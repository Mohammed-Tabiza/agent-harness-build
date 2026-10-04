"""harness - l'assemblage de v5 (exécution contrôlée), v6 (état et reprise) et
v7 (contexte maîtrisé), autour de responsabilités distinctes.

    Workspace, Memory     stockage : fichiers de l'utilisateur, faits retenus
    Tool, ToolCatalog     ce que l'agent PEUT faire (schéma + fonction + permission)
    Policy                ce qu'il a le DROIT de faire (autoriser / confirmer / refuser)
    ToolExecutor          validation, politique, timeout, erreurs structurées, troncature
    Journal, RunStore     persistance d'exécution (un journal append-only par tâche)
    ContextManager        ce qui part vers le modèle à chaque étape (budget, résumé)
    OpenAIModel           l'unique frontière avec le SDK (n'importe quel serveur compatible)
    TaskRunner            la boucle : états, limites, journal, reprise
    Harness               assemble le tout autour d'un dossier racine

Aucun état global : tout est injecté. Un test ou une éval crée un Harness(tmp) neuf,
avec un faux modèle si besoin (tout objet qui a une méthode `chat`).

Pur Python (stdlib). Le SDK `openai` n'est importé qu'à la création d'un OpenAIModel.
Pas de MCP ici : un outil MCP s'enregistrerait comme n'importe quel Tool, avec la
permission "network" (donc confirmée par défaut) et sans applied_check (donc rejoué
tel quel à la reprise : à déclarer idempotent seulement si c'est vrai).

Usage :
    python harness.py                  # REPL : chaque ligne est une TÂCHE journalisée
    python harness.py --model gpt
    /models /model <n> /tools /memory /tasks /show <id> /resume <id> /quit
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model: str
    api_key: str = "none"


# Chaque entrée est un endpoint compatible OpenAI : cloud ou local, même client.
MODELS: dict[str, ModelConfig] = {
    "local": ModelConfig(base_url="http://localhost:11434/v1", api_key="ollama",
                         model="gemma4:31b-cloud"),
    "local-small": ModelConfig(base_url="http://localhost:11434/v1", api_key="ollama",
                               model="gemma4:31b-cloud"),
    "gpt": ModelConfig(base_url="https://api.openai.com/v1",
                       api_key=os.getenv("OPENAI_API_KEY", ""), model="gpt-5.2"),
    "claude": ModelConfig(base_url="https://api.anthropic.com/v1/",
                          api_key=os.getenv("ANTHROPIC_API_KEY", ""), model="claude-sonnet-5"),
}


@dataclass(frozen=True)
class RunConfig:
    max_steps: int = 40            # appels au modèle par tâche
    timeout_s: float = 600.0       # durée totale (repart à zéro à chaque reprise)
    tool_timeout_s: float = 10.0   # durée max d'un outil
    max_tool_output: int = 2500    # caractères renvoyés au modèle par outil


@dataclass(frozen=True)
class ContextConfig:
    budget_tokens: int = 3000      # ~75 % de la fenêtre RÉELLE du modèle
    low_water: float = 0.7         # sur débordement, on redescend à 70 % du budget
    chars_per_token: float = 3.5   # estimation grossière
    keep_full: int = 2             # derniers échanges jamais réduits
    old_result_cap: int = 200      # taille d'un ancien résultat réduit
    summary_max_chars: int = 800
    llm_summary: bool = False      # résumé par le modèle (coûteux, peu fiable à 4B)
    memory_max_chars: int = 1200


# =============================================================================
# 2. Résultats d'outils structurés
# =============================================================================
# Un outil ne lève jamais d'exception vers la boucle : il renvoie un dict.
# Le modèle voit « error_type » et peut se corriger au tour suivant.

def ok(content: str) -> dict:
    return {"ok": True, "content": content}


def err(error_type: str, message: str) -> dict:
    return {"ok": False, "error_type": error_type, "content": message}


# =============================================================================
# 3. Stockage : espace de travail et mémoire
# =============================================================================

class PathError(Exception):
    pass


class Workspace:
    """Le dossier de l'utilisateur. Aucun chemin ne peut en sortir."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, filename: str) -> Path:
        path = (self.root / filename).resolve()   # suit les liens symboliques et les ".."
        if path != self.root and self.root not in path.parents:
            raise PathError(f"'{filename}' is outside the workspace")
        return path


class Memory:
    """Mémoire à long terme : un fichier markdown, un fait par ligne."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> str:
        return self.path.read_text(encoding="utf-8") if self.path.is_file() else "(nothing saved yet)"

    def append(self, fact: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"- {fact}\n")

    def has(self, fact: str) -> bool:
        return f"- {fact}" in self.load().splitlines()

    def bounded(self, max_chars: int) -> str:
        """Garde les faits les plus récents qui tiennent dans le budget."""
        text = self.load()
        if len(text) <= max_chars:
            return text
        lines, kept, size = text.splitlines(), [], 0
        for line in reversed(lines):
            if size + len(line) + 1 > max_chars:
                break
            kept.append(line)
            size += len(line) + 1
        return f"[{len(lines) - len(kept)} older fact(s) omitted]\n" + "\n".join(reversed(kept))


# =============================================================================
# 4. Catalogue d'outils : ce que l'agent PEUT faire
# =============================================================================

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: Callable[..., dict]
    properties: dict = field(default_factory=dict)
    required: tuple = ()
    permission: str = "read"                          # "read" | "write" | "network"
    applied_check: Callable[[dict], bool] | None = None
    # applied_check : « l'effet de cet appel est-il déjà visible ? ». Sert à la reprise
    # après un crash. Absent = l'outil est considéré idempotent et sera simplement rejoué.

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.properties,
                           "required": list(self.required)}}}

    def validate(self, args) -> str | None:
        """Message d'erreur, ou None si les arguments sont valides."""
        if not isinstance(args, dict):
            return "arguments must be a JSON object"
        for key in self.required:
            if key not in args:
                return f"missing required argument '{key}'"
        for key, value in args.items():
            spec = self.properties.get(key)
            if spec is None:
                return f"unexpected argument '{key}'"
            if spec["type"] == "string" and not isinstance(value, str):
                return f"argument '{key}' must be a string"
            if spec["type"] == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                return f"argument '{key}' must be an integer"
        return None


class ToolCatalog:
    def __init__(self, tools: list[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool           # remplace si le nom existe déjà

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def __iter__(self):
        return iter(self._tools.values())


PREVIEW_CHARS = 1500
CHUNK_MAX = 2000


def local_tools(ws: Workspace, memory: Memory) -> list[Tool]:
    """Les outils locaux, liés (par fermeture) à UN workspace et UNE mémoire."""

    def list_files() -> dict:
        return ok("\n".join(sorted(p.name for p in ws.root.iterdir())) or "(empty)")

    def read_file(filename: str) -> dict:
        path = ws.resolve(filename)
        if not path.is_file():
            return err("not_found", f"no file named {filename}")
        text = path.read_text(encoding="utf-8")
        if len(text) <= PREVIEW_CHARS:
            return ok(text)
        return ok(text[:PREVIEW_CHARS] + f"\n[truncated: file has {len(text)} chars, showing the "
                  f"first {PREVIEW_CHARS}. Continue with read_chunk(filename, offset={PREVIEW_CHARS}).]")

    def read_chunk(filename: str, offset: int = 0, length: int = 1500) -> dict:
        path = ws.resolve(filename)
        if not path.is_file():
            return err("not_found", f"no file named {filename}")
        if offset < 0 or length <= 0:
            return err("bad_arguments", "offset must be >= 0 and length > 0")
        text = path.read_text(encoding="utf-8")
        chunk = text[offset: offset + min(length, CHUNK_MAX)]
        if not chunk:
            return ok(f"[end of file: {len(text)} chars]")
        end = offset + len(chunk)
        where = f"next offset={end}" if end < len(text) else "end of file"
        return ok(f"{chunk}\n[chars {offset}-{end} of {len(text)}; {where}]")

    def write_file(filename: str, content: str) -> dict:
        path = ws.resolve(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ok(f"wrote {filename} ({len(content)} chars)")

    def append_file(filename: str, content: str) -> dict:
        path = ws.resolve(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(content + "\n")
        return ok(f"appended to {filename} ({len(content)} chars)")

    def save_memory(fact: str) -> dict:
        memory.append(fact)
        return ok(f"saved: {fact}")

    # Vérifications de reprise : « l'effet est-il déjà là ? »
    def write_applied(a: dict) -> bool:
        p = ws.resolve(a["filename"])
        return p.is_file() and p.read_text(encoding="utf-8") == a["content"]

    def append_applied(a: dict) -> bool:
        # Heuristique : un append volontairement répété juste après serait pris pour un doublon.
        p = ws.resolve(a["filename"])
        return p.is_file() and p.read_text(encoding="utf-8").endswith(a["content"] + "\n")

    S, I = {"type": "string"}, {"type": "integer"}
    return [
        Tool("list_files", "List the files in the user's workspace folder.", list_files),
        Tool("read_file", "Read a file. Large files return only a preview; use read_chunk to continue.",
             read_file, {"filename": S}, ("filename",)),
        Tool("read_chunk", "Read a slice of a file: offset and length in characters (length max 2000).",
             read_chunk, {"filename": S, "offset": I, "length": I}, ("filename",)),
        Tool("write_file", "Write a text file in the workspace (replaces existing content).",
             write_file, {"filename": S, "content": S}, ("filename", "content"),
             permission="write", applied_check=write_applied),
        Tool("append_file", "Append one line to a file in the workspace (creates it if needed). "
             "Use it to keep notes on long documents.",
             append_file, {"filename": S, "content": S}, ("filename", "content"),
             permission="write", applied_check=append_applied),
        Tool("save_memory", "Save one short fact about the user to long-term memory. Use whenever "
             "you learn something lasting: their name, preferences, projects, recurring tasks.",
             save_memory, {"fact": S}, ("fact",),
             permission="write", applied_check=lambda a: memory.has(a["fact"])),
    ]


# =============================================================================
# 5. Politique d'autorisation : ce que l'agent a le DROIT de faire
# =============================================================================

def confirm_cli(tool_name: str, args: dict) -> bool:
    preview = json.dumps(args, ensure_ascii=False)[:200]
    return input(f"  [confirm] {tool_name}({preview}) ? [y/N] ").strip().lower() == "y"


@dataclass
class Policy:
    """Niveau par permission ("allow" | "confirm" | "deny"), avec dérogations par outil.
    Une permission inconnue est refusée (échec fermé)."""
    levels: dict = field(default_factory=lambda: {"read": "allow", "write": "confirm",
                                                  "network": "confirm"})
    overrides: dict = field(default_factory=dict)       # nom d'outil -> niveau
    confirm_fn: Callable[[str, dict], bool] = confirm_cli

    @classmethod
    def permissive(cls) -> "Policy":
        """Tout est autorisé sans confirmation (évals, tests)."""
        return cls(levels={"read": "allow", "write": "allow", "network": "allow"})

    def authorize(self, tool: Tool, args: dict) -> dict | None:
        """None = autorisé ; sinon le résultat d'erreur à renvoyer au modèle."""
        level = self.overrides.get(tool.name, self.levels.get(tool.permission, "deny"))
        if level == "allow":
            return None
        if level == "confirm":
            if self.confirm_fn(tool.name, args):
                return None
            return err("denied", "the user refused this action")
        return err("denied", f"tool '{tool.name}' is not permitted by the policy")


# =============================================================================
# 6. Exécution d'un outil : le point de passage unique
# =============================================================================

class ToolExecutor:
    def __init__(self, catalog: ToolCatalog, policy: Policy, cfg: RunConfig):
        self.catalog, self.policy, self.cfg = catalog, policy, cfg
        self._pool = ThreadPoolExecutor(max_workers=4)

    def execute(self, name: str, raw_args: str) -> dict:
        tool = self.catalog.get(name)
        if tool is None:
            return err("unknown_tool", f"no tool named '{name}'; available: {self.catalog.names()}")
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as e:
            return err("bad_arguments", f"arguments are not valid JSON: {e}")
        problem = tool.validate(args)
        if problem:
            return err("bad_arguments", problem)
        denied = self.policy.authorize(tool, args)
        if denied:
            return denied

        # Timeout par outil. Limite honnête : un thread ne se tue pas en Python ;
        # on arrête d'attendre, mais la fonction peut finir en arrière-plan.
        future = self._pool.submit(tool.fn, **args)
        try:
            out = future.result(timeout=self.cfg.tool_timeout_s)
        except FutureTimeout:
            return err("timeout", f"tool exceeded {self.cfg.tool_timeout_s}s")
        except PathError as e:
            return err("forbidden_path", str(e))
        except Exception as e:                 # le harnais ne plante jamais à cause d'un outil
            return err("tool_failed", f"{type(e).__name__}: {e}")
        if not isinstance(out, dict) or "content" not in out:
            return err("tool_failed", "tool returned a malformed result")

        cap = self.cfg.max_tool_output
        if len(out["content"]) > cap:
            cut = len(out["content"]) - cap
            out["content"] = out["content"][:cap] + f"\n[... truncated {cut} chars]"
        return out


# =============================================================================
# 7. Persistance d'exécution : journal append-only et reconstruction d'état
# =============================================================================

class Journal:
    """Un fichier JSONL par tâche. L'état d'une tâche = la relecture de son journal."""

    def __init__(self, path: Path, task_id: str):
        self.path, self.task_id = Path(path), task_id

    def append(self, type: str, **data) -> None:
        event = {"ts": round(time.time(), 3), "task_id": self.task_id, "type": type, **data}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())   # sans fsync, un crash peut perdre les derniers événements

    def read(self) -> list[dict]:
        """Si la dernière ligne est tronquée (crash pendant l'écriture), on l'ignore ET on
        réécrit le fichier proprement, sinon le prochain append se collerait à la ligne cassée."""
        if not self.path.is_file():
            return []
        events, clean = [], True
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                clean = False
                break
        if not clean:
            self.path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events),
                                 encoding="utf-8")
        return events


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def new_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]

    def journal(self, task_id: str) -> Journal:
        return Journal(self.root / task_id / "events.jsonl", task_id)

    def ids(self) -> list[str]:
        return sorted(d.name for d in self.root.glob("*/")) if self.root.is_dir() else []


@dataclass
class State:
    messages: list
    steps: int
    unknown: set            # appels commencés (tool_started) sans résultat (tool_done)
    finished: dict | None
    prompt_tokens: int
    completion_tokens: int


def rebuild(events: list[dict]) -> State:
    start = events[0]
    messages = [{"role": "system", "content": start["system"]},
                {"role": "user", "content": start["prompt"]}]
    st = State(messages, 0, set(), None, 0, 0)
    started, done = set(), set()
    for e in events[1:]:
        if e["type"] == "model_response":
            st.steps += 1
            messages.append(e["message"])
            st.prompt_tokens += (e.get("usage") or {}).get("prompt_tokens", 0)
            st.completion_tokens += (e.get("usage") or {}).get("completion_tokens", 0)
        elif e["type"] == "tool_started":
            started.add(e["call_id"])
        elif e["type"] == "tool_done":
            done.add(e["call_id"])
            messages.append(e["message"])
        elif e["type"] == "task_finished":
            st.finished = e
    st.unknown = started - done
    return st


def pending_calls(messages: list) -> list[dict]:
    """Appels demandés par le dernier message assistant et restés sans réponse."""
    last = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
    if not last or not last.get("tool_calls"):
        return []
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    return [c for c in last["tool_calls"] if c["id"] not in answered]


def task_status(events: list[dict]) -> str:
    for e in reversed(events):
        if e["type"] == "task_finished":
            return e["status"]
    # Sans verrou ni pid, « en cours » et « planté » sont indistinguables.
    return "interrompue (reprenable)" if events else "inconnue"


# =============================================================================
# 8. Modèle : la seule frontière avec le SDK
# =============================================================================

@dataclass
class ModelReply:
    message: dict            # {"role": "assistant", "content": str, "tool_calls": [...]?}
    usage: dict              # {"prompt_tokens": int, "completion_tokens": int}


class OpenAIModel:
    """Tout objet avec une méthode chat(messages, tools, timeout) -> ModelReply peut le remplacer."""

    def __init__(self, cfg: ModelConfig):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("le client `openai` est requis : pip install openai") from e
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def chat(self, messages: list, tools: list | None = None, timeout: float = 60.0) -> ModelReply:
        kwargs = {"model": self.cfg.model, "messages": messages, "timeout": timeout}
        if tools:
            kwargs["tools"] = tools
        r = self.client.chat.completions.create(**kwargs)
        m = r.choices[0].message
        msg = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                for c in m.tool_calls]
        u = getattr(r, "usage", None)
        return ModelReply(msg, {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                                "completion_tokens": getattr(u, "completion_tokens", 0) or 0})


# =============================================================================
# 9. Gestion du contexte : ce qui part vers le modèle à cette étape
# =============================================================================
# messages = l'histoire complète (le journal la garde) ; la VUE = ce qu'on envoie.
# Du moins au plus destructeur : (1) réduire les anciens résultats d'outils,
# (2) remplacer les plus anciens échanges par un résumé, (3) en urgence, réduire
# même le résultat le plus récent. Jamais coupé : system prompt, consigne, dernier
# échange, ni un appel d'outil séparé de sa réponse.

def est_tokens(obj, chars_per_token: float) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return int(len(text) / chars_per_token) + 4


def split_units(messages: list[dict]) -> list[list[dict]]:
    """Un message assistant avec tool_calls et ses réponses forment UNE unité."""
    units: list[list[dict]] = []
    for m in messages:
        if m["role"] == "tool" and units:
            units[-1].append(m)
        else:
            units.append([m])
    return units


def shrink_tool_message(m: dict, cap: int) -> dict:
    """Réduit le contenu d'un résultat d'outil en gardant un JSON valide."""
    data = None
    try:
        data = json.loads(m["content"])
        text = data["content"]
    except Exception:
        data, text = None, m["content"]
    if len(text) <= cap:
        return m
    short = text[:cap] + f" [... réduit, {len(text)} car. à l'origine]"
    return {**m, "content": json.dumps({**data, "content": short}, ensure_ascii=False) if data else short}


def clip(text, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def digest(units: list[list[dict]]) -> list[str]:
    """Une ligne par événement. Perd du détail, mais n'invente jamais rien."""
    lines = []
    for u in units:
        m = u[0]
        if m["role"] == "assistant" and m.get("tool_calls"):
            results = {t["tool_call_id"]: t for t in u[1:]}
            for c in m["tool_calls"]:
                status, body, t = "?", "", results.get(c["id"])
                if t:
                    try:
                        d = json.loads(t["content"])
                        status, body = ("ok" if d.get("ok") else "ERR"), str(d.get("content", ""))
                    except Exception:
                        body = t["content"]
                lines.append(f"{c['function']['name']}({clip(c['function']['arguments'], 80)})"
                             f" -> {status}, {len(body)} car.: {clip(body, 60)}")
        else:
            lines.append(f"{m['role']}: {clip(m.get('content') or '', 120)}")
    return lines


def merge_summary(prev: str, new_lines: list[str], cap: int) -> str:
    text = (prev + "\n" if prev else "") + "\n".join(new_lines)
    if len(text) <= cap:
        return text
    tail = text[-cap:]
    tail = tail[tail.find("\n") + 1:] if "\n" in tail else tail   # pas de début en milieu de ligne
    return "[... début omis]\n" + tail


class ContextManager:
    """Un par tâche : retient le résumé et le nombre d'échanges déjà résumés."""

    def __init__(self, cfg: ContextConfig, schemas: list[dict], model=None,
                 log: Callable[[str], None] = print):
        self.cfg, self.model, self.log = cfg, model, log
        self.overhead = est_tokens(schemas, cfg.chars_per_token)   # les schémas coûtent aussi
        self.summary = ""
        self.covered = 0          # échanges déjà remplacés par le résumé (monotone)
        self.shrinking = False    # une fois activée, la réduction reste (préfixe stable)
        self.stats: dict = {}

    def _summarize(self, new_lines: list[str]) -> str:
        cap = self.cfg.summary_max_chars
        if self.cfg.llm_summary and self.model is not None:
            material = merge_summary(self.summary, new_lines, cap * 3)
            try:
                r = self.model.chat([{"role": "user", "content":
                    "Summarize in at most 6 short lines what has been done and learned. "
                    "Keep file names, numbers, codes and decisions exactly.\n\n" + material}],
                    None, 30)
                text = r.message["content"].strip()
                if text:
                    return text[:cap]
            except Exception:
                pass                                  # repli sur le digest
        return merge_summary(self.summary, new_lines, cap)

    def build(self, messages: list[dict]) -> list[dict]:
        """messages[0] = system, messages[1] = consigne (invariant posé par rebuild)."""
        cfg, cpt = self.cfg, self.cfg.chars_per_token
        system, goal = messages[0], messages[1]
        units = split_units(messages[2:])

        def with_summary(text: str) -> dict:
            if not text:
                return system
            return {**system, "content": system["content"] +
                    "\n\nEarlier steps (summary; older details were dropped to save context):\n" + text}

        def assemble(cut: int, shrink: bool, last_cap: int | None, summary: str) -> list[dict]:
            view = [with_summary(summary), goal]
            for i, u in enumerate(units):
                if i < cut:
                    continue
                old = shrink and i < len(units) - cfg.keep_full
                for m in u:
                    if m["role"] == "tool":
                        cap = cfg.old_result_cap if old else (last_cap if i == len(units) - 1 else None)
                        if cap:
                            m = shrink_tool_message(m, cap)
                    view.append(m)
            return view

        def cost(view: list[dict]) -> int:
            return self.overhead + sum(est_tokens(m, cpt) for m in view)

        def fits(cut: int, shrink: bool, last_cap: int | None = None, ratio: float = 1.0) -> bool:
            placeholder = "x" * cfg.summary_max_chars if cut > 0 else ""   # pire cas du résumé
            return cost(assemble(cut, shrink, last_cap, placeholder)) <= cfg.budget_tokens * ratio

        cut = self.covered
        if not self.shrinking and not fits(cut, False):                  # étape 1
            self.shrinking = True
        if not fits(cut, self.shrinking):                                # étape 2, avec hystérésis
            while not fits(cut, self.shrinking, ratio=cfg.low_water) and cut < len(units) - 1:
                cut += 1
        if cut > self.covered:
            self.summary = self._summarize(digest(units[self.covered:cut]))
            self.covered = cut
        last_cap = None                                                  # étape 3
        for cap in (1500, 800, 400, 200):
            if fits(cut, self.shrinking, last_cap):
                break
            last_cap = cap

        view = assemble(cut, self.shrinking, last_cap, self.summary)
        original = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
        shrunk = sum(1 for m in view if m["role"] == "tool"
                     and m["content"] != original[m["tool_call_id"]])
        self.stats = {"tokens": cost(view), "budget": cfg.budget_tokens,
                      "summarized_units": self.covered, "shrunk_results": shrunk,
                      "emergency": last_cap is not None}
        s = self.stats
        if s["summarized_units"] or s["shrunk_results"] or s["emergency"]:
            self.log(f"  [ctx] ~{s['tokens']}/{s['budget']} tok | {s['summarized_units']} échange(s) "
                     f"résumé(s) | {s['shrunk_results']} résultat(s) réduit(s)"
                     + (" | URGENCE" if s["emergency"] else ""))
        return view


# =============================================================================
# 10. L'exécuteur de tâches : états, limites, journal, reprise
# =============================================================================

SYSTEM_PROMPT = """You are a helpful personal assistant running inside a custom harness. Be concise.

The user's workspace folder holds their personal files: notes, todo lists, ideas. You have tools \
to list, read, and write those files. Never claim you lack access to the user's files or tasks - \
use your tools to look. Large files return only a preview: continue with read_chunk, and keep \
notes with append_file if you need to remember what you read.

Here is what you remember about the user from previous sessions:
{memory}

When you learn a new lasting fact about the user, save it with save_memory.
If a tool returns an error, read error_type and try a different approach."""


class SimulatedCrash(BaseException):
    """BaseException : les `except Exception` des outils ne l'avalent pas, comme un vrai kill."""


@dataclass
class TaskResult:
    task_id: str
    status: str              # "done" | "max_steps" | "timeout" | "error"
    answer: str
    steps: int
    elapsed: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    messages: list = field(default_factory=list)


class TaskRunner:
    def __init__(self, model, catalog: ToolCatalog, executor: ToolExecutor, store: RunStore,
                 memory: Memory, run_cfg: RunConfig, ctx_cfg: ContextConfig,
                 log: Callable[[str], None] = print):
        self.model, self.catalog, self.executor, self.store = model, catalog, executor, store
        self.memory, self.run_cfg, self.ctx_cfg, self.log = memory, run_cfg, ctx_cfg, log

    def start(self, prompt: str, faults: dict | None = None, task_id: str | None = None) -> TaskResult:
        task_id = task_id or self.store.new_id()
        journal = self.store.journal(task_id)
        if journal.read():
            raise ValueError(f"la tâche {task_id} existe déjà : utilise resume")
        system = SYSTEM_PROMPT.format(memory=self.memory.bounded(self.ctx_cfg.memory_max_chars))
        journal.append("task_started", prompt=prompt, system=system, run_cfg=asdict(self.run_cfg))
        return self._run(task_id, rebuild(journal.read()), faults)

    def resume(self, task_id: str, faults: dict | None = None) -> TaskResult:
        events = self.store.journal(task_id).read()
        if not events:
            raise ValueError(f"aucune tâche {task_id}")
        state = rebuild(events)
        if state.finished:
            f = state.finished
            self.log(f"  [resume] tâche déjà terminée ({f['status']}), rien à refaire")
            return TaskResult(task_id, f["status"], f.get("answer", ""), f.get("steps", state.steps),
                              0.0, state.prompt_tokens, state.completion_tokens, state.messages)
        self.log(f"  [resume] {state.steps} étape(s) rejouée(s) depuis le journal, "
                 f"{len(state.unknown)} appel(s) au résultat inconnu")
        return self._run(task_id, state, faults)

    def _applied(self, tool: Tool | None, raw_args: str) -> bool:
        if tool is None or tool.applied_check is None:
            return False
        try:
            return bool(tool.applied_check(json.loads(raw_args)))
        except Exception:
            return False

    def _run(self, task_id: str, state: State, faults: dict | None) -> TaskResult:
        faults, cfg = faults or {}, self.run_cfg
        journal = self.store.journal(task_id)
        ctx = ContextManager(self.ctx_cfg, self.catalog.schemas(), self.model, self.log)
        messages, steps, unknown = state.messages, state.steps, set(state.unknown)
        tokens = [state.prompt_tokens, state.completion_tokens]
        start, executed = time.monotonic(), 0

        def result(status: str, answer: str = "") -> TaskResult:
            return TaskResult(task_id, status, answer, steps, time.monotonic() - start,
                              tokens[0], tokens[1], messages)

        def finish(status: str, answer: str = "") -> TaskResult:
            journal.append("task_finished", status=status, answer=answer, steps=steps,
                           prompt_tokens=tokens[0], completion_tokens=tokens[1])
            return result(status, answer)

        while True:
            # a) solder d'abord les appels d'outils en attente (cas de la reprise)
            for call in pending_calls(messages):
                name, raw = call["function"]["name"], call["function"]["arguments"]
                tool, recovered, t0 = self.catalog.get(name), False, time.monotonic()
                if call["id"] in unknown and self._applied(tool, raw):
                    out, recovered = ok(f"already applied before the interruption: {name}"), True
                    self.log(f"  [resume] {name} déjà appliqué, non rejoué")
                else:
                    journal.append("tool_started", call_id=call["id"], name=name, arguments=raw)
                    out = self.executor.execute(name, raw)
                    executed += 1
                    if faults.get("crash_after_exec") == executed:
                        raise SimulatedCrash(f"crash simulé après l'exécution de {name}")
                self.log(f"  [tool:{'ok ' if out['ok'] else 'ERR'}] {name}({raw})"
                         + ("  (récupéré)" if recovered else ""))
                tool_msg = {"role": "tool", "tool_call_id": call["id"],
                            "content": json.dumps(out, ensure_ascii=False)}
                journal.append("tool_done", call_id=call["id"], name=name, ok=out["ok"],
                               error_type=out.get("error_type"), recovered=recovered,
                               duration_ms=round((time.monotonic() - t0) * 1000), message=tool_msg)
                messages.append(tool_msg)
                unknown.discard(call["id"])

            # b) limites
            remaining = cfg.timeout_s - (time.monotonic() - start)
            if remaining <= 0:
                return finish("timeout")
            if steps >= cfg.max_steps:
                return finish("max_steps")

            # c) un tour de modèle sur la VUE (pas sur l'histoire complète)
            view, t0 = ctx.build(messages), time.monotonic()
            try:
                reply = self.model.chat(view, self.catalog.schemas(), remaining)
            except Exception as e:
                # Panne transitoire (serveur coupé) : on NE clôt PAS la tâche, elle reste reprenable.
                journal.append("model_error", error=f"{type(e).__name__}: {e}")
                return result("error", f"{type(e).__name__}: {e}")
            steps += 1
            tokens[0] += reply.usage.get("prompt_tokens", 0)
            tokens[1] += reply.usage.get("completion_tokens", 0)
            journal.append("model_response", step=steps, message=reply.message, usage=reply.usage,
                           duration_ms=round((time.monotonic() - t0) * 1000),
                           ctx_estimate=ctx.stats.get("tokens"))
            messages.append(reply.message)
            if faults.get("crash_after_model") == steps:
                raise SimulatedCrash("crash simulé juste après la réponse du modèle")
            if not reply.message.get("tool_calls"):
                return finish("done", reply.message["content"])


# =============================================================================
# 11. Le harnais : assemblage autour d'un dossier racine
# =============================================================================

class Harness:
    """root/workspace  fichiers de l'utilisateur   root/memory.md  faits retenus
       root/runs/<id>/events.jsonl  un journal par tâche"""

    def __init__(self, root, model=None, model_name: str = "local",
                 run_cfg: RunConfig = RunConfig(), ctx_cfg: ContextConfig = ContextConfig(),
                 policy: Policy | None = None, log: Callable[[str], None] = print):
        self.root = Path(root)
        self.run_cfg, self.ctx_cfg, self.log = run_cfg, ctx_cfg, log
        self.workspace = Workspace(self.root / "workspace")
        self.memory = Memory(self.root / "memory.md")
        self.store = RunStore(self.root / "runs")
        self.catalog = ToolCatalog(local_tools(self.workspace, self.memory))
        self.policy = policy or Policy()
        self.executor = ToolExecutor(self.catalog, self.policy, run_cfg)
        self.model_name = model_name
        self.model = model or OpenAIModel(MODELS[model_name])
        self.last_task_id: str | None = None

    def use_model(self, name: str) -> None:
        self.model, self.model_name = OpenAIModel(MODELS[name]), name

    def _runner(self) -> TaskRunner:
        return TaskRunner(self.model, self.catalog, self.executor, self.store, self.memory,
                          self.run_cfg, self.ctx_cfg, self.log)

    def run(self, prompt: str, faults: dict | None = None) -> TaskResult:
        self.last_task_id = self.store.new_id()
        return self._runner().start(prompt, faults, self.last_task_id)

    def resume(self, task_id: str, faults: dict | None = None) -> TaskResult:
        self.last_task_id = task_id
        return self._runner().resume(task_id, faults)

    def tasks(self) -> list[tuple[str, str, str]]:
        out = []
        for tid in self.store.ids():
            events = self.store.journal(tid).read()
            out.append((tid, task_status(events), events[0]["prompt"] if events else "?"))
        return out

    def show(self, task_id: str) -> list[str]:
        lines = []
        for e in self.store.journal(task_id).read():
            t = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
            d = {
                "task_started": lambda: e["prompt"],
                "model_response": lambda: (f"{(e['message'].get('content') or '')[:50]} "
                                           f"[{len(e['message'].get('tool_calls', []))} appel(s)] "
                                           f"{e.get('duration_ms', '?')}ms "
                                           f"tok={e.get('usage', {}).get('prompt_tokens', '?')}"
                                           f"/estimé {e.get('ctx_estimate', '?')}"),
                "model_error": lambda: e["error"],
                "tool_started": lambda: f"{e['name']}({e['arguments']})",
                "tool_done": lambda: (f"{e['name']} ok={e['ok']}"
                                      + (f" {e['error_type']}" if e.get("error_type") else "")
                                      + f" {e.get('duration_ms', '?')}ms"
                                      + (" RÉCUPÉRÉ" if e.get("recovered") else "")),
                "task_finished": lambda: (f"{e['status']} en {e['steps']} étape(s), "
                                          f"{e.get('prompt_tokens', 0)}+{e.get('completion_tokens', 0)} tok"),
            }.get(e["type"], lambda: "")()
            lines.append(f"  {t}  {e['type']:<15} {d}")
        return lines


# =============================================================================
# 12. REPL
# =============================================================================

def report(r: TaskResult) -> None:
    print(f"\nassistant: {r.answer}\n")
    print(f"  [run {r.task_id}] status={r.status} steps={r.steps} elapsed={r.elapsed:.1f}s "
          f"tokens={r.prompt_tokens}+{r.completion_tokens}")


def handle_command(h: Harness, line: str) -> bool:
    if not line.startswith("/"):
        return False
    cmd, _, arg = line.partition(" ")
    if cmd == "/models":
        for name, cfg in MODELS.items():
            print(f" {'*' if name == h.model_name else ' '} {name:<12} {cfg.model}  ({cfg.base_url})")
    elif cmd == "/model":
        if arg in MODELS:
            h.use_model(arg)
            print(f"  switched to {arg} ({MODELS[arg].model})")
        else:
            print(f"  unknown model '{arg}' - try /models")
    elif cmd == "/tools":
        for t in h.catalog:
            print(f"  [{t.permission:<7}] {t.name}: {t.description[:60]}")
    elif cmd == "/memory":
        print(h.memory.load())
    elif cmd == "/tasks":
        for tid, status, prompt in h.tasks():
            print(f"  {tid}  {status:<26} {prompt[:50]}")
    elif cmd == "/show":
        print("\n".join(h.show(arg)) or "  (aucun journal)")
    elif cmd == "/resume":
        report(h.resume(arg))
    elif cmd == "/quit":
        raise SystemExit
    else:
        print("  commands: /models /model <n> /tools /memory /tasks /show <id> /resume <id> /quit")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).parent))
    ap.add_argument("--model", default="local", choices=list(MODELS))
    args = ap.parse_args()
    try:
        h = Harness(args.root, model_name=args.model)
    except RuntimeError as e:
        print(e)
        return 2
    print(f"harness ready - model: {h.model_name} ({MODELS[h.model_name].model}), "
          f"{len(h.catalog.names())} tools, root: {h.root}")
    print("type /models, /model <n>, /tools, /memory, /tasks, /show <id>, /resume <id>, or just talk\n")
    while True:
        try:
            line = input("you: ").strip()
        except (EOFError, KeyboardInterrupt, SystemExit):
            break
        try:
            if not line or handle_command(h, line):
                continue
            report(h.run(line))
        except KeyboardInterrupt:
            print(f"\n  interrompu. reprends avec : /resume {h.last_task_id}")
        except SystemExit:
            break
        except SimulatedCrash as e:
            print(f"\n  *** {e} *** reprends avec : /resume {h.last_task_id}")
    print("\nbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())