"""v5 - Exécution contrôlée.

v4 : le modèle décide quand s'arrêter, et toute erreur plante le programme.
v5 : le HARNAIS décide. Il borne la boucle (étapes, temps), valide chaque appel
d'outil, enferme les fichiers dans le workspace, et transforme toute erreur en
résultat structuré que le modèle peut lire et corriger.

Pur Python : stdlib uniquement (dataclasses, concurrent.futures, pathlib, json)
+ le client `openai` déjà utilisé depuis v0.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "gemma4:31b-cloud"

WORKSPACE = Path(__file__).parent / "workspace"
MEMORY_FILE = Path(__file__).parent / "memory.md"


# --- 1. configuration d'exécution -------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    max_steps: int = 8             # appels au modèle par tâche
    timeout_s: float = 120.0       # durée totale de la tâche
    tool_timeout_s: float = 10.0   # durée max d'un outil
    max_tool_output: int = 4000    # caractères renvoyés au modèle par outil
    confirm_writes: bool = True    # demander avant d'écrire (désactivé dans les évals)


@dataclass
class RunResult:
    status: str    # "done" | "max_steps" | "timeout" | "error"
    answer: str
    steps: int
    elapsed: float


# --- 2. erreurs structurées --------------------------------------------------
# Un outil ne lève jamais d'exception vers la boucle : il renvoie un dict.
# Le modèle voit "error_type" et peut se corriger au tour suivant.

def ok(content: str) -> dict:
    return {"ok": True, "content": content}


def err(error_type: str, message: str) -> dict:
    return {"ok": False, "error_type": error_type, "content": message}


# --- 3. contrôle des chemins -------------------------------------------------

class PathError(Exception):
    pass


def safe_path(filename: str) -> Path:
    """Résout le chemin et refuse tout ce qui sort du workspace."""
    root = WORKSPACE.resolve()
    path = (root / filename).resolve()
    if path != root and root not in path.parents:
        raise PathError(f"'{filename}' is outside the workspace")
    return path


# --- 4. outils ---------------------------------------------------------------

def load_memory() -> str:
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return "(nothing saved yet)"


def save_memory(fact: str) -> dict:
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    return ok(f"saved: {fact}")


def list_files() -> dict:
    if not WORKSPACE.is_dir():
        return err("not_found", "workspace folder does not exist")
    return ok("\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)")


def read_file(filename: str) -> dict:
    path = safe_path(filename)
    if not path.is_file():
        return err("not_found", f"no file named {filename}")
    return ok(path.read_text(encoding="utf-8"))


def write_file(filename: str, content: str) -> dict:
    path = safe_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return ok(f"wrote {filename} ({len(content)} chars)")


def _schema(name, description, properties=None, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": list(required),
            },
        },
    }


TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "save_memory": save_memory,
}

# Permission par outil : "read" passe librement, "write" peut demander confirmation.
PERMISSIONS = {
    "list_files": "read",
    "read_file": "read",
    "write_file": "write",
    "save_memory": "write",
}

TOOL_SCHEMAS = [
    _schema("list_files", "List the files in the user's workspace folder."),
    _schema("read_file", "Read one file from the workspace.",
            {"filename": {"type": "string"}}, ["filename"]),
    _schema("write_file", "Write a text file in the workspace.",
            {"filename": {"type": "string"}, "content": {"type": "string"}},
            ["filename", "content"]),
    _schema("save_memory",
            "Save one short fact about the user to long-term memory.",
            {"fact": {"type": "string"}}, ["fact"]),
]
SCHEMA_BY_NAME = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SCHEMAS}


# --- 5. validation des arguments ---------------------------------------------

def validate_args(name: str, args) -> str | None:
    """Renvoie un message d'erreur, ou None si les arguments sont valides."""
    schema = SCHEMA_BY_NAME[name]
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    for key in schema["required"]:
        if key not in args:
            return f"missing required argument '{key}'"
    for key, value in args.items():
        if key not in schema["properties"]:
            return f"unexpected argument '{key}'"
        expected = schema["properties"][key]["type"]
        if expected == "string" and not isinstance(value, str):
            return f"argument '{key}' must be a string"
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return f"argument '{key}' must be an integer"
    return None


# --- 6. exécution d'un outil : le point de passage unique --------------------

_pool = ThreadPoolExecutor(max_workers=4)


def confirm_cli(name: str, args: dict) -> bool:
    preview = json.dumps(args, ensure_ascii=False)[:200]
    return input(f"  [confirm] {name}({preview}) ? [y/N] ").strip().lower() == "y"


def execute_tool(name: str, raw_args: str, cfg: RunConfig, confirm=confirm_cli) -> dict:
    if name not in TOOLS:
        return err("unknown_tool", f"no tool named '{name}'; available: {sorted(TOOLS)}")

    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as e:
        return err("bad_arguments", f"arguments are not valid JSON: {e}")

    problem = validate_args(name, args)
    if problem:
        return err("bad_arguments", problem)

    if PERMISSIONS[name] == "write" and cfg.confirm_writes and not confirm(name, args):
        return err("denied", "the user refused this action")

    # Timeout par outil. Limite honnête : un thread ne se tue pas en Python ;
    # on arrête d'attendre, mais la fonction peut finir en arrière-plan.
    future = _pool.submit(TOOLS[name], **args)
    try:
        result = future.result(timeout=cfg.tool_timeout_s)
    except FutureTimeout:
        return err("timeout", f"tool exceeded {cfg.tool_timeout_s}s")
    except PathError as e:
        return err("forbidden_path", str(e))
    except Exception as e:  # le harnais ne plante jamais à cause d'un outil
        return err("tool_failed", f"{type(e).__name__}: {e}")

    if len(result["content"]) > cfg.max_tool_output:
        cut = len(result["content"]) - cfg.max_tool_output
        result["content"] = result["content"][: cfg.max_tool_output] + f"\n[... truncated {cut} chars]"
    return result


# --- 7. la boucle bornée -----------------------------------------------------

SYSTEM_PROMPT = """You are a helpful personal assistant. Be concise.

Here is what you remember about the user from previous sessions:
{memory}

When you learn a new lasting fact about the user, save it with save_memory.
If a tool returns an error, read error_type and try a different approach."""


def run_agent(messages: list, cfg: RunConfig = RunConfig(), confirm=confirm_cli) -> RunResult:
    start = time.monotonic()
    steps = 0

    def result(status: str, answer: str = "") -> RunResult:
        return RunResult(status, answer, steps, time.monotonic() - start)

    while True:
        remaining = cfg.timeout_s - (time.monotonic() - start)
        if remaining <= 0:
            return result("timeout")
        if steps >= cfg.max_steps:
            return result("max_steps")

        steps += 1
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOL_SCHEMAS, timeout=remaining
            )
        except Exception as e:
            return result("error", f"{type(e).__name__}: {e}")

        message = response.choices[0].message
        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            return result("done", message.content or "")

        # On stocke un dict simple (plus lisible, rejouable, sérialisable en v6).
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                for c in message.tool_calls
            ],
        })
        # Chaque tool_call reçoit TOUJOURS une réponse, même en erreur :
        # sinon l'API rejette le message suivant.
        for call in message.tool_calls:
            out = execute_tool(call.function.name, call.function.arguments, cfg, confirm)
            flag = "ok " if out["ok"] else "ERR"
            print(f"  [tool:{flag}] {call.function.name}({call.function.arguments})")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(out, ensure_ascii=False)})


# --- 8. REPL -----------------------------------------------------------------

if __name__ == "__main__":
    cfg = RunConfig()
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory())}]
    print(f"v5 assistant ({MODEL}) - max {cfg.max_steps} steps, {cfg.timeout_s:.0f}s - ctrl+c to quit")
    try:
        while True:
            messages.append({"role": "user", "content": input("\nyou: ")})
            r = run_agent(messages, cfg)
            print(f"\nassistant: {r.answer}")
            print(f"  [run] status={r.status} steps={r.steps} elapsed={r.elapsed:.1f}s")
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")


# --- EXERCICES ---------------------------------------------------------------
# 1. Erreur de lecture : demande « lis le fichier ../memory.md ».
#    Observe [tool:ERR] forbidden_path. Le modèle s'adapte-t-il, ou insiste-t-il ?
#    Puis demande « lis nimporte_quoi.txt » : not_found, sans crash.
#
# 2. Dépassement de limite : mets RunConfig(max_steps=3) et demande
#    « lis chaque fichier du workspace un par un puis résume-les tous ».
#    Attendu : status=max_steps, et non une boucle sans fin.
#    Variante : RunConfig(timeout_s=2) => status=timeout.
#
# 3. Bonus : retire "unexpected argument" dans validate_args et regarde ce
#    que fait un petit modèle qui invente un paramètre.