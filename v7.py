"""v7 - Contexte maîtrisé.

v6 : `messages` grossit à chaque étape. Avec un petit modèle local, la fenêtre
     se remplit vite, et Ollama tronque alors EN SILENCE le début de la
     conversation (souvent le system prompt et la consigne).
v7 : le harnais décide ce qui part vers le modèle. Distinction centrale :

     messages  = l'histoire complète (le journal v6 la garde intacte)
     la VUE    = ce qu'on envoie au modèle à CETTE étape, calculée sous budget

Stratégie, du moins au plus destructeur (on s'arrête dès que ça tient) :
  1. réduire les anciens résultats d'outils (garder le début, noter la taille d'origine)
  2. remplacer les plus anciens échanges par un court résumé
  3. en dernier recours, réduire même le résultat le plus récent
Jamais coupé : le system prompt, la consigne, le dernier échange, et jamais entre
un appel d'outil et sa réponse (l'API rejette un appel orphelin).

Pour un gros fichier : read_file renvoie un aperçu + un indice, read_chunk pagine,
append_file permet au modèle d'EXTERNALISER ses notes. Ce qui sort de la vue n'est
pas perdu s'il a été écrit dans un fichier.

Pur Python. Réutilise v5 (outils, limites) et v6 (journal, reprise).

Usage :
    python v7.py bigfile                       # crée workspace/big.txt (~22 000 car.)
    python v7.py run "ta demande" [--budget 2000] [--llm-summary]
    python v7.py resume <id>
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import random
import time
import uuid
from dataclasses import dataclass

import v5
import v6
from v5 import RunConfig


# --- 1. configuration --------------------------------------------------------

@dataclass(frozen=True)
class ContextConfig:
    budget_tokens: int = 3000       # à régler vers ~75 % de la fenêtre RÉELLE du modèle
    low_water: float = 0.7          # à l'arrêt d'un débordement, on redescend à 70 % du budget
    chars_per_token: float = 3.5    # estimation grossière ; voir « limites » dans la doc
    keep_full: int = 2              # derniers échanges jamais réduits (étape 1)
    old_result_cap: int = 200       # taille max d'un ancien résultat réduit (caractères)
    summary_max_chars: int = 800
    llm_summary: bool = False       # résumé par le modèle (coûteux, peu fiable à 4B)
    memory_max_chars: int = 1200
    verbose: bool = True


# --- 2. estimation de taille -------------------------------------------------

def est_tokens(obj, chars_per_token: float) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return int(len(text) / chars_per_token) + 4


# --- 3. unités : on ne coupe jamais au milieu d'un échange d'outils ----------

def split_units(messages: list[dict]) -> list[list[dict]]:
    """Un message assistant avec tool_calls et ses réponses forment UNE unité."""
    units: list[list[dict]] = []
    for m in messages:
        if m["role"] == "tool" and units:
            units[-1].append(m)          # réponse d'outil : rattachée à son appel
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
    content = json.dumps({**data, "content": short}, ensure_ascii=False) if data else short
    return {**m, "content": content}


# --- 4. résumé : déterministe par défaut, LLM en option ----------------------

def clip(text: str, n: int) -> str:
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
                status, body = "?", ""
                t = results.get(c["id"])
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
    tail = tail[tail.find("\n") + 1:] if "\n" in tail else tail   # ne pas commencer en milieu de ligne
    return "[... début omis]\n" + tail


def llm_summary(prev: str, new_lines: list[str], cap: int) -> str:
    """Le modèle compresse le digest. Repli sur le digest seul si l'appel échoue."""
    material = merge_summary(prev, new_lines, cap * 3)
    try:
        r = v5.client.chat.completions.create(
            model=v5.MODEL, timeout=30,
            messages=[{"role": "user", "content":
                       "Summarize in at most 6 short lines what has been done and learned. "
                       "Keep file names, numbers, codes and decisions exactly.\n\n" + material}])
        text = (r.choices[0].message.content or "").strip()
        if text:
            return text[:cap]
    except Exception:
        pass
    return merge_summary(prev, new_lines, cap)


# --- 5. le gestionnaire de contexte ------------------------------------------

class ContextManager:
    def __init__(self, cfg: ContextConfig = ContextConfig()):
        self.cfg = cfg
        self.summary = ""
        self.covered = 0          # nb d'unités déjà remplacées par le résumé (monotone)
        self.shrinking = False    # une fois activée, la réduction reste (préfixe stable)
        self.stats: dict = {}

    def build(self, messages: list[dict]) -> list[dict]:
        """messages[0] = system, messages[1] = consigne (invariant posé par v6)."""
        cfg, cpt = self.cfg, self.cfg.chars_per_token
        system, goal = messages[0], messages[1]
        units = split_units(messages[2:])
        overhead = est_tokens(v5.TOOL_SCHEMAS, cpt)   # les schémas d'outils coûtent aussi

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
            return overhead + sum(est_tokens(m, cpt) for m in view)

        def fits(cut: int, shrink: bool, last_cap: int | None = None, ratio: float = 1.0) -> bool:
            placeholder = "x" * cfg.summary_max_chars if cut > 0 else ""   # pire cas du résumé
            return cost(assemble(cut, shrink, last_cap, placeholder)) <= cfg.budget_tokens * ratio

        cut = self.covered
        # étape 1 : réduire les anciens résultats
        if not self.shrinking and not fits(cut, False):
            self.shrinking = True
        # étape 2 : résumer les plus anciens échanges (jamais le dernier).
        # Hystérésis : sur débordement, on redescend jusqu'à low_water. Sans elle on
        # résumerait à CHAQUE étape (un appel LLM par tour, préfixe instable pour le cache).
        if not fits(cut, self.shrinking):
            while not fits(cut, self.shrinking, ratio=cfg.low_water) and cut < len(units) - 1:
                cut += 1
        if cut > self.covered:
            new_lines = digest(units[self.covered:cut])
            summarize = llm_summary if cfg.llm_summary else merge_summary
            self.summary = summarize(self.summary, new_lines, cfg.summary_max_chars)
            self.covered = cut
        # étape 3 : urgence, réduire le résultat le plus récent
        last_cap = None
        for cap in (1500, 800, 400, 200):
            if fits(cut, self.shrinking, last_cap):
                break
            last_cap = cap

        view = assemble(cut, self.shrinking, last_cap, self.summary)
        original = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
        changed = sum(1 for m in view if m["role"] == "tool"
                      and m["content"] != original[m["tool_call_id"]])
        self.stats = {"tokens": cost(view), "budget": cfg.budget_tokens,
                      "summarized_units": self.covered, "shrunk_results": changed,
                      "emergency": last_cap is not None}
        if cfg.verbose:
            s = self.stats
            print(f"  [ctx] ~{s['tokens']}/{s['budget']} tok | {s['summarized_units']} échange(s) "
                  f"résumé(s) | {s['shrunk_results']} résultat(s) réduit(s)"
                  + (" | URGENCE" if s["emergency"] else ""))
        return view


# --- 6. mémoire bornée -------------------------------------------------------

def bounded_memory(max_chars: int) -> str:
    """Garde les faits les plus récents qui tiennent dans le budget."""
    text = v5.load_memory()
    if len(text) <= max_chars:
        return text
    lines, kept, size = text.splitlines(), [], 0
    for line in reversed(lines):
        if size + len(line) + 1 > max_chars:
            break
        kept.append(line)
        size += len(line) + 1
    return f"[{len(lines) - len(kept)} older fact(s) omitted]\n" + "\n".join(reversed(kept))


# --- 7. outils pour gros fichiers --------------------------------------------

PREVIEW_CHARS = 1500
CHUNK_MAX = 2000


def read_file(filename: str) -> dict:
    path = v5.safe_path(filename)
    if not path.is_file():
        return v5.err("not_found", f"no file named {filename}")
    text = path.read_text(encoding="utf-8")
    if len(text) <= PREVIEW_CHARS:
        return v5.ok(text)
    return v5.ok(text[:PREVIEW_CHARS] + f"\n[truncated: file has {len(text)} chars, showing the "
                 f"first {PREVIEW_CHARS}. Continue with read_chunk(filename, offset={PREVIEW_CHARS}).]")


def read_chunk(filename: str, offset: int = 0, length: int = 1500) -> dict:
    path = v5.safe_path(filename)
    if not path.is_file():
        return v5.err("not_found", f"no file named {filename}")
    if offset < 0 or length <= 0:
        return v5.err("bad_arguments", "offset must be >= 0 and length > 0")
    text = path.read_text(encoding="utf-8")
    chunk = text[offset: offset + min(length, CHUNK_MAX)]
    if not chunk:
        return v5.ok(f"[end of file: {len(text)} chars]")
    end = offset + len(chunk)
    where = f"next offset={end}" if end < len(text) else "end of file"
    return v5.ok(f"{chunk}\n[chars {offset}-{end} of {len(text)}; {where}]")


def append_file(filename: str, content: str) -> dict:
    path = v5.safe_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(content + "\n")
    return v5.ok(f"appended to {filename} ({len(content)} chars)")


def _append_applied(args: dict) -> bool:
    # Heuristique : un append volontairement répété juste après serait pris pour un doublon.
    path = v5.safe_path(args["filename"])
    return path.is_file() and path.read_text(encoding="utf-8").endswith(args["content"] + "\n")


def register_tool(fn, schema: dict, permission: str, applied_check=None) -> None:
    """Ajoute (ou remplace) un outil dans le catalogue de v5. Raccourci pédagogique :
    on modifie les registres d'un autre module ; harness.py aura un vrai catalogue."""
    name = schema["function"]["name"]
    v5.TOOLS[name] = fn
    v5.PERMISSIONS[name] = permission
    v5.SCHEMA_BY_NAME[name] = schema["function"]["parameters"]
    v5.TOOL_SCHEMAS[:] = [s for s in v5.TOOL_SCHEMAS if s["function"]["name"] != name] + [schema]
    if applied_check:
        v6.APPLIED_CHECKS[name] = applied_check


register_tool(read_file, v5._schema(
    "read_file", "Read a file. Large files return only a preview; use read_chunk to continue.",
    {"filename": {"type": "string"}}, ["filename"]), "read")
register_tool(read_chunk, v5._schema(
    "read_chunk", "Read a slice of a file: offset and length in characters (length max 2000).",
    {"filename": {"type": "string"}, "offset": {"type": "integer"}, "length": {"type": "integer"}},
    ["filename"]), "read")
register_tool(append_file, v5._schema(
    "append_file", "Append one line to a file in the workspace (creates it if needed). "
    "Use it to keep notes on long documents.",
    {"filename": {"type": "string"}, "content": {"type": "string"}}, ["filename", "content"]),
    "write", _append_applied)


# --- 8. assemblage : v6.run_task + gestion de contexte -----------------------

def run_task(task_id: str, prompt: str | None, cfg: RunConfig | None = None,
             ctx_cfg: ContextConfig = ContextConfig(), confirm=v5.confirm_cli, faults=None):
    """Renvoie (RunResult, ContextManager) pour que les évals lisent les stats.
    À la reprise, le résumé est recalculé depuis le journal (il n'y est pas stocké)."""
    cfg = cfg or RunConfig(max_steps=40, timeout_s=600, max_tool_output=2500)
    cm = ContextManager(ctx_cfg)
    result = v6.run_task(task_id, prompt, cfg, confirm, faults, context_fn=cm.build,
                         memory_fn=lambda: bounded_memory(ctx_cfg.memory_max_chars))
    return result, cm


# --- 9. CLI ------------------------------------------------------------------

def make_bigfile() -> None:
    """~22 000 caractères, 3 codes cachés au début, au milieu et à la fin."""
    rng = random.Random(7)
    words = ("logistique entrepôt palette convoi quai inventaire commande retard expédition "
             "contrôle lot stock transporteur livraison fournisseur planning tournée").split()
    codes = {2: "CODE-ALPHA-4821", 12: "CODE-BRAVO-1937", 23: "CODE-CHARLIE-7305"}
    sections = []
    for k in range(1, 25):
        body = " ".join(rng.choice(words) for _ in range(120))
        extra = f" Référence interne : {codes[k]}." if k in codes else ""
        sections.append(f"Section {k}. {body}.{extra}")
    v5.WORKSPACE.mkdir(exist_ok=True)
    (v5.WORKSPACE / "big.txt").write_text("\n\n".join(sections), encoding="utf-8")
    size = (v5.WORKSPACE / "big.txt").stat().st_size
    print(f"workspace/big.txt créé ({size} car.), codes cachés : {', '.join(codes.values())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bigfile")
    p_run = sub.add_parser("run")
    p_run.add_argument("prompt")
    p_run.add_argument("--budget", type=int, default=3000)
    p_run.add_argument("--llm-summary", action="store_true")
    p_res = sub.add_parser("resume")
    p_res.add_argument("task_id")
    p_res.add_argument("--budget", type=int, default=3000)
    args = ap.parse_args()

    if args.cmd == "bigfile":
        make_bigfile()
        return 0

    ctx_cfg = ContextConfig(budget_tokens=args.budget,
                            llm_summary=getattr(args, "llm_summary", False))
    if args.cmd == "run":
        task_id, prompt = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4], args.prompt
    else:
        task_id, prompt = args.task_id, None
    print(f"tâche {task_id}  (budget contexte : {ctx_cfg.budget_tokens} tok)")
    try:
        r, cm = run_task(task_id, prompt, ctx_cfg=ctx_cfg)
    except KeyboardInterrupt:
        print(f"\n  interrompu. reprends avec : python v7.py resume {task_id}")
        return 130
    print(f"\nassistant: {r.answer}")
    print(f"  [run] status={r.status} steps={r.steps} elapsed={r.elapsed:.1f}s | "
          f"{cm.stats.get('summarized_units', 0)} échange(s) résumé(s)")
    return 0 if r.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())


# --- LIMITES -----------------------------------------------------------------
# - chars_per_token = 3.5 est une approximation. Le vrai décompte dépend du
#   tokenizer du modèle ; garde de la marge (budget ~75 % de la fenêtre réelle).
#   Vérifie la fenêtre effective de ton modèle dans Ollama (num_ctx) : sur
#   l'endpoint /v1 on ne peut pas la régler par requête.
# - Un seul message utilisateur géant n'est pas réduit ici.
# - Le résumé déterministe garde noms d'outils, tailles et 60 premiers caractères :
#   il ne retient PAS le contenu lu. Ce qui doit survivre doit être externalisé
#   (append_file) ou répété par le modèle dans ses réponses.
# - Réduire d'anciens résultats change le préfixe envoyé : le cache KV d'Ollama
#   est invalidé à partir de là (coût en latence, pas en exactitude).
#
# --- EXERCICES ---------------------------------------------------------------
# 1. Le problème : python v7.py bigfile, puis
#      python v7.py run "Lis big.txt par morceaux avec read_chunk. Après chaque morceau,
#      ajoute avec append_file dans notes.md les CODE-... rencontrés (ou 'rien'). À la
#      fin, relis notes.md et donne la liste des codes." --budget 2000
#    Observe [ctx] : les premiers morceaux sortent de la vue, et pourtant les trois
#    codes sont retrouvés grâce à notes.md. `python v6.py show <id>` : le journal,
#    lui, a tout gardé.
#
# 2. Retire append_file de la consigne (« garde tout en tête ») : que devient le
#    premier code une fois son échange résumé ? C'est la limite du digest.
#
# 3. --llm-summary : compare la qualité du résumé d'un 4B avec le digest
#    déterministe. Qui a gardé « CODE-ALPHA-4821 » ?