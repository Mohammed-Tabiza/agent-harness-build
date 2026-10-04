"""v6 - État et reprise.

v5 : si le processus s'arrête (Ctrl+C, crash, coupure Ollama), tout est perdu.
v6 : chaque tâche écrit un JOURNAL (runs/<id>/events.jsonl, append-only).
     L'état d'une tâche = la relecture de son journal. On peut donc la reprendre.

Trois règles qui font la différence entre « rejouer » et « reprendre » :
  1. On journalise l'INTENTION (tool_started) avant d'exécuter, et le RÉSULTAT
     (tool_done) après. Un appel avec tool_started mais sans tool_done a un
     résultat inconnu : l'effet a pu avoir lieu.
  2. Un appel déjà terminé (tool_done) n'est JAMAIS ré-exécuté : on réutilise
     le résultat du journal.
  3. Pour un appel au résultat inconnu, on vérifie l'état réel du monde avant de
     rejouer (already_applied) : save_memory est un append, le rejouer
     dupliquerait le fait.

Réutilise v5 (outils, validation, limites). Pur Python, aucune dépendance de plus.

Deux crochets optionnels pour v7 (sans eux, comportement inchangé) :
  context_fn(messages) -> messages : ce que l'on ENVOIE au modèle. Le journal, lui,
                                     garde tout.
  memory_fn() -> str               : la mémoire injectée dans le system prompt.

Usage :
    python v6.py run "ta demande"            # nouvelle tâche
    python v6.py run "ta demande" --crash-after-exec 1   # simule un crash (exercice)
    python v6.py list                        # états des tâches
    python v6.py show <id>                   # le journal, lisible
    python v6.py resume <id>                 # reprend une tâche interrompue
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import v5
from v5 import RunConfig, RunResult

RUNS_DIR = Path(__file__).parent / "runs"


class SimulatedCrash(BaseException):
    """Hérite de BaseException : le `except Exception` des outils ne l'avale pas,
    comme un vrai kill du processus."""


# --- 1. le journal : append-only, fsync à chaque événement -------------------

class Journal:
    def __init__(self, path: Path):
        self.path = path

    def append(self, type: str, **data) -> None:
        event = {"ts": round(time.time(), 3), "type": type, **data}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())  # sans fsync, un crash peut perdre les derniers événements

    def read(self) -> list[dict]:
        """Relit le journal. Si la dernière ligne est tronquée (crash pendant
        l'écriture), on l'ignore ET on réécrit le fichier proprement, sinon le
        prochain append se collerait à la ligne cassée."""
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
            self.path.write_text(
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events),
                encoding="utf-8",
            )
        return events


# --- 2. l'état = la relecture du journal -------------------------------------

def rebuild(events: list[dict]):
    """Reconstruit (messages, nb d'étapes, appels au résultat inconnu, fin)."""
    start = events[0]
    messages = [
        {"role": "system", "content": start["system"]},
        {"role": "user", "content": start["prompt"]},
    ]
    steps, started, done, finished = 0, set(), set(), None
    for e in events[1:]:
        if e["type"] == "model_response":
            steps += 1
            messages.append(e["message"])
        elif e["type"] == "tool_started":
            started.add(e["call_id"])
        elif e["type"] == "tool_done":
            done.add(e["call_id"])
            messages.append(e["message"])
        elif e["type"] == "task_finished":
            finished = e
    return messages, steps, started - done, finished


def pending_calls(messages: list) -> list[dict]:
    """Appels demandés par le dernier message assistant et sans réponse."""
    last = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
    if not last or not last.get("tool_calls"):
        return []
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    return [c for c in last["tool_calls"] if c["id"] not in answered]


def task_status(events: list[dict]) -> str:
    if not events:
        return "inconnue"
    for e in reversed(events):
        if e["type"] == "task_finished":
            return e["status"]
    # Sans verrou ni pid, on ne peut pas distinguer « en cours » de « planté ».
    return "interrompue (reprenable)"


# --- 3. vérifier le monde avant de rejouer -----------------------------------

def _memory_applied(args: dict) -> bool:
    return f"- {args['fact']}" in v5.load_memory().splitlines()


def _write_applied(args: dict) -> bool:
    path = v5.safe_path(args["filename"])
    return path.is_file() and path.read_text(encoding="utf-8") == args["content"]


# Registre : « comment savoir si cet outil a déjà fait son effet ? ».
# Un outil absent d'ici est considéré idempotent (lectures) et sera simplement rejoué.
APPLIED_CHECKS = {"save_memory": _memory_applied, "write_file": _write_applied}


def already_applied(name: str, raw_args: str) -> bool:
    """True si l'effet d'un appel au résultat inconnu est déjà visible."""
    check = APPLIED_CHECKS.get(name)
    if check is None:
        return False
    try:
        return bool(check(json.loads(raw_args)))
    except Exception:
        return False


# --- 4. la tâche : démarrage OU reprise, même boucle -------------------------

def run_task(task_id: str, prompt: str | None, cfg: RunConfig = RunConfig(),
             confirm=v5.confirm_cli, faults: dict | None = None,
             context_fn=None, memory_fn=None) -> RunResult:
    """prompt=None => reprise d'une tâche existante."""
    faults = faults or {}
    journal = Journal(RUNS_DIR / task_id / "events.jsonl")
    events = journal.read()
    session_start = time.monotonic()  # le délai total repart à zéro à chaque reprise

    if prompt is not None:
        if events:
            raise ValueError(f"la tâche {task_id} existe déjà : utilise resume")
        system = v5.SYSTEM_PROMPT.format(memory=(memory_fn or v5.load_memory)())
        journal.append("task_started", prompt=prompt, system=system, cfg=asdict(cfg))
        messages, steps, unknown, finished = [
            {"role": "system", "content": system}, {"role": "user", "content": prompt}
        ], 0, set(), None
    else:
        if not events:
            raise ValueError(f"aucune tâche {task_id}")
        messages, steps, unknown, finished = rebuild(events)
        if finished:
            print(f"  [resume] tâche déjà terminée ({finished['status']}), rien à refaire")
            return RunResult(finished["status"], finished.get("answer", ""),
                             finished.get("steps", steps), 0.0)
        print(f"  [resume] {steps} étape(s) rejouée(s) depuis le journal, "
              f"{len(unknown)} appel(s) au résultat inconnu")

    executed = 0  # appels réellement exécutés dans cette session (pour les pannes simulées)

    def result(status: str, answer: str = "") -> RunResult:
        return RunResult(status, answer, steps, time.monotonic() - session_start)

    def finish(status: str, answer: str = "") -> RunResult:
        journal.append("task_finished", status=status, answer=answer, steps=steps)
        return result(status, answer)

    while True:
        # a) d'abord solder les appels d'outils en attente (cas de la reprise)
        for call in pending_calls(messages):
            name, raw = call["function"]["name"], call["function"]["arguments"]
            recovered = False
            if call["id"] in unknown and already_applied(name, raw):
                out = v5.ok(f"already applied before the interruption: {name}")
                recovered = True
                print(f"  [resume] {name} déjà appliqué, non rejoué")
            else:
                journal.append("tool_started", call_id=call["id"], name=name, arguments=raw)
                out = v5.execute_tool(name, raw, cfg, confirm)
                executed += 1
                if faults.get("crash_after_exec") == executed:
                    raise SimulatedCrash(f"crash simulé après l'exécution de {name}")
            flag = "ok " if out["ok"] else "ERR"
            print(f"  [tool:{flag}] {name}({raw})" + ("  (récupéré)" if recovered else ""))
            tool_msg = {"role": "tool", "tool_call_id": call["id"],
                        "content": json.dumps(out, ensure_ascii=False)}
            journal.append("tool_done", call_id=call["id"], name=name, ok=out["ok"],
                           recovered=recovered, message=tool_msg)
            messages.append(tool_msg)
            unknown.discard(call["id"])

        # b) limites (mêmes règles que v5)
        remaining = cfg.timeout_s - (time.monotonic() - session_start)
        if remaining <= 0:
            return finish("timeout")
        if steps >= cfg.max_steps:
            return finish("max_steps")

        # c) un tour de modèle
        try:
            response = v5.client.chat.completions.create(
                model=v5.MODEL, messages=context_fn(messages) if context_fn else messages,
                tools=v5.TOOL_SCHEMAS, timeout=remaining)
        except Exception as e:
            # Panne transitoire (Ollama coupé) : on NE clôt PAS la tâche, elle reste reprenable.
            return result("error", f"{type(e).__name__}: {e}")

        m = response.choices[0].message
        msg = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                for c in m.tool_calls
            ]
        steps += 1
        journal.append("model_response", step=steps, message=msg)
        messages.append(msg)
        if faults.get("crash_after_model") == steps:
            raise SimulatedCrash("crash simulé juste après la réponse du modèle")
        if not m.tool_calls:
            return finish("done", msg["content"])


# --- 5. CLI ------------------------------------------------------------------

def show(task_id: str) -> None:
    for e in Journal(RUNS_DIR / task_id / "events.jsonl").read():
        t = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
        detail = {
            "task_started": lambda: e["prompt"],
            "model_response": lambda: (e["message"].get("content") or "")[:60]
                                      + f" [{len(e['message'].get('tool_calls', []))} appel(s)]",
            "tool_started": lambda: f"{e['name']}({e['arguments']})",
            "tool_done": lambda: f"{e['name']} ok={e['ok']}" + (" RÉCUPÉRÉ" if e["recovered"] else ""),
            "task_finished": lambda: f"{e['status']} en {e['steps']} étape(s)",
        }.get(e["type"], lambda: "")()
        print(f"  {t}  {e['type']:<15} {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("prompt")
    p_run.add_argument("--crash-after-exec", type=int)
    p_run.add_argument("--crash-after-model", type=int)
    p_res = sub.add_parser("resume")
    p_res.add_argument("task_id")
    sub.add_parser("list")
    p_show = sub.add_parser("show")
    p_show.add_argument("task_id")
    args = ap.parse_args()

    if args.cmd == "list":
        for d in sorted(RUNS_DIR.glob("*/")) if RUNS_DIR.is_dir() else []:
            events = Journal(d / "events.jsonl").read()
            prompt = events[0]["prompt"][:50] if events else "?"
            print(f"  {d.name}  {task_status(events):<26} {prompt}")
        return 0
    if args.cmd == "show":
        show(args.task_id)
        return 0

    faults = {}
    if args.cmd == "run":
        task_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        if args.crash_after_exec:
            faults["crash_after_exec"] = args.crash_after_exec
        if args.crash_after_model:
            faults["crash_after_model"] = args.crash_after_model
        prompt = args.prompt
    else:
        task_id, prompt = args.task_id, None

    print(f"tâche {task_id}")
    try:
        r = run_task(task_id, prompt, faults=faults)
    except SimulatedCrash as e:
        print(f"\n  *** {e} ***\n  reprends avec : python v6.py resume {task_id}")
        return 3
    except KeyboardInterrupt:
        print(f"\n  interrompu. reprends avec : python v6.py resume {task_id}")
        return 130
    print(f"\nassistant: {r.answer}")
    print(f"  [run] status={r.status} steps={r.steps} elapsed={r.elapsed:.1f}s")
    return 0 if r.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())


# --- EXERCICES ---------------------------------------------------------------
# 1. Reprise sans doublon (le cœur de v6) :
#      python v6.py run "Je m'appelle Amine, retiens-le" --crash-after-exec 1
#    Le crash tombe APRÈS l'écriture dans memory.md mais AVANT tool_done.
#      python v6.py show <id>      -> tool_started sans tool_done
#      python v6.py resume <id>    -> "déjà appliqué, non rejoué"
#    Vérifie que memory.md contient le fait UNE seule fois.
#    Puis retire le test already_applied dans run_task et recommence : doublon.
#
# 2. Arrêt à froid : lance une tâche longue, Ctrl+C en plein milieu, `list`,
#    puis `resume`. Quelles étapes sont rejouées depuis le journal (gratuit)
#    et lesquelles appellent réellement le modèle ?
#
# 3. Journal tronqué : ouvre events.jsonl, coupe la dernière ligne en deux,
#    lance resume. Le journal se répare tout seul (voir Journal.read).