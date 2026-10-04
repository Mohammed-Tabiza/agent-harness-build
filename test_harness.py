"""Tests déterministes du harnais : aucun modèle réel, aucune dépendance (unittest).

    python -m unittest test_harness -v

Version 2 : 26 scénarios historiques + régressions + reprise en processus séparés.
Aucun appel réseau ni SDK requis. Placer ce fichier à côté de harness.py.

Les tests RegressionTests expriment le comportement attendu : ils échouent tant
que les défauts du harness fourni ne sont pas corrigés. Aucun expectedFailure
ne masque ces défauts. Les tests LimitsTests documentent les limites actuelles.
Le budget testé est celui de l'estimateur, pas celui du tokenizer du modèle.
Les arrêts os._exit simulent une mort de processus, pas une coupure électrique.

Commandes ciblées :
    python -m unittest test_harness.RegressionTests -v
    python -m unittest test_harness.ProcessRecoveryTests -v

Le scénario historique big_file conserve un état de callback en mémoire ;
les scénarios ProcessRecoveryTests vérifient séparément le redémarrage réel.
"""
import json
import os
import subprocess
import sys
import threading
from unittest.mock import patch
import re
import tempfile
import time
import unittest
from pathlib import Path

import harness as H

CANARY = "CANARY-7421"


# --- faux modèles ------------------------------------------------------------

def call(i, name, args):
    return {"id": f"c{i}", "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def say(text):
    return {"role": "assistant", "content": text}


def act(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


USAGE = {"prompt_tokens": 10, "completion_tokens": 5}


class ScriptedModel:
    """Rejoue une liste de messages assistant (ou lève une exception)."""
    def __init__(self, script):
        self.script, self.calls, self.views = list(script), 0, []

    def chat(self, messages, tools=None, timeout=60):
        self.calls += 1
        self.views.append(messages)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return H.ModelReply(item, dict(USAGE))


class FnModel:
    """Modèle piloté par callback ; le callback peut conserver un état de test."""
    def __init__(self, fn):
        self.fn, self.calls, self.views = fn, 0, []

    def chat(self, messages, tools=None, timeout=60):
        self.calls += 1
        self.views.append(messages)
        return H.ModelReply(self.fn(messages, self.calls), dict(USAGE))


def paired(view):
    """Chaque tool_call a sa réponse, dans l'ordre (sinon l'API réelle rejette la requête)."""
    ids = [c["id"] for m in view if m["role"] == "assistant" for c in m.get("tool_calls", [])]
    return ids == [m["tool_call_id"] for m in view if m["role"] == "tool"]


class Base(unittest.TestCase):
    def make(self, model=None, run_cfg=None, ctx_cfg=None, policy=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        h = H.Harness(self.tmp.name, model=model or ScriptedModel([]),
                      run_cfg=run_cfg or H.RunConfig(), ctx_cfg=ctx_cfg or H.ContextConfig(),
                      policy=policy or H.Policy.permissive(), log=lambda s: None)
        (h.workspace.root / "a.txt").write_text("hello", encoding="utf-8")
        self.addCleanup(h.executor._pool.shutdown, wait=True)
        return h

    def journal_events(self, h, task_id):
        return h.store.journal(task_id).read()


# --- outils : validation, chemins, erreurs, timeout --------------------------

class ToolTests(Base):
    def ex(self, h, name, args):
        return h.executor.execute(name, args if isinstance(args, str) else json.dumps(args))

    def test_path_escape_blocked_on_every_file_tool(self):
        h = self.make()
        (h.root / "memory.md").write_text(f"- {CANARY}\n")
        for name, args in [("read_file", {"filename": "../memory.md"}),
                           ("read_chunk", {"filename": "../memory.md"}),
                           ("write_file", {"filename": "../evil.txt", "content": "x"}),
                           ("append_file", {"filename": "../evil.txt", "content": "x"}),
                           ("read_file", {"filename": "/etc/passwd"})]:
            r = self.ex(h, name, args)
            self.assertEqual(r["error_type"], "forbidden_path", name)
            self.assertNotIn(CANARY, r["content"])
        self.assertFalse((h.root / "evil.txt").exists())

    def test_symlink_cannot_escape(self):
        h = self.make()
        (h.root / "secret.txt").write_text(CANARY)
        try:
            (h.workspace.root / "link.txt").symlink_to(h.root / "secret.txt")
        except OSError:
            self.skipTest("symlinks indisponibles")
        self.assertEqual(self.ex(h, "read_file", {"filename": "link.txt"})["error_type"], "forbidden_path")

    def test_structured_errors_instead_of_crashes(self):
        h = self.make()
        self.assertEqual(self.ex(h, "hack", {})["error_type"], "unknown_tool")
        self.assertEqual(self.ex(h, "read_file", "{bad")["error_type"], "bad_arguments")
        self.assertIn("missing", self.ex(h, "read_file", {})["content"])
        self.assertIn("unexpected", self.ex(h, "read_file", {"filename": "a.txt", "x": 1})["content"])
        self.assertIn("integer", self.ex(h, "read_chunk", {"filename": "a.txt", "offset": "5"})["content"])
        self.assertIn("integer", self.ex(h, "read_chunk", {"filename": "a.txt", "offset": True})["content"])
        self.assertEqual(self.ex(h, "read_file", {"filename": "nope.txt"})["error_type"], "not_found")
        self.assertEqual(self.ex(h, "read_chunk", {"filename": "a.txt", "offset": -1})["error_type"], "bad_arguments")

    def test_tool_exception_and_timeout_become_results(self):
        h = self.make(run_cfg=H.RunConfig(tool_timeout_s=0.1))
        h.catalog.register(H.Tool("boom", "x", lambda: 1 / 0))
        h.catalog.register(H.Tool("slow", "x", lambda: (time.sleep(0.6), H.ok("late"))[1]))
        h.catalog.register(H.Tool("bad", "x", lambda: "pas un dict"))
        self.assertEqual(self.ex(h, "boom", {})["error_type"], "tool_failed")
        self.assertEqual(self.ex(h, "slow", {})["error_type"], "timeout")
        self.assertEqual(self.ex(h, "bad", {})["error_type"], "tool_failed")

    def test_output_truncated_and_large_file_paginated(self):
        h = self.make(run_cfg=H.RunConfig(max_tool_output=300))
        (h.workspace.root / "big.txt").write_text("x" * 10000)
        r = self.ex(h, "read_chunk", {"filename": "big.txt", "offset": 0, "length": 2000})
        self.assertLessEqual(len(r["content"]), 300 + 40)
        self.assertIn("truncated", r["content"])
        h2 = self.make()
        (h2.workspace.root / "big.txt").write_text("x" * 10000)
        r = self.ex(h2, "read_file", {"filename": "big.txt"})
        self.assertIn("read_chunk", r["content"])
        self.assertIn("next offset=3000", self.ex(h2, "read_chunk", {"filename": "big.txt", "offset": 1500})["content"])
        self.assertIn("end of file", self.ex(h2, "read_chunk", {"filename": "big.txt", "offset": 99999})["content"])


# --- politique ---------------------------------------------------------------

class PolicyTests(Base):
    def test_write_requires_confirmation_and_denial_is_enforced(self):
        asked = []
        h = self.make(policy=H.Policy(confirm_fn=lambda n, a: asked.append((n, a)) or False))
        r = h.executor.execute("write_file", json.dumps({"filename": "b.txt", "content": "x"}))
        self.assertEqual(r["error_type"], "denied")
        self.assertFalse((h.workspace.root / "b.txt").exists())
        self.assertEqual(asked[0][0], "write_file")
        # une lecture ne demande rien
        h.executor.execute("read_file", json.dumps({"filename": "a.txt"}))
        self.assertEqual(len(asked), 1)

    def test_confirmed_write_goes_through(self):
        h = self.make(policy=H.Policy(confirm_fn=lambda n, a: True))
        r = h.executor.execute("write_file", json.dumps({"filename": "b.txt", "content": "x"}))
        self.assertTrue(r["ok"])

    def test_override_deny_and_unknown_permission_fails_closed(self):
        h = self.make(policy=H.Policy(overrides={"read_file": "deny"}, confirm_fn=lambda n, a: True))
        self.assertEqual(h.executor.execute("read_file", json.dumps({"filename": "a.txt"}))["error_type"], "denied")
        h.catalog.register(H.Tool("net", "x", lambda: H.ok("y"), permission="inconnue"))
        self.assertEqual(h.executor.execute("net", "{}")["error_type"], "denied")


# --- exécuteur de tâches : limites, journal, reprise -------------------------

class RunnerTests(Base):
    def test_done_and_journal_is_structured(self):
        h = self.make(ScriptedModel([act(call(1, "read_file", {"filename": "a.txt"})), say("fini")]))
        r = h.run("lis a.txt")
        self.assertEqual((r.status, r.answer, r.steps), ("done", "fini", 2))
        self.assertEqual((r.prompt_tokens, r.completion_tokens), (20, 10))
        ev = self.journal_events(h, r.task_id)
        self.assertTrue(all(e["task_id"] == r.task_id for e in ev))
        self.assertEqual([e["type"] for e in ev],
                         ["task_started", "model_response", "tool_started", "tool_done",
                          "model_response", "task_finished"])
        done = next(e for e in ev if e["type"] == "tool_done")
        self.assertIn("duration_ms", done)
        self.assertEqual(next(e for e in ev if e["type"] == "model_response")["usage"], USAGE)
        self.assertTrue(paired(r.messages))
        self.assertEqual(h.show(r.task_id)[0].split()[1], "task_started")

    def test_every_tool_call_gets_an_answer_even_in_error(self):
        h = self.make(ScriptedModel([act(call(1, "hack", {}), call(2, "read_file", "{bad"),
                                         call(3, "read_file", {"filename": "../x"})), say("ok")]))
        r = h.run("x")
        self.assertEqual(r.status, "done")
        self.assertTrue(paired(r.messages))
        errs = [json.loads(m["content"])["error_type"] for m in r.messages if m["role"] == "tool"]
        self.assertEqual(errs, ["unknown_tool", "bad_arguments", "forbidden_path"])

    def test_max_steps_is_final(self):
        h = self.make(ScriptedModel([act(call(i, "list_files", {})) for i in range(1, 9)]),
                      run_cfg=H.RunConfig(max_steps=3))
        r = h.run("x")
        self.assertEqual((r.status, r.steps), ("max_steps", 3))
        self.assertTrue(paired(r.messages))
        self.assertEqual(h.tasks()[0][1], "max_steps")

    def test_total_timeout(self):
        h = self.make(ScriptedModel([act(call(i, "slow", {})) for i in range(1, 30)]),
                      run_cfg=H.RunConfig(timeout_s=0.5, tool_timeout_s=5, max_steps=50))
        h.catalog.register(H.Tool("slow", "x", lambda: (time.sleep(0.3), H.ok("z"))[1]))
        self.assertEqual(h.run("x").status, "timeout")

    def test_model_error_leaves_task_resumable(self):
        m = ScriptedModel([RuntimeError("serveur coupé")])
        h = self.make(m)
        r = h.run("salut")
        self.assertEqual(r.status, "error")
        self.assertTrue(h.tasks()[0][1].startswith("interrompue"))
        m.script = [say("re-salut")]
        r2 = h.resume(r.task_id)
        self.assertEqual((r2.status, r2.answer), ("done", "re-salut"))

    def test_crash_after_exec_does_not_duplicate_a_memory_write(self):
        h = self.make(ScriptedModel([act(call(1, "save_memory", {"fact": "Amine"}))]))
        with self.assertRaises(H.SimulatedCrash):
            h.run("retiens Amine", faults={"crash_after_exec": 1})
        tid = h.last_task_id
        self.assertEqual(self.journal_events(h, tid)[-1]["type"], "tool_started")
        self.assertTrue(h.tasks()[0][1].startswith("interrompue"))
        h.model.script = [say("retenu")]
        r = h.resume(tid)
        self.assertEqual((r.status, r.answer), ("done", "retenu"))
        self.assertEqual(h.memory.load().count("Amine"), 1)
        self.assertTrue(any(e.get("recovered") for e in self.journal_events(h, tid)))
        self.assertTrue(paired(r.messages))

    def test_control_without_applied_check_duplicates(self):
        """Contre-épreuve : sans vérification déclarée, la reprise rejoue (et duplique)."""
        h = self.make(ScriptedModel([act(call(1, "save_memory", {"fact": "Amine"}))]))
        old = h.catalog.get("save_memory")
        h.catalog.register(H.Tool(old.name, old.description, old.fn, old.properties, old.required,
                                  permission=old.permission, applied_check=None))
        with self.assertRaises(H.SimulatedCrash):
            h.run("x", faults={"crash_after_exec": 1})
        h.model.script = [say("ok")]
        h.resume(h.last_task_id)
        self.assertEqual(h.memory.load().count("Amine"), 2)

    def test_append_file_recovery(self):
        h = self.make(ScriptedModel([act(call(1, "append_file", {"filename": "n.md", "content": "ligne"}))]))
        with self.assertRaises(H.SimulatedCrash):
            h.run("x", faults={"crash_after_exec": 1})
        h.model.script = [say("ok")]
        h.resume(h.last_task_id)
        self.assertEqual((h.workspace.root / "n.md").read_text().count("ligne"), 1)

    def test_crash_after_model_executes_pending_calls_on_resume(self):
        h = self.make(ScriptedModel([act(call(1, "read_file", {"filename": "a.txt"}),
                                         call(2, "write_file", {"filename": "b.txt", "content": "x"}))]))
        with self.assertRaises(H.SimulatedCrash):
            h.run("go", faults={"crash_after_model": 1})
        self.assertFalse((h.workspace.root / "b.txt").exists())
        h.model.script = [say("fini")]
        r = h.resume(h.last_task_id)
        self.assertEqual(r.status, "done")
        self.assertEqual((h.workspace.root / "b.txt").read_text(), "x")
        self.assertEqual(h.model.calls, 2)           # un seul NOUVEL appel au modèle
        self.assertTrue(paired(r.messages))

    def test_finished_task_resume_never_calls_the_model(self):
        h = self.make(ScriptedModel([say("fait")]))
        r = h.run("x")
        h.model.script = []
        before = h.model.calls
        r2 = h.resume(r.task_id)
        self.assertEqual((r2.status, r2.answer, h.model.calls), ("done", "fait", before))

    def test_truncated_journal_is_repaired(self):
        h = self.make(ScriptedModel([act(call(1, "list_files", {})), say("fin")]))
        r = h.run("x")
        p = h.store.journal(r.task_id).path
        lines = p.read_text().splitlines()
        p.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:25])
        j = h.store.journal(r.task_id)
        self.assertEqual(len(j.read()), len(lines) - 1)
        j.append("task_finished", status="done", answer="fin", steps=2)
        self.assertEqual(len(j.read()), len(lines))

    def test_unknown_or_duplicate_task(self):
        h = self.make(ScriptedModel([say("a")]))
        with self.assertRaises(ValueError):
            h.resume("inconnue")
        r = h.run("x")
        with self.assertRaises(ValueError):
            h._runner().start("y", task_id=r.task_id)


# --- contexte ----------------------------------------------------------------

def tmsg(i, content):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps({"ok": True, "content": content})}


def unit(i, content, name="read_chunk"):
    return [act(call(i, name, {"filename": "f"})), tmsg(i, content)]


class ContextTests(Base):
    BASE = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "GOAL"}]

    def cm(self, **kw):
        cfg = H.ContextConfig(**{"budget_tokens": 2000, **kw})
        return H.ContextManager(cfg, H.ToolCatalog(H.local_tools(
            H.Workspace(tempfile.mkdtemp()), H.Memory(Path(tempfile.mkdtemp()) / "m.md"))).schemas(),
            log=lambda s: None)

    def tokens(self, cm, view):
        return cm.overhead + sum(H.est_tokens(m, cm.cfg.chars_per_token) for m in view)

    def test_under_budget_nothing_changes(self):
        msgs = self.BASE + unit(1, "petit") + unit(2, "petit")
        cm = self.cm()
        self.assertEqual(cm.build(msgs), msgs)
        self.assertEqual((cm.covered, cm.shrinking), (0, False))

    def test_view_always_under_budget_valid_and_history_intact(self):
        cm, msgs, prev = self.cm(), list(self.BASE), 0
        for i in range(1, 31):
            msgs += unit(i, f"contenu {i} " + "x" * 2400)
            v = cm.build(msgs)
            self.assertLessEqual(self.tokens(cm, v), cm.cfg.budget_tokens, i)
            self.assertTrue(paired(v), i)
            self.assertEqual((v[1], v[-1]), (self.BASE[1], msgs[-1]))
            self.assertGreaterEqual(cm.covered, prev)
            prev = cm.covered
        self.assertIn("Earlier steps", v[0]["content"])
        self.assertEqual(len(msgs), 62)                       # l'histoire complète n'est pas touchée
        self.assertLessEqual(len(cm.summary), cm.cfg.summary_max_chars + 25)

    def test_emergency_shrinks_even_the_latest_result(self):
        msgs = self.BASE + unit(1, "y" * 30000)
        cm = self.cm()
        v = cm.build(msgs)
        self.assertLessEqual(self.tokens(cm, v), cm.cfg.budget_tokens)
        self.assertTrue(cm.stats["emergency"])
        json.loads(v[-1]["content"])                           # JSON toujours valide

    def test_hysteresis_limits_llm_summaries(self):
        model = ScriptedModel([RuntimeError("down")] * 100)
        cm = H.ContextManager(H.ContextConfig(budget_tokens=2000, llm_summary=True), [], model, lambda s: None)
        msgs = list(self.BASE)
        for i in range(1, 26):
            msgs += unit(i, "z" * 600)
            cm.build(msgs)
        self.assertTrue(1 <= model.calls <= 10, model.calls)   # pas un appel par étape
        self.assertTrue(cm.summary)                            # repli sur le digest

    def test_bounded_memory_keeps_most_recent(self):
        h = self.make()
        h.memory.path.write_text("".join(f"- fait numéro {i}\n" for i in range(200)), encoding="utf-8")
        b = h.memory.bounded(300)
        self.assertLessEqual(len(b), 330)
        self.assertIn("omitted", b)
        self.assertIn("fait numéro 199", b)
        self.assertNotIn("fait numéro 0\n", b)
        h.memory.path.write_text("- court\n", encoding="utf-8")
        self.assertEqual(h.memory.bounded(300), "- court\n")

    def test_big_file_end_to_end_with_crash_and_resume(self):
        """Un faux modèle lit 18 morceaux, prend des notes, retrouve les 3 codes cachés,
        malgré un budget qui force la compaction ET un crash au milieu."""
        h = self.make(ctx_cfg=H.ContextConfig(budget_tokens=1800),
                      run_cfg=H.RunConfig(max_steps=80, max_tool_output=2500))
        rng_words = "logistique entrepôt palette convoi quai inventaire commande retard".split()
        codes = {2: "CODE-ALPHA-4821", 12: "CODE-BRAVO-1937", 23: "CODE-CHARLIE-7305"}
        secs = [f"Section {k}. " + " ".join(rng_words[(k * j) % 8] for j in range(130))
                + (f" Réf : {codes[k]}." if k in codes else "") for k in range(1, 25)]
        (h.workspace.root / "big.txt").write_text("\n\n".join(secs), encoding="utf-8")
        size = len((h.workspace.root / "big.txt").read_text(encoding="utf-8"))
        self.assertGreater(size, 15000)

        state = {"next": 0, "n": 0}

        def fn(view, n):
            last = view[-1]
            state["n"] += 1
            nid = state["n"]
            if last["role"] == "user":
                return act(call(f"r{nid}", "read_chunk", {"filename": "big.txt", "offset": 0}))
            body = json.loads(last["content"])["content"]
            if "[chars" in body:
                found = re.findall(r"CODE-[A-Z]+-\d+", body)
                m = re.search(r"next offset=(\d+)", body)
                state["next"] = int(m.group(1)) if m else None
                return act(call(f"a{nid}", "append_file", {"filename": "notes.md", "content": ", ".join(found) or "rien"}))
            if "appended" in body or "already applied" in body:
                if state["next"] is None:
                    return act(call(f"n{nid}", "read_file", {"filename": "notes.md"}))
                return act(call(f"r{nid}", "read_chunk", {"filename": "big.txt", "offset": state["next"]}))
            return say("Codes : " + ", ".join(sorted(set(re.findall(r"CODE-[A-Z]+-\d+", body)))))

        h.model = FnModel(fn)
        with self.assertRaises(H.SimulatedCrash):
            h.run("lis big.txt par morceaux", faults={"crash_after_exec": 6})
        r = h.resume(h.last_task_id)

        self.assertEqual(r.status, "done")
        for c in codes.values():
            self.assertIn(c, r.answer)
        n_chunks = -(-size // 1500)
        self.assertEqual(len((h.workspace.root / "notes.md").read_text().splitlines()), n_chunks)  # sans doublon
        cm = h.model                                           # toutes les vues : sous budget, valides, ancrées
        for v in cm.views:
            self.assertTrue(paired(v))
            self.assertEqual(v[1]["content"], "lis big.txt par morceaux")
            self.assertTrue(v[0]["content"].startswith("You are a helpful personal assistant"))
        overhead = H.est_tokens(h.catalog.schemas(), 3.5)      # même estimateur que le harnais
        for v in cm.views:
            self.assertLessEqual(overhead + sum(H.est_tokens(m, 3.5) for m in v), 1800)
        done = [e for e in self.journal_events(h, r.task_id) if e["type"] == "tool_done"]
        self.assertGreaterEqual(len(done), 30)                 # le journal a tout gardé




# --- régressions : exigences, sans masquage des défauts ----------------------

class RegressionTests(Base):
    def assert_malformed_rejected(self, payload):
        h = self.make()
        h.catalog.register(H.Tool("malformed", "test", lambda: payload))
        try:
            out = h.executor.execute("malformed", "{}")
        except Exception as exc:
            self.fail(f"Exception non structurée : {type(exc).__name__}: {exc}")
        self.assertIs(out.get("ok"), False, out)
        self.assertEqual(out.get("error_type"), "tool_failed", out)
        self.assertIsInstance(out.get("content"), str)

    def test_missing_ok_is_rejected(self):
        self.assert_malformed_rejected({"content": "x"})

    def test_non_string_content_is_rejected(self):
        self.assert_malformed_rejected({"ok": True, "content": None})

    def test_non_boolean_ok_is_rejected(self):
        self.assert_malformed_rejected({"ok": "yes", "content": "x"})

    def assert_context_not_sent_over_budget(self, messages):
        # Contrat : produire une vue sous budget OU refuser explicitement.
        # Une erreur accidentelle (TypeError, KeyError...) ne compte pas comme refus.
        cm = H.ContextManager(H.ContextConfig(budget_tokens=200), [], log=lambda s: None)
        try:
            view = cm.build(messages)
        except Exception as exc:
            self.assertIn(type(exc).__name__, {"ContextBudgetExceeded", "ContextBudgetError"},
                          f"Refus attendu via une exception métier explicite, reçu : {exc!r}")
            return
        cost = cm.overhead + sum(H.est_tokens(m, cm.cfg.chars_per_token) for m in view)
        self.assertLessEqual(cost, 200, "Une vue impossible à réduire doit être refusée")

    def test_oversized_goal_is_rejected_or_fits(self):
        self.assert_context_not_sent_over_budget([
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "x" * 5000}])

    def test_oversized_tool_arguments_are_rejected_or_fit(self):
        self.assert_context_not_sent_over_budget([
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "GO"},
            act(call(1, "write_file", {"filename": "a", "content": "x" * 5000})),
            tmsg(1, "written")])

    def test_final_response_recovered_without_model_call(self):
        h = self.make(ScriptedModel([say("réponse originale")]))
        with self.assertRaises(H.SimulatedCrash):
            h.run("question", faults={"crash_after_model": 1})
        h.model = ScriptedModel([say("réponse indésirable")])
        result = h.resume(h.last_task_id)
        self.assertEqual(h.model.calls, 0, "La réponse finale existe déjà dans le journal")
        self.assertEqual((result.status, result.answer), ("done", "réponse originale"))

    def test_deadline_checked_before_each_pending_tool(self):
        # Horloge contrôlée : aucun seuil de performance ni sleep fragile.
        clock, effects = [0.0], []
        h = self.make(ScriptedModel([
            act(call(1, "first", {}), call(2, "second", {}))]),
            run_cfg=H.RunConfig(timeout_s=1))
        def first():
            effects.append("first")
            clock[0] = 2.0
            return H.ok("done")
        h.catalog.register(H.Tool("first", "test", first))
        h.catalog.register(H.Tool("second", "test", lambda: (effects.append("second"), H.ok("done"))[1]))
        with patch.object(H.time, "monotonic", side_effect=lambda: clock[0]):
            result = h.run("go")
        self.assertEqual(result.status, "timeout")
        self.assertEqual(effects, ["first"], "Un nouvel outil démarre après la deadline")


class LimitsTests(Base):
    def test_thread_timeout_does_not_cancel_running_effect(self):
        """Caractérisation de l'exécuteur à threads, pas une garantie d'annulation."""
        h = self.make(run_cfg=H.RunConfig(tool_timeout_s=0.05))
        started, release, completed = threading.Event(), threading.Event(), threading.Event()
        path = h.workspace.root / "late.txt"
        def delayed_write():
            started.set()
            if not release.wait(5):
                return H.err("test_timeout", "test did not release worker")
            path.write_text("late", encoding="utf-8")
            completed.set()
            return H.ok("written")
        h.catalog.register(H.Tool("delayed", "test", delayed_write, permission="write"))
        try:
            out = h.executor.execute("delayed", "{}")
            self.assertTrue(started.wait(2))
            self.assertEqual(out["error_type"], "timeout")
            self.assertFalse(path.exists())
        finally:
            release.set()
        self.assertTrue(completed.wait(2))
        self.assertEqual(path.read_text(encoding="utf-8"), "late")


# Exécuté dans deux interpréteurs indépendants. Aucun ScriptedModel ni dictionnaire
# de progression n'est transmis du processus interrompu au processus de reprise.
PROCESS_WORKER = r'''
import json, os, sys
from pathlib import Path
import harness as H
root, phase, crash_point = sys.argv[1:]
class Model:
    def __init__(self): self.calls = 0
    def chat(self, messages, tools=None, timeout=60):
        self.calls += 1
        if phase == "crash":
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"id": "append-1", "type": "function", "function": {
                    "name": "append_file", "arguments": json.dumps({
                        "filename": "notes.md", "content": "UNIQUE-MARKER"})}}]}
        else:
            assert messages[-1]["role"] == "tool", "Appel en attente non soldé"
            assert json.loads(messages[-1]["content"])["ok"] is True
            message = {"role": "assistant", "content": "reprise terminée"}
        return H.ModelReply(message, {"prompt_tokens": 0, "completion_tokens": 0})
model = Model()
h = H.Harness(root, model=model, policy=H.Policy.permissive(), log=lambda s: None)
if phase == "crash":
    try:
        h._runner().start("ajoute une ligne", task_id="process-test", faults={crash_point: 1})
    except H.SimulatedCrash:
        os._exit(73)
    raise AssertionError("Crash attendu non déclenché")
else:
    result = h.resume("process-test")
    print(json.dumps({"status": result.status, "answer": result.answer,
                      "calls": model.calls, "events": h.store.journal("process-test").read()}))
    h.executor._pool.shutdown(wait=True)
'''


class ProcessRecoveryTests(unittest.TestCase):
    def check_restart(self, crash_point, before_exists):
        with tempfile.TemporaryDirectory() as root:
            env = dict(os.environ)
            module_dir = str(Path(H.__file__).resolve().parent)
            env["PYTHONPATH"] = module_dir + os.pathsep + env.get("PYTHONPATH", "")
            def launch(phase):
                return subprocess.run(
                    [sys.executable, "-c", PROCESS_WORKER, root, phase, crash_point],
                    cwd=module_dir, env=env, capture_output=True, text=True,
                    encoding="utf-8", timeout=15)
            crashed = launch("crash")
            self.assertEqual(crashed.returncode, 73, crashed.stdout + crashed.stderr)
            notes = Path(root) / "workspace" / "notes.md"
            self.assertEqual(notes.exists(), before_exists)
            resumed = launch("resume")
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            data = json.loads(resumed.stdout)
            self.assertEqual((data["status"], data["answer"], data["calls"]),
                             ("done", "reprise terminée", 1))
            self.assertEqual(notes.read_text(encoding="utf-8").splitlines(), ["UNIQUE-MARKER"])
            dones = [e for e in data["events"] if e["type"] == "tool_done"]
            self.assertEqual(len(dones), 1)
            self.assertEqual(dones[0]["recovered"], before_exists)
            self.assertEqual(data["events"][-1]["type"], "task_finished")

    def test_process_restart_after_model_before_tool(self):
        self.check_restart("crash_after_model", False)

    def test_process_restart_after_effect_before_tool_done(self):
        self.check_restart("crash_after_exec", True)


if __name__ == "__main__":
    unittest.main()
