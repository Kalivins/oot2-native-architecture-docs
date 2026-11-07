# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the step runner of `oot2 prepare` (tools/prepare/runner.py) and its state file (tools/prepare/state.py).

The steps are synthetic: each one logs that it ran, takes its own key from a dictionary the test changes, and keeps
its "output" as a flag the test can remove. No game file is read.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, PrepareError, Reporter, Step, run_steps  # noqa: E402


class FakeStep(Step):
    def __init__(self, id, needs=(), world=None, always=False):
        self.id, self.needs, self.always, self.world = id, tuple(needs), always, world

    def key(self, ctx):
        return self.world["inputs"].get(self.id, "v1")

    def current(self, ctx, record):
        return self.id in self.world["outputs"]

    def verify(self, ctx, record):
        return self.id not in self.world["damaged"]

    def run(self, ctx):
        self.world["log"].append(self.id)
        stop = self.world["stop"].pop(self.id, None)
        if stop == "interrupt":
            raise KeyboardInterrupt
        if stop == "fail":
            raise PrepareError("synthetic failure")
        if stop == "crash":
            raise RuntimeError("synthetic bug")
        self.world["outputs"].add(self.id)
        self.world["damaged"].discard(self.id)
        return {"bytes": 10}


def new_world():
    return {"inputs": {}, "outputs": set(), "damaged": set(), "stop": {}, "log": []}


# A graph with two branches: check -> extract -> lift -> compile, and extract -> shaders.
GRAPH = (("check", (), True), ("extract", ("check",), False), ("lift", ("extract",), False),
         ("shaders", ("extract",), False), ("compile", ("lift",), False))


def steps_for(world, graph=GRAPH):
    return [FakeStep(i, needs, world, always) for i, needs, always in graph]


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2-runner-")
        self.dest = Path(self.tmp.name) / "out"
        self.world = new_world()

    def tearDown(self):
        self.tmp.cleanup()

    def run_once(self, verify=False, only=None):
        """One run as `oot2 prepare` does it: load the state, run, return (result, steps run, printed text)."""
        self.world["log"] = []
        out, err = io.StringIO(), io.StringIO()
        ctx = Context(self.dest, None, Reporter(out, err), state=statefile.load(self.dest), verify=verify)
        result = run_steps(steps_for(self.world), ctx, only=only)
        return result, list(self.world["log"]), out.getvalue() + err.getvalue()

    def recorded(self):
        return set(statefile.load(self.dest)["steps"])

    def test_first_run_runs_every_step_in_order_and_records_all_but_the_check(self):
        result, ran, _ = self.run_once()
        self.assertEqual(result, "ok")
        self.assertEqual(ran, ["check", "extract", "lift", "shaders", "compile"])
        self.assertEqual(self.recorded(), {"extract", "lift", "shaders", "compile"})

    def test_second_run_keeps_everything_but_the_check(self):
        self.run_once()
        result, ran, text = self.run_once()
        self.assertEqual((result, ran), ("ok", ["check"]))
        self.assertIn("compile: kept", text)

    def test_interrupted_run_resumes_at_the_step_that_did_not_finish(self):
        for how, expected in (("interrupt", "cancelled"), ("fail", "failed"), ("crash", "failed")):
            with self.subTest(how=how):
                self.tearDown()
                self.setUp()
                self.world["stop"]["lift"] = how
                result, ran, text = self.run_once()
                self.assertEqual(result, expected, text)
                self.assertEqual(ran, ["check", "extract", "lift"])
                self.assertEqual(self.recorded(), {"extract"})
                result, ran, _ = self.run_once()
                self.assertEqual(result, "ok")
                self.assertEqual(ran, ["check", "lift", "shaders", "compile"])

    def test_step_stopped_while_rerunning_is_not_taken_for_finished(self):
        self.run_once()
        self.world["outputs"].discard("lift")          # its output went missing: it runs again with the same key
        self.world["stop"]["lift"] = "interrupt"       # and is stopped halfway
        self.assertEqual(self.run_once()[0], "cancelled")
        self.assertNotIn("lift", self.recorded())
        self.world["outputs"].add("lift")              # a half-written output that looks present
        result, ran, text = self.run_once()
        self.assertEqual(ran, ["check", "lift"])
        self.assertIn("lift: started (not done yet)", text)

    def test_killed_process_leaves_a_whole_state_file(self):
        """A run killed without any chance to clean up (os._exit in a step) keeps the steps finished before it."""
        script = textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {str(ROOT / 'tools')!r})
            from prepare.runner import Context, Reporter, Step, run_steps
            class S(Step):
                def __init__(self, id, needs=()):
                    self.id, self.needs = id, needs
                def run(self, ctx):
                    if self.id == "lift":
                        os._exit(9)
                    return {{"bytes": 1}}
            ctx = Context({str(self.dest)!r}, None, Reporter())
            run_steps([S("extract"), S("lift", ("extract",)), S("compile", ("lift",))], ctx)
        """)
        r = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 9, r.stderr)
        self.assertEqual(set(statefile.load(self.dest)["steps"]), {"extract"})

    def test_changed_input_reruns_that_step_and_downstream_only(self):
        cases = {
            "lift": ["check", "lift", "compile"],                       # not extract, not the shaders branch
            "extract": ["check", "extract", "lift", "shaders", "compile"],
            "shaders": ["check", "shaders"],
            "compile": ["check", "compile"],
        }
        for changed, expected in cases.items():
            with self.subTest(changed=changed):
                self.tearDown()
                self.setUp()
                self.run_once()
                self.world["inputs"][changed] = "v2"
                result, ran, text = self.run_once()
                self.assertEqual((result, ran), ("ok", expected), text)
                self.assertIn(f"{changed}: started (its inputs changed)", text)
                self.assertEqual(self.run_once()[1], ["check"])   # and the new keys are recorded

    def test_missing_output_reruns_that_step_only(self):
        self.run_once()
        self.world["outputs"].discard("extract")
        result, ran, text = self.run_once()
        self.assertEqual(ran, ["check", "extract"])
        self.assertIn("extract: started (its output is missing)", text)

    def test_verify_reruns_a_damaged_output_only_when_asked(self):
        self.run_once()
        self.world["damaged"].add("lift")
        self.assertEqual(self.run_once()[1], ["check"])
        result, ran, text = self.run_once(verify=True)
        self.assertEqual(ran, ["check", "lift"])
        self.assertIn("lift: started (its output failed the full check)", text)

    def test_only_runs_the_named_steps_and_says_what_the_others_would_do(self):
        result, ran, text = self.run_once(only={"check"})
        self.assertEqual((result, ran), ("ok", ["check"]))
        self.assertIn("extract: would run (not done yet)", text)
        self.assertEqual(self.recorded(), set())

    def test_steps_must_name_earlier_steps(self):
        bad = [FakeStep("lift", ("extract",), self.world), FakeStep("extract", (), self.world)]
        ctx = Context(self.dest, None, Reporter(io.StringIO(), io.StringIO()))
        with self.assertRaises(ValueError):
            run_steps(bad, ctx)
        with self.assertRaises(ValueError):
            run_steps([FakeStep("a", (), self.world), FakeStep("a", (), self.world)], ctx)

    def test_failure_message_names_the_step(self):
        self.world["stop"]["extract"] = "fail"
        result, _, text = self.run_once()
        self.assertEqual(result, "failed")
        self.assertIn("oot2 prepare: step extract: synthetic failure", text)


class StateFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2-state-")
        self.dest = Path(self.tmp.name)
        self.path = self.dest / statefile.NAME

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_and_absence(self):
        self.assertIsNone(statefile.load(self.dest))
        data = statefile.empty()
        data["steps"]["extract"] = {"key": "ab", "finished": "2026-11-06T00:00:00Z", "seconds": 1.5, "bytes": 3}
        statefile.save(self.dest, data)
        self.assertEqual(statefile.load(self.dest), data)
        self.assertFalse((self.dest / (statefile.NAME + ".tmp")).exists())

    def test_damaged_files_are_refused_with_a_plain_message(self):
        good = {"key": "ab", "finished": "t", "seconds": 1}
        cases = {
            "truncated": b'{"format": 1, "inputs": {}, "steps": {"extract": {"key": "ab"',
            "empty": b"",
            "binary": b"\xff\xfe\x00garbage",
            "list": b"[]",
            "no format": json.dumps({"inputs": {}, "steps": {}}).encode(),
            "format text": json.dumps({"format": "1", "inputs": {}, "steps": {}}).encode(),
            "steps not an object": json.dumps({"format": 1, "inputs": {}, "steps": []}).encode(),
            "inputs missing": json.dumps({"format": 1, "steps": {}}).encode(),
            "host not an object": json.dumps({"format": 1, "inputs": {}, "steps": {}, "host": 3}).encode(),
            "record not an object": json.dumps({"format": 1, "inputs": {}, "steps": {"lift": 1}}).encode(),
            "record without key": json.dumps(
                {"format": 1, "inputs": {}, "steps": {"lift": {**good, "key": ""}}}).encode(),
            "record without time": json.dumps(
                {"format": 1, "inputs": {}, "steps": {"lift": {"key": "ab", "seconds": 1}}}).encode(),
            "record without duration": json.dumps(
                {"format": 1, "inputs": {}, "steps": {"lift": {"key": "ab", "finished": "t"}}}).encode(),
        }
        for label, raw in cases.items():
            with self.subTest(label):
                self.path.write_bytes(raw)
                with self.assertRaises(statefile.StateError) as cm:
                    statefile.load(self.dest)
                message = str(cm.exception)
                self.assertIn(str(self.path), message)
                self.assertIn("Nothing was run", message)
                self.assertNotIn("Traceback", message)

    def test_save_that_fails_midway_leaves_the_previous_file_whole(self):
        data = statefile.empty()
        data["steps"]["extract"] = {"key": "ab", "finished": "t", "seconds": 1}
        statefile.save(self.dest, data)
        broken = statefile.empty()
        broken["steps"]["extract"] = dict(data["steps"]["extract"])
        broken["steps"]["lift"] = {"key": "cd", "finished": "t", "seconds": 1, "bytes": object()}
        with self.assertRaises(TypeError):
            statefile.save(self.dest, broken)       # json.dump has written part of the file when it fails
        self.assertEqual(statefile.load(self.dest), data)

    def test_newer_format_is_refused(self):
        self.path.write_text(json.dumps({"format": statefile.FORMAT + 1, "inputs": {}, "steps": {}}))
        with self.assertRaises(statefile.NewerStateError) as cm:
            statefile.load(self.dest)
        self.assertIn("newer oot2", str(cm.exception))


class CommandLineStateTest(unittest.TestCase):
    """`oot2 prepare` refuses a damaged state file before running anything, and leaves the file as it was."""

    def test_damaged_state_file_stops_the_command(self):
        with tempfile.TemporaryDirectory(prefix="oot2-cli-") as tmp:
            dest = Path(tmp)
            raw = b'{"format": 1, "steps": {"extract": '
            (dest / statefile.NAME).write_bytes(raw)
            r = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "oot2.py"), "prepare", "--dest", tmp,
                                "--check"], capture_output=True, text=True, env=dict(os.environ, OOT2_CC="no-such-cc"))
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("is damaged: it is not valid JSON", r.stderr)
            self.assertIn("Nothing was run", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertEqual(r.stdout, "")
            self.assertEqual((dest / statefile.NAME).read_bytes(), raw)
            self.assertEqual(sorted(p.name for p in dest.iterdir()), [statefile.NAME])


if __name__ == "__main__":
    unittest.main()
