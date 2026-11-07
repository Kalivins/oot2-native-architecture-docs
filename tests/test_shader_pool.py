# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/shaders/pool.py and tools/shaders/validate.py.

The pool is exercised with a stand-in translator written in Python, which
speaks the same line protocol: no game data, no .NET build and no Vulkan SDK.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import pool  # noqa: E402
import validate  # noqa: E402

# A stand-in for `oot2-xlate --serve`: it answers each job by its "act" field.
FAKE = textwrap.dedent("""
    import json, os, sys
    for line in sys.stdin:
        req = json.loads(line)
        job = req["job"]
        act = job.get("act", "ok")
        if act == "die":
            os._exit(3)
        if act == "hang":
            sys.stdin.readline()
        if act == "fail":
            print(f"fail {job['key']} 12 unsupported opcode", flush=True)
        elif act == "wrongkey":
            print(f"ok other 5 10", flush=True)
        elif act == "nokey":
            print("fail - 0 bad_job: key does not match the pattern", flush=True)
        elif act == "slow":
            import time
            time.sleep(0.05)
            print(f"ok {job['key']} 250 1024", flush=True)
        elif act == "garbage":
            print("hello", flush=True)
        else:
            print(f"ok {job['key']} 250 1024", flush=True)
""")


class FakeTranslator:
    def __init__(self, folder: Path):
        self.script = folder / "fake_xlate.py"
        self.script.write_text(FAKE, encoding="utf-8")
        self.command = [sys.executable, "-I", str(self.script)]


class ParseReplyTest(unittest.TestCase):
    def test_ok_and_fail(self):
        self.assertEqual(pool.parse_reply("ok abc 250 1024\n"), ("ok", "abc", 250, "1024"))
        self.assertEqual(pool.parse_reply("fail abc 12 unsupported opcode 0x243"),
                         ("fail", "abc", 12, "unsupported opcode 0x243"))

    def test_malformed(self):
        for line in ("", "hello", "ok abc", "maybe abc 1 2", "ok abc notanumber 2"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                pool.parse_reply(line)


class RunPoolTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.fake = FakeTranslator(self.dir)

    def tearDown(self):
        self._tmp.cleanup()

    def run_jobs(self, acts, workers=2, timeout=20.0):
        jobs = [({"key": f"k{i}", "act": a}, self.dir / "out") for i, a in enumerate(acts)]
        results, stats = pool.run_pool(self.fake.command, jobs, workers, self.dir / "logs", timeout=timeout)
        return {r.key: r for r in results}, stats

    def test_every_job_gets_one_result(self):
        got, stats = self.run_jobs(["ok"] * 9, workers=3)
        self.assertEqual(sorted(got), sorted(f"k{i}" for i in range(9)))
        self.assertTrue(all(r.status == "ok" and r.micros == 250 and r.spv_bytes == 1024 for r in got.values()))
        self.assertEqual(stats.restarts, 0)

    def test_failures_are_reported_not_hidden(self):
        got, _ = self.run_jobs(["fail", "wrongkey", "garbage", "ok", "nokey"], workers=1)
        self.assertEqual(got["k0"].status, "fail")
        self.assertIn("unsupported opcode", got["k0"].message)
        self.assertEqual(got["k1"].status, "fail")
        self.assertIn("reply for other", got["k1"].message)
        self.assertEqual(got["k2"].status, "fail")
        self.assertEqual(got["k3"].status, "ok")
        self.assertEqual(got["k4"].status, "fail")
        self.assertIn("bad_job: key does not match", got["k4"].message)

    def test_a_dead_child_is_replaced(self):
        got, stats = self.run_jobs(["ok", "die", "ok", "ok"], workers=1)
        self.assertEqual(got["k1"].status, "crash")
        self.assertEqual([got[k].status for k in ("k0", "k2", "k3")], ["ok"] * 3)
        self.assertEqual(stats.restarts, 1)

    @unittest.skipUnless(sys.platform == "win32", "working sets are sampled on Windows only")
    def test_over_the_memory_limit_children_are_recycled_not_stalled(self):
        # A limit no child can meet: the pool must still finish, by replacing children.
        jobs = [({"key": f"k{i}", "act": "slow"}, self.dir / "out") for i in range(40)]
        results, stats = pool.run_pool(self.fake.command, jobs, 2, self.dir / "logs", timeout=20.0, memory_limit=1)
        self.assertEqual(sorted(r.key for r in results), sorted(f"k{i}" for i in range(40)))
        self.assertTrue(all(r.status == "ok" for r in results))
        self.assertGreater(stats.recycled, 0)
        self.assertEqual(stats.restarts, 0)

    def test_a_silent_child_times_out(self):
        got, stats = self.run_jobs(["hang", "ok"], workers=1, timeout=2.0)
        self.assertEqual(got["k0"].status, "timeout")
        self.assertEqual(got["k1"].status, "ok")
        self.assertEqual(stats.restarts, 1)


class ValidateHelpersTest(unittest.TestCase):
    def test_percentile_nearest_rank(self):
        values = list(range(1, 101))
        self.assertEqual(validate.percentile(values, 50), 50.0)
        self.assertEqual(validate.percentile(values, 99), 99.0)
        self.assertEqual(validate.percentile(values, 100), 100.0)
        self.assertEqual(validate.percentile([7], 99), 7.0)
        self.assertEqual(validate.percentile([], 50), 0.0)

    def test_timing_summary_in_milliseconds(self):
        s = validate.timing_summary([1000, 2000, 3000])
        self.assertEqual((s["n"], s["p50_ms"], s["max_ms"], s["sum_s"]), (3, 2.0, 3.0, 0.006))

    def test_first_error_names_the_rule(self):
        text = "warning: x\nerror: line 12: Capability Float64 is not allowed\n  %1 = OpCapability\n"
        self.assertEqual(validate.first_error(text), "error: line 12: Capability Float64 is not allowed")
        self.assertEqual(validate.first_error("something odd\n"), "something odd")


if __name__ == "__main__":
    unittest.main()
