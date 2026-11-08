# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/shaders/pool.py and tools/shaders/validate.py.

The pool is exercised with a stand-in translator written in Python, which
speaks the same line protocol: no game data, no .NET build and no Vulkan SDK.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import pool  # noqa: E402
import validate  # noqa: E402

# A stand-in for `oot2-xlate --serve`: it answers each job by its "act" field.
FAKE = textwrap.dedent("""
    import json, os, sys, base64
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
            module = base64.b64encode(b"other-module")
            desc = base64.b64encode(b'{"schema":2}')
            print(f"ok other 5 {module.decode()} {desc.decode()}", flush=True)
        elif act == "nokey":
            print("fail - 0 bad_job: key does not match the pattern", flush=True)
        elif act == "slow":
            import time
            time.sleep(0.05)
            module = base64.b64encode(b"m" * 1024)
            desc = base64.b64encode(b'{"schema":2}')
            print(f"ok {job['key']} 250 {module.decode()} {desc.decode()}", flush=True)
        elif act == "garbage":
            print("hello", flush=True)
        elif act == "noparts":          # a protocol 1 reply: the size, no module and no description
            print(f"ok {job['key']} 250 1024", flush=True)
        elif act == "oneparts":
            print(f"ok {job['key']} 250 {base64.b64encode(b'm' * 64).decode()}", flush=True)
        elif act == "emptydesc":
            print(f"ok {job['key']} 250 {base64.b64encode(b'm' * 64).decode()} ", flush=True)
        elif act == "badbase64":
            print(f"ok {job['key']} 250 !!!! ????", flush=True)
        elif act == "cut":              # the reply ends in the middle, with no line break, and the child is gone
            sys.stdout.write(f"ok {job['key']} 250 " + base64.b64encode(b"m" * 1024).decode()[:100])
            sys.stdout.flush()
            os._exit(0)
        else:
            module = base64.b64encode(b"m" * 1024)
            desc = base64.b64encode(b'{"schema":2}')
            print(f"ok {job['key']} 250 {module.decode()} {desc.decode()}", flush=True)
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


# A stand-in that speaks protocol 1: it writes its own files and answers with the size.
OLD = textwrap.dedent("""
    import json, sys
    if sys.argv[1:] == ["--version"]:
        print("oot2-xlate " + "0" * 64 + " proto=1 spirv=1.3 target=vulkan1.3")
        sys.exit(0)
    for line in sys.stdin:
        job = json.loads(line)["job"]
        print(f"ok {job['key']} 250 1024", flush=True)
""")


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

    def test_the_returned_results_hold_no_module_bytes(self):
        # A full prepare translates millions of bytes of modules: the list the pool returns keeps the metadata
        # only, and the caller's sink is the one place the bytes go.
        sink: dict[str, tuple[bytes, bytes]] = {}
        jobs = [({"key": f"k{i}", "act": "ok"}, None) for i in range(8)]
        results, _ = pool.run_pool(self.fake.command, jobs, 2, self.dir / "logs", timeout=20.0,
                                   on_result=lambda r: sink.__setitem__(r.key, (r.module, r.description)))
        self.assertEqual(sorted(r.key for r in results), sorted(f"k{i}" for i in range(8)))
        for r in results:
            self.assertEqual((r.status, r.spv_bytes), ("ok", 1024))
            self.assertEqual((r.module, r.description), (b"", b""), r.key)
        self.assertEqual(sorted(sink), sorted(f"k{i}" for i in range(8)))
        for key, (module, desc) in sink.items():
            self.assertEqual(module, b"m" * 1024, key)
            self.assertEqual(desc, b'{"schema":2}', key)
        self.assertFalse((self.dir / "out").exists())      # no folder was given, so nothing is written

    def test_an_ok_reply_without_its_two_parts_fails_the_job_by_name(self):
        acts = ["noparts", "oneparts", "emptydesc", "badbase64", "cut", "ok"]
        got, stats = self.run_jobs(acts, workers=1)
        for i, why in enumerate(("does not carry a module and a description", "does not carry a module and a description",
                                 "does not carry a module and a description", "is not base64", "in the middle of its reply")):
            r = got[f"k{i}"]
            self.assertIn(r.status, ("fail", "crash"), r)
            self.assertIn(why, r.message, r)
            self.assertEqual((r.module, r.description, r.spv_bytes), (b"", b"", 0), r)
            self.assertFalse((self.dir / "out" / f"k{i}.spv").exists(), r)
            self.assertFalse((self.dir / "out" / f"k{i}.res.json").exists(), r)
        for i in (0, 1, 2):
            self.assertIn(f"k{i}", got[f"k{i}"].message)           # named by the job's key
        self.assertEqual(got["k4"].status, "crash")                # the child died mid-line and was replaced
        self.assertEqual(got["k5"].status, "ok")                   # the pool went on
        self.assertEqual(stats.restarts, 1)

    def test_files_are_written_to_tmp_names_and_renamed_module_first(self):
        out = self.dir / "out"
        seen = []
        real = os.replace

        def spy(src, dst, *a, **kw):
            seen.append((Path(dst).name, Path(src).name, sorted(p.name for p in out.iterdir())))
            real(src, dst, *a, **kw)

        with mock.patch.object(pool.os, "replace", spy):
            got, _ = self.run_jobs(["ok"], workers=1)
        self.assertEqual(got["k0"].status, "ok")
        self.assertEqual([s[:2] for s in seen], [("k0.spv", "k0.spv.tmp"), ("k0.res.json", "k0.res.json.tmp")])
        # at the first rename both parts are still under their .tmp names; no final name exists yet
        self.assertEqual(seen[0][2], ["k0.res.json.tmp", "k0.spv.tmp"])
        self.assertEqual(seen[1][2], ["k0.res.json.tmp", "k0.spv"])
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["k0.res.json", "k0.spv"])
        self.assertEqual((out / "k0.spv").read_bytes(), b"m" * 1024)
        self.assertEqual((out / "k0.res.json").read_bytes(), b'{"schema":2}')

    def test_a_failed_write_leaves_no_tmp_file_and_fails_the_job(self):
        out = self.dir / "out"
        real = os.replace

        def failing(src, dst, *a, **kw):
            if str(dst).endswith(".res.json"):
                raise PermissionError("denied")
            real(src, dst, *a, **kw)

        with mock.patch.object(pool.os, "replace", failing):
            got, _ = self.run_jobs(["ok"], workers=1)
        self.assertEqual(got["k0"].status, "fail")
        self.assertIn("cannot be written", got["k0"].message)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["k0.spv"])   # the module landed whole, no .tmp stays

    def test_an_out_folder_is_refused_by_name_for_a_protocol_1_translator(self):
        old = self.dir / "old_xlate.py"
        old.write_text(OLD, encoding="utf-8")
        command = [sys.executable, "-I", str(old)]
        with self.assertRaisesRegex(ValueError, "speaks protocol 1.*out folder needs protocol 2"):
            pool.run_pool(command, [({"key": "k0"}, self.dir / "out")], 1, self.dir / "logs", timeout=20.0)
        self.assertFalse((self.dir / "out").exists())

    def test_the_protocol_is_asked_once_for_each_command(self):
        asked = []
        real = pool.subprocess.run

        def counting(cmd, *a, **kw):
            asked.append(list(cmd))
            return real(cmd, *a, **kw)

        with mock.patch.object(pool.subprocess, "run", counting):
            for _ in range(3):
                self.run_jobs(["ok"], workers=1)
        self.assertEqual([c[-1] for c in asked], ["--version"])    # a stand-in without --version is let through

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
