# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the census timer (runtime/core/hostfn.c).

Builds tests/runtime/hostfn_timer_test.c with runtime/core against a synthetic
layout, at -O1 and -O2, and runs its cases in their own processes. Host
functions burn thread CPU time, sleep, or call each other and a guest
function; no game code is involved.

- timed (census mode and the timer on): a function is charged its own thread
  CPU time and not a nested host call's (A burns 20 ms and dispatches B, which
  burns 10 ms: A is about 20, not 30), not a guest callback's (C calls guest
  code that burns 10 ms and dispatches D: C is small, the guest bucket about
  10, D about 5), and not time spent blocked (S sleeps 30 ms: about 0). Bounds
  are one-sided and loose, since a loaded machine stretches a burn.
- nested: 20000 empty calls from one parent leave the parent nearly nothing,
  because the calibrated cost of the timer per nested call is subtracted.
- default (no flag; and the timer variable without OOT2_CENSUS=1): no table, no
  thread, no total, and the registry's own counts as before.

Needs clang for x86-64 (OOT2_CC, else clang on PATH); without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402

CORE = ROOT / "runtime" / "core"
TEST_SOURCE = ROOT / "tests" / "runtime" / "hostfn_timer_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


def slots(out: str) -> dict[str, tuple[int, float]]:
    return {m[1]: (int(m[2]), float(m[3])) for m in re.finditer(r"^(\w+) calls (\d+) excl_ms ([\d.]+)$", out, re.M)
            if m[1] != "guest"}


class HostFnTimerTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-hostfn-timer-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": 0x8000}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"hostfn_timer_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(ROOT / "runtime" / "include"), "-I", str(CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, **env: str) -> subprocess.CompletedProcess:
        e = {k: v for k, v in os.environ.items() if not k.startswith("OOT2_CENSUS")}
        e.update(env)
        return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=120, env=e)

    def passes(self, case: str, **env: str) -> list[subprocess.CompletedProcess]:
        runs = []
        for level in LEVELS:
            with self.subTest(level=level, case=case, env=env):
                done = self.run_case(level, case, **env)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}\n", done.stdout)
                runs.append(done)
        return runs

    def test_a_function_is_charged_its_own_time_not_a_nested_call_a_guest_callback_or_a_wait(self) -> None:
        for done in self.passes("timed", OOT2_CENSUS="1", OOT2_CENSUS_TIMER="1"):
            s = slots(done.stdout)
            self.assertEqual({k: v[0] for k, v in s.items()}, {"A": 1, "B": 1, "C": 1, "D": 1, "S": 1})
            self.assertTrue(20 <= s["A"][1] < 26, s)       # an inclusive timer says about 30
            self.assertTrue(10 <= s["B"][1] < 14, s)
            self.assertLess(s["C"][1], 3, s)               # an inclusive timer says about 15
            self.assertTrue(5 <= s["D"][1] < 8, s)
            self.assertLess(s["S"][1], 5, s)               # a wall-clock timer says 30
            m = re.search(r"^guest calls (\d+) excl_ms ([\d.]+)$", done.stdout, re.M)
            self.assertEqual(int(m[1]), 1)
            self.assertTrue(10 <= float(m[2]) < 14, m[0])  # the guest function's own burn, without D's
            self.assertIn("classes nvn 0 submit 0 other 5 guest 1\n", done.stdout)   # none of them is an NVN2 name
            self.assertIn("OOT2_CENSUS_TIMER=1: every host function call is timed", done.stderr)

    def test_the_calibrated_cost_of_the_timer_is_taken_from_the_parent_of_many_calls(self) -> None:
        for done in self.passes("nested", OOT2_CENSUS="1", OOT2_CENSUS_TIMER="1"):
            s = slots(done.stdout)
            self.assertEqual(s["Z"][0], 100000)            # five calls of the parent, 20,000 children each
            overhead_ns = int(re.search(r"^overhead_ns (\d+) ", done.stdout, re.M)[1])
            self.assertGreater(overhead_ns, 0)
            runs = [(float(m[1]), float(m[2])) for m in re.finditer(r"^run \d many_ms ([\d.]+) z_ms ([\d.]+)$", done.stdout, re.M)]
            self.assertEqual(len(runs), 5, done.stdout)
            # Not taken out, the timer's cost would leave the parent 20,000 x overhead (about 16 to 27 ms), and the same sum
            # inside Z's own readings. Calibrated, each reading is a few ms: the bound is half of that sum, on the smallest of
            # five readings because a loaded machine only adds to one (a single run read 8.3 ms at the old fixed bound of 8).
            bound = 0.5 * 20000 * overhead_ns / 1e6
            self.assertLess(min(r[0] for r in runs), bound, (runs, overhead_ns))
            self.assertLess(min(r[1] for r in runs), bound, (runs, overhead_ns))   # the cost inside an empty call's own reading is taken too

    def test_without_the_flag_nothing_is_timed_and_nothing_is_kept(self) -> None:
        self.passes("default")
        for done in self.passes("default", OOT2_CENSUS_TIMER="1"):   # the timer variable alone: census mode is not on
            self.assertIn("OOT2_CENSUS_TIMER is set without OOT2_CENSUS=1: the timer stays off", done.stderr)
        for done in self.passes("default", OOT2_CENSUS="1"):
            self.assertNotIn("is timed", done.stderr)


if __name__ == "__main__":
    unittest.main()
