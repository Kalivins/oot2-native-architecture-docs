# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the virtual 60 Hz display clock (runtime/core/vsync.c).

Builds tests/runtime/vsync_test.c with runtime/core against a synthetic layout
(one module of 0x8000 bytes, main), at -O1 and -O2 with -march=x86-64-v3, then
runs each case in its own process. The guest programs are C functions written
by hand in the lifter's shape; nothing is read from a dump.

The driven cases replace the clock's time source and wait, and check every
deadline against a schedule computed in the test: an hour of ticks with
wake-ups up to 15 ms late, one wake-up 2.5 ticks late, and a guest thread
woken by a tick while another thread runs on the core it gave up. The
realtime case runs the host's clock for 5 seconds at one optimisation level
(its timing does not depend on code generation) and checks 60 signals a second
per event with the lateness guest threads see under a loose bound (4 ms at
p99), since the runners' timing margins are not measured; set
OOT2_VSYNC_P99_US to hold a quiet desktop to a tighter one. The figures are
printed with the prefix "vsync figure:".

The realtime case holds the system timer at 0.5 ms the way the program does
(runtime/core/timer.c, the timer resolution hold, on by default, in CI too) and prints the
resolution it saw. The desktop's bound is 0.6 ms (the vsync clock study), a variable
of the run and never a default; CI keeps 4 ms:

    OOT2_VSYNC_P99_US=600 py -3.14 -I -m unittest discover -s tests -p test_runtime_vsync.py -v

OOT2_TIMER_RESOLUTION=0 leaves the system's resolution, for an A/B run; this
case passes the environment through, so a shell that sets it runs the case
without the request (the output says which).

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
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
TEST_SOURCE = ROOT / "tests" / "runtime" / "vsync_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000
TIMEOUT = 180

EXIT_FATAL = 70
EXIT_HUNG = 3   # vsync_test.c's watchdog: the case had not ended after 120 s; it logs the clock's state first

HOUR_LINE = "tick 216000 due at origin + 3600.000000000 s"


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class VsyncTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-vsync-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        # Windows: without high-entropy ASLR the host's own stacks stay below 4 GiB, out of the fixed module region
        # (runtime/core/memory.c, rt_mem_init); the program is linked the same way.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"vsync_test{level}{'.exe' if os.name == 'nt' else ''}"
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

    def run_case(self, level: str, case: str) -> subprocess.CompletedProcess:
        try:
            done = subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{case} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")
        if done.returncode == EXIT_HUNG:
            self.fail(f"{case} at {level} hung (the case's watchdog):\n{done.stdout}{done.stderr}")
        return done

    def passes(self, level: str, case: str) -> str:
        done = self.run_case(level, case)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"ok {case}", done.stdout)
        self.assertNotIn("FATAL", done.stderr)
        for line in done.stdout.splitlines():
            if not line.startswith("ok "):
                print(f"vsync figure: {level} {line}")
        return done.stdout

    def test_clock_schedule(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                self.passes(level, "clock")

    def test_hour_of_driven_ticks(self) -> None:
        # Every deadline is checked against the schedule as the clock asks for it; one rounded period fails at once.
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertIn(HOUR_LINE, self.passes(level, "hour"))

    def test_late_wake_up_skips_without_a_burst(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertIn("2 skipped, last tick 22", self.passes(level, "late"))

    def test_waiting_guest_thread_woken_by_the_tick(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertIn("woke at tick 1; the worker ran while it waited: 1", self.passes(level, "wake"))

    def test_realtime_sixty_a_second(self) -> None:
        out = self.passes(LEVELS[-1], "realtime")
        self.assertEqual(out.count("realtime: event "), 4, out)

    def test_drive_after_start_is_refused(self) -> None:
        done = self.run_case(LEVELS[0], "drive_after_start")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn("FATAL: rt_vsync_drive: the clock is already running", done.stderr)
        self.assertNotIn("ok drive_after_start", done.stdout)
        fatal_lines = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
        self.assertEqual(len(fatal_lines), 1, done.stderr)

    def test_unknown_case_is_refused(self) -> None:
        done = self.run_case(LEVELS[0], "no_such_case")
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
