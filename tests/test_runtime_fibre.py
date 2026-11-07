# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of nn::os fibres on host fibres.

Builds tests/runtime/fibre_test.c with runtime/core against a synthetic
layout (one module of 0x8000 bytes, main), at -O1 and -O2 with
-march=x86-64-v3, then runs each case in its own process. The guest programs
are C functions written by hand in the lifter's shape that call the four
public fibre entry points through the host-function addresses their symbols
bind to; nothing is read from a dump. A passing case must print "ok <case>";
a failing case must end in exactly one FATAL line with the expected exit code
and text.

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
TEST_SOURCE = ROOT / "tests" / "runtime" / "fibre_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000
TIMEOUT = 60

EXIT_FATAL, EXIT_GUEST_STOP = 70, 72
EXIT_HUNG = 3   # fibre_test.c's watchdog: the case had not ended after 20 s; it logs the kernel's state first

# basic: creation, a switch and a switch back; returns: a fibre's function hands over to the next, and a finished
# fibre is initialized again; other_thread: a fibre left on one thread resumed by another, from a fibre of its own, and
# handing back to it; migrate_core: the thread
# sleeps on the fibre and wakes on another core, then resumes the fibre after sleeping on its own context; kick_on_fibre
# and pending_moves: the scheduler's kicks reach the Cpu the thread runs; exit_on_fibre: ExitThread on a fibre, then
# the fibre deleted; ring: a hundred fibres in turn, and all deleted.
PASSING = ("basic", "returns", "other_thread", "migrate_core", "kick_on_fibre", "pending_moves", "exit_on_fibre", "ring")

# case: (exit code, text the FATAL line must contain). The guest stops are where the SDK's own code aborts.
FAILING = {
    "switch_running": (EXIT_GUEST_STOP, "FATAL: guest stop: SwitchToFiber to 0x"),
    "finalize_running": (EXIT_GUEST_STOP, "): the fibre is running on thread 0x"),
    "switch_finished": (EXIT_GUEST_STOP, ", which is finished: the SDK aborts"),
    "switch_unknown": (EXIT_GUEST_STOP, ", which is not an initialized fibre: the SDK aborts"),
    "null_own": (EXIT_GUEST_STOP, "FATAL: guest stop: SwitchToFiber(nullptr) on the thread's own context"),
    "host_switch": (EXIT_FATAL, "FATAL: kernel: SwitchToFiber on a host thread, which is no guest thread"),
}
SWITCH_RUNNING_TAIL = ", which is running: the SDK aborts"


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class FibreTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-fibre-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        # Windows: without high-entropy ASLR the host's own stacks stay below 4 GiB, out of the fixed module region
        # (runtime/core/memory.c, rt_mem_init); the program is linked the same way.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"fibre_test{level}{'.exe' if os.name == 'nt' else ''}"
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

    def test_passing_cases(self) -> None:
        for level in LEVELS:
            for case in PASSING:
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)

    def test_counts(self) -> None:
        # One fibre, four switches (in, back, in, its return), none left; a hundred fibres, five rounds, none left.
        done = self.run_case(LEVELS[0], "basic")
        self.assertIn("fibres: created 1, live 0, switches 4", done.stdout, done.stdout + done.stderr)
        self.assertIn("kernel: first InitializeFiber(0x", done.stderr)   # once each, with the caller
        self.assertEqual(done.stderr.count("kernel: first SwitchToFiber(0x"), 1, done.stderr)
        done = self.run_case(LEVELS[0], "ring")
        self.assertIn("fibres: created 100, live 0, switches 502", done.stdout, done.stdout + done.stderr)

    def test_fatal_cases(self) -> None:
        for level in LEVELS:
            for case, (code, text) in FAILING.items():
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    if case == "switch_running":
                        self.assertIn(SWITCH_RUNNING_TAIL, done.stderr)
                    self.assertNotIn(f"ok {case}", done.stdout)
                    fatal_lines = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
                    self.assertEqual(len(fatal_lines), 1, done.stderr)

    def test_unknown_case_is_refused(self) -> None:
        done = self.run_case(LEVELS[0], "no_such_case")
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
