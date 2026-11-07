# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the boundary between generated code and the runtime.

Builds tests/runtime/boundary_test.c with runtime/core against a synthetic
layout written by tools/layout.py, for the target generated code uses
(-march=x86-64-v3, architecture D4) at its optimisation level (-O1) and at
-O2, then runs each case in its own
process: the cases that must end in a fatal error are checked by exit code and
by their FATAL line.

Needs clang for x86-64 (the compiler is OOT2_CC, else clang on PATH). Without
one the tests skip, unless OOT2_REQUIRE_CLANG is set, as continuous
integration sets it: there a missing compiler is a failure.

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
TEST_SOURCE = ROOT / "tests" / "runtime" / "boundary_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")

EXIT_FATAL, EXIT_UNIMPLEMENTED, EXIT_GUEST_STOP = 70, 71, 72

PASSING = ("dispatch", "hostfn", "svc", "setjmp", "exclusive", "ticks")

# case: (exit code, text the FATAL line must contain)
FAILING = {
    "svc_missing": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented svc: 0x29 GetInfo"),
    "missing_import": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented import: _ZN2nn2os12NotThereYetEv"),
    "dispatch_not_start": (EXIT_FATAL, "(main+0x104): not a function start"),
    "dispatch_outside": (EXIT_FATAL, "outside every module"),
    "setjmp_indirect": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented setjmp: setjmp called through a pointer"),
    "brk": (EXIT_GUEST_STOP, "FATAL: guest stop: brk #0x1"),
    "longjmp_stale": (EXIT_FATAL, "no live setjmp filled it"),
    "throw": (EXIT_GUEST_STOP, "FATAL: guest stop: C++ throw of N4test9TestErrorE"),
    "late_register": (EXIT_FATAL, "registered after rt_dispatch_init"),
    "late_svc": (EXIT_FATAL, "rt_svc_register(0x1) after rt_dispatch_init"),
    "refuse_unsorted": (EXIT_FATAL, "is not strictly ascending"),
    "refuse_outside": (EXIT_FATAL, "is misaligned or outside the module"),
    "refuse_overlap": (EXIT_FATAL, "modules main and sdk overlap"),
    "refuse_limit": (EXIT_FATAL, "leaves the module range"),
    "refuse_host_in_module": (EXIT_FATAL, "lies inside module main"),
}


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class BoundaryTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-boundary-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": 0x100000, "sdk": 0x40000}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        for level in LEVELS:
            exe = out / f"boundary_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror",
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
        return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=120)

    def test_passing_cases(self) -> None:
        for level in LEVELS:
            for case in PASSING:
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)

    def test_fatal_cases(self) -> None:
        for level in LEVELS:
            for case, (code, text) in FAILING.items():
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    self.assertNotIn(f"ok {case}", done.stdout)
                    fatal_lines = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
                    self.assertEqual(len(fatal_lines), 1, done.stderr)

    def test_unknown_case_is_refused(self) -> None:
        done = self.run_case(LEVELS[0], "no_such_case")
        self.assertEqual(done.returncode, 2)

    def test_layout_limit_matches_runtime_header(self) -> None:
        header = (ROOT / "runtime" / "include" / "oot2" / "runtime.h").read_text(encoding="utf-8")
        self.assertIn(f"#define RT_MODULE_LIMIT {layout.MODULE_LIMIT:#x}ull", header)


if __name__ == "__main__":
    unittest.main()
