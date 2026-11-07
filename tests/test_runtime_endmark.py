# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the end marker of a module's lifted code (the lifted-code end marker): the address one byte past a module's last
lifted function is host code, and the last function's own last byte is guest code, in the three places that ask:

- lookup: rt_host_pc_module, the lookup of a host address over the lifter's tables (runtime/core/dispatch.c);
- preempt: the asynchronous preemption range check (kern_pc_lifted, runtime/core/kernel.c), which sends an interrupted
  thread to the preemption stub only when its pc is in lifted code;
- whoread: dispatch_host_pc_guest, the host-address index behind "read by guest code" or "by runtime or back-end code"
  of the page ranges (the end-to-end case is in tests/test_runtime_pages.py).

And rt_dispatch_init refuses, with a named fatal error, a module that has no end marker and one whose marker does not
lie above all of its functions. The modules are synthetic bytes laid out by hand in tests/runtime/endmark_test.c;
nothing is read from a dump. Needs clang for x86-64 (OOT2_CC, else clang on PATH); without one the tests skip, unless
OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

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
TEST_SOURCE = ROOT / "tests" / "runtime" / "endmark_test.c"
LEVELS = ("-O1", "-O2")
EXIT_FATAL = 70

FAILING = {
    "no_marker": "dispatch: module sdk has no end marker (sdk_lifted_end)",
    "marker_at_last": "dispatch: main_fn_ptrs[2] at ",
    "marker_below": "dispatch: sdk_fn_ptrs[0] at ",
}
MOVED = "lies at or past the module's end marker"


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class EndMarkTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-endmark-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": 0x8000, "sdk": 0x8000}), out)
        sources = [str(TEST_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"endmark_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(ROOT / "runtime" / "include"), "-I", str(CORE), "-I", str(out),
                   "-I", str(ROOT / "tests" / "runtime"), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=30)

    def passes(self, case: str) -> None:
        for level in LEVELS:
            with self.subTest(level=level, case=case):
                done = self.run_case(level, case)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}", done.stdout)
                self.assertNotIn("FATAL", done.stderr)

    def test_the_byte_past_the_last_function_is_host_code_in_the_lookup(self) -> None:
        self.passes("lookup")

    def test_the_byte_past_the_last_function_is_host_code_for_the_preemption_range_check(self) -> None:
        self.passes("preempt")

    def test_the_byte_past_the_last_function_is_host_code_in_the_who_read_index(self) -> None:
        self.passes("whoread")

    def test_a_module_without_a_marker_or_with_a_marker_below_its_functions_is_refused(self) -> None:
        for level in LEVELS:
            for case, text in FAILING.items():
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    if case != "no_marker":
                        self.assertIn(MOVED, done.stderr)
                    self.assertNotIn(f"ok {case}", done.stdout)
                    self.assertEqual(len([ln for ln in done.stderr.splitlines() if ln.startswith("FATAL")]), 1)

    def test_unknown_case_is_refused(self) -> None:
        self.assertEqual(self.run_case(LEVELS[0], "no_such_case").returncode, 2)


if __name__ == "__main__":
    unittest.main()
