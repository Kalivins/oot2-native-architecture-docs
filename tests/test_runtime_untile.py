# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Checks of the CPU untiler (runtime/gpu/untile.c, runtime/include/oot2/untile.h; architecture D19, the texture upload) against
the layout functions of the NVN2 front end (runtime/nvn/nvn_tex.c, the texture and sampler work), without a device, through
tests/runtime/untile_test.c, built like tests/test_runtime_gpu.py's programs at -O1 and -O2:

- the two GOB orders (the Tegra X1's and the Switch 2's, offset bits 5 and 6 exchanged) as literal tables read off
  Lichii's untiling shader, which is a source the untiler does not share an assumption with;
- a table of sixteen formats, twelve sizes, one level and a full chain, then 900 seeded descriptions over every target
  (2D, array, 3D, cube, cube array, 1D, 1D array, rectangle), block heights and MINIMAL_LAYOUT: each region's base and span
  agree with the layout functions, and texels written into storage by a walk that uses no formula (a counter that follows
  the storage's own order and the table of pieces) come out of the untiler as written, in both orders;
- the shapes that have no region (sparse, multisampled, buffer, linear, an unsized format) say so by name.

Without clang or the Vulkan SDK the program tests skip, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set.

Run: python -m unittest discover -s tests -p test_runtime_untile.py -v
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import LEVELS, ROOT, build_program, case_env, clang_or_skip, fatal_lines, vulkan_or_skip

TEST_SOURCE = ROOT / "tests" / "runtime" / "untile_test.c"
SYNC_SOURCE = ROOT / "tests" / "runtime" / "texsync_test.c"
CASES = ("order", "geometry", "roundtrip", "stops")
SYNC_CASES = ("first", "again", "layers", "untracked", "linear", "stops")


class UntileTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-untile-", ignore_cleanup_errors=True)
        cls.exes = build_program(cc, include, glslang, TEST_SOURCE, Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, timeout: int = 300) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=timeout, env=case_env())

    def test_cases(self) -> None:
        for case in CASES:
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertEqual(fatal_lines(done.stderr), [])


class TexSyncTest(unittest.TestCase):
    """Which regions of a CPU-written texture are stale (runtime/gpu/texsync.c), with the write watch standing in."""

    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-texsync-", ignore_cleanup_errors=True)
        cls.exes = build_program(cc, include, glslang, SYNC_SOURCE, Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def test_cases(self) -> None:
        for case in SYNC_CASES:
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=120, env=case_env())
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertEqual(fatal_lines(done.stderr), [])


if __name__ == "__main__":
    unittest.main()
