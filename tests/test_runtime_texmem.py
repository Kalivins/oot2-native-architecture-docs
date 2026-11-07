# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Checks of the staging ring and the texture arena (runtime/gpu/texmem.c, runtime/include/oot2/texmem.h; architecture
D19 and section 8, ticket 65) without a device, through tests/runtime/texmem_test.c, built like
tests/test_runtime_gpu.py's programs at -O1 and -O2, one case a process:

- the ring: regions share chunks, never overlap, a chunk comes back once its submission is done, and at the 384 MiB cap
  a request waits for the oldest submission instead of growing; the chunks never held more than the cap;
- the arena: aligned disjoint extents, a freed block joined back whole, a block of its own for an image larger than a
  block, the 3.4 GiB cap (54 blocks of 64 MiB), empty blocks given back to make room for another memory type, and the
  memory budget's edge as the mirrors' rule has it;
- what no wait or reclaim can serve stops by name: one submission asking more than the ring holds, a region larger than
  the ring, the arena at its cap, the arena past the device's budget.

Without clang or the Vulkan SDK the program tests skip, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set.

Run: python -m unittest discover -s tests -p test_runtime_texmem.py -v
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXIT_FATAL, LEVELS, ROOT, build_program, case_env, clang_or_skip, fatal_lines, \
    vulkan_or_skip

TEST_SOURCE = ROOT / "tests" / "runtime" / "texmem_test.c"
PASSING = ("ring", "arena")
# case: text the one FATAL line must hold
FAILING = {
    "ring_full": "the staging ring is full inside one submission",
    "ring_big": "the staging ring holds at most 402653184 bytes",
    "arena_full": "the texture arena is full (the arena's cap)",
    "arena_budget": "the texture arena is full (the device's memory budget)",
}


class TexMemTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-texmem-", ignore_cleanup_errors=True)
        cls.exes = build_program(cc, include, glslang, TEST_SOURCE, Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, timeout: int = 120) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=timeout, env=case_env())

    def test_passing_cases(self) -> None:
        for case in PASSING:
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertEqual(fatal_lines(done.stderr), [])

    def test_what_nothing_can_serve_stops_by_name(self) -> None:
        for case, text in FAILING.items():
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                    self.assertNotIn("returned", done.stdout)
                    fatal = fatal_lines(done.stderr)
                    self.assertEqual(len(fatal), 1, done.stderr)
                    self.assertIn(text, fatal[0])


if __name__ == "__main__":
    unittest.main()
