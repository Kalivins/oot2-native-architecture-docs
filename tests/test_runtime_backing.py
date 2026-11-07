# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the Vulkan back end's backing map (runtime/gpu/backing.c, runtime/include/oot2/backing.h) that
need no GPU: the decision of each kind of page, imports laid over committed pages only (through the commit query of
oot2/memory.h), chunks committed after a pool was mapped taken in through the commit epoch, frees deferred until their
value is done, pool_fini waiting for the submissions that used its pool, a bind across an import window, an import the
device refuses, VRAM blocks made ahead of use, 4,660 pools made and finalized with nothing left over, and the binds the
map refuses by name. The device stands in as host memory (tests/runtime/backing_test.c says how); the device's own
tests are tests/test_runtime_vk.py.

Builds tests/runtime/backing_test.c like tests/test_runtime_gpu.py's programs (build_program), at -O1 and -O2, and runs
each case in its own process. The resolver's cost per lookup is printed by the bench case for the research note; it is
measured, not judged.

Needs clang and a Vulkan SDK with the headers and Volk, as tests/test_runtime_gpu.py does.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXIT_FATAL, LEVELS, ROOT, build_program, case_env, clang_or_skip, fatal_lines, \
    vulkan_or_skip

TEST_SOURCE = ROOT / "tests" / "runtime" / "backing_test.c"

PASSING = ("decisions", "uncommitted", "epoch", "deferred", "finiwait", "cross", "refused", "blocks", "flat")

# case: text the one FATAL line must hold
FAILING = {
    "span_pools": "gfx: vulkan: backing map: a bind of [0x80000ff000, 0x8000101000) runs past the end of its "
                  "CPU_NO_ACCESS pool [0x8000000000, 0x8000100000) by 0x1000 bytes: it would need two backings",
    "span_virtual": "reaches pages no mapping request mapped (16 of 32 pages mapped): it would need two backings",
    "import_fails": "the device refused to import committed host memory",
    "mirror": "the mirror hook chose a mirror for [0x",
}


class BackingTest(unittest.TestCase):
    """backing_test.c with the runtime's core, NVN2 front end and gpu module, without a device."""

    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-backing-", ignore_cleanup_errors=True)
        cls.exes = build_program(cc, include, glslang, TEST_SOURCE, Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, timeout: int = 120) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=timeout,
                              env=case_env())

    def test_passing_cases(self) -> None:
        for case in PASSING:
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertEqual(fatal_lines(done.stderr), [])

    def test_binds_the_map_cannot_back_as_one_stop_by_name(self) -> None:
        for case, text in FAILING.items():
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                    self.assertNotIn("returned", done.stdout)
                    fatal = fatal_lines(done.stderr)
                    self.assertEqual(len(fatal), 1, done.stderr)
                    self.assertIn(text, fatal[0])

    def test_the_resolver_cost_per_lookup_is_measured(self) -> None:
        done = self.run_case(LEVELS[1], "bench", timeout=300)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertRegex(done.stdout, r"bench: 4500 pools .* over 4500 of them, .* p50 [\d.]+ ns")
        self.assertRegex(done.stdout, r"bench: 4500 pools .* over 32 of them, .* p50 [\d.]+ ns")
        print("\n" + done.stdout, end="")


if __name__ == "__main__":
    unittest.main()
