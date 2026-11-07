# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Device tests of the uploads of CPU-written textures (runtime/gpu/vk_texture.c, the arena images of vk_image.c, the
untile pass runtime/gpu/shaders/untile.comp; architecture D19, ticket 65), through tests/runtime/vk_texture_test.c, built
like tests/test_runtime_gpu.py's programs. Every case runs under the validation layer with synchronization validation and
ends with its counts, which must be 0 errors and 0 warnings.

On a device with every item of oot2/vk_required.h (the desktop; the device is the RTX 3060 Ti):

- the compute pass equals the CPU untiler byte for byte over the table of tests/runtime/untile_table.h, in both GOB orders,
  and the table tells the orders apart;
- with OOT2_T65_DIR naming the local folder of real textures of a player's dump and Lichii's untiling of them (game data:
  it never enters the repository): the compute pass and the CPU untiler equal Lichii's answer in the Switch 2's order and
  the X1's order does not; without the folder the case prints a skip line;
- textures the CPU wrote (2D, mips, block-compressed, arrays, 3D, cube, packaged block heights, linear) become images of the
  arena at the first submission that binds them and copy back as written;
- a texture the CPU rewrites is uploaded again, only the regions over the written page, and the next submission reads it;
- 64 MiB textures through the staging ring: the ring and the arena never pass their caps; to the end, the arena's own cap
  stops the run by name.

Without clang or the Vulkan SDK the program tests skip, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set; without a
usable device they skip, unless OOT2_REQUIRE_GPU is set.

Run: python -m unittest discover -s tests -p test_runtime_vk_texture.py -v
"""

from __future__ import annotations

import atexit
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXIT_FATAL, ROOT, build_program, case_env, clang_or_skip, fatal_lines, vulkan_or_skip

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_texture_test.c"
LEVEL = "-O2"
EXIT_NO_DEVICE = 3
VALIDATION = {"OOT2_VK_VALIDATION": "sync"}

_build: dict[str, object] = {}


def program() -> Path:
    if "exe" not in _build:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        tmp = tempfile.TemporaryDirectory(prefix="oot2-vktex-", ignore_cleanup_errors=True)
        atexit.register(tmp.cleanup)
        _build["tmp"] = tmp
        _build["exe"] = build_program(cc, include, glslang, TEST_SOURCE, Path(tmp.name), levels=(LEVEL,))[LEVEL]
    return _build["exe"]  # type: ignore[return-value]


def run(case: str, env: dict[str, str] | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run([str(program()), case], capture_output=True, text=True, timeout=timeout, env=case_env(env))


class DeviceTest(unittest.TestCase):
    def device(self, case: str, env: dict[str, str] | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
        done = run(case, {**VALIDATION, **(env or {})}, timeout)
        if done.returncode == EXIT_NO_DEVICE:
            why = f"no Vulkan device here ({done.stdout.strip()})"
            if os.environ.get("OOT2_REQUIRE_GPU"):
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but there is {why}")
            self.skipTest(why)
        return done

    def passing(self, case: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        done = self.device(case, env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-3000:])
        self.assertEqual(fatal_lines(done.stderr), [])
        self.assertIn("validation 0 errors, 0 warnings", done.stdout)
        self.assertIn(f"ok {case}", done.stdout)
        return done

    def test_the_compute_pass_equals_the_cpu_untiler_in_both_orders(self) -> None:
        done = self.passing("untile")
        self.assertRegex(done.stdout, r"\d+ regions, \d+ bytes, equal to the CPU untiler in both orders; the orders differ on \d+ of them")

    def test_lichiis_answer_on_real_textures_is_the_switch_2s_order(self) -> None:
        done = self.device("anchor", {"OOT2_T65_DIR": os.environ.get("OOT2_T65_DIR", "")})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-3000:])
        if "skip anchor" in done.stdout:
            if os.environ.get("OOT2_REQUIRE_ANCHOR"):
                raise RuntimeError(f"OOT2_REQUIRE_ANCHOR is set: {done.stdout.strip()}")
            self.skipTest(done.stdout.strip())
        self.assertEqual(fatal_lines(done.stderr), [])
        self.assertRegex(done.stdout, r"\d+ real regions \(\d+ bytes\): the compute pass and the CPU untiler equal Lichii's answer; the X1's order differs on \d+")
        self.assertIn("validation 0 errors, 0 warnings", done.stdout)

    def test_textures_the_cpu_wrote_are_images_of_the_arena_that_read_back_as_written(self) -> None:
        done = self.passing("upload")
        self.assertEqual(len(re.findall(r"^spec \d+: .*every level and layer reads back as written$", done.stdout, re.M)), 12)
        self.assertRegex(done.stdout, r"arena images 12, uploads 12,")

    def test_submissions_beyond_the_ring_wait_for_the_earlier_ones(self) -> None:
        done = self.passing("ring")
        m = re.search(r"6 submissions of 5 textures of 64 MiB: ring peak (\d+) MiB of 384, ring waits (\d+), (\d+) MiB read from storage", done.stdout)
        self.assertIsNotNone(m, done.stdout)
        self.assertLessEqual(int(m.group(1)), 384)

    def test_the_arena_gets_its_memory_back_when_textures_are_replaced(self) -> None:
        done = self.passing("churn")
        self.assertRegex(done.stdout, r"90 textures registered one after another at one ID: 90 images made, \d+ destroyed, arena peak \d+ MiB, \d+ MiB in use now")

    def test_one_submission_beyond_the_ring_stops_by_name(self) -> None:
        done = self.device("ring_full")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr[-3000:])
        lines = fatal_lines(done.stderr)
        self.assertEqual(len(lines), 1, done.stderr[-3000:])
        self.assertIn("the staging ring is full inside one submission", lines[0])

    def test_untiled_rows_beyond_the_scratch_are_copied_out_before_it_is_reused(self) -> None:
        done = self.passing("scratch")
        self.assertRegex(done.stdout, r"5 regions in one prologue, scratch reused [1-9]\d* times, grown 0 times")

    def test_a_texture_the_cpu_rewrites_is_uploaded_again(self) -> None:
        done = self.passing("rewrite")
        self.assertIn("a bind with no write uploaded nothing", done.stdout)
        self.assertRegex(done.stdout, r"one write: \d+ regions uploaded again, \d+ lie over the page")
        self.assertRegex(done.stdout, r"a second write at the start of the texture: \d+ regions again, the image follows")

    def test_the_ring_and_the_arena_hold_their_caps_under_uploads_that_exceed_them(self) -> None:
        done = self.passing("caps")
        m = re.search(r"ring peak (\d+) MiB of 384, ring waits (\d+), arena (\d+) MiB in \d+ blocks of a cap of 3481 MiB", done.stdout)
        self.assertIsNotNone(m, done.stdout)
        self.assertLessEqual(int(m.group(1)), 384)

    def test_what_the_upload_does_not_serve_yet_stops_by_name_at_the_first_bind(self) -> None:
        for case, text in (("astc_stop", "an ASTC texture, which the transcode to BC brings (ticket 66)"),
                           ("depth_stop", "a depth or stencil texture")):
            with self.subTest(case=case):
                done = self.device(case)
                self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr[-3000:])
                self.assertNotIn("returned", done.stdout)
                lines = fatal_lines(done.stderr)
                self.assertEqual(len(lines), 1, done.stderr[-3000:])
                self.assertIn(text, lines[0])

    def test_a_full_arena_stops_by_name(self) -> None:
        done = self.device("caps_full")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr[-3000:])
        self.assertNotIn("ok caps_full", done.stdout)
        lines = fatal_lines(done.stderr)
        self.assertEqual(len(lines), 1, done.stderr[-3000:])
        # the cap binds on a device with room for it and the eighth of the budget kept free; one already holding other
        # work reaches its budget first
        heap = re.search(r"device heap: (\d+) MiB used of a budget of (\d+) MiB", done.stdout)
        self.assertIsNotNone(heap, done.stdout)
        used, budget = int(heap.group(1)), int(heap.group(2))
        room = budget - used - budget // 8 >= 3481 + 70
        self.assertRegex(lines[0], r"the texture arena is full \(the arena's cap\)" if room else
                         r"the texture arena is full \((the arena's cap|the device's memory budget)\)")
        self.assertIn("evicting textures by uploading them again is not built", lines[0])


if __name__ == "__main__":
    unittest.main()
