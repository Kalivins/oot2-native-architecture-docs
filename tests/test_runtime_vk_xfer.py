# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The replay's transfers on a real device (runtime/gpu/vk_replay_xfer.c, oot2/vk_replay.h), through
tests/runtime/vk_xfer_test.c: recordings made by hand go through the back end's own submit, on the CPU-visible pool as a
mirror (VRAM kept current from guest memory, its GPU writes written back after the fence) and as an import (the guest's own
pages), and the results are read from guest memory. The synthetic pack of GLSL modules of tests/test_runtime_vk_pipeline.py
serves the draw and the dispatches (no game data).

- buffers: UpdateUniformBuffer, buffer copies into, out of and between the CPU-visible pool and VRAM, and fills each give the
  expected bytes in guest memory after the fence and cpu_synced; held behind a semaphore, the destinations keep the CPU's
  bytes until it opens.
- textures: a buffer to a texture, a texture to a texture, a clear of a whole level and of a part of one (a pass of its own),
  and a texture to a buffer, tight and at a row stride, each texel as expected.
- uniform_draw: a draw reads a uniform in host memory (copied at submit), an update rewrites it, and the next draw reads the new
  value; with the drop rule off the second draw shows the old value (the counter-run).
- storage: a compute module's stores reach the CPU-visible pool, a uniform an update rewrote between two dispatches is read
  anew, and an indirect dispatch takes group counts an update wrote in the same submission; with the writes declared to no
  one, the CPU never sees the stores of a mirror (the counter-run).
- storage_image: a compute module stores a known value to every texel of two textures registered at texture pool IDs, one in a
  pool the CPU cannot see (an image placed in its extent) and one in CPU-visible storage (an arena image), and a copy reads each
  back; a hundred more submissions then make storage views the completion thread destroys, and the images still read back
  whole with no view left at the device's end (validation counts them); a store into a corner of a texture the CPU wrote
  leaves the rest as the CPU wrote it, read back by a later submission (the CPU's bytes are uploaded before the stores, in
  the submission that binds the texture, and not again over them).

Every case runs under synchronization validation, which must report nothing. Without a usable device they skip, unless
OOT2_REQUIRE_GPU is set.

Run: python -m unittest discover -s tests -p test_runtime_vk_xfer.py -v
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest import mock

import test_runtime_vk_pipeline as vp
from test_runtime_gpu import ROOT, build_program, case_env, clang_or_skip, vulkan_or_skip

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_xfer_test.c"
LEVEL = "-O1"
_exe: dict[str, Path] = {}

# A compute module that stores the group counts it finds in the driver bank c[0] (bytes 4, 8 and 12 of its head).
HEAD_GLSL = """#version 450
layout(local_size_x = 1) in;
layout(set = 1, binding = 2) uniform Head { uvec4 head; } h;
layout(set = 1, binding = 9) buffer Out { uint v[]; } o;
void main() {
    if (gl_GlobalInvocationID.x == 0u) { o.v[0] = h.head.y; o.v[1] = h.head.z; o.v[2] = h.head.w; }
}
"""
HEAD_KEY = "5100000000000005.00000000"

# A compute module that stores (x, y, 0, 1) to the texel (x, y) of the storage image at binding 25 (compute storage images are
# bindings 25 to 31), without a format as the translator's modules do.
IMG_GLSL = """#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 1, binding = 25) uniform writeonly image2D img;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    imageStore(img, p, vec4(float(p.x) / 255.0, float(p.y) / 255.0, 0.0, 1.0));
}
"""
IMG_KEY = "5100000000000006.00000000"


def exe() -> Path:
    if "exe" not in _exe:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        import tempfile
        tmp = tempfile.TemporaryDirectory(prefix="oot2-vkxfer-", ignore_cleanup_errors=True)
        _exe["tmp"] = tmp  # type: ignore[assignment]
        _exe["exe"] = build_program(cc, include, glslang, TEST_SOURCE, Path(tmp.name), levels=(LEVEL,))[LEVEL]
    return _exe["exe"]


def tearDownModule() -> None:
    if "tmp" in _exe:
        _exe.pop("tmp").cleanup()  # type: ignore[attr-defined]


class TransferTest(vp.DeviceTestCase):
    """The device checks, on a device with every item the renderer requires."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.head_dir = vp.scratch()
        patches = (mock.patch.dict(vp.GLSL, {"head.comp": HEAD_GLSL, "img.comp": IMG_GLSL}),
                   mock.patch.dict(vp.SYNTH_KEYS, {"head.comp": HEAD_KEY, "img.comp": IMG_KEY}),
                   mock.patch.dict(vp.SYNTH_STAGE, {"head.comp": "cs", "img.comp": "cs"}))
        for patch in patches:
            patch.start()
            cls.addClassCleanup(patch.stop)
        cls.head_pack = vp.synthetic_pack(cls.head_dir)

    def xfer(self, case: str, *args: str, validation: str = "sync") -> subprocess.CompletedProcess:
        return subprocess.run([str(exe()), case, *args], capture_output=True, text=True, timeout=300,
                              env=case_env({"OOT2_VK_VALIDATION": validation}))

    def clean(self, case: str, *args: str) -> str:
        done = self.xfer(case, *args)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-3000:])
        self.assertIn(f"ok {case}", done.stdout)
        self.assertIn("validation errors 0 warnings 0", done.stdout, done.stderr[-3000:])
        self.assertEqual([l for l in done.stderr.splitlines() if l.startswith("FATAL")], [])
        return done.stdout

    def failing(self, case: str, *args: str) -> str:
        done = self.xfer(case, *args)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr[-3000:])
        self.assertNotIn(f"ok {case}", done.stdout)
        return done.stdout + done.stderr

    def test_buffer_copies_fills_and_updates_reach_guest_memory_after_their_fence(self) -> None:
        for mode in ("mirror", "imports"):
            with self.subTest(destinations=mode):
                args = ("gated",) if mode == "mirror" else ("gated", "imports")
                out = self.clean("buffers", *args)
                self.assertIn("held by a semaphore: the destinations kept the CPU's bytes", out)
                print("\n" + out)

    def test_counter_run_a_write_declared_to_no_one_never_reaches_the_cpu_through_a_mirror(self) -> None:
        err = self.failing("buffers", "gated", "--mutant", "nodeclare")
        self.assertIn("check failed", err)

    def test_texture_copies_clears_and_readbacks_give_the_expected_texels(self) -> None:
        for mode in ("mirror", "imports"):
            with self.subTest(destinations=mode):
                out = self.clean("textures", *(("gated",) if mode == "mirror" else ("gated", "imports")))
                self.assertIn("texture B read back: blue, A's block copied, a part cleared red", out)
                self.assertIn("texture A's block read back at a row stride of 64 bytes", out)

    def test_a_draw_after_an_update_reads_the_new_uniform_with_the_snapshot_on(self) -> None:
        for mode in ("mirror", "imports"):
            with self.subTest(uniform=mode):
                out = self.clean("uniform_draw", "--pack", str(self.pack), *(("imports",) if mode == "imports" else ()))
                self.assertIn("pixels: 0 of 1024 differ from the new uniform's value, 0 show the old one", out)
                if mode == "imports":
                    self.assertIn("snapshots 1 drops 1 ", out)   # the first draw's copy, dropped by the update
                print("\n" + out)

    def test_counter_run_without_the_drop_rule_the_second_draw_shows_the_old_uniform(self) -> None:
        err = self.failing("uniform_draw", "--pack", str(self.pack), "imports", "--mutant", "nodrop")
        self.assertIn("1024 of 1024 differ from the new uniform's value, 1024 show the old one", err)

    def test_compute_stores_reach_the_cpu_and_an_update_feeds_the_next_dispatch(self) -> None:
        for mode in ("mirror", "imports"):
            with self.subTest(destinations=mode):
                out = self.clean("storage", "gated", "--pack", str(self.pack), *(("imports",) if mode == "imports" else ()))
                self.assertIn("; 0 wrong", out)
                print("\n" + out)

    def test_an_indirect_dispatch_has_its_group_counts_in_its_bank_from_the_gpu(self) -> None:
        for mode in ("mirror", "imports"):
            with self.subTest(destinations=mode):
                out = self.clean("head", "gated", "--pack", str(self.head_pack), *(("imports",) if mode == "imports" else ()))
                self.assertIn("direct (5 6 7), indirect after an update (2 1 1), direct again (3 1 1)", out)

    def test_compute_stores_to_images_by_texture_id_read_back_whole(self) -> None:
        for mode in ("mirror", "imports"):
            with self.subTest(destinations=mode):
                out = self.clean("storage_image", "--pack", str(self.head_pack), *(("imports",) if mode == "imports" else ()))
                self.assertIn("placed image: 0 of 65536 texels wrong", out)
                self.assertIn("arena image: 0 of 65536 texels wrong", out)
                self.assertIn("after 100 more submissions: placed 0, arena 0 texels wrong", out)
                self.assertIn("partial store into a texture the CPU wrote: 0 of 16384 texels wrong", out)
                print("\n" + out)

    def test_counter_run_the_stores_of_a_mirror_are_lost_when_not_declared(self) -> None:
        err = self.failing("storage", "gated", "--pack", str(self.pack), "--mutant", "nodeclare")
        self.assertIn("check failed", err)


if __name__ == "__main__":
    unittest.main()
