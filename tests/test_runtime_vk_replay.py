# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the Vulkan back end's replay (runtime/include/oot2/vk_replay.h, runtime/gpu/vk_replay.c) through
tests/runtime/vk_replay_test.c, with no device (continuous integration runs these).

Synthetic recordings go through the replay with stand-in ops and through the real recorder, whose calls write each packet
as a line. The checks: a pass begins lazily behind a full barrier with its targets loaded and stored in the general
layout, and ends at a render target change, a barrier and the frame's end; the default window origin (lower left) flips Y
through a negative viewport height, measures scissors and clear rectangles from the bottom and mirrors the front face;
the default depth mode sets depth clip control's -1 to 1 range; vertex input comes from the attribute, stream and buffer
state, and an attribute with no buffer reads (0, 0, 0, 1) of its own scalar type from a spare binding; a state that needs a
core feature the device does not enable stops by name; each stage's driver bank c[0] holds its storage table, the
largest point size and the base instance, NVN2 compute's the group counts and its uniform and storage rows; c[2] holds the
bound handles, bank 24 the pushed root tables, the support buffer the viewport; a uniform range in host memory is copied
at submit (the copy holds the bytes of that moment, and the next submission takes a copy of its own) while a mirror binds
live; a storage buffer binds from its bank's words, rounded down to 16 bytes, and one the module writes is declared and
listed; a texture's push constant slot holds its bindless slot and its sampler; draws, indexed and indirect draws, direct
and indirect dispatches and clears become their packets; a GPU write over a copied uniform range stops by name in either
order; every op of the list is replayed or stops by name with its ticket; a recording that is not whole stops by name.

Each check of the ticket's new behaviour has a counter-run, a test-only switch of the replay that breaks the mechanism
(vkr_no_flip: the origin's flip; vkr_no_snapshot: the uniform copy at submit). The check must then fail.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import re
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import (EXIT_FATAL, ROOT, build_program, case_env, clang_or_skip, fatal_lines,  # sets the path of tools/
                              vulkan_or_skip)

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_replay_test.c"
LEVEL = "-O2"

VS_KEY = "00000000000000a1.00000001"
PS_KEY = "00000000000000b2.00000001"
CS_KEY = "00000000000000c3.00000001"
REPLAYED = {
    "BEGIN", "END", "SET_RT", "CLEAR_COLOR", "CLEAR_DS", "VIEWPORTS", "SCISSOR", "DEPTH_RANGE", "BIND_PROGRAM",
    "BIND_STATE", "BIND_VATTR", "BIND_VSTREAM", "BIND_VBUF", "BIND_UBO", "BIND_SSBO", "BIND_TEX", "BIND_SAMP", "BIND_IMG",
    "STENCIL", "POLY_OFFSET", "DRAW", "DRAW_IDX", "DRAW_IDX_IND", "DISPATCH", "DISPATCH_IND", "BARRIER", "SET_TEX_POOL",
    "SET_SAMP_POOL", "BLEND_COLOR", "DEPTH_CLAMP", "RENDER_ENABLE", "RASTER_DISCARD", "PRIM_RESTART", "PUSH_CONST",
    "WAIT_SYNC", "SAMPLE_MASK", "DEPTH_BOUNDS", "CONSERVATIVE", "SHADING_RATE_IMAGE", "SHADING_RATE_MODE", "SCRATCH",
    "HINT",
}


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def packets(out: str) -> list[str]:
    return [line[len("pkt "):] for line in out.splitlines() if line.startswith("pkt ")]


def frame(out: str, n: int) -> list[str]:
    """The packets of the n-th frame (from 1): those before its "submit" line and after the previous one's."""
    frames, cur = [], []
    for p in packets(out):
        if p.startswith("submit "):
            frames.append(cur)
            cur = []
        else:
            cur.append(p)
    return frames[n - 1]


def writes(lines: list[str], binding: int) -> list[str]:
    """The descriptor writes at `binding`, in order, each as "UB buf=... off=... range=... words"."""
    return [line.split(" ", 3)[3] for line in lines if line.startswith(f" w b={binding} ")]


def word(write: str, offset: int) -> int | None:
    m = re.search(rf" {offset:#x}=(0x[0-9a-f]+)\b", write)
    return int(m.group(1), 16) if m else None


# ---- the checks, each a function of the output so that a counter-run can show it failing
def check_pass(out: str) -> None:
    f = frame(out, 1)
    assert f[0] == "PipelineBarrier2 full=1", f[:2]   # the frame after every earlier one
    assert f[1] == "PipelineBarrier2 full=1", f[:2]   # the pass after everything before it
    assert f[2] == "BeginRendering 0,0 64x32 layers=1 colors=1 [0x7125 layout=1 load=0 store=0]", f[2]
    end = f.index("EndRendering")
    assert f[end + 1] == "PipelineBarrier2 full=1" and f.count("BeginRendering 0,0 64x32 layers=1 colors=1 "
                                                               "[0x7125 layout=1 load=0 store=0]") == 1, f[end:]


def check_flip(out: str) -> None:
    f = frame(out, 1)
    assert "SetViewportWithCount 1 [0 32 64 -32 0 1]" in f, [p for p in f if p.startswith("SetViewport")]
    assert "SetScissorWithCount 1 [8 20 16 8]" in f, [p for p in f if p.startswith("SetScissor")]
    assert "SetFrontFace 1" in f, [p for p in f if p.startswith("SetFrontFace")]   # CCW mirrored to CW
    assert "ClearAttachments aspect=0x1 att=0 color=1,0,0,1 rect=8,20 16x8 layers=0+1" in f


def check_depth_clip(out: str) -> None:
    assert "SetDepthClipNegativeOneToOneEXT 1" in frame(out, 1)


def check_snapshot(out: str) -> None:
    for n, base in ((1, 0xA0), (2, 0xB0)):
        w = writes(frame(out, n), 2)
        assert w and all(x.startswith("UB buf=0x5c ") and "range=0x40" in x for x in w), w
        assert all(word(x, 4 * i) == base + i for x in w for i in range(16)), (n, w)


def check_live_mirror(out: str) -> None:
    w = writes(frame(out, 1), 7)
    assert w and all(x == "UB buf=0xb2 off=0x4000 range=0x100" for x in w), w
    w = writes(frame(out, 1), 3)   # c[1], the program's constants in its code area
    assert w and all(x == "UB buf=0xb2 off=0x1100 range=0x40" for x in w), w


def check_banks(out: str) -> None:
    f = frame(out, 1)
    vs_c0, ps_c0 = writes(f, 1), writes(f, 6)
    assert len(vs_c0) == 3 and len(ps_c0) == 3, (vs_c0, ps_c0)
    for x in vs_c0 + ps_c0:
        assert x.startswith("UB buf=0x5c ") and "range=0xa00" in x and word(x, 0x610) == f32(2047.0), x
    assert word(vs_c0[0], 0x644) is None and word(vs_c0[1], 0x644) == 1, vs_c0   # the base instance of each draw
    assert word(ps_c0[0], 0x510) == 0x40000104 and word(ps_c0[0], 0x518) == 0x40, ps_c0[0]
    ps_c2 = writes(f, 8)
    assert ps_c2 and word(ps_c2[0], 0x20 + 8 * 5) == (7 | 3 << 20), ps_c2   # combined handle row 5
    assert writes(f, 4)[0].endswith(" zero") and writes(f, 5)[0].endswith(" zero")
    support = writes(f, 0)
    assert len(support) == 3, support
    s = support[0]
    assert "range=0x570" in s, s
    assert [word(s, o) for o in (0x90, 0x94, 0x98, 0xa0, 0xa4, 0xa8, 0xc0)] == \
        [f32(2 / 64), f32(2 / 32), f32(1.0), f32(64.0), f32(32.0), f32(1.0), f32(1.0)], s


def check_storage(out: str) -> None:
    w = writes(frame(out, 1), 9)
    assert len(w) == 3 and all(x == "SB buf=0xb3 off=0x100 range=0x44" for x in w), w
    assert out.count("op wrote V+0x100 0x44 frame 1\n") == 3 and "op wrote V+0x100 0x44 frame 2\n" in out


def check_push_slot(out: str) -> None:
    pcs = [p for p in frame(out, 1) if p.startswith("PushConstants ")]
    assert len(pcs) == 3 and all(p.endswith(f" 0 128 24={1057 | 3 << 20:#x}") for p in pcs), pcs


def check_vertex_input(out: str) -> None:
    f = frame(out, 1)
    vi = [p for p in f if p.startswith("SetVertexInputEXT")]
    assert vi and all(p == "SetVertexInputEXT [b0 stride=16 rate=0 div=1] [b1 stride=0 rate=0 div=1] "
                           "[l0 b0 fmt=109 off=0] [l1 b1 fmt=108 off=16]" for p in vi), vi
    assert "BindVertexBuffers2 0 1 0xb2+0x3000" in f, f


def check_draws(out: str) -> None:
    f = frame(out, 1)
    draws = [p for p in f if p.startswith(("Draw", "BindIndexBuffer", "ClearAttachments"))]
    assert draws == ["Draw 3 1 0 0", "BindIndexBuffer 0xb2 20480 0", "DrawIndexed 6 2 0 2 1",
                     "BindIndexBuffer 0xb2 20480 1", "DrawIndexedIndirect 0xb3 512 1 20",
                     "ClearAttachments aspect=0x1 att=0 color=1,0,0,1 rect=8,20 16x8 layers=0+1"], draws
    assert "BindPipeline 0 0x9a" in f and "BindDescriptorSets 0 0x6100 0 0x5e7" in f
    assert f"op graphics {VS_KEY} {PS_KEY} colors=1 c0=37 depth=0 stencil=0" in out
    assert [p for p in frame(out, 2) if p.startswith("Draw")] == ["Draw 3 1 3 0"]


_BUILT: dict[str, object] = {}


def tearDownModule() -> None:
    if "tmp" in _BUILT:
        _BUILT.pop("tmp").cleanup()  # type: ignore[attr-defined]


class ReplayBase(unittest.TestCase):
    """vk_replay_test built once for the module (the runtime is a long compile), at one level."""

    @classmethod
    def setUpClass(cls) -> None:
        if "error" in _BUILT:
            raise RuntimeError("vk_replay_test did not build (see the first class's error)")
        if "exe" not in _BUILT:
            cc = clang_or_skip()
            include, glslang = vulkan_or_skip()
            tmp = tempfile.TemporaryDirectory(prefix="oot2-vkreplay-", ignore_cleanup_errors=True)
            _BUILT["tmp"] = tmp
            try:
                _BUILT["exe"] = build_program(cc, include, glslang, TEST_SOURCE, Path(tmp.name), levels=(LEVEL,))[LEVEL]
            except RuntimeError:
                _BUILT["error"] = True
                raise
        cls.exe = _BUILT["exe"]

    def run_case(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exe), *args], capture_output=True, text=True, timeout=60, env=case_env())

    def passing(self, case: str, *args: str) -> str:
        done = self.run_case(case, *args)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"ok {case}", done.stdout)
        return done.stdout

    def stopping(self, *args: str) -> str:
        done = self.run_case(*args)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertNotIn("returned", done.stdout)
        fatal = fatal_lines(done.stderr)
        self.assertEqual(len(fatal), 1, done.stderr)
        return fatal[0]


class GraphicsTest(ReplayBase):
    CHECKS = (check_pass, check_flip, check_depth_clip, check_snapshot, check_live_mirror, check_banks, check_storage,
              check_push_slot, check_vertex_input, check_draws)

    def test_draws_become_their_packets(self) -> None:
        out = self.passing("graphics")
        for check in self.CHECKS:
            with self.subTest(check=check.__name__):
                check(out)
        self.assertIn("stats frames 2 draws 4 clears 1 passes 2 snapshots 2 live 8 writes 4", out)

    def test_counter_run_without_the_flip(self) -> None:
        out = self.passing("graphics", "--mutant", "noflip")
        with self.assertRaises(AssertionError):
            check_flip(out)
        check_snapshot(out)

    def test_counter_run_without_the_snapshot(self) -> None:
        out = self.passing("graphics", "--mutant", "nosnapshot")
        with self.assertRaises(AssertionError):
            check_snapshot(out)
        check_flip(out)

    def test_depth_targets_and_clears(self) -> None:
        f = frame(self.passing("depth"), 1)
        self.assertEqual(f, [
            "PipelineBarrier2 full=1",
            "PipelineBarrier2 full=1",
            "BeginRendering 0,0 64x32 layers=1 colors=1 [0x7125 layout=1 load=0 store=0] "
            "depth [0x7d00 layout=1 load=0 store=0] stencil [0x7d00 layout=1 load=0 store=0]",
            "ClearAttachments aspect=0x6 att=0 depth=1 stencil=5 rect=0,0 64x32 layers=0+1",
            "ClearAttachments aspect=0x1 att=0 color=0,0.5,0,1 rect=0,24 32x8 layers=0+1",
            "EndRendering",
            "PipelineBarrier2 full=1",
            "BeginRendering 0,0 64x32 layers=1 colors=1 [0x7125 layout=1 load=0 store=0]",
            "ClearAttachments aspect=0x1 att=0 color=0,0,1,1 rect=0,24 32x8 layers=0+1",
            "EndRendering",
        ])

    def test_a_state_that_needs_a_feature_the_device_lacks_stops_by_name(self) -> None:
        self.assertEqual(self.stopping("clamp"), "FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: "
                                                 "depth clamp: the depthClamp feature is not enabled")

    def test_a_write_over_a_copied_uniform_stops_by_name(self) -> None:
        line = self.stopping("conflict_read_first")
        self.assertRegex(line, r"^FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: a GPU write "
                               r"of \[0x[0-9a-f]+, 0x[0-9a-f]+\) meets the uniform range \[0x[0-9a-f]+, 0x[0-9a-f]+\) "
                               r"copied at submit; the copies ticket \(67\) brings the drop rule$")
        line = self.stopping("conflict_write_first")
        self.assertRegex(line, r"^FATAL: gfx: vulkan: replay of submission 1, after 1 draws and dispatches: the uniform "
                               r"range \[0x[0-9a-f]+, 0x[0-9a-f]+\) is copied at submit after a GPU write of "
                               r"\[0x[0-9a-f]+, 0x[0-9a-f]+\) in the same submission; the copies ticket \(67\) brings the "
                               r"drop rule$")


class ComputeTest(ReplayBase):
    def test_dispatches_become_their_packets(self) -> None:
        out = self.passing("compute")
        f = frame(out, 1)
        self.assertEqual(out.count(f"op compute {CS_KEY}\n"), 2)
        self.assertEqual(f[0], "PipelineBarrier2 full=1")
        self.assertIn("BindPipeline 1 0x9c", f)
        self.assertIn("BindDescriptorSets 1 0x6200 0 0x5e7", f)
        self.assertEqual([p for p in f if p.startswith("Dispatch")], ["Dispatch 4 2 1", "DispatchIndirect 0xb1 512"])
        self.assertNotIn("BeginRendering", " ".join(f))
        c0 = writes(f, 1)
        self.assertEqual(len(c0), 2)
        for x, groups in zip(c0, ((4, 2, 1), (5, 6, 7))):
            self.assertEqual([word(x, o) for o in (4, 8, 12)], list(groups), x)
            self.assertEqual((word(x, 0x230), word(x, 0x238)), (0x30007000, 0x80), x)   # uniform row 2
            self.assertEqual((word(x, 0x320), word(x, 0x328)), (0x40000300, 0x20), x)   # storage row 1
        roots = writes(f, 3)
        self.assertTrue(roots and "range=0x800" in roots[0], roots)
        self.assertEqual((word(roots[0], 0x208), word(roots[0], 0x20c)), (0x11, 0x22))
        self.assertEqual(writes(f, 4), ["SB buf=0xb3 off=0x300 range=0x20"] * 2)
        self.assertNotIn("op wrote", out)   # the module only reads it


class OpListTest(ReplayBase):
    def test_every_op_is_replayed_or_stops_by_name(self) -> None:
        out = self.passing("oplist")
        rows = [re.fullmatch(r"op (\d+) (\S+) (.+)", line) for line in out.splitlines() if line.startswith("op ")]
        self.assertTrue(rows and all(rows))
        ops = {m.group(2): (int(m.group(1)), m.group(3)) for m in rows}
        self.assertEqual(len(ops), len(rows))
        self.assertEqual({name for name, (_, what) in ops.items() if what == "replayed"}, REPLAYED)
        self.assertIn("past unknown", out)
        for name, (op, what) in sorted(ops.items(), key=lambda kv: kv[1][0]):
            if what == "replayed":
                continue
            with self.subTest(op=name):
                self.assertRegex(what, r"\((ticket \d+|no ticket[^)]*|the front end inlines every one)\)$")
                line = self.stopping("stop", str(op))
                if name in ("JUMP", "CALL"):
                    self.assertEqual(line, "FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: "
                                           f"recording 0 is not whole at +8 (op {op}, 520 bytes)")
                else:
                    self.assertEqual(line, "FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: "
                                           f"{name} is not replayed yet: {what}")

    def test_a_recording_that_is_not_whole_stops_by_name(self) -> None:
        line = self.stopping("broken")
        self.assertEqual(line, "FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: recording 0 is "
                               "not whole at +0 (op 25, 32 bytes)")


if __name__ == "__main__":
    unittest.main()
