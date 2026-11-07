# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the NVN2 front end's command buffers (architecture D11).

Builds tests/runtime/nvn_cmd_test.c with runtime/core and runtime/nvn, as
tests/test_runtime_nvn.py builds its harness, at -O1 and -O2, and reads what
it prints: the entry points it calls and the records the null back end's op
log walks at submit.

- ops: every command buffer entry point with a body, called once; one
  recording holds every command, calls a second recording and a third copies
  it. Each call must leave exactly the records PROC_OPS names, the called
  recording inlined, and what the game changes once the calls returned
  (textures, the program, arrays) must not reach the records, but for the
  deferred bind, which reads its array at submit.
- begin: a recording started the way the game starts most of them, by its
  inline BeginRecording (the harness writes the console driver's fields at
  their offsets), and by nvnCommandBufferBeginRecording when control memory is
  short; the front end writes nothing into control memory.
- callbacks: the memory callback is called where the console's driver calls
  it, with its arguments, and nowhere else.
- loop: 131 command buffers created, recorded, submitted and finalized each
  frame, as research 17's census counts in the Forest, with no growth of the
  host memory they hold.
- failing cases: what the front end cannot honour stops by name.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_runtime_nvn as nvn  # noqa: E402

TEST_SOURCE = nvn.ROOT / "tests" / "runtime" / "nvn_cmd_test.c"
END_SOURCE = nvn.ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
FRAMES = 400
LOOP_BUFFERS = 131

# The records each command buffer entry point leaves in its recording.
CALLED = ["BIND_TEX", "DRAW"]          # what the called recording holds
PROC_OPS = {
    # the object, its memory, recording
    "nvnCommandBufferInitialize": [], "nvnCommandBufferFinalize": [], "nvnCommandBufferSetDebugLabel": [],
    "nvnCommandBufferSetMemoryCallback": [], "nvnCommandBufferSetMemoryCallbackData": [],
    "nvnCommandBufferGetMemoryCallback": [], "nvnCommandBufferGetMemoryCallbackData": [],
    "nvnCommandBufferSetCommandMemoryCallbackEnabled": [], "nvnCommandBufferSetControlMemoryCallbackEnabled": [],
    "nvnCommandBufferAddCommandMemory": [], "nvnCommandBufferAddControlMemory": [],
    "nvnCommandBufferGetCommandMemorySize": [], "nvnCommandBufferGetCommandMemoryUsed": [],
    "nvnCommandBufferGetCommandMemoryFree": [], "nvnCommandBufferGetControlMemorySize": [],
    "nvnCommandBufferGetControlMemoryUsed": [], "nvnCommandBufferGetControlMemoryFree": [],
    "nvnCommandBufferSetRecordingMode": [], "nvnCommandBufferBeginRecording": [], "nvnCommandBufferEndRecording": [],
    "nvnCommandBufferIsRecording": [], "nvnDeviceFinalizeCommandHandle": [],
    "nvnCommandBufferCallCommands": CALLED, "nvnCommandBufferCopyCommands": CALLED,
    # render targets and clears
    "nvnCommandBufferSetRenderTargets": ["SET_RT"], "nvnCommandBufferClearColor": ["CLEAR_COLOR"],
    "nvnCommandBufferClearDepthStencil": ["CLEAR_DS"], "nvnCommandBufferClearTexture": ["CLEAR_TEX"],
    # viewports, scissors, depth
    "nvnCommandBufferSetViewport": ["VIEWPORTS"], "nvnCommandBufferSetViewports": ["VIEWPORTS"],
    "nvnCommandBufferSetScissor": ["SCISSOR"], "nvnCommandBufferSetScissors": ["SCISSOR"],
    "nvnCommandBufferSetDepthRange": ["DEPTH_RANGE"], "nvnCommandBufferSetDepthRanges": ["DEPTH_RANGE"],
    "nvnCommandBufferSetDepthBounds": ["DEPTH_BOUNDS"],
    # render state
    "nvnCommandBufferSetDepthClamp": ["DEPTH_CLAMP"], "nvnCommandBufferSetRenderEnable": ["RENDER_ENABLE"],
    "nvnCommandBufferSetRasterizerDiscard": ["RASTER_DISCARD"],
    "nvnCommandBufferSetGeometryProcessingMode": ["GEOMETRY_MODE"], "nvnCommandBufferSetSampleMask": ["SAMPLE_MASK"],
    "nvnCommandBufferSetShadingRatePaletteMode": ["SHADING_RATE_MODE"],
    "nvnCommandBufferSetPrimitiveRestart": ["PRIM_RESTART"],
    "nvnCommandBufferSetRenderEnableConditional": ["RENDER_COND"],
    "nvnCommandBufferSetConservativeRasterEnable": ["CONSERVATIVE"],
    "nvnCommandBufferSetConservativeRasterMode": ["CONSERVATIVE"],
    # programs and state objects
    "nvnCommandBufferBindProgram": ["BIND_PROGRAM"],
    **{f"nvnCommandBufferBind{s}State": ["BIND_STATE"]
       for s in ("Blend", "Color", "ChannelMask", "Multisample", "Polygon", "DepthStencil")},
    "nvnCommandBufferSetPointSize": ["BIND_STATE"], "nvnCommandBufferSetLineWidth": ["BIND_STATE"],
    "nvnCommandBufferBindVertexAttribState": ["BIND_VATTR"], "nvnCommandBufferBindVertexStreamState": ["BIND_VSTREAM"],
    "nvnCommandBufferBindVertexBuffer": ["BIND_VBUF"],
    # buffers and handles
    "nvnCommandBufferBindUniformBuffer": ["BIND_UBO"], "nvnCommandBufferBindStorageBuffer": ["BIND_SSBO"],
    "nvnCommandBufferBindStorageBuffers": ["BIND_SSBO", "BIND_SSBO"],   # two buffers in the harness
    "nvnCommandBufferBindTexture": ["BIND_TEX"], "nvnCommandBufferBindTextures": ["BIND_TEX"],
    "nvnCommandBufferBindTexturesDeferred": ["BIND_TEX"], "nvnCommandBufferBindImage": ["BIND_IMG"],
    "nvnCommandBufferBindImages": ["BIND_IMG"], "nvnCommandBufferBindAccelerationStructuresDeferred": ["BIND_ACCEL"],
    "nvnCommandBufferSetTexturePool": ["SET_TEX_POOL"], "nvnCommandBufferSetSamplerPool": ["SET_SAMP_POOL"],
    "nvnCommandBufferSetShaderScratchMemory": ["SCRATCH"],
    "nvnCommandBufferPushGraphicsConstants": ["PUSH_CONST"], "nvnCommandBufferPushComputeConstants": ["PUSH_CONST"],
    # stencil, polygon offset, blend colour
    "nvnCommandBufferSetStencilValueMask": ["STENCIL"], "nvnCommandBufferSetStencilMask": ["STENCIL"],
    "nvnCommandBufferSetStencilRef": ["STENCIL"], "nvnCommandBufferSetPolygonOffsetClamp": ["POLY_OFFSET"],
    "nvnCommandBufferSetBlendColor": ["BLEND_COLOR"],
    # draws and dispatches
    "nvnCommandBufferDrawArrays": ["DRAW"], "nvnCommandBufferDrawArraysInstanced": ["DRAW"],
    "nvnCommandBufferDrawElements": ["DRAW_IDX"], "nvnCommandBufferDrawElementsBaseVertex": ["DRAW_IDX"],
    "nvnCommandBufferDrawElementsInstanced": ["DRAW_IDX"], "nvnCommandBufferDrawElementsIndirect": ["DRAW_IDX_IND"],
    "nvnCommandBufferDrawMeshTasks": ["DRAW_MESH"], "nvnCommandBufferDrawMeshTasksIndirect": ["DRAW_MESH_IND"],
    "nvnCommandBufferDispatchCompute": ["DISPATCH"], "nvnCommandBufferDispatchComputeIndirect": ["DISPATCH_IND"],
    "nvnCommandBufferBarrier": ["BARRIER"],
    # results the CPU sees
    "nvnCommandBufferSignalEvent": ["SIGNAL_EVENT"], "nvnCommandBufferWaitEvent": ["WAIT_EVENT"],
    "nvnCommandBufferReportCounter": ["REPORT_COUNTER"], "nvnCommandBufferResetCounter": ["RESET_COUNTER"],
    "nvnCommandBufferReportValue": ["CLEAR_BUF"], "nvnCommandBufferClearBuffer": ["CLEAR_BUF"],
    "nvnCommandBufferUpdateUniformBuffer": ["UPDATE_UB"],
    # copies
    "nvnCommandBufferCopyBufferToTexture": ["COPY_B2T"], "nvnCommandBufferCopyTextureToBuffer": ["COPY_T2B"],
    "nvnCommandBufferCopyTextureToTexture": ["COPY_T2T"], "nvnCommandBufferSetCopyRowStride": ["COPY_STRIDE"],
    "nvnCommandBufferSetCopyImageStride": ["COPY_IMAGE_STRIDE"],
    # variable-rate shading and hints
    "nvnCommandBufferBindShadingRateImage": ["SHADING_RATE_IMAGE"],
    "nvnCommandBufferShadingRateImagePalette": ["SHADING_RATE_PALETTE"],
    **{f"nvnCommandBuffer{h}": ["HINT"]
       for h in ("SetComputeOverlapMode", "SetComputeWorkloadPriority", "SetColorReductionEnable",
                 "SetColorReductionThresholds", "SetStencilCullCriteria", "SetTiledCacheAction",
                 "SetTiledCacheTileSize")},
}

# The failing cases: (case, exit code, text the FATAL line must contain)
FAILING = (
    ("inline_command_write", nvn.EXIT_UNIMPLEMENTED,
     "game code moved its command write pointer (+0x00 0xc, +0x08 0, +0x20 0): it writes GPU methods into command "
     "memory itself, which the front end cannot replay"),
    ("control_moved", nvn.EXIT_UNIMPLEMENTED,
     "game code moved its control write pointer from "),
    ("ended_outside", nvn.EXIT_UNIMPLEMENTED, "ended outside the front end"),
    ("retired_handle", nvn.EXIT_UNIMPLEMENTED,
     "QueueSubmitCommands: command handle 0x"),
    ("callback_adds_nothing", nvn.EXIT_UNIMPLEMENTED,
     "CommandBufferBeginRecording: command buffer "),
    ("vertex_attributes", nvn.EXIT_UNIMPLEMENTED,
     "CommandBufferBindVertexAttribState: first 0, 33 entries, past the 32 the console has"),
    ("not_a_texture", nvn.EXIT_UNIMPLEMENTED, "CommandBufferSetRenderTargets: the texture "),
)
FAILING_MORE = {
    "retired_handle": "is not a live recording (never recorded, or the control memory it was recorded in was added "
                      "again)",
    "callback_adds_nothing": "its memory callback left 0 bytes of control memory where 16 are needed",
    "not_a_texture": "is not initialised",
}


def records(stdout: str) -> dict[tuple[int, int], list[str]]:
    """The op log: submission fence and handle index to the ops walked, in order."""
    out: dict[tuple[int, int], list[str]] = {}
    for m in re.finditer(r"^rec (\d+) (\d+) (\w+)$", stdout, re.M):
        out.setdefault((int(m[1]), int(m[2])), []).append(m[3])
    return out


def calls_between(stdout: str, start: str, end: str) -> list[str]:
    lines = stdout.splitlines()
    i = lines.index(start)
    j = lines.index(end, i)
    return [line[5:] for line in lines[i:j] if line.startswith("call ")]


class CommandBufferTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = nvn.find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-nvncmd-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in nvn.CORE.glob("*.c")) + \
            sorted(str(p) for p in nvn.NVN.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in nvn.LEVELS:
            exe = out / f"nvn_cmd_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*nvn.compile_flags(cc), level, *extra, "-I", str(nvn.CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case, *args], capture_output=True, text=True, timeout=300)

    def assert_passes(self, case: str, *args: str) -> list[subprocess.CompletedProcess]:
        runs = []
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=case):
                done = self.run_case(level, case, *args)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                runs.append(done)
        return runs

    def test_every_command_buffer_body_has_its_records(self) -> None:
        bodies = {n for n in nvn.BODIES if n.startswith("nvnCommandBuffer") or n == "nvnDeviceFinalizeCommandHandle"}
        self.assertEqual(bodies, set(PROC_OPS))

    def test_one_recording_holds_every_command_and_a_called_and_a_copied_one(self) -> None:
        for done in self.assert_passes("ops"):
            out = done.stdout
            called = {line[5:] for line in out.splitlines() if line.startswith("call ")}
            self.assertEqual(set(PROC_OPS) - called, set())
            in_b = calls_between(out, "recording B", "recording C")
            want_b = ["BEGIN"] + [op for name in in_b for op in PROC_OPS[name]] + ["END"]
            log = records(out)
            self.assertEqual(log[(1, 0)], want_b)
            self.assertEqual(log[(1, 1)], ["BEGIN"] + CALLED + ["END"])
            self.assertTrue(set(want_b) >= {op for ops in PROC_OPS.values() for op in ops})
            # Control memory as the console's driver fills it: a header, a segment entry where command memory
            # changes, a call, or the end, the called recording's entries, an end.
            self.assertIn("control used A 0x30\n", out)
            self.assertIn("control used B 0x60\n", out)
            self.assertIn("control used C 0x30\n", out)
            used = int(re.search(r"^command used B (0x[0-9a-f]+)$", out, re.M)[1], 16)
            self.assertTrue(0 < used <= 65536, used)
            self.assertIn("checked 26\n", out)

    def test_recordings_start_inline_and_by_call(self) -> None:
        for done in self.assert_passes("begin"):
            out = done.stdout
            lines = [line for line in out.splitlines() if not line.startswith(("call ", "rec "))]
            self.assertEqual(lines[:6], ["begin inline", "control used 0x30", "cb 1 0x100", "begin call",
                                         "control used 0x30", "control memory holds the game's header only"])
            log = records(out)
            self.assertEqual(log[(1, 0)], ["BEGIN", "DRAW", "END"])
            self.assertEqual(log[(2, 0)], ["BEGIN", "DRAW", "END"])
            self.assertEqual(log[(3, 0)], ["BEGIN", "DISPATCH", "END"])
            self.assertIn("recorded before a recording began again are left out of every handle", done.stderr)
            # three begins taken up from the game's inline code, one through the entry point; four begins, three ended
            # with a handle (one began over another's records), one EndRecording with nothing open
            self.assertIn("counts inline 3 slow 1 handle 3 empty 1 left_out 1\n", out)

    def test_memory_callback_fires_where_the_console_would(self) -> None:
        for done in self.assert_passes("callbacks"):
            lines = [line for line in done.stdout.splitlines()
                     if line.startswith(("cb ", "step ", "command size ", "control used "))]
            self.assertEqual(lines, [
                "step no control memory", "cb 1 0x100",
                "step no command memory", "cb 0 0x200", "command size 0x1000",
                "step end short of control memory", "cb 1 0x100", "control used 0x20",
                "step call short of control memory", "cb 1 0x100",
                "step control memory exactly full", "control used 0x30",
                "step callbacks off",
                "step ample memory", "control used 0x48",
                "step end",
            ])
            # five begins taken up from the game's inline code and one through the entry point: the recordings row
            self.assertIn("stat| recordings 6 inline 5, slow 1, ended with a handle 6, ended empty 0\n", done.stdout)
            # three calls for control memory and one for command memory, counted: the census row has both
            self.assertIn("stat| callbacks 4 memory callbacks: control 3, command 1; recordings begun over left-out "
                          "records 0\n", done.stdout)

    def test_command_buffers_made_and_finalized_each_frame_hold_no_more_memory(self) -> None:
        per_frame = sum(2 + 3 * (1 + k % 9) for k in range(LOOP_BUFFERS))
        for done in self.assert_passes("loop", str(FRAMES)):
            out = done.stdout
            self.assertNotIn("cb ", out)
            frames = [(int(m[1]), int(m[2]), int(m[3]))
                      for m in re.finditer(r"^frame (\d+) host (\d+) copies (\d+)$", out, re.M)]
            self.assertEqual(frames[-1][0], FRAMES - 1)
            self.assertEqual({copies for _, _, copies in frames}, {LOOP_BUFFERS})
            self.assertEqual(len({host for _, host, _ in frames}), 1, frames)
            m = re.search(r"^private (\d+) (\d+)$", out, re.M)
            self.assertLess(int(m[2]) - int(m[1]), 2 << 20, m[0])
            self.assertIn(f"records {per_frame * FRAMES}\n", out)

    def test_what_the_front_end_cannot_honour_stops_by_name(self) -> None:
        for case, code, text in FAILING:
            for level in nvn.LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    self.assertIn(FAILING_MORE.get(case, ""), done.stderr)
                    self.assertEqual(done.stderr.count("FATAL"), 1, done.stderr)


if __name__ == "__main__":
    unittest.main()
