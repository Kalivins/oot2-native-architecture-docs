# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The synchronisation contract of gfx.h on a real device (runtime/gpu/vk_replay_report.c, runtime/gpu/vk_capture.c),
through tests/runtime/vk_sync_test.c: the checks the null back end passes in tests/test_runtime_nvn_queue.py, on the
Vulkan back end. Recordings made by hand go through the back end's own submit; a gate semaphore holds submissions so that
memory can be read with their work not done.

- fences: three held submissions write nothing until released, are done one at a time in submission order, and leave the
  event at 1, 2, then 12.
- timestamps: reports in 31.25 MHz ticks of the host clock, bracketed by the clock around their submit and their wait, two
  reports 20 ms apart, 64 of three types never going back, a held report stamped at its release.
- samples: draws clipped to 5x3, 4x2 and 2x2 pixels report 15, 23 and 27 samples passed; a skipped draw adds nothing; a reset
  sets 0.
- render_cond: seven draws, rendered or skipped by conditions on done reports, on a timestamp and a samples-passed report
  not done in the same submission, and on a timestamp report held in an earlier one.
- events: a CPU spinning on an event sees the copies the frame made before it (written back to a mirror first); WRITE then
  ADD leaves their sum; a WaitEvent is checked at its place.
- present: three textures presented in order reach the capture with the crop; a present right after a held submission shows
  its work.

Every case runs under synchronization validation, which must report nothing, on the CPU-visible pool as a mirror and as
an import where it matters. Each check has a counter-run (a test-only switch that breaks its mechanism) that must fail.
Without a usable device they skip, unless OOT2_REQUIRE_GPU is set.

Run: python -m unittest discover -s tests -p test_runtime_vk_sync.py -v
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import test_runtime_vk_pipeline as vp
from test_runtime_gpu import ROOT, build_program, case_env, clang_or_skip, vulkan_or_skip

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_sync_test.c"
LEVEL = "-O1"
_exe: dict[str, object] = {}


def exe() -> Path:
    if "exe" not in _exe:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        tmp = tempfile.TemporaryDirectory(prefix="oot2-vksync-", ignore_cleanup_errors=True)
        _exe["tmp"] = tmp
        _exe["exe"] = build_program(cc, include, glslang, TEST_SOURCE, Path(tmp.name), levels=(LEVEL,))[LEVEL]
    return _exe["exe"]  # type: ignore[return-value]


def tearDownModule() -> None:
    if "tmp" in _exe:
        _exe.pop("tmp").cleanup()  # type: ignore[attr-defined]


class SyncTest(vp.DeviceTestCase):
    """The device checks, on a device with every item the renderer requires."""

    def sync(self, case: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(exe()), case, *args], capture_output=True, text=True, timeout=300,
                              env=case_env({"OOT2_VK_VALIDATION": "sync"}))

    def clean(self, case: str, *args: str) -> str:
        done = self.sync(case, *args)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-3000:])
        self.assertIn(f"ok {case}", done.stdout)
        self.assertIn("validation errors 0 warnings 0", done.stdout, done.stderr[-3000:])
        self.assertEqual([l for l in done.stderr.splitlines() if l.startswith("FATAL")], [])
        print("\n" + done.stdout)
        return done.stdout

    def failing(self, case: str, *args: str) -> str:
        done = self.sync(case, *args)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr[-3000:])
        self.assertNotIn(f"ok {case}", done.stdout)
        self.assertIn("check failed", done.stderr)
        return done.stdout + done.stderr

    def both(self, case: str, *args: str) -> list[str]:
        outs = []
        for mode in ("mirror", "imports"):
            with self.subTest(pool=mode):
                outs.append(self.clean(case, *args, *(("imports",) if mode == "imports" else ())))
        return outs

    def test_fences_are_done_in_submission_order_and_write_nothing_before(self) -> None:
        for out in self.both("fences"):
            self.assertIn("held: no event, no report, no fence", out)
            for line in ("complete 1 event 1", "complete 2 event 2", "complete 3 event 12"):
                self.assertIn(line, out)

    def test_report_timestamps_are_guest_ticks_at_completion(self) -> None:
        for out in self.both("timestamps"):
            self.assertIn("64 reports stamped in order", out)

    def test_counter_run_timestamps_taken_at_replay(self) -> None:
        self.failing("timestamps", "--mutant", "replaystamp")

    def test_samples_passed_count_the_pixels_of_known_draws(self) -> None:
        for out in self.both("samples", "--pack", str(self.pack)):
            self.assertIn("samples passed: 0 15 23 27 0", out)

    def test_counter_run_without_queries_nothing_is_counted(self) -> None:
        out = self.failing("samples", "--pack", str(self.pack), "--mutant", "noqueries")
        self.assertIn("samples passed: 0 0 0 0 0", out)

    def test_conditional_rendering_on_done_and_not_done_reports(self) -> None:
        for out in self.both("render_cond", "--pack", str(self.pack)):
            self.assertIn("columns 0 to 7: .R.RR.RR", out)

    def test_counter_run_the_condition_reads_memory_under_a_report_not_done(self) -> None:
        self.failing("render_cond", "--pack", str(self.pack), "--mutant", "nopending")

    def test_events_follow_the_frames_copies_in_op_stream_order(self) -> None:
        for out in self.both("events"):
            self.assertIn("20 events seen by a spinning CPU, 0 stale words of the copies before them", out)
            self.assertIn("event 8; waits 2 unsatisfied 1", out)

    def test_counter_run_events_walked_backwards(self) -> None:
        out = self.failing("events", "--mutant", "reverse")
        self.assertIn("event 5;", out)

    def test_presents_reach_the_capture_in_order_with_their_crop(self) -> None:
        out = self.clean("present")
        for f in range(9):
            self.assertIn(f"present {f + 1}: texture {f % 3} in the capture", out)
        self.assertIn("present after a held clear: the clear's white", out)

    def test_counter_run_a_present_that_ignores_its_crop(self) -> None:
        out = self.failing("present", "--mutant", "nocrop")
        self.assertIn("capture: present 1, 64x64", out)


if __name__ == "__main__":
    unittest.main()
