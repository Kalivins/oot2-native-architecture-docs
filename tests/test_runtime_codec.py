# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the video decoder service under the SDK's movie library (runtime/hos/ov_codec.c, ticket 98).

Builds tests/runtime/codec_test.c with runtime/core, runtime/nvn and runtime/hos and a fake AV1 decoder
(tests/runtime/av1_fake.h, written against the seam of runtime/video/av1.h) against a synthetic layout, at -O1 and -O2,
then runs each case in its own process. Synthetic guest threads written as host functions call the service's nn::codec
entry points by name and wait on its events with the kernel's WaitSynchronization. The cases check that each event fires
when the decoder has room and a WaitAny over several holders returns the one that fired, that a submit returns without
waiting for the decode and the decode runs on a worker, that a buffer is not reused before it is released, that a Stop or
a Flush leaves no picture behind, that a damaged unit makes no picture and no crash, and the planes and strides of the
pictures. No game code, no SDK and no decoder is involved.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests skip, unless OOT2_REQUIRE_CLANG is set.
"""

from __future__ import annotations

import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402

RUNTIME = ROOT / "runtime"
TEST_SOURCE = ROOT / "tests" / "runtime" / "codec_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000
TIMEOUT = 120

CASES = ("events", "submit_async", "no_reuse", "stop", "flush", "stop_after", "flush_after", "planes", "damaged", "finalize", "eos", "busy", "misc", "syncpoint", "release_room", "stress", "syncpoint_finalize", "reinit")


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class CodecTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-codec-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)]
        for part in ("core", "nvn", "hos"):
            sources += sorted(str(p) for p in (RUNTIME / part).glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"codec_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str) -> subprocess.CompletedProcess:
        folder = Path(tempfile.mkdtemp(prefix=f"{case}{level}-", dir=self.tmp.name))
        env = {k: v for k, v in os.environ.items() if not k.startswith("OOT2_")}
        try:
            return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=TIMEOUT, env=env,
                                  cwd=folder)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{case} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")

    def check(self, case: str) -> None:
        for level in LEVELS:
            with self.subTest(case=case, level=level):
                done = self.run_case(level, case)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}", done.stdout)

    def test_events_fire_for_the_right_holder(self) -> None:
        self.check("events")

    def test_submit_returns_before_the_decode_and_the_decode_runs_on_a_worker(self) -> None:
        self.check("submit_async")

    def test_a_buffer_is_not_reused_before_its_release(self) -> None:
        self.check("no_reuse")

    def test_stop_leaves_no_picture(self) -> None:
        self.check("stop")

    def test_flush_leaves_no_picture(self) -> None:
        self.check("flush")

    def test_a_picture_already_in_a_buffer_stays_after_stop_and_none_waiting_is_delivered(self) -> None:
        self.check("stop_after")

    def test_a_picture_already_in_a_buffer_stays_after_flush_and_none_waiting_is_delivered(self) -> None:
        self.check("flush_after")

    def test_planes_and_strides(self) -> None:
        self.check("planes")

    def test_a_damaged_unit_makes_no_picture_and_no_crash(self) -> None:
        self.check("damaged")

    def test_finalize_leaves_the_worker_idle_and_the_slot_free(self) -> None:
        self.check("finalize")

    def test_end_of_stream(self) -> None:
        self.check("eos")

    def test_a_full_queue_answers_busy_and_the_event_returns(self) -> None:
        self.check("busy")

    def test_a_sync_point_event_fires_when_its_picture_is_in_its_buffer(self) -> None:
        self.check("syncpoint")

    def test_a_release_that_frees_room_signals_the_submission_ready_event(self) -> None:
        self.check("release_room")

    def test_a_finalised_decoder_leaves_no_armed_sync_point_event_behind(self) -> None:
        self.check("syncpoint_finalize")

    def test_a_decoder_initialised_again_waits_for_its_worker(self) -> None:
        self.check("reinit")

    def test_several_guest_threads_on_one_decoder_do_not_hang_or_corrupt_it(self) -> None:
        self.check("stress")

    def test_the_first_stop_flush_unregister_and_damaged_unit_are_logged_and_counted(self) -> None:
        # one line each for the first call of each kind, and the counts with the Finalize line (60 resets, 30 of each kind)
        for case, text in (("stop", "the first StopAv1Decoder"), ("flush", "the first FlushAv1Decoder"),
                           ("damaged", "the first damaged unit"), ("stress", "the first UnregisterAv1DecoderBuffer")):
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertEqual(done.stderr.count(text), 1, done.stderr)
        for level in LEVELS:
            with self.subTest(case="stress", level=level):
                done = self.run_case(level, "stress")
                self.assertRegex(done.stderr, r"finalised after \d+ pictures \(\d+ units refused as damaged, 30 Stop, 30 Flush, 1 Unregister\)")

    def test_the_other_entries(self) -> None:
        for level in LEVELS:
            with self.subTest(case="misc", level=level):
                done = self.run_case(level, "misc")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok misc", done.stdout)
                # both forms of the decoder's creation (by remote memory, by transfer memory) take the picture size from
                # their own argument lists: 64x64 in both
                self.assertEqual(len(re.findall(r"codec: av1 decoder \d+: 64x64, ", done.stderr)), 2, done.stderr)

    def test_every_case_has_a_test(self) -> None:
        """Each case the C program can run is named in CASES, and each name is run by a test (not only counted)."""
        sources = "".join(inspect.getsource(getattr(self, n)) for n in dir(self)
                          if n.startswith("test_") and n != "test_every_case_has_a_test")
        body = TEST_SOURCE.read_text(encoding="utf-8").split("static void g_main")[1].split("unknown case")[0]
        in_c = set(re.findall(r"\bCASE\((\w\w+)\)", body))   # the macro's own parameter, n, is one letter
        self.assertEqual(in_c, set(CASES))
        for case in CASES:
            self.assertIn(f'"{case}"', sources, f"no test runs the case {case}")


if __name__ == "__main__":
    unittest.main()
