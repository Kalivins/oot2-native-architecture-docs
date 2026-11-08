# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the Opus decoder behind the seam of runtime/audio/opus.h (runtime/audio/opus_libopus.c).

Builds tests/runtime/opus_test.c with the seam and a libopus kit (tools/fetch_opus.py lays one out; OPUS_DIR names it), at
-O1 and -O2, and runs each case in its own process. The streams are made inside the test by libopus's own encoder from
generated tones (nothing is from a game): the decoder must give back the frames the encoder was given, each channel with its
own tone at its own level, and the final range of every packet must be the encoder's; an invalid packet and an output
buffer that is too small are refused and the decoder goes on; a packet of no bytes makes no sample; a reset gives the first
packet's samples again.

The decoder tests need a libopus kit. Without one they skip with the reason said, unless OOT2_REQUIRE_OPUS is set, as the
Windows continuous-integration job sets it (the job fetches the kit with tools/fetch_opus.py first). The test of a build with
no kit needs only a compiler.

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

import build_game  # noqa: E402
import fetch_opus  # noqa: E402
import layout  # noqa: E402

TEST_SOURCE = ROOT / "tests" / "runtime" / "opus_test.c"
SEAM = [ROOT / "runtime" / "audio" / "opus.c", ROOT / "runtime" / "audio" / "opus_libopus.c"]
HOS_SOURCE = ROOT / "tests" / "runtime" / "hos_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"
RUNTIME = ROOT / "runtime"
MODULE_SIZE = 0x8000
LEVELS = ("-O1", "-O2")
CASES = ("stereo_tones", "mono_16k", "all_formats", "frame_sizes", "garbage", "too_small", "empty", "reset", "create", "table")


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


def build(cc: str, out: Path, level: str, kit: Path | None) -> Path:
    exe = out / f"opus_test{'_kit' if kit else '_nokit'}{level}{'.exe' if os.name == 'nt' else ''}"
    cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", str(TEST_SOURCE), *map(str, SEAM)]
    if kit:
        cmd += ["-DOOT2_OPUS=1", "-I", str(kit / "include"), str(kit / "lib" / "libopus.a"), "-lm"]
    cmd += ["-o", str(exe)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
    return exe


class OpusTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        kit = build_game.find_opus()
        if kit is None:
            if os.environ.get("OOT2_REQUIRE_OPUS"):
                raise RuntimeError("OOT2_REQUIRE_OPUS is set but OPUS_DIR holds no libopus kit (tools/fetch_opus.py)")
            raise unittest.SkipTest("no libopus kit: run tools/fetch_opus.py --dest DIR and set OPUS_DIR to DIR")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        for level in LEVELS:
            cls.exes[level] = build(cc, out, level, kit)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str) -> subprocess.CompletedProcess:
        done = subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=180)
        self.assertEqual(done.returncode, 0, f"{case} at {level}:\n{done.stdout}{done.stderr}")
        self.assertIn(f"ok {case}", done.stdout)
        return done

    def cases(self, names: tuple[str, ...]) -> None:
        for level in LEVELS:
            for case in names:
                with self.subTest(level=level, case=case):
                    self.run_case(level, case)

    def test_a_stream_made_by_libopus_decodes_to_its_tones_in_their_channels(self) -> None:
        self.cases(("stereo_tones", "mono_16k", "all_formats"))

    def test_packets_of_every_duration_decode_to_their_frames_and_final_range(self) -> None:
        self.cases(("frame_sizes",))

    def test_an_invalid_packet_and_a_small_buffer_are_refused_and_the_decoder_goes_on(self) -> None:
        self.cases(("garbage", "too_small"))

    def test_a_packet_of_no_bytes_makes_no_sample(self) -> None:
        self.cases(("empty",))

    def test_a_reset_starts_the_stream_again(self) -> None:
        self.cases(("reset",))

    def test_the_formats_the_hardware_decoder_does_not_take_are_refused_with_a_reason(self) -> None:
        self.cases(("create",))

    def test_the_build_has_this_table_and_an_installed_one_wins(self) -> None:
        self.cases(("table",))

    def test_every_case_of_the_binary_is_run_by_a_test(self) -> None:
        text = TEST_SOURCE.read_text(encoding="utf-8")
        for case in CASES:
            self.assertIn(f'"{case}"', text)
        self.assertEqual(len(CASES), 10)

    def test_the_kit_the_tests_use_is_the_pinned_one(self) -> None:
        kit = build_game.find_opus()
        self.assertTrue(fetch_opus.is_current(kit), f"{kit} is not what tools/fetch_opus.py lays out for its pins")


class OverrideTest(unittest.TestCase):
    """The hardware Opus override (runtime/hos/ov_opus.c) with libopus behind it, end to end: the Horizon test program built
    with the kit runs its case opus_libopus, which wraps packets of libopus's own encoder in the 8-byte frame header and
    calls the override as the guest would."""

    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        kit = build_game.find_opus()
        if kit is None:
            if os.environ.get("OOT2_REQUIRE_OPUS"):
                raise RuntimeError("OOT2_REQUIRE_OPUS is set but OPUS_DIR holds no libopus kit (tools/fetch_opus.py)")
            raise unittest.SkipTest("no libopus kit: run tools/fetch_opus.py --dest DIR and set OPUS_DIR to DIR")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-opus-ov-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(HOS_SOURCE), str(END_SOURCE)]
        for part in ("core", "nvn", "hos", "audio"):
            sources += sorted(str(p) for p in (RUNTIME / part).glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]   # as test_runtime_hos builds
        libs = [str(kit / "lib" / "libopus.a")] + ([] if os.name == "nt" else ["-lm"])
        for level in LEVELS:
            exe = out / f"hos_opus{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra, "-DOOT2_OPUS=1",
                   "-I", str(kit / "include"), "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(out),
                   *sources, "-o", str(exe), *libs]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def test_the_override_decodes_framed_libopus_packets_to_libopus_samples(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                folder = Path(tempfile.mkdtemp(prefix=f"ov{level}-", dir=self.tmp.name))
                done = subprocess.run([str(self.exes[level]), "opus_libopus", str(folder)], capture_output=True, text=True,
                                      timeout=120)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok opus_libopus", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                # the case gives two packets a wrong header range: one log line, and the samples are checked all the same
                # (a header range of 0 is not a recorded value: the line names a real one, not 0)
                self.assertEqual(done.stderr.count("is not the stream's"), 1, done.stderr)
                self.assertRegex(done.stderr, r"is not the stream's 0x[0-9a-f]+ ")

    def test_the_case_is_in_the_program(self) -> None:
        text = HOS_SOURCE.read_text(encoding="utf-8")
        self.assertIn('"opus_libopus"', text)


class NoKitTest(unittest.TestCase):
    """A build with no libopus kit has no decoder of its own: the hardware Opus override then consumes and makes no sample."""

    def test_a_build_without_the_kit_has_no_table_until_one_is_installed(self) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            self.skipTest("no clang on PATH (set OOT2_CC)")
        with tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True) as tmp:
            for level in LEVELS:
                exe = build(cc, Path(tmp), level, None)
                done = subprocess.run([str(exe), "nokit"], capture_output=True, text=True, timeout=60)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok nokit", done.stdout)


class BuildPlanTest(unittest.TestCase):
    """build_game finds the kit, compiles the runtime with it and links its library (the plan's text, no compiler run)."""

    def make_kit(self, root: Path) -> Path:
        (root / "include" / "opus").mkdir(parents=True)
        (root / "lib").mkdir()
        (root / "include" / "opus" / "opus.h").write_text("", encoding="utf-8")
        (root / "lib" / "libopus.a").write_bytes(b"!<arch>\n")
        return root

    def test_a_kit_is_found_by_its_header_and_library_and_keys_the_link(self) -> None:
        from unittest import mock

        with tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True) as tmp:
            kit = self.make_kit(Path(tmp) / "kit")
            self.assertEqual(build_game.find_opus(kit), kit.resolve())
            with mock.patch.dict(os.environ, {"OPUS_DIR": str(kit)}):
                self.assertEqual(build_game.find_opus(None), kit.resolve())   # else the OPUS_DIR variable
                os.environ.pop("OPUS_DIR")
                self.assertIsNone(build_game.find_opus(None))
            self.assertIsNone(build_game.find_opus(Path(tmp) / "no kit here"))
            self.assertEqual(build_game.opus_inputs(kit), [kit / "lib" / "libopus.a", kit / "include" / "opus" / "opus.h"])
            self.assertEqual(build_game.opus_inputs(None), ["no libopus kit"])
            (kit / "lib" / "libopus.a").unlink()
            self.assertIsNone(build_game.find_opus(kit))   # a kit with no library is no kit
            (kit / "lib" / "libopus.a").write_bytes(b"x")
            (kit / "include" / "opus" / "opus.h").unlink()
            self.assertIsNone(build_game.find_opus(kit))   # nor one with no header


if __name__ == "__main__":
    unittest.main()
