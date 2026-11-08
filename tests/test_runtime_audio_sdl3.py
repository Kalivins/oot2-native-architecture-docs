# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the audio sink's platform stream (runtime/platform/audio_sdl3.c, called by runtime/audio/sink.c).

Builds tests/runtime/audio_sdl3_test.c with the sink, its driver and an SDL3 kit (tools/fetch_sdl3.py lays one out; SDL3_DIR
names it), at -O1 and -O2, and runs each case in its own process with SDL's dummy audio driver (SDL_AUDIO_DRIVER=dummy): no
sound card is opened. The cases check that the sink opens and says which driver it has, that the queue is held to about 30 ms
and a write past that is refused whole and counted, that a queue the device drained is an underrun at the next write, that a
paused sink holds nothing, that the device takes what was written at its own rate, and, with SDL's disk driver that writes
the device's input to a file, that the frames reach the device in order and unchanged.

The tests need an SDL3 kit. Without one they skip with the reason said, unless OOT2_REQUIRE_SDL3 is set. Also checked, with
no kit and no compiler: the build defines OOT2_AUDIO_SDL3 together with OOT2_WITH_SDL3 and only then.

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
import fetch_sdl3  # noqa: E402

RUNTIME = ROOT / "runtime"
TEST_SOURCE = ROOT / "tests" / "runtime" / "audio_sdl3_test.c"
SOURCES = [RUNTIME / "platform" / "audio_sdl3.c", RUNTIME / "audio" / "sink.c", RUNTIME / "audio" / "frameclock.c",
           RUNTIME / "core" / "plat.c", RUNTIME / "core" / "plat_sys.c", RUNTIME / "core" / "fatal.c"]
LEVELS = ("-O1", "-O2")
CASES = ("open", "queue_cap", "underrun", "pause", "drain", "samples")


class AudioSdl3Test(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        try:
            kit = build_game.find_sdl3()
        except build_game.BuildError as e:
            if os.environ.get("OOT2_REQUIRE_SDL3"):
                raise RuntimeError(f"OOT2_REQUIRE_SDL3 is set and the SDL3 kit cannot be used: {e}") from None
            raise unittest.SkipTest(f"the SDL3 kit cannot be used: {e}") from None
        if kit is None:
            if os.environ.get("OOT2_REQUIRE_SDL3"):
                raise RuntimeError("OOT2_REQUIRE_SDL3 is set but SDL3_DIR holds no SDL3 kit (tools/fetch_sdl3.py)")
            raise unittest.SkipTest("no SDL3 kit: run tools/fetch_sdl3.py --dest DIR and set SDL3_DIR to DIR")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-audio-sdl3-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        extra = [] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"audio_sdl3_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", "-DOOT2_WITH_SDL3=1",
                   "-DOOT2_AUDIO_SDL3=1", "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"),
                   "-I", str(RUNTIME / "audio"), "-I", str(kit / "include"), str(TEST_SOURCE), *map(str, SOURCES),
                   str(kit / "lib" / "libSDL3.a"), *fetch_sdl3.LINK_LIBS, *extra, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe
        cls.out = out

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str) -> subprocess.CompletedProcess:
        env = {**os.environ, "SDL_AUDIO_DRIVER": "dummy", "OOT2_TEST_DEVICE_FILE": str(self.out / f"device-{level}.raw")}
        done = subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(done.returncode, 0, f"{case} at {level}:\n{done.stdout}{done.stderr}")
        self.assertIn(f"ok {case}", done.stdout)
        return done

    def cases(self, *names: str) -> None:
        for level in LEVELS:
            for case in names:
                with self.subTest(level=level, case=case):
                    self.run_case(level, case)

    def test_the_sink_opens_on_the_platform_stream_and_falls_back_when_auto_cannot(self) -> None:
        self.cases("open")

    def test_the_queue_is_held_to_about_30_ms_and_a_write_past_it_is_refused_whole_and_counted(self) -> None:
        self.cases("queue_cap")

    def test_a_drained_queue_is_an_underrun_at_the_next_write_and_the_prefill_starts_again(self) -> None:
        self.cases("underrun")

    def test_a_paused_sink_holds_no_queue_and_counts_no_underrun(self) -> None:
        self.cases("pause")

    def test_the_device_takes_what_was_written_at_its_own_rate(self) -> None:
        self.cases("drain")

    def test_the_frames_reach_the_device_in_order_and_unchanged(self) -> None:
        self.cases("samples")

    def test_every_case_of_the_binary_is_run_by_a_test(self) -> None:
        text = TEST_SOURCE.read_text(encoding="utf-8")
        for case in CASES:
            self.assertIn(f'"{case}"', text)


class BuildFlagsTest(unittest.TestCase):
    """The sink calls the driver only where the driver file is compiled with the kit's headers (tools/build_game.py)."""

    def test_both_flags_come_with_the_kit_and_only_with_it(self) -> None:
        import dataclasses
        import re

        sys.path.insert(0, str(ROOT / "tests"))
        from test_build_game import fake_gen, fake_tool
        from test_fetch_sdl3 import fake_sdl3_kit

        with tempfile.TemporaryDirectory(prefix="oot2-audio-sdl3-", ignore_cleanup_errors=True) as tmp:
            folder = Path(tmp)
            gen = folder / "gen"
            fake_gen(gen, {"main": 1})
            tool = fake_tool(folder)

            def flags(t: build_game.Toolchain, name: str) -> str:
                b = build_game.Build(folder / "build", t, gen=gen, exefs=folder, jobs=1)
                _, lifted = build_game.lifted_modules(gen)
                text = build_game.plan(b, lifted).text
                return next(line for line in text.splitlines() if line.startswith(f"{name} = "))

            with_kit = dataclasses.replace(tool, sdl3=fake_sdl3_kit(folder / "kit"))
            for name in ("rtflags", "cxxflags"):
                self.assertRegex(flags(with_kit, name), r" -DOOT2_WITH_SDL3=1 -DOOT2_AUDIO_SDL3=1 ")
                self.assertNotRegex(flags(tool, name), re.compile("AUDIO_SDL3"))
            for name in ("genflags", "glueflags"):
                self.assertNotIn("AUDIO_SDL3", flags(with_kit, name))


if __name__ == "__main__":
    unittest.main()
