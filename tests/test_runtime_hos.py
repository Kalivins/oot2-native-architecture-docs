# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the Horizon layer under the SDK (runtime/hos).

Builds tests/runtime/hos_test.c with runtime/core, runtime/nvn and runtime/hos
against a synthetic layout (one module of 0x8000 bytes, main), at -O1 and -O2
with -march=x86-64-v3, then runs each case in its own process with a fresh
game folder. The guest programs are C functions written by hand in the
lifter's shape; the messages are built by hand in a guest thread's message
buffer; nothing is read from a dump. Every file of runtime/hos is compiled
here, so this is also the compile check of the overrides.

A passing case must print "ok <case>"; a failing case must end in exactly one
FATAL line with the expected exit code and text, and leave census.tsv with
its stop row in the game folder.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

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
TEST_SOURCE = ROOT / "tests" / "runtime" / "hos_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000
TIMEOUT = 60

EXIT_FATAL, EXIT_UNIMPLEMENTED, EXIT_GUEST_STOP, EXIT_CENSUS = 70, 71, 72, 73

MOVIE = ("_ZN5movie12VideoDecoder10InitializeEPvmN2nn5movie13MediaMimeTypeENS_24VideoDecoderOutputFormatENS_14"
         "VideoCodecModeE")
OPEN = ("_ZN2nn5audio17OpenAudioRendererEPNS0_19AudioRendererHandleEPNS_2os11SystemEventERKNS0_22AudioRendererParameterEPvm")
ACQUIRE = "_ZN2nn5audio36AcquireAudioOutputDeviceNotificationEPNS_2os11SystemEventE"

PASSING = ("sm", "notes", "audio")

# case: (exit code, text the FATAL line must contain, a row census.tsv must hold)
FAILING = {
    "service": (EXIT_UNIMPLEMENTED, 'FATAL: unimplemented service: sm: GetServiceHandle("lm"): no service of that '
                                    'name is served', "service\tlm\tnamed-fatal\t1\t"),
    "port": (EXIT_UNIMPLEMENTED, 'FATAL: unimplemented port: ConnectToNamedPort("xyz:"): only sm: is served',
             "svc\t0x1f ConnectToNamedPort\timplemented\t1\t"),
    "command": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented service command: sm: tipc command 7 (0 bytes of arguments",
                "command\tsm: tipc 7\tnamed-fatal\t1\t"),
    "domain": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented service command: sm: control 0 "
                                   "ConvertCurrentObjectToDomain: CMIF domains come with",
               "command\tsm: control 0\tnamed-fatal\t1\t"),
    "boot": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented boot: the SDK function _ZN2nn4init5StartEmmPFvvES2_S2_ is not "
                                 "defined by any module", "boot\t<rtld>\timplemented\t1\t"),
    # the movie player's first step is a named fatal until the movie ticket
    "movie": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented movie: movie::VideoDecoder::Initialize: the game's movie player "
                                  "starts, and the host has no video decoder and no media extractor yet",
              f"import\t{MOVIE}\tnamed-fatal\t1\t0x7000000250 (main+0x250)\t"),
    # the audio stand-ins that bind the game's SystemEvent reach the SDK's own attach function, which this synthetic
    # module does not define; a null handle or event stops the guest; a second renderer is a gap
    "audio_open_binds": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented boot: the SDK function "
                                             "_ZN2nn2os33AttachReadableHandleToSystemEventEPNS0_15SystemEventTypeEjbNS0_14"
                                             "EventClearModeE is not defined by any module",
                         f"import\t{OPEN}\tstub\t1\t0x7000000250 (main+0x250)\t"),
    "audio_open_null": (EXIT_GUEST_STOP, "FATAL: guest stop: nn::audio::OpenAudioRenderer: null handle or event",
                        f"import\t{OPEN}\tstub\t1\t0x7000000250 (main+0x250)\t"),
    "audio_notification_binds": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented boot: the SDK function "
                                                     "_ZN2nn2os33AttachReadableHandleToSystemEventEPNS0_15SystemEventTypeE"
                                                     "jbNS0_14EventClearModeE is not defined by any module",
                                 f"import\t{ACQUIRE}\tstub\t1\t0x7000000250 (main+0x250)\t"),
    "audio_second": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented audio: nn::audio::OpenAudioRenderer: a second renderer; "
                                         "the stand-in keeps one", "svc\t0x01 SetHeapSize\timplemented\t1\t"),
}


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class HosTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-hos-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)]
        for part in ("core", "nvn", "hos"):
            sources += sorted(str(p) for p in (RUNTIME / part).glob("*.c"))
        # Windows: without high-entropy ASLR the host's own stacks stay below 4 GiB, out of the fixed module region
        # (runtime/core/memory.c, rt_mem_init); the program is linked the same way.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"hos_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(out), *sources, "-o",
                   str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, **census: str) -> tuple[subprocess.CompletedProcess, Path]:
        folder = Path(tempfile.mkdtemp(prefix=f"{case}{level}-", dir=self.tmp.name))
        env = {k: v for k, v in os.environ.items() if not k.startswith("OOT2_CENSUS")}   # a default run unless asked
        env.update(census)
        try:
            done = subprocess.run([str(self.exes[level]), case, str(folder)], capture_output=True, text=True,
                                  timeout=TIMEOUT, env=env)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{case} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")
        return done, folder

    def test_passing_cases(self) -> None:
        for level in LEVELS:
            for case in PASSING:
                with self.subTest(level=level, case=case):
                    done, _ = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)

    def test_census_mode_refuses_an_unserved_service_and_goes_on(self) -> None:
        env = dict(os.environ, OOT2_CENSUS="1")
        for level in LEVELS:
            with self.subTest(level=level):
                folder = Path(tempfile.mkdtemp(prefix=f"census{level}-", dir=self.tmp.name))
                done = subprocess.run([str(self.exes[level]), "service_census", str(folder)], capture_output=True,
                                      text=True, timeout=TIMEOUT, env=env)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok service_census", done.stdout)
                self.assertIn('[census mode] unserved service: sm: GetServiceHandle("lm") refused with 0x415', done.stderr)
                self.assertNotIn("FATAL", done.stderr)
                # each name is listed once, the same name again is not, and the list holds 32 names (lm and 31 of 40)
                self.assertEqual(done.stderr.count('GetServiceHandle("lm") refused'), 1, done.stderr)
                self.assertEqual(done.stderr.count("unserved service: sm: GetServiceHandle("), 32, done.stderr)
                self.assertIn('GetServiceHandle("svc00030") refused', done.stderr)
                self.assertNotIn('GetServiceHandle("svc00031") refused', done.stderr)
        # a default run of the same case is the named fatal
        done, _ = self.run_case(LEVELS[0], "service")
        self.assertEqual(done.returncode, EXIT_UNIMPLEMENTED)
        self.assertNotIn("census mode", done.stderr)

    def test_every_line_of_a_census_mode_run_is_marked_and_the_notice_comes_first(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, folder = self.run_case(level, "movie", OOT2_CENSUS="1")   # a run that ends in a named fatal
                self.assertEqual(done.returncode, EXIT_UNIMPLEMENTED, done.stdout + done.stderr)
                lines = done.stderr.splitlines()
                self.assertRegex(lines[0], r"^\[rt\] \[census mode\] OOT2_CENSUS=1: gaps in NVN2 are listed and passed")
                self.assertTrue(all("[census mode]" in line for line in lines), done.stderr)
                fatal = [line for line in lines if line.startswith("FATAL")]
                self.assertEqual(len(fatal), 1, done.stderr)
                self.assertTrue(fatal[0].endswith(" [census mode]"), fatal[0])
                census = (folder / "census.tsv").read_text(encoding="utf-8")
                self.assertIn(f"\nstop\texit 71\tnamed-fatal\t1\t\t{fatal[0]}\n", census)
        # the same case in a default run has no mark anywhere
        done, _ = self.run_case(LEVELS[0], "movie")
        self.assertNotIn("census mode", done.stderr)

    def test_census_mode_passes_an_nvn_gap_and_marks_the_row(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.run_case(level, "nvn_census", OOT2_CENSUS="1")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok nvn_census", done.stdout)
                self.assertRegex(done.stderr, r"(?m)^\[rt\] \[census mode\] unbodied nvn: nvn\w+, answered 0, at ")
                self.assertNotIn("FATAL", done.stderr)

    def test_census_files_in_both_modes(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level, mode="census"):
                done, folder = self.run_case(level, "census_files", OOT2_CENSUS="1")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok census_files", done.stdout)
                nvn = (folder / "census-nvn.tsv").read_text(encoding="utf-8")
                self.assertTrue(nvn.startswith("#census-mode OOT2_CENSUS=1: not a default run\n"), nvn[:80])
            with self.subTest(level=level, mode="default"):
                done, folder = self.run_case(level, "census_files")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok census_files", done.stdout)
                self.assertFalse((folder / "census-nvn.tsv").exists())
                self.assertNotIn("census mode", done.stderr)

    def test_a_census_mode_run_ends_with_its_own_exit_code_and_its_census(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, folder = self.run_case(level, "census_end", OOT2_CENSUS="1")
                self.assertEqual(done.returncode, EXIT_CENSUS, done.stdout + done.stderr)
                census = (folder / "census.tsv").read_text(encoding="utf-8")
                self.assertTrue(census.startswith("#census-mode OOT2_CENSUS=1"), census[:80])
                self.assertIn("\nstop\texit 73\tcensus\t1\t\t[census mode] the case ended at its frame limit\n", census)
                # the files are named in the log, each line marked
                self.assertRegex(done.stderr, r"(?m)^\[rt\] \[census mode\] census: written to \S*census\.tsv$")
                self.assertRegex(done.stderr, r"(?m)^\[rt\] \[census mode\] census: calls per frame written to "
                                              r"\S*census-nvn\.tsv$")
                self.assertTrue((folder / "census-nvn.tsv").read_text(encoding="utf-8").startswith("#census-mode"))

    def test_the_audio_renderer_signals_its_frame_event_while_it_runs(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.run_case(level, "audio_events")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok audio_events", done.stdout)
                opened = re.findall(r"^\[rt\] audio: stand-in renderer opened: (.*), nothing is mixed$", done.stderr, re.M)
                self.assertEqual(opened, ["48000 Hz, 480 samples a frame (10.00 ms)"] +
                                 ["16000 Hz, 160 samples a frame (10.00 ms)"] * 2 +
                                 ["48000 Hz, 240 samples a frame (5.00 ms)"] * 3, done.stderr)

    def test_a_memory_fault_names_the_guest_function_and_the_threads_frames(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.run_case(level, "fault")
                self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                self.assertIn("[rt] guest state as last saved before the fault: x30 0x7000000124 (main+0x124), x29 chain "
                              "follows\n", done.stderr)
                self.assertIn("[rt]   frame 0: 0x7000000144 (main+0x144)\n", done.stderr)
                fatal = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
                self.assertEqual(len(fatal), 1, done.stderr)
                self.assertRegex(fatal[0], r"^FATAL: guest memory: read of 0x7f00000000: no guest memory is mapped there "
                                           r"\(host pc 0x[0-9a-f]+, in the guest function 0x7000000200 \(main\+0x200\)\)$")

    def test_a_fault_while_reading_the_frame_chain_keeps_the_first_fault_as_the_reason(self) -> None:
        # x29 aligned, below the address limit, and mapped nowhere: the report's own read of the chain faults
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.run_case(level, "fault_chain")
                self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                self.assertIn("[rt] guest state as last saved before the fault: x30 0x7000000124 (main+0x124), x29 chain "
                              "follows\n", done.stderr)
                fatal = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
                self.assertEqual(len(fatal), 1, done.stderr)
                self.assertRegex(fatal[0], r"^FATAL: guest memory: read of 0x7f00000000: no guest memory is mapped there "
                                           r"\(host pc 0x[0-9a-f]+, in the guest function 0x7000000200 \(main\+0x200\)\); "
                                           r"reading the thread's frame chain for this report then faulted: read of "
                                           r"0x7e00000000: no guest memory is mapped there$")

    def test_unbound_overrides_are_logged(self) -> None:
        done, _ = self.run_case(LEVELS[0], "notes")
        self.assertIn("hos: the override of hos_test_unbound binds nothing in these modules", done.stderr)
        self.assertNotIn("the override of hos_test_bound binds nothing", done.stderr)

    def test_fatal_cases(self) -> None:
        for level in LEVELS:
            for case, (code, text, row) in FAILING.items():
                with self.subTest(level=level, case=case):
                    done, folder = self.run_case(level, case)
                    self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    self.assertNotIn(f"ok {case}", done.stdout)
                    fatal_lines = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
                    self.assertEqual(len(fatal_lines), 1, done.stderr)
                    census = (folder / "census.tsv").read_text(encoding="utf-8")
                    self.assertIn("\n" + row, census)
                    stop = [line for line in census.splitlines() if line.startswith("stop\t")]
                    self.assertEqual(len(stop), 1, census)
                    self.assertTrue(stop[0].startswith(f"stop\texit {code}\tnamed-fatal\t1\t\t{fatal_lines[0]}"),
                                    stop[0])

    def test_an_unbodied_nvn_entry_point_is_a_named_fatal_in_the_census(self) -> None:
        done, folder = self.run_case(LEVELS[0], "nvn")
        self.assertEqual(done.returncode, EXIT_UNIMPLEMENTED, done.stdout + done.stderr)
        name = re.search(r"^case nvn: (nvn\w+)$", done.stderr, re.M)
        self.assertIsNotNone(name, done.stderr)
        self.assertIn(f"FATAL: unimplemented nvn: {name[1]} ", done.stderr)
        census = (folder / "census.tsv").read_text(encoding="utf-8")
        self.assertIn(f"\nnvn\t{name[1]}\tnamed-fatal\t1\t0x7000000250 (main+0x250)\tan NVN2 entry point with no body "
                      "yet: it stops with its name\n", census)

    def test_unknown_case_is_refused(self) -> None:
        done, _ = self.run_case(LEVELS[0], "no_such_case")
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
