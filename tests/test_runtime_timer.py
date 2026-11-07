# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the system timer resolution (ticket 116): the request at start and its release on every exit path.

Builds tests/runtime/timer_test.c with runtime/core against a synthetic layout, at -O1 and -O2 with -march=x86-64-v3,
then runs each case in its own process. The machine-wide timer resolution is shared with every process (a browser, another
run of the game), so the cases never assume what the system default is. Each installs a recorder through plat_timer_ops
that answers the request and prints one unbuffered line per call, "timer seam: set 5000 on" or "off" (units of 100 ns), so
the request and the release are read from the process's output on every platform, the Linux leg included. The one case
that asks the real system, `held`, reads the resolution back while the request is held and asserts only that it is at or
under 0.5 ms, or the finest the system allows when that is coarser (a virtual machine's timer may be); the finest is
printed. It also asks the system directly, which no other process can answer for: taking this process's request back
with NtSetTimerResolution(off) answers 0 and then STATUS_TIMER_RESOLUTION_NOT_SET, and after plat_timer_release a direct
take-back answers NOT_SET (the release really gave the request back). Nothing asserts that the resolution went back to
the default after a release.

The exit paths: a normal return from main, ExitProcess (SVC 0x07) through plat_exit, rt_fatal, a fatal whose stop hook
raises another fatal, a named unimplemented (exit 71), a guest stop (exit 72), the census mode's end, and on Windows a
console control event (Ctrl+Break, raised by the process itself in a group of its own; the system's exit code 0xC000013A
stays). The setting off, through the environment and through --set, makes no call at all.

The settings variable OOT2_TIMER_RESOLUTION is removed from the environment the cases inherit, except in the case that
sets it.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests skip, unless OOT2_REQUIRE_CLANG is set, as
continuous integration sets it.

Run: python -m unittest discover -s tests -p test_runtime_timer.py -v
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

import layout  # noqa: E402

CORE = ROOT / "runtime" / "core"
TEST_SOURCE = ROOT / "tests" / "runtime" / "timer_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000
TIMEOUT = 120

EXIT_FATAL = 70
EXIT_UNIMPLEMENTED = 71
EXIT_GUEST_STOP = 72
STATUS_CONTROL_C_EXIT = 0xC000013A

ON = "timer seam: set 5000 on"
OFF = "timer seam: set 5000 off"


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class TimerTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-timer-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        # Windows: without high-entropy ASLR the host's own stacks stay below 4 GiB, out of the fixed module region
        # (runtime/core/memory.c, rt_mem_init); the program is linked the same way.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"timer_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(ROOT / "runtime" / "include"), "-I", str(CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, env: dict[str, str] | None = None,
                 creationflags: int = 0) -> subprocess.CompletedProcess:
        full = {k: v for k, v in os.environ.items() if k != "OOT2_TIMER_RESOLUTION"}
        full.update(env or {})
        try:
            return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=TIMEOUT,
                                  env=full, creationflags=creationflags)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{case} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")

    def check_calls(self, done: subprocess.CompletedProcess, ons: int, offs: int) -> None:
        lines = done.stdout.splitlines()
        self.assertEqual(lines.count(ON), ons, done.stdout + done.stderr)
        self.assertEqual(lines.count(OFF), offs, done.stdout + done.stderr)
        self.assertEqual(sum(1 for line in lines if line.startswith("timer seam:")), ons + offs, done.stdout)

    def ends_with(self, level: str, case: str, code: int, ons: int = 1, offs: int = 1, env: dict[str, str] | None = None,
                  ) -> subprocess.CompletedProcess:
        done = self.run_case(level, case, env)
        self.assertEqual(done.returncode, code, done.stdout + done.stderr)
        self.check_calls(done, ons, offs)
        if ons and offs:
            lines = done.stdout.splitlines()
            self.assertLess(lines.index(ON), lines.index(OFF), "the release came before the request")
        return done

    def test_the_request_is_made_once_before_the_clock_thread_exists(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.ends_with(level, "start", 0)
                self.assertIn("start: one request of 0.5 ms with 0 events bound and 0 ticks sent", done.stdout)
                self.assertIn("ok start", done.stdout)

    def test_a_normal_return_from_main_releases_once(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.ends_with(level, "normal", 0)
                self.assertIn("ok normal", done.stdout)

    def test_exit_process_releases_once(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.ends_with(level, "exit_process", 0)
                self.assertIn("guest ExitProcess at", done.stderr)

    def test_a_fatal_error_releases_once(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.ends_with(level, "fatal", EXIT_FATAL)
                self.assertIn("FATAL: timer test: a fatal error", done.stderr)

    def test_a_second_fatal_error_inside_the_stop_hook_still_releases_once(self) -> None:
        done = self.ends_with(LEVELS[0], "fatal_in_hook", EXIT_FATAL)
        self.assertIn("a fatal error whose hook fails", done.stderr)

    def test_a_named_unimplemented_releases_once(self) -> None:
        done = self.ends_with(LEVELS[0], "unimplemented", EXIT_UNIMPLEMENTED)
        self.assertIn("FATAL: unimplemented test: timer test", done.stderr)

    def test_a_guest_stop_releases_once(self) -> None:
        done = self.ends_with(LEVELS[0], "guest_stop", EXIT_GUEST_STOP)
        self.assertIn("FATAL: guest stop: brk #0x7", done.stderr)

    def test_the_census_modes_end_releases_once(self) -> None:
        self.ends_with(LEVELS[0], "census_end", 0)

    @unittest.skipUnless(os.name == "nt", "console control events are Windows's")
    def test_a_console_control_event_releases_once_and_the_exit_code_stays_the_systems(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "ctrl", creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                self.assertEqual(done.returncode & 0xFFFFFFFF, STATUS_CONTROL_C_EXIT, done.stdout + done.stderr)
                self.check_calls(done, 1, 1)
                self.assertNotIn("FAIL", done.stderr)

    def test_the_setting_off_through_the_environment_makes_no_call(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.ends_with(level, "off", 0, ons=0, offs=0, env={"OOT2_TIMER_RESOLUTION": "0"})
                self.assertIn("ok off", done.stdout)

    def test_the_setting_off_through_set_makes_no_call(self) -> None:
        done = self.ends_with(LEVELS[0], "off_set", 0, ons=0, offs=0)
        self.assertIn("ok off_set", done.stdout)

    def test_the_default_is_on(self) -> None:
        done = self.ends_with(LEVELS[0], "on_by_default", 0)
        self.assertIn("ok on_by_default", done.stdout)

    def test_a_refused_request_is_not_fatal_and_is_not_released(self) -> None:
        done = self.ends_with(LEVELS[0], "refused", 0, ons=1, offs=0)
        self.assertIn("[rt] timer: the system would not hold 0.500 ms", done.stderr)

    def test_the_log_line_names_the_request_and_the_setting(self) -> None:
        done = self.run_case(LEVELS[0], "normal")
        self.assertIn("[rt] timer: holding 0.500 ms (the system granted 0.500 ms, 0.500 ms in force now; "
                      "timer_resolution = 1, the default)", done.stderr)
        done = self.run_case(LEVELS[0], "off", {"OOT2_TIMER_RESOLUTION": "0"})
        self.assertIn("[rt] timer: the system's resolution is left as it is, not requested (timer_resolution = 0, "
                      "OOT2_TIMER_RESOLUTION)", done.stderr)

    @unittest.skipUnless(os.name == "nt", "the real timer resolution is Windows's")
    def test_the_real_resolution_read_back_while_held_is_at_or_under_what_the_system_allows(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "held")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok held", done.stdout)
                for line in done.stdout.splitlines():
                    if line.startswith("held:"):
                        print(f"timer figure: {level} {line}")

    def test_an_unknown_case_is_refused(self) -> None:
        done = self.run_case(LEVELS[0], "no_such_case")
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
