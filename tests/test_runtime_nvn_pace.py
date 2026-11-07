# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the acquire gate (architecture D22): AcquireTexture held on
the virtual 60 Hz clock, QueuePresentTexture returning at once, the per-present
records, the seam to the back end and the opt-in E2 hold.

Builds tests/runtime/nvn_pace_test.c with runtime/core and runtime/nvn at -O1
and -O2, as tests/test_runtime_nvn_queue.py builds its harness, and runs each
case in its own process. The game is a synthetic guest thread written as a host
function; the clock is the vsync ticket's driven clock, so an hour of ticks runs
in seconds and nothing waits on the wall clock but the one realtime case.

- hour1 / hour2: 216,000 driven ticks, wake-ups 0 to 15 ms into each tick:
  interval 1 puts a present on every tick, interval 2 on every other tick; each
  acquire came back at the tick of the model and not one earlier.
- block: AcquireTexture gives its virtual core up while it waits, so a worker of
  a worse priority on the same core runs, and the game takes the core back.
- present_nowait: QueuePresentTexture comes back with the clock stopped.
- first: an acquire with no previous present waits for nothing.
- late / change: a present 1.5 ticks late, exactly one tick late and 10.2 ticks
  late; an interval change 1, 2, 1 that applies at the next acquire.
- hold / hold_max / hold_default / hold_env: experiment E2.
- seam / ring: the due tick and the shown tick across gfx.h; the record ring.
- realtime: the host's clock, never early.

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

TEST_SOURCE = nvn.ROOT / "tests" / "runtime" / "nvn_pace_test.c"
TIMEOUT = 180
EXIT_HUNG = 3   # nvn_pace_test.c's watchdog: the case had not ended after 90 s; it logs the kernel's state first


class PaceTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = nvn.find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-nvnpace-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(nvn.END_SOURCE)] + sorted(str(p) for p in nvn.CORE.glob("*.c")) + \
            sorted(str(p) for p in nvn.NVN.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in nvn.LEVELS:
            exe = out / f"nvn_pace_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*nvn.compile_flags(cc), level, *extra, "-I", str(nvn.CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, hold_env: str | None = None) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != "OOT2_HOLD_TICKS"}
        env["OOT2_HOLD_PLACEMENT"] = "acquire"   # these cases are the old placement's: the hold inside the call (the acquire sync hold)
        if hold_env is not None:
            env["OOT2_HOLD_TICKS"] = hold_env
        try:
            done = subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=TIMEOUT,
                                  env=env)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{case} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")
        if done.returncode == EXIT_HUNG:
            self.fail(f"{case} at {level} hung (the case's watchdog):\n{done.stdout}{done.stderr}")
        return done

    def passes(self, case: str, hold_env: str | None = None) -> list[str]:
        outs = []
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=case):
                done = self.run_case(level, case, hold_env)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}\n", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                outs.append(done.stdout)
        return outs

    def test_interval_1_puts_an_hour_of_presents_on_consecutive_ticks(self) -> None:
        for out in self.passes("hour1"):
            self.assertRegex(out, r"hour: 216001 presents at interval 1, every 1 ticks from tick 0 to tick 216000,")

    def test_interval_2_puts_an_hour_of_presents_on_every_other_tick(self) -> None:
        for out in self.passes("hour2"):
            self.assertRegex(out, r"hour: 108001 presents at interval 2, every 2 ticks from tick 0 to tick 216000,")

    def test_acquire_blocks_while_another_guest_thread_runs_on_the_freed_core(self) -> None:
        for out in self.passes("block"):
            self.assertRegex(out, r"block: the worker counted \d+ while the game thread waited in AcquireTexture")
            self.assertIn("block: acquire 2 came back at tick 1\n", out)

    def test_present_returns_without_waiting_for_a_tick(self) -> None:
        for out in self.passes("present_nowait"):
            self.assertIn("present: 3 presents came back with the clock stopped at tick 5", out)

    def test_first_acquire_with_no_previous_present_waits_for_nothing(self) -> None:
        for out in self.passes("first"):
            self.assertIn("first: acquires with no previous present came back at once at tick 7", out)

    def test_a_late_present_takes_the_tick_it_arrives_in_and_the_grid_keeps_its_phase(self) -> None:
        # frame: acquired at tick, due tick (the model, written by hand in the test)
        want = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 6), (7, 17), (18, 18), (19, 19)]
        for out in self.passes("late"):
            got = [(int(a), int(d)) for a, d in re.findall(r"^frame \d+ acquired at tick (\d+) due (\d+) ", out, re.M)]
            self.assertEqual(got, want)

    def test_an_interval_change_applies_at_the_next_acquire_and_the_record_keeps_the_games_interval(self) -> None:
        want = [(0, 0, 1), (2, 2, 2), (4, 4, 2), (5, 5, 1), (6, 6, 3), (8, 8, 2), (9, 9, 1)]
        for out in self.passes("change"):
            got = [(int(a), int(d), int(r)) for a, d, r in
                   re.findall(r"^frame \d+ acquired at tick (\d+) due (\d+) requested (\d+) ", out, re.M)]
            self.assertEqual(got, want)

    def test_the_hold_puts_presents_two_ticks_apart_while_the_records_show_interval_1(self) -> None:
        for out in self.passes("hold"):
            frames = re.findall(r"^frame \d+ acquired at tick (\d+) due (\d+) requested (\d+) held (\d+)$", out, re.M)
            self.assertEqual([tuple(map(int, f)) for f in frames],
                             [(2 * k, 2 * k, 1, 2) for k in range(6)])

    def test_the_hold_raises_the_interval_and_never_lowers_it(self) -> None:
        for out in self.passes("hold_max"):
            frames = re.findall(r"^frame \d+ acquired at tick (\d+) due (\d+) requested (\d+) held (\d+)$", out, re.M)
            self.assertEqual([tuple(map(int, f)) for f in frames], [(0, 0, 1, 2), (3, 3, 3, 2), (6, 6, 3, 2), (9, 9, 3, 2)])

    def test_the_hold_is_off_by_default_and_only_the_environment_or_a_call_turns_it_on(self) -> None:
        for out in self.passes("hold_default"):
            frames = re.findall(r"^frame \d+ acquired at tick (\d+) due \d+ requested \d+ held (\d+)$", out, re.M)
            self.assertEqual([tuple(map(int, f)) for f in frames], [(k, 0) for k in range(4)])
        for out in self.passes("hold_env", hold_env="3"):
            frames = re.findall(r"^frame \d+ acquired at tick (\d+) due \d+ requested \d+ held (\d+)$", out, re.M)
            self.assertEqual([tuple(map(int, f)) for f in frames], [(3 * k, 3) for k in range(4)])
        for out in self.passes("hold_call", hold_env="3"):   # the call wins over the environment
            frames = re.findall(r"^frame \d+ acquired at tick (\d+) due \d+ requested \d+ held (\d+)$", out, re.M)
            self.assertEqual([tuple(map(int, f)) for f in frames], [(k, 0) for k in range(4)])

    def test_the_back_end_is_told_each_due_tick_and_reports_the_shown_tick_back(self) -> None:
        for out in self.passes("seam"):
            self.assertIn("seam: 41 presents; the back end was told each present's number and due tick", out)

    def test_the_record_ring_keeps_the_newest_presents_and_intervals_as_written(self) -> None:
        for out in self.passes("ring"):
            self.assertIn("ring: 600 presents; the newest 512 are kept (89 to 600)", out)

    def test_acquire_waits_on_the_host_clock_and_never_comes_back_early(self) -> None:
        for out in self.passes("realtime"):
            self.assertRegex(out, r"realtime: 12 presents at interval 2 took \d+ ms of the host's clock")

    def test_unknown_case_is_refused(self) -> None:
        done = self.run_case(nvn.LEVELS[0], "no_such_case")
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
