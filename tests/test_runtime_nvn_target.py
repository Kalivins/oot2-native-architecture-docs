# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the fixed frame rate (the fixed frame rate gate, architecture D22, the frame-rate study): the
settings store's frame_rate_target N puts the acquire gate on a second schedule of N slots a
second, anchored on the clock's origin, and the game's own interval is ignored while it is set.

Builds tests/runtime/nvn_target_test.c with runtime/core and runtime/nvn at -O1 and -O2, as
tests/test_runtime_nvn_sync.py builds its harness, and runs each case in its own process. The game
is a synthetic guest thread written as a host function; the clock is the vsync ticket's driven
clock, so an hour of slots runs in seconds and nothing waits on the wall clock but the realtime
cases (their bounds: never early, and the waiting thread burns no processor).

- hour: N = 40 over a driven hour (144,001 presents) and N = 100 (360,001), every acquire back
  at once, every wait ended in its slot, every due count equal to ceil(m * 31,250,000 / N), and a
  nanosecond before the due count the wait still waiting. The game presenting before its wait too.
- identity: N = 60 gives the very records of the run without a target at interval 1, N = 30 of
  interval 2 and N = 20 of interval 3 (a digest of every due count of 10 driven minutes).
- the rule when the periods differ: the game asking for 2 at N = 60, for 3 at N = 100, for 1 at
  N = 40, and the records keeping what the game asked for beside the target.
- events: the game's vsync events are 216,000 over the hour at N = 40 (and 36,000 over ten
  minutes at N = 100), none skipped, with the gate running on its own schedule.
- change: N changes at the next acquire (40, 100, 0, 50), the interval the game asked for coming
  back when N is off.
- script: the default placement signals the sync at the due count and the acquire returns at
  once; the old placement blocks inside the call; the E2 hold floors the period under both.
- late / null_sync / seam: no burst after a late present, no sync no hold, the back end told the
  due count.
- schedule: the primitives against the model by hand.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests skip, unless
OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

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

TEST_SOURCE = nvn.ROOT / "tests" / "runtime" / "nvn_target_test.c"
TIMEOUT = 400
EXIT_HUNG = 3   # nvn_target_test.c's watchdog: the case had not ended after 170 s; it logs the kernel's state first

WAIT_PRESENT, PRESENT_WAIT = 0, 1   # where the game presents: after its wait on the acquire sync, or before
SYNC, ACQUIRE = 0, 1                # NvnAcquireWait
HOUR, TEN_MINUTES = 3600, 600

DIGEST = re.compile(r"^digest ([0-9a-f]{16})$", re.M)
SCRIPT_FRAME = re.compile(r"^frame (\d+) acquired at count (\d+) synced at count (\d+) due count (\d+) requested (\d+) "
                          r"held (\d+) target (\d+)$", re.M)


class TargetTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = nvn.find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-nvntarget-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(nvn.END_SOURCE)] + sorted(str(p) for p in nvn.CORE.glob("*.c")) + \
            sorted(str(p) for p in nvn.NVN.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in nvn.LEVELS:
            exe = out / f"nvn_target_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*nvn.compile_flags(cc), level, *extra, "-I", str(nvn.CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, args: list[str], rate: int | None = None,
                 extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        # no setting and no debug variable comes from the environment of the run unless the case says so
        env = {k: v for k, v in os.environ.items()
               if k not in ("OOT2_HOLD_TICKS", "OOT2_HOLD_PLACEMENT", "OOT2_FRAME_RATE_TARGET", "OOT2_PACE_TRACE",
                            "OOT2_PROBE_FILE")}
        if rate is not None:
            env["OOT2_FRAME_RATE_TARGET"] = str(rate)
        env.update(extra or {})
        try:
            done = subprocess.run([str(self.exes[level]), *args], capture_output=True, text=True, timeout=TIMEOUT,
                                  env=env)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{args} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")
        if done.returncode == EXIT_HUNG:
            self.fail(f"{args} at {level} hung (the case's watchdog):\n{done.stdout}{done.stderr}")
        return done

    def passes(self, args: list[str], rate: int | None = None, extra: dict[str, str] | None = None) -> list[str]:
        outs = []
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=args, rate=rate):
                done = self.run_case(level, args, rate, extra)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {args[0]}\n", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                outs.append(done.stdout)
        return outs

    def hour(self, hz: int, periods: int, interval: int, order: int, seconds: int, rate: int | None) -> list[str]:
        return self.passes(["hour", str(hz), str(periods), str(interval), str(order), str(seconds)], rate)

    def digests(self, outs: list[str]) -> list[str]:
        found = [DIGEST.search(o) for o in outs]
        self.assertTrue(all(found), outs)
        return [m[1] for m in found]

    # ---- the schedule
    def test_the_primitives_are_the_model_by_hand_and_an_hour_never_drifts(self) -> None:
        self.passes(["schedule"])

    # ---- the hours
    def test_n40_over_a_driven_hour_has_every_due_count_on_the_model(self) -> None:
        for out in self.hour(40, 1, 1, WAIT_PRESENT, HOUR, 40):
            self.assertRegex(out, r"hour: 144001 presents, 40 slots a second, the game asking for interval 1; every "
                                  r"acquire came back at once, every wait ended in the slot 1 slots apart from slot 0 to "
                                  r"slot 144000, every due count the model's,")

    def test_n100_over_a_driven_hour_has_every_due_count_on_the_model(self) -> None:
        for out in self.hour(100, 1, 1, WAIT_PRESENT, HOUR, 100):
            self.assertRegex(out, r"hour: 360001 presents, 100 slots a second, the game asking for interval 1; every "
                                  r"acquire came back at once, every wait ended in the slot 1 slots apart from slot 0 to "
                                  r"slot 360000, every due count the model's,")

    def test_a_game_that_presents_before_its_wait_gets_the_same_due_counts_at_n40(self) -> None:
        after = self.digests(self.hour(40, 1, 1, WAIT_PRESENT, TEN_MINUTES, 40))
        before = self.digests(self.hour(40, 1, 1, PRESENT_WAIT, TEN_MINUTES, 40))
        self.assertEqual(after, before)

    # ---- identity with the acquire gate where the target is a whole number of ticks
    def test_n60_is_the_acquire_gates_behaviour_tick_for_tick_at_interval_1(self) -> None:
        target = self.digests(self.hour(60, 1, 1, WAIT_PRESENT, TEN_MINUTES, 60))
        none = self.digests(self.hour(60, 1, 1, WAIT_PRESENT, TEN_MINUTES, None))
        self.assertEqual(target, none)

    def test_n30_is_interval_2_and_n20_is_interval_3_tick_for_tick(self) -> None:
        self.assertEqual(self.digests(self.hour(30, 1, 1, WAIT_PRESENT, TEN_MINUTES, 30)),
                         self.digests(self.hour(60, 2, 2, WAIT_PRESENT, TEN_MINUTES, None)))
        self.assertEqual(self.digests(self.hour(20, 1, 1, WAIT_PRESENT, TEN_MINUTES, 20)),
                         self.digests(self.hour(60, 3, 3, WAIT_PRESENT, TEN_MINUTES, None)))

    # ---- the rule when the periods differ
    def test_the_targets_period_replaces_the_games_interval_and_the_records_keep_both(self) -> None:
        for out in self.hour(60, 1, 2, WAIT_PRESENT, 60, 60):    # the game asks for 2, N is 60: one frame a tick
            self.assertIn("the game asking for interval 2;", out)
        for out in self.hour(100, 1, 3, WAIT_PRESENT, 60, 100):  # the game asks for 3, N is 100: one frame in 10 ms
            self.assertIn("the game asking for interval 3;", out)
        for out in self.hour(40, 1, 1, WAIT_PRESENT, 60, 40):    # the game asks for 1, N is 40: 1.5 ticks
            self.assertIn("the game asking for interval 1;", out)

    # ---- the vsync events
    def test_the_games_vsync_events_stay_at_216000_an_hour_whatever_n_is(self) -> None:
        for out in self.passes(["events", "40", str(HOUR)], 40):
            self.assertIn("events: 144001 presents at 40 a second and 216000 vsync events over 3600 s, none skipped", out)
        for out in self.passes(["events", "100", str(TEN_MINUTES)], 100):
            self.assertIn("events: 60001 presents at 100 a second and 36000 vsync events over 600 s, none skipped", out)

    # ---- the setting, live
    def test_a_change_of_n_applies_at_the_next_acquire(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, ["change"])
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok change\n", done.stdout)
                for line in ("present 1: due count 0 tick 0 requested 2 target 40",
                             "present 2: due count 781250 tick 1 requested 2 target 100",
                             "present 3: due count 937500 tick 1 requested 2 target 100",
                             "present 4: due count 1562500 tick 3 requested 2 target 0",
                             "present 5: due count 1875000 tick 3 requested 2 target 50"):
                    self.assertIn(line, done.stdout)
                for rate in ("40", "100", "50"):
                    self.assertRegex(done.stderr, rf"nvn: fixed frame rate: presents are paced at {rate} a second on a "
                                                  rf"schedule of their own, whatever interval the game asks for "
                                                  rf"\(frame_rate_target = {rate},")
                self.assertIn("nvn: no fixed frame rate: the game's own interval on the 60 Hz clock", done.stderr)

    # ---- where the gate acts
    def script(self, placement: int, hold: int, periods: int, rate: int, frames: int = 60) -> list[str]:
        return self.passes(["script", str(placement), str(hold), str(periods), str(frames)], rate)

    def test_the_default_placement_signals_the_sync_at_the_due_count_and_the_acquire_returns_at_once(self) -> None:
        for out in self.script(SYNC, 0, 1, 40):
            frames = [tuple(int(v) for v in f) for f in SCRIPT_FRAME.findall(out)]
            self.assertEqual(len(frames), 60)
            self.assertEqual(frames[1][1:4], (0, 781250, 781250))      # acquired at count 0, synced at 781,250
            self.assertEqual(frames[2][1:4], (781250, 1562500, 1562500))

    def test_the_old_placement_blocks_inside_the_call_until_the_due_count(self) -> None:
        for out in self.script(ACQUIRE, 0, 1, 40):
            frames = [tuple(int(v) for v in f) for f in SCRIPT_FRAME.findall(out)]
            self.assertEqual(frames[1][1:4], (781250, 781250, 781250))   # the acquire itself came back at the gate
            self.assertEqual(frames[2][1:4], (1562500, 1562500, 1562500))

    def test_the_e2_hold_floors_the_period_in_whole_slots_under_both_placements(self) -> None:
        for placement in (SYNC, ACQUIRE):
            for hold, rate, periods in ((3, 40, 2),     # 3 ticks are exactly 2 slots of 40
                                        (1, 100, 2),    # 1 tick is 1.67 slots: rounded up, never shorter than the hold
                                        (2, 100, 4),    # 2 ticks are 3.33: 4 slots
                                        (1, 20, 1)):    # one slot of 20 is 3 ticks, longer than the hold: no floor
                with self.subTest(placement=placement, hold=hold, rate=rate):
                    for out in self.script(placement, hold, periods, rate, frames=30):
                        frames = [tuple(int(v) for v in f) for f in SCRIPT_FRAME.findall(out)]
                        self.assertTrue(all(f[5] == hold and f[6] == rate and f[4] == 1 for f in frames), frames[:3])

    # ---- late presents, no sync, the back end
    def test_a_late_present_takes_its_own_slot_and_there_is_no_burst_of_syncs_to_catch_up(self) -> None:
        self.passes(["late"], 40)

    def test_a_game_that_passes_no_sync_is_not_held_with_a_target_either(self) -> None:
        self.passes(["null_sync"], 50)

    def test_the_back_end_is_told_the_due_count_and_the_tick_it_falls_in(self) -> None:
        self.passes(["seam"], 40)

    # ---- the cost of the wait
    def test_the_wait_never_ends_early_and_the_waiting_thread_burns_no_processor(self) -> None:
        for rate in (40, 100):
            for out in self.passes(["realtime", "3"], rate):
                self.assertRegex(out, rf"realtime: {3 * rate + 1} presents at {rate} a second took \d+ ms of the host's "
                                      rf"clock; no wait ended early; the game thread used")


if __name__ == "__main__":
    unittest.main()
