# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the acquire sync (ticket 95, architecture D22): under the
default placement AcquireTexture returns at once and the sync it returns is
signalled at the gate tick; the old placement (the hold inside the call) is
behind the settings store's key hold_placement = acquire (OOT2_HOLD_PLACEMENT, the
file, --set) or a call to nvn_acquire_wait.

Builds tests/runtime/nvn_sync_test.c with runtime/core and runtime/nvn at -O1
and -O2, as tests/test_runtime_nvn_pace.py builds its harness, and runs each
case in its own process. The game is a synthetic guest thread written as a host
function; the clock is the vsync ticket's driven clock, so an hour of ticks runs
in seconds and nothing waits on the wall clock but the two realtime cases, which exist because the host clock's own waits cannot
be shown on a driven one (their bounds: never early, and one generous upper bound).

- hour1 / hour2: 216,000 driven ticks, wake-ups 0 to 15 ms into each tick: every
  acquire comes back at once, and the wait on its sync ends on consecutive ticks
  (interval 1) or on every other tick (interval 2). hour1_pwait / hour2_pwait:
  the same with the game presenting before its wait, where the present is due at
  the tick its sync is signalled at.
- block: a wait on the acquire sync gives its virtual core up, so a worker of a
  worse priority on the same core runs.
- acquire_nowait / first / null_sync: the acquire never waits, the sync is not
  signalled before its tick and is at it, the first acquire is free, a game that
  passes no sync is not held.
- fence_and_tick / fence_clears_tick: the sync needs the tick and the fence value;
  a queue's fence replaces the tick.
- timeout / realtime_timeout: timeout 0, a short timeout and a long one, on the
  driven clock and on the host's.
- pace_seam: a present before its wait tells the back end the due tick it took.
- wake: an acquire wakes a wait parked on a sync not fenced yet.
- late / change / change_mid: the model of oot2/nvn_pace.h at the sync, with the
  interval read at the acquire.
- hold_*: experiment E2 under both placements.
- setting_*: the placement set by the environment, by the store's layers and live
  through its hook, and by the call (the call wins); an unknown value refused by
  name with exit code 2; a flip between two acquires.

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

TEST_SOURCE = nvn.ROOT / "tests" / "runtime" / "nvn_sync_test.c"
TIMEOUT = 180
EXIT_HUNG = 3   # nvn_sync_test.c's watchdog: the case had not ended after 90 s; it logs the kernel's state first

FRAME = re.compile(r"^frame (\d+) acquired at tick (\d+) synced at tick (\d+) due (\d+) requested (\d+) held (\d+) "
                   r"placement (\d+)$", re.M)


class SyncTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = nvn.find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-nvnsync-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(nvn.END_SOURCE)] + sorted(str(p) for p in nvn.CORE.glob("*.c")) + \
            sorted(str(p) for p in nvn.NVN.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in nvn.LEVELS:
            exe = out / f"nvn_sync_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*nvn.compile_flags(cc), level, *extra, "-I", str(nvn.CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, wait_env: str | None = None,
                 extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        # neither setting, and neither debug variable, comes from the environment of the run unless the case says so
        env = {k: v for k, v in os.environ.items()
               if k not in ("OOT2_HOLD_TICKS", "OOT2_HOLD_PLACEMENT", "OOT2_PACE_TRACE", "OOT2_PROBE_FILE")}
        if wait_env is not None:
            env["OOT2_HOLD_PLACEMENT"] = wait_env
        env.update(extra or {})
        try:
            done = subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=TIMEOUT,
                                  env=env)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{case} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")
        if done.returncode == EXIT_HUNG:
            self.fail(f"{case} at {level} hung (the case's watchdog):\n{done.stdout}{done.stderr}")
        return done

    def passes(self, case: str, wait_env: str | None = None) -> list[str]:
        outs = []
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=case):
                done = self.run_case(level, case, wait_env)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}\n", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                outs.append(done.stdout)
        return outs

    def frames(self, out: str) -> list[tuple[int, ...]]:
        return [tuple(int(v) for v in f) for f in FRAME.findall(out)]

    # ---- the default placement
    def test_interval_1_syncs_end_their_waits_on_consecutive_ticks_and_every_acquire_returns_at_once(self) -> None:
        for out in self.passes("hour1"):
            self.assertRegex(out, r"hour: 216001 presents at interval 1, the game presenting after its wait on the "
                                  r"acquire sync; every acquire came back at once, every wait ended 1 ticks apart "
                                  r"from tick 0 to tick 216000,")

    def test_interval_2_syncs_end_their_waits_on_every_other_tick_and_every_acquire_returns_at_once(self) -> None:
        for out in self.passes("hour2"):
            self.assertRegex(out, r"hour: 108001 presents at interval 2, the game presenting after its wait on the "
                                  r"acquire sync; every acquire came back at once, every wait ended 2 ticks apart "
                                  r"from tick 0 to tick 216000,")

    def test_a_game_that_presents_before_it_waits_on_the_sync_still_gets_one_frame_per_interval_1(self) -> None:
        for out in self.passes("hour1_pwait"):
            self.assertRegex(out, r"hour: 216001 presents at interval 1, the game presenting before its wait on the "
                                  r"acquire sync; every acquire came back at once, every wait ended 1 ticks apart "
                                  r"from tick 0 to tick 216000,")

    def test_a_game_that_presents_before_it_waits_on_the_sync_still_gets_one_frame_per_interval_2(self) -> None:
        for out in self.passes("hour2_pwait"):
            self.assertRegex(out, r"hour: 108001 presents at interval 2, the game presenting before its wait on the "
                                  r"acquire sync; every acquire came back at once, every wait ended 2 ticks apart "
                                  r"from tick 0 to tick 216000,")

    def test_a_present_before_its_wait_tells_the_back_end_the_due_tick_it_took(self) -> None:
        for out in self.passes("pace_seam"):
            self.assertIn("pace_seam: presents 2 and 3 were made at ticks 5 and 8 and told the back end the due ticks 7 "
                          "and 9, the tick of the acquire's sync; each is the due tick of its record", out)

    def test_an_acquire_wakes_a_wait_parked_on_a_sync_that_was_not_fenced_yet(self) -> None:
        for out in self.passes("wake"):
            self.assertIn("wake: a game thread parked on a sync not yet fenced was woken by the acquire that fenced it, "
                          "at tick 0", out)

    def test_a_wait_on_the_acquire_sync_gives_its_virtual_core_up_to_a_worker_on_the_same_core(self) -> None:
        for out in self.passes("block"):
            self.assertRegex(out, r"block: the worker counted \d+ while the game thread waited on the acquire sync")
            self.assertIn("block: acquire 2 came back at tick 0, its sync ended the wait at tick 1\n", out)

    def test_the_acquire_never_waits_and_its_sync_is_signalled_at_the_gate_tick_and_not_before(self) -> None:
        for out in self.passes("acquire_nowait"):
            self.assertIn("acquire: the second acquire came back at once at tick 5; its sync said TIMEOUT_EXPIRED "
                          "until tick 7 and ALREADY_SIGNALED at it; the present made before the wait is due at tick 7; the "
                          "acquire after it is signalled at tick 9",
                          out)

    def test_the_first_acquire_signals_its_sync_at_once_whatever_the_interval_and_the_hold(self) -> None:
        for out in self.passes("first"):
            self.assertIn("first: acquires with no previous present signalled their sync at once at tick 7 "
                          "(interval 3, hold 10); the acquire after the present did not", out)

    def test_a_game_that_passes_no_sync_is_not_held(self) -> None:
        for out in self.passes("null_sync"):
            self.assertIn("null_sync: acquires with no sync came back at once at tick 5 and their presents are due at "
                          "tick 5", out)

    def test_the_sync_is_signalled_only_when_the_tick_is_reached_and_the_fence_value_is_done(self) -> None:
        for out in self.passes("fence_and_tick"):
            self.assertIn("fence_and_tick: the sync was signalled only with the tick reached and the fence value done",
                          out)

    def test_a_queue_fence_on_a_sync_clears_the_tick_an_acquire_gave_it(self) -> None:
        for out in self.passes("fence_clears_tick"):
            self.assertIn("fence: QueueFenceSync on a sync an acquire gave a tick to cleared the tick", out)

    def test_wait_timeouts_run_on_the_clocks_own_time_and_zero_never_waits(self) -> None:
        for out in self.passes("timeout"):
            self.assertRegex(out, r"timeout: timeout 0 said TIMEOUT_EXPIRED at tick 0; 5 ms ended at tick [0-3] with "
                                  r"TIMEOUT_EXPIRED; 100 ms carried the wait to tick [4-9] \(CONDITION_SATISFIED\), "
                                  r"then ALREADY_SIGNALED")

    def test_a_late_present_takes_the_tick_it_arrives_in_and_the_next_sync_keeps_the_grid(self) -> None:
        # frame: acquired at tick, synced at tick, due tick (the model, written by hand in the test)
        want = [(0, 0, 0), (0, 1, 1), (1, 2, 3), (3, 4, 5), (5, 6, 6), (6, 7, 17), (17, 18, 18), (18, 19, 19)]
        for out in self.passes("late"):
            self.assertEqual([(f[1], f[2], f[3]) for f in self.frames(out)], want)

    def test_an_interval_change_applies_at_the_next_acquire_and_the_record_keeps_the_games_interval(self) -> None:
        want = [(0, 0, 1), (2, 2, 2), (4, 4, 2), (5, 5, 1), (6, 6, 3), (8, 8, 2), (9, 9, 1)]
        for out in self.passes("change"):
            self.assertEqual([(f[2], f[3], f[4]) for f in self.frames(out)], want)

    def test_an_interval_set_between_the_acquire_and_the_wait_applies_at_the_next_acquire(self) -> None:
        want = [(0, 0, 1), (1, 1, 3), (4, 4, 3), (7, 7, 1), (8, 8, 1)]
        for out in self.passes("change_mid"):
            self.assertEqual([(f[2], f[3], f[4]) for f in self.frames(out)], want)

    def test_the_hold_puts_syncs_two_ticks_apart_while_the_records_show_interval_1(self) -> None:
        for out in self.passes("hold_sync"):
            self.assertEqual([f[1:] for f in self.frames(out)],
                             [(max(0, 2 * k - 2), 2 * k, 2 * k, 1, 2, 0) for k in range(6)])

    def test_the_hold_raises_the_interval_and_never_lowers_it_under_the_default(self) -> None:
        for out in self.passes("hold_max_sync"):
            self.assertEqual([f[1:] for f in self.frames(out)],
                             [(0, 0, 0, 1, 2, 0), (0, 3, 3, 3, 2, 0), (3, 6, 6, 3, 2, 0), (6, 9, 9, 3, 2, 0)])

    def test_a_game_that_asks_for_interval_0_is_held_by_the_hold_under_the_default(self) -> None:
        for out in self.passes("hold_zero_sync"):
            self.assertEqual([f[1:] for f in self.frames(out)], [(0, 0, 0, 0, 2, 0), (0, 2, 2, 0, 2, 0), (2, 4, 4, 0, 2, 0)])

    # ---- the old placement, behind the setting
    def test_the_hold_puts_acquires_two_ticks_apart_under_the_old_placement(self) -> None:
        for out in self.passes("hold_call", wait_env="acquire"):
            self.assertEqual([f[1:] for f in self.frames(out)], [(2 * k, 2 * k, 2 * k, 1, 2, 1) for k in range(6)])

    def test_the_hold_raises_the_interval_and_never_lowers_it_under_the_old_placement(self) -> None:
        for out in self.passes("hold_max_call", wait_env="acquire"):
            self.assertEqual([f[1:] for f in self.frames(out)],
                             [(0, 0, 0, 1, 2, 1), (3, 3, 3, 3, 2, 1), (6, 6, 6, 3, 2, 1), (9, 9, 9, 3, 2, 1)])

    def test_a_game_that_asks_for_interval_0_is_held_by_the_hold_under_the_old_placement(self) -> None:
        for out in self.passes("hold_zero_call", wait_env="acquire"):
            self.assertEqual([f[1:] for f in self.frames(out)], [(0, 0, 0, 0, 2, 1), (2, 2, 2, 0, 2, 1), (4, 4, 4, 0, 2, 1)])

    # ---- the setting
    def test_the_placement_is_the_sync_unless_the_environment_or_a_call_says_otherwise(self) -> None:
        for env in (None, "", "sync"):
            for out in self.passes("setting_default", wait_env=env):
                # the acquire of frame k came back at the last present's tick, and the sync ended the wait at its own
                self.assertEqual([(f[1], f[2], f[6]) for f in self.frames(out)],
                                 [(0, 0, 0), (0, 2, 0), (2, 4, 0), (4, 6, 0)])

    def test_the_environment_asks_for_the_old_placement_and_the_records_say_so(self) -> None:
        for out in self.passes("setting_env", wait_env="acquire"):
            self.assertEqual([(f[1], f[2], f[6]) for f in self.frames(out)],
                             [(0, 0, 1), (2, 2, 1), (4, 4, 1), (6, 6, 1)])

    def test_the_stores_command_line_layer_asks_for_the_old_placement_and_its_hook_says_so(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "setting_store")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok setting_store\n", done.stdout)
                self.assertIn("nvn: AcquireTexture holds inside the call (hold_placement = acquire, --set)", done.stderr)
                self.assertEqual([f[6] for f in self.frames(done.stdout)], [1, 1, 1])

    def test_the_key_is_live_a_change_applies_at_the_next_acquire_both_ways_and_a_call_overrides_it(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "setting_live")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok setting_live\n", done.stdout)
                self.assertRegex(done.stderr, r"settings: hold_placement = acquire .*applied now; it was sync")
                self.assertIn("nvn: AcquireTexture holds inside the call (hold_placement = acquire,", done.stderr)
                self.assertIn("nvn: AcquireTexture returns at once and its sync is signalled at the due tick "
                              "(hold_placement = sync,", done.stderr)

    def test_a_call_to_the_setting_wins_over_the_environment_both_ways(self) -> None:
        for out in self.passes("setting_call_sync", wait_env="acquire"):
            self.assertEqual([f[6] for f in self.frames(out)], [0, 0, 0, 0])
        for out in self.passes("setting_call_call", wait_env="sync"):
            self.assertEqual([f[6] for f in self.frames(out)], [1, 1, 1, 1])

    def test_an_unknown_value_of_the_setting_is_refused_by_name_with_exit_code_2(self) -> None:
        for bad in ("fast", "call", "1"):
            for level in nvn.LEVELS:
                with self.subTest(level=level, value=bad):
                    done = self.run_case(level, "setting_bad", wait_env=bad)
                    self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
                    self.assertIn(f"oot2: OOT2_HOLD_PLACEMENT={bad}: hold_placement takes sync or acquire\n", done.stderr)
                    self.assertNotIn("ok setting_bad", done.stdout)

    def test_the_setting_flipped_between_two_acquires_leaves_the_first_sync_its_tick(self) -> None:
        for out in self.passes("flip"):
            self.assertIn("flip: the placement flipped between two acquires; the sync of the first kept its tick; "
                          "each record names its placement; a present of the old placement takes the tick in progress, "
                          "not a gate tick left by the sync", out)

    # ---- the pacing trace
    def test_the_trace_names_every_acquire_wait_and_present_with_the_placement_and_the_probes_it_was_given(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level), tempfile.TemporaryDirectory(prefix="oot2-nvnsynctr-",
                                                                       ignore_cleanup_errors=True) as d:
                trace, probes = Path(d) / "trace.txt", Path(d) / "probes.txt"
                done = self.run_case(level, "trace", extra={"OOT2_PACE_TRACE": str(trace), "OOT2_PROBE_FILE": str(probes)})
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok trace\n", done.stdout)
                self.assertIn("OOT2_PACE_TRACE: the pacing trace is on; this is not a default run", done.stderr)
                lines = trace.read_text().splitlines()
                self.assertEqual(lines[0], "#pace-trace OOT2_PACE_TRACE: not a default run")
                self.assertEqual(lines[1:4], ["#probe frames", "#probe ratio", "#probe nowhere"])
                events = [ln.split()[0] for ln in lines[4:]]
                # three frames: acquire, wait, present (the first wait is for nothing: also a wait)
                self.assertEqual(events, ["acq-enter", "acq-return", "sync-enter", "sync-return", "present"] * 3)
                acq = [ln for ln in lines if ln.startswith("acq-enter")]
                self.assertTrue(all(" interval 2 hold 0 placement 0 | frames=4242 ratio=2.5 nowhere=?" in ln for ln in acq), acq)
                pres = [ln for ln in lines if ln.startswith("present")]
                self.assertTrue(all("placement 0 | frames=4242 ratio=2.5 nowhere=?" in ln for ln in pres), pres)
                self.assertRegex(pres[2], r"^present \d+ tick 4 seq 3 due 4 interval 2 hold 0 placement 0 \|")
                gates = re.findall(r"^acq-return \d+ tick (\d+) gate (\d+) ", "\n".join(lines), re.M)
                self.assertEqual([(int(a), int(g)) for a, g in gates], [(0, 0), (0, 2), (2, 4)])
                syncs = re.findall(r"^sync-enter \d+ tick \d+ timeout (\d+) ready (\d+) fenced 2$", "\n".join(lines), re.M)
                self.assertEqual(syncs, [("18446744073709551615", "0"), ("18446744073709551615", "2"),
                                         ("18446744073709551615", "4")])

    def test_without_the_trace_variable_the_probe_file_is_never_read_and_no_trace_is_written(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level), tempfile.TemporaryDirectory(prefix="oot2-nvnsynctr-",
                                                                       ignore_cleanup_errors=True) as d:
                probes = Path(d) / "probes.txt"   # the case writes it with a line that does not parse
                done = self.run_case(level, "trace_off", extra={"OOT2_PROBE_FILE": str(probes)})
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok trace_off\n", done.stdout)
                self.assertNotIn("pacing trace", done.stderr)

    def test_a_probe_line_that_does_not_parse_stops_the_run_by_name(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level), tempfile.TemporaryDirectory(prefix="oot2-nvnsynctr-",
                                                                       ignore_cleanup_errors=True) as d:
                trace, probes = Path(d) / "trace.txt", Path(d) / "probes.txt"
                done = self.run_case(level, "trace_off", extra={"OOT2_PACE_TRACE": str(trace), "OOT2_PROBE_FILE": str(probes)})
                self.assertNotEqual(done.returncode, 0)
                self.assertIn("FATAL", done.stderr)
                self.assertIn('OOT2_PROBE_FILE: "this line does not parse"', done.stderr)

    # ---- the host's clock
    def test_waits_on_the_acquire_sync_end_on_the_host_clock_and_never_early(self) -> None:
        for out in self.passes("realtime"):
            self.assertRegex(out, r"realtime: 12 presents at interval 2 took \d+ ms of the host's clock")

    def test_a_timeout_on_the_host_clock_ends_a_wait_for_a_tick_still_far_off(self) -> None:
        for out in self.passes("realtime_timeout"):
            self.assertRegex(out, r"realtime: a 5 ms timeout ended a wait for a tick 4 ticks away after \d+ ms of the host's "
                                  r"clock, then a 10 s wait reached the tick after \d+ ms")

    def test_unknown_case_is_refused(self) -> None:
        done = self.run_case(nvn.LEVELS[0], "no_such_case")
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
