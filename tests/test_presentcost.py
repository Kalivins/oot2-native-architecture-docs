# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the windowed present spike (tools/presentcost).

The driver's pure parts run everywhere: the path names, the reading of the harness's key=value lines, the conversion
of present-timing values to the performance counter, and the analysis, fed with synthetic frame rows built here from
a model of a display (a refresh grid, frames shown at the first refresh after they are ready). The model covers the
two cases the spike has to tell apart: a 60 Hz clock on a 100 Hz display (two thirds of the intervals at 20 ms, one
third at 10, no tick shown twice or never) and a producer that skips one tick in ten (the counter-run: one repeat and
one never-shown tick per skip), plus a frame the engine reports as never shown and a late frame on a 60 Hz display.
On Windows the harness is built by tools/build_vkdraw.py and its argument checks run; a real run opens a window on the
primary monitor for two seconds, so it only runs with OOT2_PRESENT_WINDOW=1. Needs clang (OOT2_CC, else PATH) and a
Vulkan SDK (VULKAN_SDK); without them the build test skips, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set,
as continuous integration sets them.

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
sys.path.insert(0, str(ROOT / "tools" / "presentcost"))

import build_vkdraw  # noqa: E402
import present_cost as pc  # noqa: E402

QPF = 10_000_000
T0 = 1_000_000_000


def stdout_text(calib: list[tuple[int, int, int]], refresh_ns: int = 10_000_000) -> str:
    """Harness lines: counter frequency, TSC rate, timing properties and calibration pairs (stage, value, qpc)."""
    lines = [f"start qpf={QPF} frames=0 skip=0 tick_hz=60", "tsc hz=3700000000",
             f"timing_props when=start result=0 counter=1 refresh_duration_ns={refresh_ns} "
             f"refresh_interval_ns={refresh_ns} qpc=1",
             'env layers_disable="~implicit~"', "layers loaded=0",
             "vrr when=start init=0 primary=0 display_id=0x1 result=0 enabled=0 possible=0 requested=0 indicator=0 "
             "in_vrr_mode=0"]
    for stage, value, q in calib:
        lines.append(f"calib when=first kind=1000208000 id=1000208000 stage={stage} result=0 value={value} qpc={q} "
                     "max_deviation_ns=100")
    return "\n".join(lines) + "\n"


def model(ticks: int, refresh_ms: float, skip: int = 0, ready_ms: float = 2.0, late: dict[int, float] | None = None,
          dropped: set[int] | None = None) -> tuple[str, list[dict], list[int]]:
    """Frame rows, vertical blanks and harness text for a display that shows each frame at the first refresh after
    due + ready_ms (+ late[k] for chosen ticks). Present timing reports first-pixel-out in nanoseconds of the counter
    (value = qpc * 100), queue-operations-end at due + 0.5 ms in a clock offset by 7 s (value in ns)."""
    period = round(refresh_ms * QPF / 1000)
    phase = 3_333
    vblanks = [T0 + phase + i * period for i in range(int((ticks / 60 + 1) * 1000 / refresh_ms) + 4)]
    rows = []
    for k in range(ticks):
        due = T0 + (k * QPF + 59) // 60
        state = pc.ST_SKIPPED if skip and k % skip == skip - 1 else pc.ST_PRESENTED
        row = {"tick": k, "state": state, "due": due, "produced": due + 10, "wake": due + 50, "acquired": due + 300,
               "present_call": due + 2_000, "present_ret": due + 2_500, "wait_ret": 0, "lock_wait": 0,
               "cycles": 370_000 if state == pc.ST_PRESENTED else 0, "pt_have": 0, "pt_complete": 0, "pt_domain": 0,
               "pt_qoe": 0, "pt_fpo": 0, "acq_res": 0, "pres_res": 0, "wait_res": 0, "image": k % 3}
        if state == pc.ST_PRESENTED:
            ready = due + round((ready_ms + (late or {}).get(k, 0.0)) * QPF / 1000)
            shown = next(v for v in vblanks if v >= ready)
            row.update(pt_have=3, pt_complete=1, pt_qoe=(due + 5_000) * 100 + 7_000_000_000,
                       pt_fpo=0 if dropped and k in dropped else shown * 100, wait_ret=shown + 3_000)
            if dropped and k in dropped:
                row["wait_ret"] = 0
                row["wait_res"] = 2
        rows.append(row)
    text = stdout_text([(pc.STAGE_QOE, 7_000_000_000 + 100 * T0, T0), (pc.STAGE_FPO, 100 * T0, T0)],
                       round(refresh_ms * 1e6))
    return text, rows, vblanks


class PathTest(unittest.TestCase):
    def test_paths_name_their_arguments(self) -> None:
        self.assertEqual(pc.path_args("composed-fifo"), (["--window", "composed", "--mode", "fifo"], False))
        self.assertEqual(pc.path_args("borderless-fifo-rel16667-nearest"),
                         (["--window", "borderless", "--mode", "fifo", "--relative", "16667000", "--nearest"], False))
        self.assertEqual(pc.path_args("composed-mailbox-layers")[1], True)
        self.assertEqual(pc.path_args("composed-fifo-occluded")[0][-1], "--occlude")
        self.assertEqual(pc.path_args("borderless-latest-notiming-skip10-img2-nolock")[0],
                         ["--window", "borderless", "--mode", "latest", "--timing", "off", "--skip", "10",
                          "--images", "2", "--wait-lock", "off"])

    def test_bad_paths_are_refused(self) -> None:
        for name in ("fifo", "windowed-fifo", "composed-vsync", "composed-fifo-rel", "composed-fifo-nearest",
                     "composed-fifo-notiming-rel1000", "composed-fifo-skip1", "composed-fifo-bogus"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pc.path_args(name)


class ReadingTest(unittest.TestCase):
    def test_stdout_lines_are_read(self) -> None:
        text = stdout_text([(1, 5, 6)]) + 'layer_dll path="C:\\x y\\l.dll"\nresult frames=3 presented=3\nnoise\n'
        m = pc.parse_stdout(text)
        self.assertEqual(m["start"]["qpf"], str(QPF))
        self.assertEqual(m["layer_dll"][0]["path"], "C:\\x y\\l.dll")
        self.assertEqual(m["calib"][0]["value"], "5")
        self.assertEqual(m["vrr"][0]["possible"], "0")
        self.assertEqual(m["result"]["presented"], "3")
        self.assertEqual(m["other"], ["noise"])

    def test_calibration_uses_the_measured_slope(self) -> None:
        # a slope of 0.02 counts per value unit, twice what nanoseconds would give at 10 MHz
        calib = [{"kind": "1", "id": "9", "stage": "4", "result": "0", "value": "1000", "qpc": "50"},
                 {"kind": "1", "id": "9", "stage": "4", "result": "0", "value": "3000", "qpc": "90"}]
        conv = pc.converters(calib, QPF)
        self.assertAlmostEqual(pc.to_qpc(conv, 0, pc.STAGE_FPO, 2000), 70.0)
        # the other stage of the same domain falls back to the same pairs; a value of 0 is "never shown"
        self.assertAlmostEqual(pc.to_qpc(conv, 0, pc.STAGE_QOE, 2000), 70.0)
        self.assertIsNone(pc.to_qpc(conv, 0, pc.STAGE_FPO, 0))
        self.assertIsNone(pc.to_qpc(conv, 1, pc.STAGE_FPO, 2000))

    def test_one_pair_reads_nanoseconds(self) -> None:
        conv = pc.converters([{"kind": "3", "id": "0", "stage": "1", "result": "0", "value": "0", "qpc": "0"}], QPF)
        self.assertAlmostEqual(pc.to_qpc(conv, 0, pc.STAGE_QOE, 1_000_000_000), QPF)

    def test_domains_are_numbered_in_order_of_appearance(self) -> None:
        # appearance order, not the order of the kinds: the harness numbers domains as it first meets them
        calib = [{"kind": "8", "id": "1", "stage": "1", "result": "0", "value": "0", "qpc": "0"},
                 {"kind": "7", "id": "2", "stage": "1", "result": "0", "value": "0", "qpc": "100"},
                 {"kind": "8", "id": "1", "stage": "1", "result": "0", "value": "1000", "qpc": "10"}]
        conv = pc.converters(calib, QPF)
        self.assertAlmostEqual(pc.to_qpc(conv, 0, pc.STAGE_QOE, 500), 5.0)
        self.assertAlmostEqual(pc.to_qpc(conv, 1, pc.STAGE_QOE, 0 + 1), 100.0 + QPF / 1e9)


class AnalysisTest(unittest.TestCase):
    def run_model(self, *args, **kw) -> dict:
        text, rows, vblanks = model(*args, **kw)
        return pc.analyse(text, rows, vblanks)

    def test_sixty_on_a_hundred_hertz_alternates_twenty_and_ten(self) -> None:
        a = self.run_model(3060, 10.0)
        f = a["fpo"]
        self.assertEqual(a["ticks"], 3000)
        self.assertEqual(f["frames"], 3000)
        self.assertAlmostEqual(f["interval_shares"]["20.0"], 66.7, delta=0.1)
        self.assertAlmostEqual(f["interval_shares"]["10.0"], 33.3, delta=0.1)
        self.assertEqual((f["ticks"]["repeats"], f["ticks"]["never_shown"], f["ticks"]["late"]), (0, 0, 0))
        self.assertAlmostEqual(a["refresh_ms"], 10.0, places=3)
        self.assertEqual(f["to_vblank_ms"]["max"], 0.0)       # first-pixel-out sits on the grid
        self.assertAlmostEqual(a["present_thread_us"]["p50"], 100.0, places=3)
        self.assertEqual(a["wait"]["ticks"]["repeats"], 0)
        self.assertAlmostEqual(a["wait_after_fpo_ms"]["p50"], 0.3, places=3)
        # queue-operations-end converts through its own offset domain
        self.assertAlmostEqual(a["qoe"]["from_due_ms"]["p50"], 0.5, places=2)

    def test_counter_run_skipping_one_tick_in_ten_is_caught_by_both_instruments(self) -> None:
        a = self.run_model(3060, 10.0, skip=10)
        self.assertEqual(a["skipped"], 300)
        for inst in ("fpo", "wait"):
            with self.subTest(instrument=inst):
                t = a[inst]["ticks"]
                self.assertEqual(t["repeats"], 300)
                self.assertEqual(t["never_shown"], 300)
        # each skip leaves the previous frame on screen for one whole tick more (30 or 40 ms on a 100 Hz grid)
        self.assertGreater(a["fpo"]["interval_shares"].get("30.0", 0) + a["fpo"]["interval_shares"].get("40.0", 0),
                           9.0)

    def test_a_frame_the_engine_never_showed_is_counted(self) -> None:
        a = self.run_model(1060, 10.0, dropped={500, 700})
        self.assertEqual(a["engine_dropped"], 2)
        self.assertEqual(a["fpo"]["frames"], 998)
        self.assertEqual(a["fpo"]["ticks"]["never_shown"], 2)
        self.assertEqual(a["fpo"]["ticks"]["repeats"], 2)

    def test_a_late_frame_on_a_sixty_hertz_display_repeats_one_tick(self) -> None:
        a = self.run_model(1060, 1000.0 / 60, late={400: 17.0})
        t = a["fpo"]["ticks"]
        self.assertEqual(t["repeats"], 1)
        self.assertEqual(t["late"], 1)
        self.assertEqual(t["never_shown"], 1)

    def test_shares_round_to_whole_refreshes(self) -> None:
        self.assertEqual(pc.interval_shares([9.9, 10.2, 20.1, 31.0], 10.0), {"10.0": 50.0, "20.0": 25.0, "30.0": 25.0})

    def test_report_names_skipped_paths(self) -> None:
        a = self.run_model(200, 10.0)
        lines = pc.report_lines([{"path": "composed-fifo", "machine": {"cpu_busy_pct": 3.0, "avail_mib": 9000},
                                  "analysis": a},
                                 {"path": "borderless-fifo", "skipped": "heavy processes: clang.exe",
                                  "machine": {}}])
        self.assertIn("| composed-fifo (fpo) | 140 |", lines[2])
        self.assertIn("heavy processes: clang.exe", lines[3])


@unittest.skipUnless(os.name == "nt", "the harness opens a Win32 window")
class HarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
        sdk = os.environ.get("VULKAN_SDK")
        have_sdk = bool(sdk) and (Path(sdk) / "Include" / "Volk" / "volk.c").is_file()
        if not cc or not have_sdk:
            if os.environ.get("OOT2_REQUIRE_CLANG") or os.environ.get("OOT2_REQUIRE_VULKAN"):
                raise RuntimeError("clang or the Vulkan SDK (VULKAN_SDK) is missing but the environment requires them")
            raise unittest.SkipTest("no clang or no Vulkan SDK")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-present-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        code = build_vkdraw.main(["--harness", "vkpresent", "--out", str(out), "--cc", cc])
        if code:
            raise RuntimeError(f"build_vkdraw.py --harness vkpresent exited {code}")
        cls.exe = out / "vkpresent.exe"

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def harness(self, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, VK_LOADER_LAYERS_DISABLE="~implicit~")
        return subprocess.run([str(self.exe), *args], capture_output=True, text=True, timeout=120, env=env)

    def test_usage_errors_exit_1(self) -> None:
        for args in ((), ("info",), ("run", "--window", "composed", "--mode", "fifo"),
                     ("run", "--window", "tiny", "--mode", "fifo", "--out", "x"),
                     ("run", "--window", "composed", "--mode", "vsync", "--out", "x"),
                     ("run", "--window", "composed", "--mode", "fifo", "--out", "x", "--skip", "1"),
                     ("run", "--window", "composed", "--mode", "fifo", "--out", "x", "--timing", "off",
                      "--relative", "1000"),
                     ("run", "--window", "composed", "--mode", "fifo", "--out", "x", "--nearest")):
            with self.subTest(args=args):
                done = self.harness(*args)
                self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
                self.assertIn("usage: vkpresent", done.stderr)

    @unittest.skipUnless(os.environ.get("OOT2_PRESENT_WINDOW") == "1", "opens a window: set OOT2_PRESENT_WINDOW=1")
    def test_a_short_run_reports_every_frame(self) -> None:
        prefix = Path(self.tmp.name) / "short"
        done = self.harness("run", "--window", "composed", "--mode", "fifo", "--frames", "180", "--out", str(prefix))
        if done.returncode == 2:   # no device or no surface, as on a runner without a GPU
            self.assertRegex(done.stderr, r"Vulkan|device|surface|monitor")
            return
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        a = pc.analyse(done.stdout, pc.read_frames(Path(f"{prefix}.frames.tsv")),
                       pc.read_vblanks(Path(f"{prefix}.vblanks.tsv")))
        self.assertEqual(a["presented"], 120)
        self.assertEqual(a["fpo"]["frames"], 120)
        self.assertEqual(a["fpo"]["ticks"]["never_shown"], 0)


if __name__ == "__main__":
    unittest.main()
