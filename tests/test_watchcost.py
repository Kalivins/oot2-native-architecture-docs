# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of spike S5's write-watch and host-import probe (tools/watchcost).

The driver's pure parts run everywhere: the nearest-rank percentile, the reading of `tasklist` and of Python command
lines for heavy processes, and the parse of the probe's output (synthetic lines written here). On Windows the probe
itself is built by tools/build_vkdraw.py (runtime/core, Volk, the Vulkan headers) and its write-watch cells run on
small heaps, each checking that the query reported exactly the pages written; its behaviour cases run where a Vulkan
device with VK_EXT_external_memory_host exists and must then pass, and must otherwise stop with exit 2 naming Vulkan
(continuous integration has no device). Needs clang (OOT2_CC, else PATH) and a Vulkan SDK (VULKAN_SDK); without them
the build test skips, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set, as continuous integration sets them.

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
sys.path.insert(0, str(ROOT / "tools" / "watchcost"))

import build_vkdraw  # noqa: E402
import watch_cost  # noqa: E402

PROBE_OUTPUT = """\
cell mode=watch commit_mib=64 scope=heap dirty=3 iters=2 avail_mib=9000
setup commit_mib=64 touch=1 ms=20.5 heap=0x1080000000 heap_region_mib=8192 alias=0x1280000000 alias_region_mib=65536
reset first_pages=16384
samples kind=query n=2 values=10.50,12.25
samples kind=write n=2 values=1.00,2.00
memory stage=imported avail_mib=7000 private_mib=1500.0 ws_mib=1400.5 imported_resident_mib=1024.0 other_resident_mib=0.0
verdict B1 PASS: an import over a 64 MiB window with one uncommitted 64 KiB chunk fails
verdict B2 FAIL: an import over committed pages succeeds
B1 import_ok=0 failed_at="import (vkAllocateMemory)" result=-2
B5 written=100 buffer=10 returned=10 still_marked=90
result mode=watch commit_mib=64 scope=heap dirty=3 iters=2 count_ok=1 min_avail_mib=8990 below_1gib=0
"""


class DriverTest(unittest.TestCase):
    def test_percentile_is_nearest_rank(self) -> None:
        v = [float(i) for i in range(1, 101)]
        self.assertEqual(watch_cost.percentile(v, 50), 50.0)
        self.assertEqual(watch_cost.percentile(v, 99), 99.0)
        self.assertEqual(watch_cost.percentile(v, 100), 100.0)
        self.assertEqual(watch_cost.percentile([3.0, 1.0, 2.0], 50), 2.0)
        self.assertEqual(watch_cost.percentile([5.0], 99), 5.0)
        with self.assertRaises(ValueError):
            watch_cost.percentile([], 50)

    def test_heavy_processes_are_read_from_tasklist(self) -> None:
        text = ('"System Idle Process","0","Services","0","8 K"\r\n"clang.exe","100","Console","1","90 000 K"\r\n'
                '"python.exe","101","Console","1","9 000 K"\r\n"NINJA.EXE","102","Console","1","5 000 K"\r\n'
                '"clang.exe","103","Console","1","90 000 K"\r\n"clangd.exe","104","Console","1","1 K"\r\n')
        self.assertEqual(watch_cost.heavy_from_tasklist(text), ["clang.exe", "ninja.exe"])
        self.assertEqual(watch_cost.heavy_from_tasklist(""), [])

    def test_shader_pools_are_found_by_command_line(self) -> None:
        lines = ['"C:\\Python314\\python.exe" -I tools/shaders/pool.py --jobs 12',
                 '"C:\\Python314\\python.exe" -I tools/watchcost/watch_cost.py run',
                 'C:\\x\\notepad.exe tools/shaders/pool.py']
        self.assertEqual(watch_cost.pool_pythons(lines), [lines[0]])

    def test_probe_output_is_parsed(self) -> None:
        p = watch_cost.parse_probe(PROBE_OUTPUT)
        self.assertEqual(p["samples"]["query"], [10.5, 12.25])
        self.assertEqual(p["result"]["count_ok"], "1")
        self.assertEqual(p["cell"]["scope"], "heap")
        self.assertEqual(p["memory"][0]["imported_resident_mib"], "1024.0")
        self.assertTrue(p["verdicts"]["B1"]["pass"])
        self.assertFalse(p["verdicts"]["B2"]["pass"])
        self.assertEqual(p["facts"]["B1"]["failed_at"], "import (vkAllocateMemory)")
        self.assertEqual(p["facts"]["B5"]["still_marked"], "90")

    def test_import_jobs_are_keyed_by_size_window_and_place(self) -> None:
        self.assertEqual(watch_cost.import_key(["import", "4096", "100", "16", "bounded"]), (4096, 16, "bounded"))
        self.assertEqual(watch_cost.import_key(["import", "1024", "100"]), (1024, 64, "cycle"))


@unittest.skipUnless(os.name == "nt", "the probe needs the Windows write watch")
class ProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
        sdk = os.environ.get("VULKAN_SDK")
        have_sdk = bool(sdk) and (Path(sdk) / "Include" / "Volk" / "volk.c").is_file()
        if not cc or not have_sdk:
            if os.environ.get("OOT2_REQUIRE_CLANG") or os.environ.get("OOT2_REQUIRE_VULKAN"):
                raise RuntimeError("clang or the Vulkan SDK (VULKAN_SDK) is missing but the environment requires them")
            raise unittest.SkipTest("no clang or no Vulkan SDK")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-watchcost-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        code = build_vkdraw.main(["--harness", "vkwatchcost", "--out", str(out), "--cc", cc])
        if code:
            raise RuntimeError(f"build_vkdraw.py --harness vkwatchcost exited {code}")
        cls.exe = out / "vkwatchcost.exe"

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def probe(self, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, VK_LOADER_LAYERS_DISABLE="~implicit~")
        return subprocess.run([str(self.exe), *args], capture_output=True, text=True, timeout=300, env=env)

    def test_write_watch_cells_report_exactly_the_pages_written(self) -> None:
        for scope, size, dirty in (("heap", "64", "50"), ("region", "64", "50"), ("pools", "512", "300"),
                                   ("alias", "64", "50")):
            with self.subTest(scope=scope):
                done = self.probe("watch", size, scope, dirty, "3")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                p = watch_cost.parse_probe(done.stdout)
                self.assertEqual(p["result"]["count_ok"], "1", done.stdout)
                self.assertEqual(len(p["samples"]["query"]), 3)
                if scope == "pools":
                    self.assertEqual(p["pools"]["n"], "4500")
                    self.assertEqual(p["pools"]["covered_mib"], "512")

    def test_usage_errors_exit_2(self) -> None:
        for args in (("watch", "64", "nowhere", "1", "1"), ("watch", "63", "heap", "1", "1"), ("behave", "nothing"),
                     ("import", "64", "1", "64", "bounded"), ("frobnicate",)):
            with self.subTest(args=args):
                done = self.probe(*args)
                self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
                self.assertIn("usage: vkwatchcost", done.stderr)

    def test_behaviours_pass_or_vulkan_is_missing(self) -> None:
        done = self.probe("behave")
        if done.returncode == 2:   # no loader or no device, as on the continuous-integration runner
            self.assertRegex(done.stderr, r"Vulkan|external_memory_host|bufferDeviceAddress")
            return
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        p = watch_cost.parse_probe(done.stdout)
        self.assertEqual(sorted(p["verdicts"]), ["B1", "B2", "B3"])
        self.assertTrue(all(v["pass"] for v in p["verdicts"].values()), done.stdout)


if __name__ == "__main__":
    unittest.main()
