# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the check step of `oot2 prepare` (tools/prepare/check.py) and its host probe (hostcheck.c).

The dump is a few megabytes of random bytes written by the test, the probe answers are dictionaries written here,
and the free disk space is patched. The probe itself is built and run on this machine: it needs clang (OOT2_CC, else
clang on PATH), and without one those tests skip unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.
Its Vulkan build needs a Vulkan SDK (VULKAN_SDK); without one it skips, unless OOT2_REQUIRE_VULKAN is set.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import oot2  # noqa: E402
from prepare import check  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, PrepareError, Reporter  # noqa: E402

GiB = 1 << 30
V3 = {flag: True for flag, _ in check.CPU_V3}


def find_compiler() -> str | None:
    found = shutil.which(os.environ.get("OOT2_CC", "clang"))
    if not found and os.environ.get("OOT2_REQUIRE_CLANG"):
        raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
    return found


def has_capstone() -> bool:
    """The check step requires capstone (check.lift_tools); without it the end-to-end check skips, unless
    OOT2_REQUIRE_CAPSTONE is set, as continuous integration sets it."""
    try:
        import capstone  # noqa: F401
    except ImportError as e:
        if os.environ.get("OOT2_REQUIRE_CAPSTONE"):
            raise RuntimeError(f"OOT2_REQUIRE_CAPSTONE is set but capstone cannot be imported: {e}") from None
        return False
    return True


def find_translator() -> Path | None:
    """The built translator (tools/build_xlate.py), which the check step requires; without it the end-to-end check
    skips, unless OOT2_REQUIRE_XLATE is set, as the Windows CI job sets it."""
    name = "oot2-xlate.exe" if os.name == "nt" else "oot2-xlate"
    hits = sorted((ROOT / "build" / "xlate" / "publish").rglob(name))
    if not hits and os.environ.get("OOT2_REQUIRE_XLATE"):
        raise RuntimeError(f"OOT2_REQUIRE_XLATE is set but {name} is not built under build/xlate")
    return hits[0] if hits else None


def has_zstd() -> bool:
    try:
        from compression import zstd  # noqa: F401
    except ImportError:
        return False
    return True


# A synthetic list in the shape the probe prints ("vulkan_items", from runtime/include/oot2/vk_required.h). The check
# judges by the probe's verdicts and keeps no list of its own, so any list tests it.
ITEMS = [["vulkan_1_3", "Vulkan 1.3"], ["mesh_shaders", "mesh and task shaders"],
         ["push_descriptors", "32 push descriptors"], ["memory_budget", "memory budget"],
         ["bc_textures", "BC texture compression"]]
GPU_H = ROOT / "runtime" / "include" / "oot2" / "gpu.h"


def device(**over):
    """A synthetic device, as the probe prints it, that has every item of ITEMS."""
    d = {
        "name": "Test GPU", "api": [1, 3, 280], "type": 2,
        "properties": {"driver_info": "1.0"},
        "heaps": [{"size": 8 * GiB, "device_local": True, "budget": 7 * GiB}, {"size": 16 * GiB, "device_local": False}],
        "items": {ident: True for ident, _ in ITEMS},
        "extensions": [],
    }
    d.update(over)
    return d


class CheckTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2-check-")
        self.root = Path(self.tmp.name)
        self.dump = self.root / "game.nup"
        self.dump.write_bytes(os.urandom(3 << 20))
        self.dest = self.root / "out"
        self.out = io.StringIO()

    def tearDown(self):
        self.tmp.cleanup()

    def ctx(self, dump=None, verify=False, **options):
        opts = argparse.Namespace(dump=dump, cc=None, vulkan_sdk=None, **options)
        return Context(self.dest, opts, Reporter(self.out, self.out), state=statefile.load(self.dest), verify=verify)


class DestAndDiskTest(CheckTestBase):
    def test_output_folder_must_be_empty_on_a_first_run(self):
        self.dest.mkdir()
        (self.dest / "notes.txt").write_text("mine")
        with self.assertRaisesRegex(PrepareError, "not empty and holds no prepare state .*notes.txt"):
            check.check_dest(self.ctx())

    def test_own_scratch_folder_and_a_state_file_are_allowed(self):
        (self.dest / check.TEMP).mkdir(parents=True)
        check.check_dest(self.ctx())
        statefile.save(self.dest, statefile.empty())
        (self.dest / "partial.bin").write_bytes(b"x")    # a resumed run's own outputs
        check.check_dest(self.ctx())

    def test_output_folder_must_not_be_a_file(self):
        self.dest.write_text("x")
        with self.assertRaisesRegex(PrepareError, "is a file"):
            check.check_dest(self.ctx())

    def test_disk_needs_38_gib_less_what_kept_steps_wrote(self):
        with mock.patch.object(check, "free_bytes", return_value=37 * GiB):
            with self.assertRaisesRegex(PrepareError, "37.0 GiB free and prepare needs about 38.0 GiB"):
                check.check_disk(self.ctx())
        state = statefile.empty()
        state["steps"]["extract"] = {"key": "k", "finished": "t", "seconds": 1, "bytes": 2 * GiB}
        self.dest.mkdir()
        statefile.save(self.dest, state)
        with mock.patch.object(check, "free_bytes", return_value=37 * GiB):
            check.check_disk(self.ctx())
        self.assertIn("36.0 GiB needed", self.out.getvalue())

    def test_free_bytes_of_a_folder_not_yet_made(self):
        self.assertGreater(check.free_bytes(self.root / "a" / "b" / "c"), 0)


class DumpTest(CheckTestBase):
    def identify(self, dump, verify=False):
        ctx = self.ctx(dump=dump, verify=verify)
        calls = []
        real = check.file_sha256

        def counting(path, progress=None):
            calls.append(path)
            return real(path, progress)

        with mock.patch.object(check, "file_sha256", counting):
            check.identify_dump(ctx, check.dump_path(ctx))
        return ctx, len(calls)

    def test_hash_is_recorded_and_reused_until_the_file_changes(self):
        ctx, hashed = self.identify(self.dump)
        self.assertEqual(hashed, 1)
        rec = statefile.load(self.dest)["inputs"]["dump"]
        self.assertEqual(rec["size"], 3 << 20)
        self.assertEqual(ctx.facts["dump"], rec)
        self.assertEqual(self.identify(self.dump)[1], 0)               # same size and date: not hashed again
        self.assertEqual(self.identify(None)[1], 0)                    # resumed run without --dump: the recorded one
        os.utime(self.dump, ns=(rec["mtime_ns"] + 10**9, rec["mtime_ns"] + 10**9))
        ctx, hashed = self.identify(self.dump)
        self.assertEqual(hashed, 1)                                    # date changed: hashed again
        self.assertEqual(ctx.facts["dump"]["sha256"], rec["sha256"])

    def test_verify_hashes_again_and_catches_a_silent_change(self):
        self.identify(self.dump)
        rec = statefile.load(self.dest)["inputs"]["dump"]
        data = bytearray(self.dump.read_bytes())
        data[100] ^= 0xFF
        self.dump.write_bytes(bytes(data))
        os.utime(self.dump, ns=(rec["mtime_ns"], rec["mtime_ns"]))   # same size, same date
        ctx, hashed = self.identify(self.dump)
        self.assertEqual((hashed, ctx.facts["dump"]["sha256"]), (0, rec["sha256"]))
        ctx, hashed = self.identify(self.dump, verify=True)
        self.assertEqual(hashed, 1)
        self.assertNotEqual(ctx.facts["dump"]["sha256"], rec["sha256"])
        self.assertIn("contents changed although its size and date did not", self.out.getvalue())

    def test_another_dump_is_named_as_such(self):
        self.identify(self.dump)
        other = self.root / "other.nup"
        other.write_bytes(b"other")
        ctx, hashed = self.identify(other)
        self.assertEqual(hashed, 1)
        self.assertIn("not the dump of the last run", self.out.getvalue())

    def test_missing_dump_is_refused(self):
        with self.assertRaisesRegex(PrepareError, "give the dump with --dump"):
            check.dump_path(self.ctx())
        with self.assertRaisesRegex(PrepareError, "is not a file"):
            check.dump_path(self.ctx(dump=self.root / "absent.nup"))


class HostAnswerTest(unittest.TestCase):
    def test_cpu_with_every_v3_feature_passes(self):
        self.assertEqual(check.cpu_missing(V3), [])

    def test_cpu_missing_features_are_named(self):
        self.assertEqual(check.cpu_missing({**V3, "avx2": False}), ["AVX2"])
        self.assertEqual(check.cpu_missing({**V3, "bmi2": False, "movbe": False}), ["BMI2", "MOVBE"])
        self.assertEqual(check.cpu_missing({**V3, "os_ymm": False}),
                         ["operating-system support for the YMM registers"])
        self.assertEqual(len(check.cpu_missing({})), len(check.CPU_V3))

    def test_full_device_has_every_item(self):
        lines, warnings = check.vulkan_report({"loader": True, "instance": 0, "devices": [device()]}, ITEMS)
        self.assertEqual(warnings, [])
        self.assertIn(f"has {len(ITEMS)} of the {len(ITEMS)} features", lines[-1])
        self.assertIn("device-local memory 8.00 GiB, budget 7.00 GiB", "\n".join(lines))

    def test_missing_items_are_reported_one_by_one_from_the_probes_verdicts(self):
        d = device()
        for ident in ("vulkan_1_3", "mesh_shaders", "push_descriptors"):
            d["items"][ident] = False
        del d["items"]["memory_budget"]        # an item the probe gave no verdict for counts as missing
        _, warnings = check.vulkan_report({"loader": True, "instance": 0, "devices": [d]}, ITEMS)
        lacks = {w.split(" lacks ")[1].split(" (")[0] for w in warnings}
        self.assertEqual(lacks, {"Vulkan 1.3", "mesh and task shaders", "32 push descriptors", "memory budget"})
        self.assertTrue(all(w.endswith("(architecture section 1; not required yet)") for w in warnings))

    def test_the_1_3_discrete_device_is_the_one_judged(self):
        old = device(name="Old iGPU", api=[1, 2, 0], type=1, items={})
        new = device(name="New dGPU")
        lines, warnings = check.vulkan_report({"loader": True, "instance": 0, "devices": [old, new]}, ITEMS)
        self.assertEqual(warnings, [])
        self.assertTrue(lines[-1].startswith("New dGPU: has"))

    def test_no_vulkan_is_a_warning_never_a_failure(self):
        for vk in (None, {"loader": False}, {"loader": True, "instance": -9}, {"loader": True, "instance": 0,
                                                                                "devices": []}):
            lines, warnings = check.vulkan_report(vk, ITEMS)
            self.assertEqual(len(warnings), 1, vk)
        _, warnings = check.vulkan_report({"loader": True, "instance": 0, "devices": [device()]}, [])
        self.assertEqual(warnings, ["the host probe printed no Vulkan device items "
                                    "(runtime/include/oot2/vk_required.h)."])

    def test_required_items_fail_the_check(self):
        d = device()
        d["items"]["mesh_shaders"] = False
        with self.assertRaisesRegex(PrepareError, r"Test GPU lacks mesh and task shaders, which the renderer requires"):
            check.vulkan_report({"loader": True, "instance": 0, "devices": [d]}, ITEMS, require=True)
        lines, warnings = check.vulkan_report({"loader": True, "instance": 0, "devices": [device()]}, ITEMS,
                                              require=True)
        self.assertEqual(warnings, [])
        for vk in (None, {"loader": False}, {"loader": True, "instance": 0, "devices": []}):
            with self.assertRaises(PrepareError):
                check.vulkan_report(vk, ITEMS, require=True)

    def test_the_check_requires_every_item_once_vulkan_is_the_default_back_end(self):
        found = re.findall(r'^#define GFX_DEFAULT_BACKEND "(\w+)"$', GPU_H.read_text(encoding="utf-8"), re.M)
        self.assertEqual(len(found), 1, "gpu.h must define GFX_DEFAULT_BACKEND once, as a string")
        self.assertEqual(check.REQUIRE_VULKAN_ITEMS, found[0] == "vulkan",
                         "the commit that makes the Vulkan back end the default sets REQUIRE_VULKAN_ITEMS too")


class ProbeTest(unittest.TestCase):
    """hostcheck.c builds without warnings, with and without the Vulkan headers, and prints the agreed JSON."""

    @classmethod
    def setUpClass(cls):
        cls.cc = find_compiler()
        if not cls.cc:
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-probe-")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def probe(self, name, include, extra):
        exe = check.build_probe(self.cc, Path(self.tmp.name) / name, include, ("-Wall", "-Wextra", "-Werror", *extra))
        return check.run_probe(exe)

    def test_cpu_only_build(self):
        host = self.probe("cpu", None, ("-DOOT2_PROBE_NO_VULKAN",))
        self.assertIsNone(host["vulkan"])
        self.assertEqual(set(host["cpu"]) - {"brand"}, {flag for flag, _ in check.CPU_V3})
        self.assertTrue(all(isinstance(v, bool) for k, v in host["cpu"].items() if k != "brand"))
        items = host["vulkan_items"]      # the list needs no Vulkan header
        self.assertGreaterEqual(len(items), 19)
        self.assertTrue(all(len(pair) == 2 and all(isinstance(s, str) and s for s in pair) for pair in items))
        self.assertEqual(len({ident for ident, _ in items}), len(items))

    def test_vulkan_build(self):
        include = check.vulkan_include(argparse.Namespace(options=argparse.Namespace(vulkan_sdk=None)))
        if include is None:
            if os.environ.get("OOT2_REQUIRE_VULKAN"):
                raise RuntimeError("OOT2_REQUIRE_VULKAN is set but VULKAN_SDK names no SDK with the headers and Volk")
            self.skipTest("no Vulkan SDK headers (VULKAN_SDK)")
        host = self.probe("vulkan", include, ())
        vk = host["vulkan"]
        self.assertIsInstance(vk, dict)
        self.assertIn("loader", vk)
        ids = [ident for ident, _ in host["vulkan_items"]]
        self.assertEqual(host["vulkan_items"], self.probe("list", None, ("-DOOT2_PROBE_NO_VULKAN",))["vulkan_items"])
        for d in vk.get("devices", []):
            self.assertEqual(len(d["api"]), 3)
            self.assertIsInstance(d["extensions"], list)
            self.assertEqual(list(d["items"]), ids)
            self.assertTrue(all(isinstance(v, bool) for v in d["items"].values()))
            # the pipelines step keys on the pipeline cache UUID: a driver update builds every pipeline again
            self.assertRegex(d["pipeline_cache_uuid"], r"^[0-9a-f]{32}$")
        check.vulkan_report(vk, host["vulkan_items"], require=False)   # whatever this machine has, the report reads it

    def test_broken_compiler_is_a_plain_failure(self):
        with self.assertRaisesRegex(PrepareError, "could not build the host probe"):
            check.build_probe(self.cc, Path(self.tmp.name) / "broken", None, ("-DOOT2_PROBE_NO_VULKAN",
                                                                               "-include", "no-such-header.h"))


class PrepareCheckCommandTest(CheckTestBase):
    """`oot2 prepare --check` end to end, on this machine's CPU and toolchain, with the disk patched."""

    def run_command(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.object(check, "free_bytes", return_value=64 * GiB):
            code = oot2.main(["prepare", *args])
        return code, out.getvalue() + err.getvalue()

    def test_check_records_the_dump_and_the_host(self):
        if not find_compiler():
            self.skipTest("no clang on PATH (set OOT2_CC)")
        if not has_capstone():
            self.skipTest("capstone is not installed (pip install capstone==5.0.7)")
        if check.vulkan_include(self.ctx()) is None and not os.environ.get("OOT2_REQUIRE_VULKAN"):
            self.skipTest("no Vulkan SDK with the headers and Volk (VULKAN_SDK), which the check step requires")
        translator = find_translator()
        if translator is None:
            self.skipTest("the translator is not built (py -3 tools/build_xlate.py), and the check step requires it")
        if not has_zstd():
            self.skipTest("no compression.zstd (Python 3.14), which the check step requires")
        with mock.patch.dict(os.environ, {"OOT2_XLATE": str(translator)}):
            code, text = self.run_command("--dump", str(self.dump), "--dest", str(self.dest), "--check")
            self.assertEqual(code, 0, text)
            self.assertIn("Vulkan headers 1.", text)
            self.assertIn("x86-64-v3", text)
            self.assertIn(f"with compression.zstd, translator {translator.resolve()} (protocol 2, identity ", text)
            self.assertIn("translate: would run (not done yet)", text)
            state = statefile.load(self.dest)
            self.assertEqual(state["inputs"]["dump"]["size"], 3 << 20)
            self.assertIn("cpu", state["host"])
            self.assertEqual(state["steps"], {})
            code, text = self.run_command("--dest", str(self.dest), "--check")     # resumed: no --dump, no hashing
            self.assertEqual(code, 0, text)
            self.assertIn("SHA-256 as recorded", text)

    def test_bad_command_line_exits_2(self):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            oot2.main(["prepare", "--check"])
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
