# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/meshcost (spike S4, the mesh pipeline cost spike): the sets of task, mesh and pixel programs read from the splitter's
pairing file, the interface gap, the job lists, the statistics, the harness's refusal of a wrong job file, and the harness on
the GPU.

Three groups. The first needs nothing but Python: every input is built here, no game data. The second builds
tools/meshcost/vkmeshcost.c with clang and the Vulkan SDK headers and feeds it job files it must refuse before it touches
Vulkan; it skips without them, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set. The third translates the mesh bindings work's
synthetic task, mesh and pixel programs, builds their libraries and links on the GPU with validation on, and runs the counter-run
(a layout without the push constants, which validation must report); it runs only when OOT2_GPU_TESTS is set, the translator is
built and tools/build_vkdraw.py --harness vkmeshcost has been run: no CI runner has a GPU.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "meshcost"))
sys.path.insert(0, str(ROOT / "tools" / "firstdraw"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import build_vkdraw  # noqa: E402
import mesh_cost as mc  # noqa: E402

import test_xlate_host as host  # noqa: E402

PAIRING_HEAD = "variation\tvertex\tpixel\ttask\tmesh\tproduct100\n"


def pairing_text(rows: list[tuple[str, str, str, str]]) -> str:
    """rows of (variation, pixel, task, mesh) in the splitter's columns; the vertex column is left empty."""
    return PAIRING_HEAD + "".join(f"{v}\t\t{p}\t{t}\t{m}\t0\n" for v, p, t, m in rows)


CORE = "/Shader/Model.Product.110.product.Nin_Ounce_NVN2.bfsha.zs#BNSH@0x1000#var"
PACKAGE = "/Shader/ApplicationPackage.Nin_Ounce_NVN2.release.sarc.zs!Model.Nin_Ounce_NVN2.bfsha#BNSH@0x20#var"
FOLIAGE = "/Shader/Foliage.Product.110.product.Nin_Ounce_NVN2.bfsha.zs#BNSH@0x30#var"


class SetsTest(unittest.TestCase):
    def rows(self):
        return [
            (CORE + "0", "p1", "t1", "m1"),
            (CORE + "1", "p2", "t1", "m1"),       # the same pair, another pixel stage: one pair, two triples
            (CORE + "2", "p1", "", "m2"),         # a mesh stage with no task is a pair with an empty task
            (FOLIAGE + "0", "p3", "t2", "m3"),
            (PACKAGE + "0", "p1", "t3", "m4"),
            (PACKAGE + "1", "p1", "t1", "m1"),    # a pair of the core scope seen again in the package: still core
            ("/Shader/Model.Product.110.x#BNSH@0#var9", "p9", "t1", ""),   # no mesh stage: not a row of this ticket
        ]

    def test_only_rows_with_a_mesh_stage_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairing.tsv"
            path.write_text(pairing_text(self.rows()), encoding="utf-8")
            read = mc.read_pairing(path)
        self.assertEqual(len(read), 6)
        self.assertEqual({r["mesh"] for r in read}, {"m1", "m2", "m3", "m4"})

    def test_scope_is_the_archive_of_the_variation(self):
        self.assertEqual(mc.scope_of(CORE + "0"), "core")
        self.assertEqual(mc.scope_of(FOLIAGE + "0"), "core")
        self.assertEqual(mc.scope_of(PACKAGE + "0"), "package")
        self.assertEqual(mc.scope_of("/Shader/Model.Nin_Ounce_NVN2.bfsha#BNSH@0#var0"), "package")

    def test_pairs_triples_scopes_and_the_unpaired(self):
        rows = [{"variation": v, "pixel": p, "task": t, "mesh": m, "product100": "0"} for v, p, t, m in self.rows() if m]
        stages = {"t1": "ts", "t2": "ts", "t3": "ts", "t9": "ts", "m1": "ms", "m2": "ms", "m3": "ms", "m4": "ms", "m9": "ms",
                  "p1": "ps"}
        sel = mc.select_sets(rows, stages)
        self.assertEqual(sel["counts"], {"pairs": 4, "pairs_core": 3, "pairs_tasked": 3, "triples": 5, "triples_core": 4,
                                         "pixel_modules": 3, "unpaired_task": 1, "unpaired_mesh": 1})
        by_pair = {(p["task"], p["mesh"]): p["scope"] for p in sel["pairs"]}
        self.assertEqual(by_pair, {("t1", "m1"): "core", ("", "m2"): "core", ("t2", "m3"): "core", ("t3", "m4"): "package"})
        self.assertEqual(sel["unpaired"], {"ts": ["t9"], "ms": ["m9"]})

    def test_a_stage_the_splitter_could_not_read_makes_no_pair_and_a_missing_pixel_no_triple(self):
        rows = [{"variation": CORE + "0", "pixel": "p1", "task": "t1", "mesh": "m1", "product100": "0"},
                {"variation": CORE + "1", "pixel": "", "task": "t1", "mesh": "m2", "product100": "0"},    # no pixel stage
                {"variation": CORE + "2", "pixel": "?", "task": "", "mesh": "m3", "product100": "0"},     # pixel unreadable
                {"variation": CORE + "3", "pixel": "p1", "task": "?", "mesh": "m4", "product100": "0"},   # task unreadable
                {"variation": CORE + "4", "pixel": "p1", "task": "t1", "mesh": "?", "product100": "0"}]   # mesh unreadable
        sel = mc.select_sets(rows, {})
        self.assertEqual([(p["task"], p["mesh"]) for p in sel["pairs"]], [("", "m3"), ("t1", "m1"), ("t1", "m2")])
        self.assertEqual([(t["mesh"], t["pixel"]) for t in sel["triples"]], [("m1", "p1")])
        self.assertEqual(mc.needed_keys(sel), ["m1", "m2", "m3", "p1", "t1"])

    def test_the_splitters_path_separator_gives_the_same_scope(self):
        arc = "/Shader/ApplicationPackage.Nin_Ounce_NVN2.release.sarc.zs/Model.Nin_Ounce_NVN2.bfsha#BNSH@0x20#var0"
        self.assertEqual(mc.scope_of(arc), "package")
        self.assertEqual(mc.scope_of("/Shader/Foliage.Product.110.product.Nin_Ounce_NVN2.bfsha.zs#BNSH@0x30#var0"), "core")

    def test_the_selection_names_every_program_it_needs(self):
        rows = [{"variation": CORE + "0", "pixel": "p1", "task": "t1", "mesh": "m1", "product100": "0"}]
        sel = mc.select_sets(rows, {"t1": "ts", "m1": "ms", "m7": "ms"})
        self.assertEqual(mc.needed_keys(sel), ["m1", "m7", "p1", "t1"])


class InterfaceGapTest(unittest.TestCase):
    def out(self, location, components, first=0):
        return [{"location": location, "component": first + i, "components": 1} for i in range(components)]

    def test_a_pixel_vec4_over_two_written_components_has_a_gap_of_two(self):
        mesh = self.out(5, 2) + self.out(1, 4)
        pixel = [{"location": 5, "component": 0, "components": 4}, {"location": 1, "component": 0, "components": 4}]
        self.assertEqual(mc.interface_gap(mesh, pixel), 2)

    def test_a_pixel_stage_the_mesh_stage_covers_has_none(self):
        mesh = self.out(1, 4) + self.out(2, 3)
        pixel = [{"location": 1, "component": 0, "components": 4}, {"location": 2, "component": 0, "components": 3}]
        self.assertEqual(mc.interface_gap(mesh, pixel), 0)

    def test_extra_mesh_outputs_are_not_a_gap(self):
        self.assertEqual(mc.interface_gap(self.out(1, 4) + self.out(9, 4), [{"location": 1, "component": 0, "components": 2}]), 0)

    def test_the_pad_set_is_the_sorted_pairs_the_runtime_would_add(self):
        pixel = [{"location": 5, "component": 0, "components": 4}, {"location": 2, "component": 2, "components": 2}]
        self.assertEqual(mc.missing_components(self.out(5, 2), pixel), ((2, 2), (2, 3), (5, 2), (5, 3)))
        self.assertEqual(mc.missing_components(self.out(5, 4) + self.out(2, 4), pixel), ())

    def test_a_location_the_mesh_stage_never_names_counts_every_component(self):
        self.assertEqual(mc.interface_gap([], [{"location": 3, "component": 1, "components": 3}]), 3)


def write_module(work: Path, key: str, stage: str, size: int, exts: str = "", raw: int = 0, targets: int = 1,
                 inputs=None, outputs=None) -> dict:
    """A fake translated module: SPIR-V magic and filler of `size` bytes, a description, and its row of translated.tsv."""
    spv = work / "spv"
    spv.mkdir(parents=True, exist_ok=True)
    (spv / f"{key}.spv").write_bytes(b"\x03\x02\x23\x07" + b"\0" * (size - 4))
    frag_out = sum(0xF << (4 * t) for t in range(targets))
    res = {"stage": stage, "graphics": {"frag_out_map": frag_out},
           "interface": {"inputs": inputs or [], "outputs": outputs or []}}
    (spv / f"{key}.res.json").write_text(json.dumps(res), encoding="utf-8")
    return {"key": key, "stage": stage, "status": "ok", "bytes": size, "raw": raw, "exts": exts, "valid": 1, "message": ""}


class ListsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        mods = [write_module(self.work, "t1", "ts", 1000, "SPV_EXT_mesh_shader"),
                write_module(self.work, "m1", "ms", 5000, "SPV_EXT_mesh_shader,SPV_KHR_physical_storage_buffer", raw=1),
                write_module(self.work, "m2", "ms", 3000, "SPV_EXT_mesh_shader"),
                write_module(self.work, "m3", "ms", 2000, "SPV_EXT_mesh_shader"),
                write_module(self.work, "p1", "ps", 4000, "SPV_EXT_descriptor_indexing", targets=2),
                write_module(self.work, "p2", "ps", 8000)]
        self.tr = {m["key"]: m for m in mods}
        self.sel = {"pairs": [{"task": "t1", "mesh": "m1", "scope": "core"}, {"task": "", "mesh": "m2", "scope": "package"}],
                    "triples": [{"task": "t1", "mesh": "m1", "pixel": "p1", "scope": "core"},
                                {"task": "", "mesh": "m2", "pixel": "p2", "scope": "package"}],
                    "unpaired": {"ts": [], "ms": ["m3"]}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_pre_items_are_the_pairs_then_the_unpaired_mesh_programs_largest_first(self):
        lists = mc.build_lists(self.sel, self.tr)
        self.assertEqual([(i["task"], i["mesh"], i["scope"], i["bytes"], i["kind"], i["raw"]) for i in lists["pre"]],
                         [("t1", "m1", "core", 6000, "tm", 1), ("", "m2", "package", 3000, "m", 0),
                          ("", "m3", "unpaired", 2000, "m", 0)])
        self.assertEqual([f["pixel"] for f in lists["frag"]], ["p2", "p1"])

    def test_a_pair_with_a_module_that_did_not_translate_is_left_out_and_counted(self):
        self.tr["t1"]["status"] = "failed"
        lists = mc.build_lists(self.sel, self.tr)
        self.assertEqual(lists["excluded_pairs"], 1)
        self.assertEqual([i["mesh"] for i in lists["pre"]], ["m2", "m3"])
        self.assertEqual([f["pixel"] for f in lists["frag"]], ["p2"])

    def test_an_invalid_module_is_left_out_too(self):
        self.tr["m2"]["valid"] = 0
        self.assertEqual([i["mesh"] for i in mc.build_lists(self.sel, self.tr)["pre"]], ["m1", "m3"])

    def test_the_job_lines_index_modules_once_and_name_the_device_extensions(self):
        lists = mc.build_lists(self.sel, self.tr)
        lists["triples"] = self.sel["triples"]
        lines = mc.job_lines(self.work, lists, self.tr, [0, 1], self.sel["triples"], lists["pre"][-1], "pre,frag,link")
        spv = self.work / "spv"
        self.assertEqual(lines, [
            "extension VK_EXT_mesh_shader",
            f"module t {spv / 't1.spv'}", f"module m {spv / 'm1.spv'}", f"module m {spv / 'm2.spv'}",
            f"module f {spv / 'p2.spv'}", f"module f {spv / 'p1.spv'}", f"module m {spv / 'm3.spv'}",
            "warmup - 5",
            "pre 0 1", "pre - 2",
            "frag 3", "frag 4",
            # the first triple is (t1, m1) = pre 0 with p1 = frag 1 at 2 targets, the second (m2) = pre 1 with p2 = frag 0 at 1
            "link 0 1 2", "link 1 0 1",
        ])

    def test_without_the_link_phase_no_link_is_written(self):
        lists = mc.build_lists(self.sel, self.tr)
        lines = mc.job_lines(self.work, lists, self.tr, [0], [], None, "pre")
        self.assertFalse([l for l in lines if l.startswith(("link", "frag", "warmup"))])
        self.assertEqual([l for l in lines if l.startswith("pre")], ["pre 0 1"])

    def test_the_target_count_comes_from_the_pixel_description(self):
        self.assertEqual(mc.targets_of(self.work, "p1"), 2)
        self.assertEqual(mc.targets_of(self.work, "p2"), 1)

    def test_the_gap_of_a_triple_reads_the_two_descriptions(self):
        write_module(self.work, "mx", "ms", 100, outputs=[{"location": 1, "component": c, "components": 1} for c in range(2)])
        write_module(self.work, "px", "ps", 100, inputs=[{"location": 1, "component": 0, "components": 4}])
        self.assertEqual(mc.triple_gaps(self.work, [{"mesh": "mx", "pixel": "px"}]), [2])


class ExtensionTest(unittest.TestCase):
    def test_the_mesh_extension_maps_and_core_ones_drop(self):
        self.assertEqual(mc.device_extensions(["SPV_EXT_mesh_shader", "SPV_KHR_physical_storage_buffer",
                                               "SPV_EXT_shader_atomic_float_add", "SPV_EXT_shader_atomic_float_add"]),
                         ["VK_EXT_mesh_shader", "VK_EXT_shader_atomic_float"])

    def test_an_unknown_extension_stops_the_preparation(self):
        with self.assertRaises(ValueError):
            mc.device_extensions(["SPV_VENDOR_new"])


class StatsTest(unittest.TestCase):
    def test_spread_takes_both_ends_and_all_when_few(self):
        self.assertEqual(mc.spread_ids(100, 5), [0, 25, 50, 74, 99])
        self.assertEqual(mc.spread_ids(3, 10), [0, 1, 2])
        self.assertEqual(mc.spread_ids(10, 1), [0])

    def test_ratio_is_per_item_over_the_items_both_runs_built(self):
        r = mc.ratio_stats({0: 20.0, 1: 15.0, 2: 99.0}, {0: 10.0, 1: 10.0, 3: 1.0})
        self.assertEqual(r["n"], 2)
        self.assertEqual((r["min"], r["max"]), (1.5, 2.0))
        self.assertEqual(r["sum_ratio"], 1.75)
        self.assertEqual(mc.ratio_stats({0: 1.0}, {1: 1.0}), {"n": 0})

    def test_stats_are_nearest_rank(self):
        s = mc.stats([float(v) for v in range(1, 101)])
        self.assertEqual((s["n"], s["p50"], s["p99"], s["max"], s["sum"]), (100, 50.0, 99.0, 100.0, 5050.0))
        self.assertEqual(mc.stats([]), {"n": 0})

    def test_a_work_folder_inside_the_repository_is_refused(self):
        self.assertEqual(mc.main(["report", "--work", str(ROOT / "build" / "s4")]), 2)


class ReportTest(unittest.TestCase):
    """The report from rows and summaries written by hand: the thread curve, the pin's overhead, the warm path, one miss."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        res = self.work / "results"
        res.mkdir()
        items = [("core", "tm", 4000, 1), ("core", "m", 3000, 0), ("package", "tm", 2000, 1), ("unpaired", "m", 1000, 0)]
        (self.work / "lists.json").write_text(json.dumps({
            "pre": [{"task": "t" if k == "tm" else "", "mesh": f"m{i}", "scope": s, "kind": k, "bytes": b, "raw": r}
                    for i, (s, k, b, r) in enumerate(items)], "frag": [], "triples": [], "excluded_pairs": 0}), encoding="utf-8")
        (self.work / "translated.tsv").write_text("key\tstage\tstatus\tbytes\traw\texts\tvalid\tmessage\n", encoding="utf-8")
        (self.work / "pre_items.tsv").write_text("index\tscope\tkind\tbytes\ttask_bytes\tmesh_bytes\traw\n" + "".join(
            f"{i}\t{s}\t{k}\t{b}\t0\t{b}\t{r}\n" for i, (s, k, b, r) in enumerate(items)), encoding="utf-8")
        self.summaries = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_rows(self, tag, times, wall_ms, threads, pin32, **summary):
        rows = "phase\twhat\tindex\tms\tworker\textra\n" + "".join(
            f"pre\tpre_raster_lib\t{i}\t{ms}\t0\tresult=0 cpu_ms={ms} module_ms=1 bytes=10\n" for i, ms in enumerate(times))
        (self.work / "results" / f"{tag}.tsv").write_text(rows, encoding="utf-8")
        self.summaries.append({"tag": tag, "threads": threads, "pin32": pin32, "pre_wall_ms": wall_ms, "failed": 0,
                               "status": "ok", "pre_items": len(times), "peak_working_set_mb": 500, **summary})
        (self.work / "results" / "summary.jsonl").write_text("".join(json.dumps(s) + "\n" for s in self.summaries), encoding="utf-8")

    def report(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = mc.report(self.work, (12.0, 32.0))
        self.assertEqual(code, 0)
        return json.loads((self.work / "report.json").read_text(encoding="utf-8"))

    def test_curve_pin_overhead_warm_and_miss(self):
        self.run_rows("t1-pin", [1000, 500, 300, 200], 2000, 1, 1)
        self.run_rows("t1-nopin", [800, 400, 300, 200], 1700, 1, 0)
        self.run_rows("t6-pin", [1500, 800, 500, 300], 600, 6, 1)
        self.run_rows("t12-pin", [2000, 1000, 600, 400], 4_000_000, 12, 1)
        self.run_rows("t12-cachew", [2000, 1000, 600, 400], 1000, 12, 1, pipeline_cache_out_bytes=1234)
        self.run_rows("t12-cacher", [1, 1, 1, 1], 5, 12, 1, pipeline_cache_in_bytes=1234)
        (self.work / "results" / "screen.tsv").write_text(
            "index\tscope\tkind\tbytes\tstatus\tms\tcpu_ms\tmodule_ms\tresult\twarmup_ms\tprocess_s\tpeak_ws_mb\tmin_available_mb\n"
            "0\tcore\ttm\t4000\tok\t900\t900\t1\t0\t5\t2\t610\t3000\n1\tcore\tm\t3000\tok\t400\t400\t1\t0\t5\t2\t120\t3000\n"
            "2\tpackage\ttm\t2000\tkilled_memory\t99999\t\t\t\t\t30\t7000\t900\n", encoding="utf-8")
        r = self.report()
        self.assertEqual(r["pin_overhead_1thread"]["p50"], 1.0, "nearest rank, as stats: the lower middle of [1, 1, 1.25, 1.25]")
        self.assertEqual(r["pin_overhead_1thread"]["sum_ratio"], round(2000 / 1700, 3))
        self.assertEqual(r["thread_curve_pin_on"]["t6-pin"], {"threads": 6, "wall_min": 0.01, "throughput": 3.33})
        self.assertTrue(r["pre_libraries_minutes_vs_budget"]["t1-pin"]["within_B"])
        self.assertFalse(r["pre_libraries_minutes_vs_budget"]["t12-pin"]["within_B"], "4,000 s is 66.7 minutes: over budget B")
        self.assertEqual(r["one_miss_ms"]["alone"]["max"], 900.0)
        self.assertEqual(r["screen"]["killed"], 1)
        self.assertEqual(r["one_miss_ms"]["warm_from_cache"]["max"], 1.0)
        self.assertEqual(r["warm_from_cache"]["cache_bytes"], 1234)
        self.assertEqual(r["runs"]["t1-pin"]["pre_by_scope"]["core"]["n"], 2)
        self.assertEqual(sorted(r["runs"]["t1-pin"]["pre_by_kind"]), ["m", "tm"])
        self.assertEqual(r["runs"]["t1-pin"]["pre_by_raw"]["1"]["sum"], 1300.0)
        self.assertEqual(r["items"]["by_scope"], {"core": 2, "package": 1, "unpaired": 1})

    def test_a_killed_run_is_not_a_measurement(self):
        self.run_rows("t1-pin", [1000, 500, 300, 200], 2000, 1, 1)
        self.run_rows("t6-pin", [1500, 800], 90, 6, 1, status="killed_memory")   # two of four items: the largest
        self.run_rows("t12-pin", [2000, 1000, 600, 400], 4000, 12, 1, failed=1)  # a refused library
        r = self.report()
        self.assertEqual(sorted(r["thread_curve_pin_on"]), ["t1-pin"])
        self.assertFalse(r["runs"]["t6-pin"]["complete"])
        self.assertFalse(r["runs"]["t12-pin"]["complete"])
        self.assertNotIn("while_threads_build", r["one_miss_ms"])

    def test_a_run_that_built_some_of_the_libraries_is_grouped_by_the_items_it_built(self):
        # the link run and smoke build a subset: row 1 of such a run is lists["pre"][2], not [1]
        (self.work / "jobs").mkdir()
        (self.work / "jobs" / "subset.txt").write_text(
            "validation off\nmodule t C:\\x\\spv\\t.spv\nmodule m C:\\x\\spv\\m0.spv\nmodule m C:\\x\\spv\\m2.spv\n"
            "module m C:\\x\\spv\\m3.spv\nwarmup - 3\npre 0 1\npre 0 2\npre - 3\n", encoding="utf-8")
        lists = json.loads((self.work / "lists.json").read_text(encoding="utf-8"))
        lists["pre"][0]["task"] = lists["pre"][2]["task"] = "t"
        self.assertEqual(mc.job_pre_ids(self.work, "subset", lists), [0, 2, 3])
        self.assertIsNone(mc.job_pre_ids(self.work, "no-such-job", lists))
        self.run_rows("subset", [10, 20, 30], 60, 1, 1, pre_items=3)
        r = self.report()
        by_scope = r["runs"]["subset"]["pre_by_scope"]
        self.assertEqual({k: v["sum"] for k, v in by_scope.items()}, {"core": 10.0, "package": 20.0, "unpaired": 30.0})

    def test_a_refused_library_is_not_in_the_statistics(self):
        self.run_rows("t1-pin", [1000, 500, 300, 200], 2000, 1, 1)
        path = self.work / "results" / "t1-pin.tsv"
        path.write_text(path.read_text(encoding="utf-8").replace("\t1000\t0\tresult=0", "\t1000\t0\tresult=-1000000000"),
                        encoding="utf-8")
        self.assertEqual(self.report()["runs"]["t1-pin"]["pre"]["n"], 3)


@unittest.skipUnless(os.name == "nt", "the harness is Windows only")
class HarnessJobFileTest(unittest.TestCase):
    """The harness refuses a wrong job file before it loads Vulkan, so this needs the build and no device."""

    @classmethod
    def setUpClass(cls) -> None:
        cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
        sdk = os.environ.get("VULKAN_SDK")
        have_sdk = bool(sdk) and (Path(sdk) / "Include" / "Volk" / "volk.c").is_file()
        if not cc or not have_sdk:
            if os.environ.get("OOT2_REQUIRE_CLANG") or os.environ.get("OOT2_REQUIRE_VULKAN"):
                raise RuntimeError("clang or the Vulkan SDK (VULKAN_SDK) is missing but the environment requires them")
            raise unittest.SkipTest("no clang or no Vulkan SDK")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-meshcost-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name) / "exe"
        if build_vkdraw.main(["--harness", "vkmeshcost", "--out", str(out), "--cc", cc]):
            raise RuntimeError("build_vkdraw.py --harness vkmeshcost failed")
        cls.exe = out / "vkmeshcost.exe"

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def refuse(self, text: str, expect: str) -> None:
        job = Path(self.tmp.name) / "job.txt"
        job.write_text(text, encoding="utf-8")
        done = subprocess.run([str(self.exe), str(job)], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn(expect, done.stderr)

    def test_an_unknown_directive(self):
        self.refuse("flavour sweet\n", "not understood")

    def test_a_module_kind_that_is_not_task_mesh_or_fragment(self):
        self.refuse("module x a.spv\n", "a module is t, m or f")

    def test_a_library_that_refers_past_the_module_list(self):
        self.refuse("module m a.spv\npre - 3\n", "refers past the module list")

    def test_a_mesh_stage_is_required(self):
        self.refuse("module t a.spv\npre 0 -\n", "refers past the module list")

    def test_a_link_to_a_target_count_outside_one_to_eight(self):
        self.refuse("module m a.spv\nmodule f b.spv\npre - 0\nfrag 1\nlink 0 0 9\n", "target count")

    def test_a_link_phase_without_the_libraries_it_links(self):
        self.refuse("module m a.spv\npre - 0\nparam phases pre,link\n", "link phase needs")

    def test_an_unknown_phase_and_an_unknown_format(self):
        self.refuse("module m a.spv\npre - 0\nparam phases pre,bake\n", "unknown phase")
        self.refuse("module m a.spv\npre - 0\nparam format rgb565\n", "unknown colour format")


@unittest.skipUnless(os.environ.get("OOT2_GPU_TESTS") and host.EXE is not None
                     and (ROOT / "build" / "vkmeshcost" / "vkmeshcost.exe").is_file(),
                     "set OOT2_GPU_TESTS, build the translator and tools/build_vkdraw.py --harness vkmeshcost to run on the GPU")
class GpuTest(unittest.TestCase):
    """The mesh bindings work's synthetic task, mesh and pixel programs through the harness: libraries and links build clean, and a layout
    without the push constants is caught by validation."""

    @classmethod
    def setUpClass(cls) -> None:
        import mesh_draw as md
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.work = Path(cls.tmp.name) / "work"
        modules, res = md.translate(Path(cls.tmp.name) / "synthetic", str(host.EXE))
        cls.tr = {}
        for name, key in (("ts", "syn-ts"), ("ms", "syn-ms"), ("ms_only", "syn-mso"), ("ps", "syn-ps")):
            spv = cls.work / "spv"
            spv.mkdir(parents=True, exist_ok=True)
            data = modules[name].read_bytes()
            (spv / f"{key}.spv").write_bytes(data)
            (spv / f"{key}.res.json").write_text(json.dumps(res[name]), encoding="utf-8")
            exts = ",".join(mc.cc.spirv_extensions(data))
            cls.tr[key] = {"key": key, "stage": res[name]["stage"], "status": "ok", "bytes": len(data), "raw": 0, "exts": exts,
                           "valid": 1, "message": ""}
        sel = {"pairs": [{"task": "syn-ts", "mesh": "syn-ms", "scope": "core"},
                         {"task": "", "mesh": "syn-mso", "scope": "core"}],
               "triples": [{"task": "syn-ts", "mesh": "syn-ms", "pixel": "syn-ps", "scope": "core"},
                           {"task": "", "mesh": "syn-mso", "pixel": "syn-ps", "scope": "core"}],
               "unpaired": {"ts": [], "ms": []}}
        cls.lists = mc.build_lists(sel, cls.tr)
        cls.lists["triples"] = sel["triples"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def run_job(self, tag: str, pin32: int, extra: dict | None = None) -> tuple[dict, list[dict]]:
        n = len(self.lists["pre"])
        body = mc.job_lines(self.work, self.lists, self.tr, list(range(n)), self.lists["triples"], self.lists["pre"][-1],
                            "pre,frag,link")
        params = {"phases": "pre,frag,link", "pin32": pin32, "heartbeat_s": 0, **(extra or {})}
        s = mc.run_job(self.work, tag, mc.header(2, "disabled", "on", params), body, 300, 4000, tag)
        return s, mc.cc.rows_of(self.work, tag)

    def test_libraries_and_links_build_clean_with_and_without_the_pin(self):
        for pin32 in (1, 0):
            with self.subTest(pin32=pin32):
                s, rows = self.run_job(f"clean-{pin32}", pin32)
                self.assertEqual(s["status"], "ok", s)
                self.assertEqual((s["failed"], s["validation_errors"], s["validation_warnings"]), (0, 0, 0), s)
                self.assertEqual(s["pin32"], pin32)
                kinds = sorted((r["phase"], r["what"], r.get("result", "")) for r in rows)
                self.assertEqual(kinds, sorted([("warmup", "pre_raster_lib", "0")] + [("pre", "pre_raster_lib", "0")] * 2
                                               + [("frag", "fragment_lib", "0")] + [("link", "fast_link", "0")] * 2
                                               + [("setup", "output_lib", "")] * 8))

    def test_a_layout_without_push_constants_is_caught(self):
        s, _ = self.run_job("counter-nopush", 1, {"counter": "nopush"})
        self.assertEqual(s["status"], "validation_errors", s)
        self.assertGreater(s["validation_errors"], 0)

    def test_the_pipeline_cache_written_by_one_run_serves_the_next(self):
        cache = self.work / "cache.vkcache"
        cache.unlink(missing_ok=True)
        first, cold = self.run_job("cache-write", 1, {"pipeline_cache": cache})
        self.assertGreater(first["pipeline_cache_out_bytes"], 0, first)
        second, warm = self.run_job("cache-read", 1, {"pipeline_cache": cache})
        self.assertGreater(second["pipeline_cache_in_bytes"], 0, second)
        self.assertEqual(second["failed"], 0)
        total = lambda rows: sum(float(r["ms"]) for r in rows if r["phase"] == "pre")  # noqa: E731
        self.assertLess(total(warm), total(cold), "libraries read back from the cache cost less than building them")


if __name__ == "__main__":
    unittest.main()
