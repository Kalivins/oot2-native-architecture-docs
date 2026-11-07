# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the mesh draw harness of the mesh bindings work (tools/firstdraw/mesh_programs.py, mesh_draw.py and vkmesh.c).

Three groups. The first needs nothing but Python: the instruction words, the float bit patterns the programs build positions from,
the CPU reference, the comparison and its counter-runs, and the draw file written from descriptions. The second translates the
synthetic programs and skips where the translator is not built. The third draws them on the GPU and runs only when the harness is built
and OOT2_GPU_TESTS is set: no CI runner has a GPU.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools" / "firstdraw"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import mesh_draw as md  # noqa: E402
import mesh_programs as mp  # noqa: E402
import sass_eval as se  # noqa: E402

import test_xlate_host as host  # noqa: E402
import test_xlate_mesh as meshtests  # noqa: E402
import test_xlate_uni as unitests  # noqa: E402


def bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def words(ins: bytes) -> tuple[int, int]:
    return struct.unpack("<QQ", ins)


class InstructionWordsTest(unittest.TestCase):
    def test_the_words_are_those_the_translator_tests_assemble(self):
        self.assertEqual(mp.EXIT, host.EXIT)
        self.assertEqual(mp.ampere(0x927, ra=255, rb=4, low_bits=0x80 << 40, high_bits=0xF8 << 72),
                         host.ampere(0x927, ra=255, rb=4, low_bits=0x80 << 40, high_bits=host.bit(72, 0xF8)))
        self.assertEqual(mp.scheduled(mp.EXIT), unitests.scheduled(host.EXIT))
        for mode, offset, reg in ((0xF8, 0x80, 4), (0x88, 3, 5), (0x80, 4, 255), (0xB8, 0x10, 2)):
            self.assertEqual(mp.isbewr(mode, offset, reg), meshtests.isbewr(mode, offset, reg))
        for mode, rd, offset in ((0x78, 2, 0), (0x38, 4, 0x10)):
            self.assertEqual(mp.isberd(mode, rd, offset), meshtests.isberd(mode, rd, offset))
        self.assertEqual(mp.mov_cbuf(2, 3, 0x10), unitests.mov_cbuf(2, 3, 0x10))

    def test_a_move_of_an_immediate_and_a_multiply_add(self):
        lo, hi = words(mp.mov_imm(6, mp.HALF))
        self.assertEqual((lo & 0xFFF, (lo >> 16) & 0xFF, lo >> 32), (0x802, 6, mp.HALF))
        self.assertEqual((lo & 0xFFF) >> 9, 4, "operand form 4: the immediate is the second source")
        lo, hi = words(mp.imad_imm(7, 3, 1 << 19, 6))
        self.assertEqual((lo & 0xFFF, (lo >> 16) & 0xFF, (lo >> 24) & 0xFF, lo >> 32, hi & 0xFF), (0x824, 7, 3, 1 << 19, 6))
        self.assertEqual((hi >> 17) & 7, 7, "the predicate output is the true predicate")
        self.assertEqual((hi >> 40) & 0xFFF, 0xFE2, "the scheduling bits that make each result reach the next reader")

    def test_ipa_names_the_attribute_word(self):
        for c in range(4):
            lo, hi = words(mp.ipa(c, 0x80 + 4 * c))
            self.assertEqual((lo & 0xFFF, (lo >> 16) & 0xFF, hi & 0xFF), (0x326, c, 0x20 + c))
            self.assertEqual((hi >> 12) & 0xF, 0, "pass, at the default sample location")

    def test_the_pixel_header_reads_back_as_a_pixel_header(self):
        program = mp.pixel_program()
        sph = se.parse_sph(program[:0x80])
        self.assertEqual((sph["sph_type"], sph["shader_type"]), (2, 5))
        self.assertEqual(sph["imap_generic"][0], (2, 2, 2, 2))
        self.assertEqual(sph["imap_generic"][1], (0, 0, 0, 0))
        self.assertEqual(sph["omap_target"][0], 0xF)
        self.assertEqual(len(program), 0x80 + 5 * 16)


class FloatBitsTest(unittest.TestCase):
    def test_a_step_of_the_mantissa_is_a_step_of_a_cell(self):
        for n in range(mp.MAX_CELLS):
            self.assertEqual(bits_to_float(mp.HALF + (n << mp.CELL_SHIFT)), 0.5 + n / 32)
        self.assertEqual(bits_to_float(mp.HALF + mp.LEG) - 0.5, 5 / 256)
        self.assertEqual(bits_to_float(mp.HALF), 0.5)
        self.assertEqual(bits_to_float(mp.ONE), 1.0)

    def test_no_position_leaves_its_binade(self):
        top = bits_to_float(mp.HALF + ((mp.MAX_CELLS - 1) << mp.CELL_SHIFT) + mp.LEG)
        self.assertLess(top, 1.0)
        self.assertEqual(bits_to_float(mp.HALF + ((mp.MAX_CELLS - 1) << mp.CELL_SHIFT) + mp.LEG),
                         0.5 + (mp.MAX_CELLS - 1) / 32 + 5 / 256, "the sum of mantissa steps is exact in the binade")

    def test_the_mesh_programs_end_in_one_exit_and_store_every_line(self):
        for code in (mp.mesh_program(), mp.mesh_only_program(mp.ROW)):
            self.assertEqual(code[-1], mp.EXIT)
            self.assertEqual(code.count(mp.EXIT), 1)
            stores = [ins for ins in code if words(ins)[0] & 0xFFF == 0x927]
            attribute = [ins for ins in stores if (words(ins)[1] >> 8) & 0xFF == 0xF8]
            self.assertEqual(len(attribute), 24, "eight lines of three vertices")
            self.assertEqual(len({words(ins)[0] >> 40 for ins in attribute}), 24, "every word of every line once")

    def test_the_jobs_carry_what_the_translator_needs(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = mp.jobs(tmp)
            self.assertEqual(sorted(jobs), ["ms", "ms_only", "ps", "ts"])
            self.assertTrue(all(Path(j["code"]).is_file() for j in jobs.values()))
            self.assertEqual(jobs["ts"]["stage"], "ts")
            self.assertEqual(jobs["ms"]["mesh"]["has_task"], True)
            self.assertEqual(jobs["ms_only"]["mesh"]["has_task"], False)
            self.assertEqual(jobs["ms"]["outputs"]["vertex"], mp.VERTEX_MAP)
            self.assertEqual((jobs["ms"]["mesh"]["max_vertices"], jobs["ms"]["mesh"]["max_primitives"]), (3, 1))
            self.assertEqual(len({j["key"] for j in jobs.values()}), 4)
            self.assertEqual(Path(jobs["ps"]["code"]).read_bytes(), mp.pixel_program())


class ReferenceTest(unittest.TestCase):
    def test_every_triangle_covers_three_pixels_and_none_is_on_an_edge(self):
        for n in range(mp.MAX_CELLS):
            for j in range(mp.MAX_CELLS):
                kind, rgba = md.expected_image([(n, j)])
                self.assertEqual(kind.count(md.COLOR), 3, (n, j))
                self.assertEqual(kind.count(md.EDGE), 0, (n, j))

    def test_the_covered_pixels_are_the_corner_of_the_cell(self):
        kind, rgba = md.expected_image([(3, 1)])
        x0, y0 = 192 + 4 * 3, 192 + 4 * 1
        got = sorted((k % mp.IMAGE, k // mp.IMAGE) for k in range(len(kind)) if kind[k] == md.COLOR)
        self.assertEqual(got, [(x0, y0), (x0, y0 + 1), (x0 + 1, y0)])
        colour = rgba[y0 * mp.IMAGE + x0]
        self.assertEqual(colour, (0.5 + 3 / 32, 0.5 + 1 / 32, 0.5, 1.0))
        self.assertEqual(md.cell_of(colour[0], colour[1]), (3, 1))

    def test_what_a_scenario_launches(self):
        by = {s.name: s for s in md.scenarios()}
        self.assertEqual(md.workgroups(by["direct-first3"]), [(n, j) for n in (3, 4, 5, 6) for j in (0, 1, 2)])
        self.assertEqual(md.workgroups(by["indirect-first3"]), md.workgroups(by["direct-first3"]))
        self.assertEqual(md.workgroups(by["direct-first0"]), [(n, j) for n in (0, 1, 2) for j in (0, 1, 2)])
        self.assertEqual(md.workgroups(by["indirect-zero-count"]), [])
        self.assertEqual(md.workgroups(by["direct-no-primitives"]), [])
        self.assertEqual(md.workgroups(by["direct-primitives-over-max"]), md.workgroups(by["direct-first3"]))
        self.assertEqual(md.workgroups(by["direct-mesh-only-first3"]), [(n, mp.ROW) for n in (3, 4, 5, 6)])
        self.assertEqual(len(by), 12)

    def test_the_largest_launch_fits_the_binade(self):
        for scn in md.scenarios():
            for n, j in md.workgroups(scn):
                self.assertLess(n, mp.MAX_CELLS)
                self.assertLess(j, mp.MAX_CELLS)


class ComparisonTest(unittest.TestCase):
    def image(self, cells):
        kind, rgba = md.expected_image(cells)
        return [tuple(c) if c else md.CLEAR for c in rgba]

    def test_the_reference_agrees_with_itself_and_names_the_workgroups(self):
        cells = [(3, 0), (3, 1), (4, 2)]
        cmp = md.compare(self.image(cells), md.expected_image(cells))
        self.assertEqual((cmp["mismatches"], cmp["ran_ok"], cmp["pixels"][md.COLOR]), (0, True, 9))
        self.assertEqual(cmp["ran"], sorted(cells))
        self.assertEqual(cmp["ran_pixels"], {"3,0": 3, "3,1": 3, "4,2": 3})

    def test_a_first_task_ignored_is_caught(self):
        want = [(n, j) for n in (3, 4, 5, 6) for j in (0, 1, 2)]
        ignored = [(n, j) for n in (0, 1, 2, 3) for j in (0, 1, 2)]
        cmp = md.compare(self.image(ignored), md.expected_image(want))
        self.assertGreater(cmp["mismatches"], 0)
        self.assertFalse(cmp["ran_ok"])

    def test_a_clamped_count_of_zero_draws_nothing_and_a_draw_of_something_is_caught(self):
        empty = md.expected_image([])
        self.assertEqual(md.compare(self.image([]), empty)["mismatches"], 0)
        cmp = md.compare(self.image([(3, 0)]), empty)
        self.assertEqual(cmp["mismatches"], 3)
        self.assertFalse(cmp["ran_ok"])

    def test_a_wrong_output_count_is_caught(self):
        want = [(n, j) for n in (3, 4, 5, 6) for j in (0, 1, 2)]
        for got in ([c for c in want if c[0] < 6], want + [(7, j) for j in (0, 1, 2)], []):
            cmp = md.compare(self.image(got), md.expected_image(want))
            self.assertGreater(cmp["mismatches"], 0)
            self.assertFalse(cmp["ran_ok"])

    def test_a_wrong_colour_is_caught_by_value_and_a_wrong_place_by_pixels(self):
        cells = [(3, 0)]
        got = self.image(cells)
        k = next(i for i, c in enumerate(got) if c != md.CLEAR)
        got[k] = (got[k][0] + 0.01, got[k][1], got[k][2], got[k][3])
        cmp = md.compare(got, md.expected_image(cells))
        self.assertEqual(cmp["mismatches"], 1)
        self.assertGreater(cmp["max_abs_err"], 0.009)

    def test_a_pixel_within_the_tolerance_passes(self):
        cells = [(5, 1)]
        got = [tuple(c + 1e-7 for c in px) if px != md.CLEAR else px for px in self.image(cells)]
        self.assertEqual(md.compare(got, md.expected_image(cells))["mismatches"], 0)


class DrawFileTest(unittest.TestCase):
    def descriptions(self):
        def mod(stage, banks, caps):
            return {"schema": 2, "layout": {"name": "uni", "fits": True}, "sbufs": [], "textures": [], "images": [],
                    "raw_access": {"page_table": False}, "cbufs": banks, "capabilities": caps,
                    "mesh_stage": None if stage == "ps" else {"requires_subgroup_32": True, "threads": 32,
                                                              "first_task": {"push_offset": 24, "form": "device_address"}}}
        user = lambda set_no, binding: {"role": "user", "slot": 3, "set": set_no, "binding": binding}   # noqa: E731
        return {"ts": mod("ts", [user(2, 0)], ["Shader", "MeshShadingEXT", "Int64", "PhysicalStorageBufferAddresses"]),
                "ms": mod("ms", [user(1, 1)], ["Shader", "MeshShadingEXT"]),
                "ms_only": mod("ms", [user(1, 1)], ["Shader", "MeshShadingEXT", "Int64", "PhysicalStorageBufferAddresses"]),
                "ps": mod("ps", [], ["Shader"])}

    def lines(self, scn, counter=None, validation="sync"):
        modules = {k: Path(f"/m/{k}.spv") for k in ("ts", "ms", "ms_only", "ps")}
        with tempfile.TemporaryDirectory() as tmp:
            draw, image = md.draw_file(scn, self.descriptions(), modules, Path(tmp) / "run", counter, validation)
            text = draw.read_text()
            banks = {p.name: p.read_bytes() for p in Path(tmp, "run").glob("*-bank*.dat")}
            records = Path(tmp, "run", "records.dat").read_bytes() if Path(tmp, "run", "records.dat").exists() else None
        return text.splitlines(), banks, records, image

    def test_a_direct_draw_of_a_task_pair(self):
        by = {s.name: s for s in md.scenarios()}
        lines, banks, records, image = self.lines(by["direct-first3"])
        self.assertEqual(lines[0].split()[:1], ["ts"])
        self.assertIn("validation sync", lines)
        self.assertIn("draw direct 4 3", lines)
        self.assertEqual([l.split()[0] for l in lines if l.startswith(("uniform", "taskuniform"))], ["taskuniform", "uniform"])
        self.assertTrue(any(l.startswith("uniform 1 m ") for l in lines), "the mesh stage's bank at set 1, binding 1")
        self.assertTrue(any(l.startswith("taskuniform 0 ") for l in lines), "the task stage's bank at set 2, binding 0")
        self.assertIn("feature bufferDeviceAddress", lines)
        self.assertIn("feature shaderInt64", lines)
        self.assertIsNone(records)
        # c[3][0x10] = 3 mesh workgroups for the task, c[3][0x14] = 1 primitive for the mesh stage
        task = next(v for k, v in banks.items() if k.startswith("ts-"))
        mesh = next(v for k, v in banks.items() if k.startswith("ms-"))
        self.assertEqual((struct.unpack_from("<I", task, 0x10)[0], len(task)), (3, 0x10000))
        self.assertEqual(struct.unpack_from("<I", mesh, 0x14)[0], 1)

    def test_an_indirect_draw_writes_the_conversion_passs_records(self):
        by = {s.name: s for s in md.scenarios()}
        lines, _, records, _ = self.lines(by["indirect-first3"])
        self.assertTrue(any(l.startswith("draw indirect 1 ") for l in lines))
        self.assertEqual(len(records), 48)
        self.assertEqual(struct.unpack_from("<4I", records, 16), (4, 1, 1, 3), "count, 1, 1, then the first task")
        self.assertEqual(struct.unpack_from("<4I", records, 0)[3] != 3, True, "the records around it are other draws'")

    def test_a_mesh_stage_alone_has_no_task_stage_and_its_own_module(self):
        by = {s.name: s for s in md.scenarios()}
        lines, _, _, _ = self.lines(by["direct-mesh-only-first3"])
        self.assertFalse(any(l.startswith(("ts ", "taskuniform")) for l in lines))
        self.assertIn(f"ms {Path('/m/ms_only.spv')}", lines)

    def test_every_counter_run_changes_the_gpu_side_only(self):
        by = {s.name: s for s in md.scenarios()}
        lines, _, _, _ = self.lines(by["direct-first3"], "ignore-first")
        self.assertIn("flag ignorefirst", lines)
        self.assertIn("draw direct 4 3", lines, "the draw still says first 3: the host just points at the zero word")
        lines, _, records, _ = self.lines(by["indirect-first3"], "ignore-first")
        self.assertEqual(struct.unpack_from("<4I", records, 16), (4, 1, 1, 0))
        lines, _, _, _ = self.lines(by["indirect-first3"], "no-barrier")
        self.assertIn("flag nobarrier", lines)
        _, _, records, _ = self.lines(by["indirect-first3"], "unconverted-record")
        self.assertEqual(struct.unpack_from("<4I", records, 16), (4, 3, 0, 0), "the game's {count, first} where the extension reads three counts")
        _, _, records, _ = self.lines(by["indirect-first3"], "wrong-count")
        self.assertEqual(struct.unpack_from("<4I", records, 16)[0], 8)
        _, banks, _, _ = self.lines(by["direct-no-primitives"], "count-ignored")
        mesh = next(v for k, v in banks.items() if k.startswith("ms-"))
        self.assertEqual(struct.unpack_from("<I", mesh, 0x14)[0], 1)
        _, banks, _, _ = self.lines(by["direct-no-primitives"])
        mesh = next(v for k, v in banks.items() if k.startswith("ms-"))
        self.assertEqual(struct.unpack_from("<I", mesh, 0x14)[0], 0)

    def test_a_counter_run_means_something_only_for_the_draws_it_breaks(self):
        by = {s.name: s for s in md.scenarios()}
        self.assertTrue(md.applies("ignore-first", by["direct-first3"]))
        self.assertFalse(md.applies("ignore-first", by["direct-first0"]), "a first task of 0 cannot be ignored")
        self.assertFalse(md.applies("no-barrier", by["direct-first3"]), "a direct draw copies nothing")
        self.assertTrue(md.applies("no-barrier", by["indirect-first3"]))
        self.assertFalse(md.applies("wrong-count", by["indirect-zero-count"]))
        self.assertTrue(md.applies("count-ignored", by["direct-no-primitives"]))
        self.assertFalse(md.applies("count-ignored", by["direct-first3"]))
        self.assertTrue(all(md.applies(None, s) for s in md.scenarios()))

    def test_a_description_that_does_not_fit_is_refused(self):
        res = self.descriptions()
        md.check_description("ts", res["ts"], "ts")
        bad = json.loads(json.dumps(res["ts"]))
        bad["layout"]["fits"] = False
        bad["layout"]["overflow"] = ["no storage slot left"]
        with self.assertRaises(md.InputError):
            md.check_description("ts", bad, "ts")
        bad = json.loads(json.dumps(res["ts"]))
        bad["raw_access"]["page_table"] = True
        with self.assertRaises(md.InputError):
            md.check_description("ts", bad, "ts")
        bad = json.loads(json.dumps(res["ts"]))
        bad["mesh_stage"]["first_task"]["push_offset"] = 20
        with self.assertRaises(md.InputError):
            md.check_description("ts", bad, "ts")
        bad = json.loads(json.dumps(res["ts"]))
        bad["capabilities"].append("NoSuchCapability")
        with self.assertRaises(md.InputError):
            md.device_needs([bad])


@unittest.skipIf(host.EXE is None, host.SKIP_NO_EXE)
class TranslatedProgramsTest(unittest.TestCase):
    """The synthetic programs through the real translator: what the harness binds from is what the descriptions say."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.modules, cls.res = md.translate(Path(cls.tmp.name), str(host.EXE))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_stages_and_where_they_read_the_first_task(self):
        self.assertEqual([self.res[k]["exec_model"] for k in ("ts", "ms", "ms_only")], ["TaskEXT", "MeshEXT", "MeshEXT"])
        self.assertEqual(self.res["ps"]["stage"], "ps")
        self.assertEqual([self.res[k]["mesh_stage"]["first_task"]["read"] for k in ("ts", "ms", "ms_only")], [True, False, True])
        self.assertIsNone(self.res["ps"]["mesh_stage"])

    def test_the_banks_are_at_the_places_the_layout_gives(self):
        (task,) = self.res["ts"]["cbufs"]
        (mesh,) = self.res["ms"]["cbufs"]
        self.assertEqual((task["set"], task["binding"], task["slot"]), (2, 0, 3))
        self.assertEqual((mesh["set"], mesh["binding"], mesh["slot"]), (1, 1, 3))
        self.assertEqual(self.res["ps"]["cbufs"], [])

    def test_the_modules_fit_and_need_the_features_the_harness_enables(self):
        for name, res in self.res.items():
            self.assertTrue(res["layout"]["fits"], name)
        needs = md.device_needs([self.res[k] for k in ("ts", "ms", "ps")])
        self.assertIn("feature bufferDeviceAddress", needs)
        self.assertIn("feature shaderInt64", needs)

    def test_the_mesh_outputs_are_what_the_pixel_stage_reads(self):
        self.assertEqual(self.res["ms"]["mesh_stage"]["vertex_components"], 8)
        self.assertEqual(self.res["ms"]["mesh_stage"]["vertex_locations"], 2)
        self.assertEqual((self.res["ms"]["mesh_stage"]["max_vertices"], self.res["ms"]["mesh_stage"]["max_primitives"]), (3, 1))


@unittest.skipUnless(os.environ.get("OOT2_GPU_TESTS") and host.EXE is not None
                     and (ROOT / "build" / "vkmesh" / ("vkmesh.exe" if os.name == "nt" else "vkmesh")).is_file(),
                     "set OOT2_GPU_TESTS, build the translator and tools/build_vkdraw.py --harness vkmesh to draw")
class GpuDrawTest(unittest.TestCase):
    """The synthetic pair on the GPU: every scenario passes, and every counter-run is caught."""

    def run_main(self, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            return md.main(["--work", tmp, "--xlate", str(host.EXE), *extra])

    def test_every_scenario_draws_what_the_reference_says(self):
        self.assertEqual(self.run_main(), 0)

    def test_every_counter_run_is_caught(self):
        for counter in md.COUNTERS:
            with self.subTest(counter=counter):
                self.assertEqual(self.run_main("--counter", counter), 0)


if __name__ == "__main__":
    unittest.main()
