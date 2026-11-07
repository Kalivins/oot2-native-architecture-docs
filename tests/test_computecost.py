# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/computecost/compute_cost.py: the stratum of a program, the
tensor scan, the SPIR-V extension scan, the job bodies and the projection.

Every input is built here; no game data, no Vulkan device, no harness run.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "computecost"))

import compute_cost as cc  # noqa: E402


def instr(opcode: int, rest: int = 0) -> bytes:
    """One 16-byte Ampere instruction whose low 12 bits are `opcode`."""
    return struct.pack("<QQ", (rest << 12) | opcode, 0)


def spirv(*insts: tuple[int, list[int]]) -> bytes:
    words = [0x07230203, 0x00010600, 0, 100, 0]
    for op, operands in insts:
        words += [((1 + len(operands)) << 16) | op] + operands
    return struct.pack(f"<{len(words)}I", *words)


def string_words(s: str) -> list[int]:
    b = s.encode() + b"\0"
    b += b"\0" * (-len(b) % 4)
    return list(struct.unpack(f"<{len(b) // 4}I", b))


class Strata(unittest.TestCase):
    def test_bins_are_half_open(self):
        self.assertEqual(cc.size_bin(0), "<10K")
        self.assertEqual(cc.size_bin(9_999), "<10K")
        self.assertEqual(cc.size_bin(10_000), "10-20K")
        self.assertEqual(cc.size_bin(1_999_999), "1-2M")
        self.assertEqual(cc.size_bin(11_898_332), ">2M")

    def test_kind_prefers_tensor_over_raw(self):
        self.assertEqual(cc.kind_of({"hmma": 1, "ldsm": 0, "movm": 0, "raw": 1}), "T")
        self.assertEqual(cc.kind_of({"hmma": 0, "ldsm": 0, "movm": 0, "raw": "1"}), "R")
        self.assertEqual(cc.kind_of({"hmma": 0, "ldsm": 0, "movm": 0, "raw": 0}), "P")


class TensorScan(unittest.TestCase):
    def test_counts_each_opcode_on_instruction_boundaries(self):
        code = instr(0x23C) + instr(0x23C, 0x5) + instr(0x83B) + instr(0x23A) + instr(0x947)
        self.assertEqual(cc.tensor_counts(code), {"hmma": 2, "ldsm": 1, "movm": 1})

    def test_only_the_low_twelve_bits_decide(self):
        # 0x123C has 0x23C in its low 12 bits; 0x03C does not.
        self.assertEqual(cc.tensor_counts(instr(0x23C, 0x1) + instr(0x03C)), {"hmma": 1, "ldsm": 0, "movm": 0})

    def test_a_partial_trailing_instruction_is_ignored(self):
        self.assertEqual(cc.tensor_counts(instr(0x947) + instr(0x23C)[:8])["hmma"], 0)


class Extensions(unittest.TestCase):
    def test_reads_extensions_up_to_the_memory_model(self):
        data = spirv((17, [1]), (10, string_words("SPV_EXT_shader_atomic_float_add")),
                     (10, string_words("SPV_KHR_physical_storage_buffer")), (14, [5348, 1]),
                     (10, string_words("SPV_KHR_after_the_preamble")))
        self.assertEqual(cc.spirv_extensions(data),
                         ["SPV_EXT_shader_atomic_float_add", "SPV_KHR_physical_storage_buffer"])

    def test_refuses_what_is_not_spirv(self):
        with self.assertRaises(ValueError):
            cc.spirv_extensions(b"\0" * 64)

    def test_maps_to_device_extensions_and_drops_core_ones(self):
        self.assertEqual(cc.device_extensions(["SPV_KHR_physical_storage_buffer", "SPV_EXT_shader_atomic_float_add",
                                               "SPV_EXT_shader_atomic_float_add"]),
                         ["VK_EXT_shader_atomic_float"])

    def test_an_unknown_extension_stops(self):
        with self.assertRaises(ValueError):
            cc.device_extensions(["SPV_VENDOR_something_new"])


class JobBody(unittest.TestCase):
    def prog(self, path, set_text, exts=""):
        return {"module": path, "set": set_text, "extensions": exts}

    def test_warmup_first_sets_shared_extensions_once(self):
        a = "set | bind 0 ubo 1 | bind 1 ssbo 1"
        b = "set | bind 0 ubo 1 | bind 3 img 1"
        warm = self.prog("w.spv", a)
        lines = cc.job_body([self.prog("x.spv", b, "VK_EXT_shader_atomic_float"), self.prog("y.spv", a,
                             "VK_EXT_shader_atomic_float")], warm)
        self.assertEqual(lines, [
            "extension VK_EXT_shader_atomic_float",
            "module w.spv", "module x.spv", "module y.spv",
            "set", "bind 0 ubo 1", "bind 1 ssbo 1",
            "set", "bind 0 ubo 1", "bind 3 img 1",
            "warmup 0 0", "compute 1 1", "compute 2 0",
        ])

    def test_without_warmup_every_record_is_measured(self):
        lines = cc.job_body([self.prog("x.spv", "set | bind 0 ubo 1")], None)
        self.assertEqual(lines[-1], "compute 0 0")
        self.assertNotIn("warmup", " ".join(lines))

    def test_set_lines_refuse_set_one(self):
        res = {"stage": "cs", "spv_bytes": 4, "support_buffer": {"binding": 0}, "cbufs": [], "sbufs": [],
               "textures": [{"set": 1, "binding": 0}], "images": []}
        with self.assertRaises(ValueError):
            cc.set_lines(res)

    def test_set_lines_follow_the_description(self):
        res = {"stage": "cs", "spv_bytes": 4, "support_buffer": {"binding": 0}, "cbufs": [{"binding": 1}],
               "sbufs": [{"binding": 2}], "textures": [{"set": 0, "binding": 3, "kind": "texture"}],
               "images": [{"set": 0, "binding": 4, "type": "2d"}]}
        self.assertEqual(cc.set_lines(res), ["set", "bind 0 ubo 1", "bind 1 ubo 1", "bind 2 ssbo 1", "bind 3 tex 1",
                                             "bind 4 img 1"])


class Projection(unittest.TestCase):
    pop = {("P", "<10K"): 100, ("R", "1-2M"): 10}
    sample = {("P", "<10K"): [10.0, 30.0], ("R", "1-2M"): [1000.0, 3000.0, 2000.0]}

    def test_cells_weighted_by_population(self):
        total, modelled = cc.stratified_total(self.pop, self.sample)
        self.assertAlmostEqual(total, 100 * 20.0 + 10 * 2000.0)
        self.assertEqual(modelled, [])

    def test_an_empty_cell_takes_the_model_and_is_named(self):
        pop = {**self.pop, ("T", "0.5-1M"): 3}
        total, modelled = cc.stratified_total(pop, self.sample, model=lambda cell: 900.0)
        self.assertAlmostEqual(total, 22000.0 + 900.0)
        self.assertEqual(modelled, [("T", "0.5-1M")])

    def test_an_empty_cell_without_a_model_is_an_error(self):
        with self.assertRaises(ValueError):
            cc.stratified_total({**self.pop, ("T", ">2M"): 1}, self.sample)

    def test_bootstrap_is_repeatable_and_brackets_the_estimate(self):
        stat = lambda s: cc.stratified_total(self.pop, {**self.sample, **s})[0]  # noqa: E731
        a = cc.bootstrap(self.pop, self.sample, stat, reps=500, seed=3)
        b = cc.bootstrap(self.pop, self.sample, stat, reps=500, seed=3)
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], 22000.0)
        self.assertGreaterEqual(a[1], 22000.0)

    def test_a_cell_measured_whole_is_not_resampled(self):
        pop = {("T", "1-2M"): 3}
        sample = {("T", "1-2M"): [10.0, 50.0, 90.0]}
        stat = lambda s: cc.stratified_total(pop, {**sample, **s})[0]  # noqa: E731
        self.assertEqual(cc.bootstrap(pop, sample, stat, reps=200, seed=4), (150.0, 150.0))

    def test_weighted_percentile_counts_programs_not_measurements(self):
        # 100 plain programs at about 20 ms outweigh 10 large ones: the median is plain.
        self.assertLessEqual(cc.weighted_percentile(self.pop, self.sample, 50), 30.0)
        self.assertGreaterEqual(cc.weighted_percentile(self.pop, self.sample, 95), 1000.0)

    def test_projected_over_uses_cell_rates(self):
        count, byts = cc.projected_over(self.pop, self.sample, {("P", "<10K"): 5_000, ("R", "1-2M"): 1_500_000}, 1500)
        self.assertAlmostEqual(count, 10 * 2 / 3)
        self.assertAlmostEqual(byts, 10 * 2 / 3 * 1_500_000)

    def test_loglog_fit_recovers_a_power_law(self):
        pairs = [(b, 0.002 * b ** 1.5) for b in (1e4, 3e4, 1e5, 3e5, 1e6)]
        fit = cc.loglog_fit(pairs)
        self.assertAlmostEqual(fit["b"], 1.5, places=6)
        self.assertAlmostEqual(fit["smear"], 1.0, places=6)
        self.assertAlmostEqual(cc.predict(fit, 2e5), 0.002 * 2e5 ** 1.5, delta=1e-6 * 0.002 * 2e5 ** 1.5)
        self.assertAlmostEqual(fit["spearman"], 1.0)

    def test_spearman_handles_ties_and_order(self):
        self.assertAlmostEqual(cc.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertTrue(math.isfinite(cc.spearman([1, 1, 2, 2], [3, 4, 3, 4])))


class Guards(unittest.TestCase):
    def test_work_inside_the_repository_is_refused(self):
        self.assertEqual(cc.main(["report", "--work", str(ROOT / "build" / "s6"), "--census", "x.tsv"]), 2)

    def test_parse_extra(self):
        self.assertEqual(cc.parse_extra("result=0 cpu_ms=12.5 bytes=100"),
                         {"result": "0", "cpu_ms": "12.5", "bytes": "100"})

    def test_spread_takes_both_ends(self):
        progs = [{"bytes": b} for b in range(0, 1000, 10)]
        picked = cc.spread(progs, 5, 10_000)
        self.assertEqual([p["bytes"] for p in picked][0], 0)
        self.assertEqual([p["bytes"] for p in picked][-1], 990)
        self.assertEqual(len(picked), 5)


if __name__ == "__main__":
    unittest.main()
