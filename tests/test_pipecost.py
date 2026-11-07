# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/pipecost/pipe_cost.py: the module census, the pairing, the
binding schemes the measured layouts use, the synthetic shaders and the job
text the harness reads.

Every resource description here is written by hand in the shape of the
translator's res.json; no game data, no Vulkan device, no glslang.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "pipecost"))

import pipe_cost as pc  # noqa: E402


def res(stage: str, cbufs=(), sbufs=(), textures=(), images=(), frag_out_map=0, raw=False) -> dict:
    """A resource description as the translator writes it, with only what describe reads."""
    return {
        "stage": stage, "spv_bytes": 1000,
        "support_buffer": {"set": 0, "binding": 0, "size": 1392},
        "cbufs": [{"set": 0, "binding": 1 + s, "slot": s} for s in cbufs],
        "sbufs": [{"set": 0, "binding": 32 + n} for n in sbufs],
        "textures": [dict(t) for t in textures], "images": [dict(t) for t in images],
        "graphics": {"frag_out_map": frag_out_map},
        "raw_access": {"physical_pointers": raw, "page_table": raw},
        "capabilities": ["Shader"],
    }


def tex(binding: int, set_: int = 0, kind: str = "combined", type_: str = "2d", length: int = 1) -> dict:
    return {"set": set_, "binding": binding, "kind": kind, "type": type_, "array_length": length}


class CensusTests(unittest.TestCase):
    def test_describe_counts_every_set0_resource_and_marks_set1(self):
        d = pc.describe(res("ps", cbufs=[0, 7], sbufs=[0], frag_out_map=0xF,
                            textures=[tex(64), tex(65, length=4), tex(0, set_=1, kind="separate_texture", length=33025),
                                      tex(1, set_=1, kind="sampler", type_="sampler", length=4096)],
                            images=[tex(66, type_="2d")]))
        self.assertEqual(d["binds"], [(0, "ubo", 1), (1, "ubo", 1), (8, "ubo", 1), (32, "ssbo", 1), (64, "tex", 1),
                                      (65, "tex", 4), (66, "img", 1)])
        self.assertEqual(d["set1"], [0, 1])
        self.assertEqual(pc.count(d), 10)
        self.assertEqual(pc.count(d, shared_support=True), 9)

    def test_a_schema_2_description_is_refused(self):
        # Ticket 23's numbering (layout uni) would read here as a different set 0; the census measured schema 1.
        with self.assertRaises(ValueError):
            pc.describe(dict(res("vs"), schema=2))
        self.assertEqual(pc.describe(dict(res("vs"), schema=1))["binds"], [(0, "ubo", 1)])

    def test_texture_kinds(self):
        self.assertEqual(pc.texture_kind(tex(64, type_="buffer")), pc.TEXBUF)
        self.assertEqual(pc.texture_kind(tex(64, kind="separate_texture")), pc.SEPTEX)
        self.assertEqual(pc.texture_kind(tex(64, kind="sampler")), pc.SAMPLER)
        self.assertEqual(pc.texture_kind(tex(64)), pc.TEX)

    def test_pair_count_shares_the_support_buffer(self):
        vs = pc.describe(res("vs", cbufs=[0, 4], sbufs=[0, 1]))
        ps = pc.describe(res("ps", cbufs=[7], textures=[tex(64), tex(65)]))
        self.assertEqual(pc.pair_count(vs, ps), 5 + 3)

    def test_pairs_need_one_vertex_and_one_pixel_module_per_label(self):
        rows = [dict(pc.describe(res(st)), key=f"{st}{i}", label=label, valid=True)
                for i, (st, label) in enumerate([("vs", "a"), ("ps", "a"), ("vs", "b"), ("ps", "b"), ("ps", "b"),
                                                 ("vs", "c")])]
        pairs = pc.make_pairs(rows)
        self.assertEqual([(p["vs"], p["ps"]) for p in pairs], [("vs0", "ps1")])

    def test_nearest_rank_percentile(self):
        v = list(range(1, 101))
        self.assertEqual(pc.percentile(v, 50), 50)
        self.assertEqual(pc.percentile(v, 99), 99)
        self.assertEqual(pc.percentile(v, 100), 100)
        self.assertEqual(pc.percentile([7.0], 99), 7.0)


class LayoutTests(unittest.TestCase):
    def test_targets_from_the_output_map(self):
        self.assertEqual(pc.targets(0), 0)
        self.assertEqual(pc.targets(0xF), 1)
        self.assertEqual(pc.targets(0x100F), 4)

    def test_pair_set_moves_pixel_bindings_and_shares_the_support_buffer(self):
        vs = pc.describe(res("vs", cbufs=[0], sbufs=[0], textures=[tex(64)]))
        ps = pc.describe(res("ps", cbufs=[0], textures=[tex(64, length=2)]))
        s = pc.pair_set(vs, ps)
        self.assertEqual(s, ((0, "ubo", 1, "vf"), (1, "ubo", 1, "v"), (32, "ssbo", 1, "v"), (64, "tex", 1, "v"),
                             (1 + pc.REBIND, "ubo", 1, "f"), (64 + pc.REBIND, "tex", 2, "f")))

    def test_pair_set_refuses_a_collision(self):
        vs = {"binds": [(0, "ubo", 1), (1 + pc.REBIND, "ubo", 1)]}
        ps = {"binds": [(0, "ubo", 1), (1, "ubo", 1)]}
        with self.assertRaises(ValueError):
            pc.pair_set(vs, ps)

    def test_universal_slots_are_disjoint_and_fit_the_push_limit(self):
        used = [b for r in pc.UNI_SLOTS.values() for b in r]
        self.assertEqual(len(used), len(set(used)))
        self.assertNotIn(0, used)
        self.assertLessEqual(1 + len(used), pc.PUSH_LIMIT)
        # The corpus maxima the slots are sized by (census of S1's modules, support buffer apart).
        self.assertEqual({k: len(r) for k, r in pc.UNI_SLOTS.items()},
                         {("v", "ubo"): 8, ("v", "ssbo"): 4, ("f", "ubo"): 11, ("f", "ssbo"): 5})

    def test_synthetic_bindings(self):
        comp = {"v": {"ubo": 2, "ssbo": 1, "tex": 1}, "f": {"ubo": 1, "ssbo": 0, "tex": 3}}
        dkb = pc.synth_bindings("dkb", comp)
        self.assertEqual([b for b, _, _ in dkb["v"] + dkb["f"]], list(range(1, 9)))
        uni = pc.synth_bindings("uni", comp)
        self.assertEqual(uni["v"], [(1, "ubo", 0), (2, "ubo", 1), (9, "ssbo", 0), (pc.TEX_FIRST, "tex", 0)])
        self.assertEqual(uni["f"], [(13, "ubo", 0), (pc.TEX_FIRST + 1, "tex", 1), (pc.TEX_FIRST + 2, "tex", 2),
                                    (pc.TEX_FIRST + 3, "tex", 3)])
        self.assertEqual(pc.composition_count(comp), 1 + 4 + 4)

    def test_synthetic_bindings_refuse_too_many_textures(self):
        comp = {"v": {"ubo": 0, "ssbo": 0, "tex": 5}, "f": {"ubo": 0, "ssbo": 0, "tex": pc.TEX_SLOTS}}
        with self.assertRaises(ValueError):
            pc.synth_bindings("uni", comp)

    def test_synthetic_shaders_use_each_layouts_sets(self):
        comp = {"v": {"ubo": 1, "ssbo": 1, "tex": 1}, "f": {"ubo": 1, "ssbo": 0, "tex": 1}}
        dkb_vs = pc.synth_glsl("v", "dkb", pc.synth_bindings("dkb", comp))
        self.assertIn("layout(set = 0, binding = 3) uniform sampler2D texv0;", dkb_vs)
        self.assertNotIn("gImages", dkb_vs)
        uni_fs = pc.synth_glsl("f", "uni", pc.synth_bindings("uni", comp))
        self.assertIn("layout(set = 1, binding = 13) uniform U_ubof0", uni_fs)
        self.assertIn("gImages[pc.tex[1]]", uni_fs)
        self.assertNotIn("sampler2D tex", uni_fs)
        d13_fs = pc.synth_glsl("f", "d13", pc.synth_bindings("d13", comp))
        self.assertIn("pc.pt.x", d13_fs)
        self.assertIn("layout(set = 1, binding = 5) uniform sampler2D texf0;", d13_fs)

    def test_body_text(self):
        text = pc.body_text([("v", "a.spv"), ("f", "b.spv")], [((0, "ubo", 1, "vf"), (129, "tex", 2, "f"))],
                            [(0, 1, 0, 1, 0)], rebind=128, global_set=1, depth="d32s8", params={"uni_set": 0})
        lines = text.splitlines()
        self.assertIn("rebind 128", lines)
        self.assertIn("global 1", lines)
        self.assertIn("bind 0 septex 33025 vf", lines)
        self.assertEqual(lines[lines.index("set") + 2], "bind 129 tex 2 f")
        self.assertIn("pair 0 1 0 1 0", lines)
        self.assertEqual(lines[-1], "param uni_set 0")
        self.assertNotIn("rebind 0", pc.body_text([], [], [], rebind=0, global_set=0, depth="none"))

    def test_item_stats(self):
        s = pc.item_stats([1.0, 2.0, 3.0, 4.0])
        self.assertEqual((s["n"], s["p50"], s["max"], s["sum"]), (4, 2.0, 4.0, 10.0))


if __name__ == "__main__":
    unittest.main()
