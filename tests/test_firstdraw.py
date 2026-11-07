# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/firstdraw: the SM86 listing evaluator, the CPU reference and
the driver's input builders.

Every program here is written by hand in nvdisasm's listing format; no game
data, no CUDA toolkit, no Vulkan device.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import struct
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "firstdraw"))

import candidates  # noqa: E402
import first_draw  # noqa: E402
import reference as ref  # noqa: E402
import sass_eval as se  # noqa: E402

f2b, b2f = se.float_to_bits, se.bits_to_float


def listing(*instrs: tuple) -> str:
    """nvdisasm -hex text: (text,) or (text, lo, hi) per instruction, 16 bytes apart."""
    lines = ['\t.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"']
    for i, ins in enumerate(instrs):
        text, lo, hi = (ins + (0, 0))[:3] if len(ins) == 1 else ins
        lines.append(f"        /*{16 * i:04x}*/                   {text} ;   /* 0x{lo:016x} */")
        lines.append(f"                                                       /* 0x{hi:016x} */")
    return "\n".join(lines) + "\n"


def run(text: str, **env) -> se.Result:
    return se.Machine(se.parse_listing(text), se.Env(**env)).run()


def sph_pixel(imap: dict[int, tuple[int, int, int, int]], targets: dict[int, int], kills: bool = False) -> bytes:
    """A pixel shader program header (type 2) with the given input and output maps."""
    w = [0] * 32
    w[0] = 2 | (4 << 5) | (5 << 10) | (int(kills) << 15)
    for vec, comps in imap.items():
        byte = sum(m << (2 * c) for c, m in enumerate(comps))
        w[6 + (vec >> 2)] |= byte << ((vec & 3) * 8)
    for t, mask in targets.items():
        w[18] |= mask << (4 * t)
    return struct.pack("<32I", *w)


class RoundingTests(unittest.TestCase):
    def test_nearest_even_and_toward_zero(self):
        third = Fraction(1, 3)
        rn, rz = se.round_f32(third), se.round_f32(third, rz=True)
        self.assertEqual(rn, f2b(1 / 3))
        self.assertEqual(rz, rn - 1)  # 1/3 rounds up to nearest, so toward zero is one ulp lower
        self.assertLess(Fraction(b2f(rz)), third)

    def test_ties_go_to_even(self):
        one = Fraction(1)
        half_ulp = Fraction(1, 1 << 24)
        self.assertEqual(se.round_f32(one + half_ulp), 0x3F800000)  # tie, even mantissa kept
        self.assertEqual(se.round_f32(one + 3 * half_ulp), 0x3F800002)  # tie, rounds to even

    def test_overflow_and_flush(self):
        big = Fraction(2) ** 128
        self.assertEqual(se.round_f32(big), 0x7F800000)
        self.assertEqual(se.round_f32(-big, rz=True), 0xFF7FFFFF)
        tiny = Fraction(1, 1 << 140)
        self.assertNotEqual(se.round_f32(tiny), 0)
        self.assertEqual(se.round_f32(tiny, ftz=True), 0)

    def test_fused_multiply_add_rounds_once(self):
        # a*b+c with a = b = 1 + 2^-12, c = -(1 + 2^-11): the exact result is 2^-24,
        # which an unfused product would lose to rounding.
        a = f2b(1 + 2 ** -12)
        c = f2b(-(1 + 2 ** -11))
        res = run(listing(("MOV R1, 0x%x" % a,), ("MOV R2, 0x%x" % c,), ("FFMA R0, R1, R1, R2",), ("EXIT",)))
        self.assertEqual(b2f(res.regs[0]), 2 ** -24)


class MachineTests(unittest.TestCase):
    def test_arithmetic_modifiers(self):
        text = listing(
            ("MOV R1, 0x%x" % f2b(3.0),), ("MOV R2, 0x%x" % f2b(0.5),),
            ("FMUL.FTZ.D2 R3, R1, R2",),          # 3 * 0.5 / 2
            ("FADD.FTZ R4, -R1, 1",),             # 1 - 3
            ("FFMA.FTZ.SAT R5, R1, R2, -|R4|",),  # 1.5 - 2, saturated
            ("FADD.FTZ R6, R1, c[0x3][0x8]",),
            ("EXIT",))
        res = run(text, cbank=lambda bank, off: f2b(10.0) if (bank, off) == (3, 8) else 0)
        self.assertEqual([b2f(res.regs[k]) for k in (3, 4, 5, 6)], [0.75, -2.0, 0.0, 13.0])

    def test_predicates_branches_and_exit(self):
        text = listing(
            ("MOV R1, 0x%x" % f2b(2.0),),
            ("FSETP.GEU.FTZ.AND P0, PT, R1, 1, PT",),
            ("@!P0 BRA 0x50",),
            ("MOV R0, 0x7",),
            ("EXIT",),
            ("MOV R0, 0x9",),
            ("EXIT",))
        self.assertEqual(run(text).regs[0], 7)

    def test_lop3_predicate_and_kill(self):
        text = listing(
            ("MOV R5, 0x6",),
            ("LOP3.LUT P0, RZ, R5, 0x2, RZ, 0xc0, !PT",),
            ("@P0 KILL",),
            ("EXIT",))
        self.assertTrue(run(text).killed)
        self.assertFalse(run(text.replace("0x6", "0x4", 1)).killed)

    def test_attribute_loads_stores_and_global_memory(self):
        mem = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        text = listing(
            ("ULDC.64 UR4, c[0x0][0x170]",),
            ("LDG.E.128.CONSTANT R4, [UR4+0x10]",),
            ("ALD.64 R0, a[0x80]",),
            ("AST.128 a[0x70], R4",),
            ("AST.64 a[0xa0], R0",),
            ("EXIT",))
        base = 0x100000000
        res = run(text,
                  cbank=lambda bank, off: {0x170: base & 0xFFFFFFFF, 0x174: base >> 32}.get(off, 0),
                  gload=lambda addr, n: mem[addr - base - 0x10:addr - base - 0x10 + n],
                  ald=lambda a: f2b(a / 16.0))
        self.assertEqual([b2f(res.stores[0x70 + 4 * k]) for k in range(4)], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual([b2f(res.stores[a]) for a in (0xA0, 0xA4)], [8.0, 8.25])

    def test_texture_operands_checked_against_the_encoding(self):
        # TEX R6, R8, R6, R7: Rd = R8 (bits 16-23), Ra = R6, Rb = R7, handle 0x18,
        # Rd2 = R6 (bits 64-71), write mask 0xf (bits 72-75).
        lo = 0x08 << 16 | 0x06 << 24 | 0x07 << 32 | 0x18 << 40
        hi = 0x06 | 0xF << 8
        good = listing(("MOV R6, 0x%x" % f2b(0.25),), ("MOV R7, 0x%x" % f2b(0.75),),
                       ("TEX.SCR R6, R8, R6, R7, 0x2, 0x18, 2D", lo, hi), ("EXIT",))
        seen = []
        res = run(good, tex=lambda h, dim, c, layer, lod: seen.append((h, dim, c, layer, lod)) or (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(seen, [(0x18, "2D", [0.25, 0.75], None, None)])
        self.assertEqual([round(b2f(res.regs[r]), 6) for r in (8, 9, 6, 7)], [0.1, 0.2, 0.3, 0.4])
        bad = good.replace("0x%016x" % lo, "0x%016x" % (lo ^ (0x01 << 16)))
        with self.assertRaises(se.Unsupported):
            run(bad, tex=lambda *a: (0, 0, 0, 0))

    def test_unknown_instructions_and_modifiers_refuse(self):
        for text in ("IMAD.WIDE R0, R1, R2, R3", "FMUL.FTZ.XYZ R0, R1, R2", "LOP3.LUT P0, RZ, R1, 0x2, RZ, 0xc0, P1",
                     "SHF.L.U32.XYZ R0, R1, 0x2, RZ", "F2I.F64 R0, R1", "HADD2 R0, R1, R2"):
            with self.subTest(text=text), self.assertRaises(se.Unsupported):
                run(listing((text,), ("EXIT",)))


def tex_words(rd2: int, rd: int, ra: int, rb: int, handle: int, mask: int = 0xF) -> tuple[int, int]:
    """The two words of an SM70+ TEX whose operands are these registers (255 is RZ)."""
    return rd << 16 | ra << 24 | rb << 32 | handle << 40, rd2 | mask << 8


class IntegerTests(unittest.TestCase):
    """The integer, select and conversion instructions world-material programs use (ticket 23)."""

    def regs(self, *instrs, **env) -> dict:
        return run(listing(*[(i,) for i in instrs], ("EXIT",)), **env).regs

    def test_add_shift_and_multiply(self):
        r = self.regs("MOV R1, 0x10", "IADD3 R2, R1, -0x64, RZ", "IADD3 R3, -R1, 0x5, RZ", "SHF.L.U32 R4, R1, 0x8, RZ",
                      "MOV R5, 0x80000000", "SHF.R.S32.HI R6, RZ, 0x1f, R5", "SHF.R.U32.HI R7, RZ, 0x8, R5",
                      "IMAD R8, R1, R1, 0x10", "LEA R9, R1, 0x80, 0x8", "UIADD3 UR4, URZ, -0x1, URZ", "MOV R10, UR4")
        self.assertEqual(r[2], (0x10 - 0x64) & 0xFFFFFFFF)
        self.assertEqual(r[3], (0x5 - 0x10) & 0xFFFFFFFF)
        self.assertEqual((r[4], r[6], r[7]), (0x1000, 0xFFFFFFFF, 0x800000))
        self.assertEqual((r[8], r[9], r[10]), (0x110, 0x1080, 0xFFFFFFFF))

    def test_64_bit_address_with_carry(self):
        # LEA R0, P0, ... then IADD3.X carries into the high word: c[0][0x200] holds 0xFFFFFFF0, 0x1.
        bank = {0x200: 0xFFFFFFF0, 0x204: 0x1}
        prog = ("LEA R0, P0, R0, c[0x0][0x200], 0x4", "IADD3.X R1, RZ, c[0x0][0x204], RZ, P0, !PT")
        r = self.regs("MOV R0, 0x3", *prog, cbank=lambda b, o: bank.get(o, 0))
        self.assertEqual((r[0], r[1]), (0x20, 0x2))
        r = self.regs("MOV R0, 0x0", *prog, cbank=lambda b, o: bank.get(o, 0))
        self.assertEqual((r[0], r[1]), (0xFFFFFFF0, 0x1))

    def test_compare_min_max_and_predicate_logic(self):
        r = self.regs("MOV R1, 0xfffffffe", "IMNMX R2, RZ, R1, !PT", "IMNMX.U32 R3, R1, 0x4, PT", "IMNMX R4, R1, 0x4, PT",
                      "ISETP.GT.AND P0, PT, R1, RZ, PT", "ISETP.GT.U32.AND P1, PT, R1, RZ, PT",
                      "PLOP3.LUT P2, PT, PT, PT, PT, 0x80, 0x0", "@P0 MOV R6, 0x5", "@P2 MOV R6, 0x7", "@!P1 MOV R6, 0x9")
        self.assertEqual((r[2], r[3], r[4]), (0, 4, 0xFFFFFFFE))  # max(0, -2), unsigned min, signed min
        self.assertEqual(r[6], 7)  # -2 > 0 is false signed, true unsigned; PLOP3's LUT 0x80 of three trues

    def test_float_select_min_max_and_set(self):
        nan, one, two = 0x7FC00000, f2b(1.0), f2b(2.0)
        r = self.regs(f"MOV R1, {nan:#x}", f"MOV R2, {two:#x}", "FMNMX.FTZ R3, R1, R2, PT", "FMNMX.FTZ R4, R2, 1, PT",
                      "FMNMX.FTZ R5, R2, 1, !PT", "FMNMX.FTZ R6, -RZ, RZ, PT", "FMNMX.FTZ R7, -RZ, RZ, !PT",
                      "FSEL R8, R2, -1, !PT", "FSET.BF.GE.FTZ.AND R9, R2, 1, PT", "FSET.BF.GE.FTZ.AND R10, R1, 1, PT")
        self.assertEqual((r[3], r[4], r[5]), (two, one, two))  # a NaN loses to a number
        self.assertEqual((r[6], r[7]), (0x80000000, 0))  # min(-0, +0) is -0, max is +0
        self.assertEqual((r[8], r[9], r[10]), (f2b(-1.0), one, 0))

    def test_conversions(self):
        r = self.regs(f"MOV R1, {f2b(-2.5):#x}", "F2I.FTZ.TRUNC.NTZ R2, R1", "FRND.FLOOR R3, R1", "FRND R4, R1",
                      "MOV R5, 0x7fc00000", "F2I.FTZ.TRUNC.NTZ R6, R5", f"MOV R7, {f2b(3e9):#x}", "F2I.FTZ.TRUNC.NTZ R8, R7",
                      "MOV R9, 0xfffffffd", "I2FP.F32.S32 R10, R9", "MOV R11, 0x3c003800", "HADD2.F32 R12, -RZ, R11.H0_H0",
                      "HADD2.F32 R13, -RZ, R11.H1_H1")
        self.assertEqual((r[2], b2f(r[3]), b2f(r[4])), (0xFFFFFFFE, -3.0, -2.0))  # trunc, floor, nearest even
        self.assertEqual((r[6], r[8]), (0, 0x7FFFFFFF))  # NaN gives 0, out of range saturates
        self.assertEqual((b2f(r[10]), b2f(r[12]), b2f(r[13])), (-3.0, 0.5, 1.0))

    def test_sine_and_cosine_take_turns(self):
        r = self.regs(f"MOV R1, {f2b(0.25):#x}", "MUFU.SIN R2, R1", "MUFU.COS R3, R1", "MUFU.SIN R4, RZ")
        self.assertAlmostEqual(b2f(r[2]), 1.0, places=6)
        self.assertAlmostEqual(b2f(r[3]), 0.0, places=6)
        self.assertEqual(r[4], 0)


class ControlTests(unittest.TestCase):
    def test_jump_table_through_c1_and_brx(self):
        # LDC reads the target from c[1] at a register offset; BRX R, imm adds imm and the next address.
        text = listing(("MOV R2, 0x4",), ("LDC R4, c[0x1][R2]",), ("SHF.R.S32.HI R5, RZ, 0x1f, R4",),
                       ("BRX R4 -0x40",), ("MOV R0, 0x1",), ("EXIT",), ("MOV R0, 0x2",), ("EXIT",))
        c1 = {0: 0x40, 4: 0x60}
        self.assertEqual(run(text, cbank=lambda b, o: c1.get(o, 0) if b == 1 else 0).regs[0], 2)
        c1[4] = 0x40
        self.assertEqual(run(text, cbank=lambda b, o: c1.get(o, 0) if b == 1 else 0).regs[0], 1)

    def test_call_barriers_and_predicated_branch(self):
        text = listing(("BSSY B0, 0x50",), ("DEPBAR.LE SB5, 0x1",), ("ISETP.GE.AND P0, PT, RZ, RZ, PT",),
                       ("@P0 CALL.REL.NOINC 0x50",), ("MOV R0, 0x9",), ("BSYNC B0",),
                       ("PLOP3.LUT P1, PT, PT, PT, PT, 0x0, 0x0",), ("@P0 BRA P1, 0x90",), ("MOV R1, 0x3",), ("EXIT",))
        r = run(text).regs
        self.assertEqual((r[0], r[1]), (0, 3))  # the call skipped R0; the branch needs P1 too

    def test_global_load_address_forms(self):
        mem = {0x1000 + 4 * k: k for k in range(64)}
        bank = {0x1f0: 0x1000, 0x1f4: 0}
        r = run(listing(("ULDC.64 UR6, c[0x0][0x1f0]",), ("MOV R5, 0x10",), ("LDG.E.STRONG.SM R1, [R5.U32+UR6]",),
                        ("LDG.E.STRONG.SM R2, [R5.U32+UR6+0x4]",), ("MOV R8, 0x1008",), ("MOV R9, RZ",),
                        ("LDG.E.STRONG.SM R3, [R8]",), ("LDG.E.STRONG.SM R4, [UR6+0x70]",), ("EXIT",)),
                cbank=lambda b, o: bank.get(o, 0), gload=lambda a, n: struct.pack("<I", mem[a])).regs
        self.assertEqual((r[1], r[2], r[3], r[4]), (4, 5, 2, 28))

    def test_flat_attributes_read_their_own_callback(self):
        r = run(listing(("IPA.CONSTANT R0, a[0x14c]",), ("EXIT",)), ipa_flat=lambda a: a, ipa=lambda a: 0).regs
        self.assertEqual(r[0], 0x14C)


class TextureInstructionTests(unittest.TestCase):
    """Sources split between Ra and Rb, results packed by the mask (ticket 23)."""

    def sample(self, text: str, words: tuple[int, int], regs: dict[int, float], from_ra: bool = False):
        prog = [(f"MOV R{k}, 0x{f2b(v):x}",) for k, v in regs.items()] + [(text, *words), ("EXIT",)]
        seen = []
        m = se.Machine(se.parse_listing(listing(*prog)), se.Env(tex=lambda *a: seen.append(a) or (0.1, 0.2, 0.3, 0.4)), from_ra)
        m.run()
        return seen[0], m

    def test_3d_reads_x_y_from_ra_and_z_from_rb(self):
        text, words = "TEX.SCR.LZ RZ, R10, R12, R3, 0x2, 0x36, 3D, 0x1", tex_words(255, 10, 12, 3, 0x36, 0x1)
        (h, dim, coords, layer, lod), m = self.sample(text, words, {12: 0.1, 13: 0.2, 14: 0.7, 3: 0.3})
        self.assertEqual((h, dim, layer, lod), (0x36, "3D", None, 0.0))
        self.assertEqual([round(c, 6) for c in coords], [0.1, 0.2, 0.3])
        self.assertEqual(round(b2f(m.r[10]), 6), 0.1)  # mask 0x1: red to Rd
        (_, _, coords, _, _), _ = self.sample(text, words, {12: 0.1, 13: 0.2, 14: 0.7, 3: 0.3}, from_ra=True)
        self.assertEqual([round(c, 6) for c in coords], [0.1, 0.2, 0.7])  # the counter-run's order

    def test_cube_with_explicit_level_and_2d_with_one_register_each(self):
        (_, dim, coords, _, lod), m = self.sample("TEX.SCR.LL.NODEP R18, R16, R16, R18, 0x2, 0x1e, CUBE",
                                                  tex_words(18, 16, 16, 18, 0x1E), {16: -0.5, 17: 1.0, 18: 0.25, 19: 2.0})
        self.assertEqual((dim, coords, lod), ("CUBE", [-0.5, 1.0, 0.25], 2.0))
        self.assertEqual([round(b2f(m.r[k]), 6) for k in (16, 17, 18, 19)], [0.1, 0.2, 0.3, 0.4])
        (_, dim, coords, _, lod), m = self.sample("TEX.SCR RZ, R4, R13, R8, 0x2, 0x18, 2D, 0x8",
                                                  tex_words(255, 4, 13, 8, 0x18, 0x8), {13: 0.25, 8: 0.75, 14: 9.0})
        self.assertEqual((dim, coords, lod), ("2D", [0.25, 0.75], None))
        self.assertEqual(round(b2f(m.r[4]), 6), 0.4)  # mask 0x8: alpha packed into Rd

    def test_mask_packs_three_components(self):
        _, m = self.sample("TEX.SCR.LZ R19, R16, R24, R17, 0x2, 0x12, 2D, 0x7", tex_words(19, 16, 24, 17, 0x12, 0x7),
                           {24: 0.5, 17: 0.5})
        self.assertEqual([round(b2f(m.r[k]), 6) for k in (16, 17, 19)], [0.1, 0.2, 0.3])


class HeaderTests(unittest.TestCase):
    def test_pixel_header(self):
        sph = se.parse_sph(sph_pixel({0: (2, 2, 2, 2), 2: (3, 3, 0, 0)}, {0: 0xF, 1: 0x3}, kills=True))
        self.assertEqual((sph["sph_type"], sph["shader_type"], sph["kills_pixels"]), (2, 5, True))
        self.assertEqual(sph["imap_generic"][0], (2, 2, 2, 2))
        self.assertEqual(sph["imap_generic"][2], (3, 3, 0, 0))
        self.assertEqual(se.pixel_output_registers(sph), {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (1, 0): 4, (1, 1): 5})


class ReferenceTests(unittest.TestCase):
    # A vertex program that passes position (input 0) and one attribute (input 1)
    # through, and a pixel program that writes the perspective-corrected
    # attribute 0 to colour target 0 the way compiled code does: IPA.PASS of the
    # attribute times the reciprocal of IPA.PASS of position W.
    VS = listing(("ALD.128 R0, a[0x80]",), ("AST.128 a[0x70], R0",), ("ALD.128 R4, a[0x90]",),
                 ("AST.128 a[0x80], R4",), ("EXIT",))
    PS = listing(("IPA.PASS R4, a[0x7c]",), ("MUFU.RCP R5, R4",), ("IPA.PASS R0, a[0x80]",),
                 ("IPA.PASS R1, a[0x84]",), ("FMUL.FTZ R0, R0, R5",), ("FMUL.FTZ R1, R1, R5",),
                 ("MOV R2, RZ",), ("MOV R3, 0x%x" % f2b(1.0),), ("EXIT",))

    def draw(self, **kw) -> ref.Draw:
        # One triangle with w of 1, 2 and 4, attribute = its own vertex index.
        verts = []
        for i, (x, y, w) in enumerate(((-0.9, -0.9, 1.0), (1.8, -1.8, 2.0), (-3.6, 3.6, 4.0))):
            words = {0x80 + 4 * c: f2b(v) for c, v in enumerate((x, y, 0.5 * w, w))}
            words.update({0x90 + 4 * c: f2b(v) for c, v in enumerate((float(i), 1.0 - i, 0.0, 1.0))})
            verts.append(words)
        sph = se.parse_sph(sph_pixel({0: (2, 3, 0, 0)}, {0: 0xF}))
        return ref.Draw(16, 16, se.parse_listing(self.VS), se.parse_listing(self.PS), sph, verts, [(0, 1, 2)],
                        lambda b, o: 0, lambda b, o: 0, lambda a, n: b"", {}, **kw)

    def test_perspective_and_screen_linear_interpolation(self):
        d = self.draw()
        exp = ref.render(d)
        screen = [ref._screen(d, o) for o in exp.vertex_outputs]
        checked = 0
        for k, kind in enumerate(exp.kind):
            if kind != ref.COLOR:
                continue
            cx, cy = k % 16 + 0.5, k // 16 + 0.5
            area = ((screen[1][0] - screen[0][0]) * (screen[2][1] - screen[0][1])
                    - (screen[2][0] - screen[0][0]) * (screen[1][1] - screen[0][1]))
            b = []
            for i in range(3):
                a, c = screen[(i + 1) % 3], screen[(i + 2) % 3]
                b.append(((c[0] - a[0]) * (cy - a[1]) - (c[1] - a[1]) * (cx - a[0])) / area)
            w = [s[3] for s in screen]
            persp = sum(b[i] * i / w[i] for i in range(3)) / sum(b[i] / w[i] for i in range(3))
            linear = sum(b[i] * (1.0 - i) for i in range(3)) * (1 / sum(b[i] / w[i] for i in range(3)))
            got = exp.rgba[k]
            self.assertAlmostEqual(got[0], persp, places=5)
            self.assertAlmostEqual(got[1], linear, places=5)
            checked += 1
        self.assertGreater(checked, 20)

    def test_counter_run_changes_the_image(self):
        a = ref.render(self.draw())
        b = ref.render(self.draw(swap_interpolation=True))
        diffs = sum(1 for x, y in zip(a.rgba, b.rgba) if x and y and abs(x[0] - y[0]) > 1e-3)
        self.assertGreater(diffs, 10)

    def test_nearest_texel_and_ambiguity(self):
        t = ref.Texture(4, 2, [(float(i), 0.0, 0.0, 1.0) for i in range(8)])
        self.assertEqual(t.sample([0.6, 0.75], None, 1e-3), ((6.0, 0.0, 0.0, 1.0), False))
        self.assertTrue(t.sample([0.5, 0.25], None, 1e-3)[1])  # exactly on the boundary between texels 1 and 2
        self.assertEqual(t.sample([-3.0, 9.0], None, 1e-3)[0][0], 4.0)  # clamped to column 0, row 1
        flat = ref.Texture(4, 2, [(1.0, 0.0, 0.0, 1.0)] * 4 + [(float(i), 0.0, 0.0, 1.0) for i in range(4)])
        self.assertFalse(flat.sample([0.5, 0.25], None, 1e-3)[1])  # a boundary between equal texels is not ambiguous
        self.assertTrue(flat.sample([0.5, 0.75], None, 1e-3)[1])

    def test_3d_and_array_texels(self):
        texels = [(float(i), 0.0, 0.0, 1.0) for i in range(2 * 2 * 3)]
        t3 = ref.Texture(2, 2, texels, kind="3d", layers=3)
        self.assertEqual(t3.sample([0.75, 0.2, 0.9], None, 1e-3), ((9.0, 0.0, 0.0, 1.0), False))  # slice 2, row 0, column 1
        self.assertTrue(t3.sample([0.75, 0.2, 2 / 3], None, 1e-3)[1])  # between slices 1 and 2
        ta = ref.Texture(2, 2, texels, kind="2darray", layers=3)
        self.assertEqual(ta.sample([0.2, 0.8], 1, 1e-3)[0][0], 6.0)
        self.assertEqual(ta.sample([0.2, 0.8], 9, 1e-3)[0][0], 10.0)  # the layer clamps

    def test_cube_face_selection(self):
        # One texel per face (face index in red): the major axis and its sign pick the face, +X, -X, +Y, -Y, +Z, -Z.
        cube = ref.Texture(1, 1, [(float(f), 0.0, 0.0, 1.0) for f in range(6)], kind="cube", layers=6)
        for direction, face in (((1, 0.2, 0.1), 0), ((-1, 0.2, 0.1), 1), ((0.2, 1, 0.1), 2), ((0.2, -1, 0.1), 3),
                                ((0.1, 0.2, 1), 4), ((0.1, 0.2, -1), 5)):
            with self.subTest(direction=direction):
                self.assertEqual(cube.sample(list(direction), None, 1e-3), ((float(face), 0.0, 0.0, 1.0), False))
        self.assertTrue(cube.sample([1.0, 1.0, 0.0], None, 1e-3)[1])  # a tie between two faces
        # s and t on a face: +Z takes s = (x / |z| + 1) / 2 and t = (-y / |z| + 1) / 2.
        grid = ref.Texture(2, 2, [(float(i), 0.0, 0.0, 1.0) for i in range(24)], kind="cube", layers=6)
        self.assertEqual(grid.sample([0.5, 0.5, 1.0], None, 1e-3)[0][0], 4 * 4 + 1)  # face 4, row 0, column 1
        self.assertEqual(grid.sample([-0.5, -0.5, 1.0], None, 1e-3)[0][0], 4 * 4 + 2)  # row 1, column 0

    def test_vertex_and_instance_indices(self):
        # The vertex program stores its vertex index (a[0x2fc]) and instance index (a[0x2f8]) as outputs.
        vs = se.parse_listing(listing(("ALD R0, a[0x2fc]",), ("ALD R1, a[0x2f8]",), ("AST.64 a[0x90], R0",), ("EXIT",)))
        d = self.draw()
        d.vs = vs
        outs = ref.shade_vertices(d)
        self.assertEqual([(o[0x90], o[0x94]) for o in outs], [(0, 0), (1, 0), (2, 0)])


class DriverHelperTests(unittest.TestCase):
    def test_bank_bytes(self):
        b = first_draw.bank_bytes({"0x170": {"u64": 0x123456700}, "0x10": {"f32": [1.5]}, "0x20": {"u32": ["0x2"]}})
        self.assertEqual(len(b), first_draw.BANK_BYTES)
        self.assertEqual(struct.unpack_from("<Q", b, 0x170)[0], 0x123456700)
        self.assertEqual(struct.unpack_from("<f", b, 0x10)[0], 1.5)
        self.assertEqual(struct.unpack_from("<I", b, 0x20)[0], 2)
        with self.assertRaises(first_draw.InputError):
            first_draw.bank_bytes({"0x2": {"u32": [1]}})

    def test_merge_banks_overrides_one_value(self):
        base = {"ps": {"7": {"0x0": {"f32": [1.0]}, "0x28": {"u32": [0]}}}}
        out = first_draw.merge_banks(base, {"ps": {"7": {"0x28": {"u32": [2]}}}})
        self.assertEqual(out["ps"]["7"], {"0x0": {"f32": [1.0]}, "0x28": {"u32": [2]}})
        self.assertEqual(base["ps"]["7"]["0x28"], {"u32": [0]})

    def test_memory_window_and_load(self):
        m = first_draw.Memory([{"address": 0x1000, "f32": [1.0, 2.0]}])
        self.assertEqual(m.load(0x1004, 4), struct.pack("<f", 2.0))
        self.assertEqual(m.window(0xFF8, 16), bytes(8) + struct.pack("<2f", 1.0, 2.0))
        with self.assertRaises(se.Unsupported):
            m.load(0x1008, 4)

    def test_support_buffer_layout(self):
        b = first_draw.support_buffer(128, 64)
        self.assertEqual(len(b), 1392)
        self.assertEqual(struct.unpack_from("<2f", b, 160), (128.0, 64.0))  # viewport_size
        self.assertEqual(struct.unpack_from("<f", b, 192)[0], 1.0)  # render_scale[0]
        self.assertEqual(struct.unpack_from("<4i", b, 0), (0, 0, 0, 0))  # no alpha test

    def test_png_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.png"
            first_draw.write_png(p, 2, 1, [(1.0, 0.0, 0.0, 1.0), (math.nan, 2.0, -1.0, 0.5)])
            data = p.read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", data[16:24]), (2, 1))

    def test_stores_and_atomics_only_for_writable_storage(self):
        need = first_draw.needs_stores_and_atomics
        self.assertFalse(need({"sbufs": [], "images": []}))
        self.assertFalse(need({"sbufs": [{"non_writable": True}], "images": [{"non_writable": True}]}))
        self.assertTrue(need({"sbufs": [{"non_writable": True}], "images": [{"non_writable": False}]}))
        self.assertTrue(need({"sbufs": [{"write": False}], "images": []}))  # a description from before the post-pass marked them
        # The raw helper's miss path writes the table's miss log, so a module that only loads through it needs the feature too
        self.assertTrue(need({"sbufs": [], "images": [], "raw_access": {"miss_log": True}}))
        self.assertFalse(need({"sbufs": [], "images": [], "raw_access": {"miss_log": False}}))


def description(stage: str, *, cbufs=(), sbufs=(), textures=(), inputs=None, fits=True, schema=2, images=()) -> dict:
    """A schema 2 resource description with only the fields the binding reads (layout `uni`)."""
    return {
        "schema": schema, "stage": stage,
        "layout": {"name": "uni", "fits": fits, "overflow": [] if fits else ["texture: no texture slot left"]},
        "answers": {"heap_images": 33025, "heap_samplers": 4096},
        "support_buffer": {"set": 1, "binding": 0, "size": 1392},
        "storage_buffer_contract": {"alignment": 16},
        "cbufs": [{"set": 1, "binding": b, "slot": s} for b, s in cbufs],
        "sbufs": [{"set": 1, "binding": b, "cb_slot": s, "address_byte_offset": o, "size_byte_offset": o + 8, "role": "guest"}
                  for b, s, o in sbufs],
        "textures": [{"set": 0, "binding": 0, "via": via, "push_slot": slot, "push_offset": 16 + 4 * slot,
                      "kind": "combined", "type": ty, "array_length": 1, "tex_word_offset": h}
                     for slot, h, ty, via in textures],
        "images": list(images),
        "vertex_inputs": inputs,
    }


class BindingTests(unittest.TestCase):
    """The draw's bindings come from the two descriptions alone (ticket 23, layout `uni`)."""

    def setUp(self):
        bank = first_draw.bank_bytes({"0x110": {"u64": 0x1004}, "0x118": {"u32": [8]}})
        self.banks = {"vs": {0: bank, 4: first_draw.bank_bytes({"0x0": {"f32": [2.0]}})}, "ps": {7: bytearray(first_draw.BANK_BYTES)}}
        self.mem = first_draw.Memory([{"address": 0x1000, "f32": [1.0, 2.0, 3.0, 4.0]}])
        tex = first_draw.make_texture({"width": 2, "height": 2})
        self.textures = {"vs": {0x10: tex}, "ps": {0x10: tex, 0x12: tex}}

    def pair(self, **ps) -> dict:
        vs = description("vs", cbufs=[(1, 0), (2, 4)], sbufs=[(9, 0, 0x110)], textures=[(0, 0x10, "2d", "push_slot")],
                         inputs=[])
        return {"vs": vs, "ps": description("ps", **ps)}

    def test_every_resource_has_its_own_place(self):
        res = self.pair(cbufs=[(14, 7)], textures=[(5, 0x10, "2d", "push_slot"), (6, 0x12, "2d", "push_slot")])
        out = first_draw.bind(res, self.banks, self.mem, self.textures, 64, 32)
        buffers = [(k, b, st) for k, b, st, *_ in (o for o in out if o[0] in ("uniform", "storage"))]
        self.assertEqual(buffers, [("uniform", 0, "vf"), ("uniform", 1, "v"), ("uniform", 2, "v"), ("storage", 9, "v"),
                                   ("uniform", 14, "f")])
        self.assertEqual(len({b for _, b, _ in buffers}), len(buffers))  # disjoint, the support buffer shared
        by = {(o[0], o[1]): o for o in out}
        self.assertEqual(by[("uniform", 2)][3][:4], struct.pack("<f", 2.0))  # c4 at its slot
        # The storage buffer is bound at its address rounded down to 16 bytes, its size plus what was rounded off.
        self.assertEqual(by[("storage", 9)][3], struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
        # Each texture an image and a sampler entry of its own; its slot holds the handle (ID | sampler ID << 20).
        pushes = sorted((o[1], o[2]) for o in out if o[0] == "push")
        self.assertEqual(pushes, [(16, 1 | 1 << 20), (36, 2 | 2 << 20), (40, 3 | 3 << 20)])
        self.assertEqual(sorted(o[1] for o in out if o[0] == "image"), [1, 2, 3])
        self.assertIn(("global", 33025, 4096), out)

    def test_what_the_layout_does_not_hold_is_refused(self):
        for ps, needle in ((dict(schema=1), "schema 1"), (dict(fits=False), "does not fit"),
                           (dict(textures=[(5, 0x10, "2d", "pool")]), "pool texture"),
                           (dict(images=[{"binding": 30}]), "storage images"),
                           (dict(textures=[(5, 0x14, "2d", "push_slot")]), "no spec entry"),
                           (dict(textures=[(5, 0x12, "3d", "push_slot")]), "3d in the description")):
            with self.subTest(needle=needle), self.assertRaises(first_draw.InputError) as cm:
                first_draw.bind(self.pair(**ps), self.banks, self.mem, self.textures, 64, 32)
            self.assertIn(needle, str(cm.exception))

    def test_a_module_with_raw_global_access_is_refused(self):
        # No buffer is listed for raw access any more: the description's raw_access is the only sign, and this harness has no table to give
        for flag in ("page_table", "physical_pointers"):
            with self.subTest(flag=flag), self.assertRaises(first_draw.InputError) as cm:
                res = self.pair()
                res["ps"]["raw_access"] = {"physical_pointers": False, "page_table": False, flag: True}
                first_draw.bind(res, self.banks, self.mem, self.textures, 64, 32)
            self.assertIn("raw global access", str(cm.exception))

    def test_vertex_inputs_come_from_the_description(self):
        vs = description("vs", inputs=[{"location": 0, "component": 0, "components": 4, "type": "float32"},
                                       {"location": 2, "component": 0, "components": 2, "type": "sint32"}])
        verts, blob, lines = first_draw.vertex_stream({"vertex_stream": {"vertices": [[1, 2, 3, 4, -1, 7]]}}, vs)
        self.assertEqual(lines, ["attribute 0 rgba32f 0", "attribute 2 rg32i 16"])
        self.assertEqual(blob, struct.pack("<4f2i", 1, 2, 3, 4, -1, 7))
        self.assertEqual(verts[0][0x80], f2b(1.0))
        self.assertEqual((verts[0][0xA0], verts[0][0xA4], verts[0][0xA8], verts[0][0xAC]), (0xFFFFFFFF, 7, 0, 1))
        with self.assertRaises(first_draw.InputError):
            first_draw.vertex_stream({"vertex_stream": {"vertices": [[1, 2]]}}, vs)
        with self.assertRaises(first_draw.InputError):  # the hand-written spec of spike S2 is no longer read
            first_draw.vertex_stream({"vertex_stream": {"attributes": [], "vertices": []}}, vs)
        self.assertEqual(first_draw.vertex_stream({"vertex_stream": {"vertices": [[], []]}}, description("vs", inputs=[]))[1], b"")

    def test_textures_by_stage_and_type(self):
        specs = first_draw.texture_specs({"textures": [{"handle": "0x10", "width": 1, "height": 1},
                                                       {"stage": "ps", "handle": "0x12", "type": "cube", "width": 2, "height": 2}]})
        self.assertEqual((sorted(specs["vs"]), sorted(specs["ps"])), ([0x10], [0x10, 0x12]))
        cube = first_draw.make_texture(specs["ps"][0x12])
        self.assertEqual((cube.kind, cube.layers, len(cube.texels)), ("cube", 6, 24))
        set_rule = first_draw.make_texture({"width": 4, "height": 4, "set": [
            {"channel": 0, "value": 0.0, "block": 2, "every": 2, "invert": True}, {"channel": 2, "value": 0.5}]})
        reds = [t[0] for t in set_rule.texels]
        self.assertEqual([r != 0 for r in reds[:4]], [True, True, False, False])  # kept where (x//2 + y//2) % 2 == 0
        self.assertTrue(all(t[2] == 0.5 for t in set_rule.texels))

    def test_c1_is_the_programs_own_bank(self):
        progs = {"vs": {"c1": b""}, "ps": {"c1": struct.pack("<2I", 0x2C80, 0x3790)}}
        banks = first_draw.stage_banks({"ps": {"7": {"0x0": {"f32": [1.0]}}}}, progs)
        self.assertEqual(struct.unpack_from("<2I", banks["ps"][1]), (0x2C80, 0x3790))
        self.assertEqual(len(banks["ps"][1]), first_draw.BANK_BYTES)
        self.assertNotIn(1, banks["vs"])
        with self.assertRaises(first_draw.InputError):
            first_draw.stage_banks({"ps": {"1": {"0x0": {"u32": [0]}}}}, progs)

    def test_memory_words_mix_floats_and_integers(self):
        m = first_draw.Memory([{"address": 0x100, "words": [1.5, "0x64"]}])
        self.assertEqual(m.load(0x100, 8), struct.pack("<fI", 1.5, 100))
        with self.assertRaises(first_draw.InputError):  # a scenario's region over the spec's would read two ways
            first_draw.Memory([{"address": 0x100, "f32": [1.0, 2.0]}, {"address": 0x104, "f32": [3.0]}])


class CandidateTests(unittest.TestCase):
    """The pairs a full-binding draw may use (tools/firstdraw/candidates.py)."""

    def module(self, ubo: int, ssbo: int, tex: int, **extra) -> dict:
        res = {"schema": 2, "spv_bytes": 100, "cbufs": [{}] * ubo, "sbufs": [{}] * ssbo, "images": [],
               "textures": [{"set": 0, "via": "push_slot", "array_length": 1}] * tex
                           + [{"set": 0, "via": "pool", "array_length": 33025}],
               "raw_access": {"physical_pointers": False, "page_table": False}, "writable_storage": False}
        res.update(extra)
        return candidates.summary(res)

    def test_a_pair_qualifies_over_32_with_both_kinds_in_both_stages(self):
        vs, ps = self.module(4, 3, 1), self.module(7, 2, 15)
        self.assertEqual((vs["tex"], candidates.s2_count(vs) + candidates.s2_count(ps) - 1), (1, 33))  # pool left out
        self.assertTrue(candidates.qualifies(vs, ps))
        self.assertFalse(candidates.qualifies(vs, self.module(7, 2, 14)))  # 32: fits one push set
        self.assertFalse(candidates.qualifies(self.module(4, 0, 1), self.module(7, 5, 15)))  # no storage in the vertex stage
        self.assertFalse(candidates.qualifies(vs, self.module(7, 2, 15, writable_storage=True)))
        self.assertFalse(candidates.qualifies(vs, self.module(7, 2, 15, raw_access={"page_table": True})))

    def test_schema_1_counts_set_0_textures(self):
        res = {"spv_bytes": 1, "cbufs": [], "sbufs": [], "images": [],
               "textures": [{"set": 0, "array_length": 4}, {"set": 1, "array_length": 33025}]}
        self.assertEqual((candidates.summary(res)["tex"], candidates.summary(res)["tex_elements"]), (1, 4))

    def test_keys_stay_outside_the_repository(self):
        self.assertTrue(candidates.inside_repo(ROOT / "build" / "x.jsonl"))
        self.assertEqual(candidates.main(["--split", "x", "--pass-dir", "y", "--out", str(ROOT / "c.jsonl")]), 2)
        self.assertFalse((ROOT / "c.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
