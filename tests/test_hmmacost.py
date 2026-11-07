# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/hmmacost/hmma_cost.py: the instruction words it assembles, its CPU model of
HMMA, LDSM and MOVM, the exact inputs of the cooperative matrix kernel, and the counter-run
mutations of translated modules.

The model is checked against matrices built here and scattered to lanes by this file's own
reading of NVIDIA's fragment layouts (PTX mma.m16n8k16, ldmatrix, movmatrix), so a layout error
in the model would have to be made twice. The tests that translate programs skip when the
translator has not been built (py -3 tools/build_xlate.py), unless OOT2_REQUIRE_XLATE is set;
the GLSL kernels compile only where glslangValidator is (the Vulkan SDK). No GPU, no game data.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "hmmacost"))

import hmma_cost as hc  # noqa: E402


def field(ins: bytes, lo: int, hi: int) -> int:
    v = int.from_bytes(ins, "little")
    return (v >> lo) & ((1 << (hi - lo)) - 1)


# --- fragments, by this file's reading of the PTX layouts ---------------------------------------------
# Lane L, g = L / 4, t = L % 4. A (16 x 16): register r holds (row g + 8 (r % 2), columns 8 (r / 2) + 2t,
# + 1). B (16 x 8, k by n): register r holds (k 8 r + 2t, + 1; column g). C / D f16: register r holds
# (row g + 8 r, columns 2t, 2t + 1); f32: register r holds (row g + 8 (r / 2), column 2t + r % 2).

def words_of(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo.astype(np.float16).view(np.uint16).astype(np.uint32) | (hi.astype(np.float16).view(np.uint16).astype(np.uint32) << 16)


def scatter_a(A: np.ndarray) -> list[np.ndarray]:
    out = []
    for r in range(4):
        regs = np.zeros(32, np.uint32)
        for L in range(32):
            g, t = divmod(L, 4)
            row, col = g + 8 * (r % 2), 8 * (r // 2) + 2 * t
            regs[L] = words_of(np.array([A[row, col]]), np.array([A[row, col + 1]]))[0]
        out.append(regs)
    return out


def scatter_b(B: np.ndarray) -> list[np.ndarray]:
    out = []
    for r in range(2):
        regs = np.zeros(32, np.uint32)
        for L in range(32):
            g, t = divmod(L, 4)
            k = 8 * r + 2 * t
            regs[L] = words_of(np.array([B[k, g]]), np.array([B[k + 1, g]]))[0]
        out.append(regs)
    return out


def scatter_c16(C: np.ndarray) -> list[np.ndarray]:
    out = []
    for r in range(2):
        regs = np.zeros(32, np.uint32)
        for L in range(32):
            g, t = divmod(L, 4)
            regs[L] = words_of(np.array([C[g + 8 * r, 2 * t]]), np.array([C[g + 8 * r, 2 * t + 1]]))[0]
        out.append(regs)
    return out


def gather_c16(regs: list[np.ndarray]) -> np.ndarray:
    D = np.zeros((16, 8), np.float32)
    for r in range(2):
        for L in range(32):
            g, t = divmod(L, 4)
            D[g + 8 * r, 2 * t] = np.uint16(regs[r][L] & 0xFFFF).view(np.float16)
            D[g + 8 * r, 2 * t + 1] = np.uint16(regs[r][L] >> 16).view(np.float16)
    return D


def scatter_c32(C: np.ndarray) -> list[np.ndarray]:
    out = []
    for r in range(4):
        regs = np.zeros(32, np.uint32)
        for L in range(32):
            g, t = divmod(L, 4)
            regs[L] = np.float32(C[g + 8 * (r // 2), 2 * t + r % 2]).view(np.uint32)
        out.append(regs)
    return out


def gather_c32(regs: list[np.ndarray]) -> np.ndarray:
    D = np.zeros((16, 8), np.float32)
    for r in range(4):
        for L in range(32):
            g, t = divmod(L, 4)
            D[g + 8 * (r // 2), 2 * t + r % 2] = regs[r][L].view(np.float32)
    return D


def small_halves(rng, shape, lo=-3, hi=0) -> np.ndarray:
    return (2.0 ** rng.uniform(lo, hi, shape) * rng.choice([-1.0, 1.0], shape)).astype(np.float16).astype(np.float32)


def one_warp(regs_by_number: dict[int, np.ndarray]) -> dict:
    regs = {r: np.zeros(32, np.uint32) for r in range(256)}
    for r, v in regs_by_number.items():
        regs[r] = v.copy()
    return regs


class Encoding(unittest.TestCase):
    def test_common_fields_and_scheduling(self):
        ins = hc.i_hmma(40, 16, 32, 40, f32=False)
        self.assertEqual(len(ins), 16)
        self.assertEqual(field(ins, 0, 12), 0x23C)
        self.assertEqual(field(ins, 12, 15), 7)            # guard PT
        self.assertEqual((field(ins, 16, 24), field(ins, 24, 32), field(ins, 32, 40), field(ins, 64, 72)), (40, 16, 32, 40))
        self.assertEqual(field(ins, 104, 116), 0xFE2)      # stall 1, no barriers
        self.assertEqual(field(ins, 113, 116), 7)          # read barrier field
        self.assertEqual(hc.EXIT[:2], struct.pack("<H", 0x794D))

    def test_hmma_shape_and_accumulator(self):
        for f32 in (False, True):
            ins = hc.i_hmma(40, 16, 32, 40, f32=f32)
            size = field(ins, 75, 76) | (field(ins, 78, 80) << 1)
            self.assertEqual(size, 1)                       # m16n8k16
            self.assertEqual(field(ins, 76, 77), int(f32))
            self.assertEqual(field(ins, 82, 85), 0)         # f16 sources

    def test_ldsm_count_mode_and_offset(self):
        ins = hc.i_ldsm4(44, 14, 0x600, trans=True)
        self.assertEqual(field(ins, 0, 12), 0x83B)
        self.assertEqual(1 << field(ins, 72, 74), 4)
        self.assertEqual(field(ins, 78, 80), 1)
        self.assertEqual(field(ins, 40, 64), 0x600)
        self.assertEqual(field(ins, 91, 92), 0)
        self.assertEqual(field(hc.i_ldsm4(44, 14, 0, trans=False), 78, 80), 0)

    def test_immediate_forms_keep_the_immediate_whole(self):
        # The B register field shares bits 32..39 with the immediate: it must not be OR-ed in.
        for ins, imm in ((hc.i_imad(10, 1, 128, 0), 128), (hc.i_imad_wide(6, 0, 160, 2), 160),
                         (hc.i_xor_imm(60, 16, 0x00110011), 0x00110011)):
            self.assertEqual(field(ins, 32, 64), imm)
            self.assertEqual(field(ins, 9, 12), 4)          # operand form 4: an immediate
            self.assertEqual(field(ins, 81, 84), 7)         # no predicate output
        self.assertEqual(field(hc.i_xor_imm(60, 16, 1), 72, 80), 0x3C)
        self.assertEqual(field(hc.i_xor_reg(40, 40, 44), 9, 12), 1)

    def test_memory_forms(self):
        ldg, stg, sts = hc.i_ldg128(16, 6, 32), hc.i_stg128(8, 40, 16), hc.i_sts128(11, 16, 48)
        self.assertEqual((field(ldg, 0, 12), field(ldg, 72, 73), field(ldg, 73, 76), field(ldg, 40, 64)), (0x381, 1, 6, 32))
        self.assertEqual((field(stg, 0, 12), field(stg, 32, 40), field(stg, 40, 64)), (0x386, 40, 16))
        self.assertEqual((field(sts, 0, 12), field(sts, 73, 76), field(sts, 78, 80)), (0x388, 6, 0))
        self.assertEqual(field(hc.i_mov_c0(2, 0x110), 38, 54), 0x110)
        self.assertEqual(field(hc.i_s2r(0, hc.SR_TID_X), 72, 80), 33)

    def test_masks_touch_only_low_mantissa_bits(self):
        seen = set()
        for i in range(512):
            for salt in (hc.SALT_A, hc.SALT_B):
                m = hc.mask(i, salt)
                self.assertNotEqual(m, 0)
                self.assertEqual(m & ~0x001F001F, 0)
                self.assertEqual(m & 0xFFFF, m >> 16)
            seen.add(hc.mask(i, hc.SALT_A))
        self.assertGreater(len(seen), 20)

    def test_the_glsl_kernel_computes_the_same_masks(self):
        text = (ROOT / "tools" / "hmmacost" / "coop_helper.comp").read_text(encoding="utf-8")
        self.assertIn("(((i + 1u) * salt) >> 11) & 0x1Fu", text)
        self.assertIn(f"{hc.SALT_A:#010X}u".replace("0X", "0x"), text)
        self.assertIn(f"{hc.SALT_B:#010X}u".replace("0X", "0x"), text)


class Model(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.w = slice(0, 32)

    def test_hmma_f32_matches_a_matrix_product(self):
        A, B = small_halves(self.rng, (16, 16)), small_halves(self.rng, (16, 8))
        C = self.rng.uniform(-1, 1, (16, 8)).astype(np.float32)
        regs = one_warp({**dict(enumerate(scatter_a(A), 16)), **dict(enumerate(scatter_b(B), 32)),
                         **dict(enumerate(scatter_c32(C), 40))})
        hc.warp_hmma(regs, self.w, 40, 16, 32, 40, True)
        D = gather_c32([regs[40 + r] for r in range(4)])
        want = C.astype(np.float64) + A.astype(np.float64) @ B.astype(np.float64)
        self.assertLess(float(np.max(np.abs(D - want))), 1e-5)

    def test_hmma_f32_adds_in_k_order_with_one_rounding_each(self):
        A, B = small_halves(self.rng, (16, 16)), small_halves(self.rng, (16, 8))
        C = self.rng.uniform(-1, 1, (16, 8)).astype(np.float32)
        regs = one_warp({**dict(enumerate(scatter_a(A), 16)), **dict(enumerate(scatter_b(B), 32)),
                         **dict(enumerate(scatter_c32(C), 40))})
        hc.warp_hmma(regs, self.w, 40, 16, 32, 40, True)
        want = C.copy()
        for k in range(16):
            want = (want + np.float32(1) * (A[:, k:k + 1] * B[k:k + 1, :])).astype(np.float32)
        np.testing.assert_array_equal(gather_c32([regs[40 + r] for r in range(4)]), want)

    def test_hmma_f16_rounds_the_result(self):
        A, B, C = small_halves(self.rng, (16, 16)), small_halves(self.rng, (16, 8)), small_halves(self.rng, (16, 8))
        regs = one_warp({**dict(enumerate(scatter_a(A), 16)), **dict(enumerate(scatter_b(B), 32)),
                         **dict(enumerate(scatter_c16(C), 40))})
        hc.warp_hmma(regs, self.w, 40, 16, 32, 40, False)
        want = C.copy()
        for k in range(16):
            want = want + A[:, k:k + 1] * B[k:k + 1, :]
        np.testing.assert_array_equal(gather_c16([regs[40], regs[41]]), want.astype(np.float16).astype(np.float32))

    def test_a_wrong_lane_layout_would_not_pass(self):
        # Transposing A before scattering must change D: the test above depends on the layout.
        A, B, C = small_halves(self.rng, (16, 16)), small_halves(self.rng, (16, 8)), small_halves(self.rng, (16, 8))
        good = one_warp({**dict(enumerate(scatter_a(A), 16)), **dict(enumerate(scatter_b(B), 32)), **dict(enumerate(scatter_c16(C), 40))})
        bad = one_warp({**dict(enumerate(scatter_a(A.T.copy()), 16)), **dict(enumerate(scatter_b(B), 32)), **dict(enumerate(scatter_c16(C), 40))})
        hc.warp_hmma(good, self.w, 40, 16, 32, 40, False)
        hc.warp_hmma(bad, self.w, 40, 16, 32, 40, False)
        self.assertFalse(np.array_equal(good[40], bad[40]))

    def test_f32_chains_skip_the_rounding_between_two_hmma(self):
        A, B, C = small_halves(self.rng, (16, 16)), small_halves(self.rng, (16, 8), -9, -6), small_halves(self.rng, (16, 8))
        start = {**dict(enumerate(scatter_a(A), 16)), **dict(enumerate(scatter_b(B), 32)), **dict(enumerate(scatter_c16(C), 40))}
        console, kept = one_warp(start), one_warp(start)
        chain: dict = {}
        for _ in range(2):
            hc.warp_hmma(console, self.w, 40, 16, 32, 40, False)
            hc.warp_hmma(kept, self.w, 40, 16, 32, 40, False, chain=chain)
        exact = C.copy()
        for _ in range(2):
            for k in range(16):
                exact = exact + A[:, k:k + 1] * B[k:k + 1, :]
        np.testing.assert_array_equal(gather_c16([kept[40], kept[41]]), exact.astype(np.float16).astype(np.float32))
        self.assertFalse(np.array_equal(console[40], kept[40]) and np.array_equal(console[41], kept[41]))

    def test_ldsm_normal_and_transposed(self):
        shared = np.zeros(4096, np.uint8)
        halves = shared.view(np.uint16)
        halves[:] = np.arange(halves.size, dtype=np.uint16)
        rows = (np.arange(32) * 16 + 512).astype(np.uint32)       # lane i gives row i % 8 of matrix i / 8
        for trans in (False, True):
            regs = one_warp({14: rows})
            hc.warp_ldsm4(regs, self.w, shared, 44, 14, 0, trans)
            for m in range(4):
                for L in range(32):
                    g, t = divmod(L, 4)
                    mat = lambda r, c: int(halves[(512 + (8 * m + r) * 16) // 2 + c])  # noqa: E731
                    lo, hi = (mat(2 * t, g), mat(2 * t + 1, g)) if trans else (mat(g, 2 * t), mat(g, 2 * t + 1))
                    self.assertEqual(int(regs[44 + m][L]), lo | (hi << 16), (trans, m, L))

    def test_movm_transposes_and_twice_is_identity(self):
        src = self.rng.integers(0, 1 << 32, 32, dtype=np.uint64).astype(np.uint32)
        regs = one_warp({40: src})
        hc.warp_movm(regs, self.w, 41, 40)
        hc.warp_movm(regs, self.w, 42, 41)
        np.testing.assert_array_equal(regs[42], src)
        M = np.zeros((8, 8), np.uint32)
        for L in range(32):
            g, t = divmod(L, 4)
            M[g, 2 * t], M[g, 2 * t + 1] = src[L] & 0xFFFF, src[L] >> 16
        for L in range(32):
            g, t = divmod(L, 4)
            self.assertEqual(int(regs[41][L]), int(M[2 * t, g]) | (int(M[2 * t + 1, g]) << 16))


class Programs(unittest.TestCase):
    def test_model_runs_each_program(self):
        inp = hc.lane_input(1)
        for p in (hc.hmma_program(32, False, True), hc.hmma_program(32, True, False), hc.ldsm_program(32), hc.movm_program(32)):
            out = hc.model(p, inp)
            self.assertEqual(out.size, hc.LOCAL * p.out_words, p.name)
            self.assertGreater(int(np.count_nonzero(out)), out.size // 2, p.name)
            self.assertEqual(len(p.binary()), 16 * (len(p.ops) + 1))

    def test_halving_the_count_halves_the_measured_instructions(self):
        for make in (lambda n: hc.hmma_program(n, False, True), hc.ldsm_program, hc.movm_program):
            full, half = make(64), make(32)
            kind = full.kind
            self.assertEqual(sum(op[0].startswith(kind) for op in full.ops), 64)
            self.assertEqual(sum(op[0].startswith(kind) for op in half.ops), 32)

    def test_perturbed_hmma_operands_differ_every_time(self):
        p = hc.hmma_program(64, True, True)
        pairs = [(op[1], op[2], op[3]) for op in p.ops if op[0] == "xor_imm"]
        a_masks = [m for (_, src, m) in pairs if 16 <= src < 32]
        keys = {(src, m) for (_, src, m) in pairs if 16 <= src < 32}
        self.assertEqual(len(a_masks), 4 * 64)
        self.assertGreater(len(keys), 3 * 64)

    def test_ldsm_addresses_never_repeat(self):
        p = hc.ldsm_program(128)
        loads, perm = [], None
        for op in p.ops:
            if op[0] == "xor_imm":
                perm = op[3]
            elif op[0] == "ldsm4":
                loads.append((perm, op[3]))
        self.assertEqual(len(set(loads)), 128)

    def test_direct_inputs_stay_exact_in_f16(self):
        h, f = hc.coop_direct_input(50)
        for f32 in (False, True):
            exact, peak = hc.coop_direct_model(256, f32, h, f)
            self.assertLess(peak, 8.0)
            scaled = exact * 256.0
            np.testing.assert_array_equal(scaled, np.round(scaled))   # multiples of 1/256
            np.testing.assert_array_equal(exact.astype(np.float16).astype(np.float64), exact)

    def test_c0_bank_holds_both_buffers(self):
        bank = hc.c0_bank(5120, 4096)
        self.assertEqual(len(bank), 65536)
        self.assertEqual(struct.unpack_from("<QI", bank, hc.SB_IN), (hc.IN_ADDR, 5120))
        self.assertEqual(struct.unpack_from("<QI", bank, hc.SB_OUT), (hc.OUT_ADDR, 4096))
        self.assertEqual(hc.IN_ADDR % 16, 0)
        self.assertEqual(hc.OUT_ADDR % 16, 0)

    def test_summary_and_the_count_rule(self):
        s = hc.summarise([3.0, 1.0, 2.0, 4.0])
        self.assertEqual((s["min"], s["p50"], s["max"], s["n"]), (1.0, 2.5, 4.0, 4))
        self.assertEqual(hc.instructions_per_dispatch(16384, 128), 16384 * 4 * 128)
        with self.assertRaises(SystemExit):
            hc.main(["run", "--out", "x", "--count", "100"])


XLATE = hc.find_xlate()
# The Vulkan SDK's glslang only: an older one on a runner's PATH may lack GL_KHR_cooperative_matrix.
GLSLANG = hc.find_sdk_tool("glslangValidator") if os.environ.get("VULKAN_SDK") else None


class Translated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if XLATE is None:
            if os.environ.get("OOT2_REQUIRE_XLATE"):
                raise RuntimeError("OOT2_REQUIRE_XLATE is set but oot2-xlate is not built under build/xlate")
            raise unittest.SkipTest("oot2-xlate is not built (run: py -3 tools/build_xlate.py)")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def names(self, spv: bytes) -> set[str]:
        words = struct.unpack(f"<{len(spv) // 4}I", spv)
        out, i = set(), 5
        while i < len(words):
            wc, op = words[i] >> 16, words[i] & 0xFFFF
            if op == 5:
                out.add(struct.pack(f"<{wc - 2}I", *words[i + 2:i + wc]).split(b"\0")[0].decode())
            i += wc
        return out

    def opcodes(self, spv: bytes) -> list[int]:
        words = struct.unpack(f"<{len(spv) // 4}I", spv)
        ops, i = [], 5
        while i < len(words):
            ops.append(words[i] & 0xFFFF)
            i += words[i] >> 16
        return ops

    def test_both_buffers_bind_as_storage_buffers(self):
        for p in (hc.hmma_program(32, False, True), hc.hmma_program(32, True, True), hc.ldsm_program(32), hc.movm_program(32)):
            spv, res = hc.translate(XLATE, p, self.dir)
            self.assertEqual(sorted(b["cb_byte_offset"] for b in res["sbufs"]), [hc.SB_IN, hc.SB_OUT], p.name)
            self.assertFalse(res["raw_access"]["physical_pointers"], p.name)
            self.assertEqual([c["slot"] for c in res["cbufs"]], [0], p.name)

    def test_the_two_hmma_forms(self):
        scratch, _ = hc.translate(XLATE, hc.hmma_program(32, False, True), self.dir)
        shuffle, _ = hc.translate(XLATE, hc.hmma_program(32, True, True), self.dir)
        self.assertIn("dmsh_hmma16816_f16", self.names(scratch.read_bytes()))
        self.assertNotIn("dmsh_hmma16816_f16", self.names(shuffle.read_bytes()))
        self.assertGreaterEqual(self.opcodes(shuffle.read_bytes()).count(345), 32 * 32)   # 32 shuffles an HMMA

    def test_mutations_change_the_module(self):
        scratch, _ = hc.translate(XLATE, hc.hmma_program(32, False, True), self.dir)
        shuffle, _ = hc.translate(XLATE, hc.hmma_program(32, True, True), self.dir)
        for spv, mutate, changed in ((shuffle, hc.swap_shuffle_lanes, 1), (scratch, hc.swap_helper_stores, 2)):
            before = spv.read_bytes()
            after = mutate(before)
            self.assertEqual(len(after), len(before))
            diff = sum(a != b for a, b in zip(struct.unpack(f"<{len(before) // 4}I", before),
                                              struct.unpack(f"<{len(after) // 4}I", after)))
            self.assertEqual(diff, changed, mutate.__name__)
        with self.assertRaises(RuntimeError):
            hc.swap_helper_stores(shuffle.read_bytes())


@unittest.skipIf(GLSLANG is None, "glslangValidator of a Vulkan SDK (VULKAN_SDK) is not present")
class Kernels(unittest.TestCase):
    def test_kernels_compile_with_every_define(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for n, (src, defines) in enumerate((("coop_direct.comp", {"N": 64}), ("coop_direct.comp", {"N": 64, "ACC_F32": 1}),
                                                ("coop_direct.comp", {"N": 64, "BAD_B_LAYOUT": 1}),
                                                ("coop_helper.comp", {"N": 32, "PERTURB": 1, "BAD_B_LAYOUT": 0}),
                                                ("coop_helper.comp", {"N": 32, "PERTURB": 0, "BAD_B_LAYOUT": 1}))):
                spv = hc.compile_glsl(hc.HERE / src, out / f"{n}.spv", defines)
                self.assertEqual(spv.read_bytes()[:4], bytes.fromhex("03022307"))


if __name__ == "__main__":
    unittest.main()
