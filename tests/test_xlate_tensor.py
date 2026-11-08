# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the translator's cleanup of tensor code: HMMA with an f16 accumulator, and what feeds it.

HMMA.16816 becomes one operation with two results (the two packed words of D). The cleanup
loop of the vendored optimizer removes an operation whose results nobody reads, and, when it
removes a reader, lets the producers of that reader's sources go once they have no reader left.
It used to do the second step for an operation whose other result was still read: a multiply
that stays in the module lost its claim on its sources, and the loads that fed it (the shared
memory loads of LDS and LDSM) and the HMMA before it were deleted as dead. The module kept one
HMMA that read values nothing defined.

The programs are assembled here from instruction words, never read from game files. Each
translation is counted by tools/shaders/spvstats.py. The helper that stands in for an HMMA
reads a scratch array of its own in workgroup memory, so the loads of the program are the
module's workgroup loads minus those of a control program with the same HMMA and no loads of
its own. The tests skip when the translator has not been built
(py -3 tools/build_xlate.py), unless OOT2_REQUIRE_XLATE is set.

Run: python -m unittest discover -s tests -p test_xlate_tensor.py -v
"""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "shaders"))

import spvstats  # noqa: E402
import validate  # noqa: E402
from test_xlate_host import Harness, SPIRV_VAL, ampere, bit, compute_job, EXIT  # noqa: E402
from test_xlate_phi import s2r  # noqa: E402
from test_xlate_raw import TID_X, scheduled  # noqa: E402
from test_xlate_optimizer import lds, sts  # noqa: E402

RZ = 255
A, B = 16, 32        # first register of the A fragment (4 registers) and of the B fragment (2)
D = 40               # the registers D is written to
X = 10               # the thread's x


def hmma(rd: int, ra: int, rb: int, rc: int) -> bytes:
    """HMMA.16816 Rd, Ra, Rb, Rc with f16 sources and an f16 accumulator (size bit 75, no f32 bit 76)."""
    return scheduled(ampere(0x23C, rd=rd, ra=ra, rb=rb, rc=rc, high_bits=bit(75)))


def ldsm(rd: int, ra: int, offset: int = 0, count_log2: int = 2) -> bytes:
    """LDSM.16.M88.<1 << count_log2> Rd.., [R<ra> + offset] (count at bits 72..74, mode 78..80 = 0)."""
    return scheduled(ampere(0x83B, rd=rd, ra=ra, low_bits=(offset & 0xFFFFFF) << 40, high_bits=bit(72, count_log2)))


class TensorCleanupTest(Harness):
    def translate(self, key: str, instructions: list[bytes]) -> bytes:
        """Translate the program with the default job options and return its module; it must validate when spirv-val is present."""
        code = self.dir / f"{key}.bin"
        code.write_bytes(b"".join(instructions) + EXIT)
        got = self.run_jobs([compute_job(key, code, shared=256)])[key]
        self.assertEqual((got.status, got.message), ("ok", ""), key)
        path = self.out / f"{key}.spv"
        if SPIRV_VAL:
            v = validate.validate_one(SPIRV_VAL, key, path)
            self.assertTrue(v.ok, v.message)
        return path.read_bytes()

    def workgroup_loads(self, key: str, instructions: list[bytes]) -> int:
        return spvstats.analyze(self.translate(key, instructions))["loads"]["Workgroup"]

    def helper_loads(self) -> int:
        """Workgroup loads of one HMMA with no loads of the program's own: what the helper's body reads."""
        loads = self.workgroup_loads("control", [s2r(X, TID_X), hmma(D, RZ, RZ, RZ), sts(RZ, D + 1, 0)])
        self.assertGreater(loads, 0, "the control program must carry the helper")
        return loads

    def helper_calls(self, module: bytes) -> int:
        """Calls of the HMMA helper, the only function the translator gives eleven arguments here."""
        words = struct.unpack(f"<{len(module) // 4}I", module)
        i, calls = 5, 0
        while i < len(words):
            length, op = words[i] >> 16, words[i] & 0xFFFF
            if op == 57 and length - 4 == 11:      # OpFunctionCall: result type, result id, function, then the arguments
                calls += 1
            i += length
        return calls

    def fragments(self) -> list[bytes]:
        """The thread's x stored, then read back as the A fragment (four words) and the B fragment (two)."""
        return [s2r(X, TID_X), sts(RZ, X, 0)] + [lds(A + i, RZ, 4 * i) for i in range(4)] + [lds(B + i, RZ, 16 + 4 * i) for i in range(2)]

    # -- the shape that lost its inputs ---------------------------------------------------------------

    def test_loads_feeding_an_hmma_whose_first_result_is_unused_stay(self):
        """R40 is written and never read, R41 is stored: the multiply keeps one result, and its inputs must stay."""
        base = self.helper_loads()
        loads = self.workgroup_loads("lost", self.fragments() + [hmma(D, A, B, RZ), sts(RZ, D + 1, 64)])
        self.assertEqual(loads - base, 6)

    def test_an_hmma_feeding_a_second_one_stays(self):
        """The first multiply gives the accumulator of the second, whose first result is unused."""
        module = self.translate("chain", self.fragments() + [hmma(D, A, B, RZ), hmma(D + 2, A, B, D), sts(RZ, D + 3, 64)])
        self.assertEqual(self.helper_calls(module), 2)
        self.assertEqual(spvstats.analyze(module)["loads"]["Workgroup"] - self.helper_loads(), 6)

    # -- shapes that were never broken ------------------------------------------------------------------

    def test_loads_feeding_an_hmma_whose_results_are_both_read_stay(self):
        base = self.helper_loads()
        loads = self.workgroup_loads("both", self.fragments() + [hmma(D, A, B, RZ), sts(RZ, D, 64), sts(RZ, D + 1, 68)])
        self.assertEqual(loads - base, 6)

    def test_a_load_nobody_reads_is_still_removed(self):
        """The cleanup keeps removing what is dead: seven loads, one of them never read, leave six."""
        base = self.helper_loads()
        loads = self.workgroup_loads("dead", self.fragments() + [lds(30, RZ, 20), hmma(D, A, B, RZ), sts(RZ, D + 1, 64)])
        self.assertEqual(loads - base, 6)


if __name__ == "__main__":
    unittest.main()
