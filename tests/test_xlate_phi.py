# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for a global access whose address is a phi of a traced pointer and a pointer the shader loaded (the mixed phi fix).

The translator traces a global address to the constant buffer word pair that holds a storage
buffer's base. When the address is a phi, it walks the sources, and a helper checks the address
against each traced buffer in turn and gives a load zero when the address is in none of them.
A source that traces to nothing, such as a pointer loaded through another buffer, used to be left
out of that list. A phi of a traced pointer and a loaded pointer then got a helper over the traced
buffer alone, and every access through the loaded pointer read zero. Such an address now takes the
raw helper, the page-table walk of architecture D15, as a select already did. A phi whose sources
all trace, or whose other sources hold no pointer at all (a register no path wrote, a constant),
still takes the buffer helper.

The programs are assembled here from instruction words. A buffer is a word pair of c[0] that a MOV
copies to a register pair; the predicate comes from ISETP on the thread's x, so a guarded
instruction makes a branch the translator cannot fold, and the register it writes becomes a phi at
the join. The tests skip when the translator has not been built (py -3 tools/build_xlate.py),
unless OOT2_REQUIRE_XLATE is set.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import struct
import unittest

from raw_abi import A
from test_xlate_host import Harness, ampere, bit, compute_job, EXIT, SPIRV_VAL, validate
from test_xlate_raw import Memory, Module, PageTable, TID_X, run, scheduled
from test_xlate_uni import mov_cbuf

B32 = 4
M32 = 0xFFFFFFFF

# c[0] byte offsets of three storage buffers' address pairs (base low, base high), each 0x10 apart
BUF_A, BUF_B, BUF_C = 0x100, 0x110, 0x120
TID, VALUE, PTR_LO, PTR_HI, TABLE_LO, TABLE_HI, RESULT = 10, 7, 2, 3, 4, 5, 6


def s2r(rd: int, sreg: int) -> bytes:
    return scheduled(ampere(0x919, rd=rd, ra=0, rb=0, rc=0, high_bits=bit(72, sreg)))


def isetp_ne_zero(ra: int) -> bytes:
    """ISETP.NE.U32 P0, PT, R<ra>, RZ, PT: opcode 0x00c in the register-register form (bits 9..11 = 1), the comparison
    at 76..78 (5: not equal), the destination predicates at 81..83 and 84..86 (P0, PT), the source predicate at 87..89."""
    return scheduled(ampere(0x20C, ra=ra, rb=255, high_bits=bit(76, 5) | bit(81, 0) | bit(84, 7) | bit(87, 7)))


def guard(ins: bytes, pred: int = 0, negate: bool = False) -> bytes:
    """The instruction under @P<pred> (@!P<pred>): the guard is bits 12..14 and its negation bit 15."""
    lo, hi = struct.unpack("<QQ", ins)
    lo = (lo & ~(0xF << 12)) | (pred << 12) | (int(negate) << 15)
    return struct.pack("<QQ", lo, hi)


def mov_zero(rd: int) -> bytes:
    """MOV Rd, 0: MOV (0x002) in the immediate form (bits 9..11 = 4), the immediate in bits 32..63."""
    return scheduled(ampere(0x802, rd=rd, ra=0, rb=0))


def ldg(rd: int, ra: int, offset: int = 0) -> bytes:
    """LDG.E Rd, [R<ra>.64 + offset], 32 bits; the uniform register field (32..37) names UR63, which is zero."""
    return scheduled(ampere(0x381, rd=rd, ra=ra, rb=255, rc=0, low_bits=(offset & 0xFFFFFF) << 40, high_bits=bit(72) | bit(73, B32)))


def stg(ra: int, rb: int) -> bytes:
    """STG.E [R<ra>.64], R<rb>, 32 bits."""
    return scheduled(ampere(0x386, ra=ra, rb=rb, rc=0, high_bits=bit(72) | bit(73, B32)))


def sts(rb: int) -> bytes:
    """STS [RZ], R<rb>: keeps a loaded value from being dead code (the job declares shared memory)."""
    return scheduled(ampere(0x388, ra=255, rb=rb))


def pair(rd: int, offset: int) -> list[bytes]:
    return [mov_cbuf(rd, 0, offset), mov_cbuf(rd + 1, 0, offset + 4)]


def loaded_pair(rd: int, table: int, guarded: bool = True) -> list[bytes]:
    """The register pair rd as the 64-bit pointer stored in the buffer whose base is in R<table>, read under @P0."""
    loads = [ldg(rd, table, 0), ldg(rd + 1, table, 4)]
    return [guard(i) for i in loads] if guarded else loads


def prologue() -> list[bytes]:
    """The thread's x in R10, P0 = (x != 0), buffer A's base in R2:R3 (the pointer on the path where P0 is false) and buffer B's in R4:R5."""
    return [s2r(TID, TID_X), isetp_ne_zero(TID)] + pair(PTR_LO, BUF_A) + pair(TABLE_LO, BUF_B)


LOAD = [ldg(RESULT, PTR_LO), sts(RESULT)]
STORE = [stg(PTR_LO, TID)]


class PhiAddressTest(Harness):
    def translate(self, key: str, instructions: list[bytes]) -> tuple[Module, dict]:
        code = self.dir / f"{key}.bin"
        code.write_bytes(b"".join(instructions) + EXIT)
        got = self.run_jobs([compute_job(key, code, shared=64)])[key]
        self.assertEqual((got.status, got.message), ("ok", ""))
        if SPIRV_VAL:
            v = validate.validate_one(SPIRV_VAL, key, self.out / f"{key}.spv")
            self.assertTrue(v.ok, v.message)
        return Module((self.out / f"{key}.spv").read_bytes()), json.loads((self.out / f"{key}.res.json").read_text(encoding="utf-8"))

    def buffers(self, res: dict) -> list[int]:
        """The c[0] byte offsets of the storage buffers the module binds, in order."""
        return sorted(s["cb_byte_offset"] for s in res["sbufs"])

    def assert_raw(self, module: Module, res: dict, params: int):
        ra = res["raw_access"]
        self.assertEqual((ra["physical_pointers"], ra["page_table"]), (True, True), "the access did not go through the page table")
        helpers = module.raw_helpers()
        self.assertEqual(len(helpers), 1, "one raw helper for the one access")
        self.assertEqual(len(helpers[0]["params"]), params)
        calls = [w for f in module.functions.values() for name, w in f["code"] if name == "FunctionCall" and w[2] == helpers[0]["id"]]
        self.assertEqual(len(calls), 1, "the phi's access calls the raw helper")
        return helpers[0]

    def assert_buffers(self, module: Module, res: dict, offsets: list[int]):
        ra = res["raw_access"]
        self.assertEqual((ra["physical_pointers"], ra["page_table"], ra["miss_log"]), (False, False, False), "the access went through the page table")
        self.assertEqual(module.raw_helpers(), [])
        self.assertEqual(self.buffers(res), offsets)

    def test_a_phi_of_a_traced_and_a_loaded_pointer_loads_through_the_page_table(self):
        # R2:R3 is buffer A's base where P0 is false and the pointer stored at the start of buffer B where P0 is true
        module, res = self.translate("mixed_load", prologue() + loaded_pair(PTR_LO, TABLE_LO) + LOAD)
        helper = self.assert_raw(module, res, 2)
        # Only the buffer the pointer is loaded from stays bound: a helper over A's range would have read zero at any
        # address outside A, which is where the loaded pointer points
        self.assertEqual(self.buffers(res), [BUF_B])
        # The helper walks the table: an address the shader loaded reads the mapped page at its delta, not zero
        table = PageTable()
        guest, delta = 0x12_3456_7890 & ~3, 0x7F_0000_1000 - 0x12_3456_7000
        table.map_page(guest, delta)
        mem = Memory(table.address, table.device, {guest + delta: 0xCAFEF00D}, table.push)
        self.assertEqual(run(module, helper, [guest & M32, guest >> 32], mem), 0xCAFEF00D)
        self.assertEqual(table.misses, 0)

    def test_a_phi_of_a_traced_and_a_loaded_pointer_stores_through_the_page_table(self):
        module, res = self.translate("mixed_store", prologue() + loaded_pair(PTR_LO, TABLE_LO) + STORE)
        self.assert_raw(module, res, 3)
        self.assertEqual(self.buffers(res), [BUF_B])
        self.assertTrue(res["raw_access"]["store_bits"], "a raw store sets its page's bit")

    def test_the_loaded_pointer_may_sit_in_an_inner_phi(self):
        # phi(phi(A, loaded), C): the untraced source is one level down
        module, res = self.translate(
            "nested", prologue() + loaded_pair(PTR_LO, TABLE_LO) + [guard(i, negate=True) for i in pair(PTR_LO, BUF_C)] + LOAD)
        self.assert_raw(module, res, 2)
        self.assertEqual(self.buffers(res), [BUF_B])

    def test_a_phi_of_two_traced_pointers_still_checks_the_two_buffers(self):
        module, res = self.translate("two_traced", prologue() + [guard(i) for i in pair(PTR_LO, BUF_C)] + LOAD)
        self.assert_buffers(module, res, [BUF_A, BUF_C])

    def test_a_source_that_holds_no_pointer_is_left_out_as_before(self):
        # phi(phi(A, C), 0): the zero is a path where no pointer was written, not an address to look up
        module, res = self.translate(
            "with_zero", prologue() + [guard(i) for i in pair(PTR_LO, BUF_C)] + [guard(mov_zero(PTR_LO), negate=True), guard(mov_zero(PTR_HI), negate=True)] + LOAD)
        self.assert_buffers(module, res, [BUF_A, BUF_C])

    def test_a_single_traced_pointer_is_not_a_phi_and_stays_inline(self):
        module, res = self.translate("one_traced", prologue() + LOAD)
        self.assert_buffers(module, res, [BUF_A])


if __name__ == "__main__":
    unittest.main()
