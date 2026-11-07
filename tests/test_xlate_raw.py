# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the translator's raw-access helper: the page-table walk of architecture D15.

A global access whose address the translator cannot trace to a constant buffer
goes through a helper function that looks the address up in the page table whose
device address and span are the first four push constant words (two loads: one
word per 2 MiB chunk of the span, then the page's 64-bit delta), then reads or
writes through a physical storage buffer pointer at the address plus the delta.
A store sets its leaf slot's bit in the table; a miss, a load's too, takes an
entry of the table's miss log. The tests translate small programs assembled
here, then run the helper functions of the translator's own SPIR-V on a model
page table with a small evaluator of the instructions the helper uses, so the
walk is checked for what it does, not only for being valid SPIR-V. The table's
layout is read from runtime/include/oot2/raw_table.h (tests/raw_abi.py), which
tests/test_raw_abi.py keeps equal to the translator's copy.

The tests skip when the translator has not been built (py -3 tools/build_xlate.py),
unless OOT2_REQUIRE_XLATE is set.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import struct
import unittest

from raw_abi import A
from test_xlate_host import Harness, ampere, bit, compute_job, EXIT, SPIRV_VAL, validate

CHUNK_LOG2 = A["ChunkLog2"]
PAGE_LOG2 = A["PageLog2"]
LEAF_ENTRIES = 1 << A["LeafEntriesLog2"]

M32, M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF

# --- a SPIR-V function evaluator, for the instructions the raw helper uses ------------------------

OP_NAMES = {
    1: "Undef", 12: "ExtInst", 19: "TypeVoid", 20: "TypeBool", 21: "TypeInt", 22: "TypeFloat", 23: "TypeVector",
    28: "TypeArray", 29: "TypeRuntimeArray", 30: "TypeStruct", 32: "TypePointer", 33: "TypeFunction",
    41: "ConstantTrue", 42: "ConstantFalse", 43: "Constant", 44: "ConstantComposite", 54: "Function",
    55: "FunctionParameter", 56: "FunctionEnd", 57: "FunctionCall", 59: "Variable", 61: "Load", 62: "Store",
    65: "AccessChain", 66: "InBoundsAccessChain", 71: "Decorate", 80: "CompositeConstruct",
    81: "CompositeExtract", 120: "ConvertUToPtr", 124: "Bitcast", 128: "IAdd", 130: "ISub", 132: "IMul",
    164: "LogicalEqual", 165: "LogicalNotEqual", 166: "LogicalOr", 167: "LogicalAnd", 168: "LogicalNot",
    169: "Select", 170: "IEqual", 171: "INotEqual", 172: "UGreaterThan", 173: "SGreaterThan",
    174: "UGreaterThanEqual", 175: "SGreaterThanEqual", 176: "ULessThan", 177: "SLessThan",
    178: "ULessThanEqual", 179: "SLessThanEqual", 194: "ShiftRightLogical", 195: "ShiftRightArithmetic",
    196: "ShiftLeftLogical", 197: "BitwiseOr", 198: "BitwiseXor", 199: "BitwiseAnd", 200: "Not",
    230: "AtomicCompareExchange", 234: "AtomicIAdd", 241: "AtomicOr", 246: "LoopMerge",
    247: "SelectionMerge", 248: "Label", 249: "Branch", 250: "BranchConditional", 253: "Return",
    254: "ReturnValue",
}
DECORATION_BINDING = 33
STORAGE_PUSH_CONSTANT = 9
GLSL_UMIN = 38
STORAGE_PHYSICAL = 5349
FUNCTION_DONT_INLINE = 0x2


def signed(v: int, width: int) -> int:
    return v - (1 << width) if v >> (width - 1) else v


class Module:
    """Types, constants, bindings and functions of a SPIR-V binary."""

    def __init__(self, data: bytes):
        words = struct.unpack(f"<{len(data) // 4}I", data)
        self.types: dict[int, tuple] = {}
        self.consts: dict[int, object] = {}
        self.binding: dict[int, int] = {}
        self.push_vars: set[int] = set()
        self.functions: dict[int, dict] = {}
        cur = None
        i = 5
        while i < len(words):
            wc, op = words[i] >> 16, words[i] & 0xFFFF
            ins = (OP_NAMES.get(op, op), list(words[i + 1:i + wc]))
            name, w = ins
            if name == "TypeInt":
                self.types[w[0]] = ("int", w[1])
            elif name == "TypeFloat":
                self.types[w[0]] = ("float", w[1])
            elif name == "TypeBool":
                self.types[w[0]] = ("bool",)
            elif name == "TypeVector":
                self.types[w[0]] = ("vec", w[1], w[2])
            elif name == "TypePointer":
                self.types[w[0]] = ("ptr", w[1], w[2])
            elif name in ("Constant",):
                width = self.width(w[0])
                self.consts[w[1]] = w[2] | (w[3] << 32 if width == 64 else 0)
            elif name == "ConstantTrue":
                self.consts[w[1]] = True
            elif name == "ConstantFalse":
                self.consts[w[1]] = False
            elif name == "Decorate" and w[1] == DECORATION_BINDING:
                self.binding[w[0]] = w[2]
            elif name == "Variable" and cur is None and w[2] == STORAGE_PUSH_CONSTANT:
                self.push_vars.add(w[1])
            elif name == "Function":
                cur = {"id": w[1], "control": w[2], "params": [], "code": [], "labels": {}}
            elif cur is not None:
                if name == "FunctionParameter":
                    cur["params"].append(w[1])
                elif name == "FunctionEnd":
                    self.functions[cur["id"]] = cur
                    cur = None
                else:
                    if name == "Label":
                        cur["labels"][w[0]] = len(cur["code"])
                    cur["code"].append(ins)
            i += wc

    def width(self, type_id: int) -> int:
        t = self.types.get(type_id, ("other",))
        return t[1] if t[0] in ("int", "float") else 32

    def raw_helpers(self) -> list[dict]:
        """Functions that access memory through a physical storage buffer pointer."""
        return [f for f in self.functions.values() if any(n == "ConvertUToPtr" for n, _ in f["code"])]


class Memory:
    """The host's view: the push constants, and device memory as 32-bit words by address, the page table's words apart from the data's."""

    TABLE_BYTES = 4 * 0x40_0000      # the model table's reach: enough for every layout the tests build

    def __init__(self, table_address: int, table: dict[int, int], data: dict[int, int], push: dict[int, int]):
        self.table_address, self.table, self.data, self.push = table_address, table, data, push
        self.trace: list[tuple] = []

    def in_table(self, address: int) -> bool:
        return self.table_address <= address < self.table_address + self.TABLE_BYTES

    def load(self, ref: tuple) -> int:
        self.trace.append(("load",) + ref)
        if ref[0] == "push":
            return self.push.get(ref[1], 0)
        if ref[1] % 4:
            raise AssertionError(f"misaligned device access at {ref[1]:#x}")
        return (self.table if self.in_table(ref[1]) else self.data).get(ref[1], 0)

    def store(self, ref: tuple, value: int):
        self.trace.append(("store",) + ref + (value,))
        if ref[0] != "dev":
            raise AssertionError("a store to the push constants")
        (self.table if self.in_table(ref[1]) else self.data)[ref[1]] = value & M32


def run(module: Module, func: dict, args: list[int], mem: Memory) -> int | None:
    """Runs one function whose parameters are pointers to 32-bit locals holding args; the push constants are read from mem."""
    vals: dict[int, object] = dict(module.consts)
    cells: dict[int, object] = {}
    for n, (pid, a) in enumerate(zip(func["params"], args)):
        cells[pid] = a & M32
        vals[pid] = ("fn", pid)
    code, labels = func["code"], func["labels"]
    pc = 0
    steps = 0

    def get(i):
        return vals[i]

    def deref_load(ptr):
        if ptr[0] == "fn":
            return cells.get(ptr[1], 0)
        return mem.load(ptr)

    def deref_store(ptr, v):
        if ptr[0] == "fn":
            cells[ptr[1]] = v
        else:
            mem.store(ptr, v)

    while True:
        steps += 1
        if steps > 100000:
            raise AssertionError("the helper did not return")
        name, w = code[pc]
        pc += 1
        if name in ("Label", "SelectionMerge", "LoopMerge"):
            continue
        if name == "Variable":
            vals[w[1]] = ("fn", w[1])
            cells.setdefault(w[1], 0)
        elif name == "Load":
            vals[w[1]] = deref_load(get(w[2]))
        elif name == "Store":
            deref_store(get(w[0]), get(w[1]))
        elif name in ("AccessChain", "InBoundsAccessChain"):
            base = w[2]
            if base not in module.push_vars or get(w[3]) != 0 or len(w) != 6:
                raise AssertionError(f"unexpected access chain {w}")
            vals[w[1]] = ("push", get(w[4]) * 4 + get(w[5]))
        elif name == "Bitcast":
            src = get(w[2])
            t = module.types[w[0]]
            if isinstance(src, tuple) and t[0] == "int" and t[1] == 64:
                vals[w[1]] = src[0] | (src[1] << 32)
            elif t[0] == "vec" and not isinstance(src, tuple):
                vals[w[1]] = (src & M32, src >> 32)
            else:
                vals[w[1]] = src
        elif name == "CompositeConstruct":
            vals[w[1]] = tuple(get(x) for x in w[2:])
        elif name == "CompositeExtract":
            vals[w[1]] = get(w[2])[w[3]]
        elif name == "ConvertUToPtr":
            if module.types[w[0]][1] != STORAGE_PHYSICAL:
                raise AssertionError("pointer to another storage class")
            vals[w[1]] = ("dev", get(w[2]) & M64)
        elif name in ("IAdd", "ISub", "IMul", "BitwiseOr", "BitwiseXor", "BitwiseAnd",
                      "ShiftLeftLogical", "ShiftRightLogical", "ShiftRightArithmetic"):
            width = module.width(w[0])
            mask = (1 << width) - 1
            a, b = get(w[2]), get(w[3])
            r = {"IAdd": lambda: a + b, "ISub": lambda: a - b, "IMul": lambda: a * b,
                 "BitwiseOr": lambda: a | b, "BitwiseXor": lambda: a ^ b, "BitwiseAnd": lambda: a & b,
                 "ShiftLeftLogical": lambda: a << (b % width), "ShiftRightLogical": lambda: (a & mask) >> (b % width),
                 "ShiftRightArithmetic": lambda: signed(a & mask, width) >> (b % width)}[name]()
            vals[w[1]] = r & mask
        elif name == "ExtInst":
            if w[3] != GLSL_UMIN:
                raise AssertionError(f"the evaluator knows no GLSL.std.450 instruction {w[3]}")
            vals[w[1]] = min(get(w[4]) & M32, get(w[5]) & M32)
        elif name == "Not":
            vals[w[1]] = ~get(w[2]) & ((1 << module.width(w[0])) - 1)
        elif name in ("IEqual", "INotEqual", "ULessThan", "ULessThanEqual", "UGreaterThan", "UGreaterThanEqual",
                      "SLessThan", "SLessThanEqual", "SGreaterThan", "SGreaterThanEqual"):
            a, b = get(w[2]) & M32, get(w[3]) & M32
            if name[0] == "S":
                a, b = signed(a, 32), signed(b, 32)
            vals[w[1]] = {"IEqual": a == b, "INotEqual": a != b, "ULessThan": a < b, "ULessThanEqual": a <= b,
                          "UGreaterThan": a > b, "UGreaterThanEqual": a >= b, "SLessThan": a < b,
                          "SLessThanEqual": a <= b, "SGreaterThan": a > b, "SGreaterThanEqual": a >= b}[name]
        elif name in ("LogicalAnd", "LogicalOr", "LogicalEqual", "LogicalNotEqual"):
            a, b = bool(get(w[2])), bool(get(w[3]))
            vals[w[1]] = {"LogicalAnd": a and b, "LogicalOr": a or b, "LogicalEqual": a == b,
                          "LogicalNotEqual": a != b}[name]
        elif name == "LogicalNot":
            vals[w[1]] = not get(w[2])
        elif name == "Select":
            vals[w[1]] = get(w[3]) if get(w[2]) else get(w[4])
        elif name in ("AtomicOr", "AtomicIAdd"):
            ptr = get(w[2])
            old = mem.load(ptr)
            mem.trace.append(("atomic", name) + ptr)
            mem.store(ptr, (old | get(w[5])) if name == "AtomicOr" else old + get(w[5]))
            vals[w[1]] = old
        elif name == "AtomicCompareExchange":
            ptr = get(w[2])
            old = mem.load(ptr)
            mem.trace.append(("atomic", name) + ptr)
            if old == get(w[7]) & M32:
                mem.store(ptr, get(w[6]))
            vals[w[1]] = old
        elif name == "Branch":
            pc = labels[w[0]]
        elif name == "BranchConditional":
            pc = labels[w[1]] if get(w[0]) else labels[w[2]]
        elif name == "Return":
            return None
        elif name == "ReturnValue":
            return get(w[0]) & M32
        else:
            raise AssertionError(f"the evaluator does not know Op{name}")


# --- the model page table ------------------------------------------------------------------------

# Where the model puts its table: low word near the top of a 4 GiB boundary, so that the words of the table
# (about 1 MiB of them) carry into the high word
TABLE_ADDRESS = 0x1_FFF0_0000
SPAN_ALL = (0, 1 << (40 - CHUNK_LOG2))      # the 2^40 window: guest addresses and the window at 2^39


class PageTable:
    """Builds the table in device memory the way the runtime writes it (raw_table.h), and reads its logs back."""

    def __init__(self, span_base: int = SPAN_ALL[0], span_count: int = SPAN_ALL[1], address: int = TABLE_ADDRESS):
        self.address = address
        self.span_base, self.span_count = span_base, span_count
        self.device: dict[int, int] = {}
        self.blocks = 0
        self.put(A["HeaderVersion"], A["Version"])
        self.put(A["HeaderSpanBase"], span_base)
        self.put(A["HeaderSpanCount"], span_count)

    @property
    def push(self) -> dict[int, int]:
        return {A["PushTableLow"]: self.address & M32, A["PushTableHigh"]: self.address >> 32,
                A["PushSpanBase"]: self.span_base, A["PushSpanCount"]: self.span_count}

    def put(self, word: int, value: int):
        self.device[self.address + 4 * word] = value & M32

    def get(self, word: int) -> int:
        return self.device.get(self.address + 4 * word, 0)

    def top_word(self, guest: int) -> int:
        return A["TopWord"] + (guest >> CHUNK_LOG2) - self.span_base

    def leaf_slot(self, guest: int) -> int:
        return (self.get(self.top_word(guest)) - 1) * LEAF_ENTRIES + ((guest >> PAGE_LOG2) & (LEAF_ENTRIES - 1))

    def entry_word(self, guest: int) -> int:
        return A["TopWord"] + self.span_count + A["EntryWords"] * self.leaf_slot(guest)

    def map_page(self, guest: int, delta: int) -> int:
        """Maps the page of guest to guest + delta; returns its leaf slot."""
        assert 0 <= (guest >> CHUNK_LOG2) - self.span_base < self.span_count
        if not self.get(self.top_word(guest)):
            self.blocks += 1
            self.put(self.top_word(guest), self.blocks)
            self.put(A["HeaderLeafBlocks"], self.blocks)
        self.set_entry(guest, (delta & M64) | 1)
        return self.leaf_slot(guest)

    def set_entry(self, guest: int, entry: int):
        self.put(self.entry_word(guest), entry & M32)
        self.put(self.entry_word(guest) + 1, entry >> 32)

    def stored(self, slot: int) -> bool:
        return bool(self.get(A["StoreBitsWord"] + (slot >> 5)) >> (slot & 31) & 1)

    def store_bit_words(self) -> dict[int, int]:
        lo = self.address + 4 * A["StoreBitsWord"]
        return {a: v for a, v in self.device.items() if lo <= a < lo + 4 * A["StoreBitsWords"] and v}

    def log_end(self) -> int:
        return A["MissLogWord"] + A["MissEntries"] * A["MissEntryWords"]

    @property
    def misses(self) -> int:
        return self.get(A["HeaderMissCount"])

    def miss(self, index: int) -> tuple[int, int, int]:
        """The address, the kind and the reason of a miss log entry."""
        base = A["MissLogWord"] + index * A["MissEntryWords"]
        return (self.get(base + A["MissAddressLow"]) | self.get(base + A["MissAddressHigh"]) << 32,
                self.get(base + A["MissKind"]), self.get(base + A["MissReason"]))


# S2R (0x919): the special register is bits 72..79 (33 and 34: the thread's x and y). LDG (0x381) and
# STG (0x386), non-uniform forms: .E (a 64-bit address pair) is bit 72, the size 73..76 (4: 32 bits),
# the signed offset 40..63. The address pair R2, R3 comes from the thread's position, so the translator
# cannot trace it to a constant buffer (nor fold it, as it would a register never written): both
# accesses go through the raw helper.
B32 = 4
TID_X, TID_Y = 33, 34
MANY_SITES = 64     # SpirvGenerator.RawHelperCallMinSites
# The scheduling fields of bits 104..115: a stall of one cycle and no write or read barrier (7). With
# these bits clear an instruction names barrier 0, and the decoder then hands its result to no reader
# that did not wait on that barrier: the producers below would read as zero.
NO_BARRIERS = 0xFE2 << 40


def scheduled(ins: bytes) -> bytes:
    lo, hi = struct.unpack("<QQ", ins)
    return struct.pack("<QQ", lo, hi | NO_BARRIERS)


TID = [ampere(0x919, rd=2, ra=0, rb=0, rc=0, high_bits=bit(72, TID_X)),
       ampere(0x919, rd=3, ra=0, rb=0, rc=0, high_bits=bit(72, TID_Y))]
LDG = ampere(0x381, rd=0, ra=2, rb=0, rc=0, high_bits=bit(72) | bit(73, B32))
STG = ampere(0x386, ra=2, rb=0, rc=0, low_bits=0x10 << 40, high_bits=bit(72) | bit(73, B32))
# ATOMG (0x3a8) with operation ADD (0) and type U32 (0), ATOMG.CAS (0x3a9): the address pair R2, R3 (.E, bit 72), the value in R3 (Rb), the
# swap value of the CAS in R1 (64..71); the old value goes to R0, which the STS after it stores so that the atomic is not dead code
ATOMG_ADD = ampere(0x3a8, rd=0, ra=2, rb=3, rc=0, high_bits=bit(72))
ATOMG_CAS = ampere(0x3a9, rd=0, ra=2, rb=3, rc=1, high_bits=bit(72))


class RawHelperTest(Harness):
    def translate(self, key: str, instructions: list[bytes], **job) -> tuple[Module, dict]:
        code = self.dir / f"{key}.bin"
        code.write_bytes(b"".join(scheduled(i) for i in instructions) + EXIT)
        got = self.run_jobs([compute_job(key, code, **job)])[key]
        self.assertEqual((got.status, got.message), ("ok", ""))
        if SPIRV_VAL:
            v = validate.validate_one(SPIRV_VAL, key, self.out / f"{key}.spv")
            self.assertTrue(v.ok, v.message)
        return Module((self.out / f"{key}.spv").read_bytes()), json.loads((self.out / f"{key}.res.json").read_text(encoding="utf-8"))

    def setUp(self):
        super().setUp()
        self.module, self.res = self.translate("raw", TID + [LDG, STG])
        helpers = self.module.raw_helpers()
        self.assertEqual(len(helpers), 2, "one load helper and one store helper")
        self.load_fn = next(f for f in helpers if len(f["params"]) == 2)
        self.store_fn = next(f for f in helpers if len(f["params"]) == 3)

    def memory(self, table: PageTable, device: dict[int, int]) -> Memory:
        return Memory(table.address, table.device, device, table.push)

    def load(self, table: PageTable, device: dict[int, int], guest: int) -> tuple[int, Memory]:
        mem = self.memory(table, device)
        return run(self.module, self.load_fn, [guest & M32, guest >> 32], mem), mem

    def store(self, table: PageTable, device: dict[int, int], guest: int, value: int) -> Memory:
        mem = self.memory(table, device)
        run(self.module, self.store_fn, [guest & M32, guest >> 32, value], mem)
        return mem

    def outside_the_table(self, table: PageTable, mem: Memory) -> list[tuple]:
        """The device accesses of a run that are not to the table."""
        return [t for t in mem.trace if t[1] == "dev" and not mem.in_table(t[2])]

    def test_the_description_says_how_the_table_is_reached(self):
        ra = self.res["raw_access"]
        self.assertEqual((ra["physical_pointers"], ra["page_table"], ra["store_bits"], ra["miss_log"]), (True, True, True, True))
        self.assertEqual((ra["abi_version"], ra["push_offset"], ra["push_bytes"]), (A["Version"], 0, A["PushBytes"]))
        self.assertEqual(self.res["sbufs"], [], "the table is reached through push constants: no storage buffer is bound for it")
        self.assertTrue(self.res["writable_storage"])

    def test_a_module_that_only_loads_sets_no_store_bits(self):
        # The loaded value goes to shared memory (STS, 0x388), so that the load is not dead code
        module, res = self.translate("loads", TID + [LDG, ampere(0x388, ra=255, rb=0)], shared=64)
        ra = res["raw_access"]
        self.assertEqual((ra["page_table"], ra["store_bits"], ra["miss_log"]), (True, False, True))
        self.assertEqual(res["sbufs"], [])
        self.assertTrue(res["writable_storage"], "the miss path of a load writes the miss log: the stage needs its stores-and-atomics feature")

    def test_a_module_without_raw_access_does_not_mention_the_table(self):
        got = self.run_jobs([compute_job("plain", self.exit_bin)])["plain"]
        self.assertEqual((got.status, got.message), ("ok", ""))
        res = json.loads((self.out / "plain.res.json").read_text(encoding="utf-8"))
        ra = res["raw_access"]
        self.assertEqual((ra["physical_pointers"], ra["page_table"], ra["store_bits"], ra["miss_log"]), (False, False, False, False))
        self.assertFalse(res["writable_storage"])
        self.assertNotIn(b"push_constants\0", (self.out / "plain.spv").read_bytes())

    def test_the_helper_reads_the_table_address_and_span_from_the_first_push_words(self):
        # The words the walk reads, by the index into the push constant block: the table's two address words, the span's base and count
        table = PageTable()
        table.map_page(0x1000_0000, 0x5000)
        mem = self.memory(table, {})
        run(self.module, self.load_fn, [0x1000_0000, 0], mem)
        self.assertEqual(sorted({t[2] for t in mem.trace if t[:2] == ("load", "push")}),
                         [A["PushTableLow"], A["PushTableHigh"], A["PushSpanBase"], A["PushSpanCount"]])

    def test_a_helper_with_few_call_sites_is_left_to_inline(self):
        for f in (self.load_fn, self.store_fn):
            self.assertFalse(f["control"] & FUNCTION_DONT_INLINE, f"function control {f['control']:#x}")

    def test_a_helper_with_many_call_sites_stays_a_call(self):
        # Inlined at each of a module's sites, the helper multiplies the code the driver compiles (the raw-access compile fix):
        # from 64 sites on, the translator declares it DontInline.
        # Each load gets its own register and its own store, so none of them is dead code.
        loads = [ampere(0x381, rd=4 + n, ra=2, rb=0, rc=0, low_bits=(4 * n) << 40, high_bits=bit(72) | bit(73, B32))
                 for n in range(MANY_SITES)]
        stores = [ampere(0x386, ra=2, rb=4 + k, rc=0, low_bits=(0x1000 + 4 * k) << 40, high_bits=bit(72) | bit(73, B32))
                  for k in range(MANY_SITES)]
        module, _ = self.translate("many", TID + loads + stores)
        load = next(f for f in module.raw_helpers() if len(f["params"]) == 2)
        calls = sum(1 for f in module.functions.values() for name, w in f["code"] if name == "FunctionCall" and w[2] == load["id"])
        self.assertGreaterEqual(calls, MANY_SITES)
        self.assertTrue(load["control"] & FUNCTION_DONT_INLINE, f"function control {load['control']:#x}")

    def test_a_mapped_page_reads_at_the_address_plus_its_delta(self):
        table = PageTable()
        guest, delta = 0x12_3456_7890 & ~3, 0x7F_0000_1000 - 0x12_3456_7000
        table.map_page(guest, delta)
        got, mem = self.load(table, {guest + delta: 0xCAFEF00D}, guest)
        self.assertEqual(got, 0xCAFEF00D)
        self.assertEqual(table.misses, 0)
        walk = [t for t in mem.trace if t[:2] == ("load", "dev") and mem.in_table(t[2])]
        self.assertEqual(len(walk), 3, "the walk is the top-level word and the leaf entry's two words")
        self.assertEqual([t for t in mem.trace if t[0] != "load"], [], "a load that lands writes nothing")

    def test_the_delta_carries_into_the_high_word(self):
        table = PageTable()
        guest = 0x05_FFFF_F010
        delta = 0x2000            # guest + delta = 0x06_0000_1010
        table.map_page(guest, delta)
        got, _ = self.load(table, {0x06_0000_1010: 0x1234}, guest)
        self.assertEqual(got, 0x1234)

    def test_a_negative_delta_borrows_from_the_high_word(self):
        table = PageTable()
        guest = 0x40_0000_0020
        delta = -0x30_0000_0000   # a device address below the guest address
        table.map_page(guest, delta)
        got, _ = self.load(table, {0x10_0000_0020: 77}, guest)
        self.assertEqual(got, 77)

    def test_the_window_for_pools_the_cpu_cannot_see(self):
        table = PageTable()
        guest = (1 << 39) + 0x1234_5000
        table.map_page(guest, 0x1000_0000)
        got, _ = self.load(table, {guest + 0x1000_0000: 9}, guest)
        self.assertEqual(got, 9)

    def test_each_page_has_its_own_delta(self):
        table = PageTable()
        base = 0x20_0000_0000
        table.map_page(base, 0x100000)
        table.map_page(base + 0x1000, 0x900000)
        device = {base + 0xFFC + 0x100000: 1, base + 0x1000 + 0x900000: 2}
        self.assertEqual(self.load(table, device, base + 0xFFC)[0], 1)
        self.assertEqual(self.load(table, device, base + 0x1000)[0], 2)

    def test_the_table_may_sit_across_a_4_gib_boundary(self):
        # The table's words are at the table's address plus 4 times their index: the sum carries into the high word
        for address in (0x0_FFFF_F000, 0x1_FFFE_0000, 0x2_FFFF_FFF0):
            with self.subTest(address=hex(address)):
                table = PageTable(address=address)
                self.assertGreater((address & M32) + 4 * A["TopWord"], 1 << 32)
                guest = 0x17_0000_5000
                table.map_page(guest, 0x4000)
                self.assertEqual(self.load(table, {guest + 0x4000: 5}, guest)[0], 5)
                self.assertEqual(table.misses, 0)

    def test_a_delta_of_zero_is_a_mapped_page(self):
        # An unmapped leaf is an entry with bit 0 clear, not a zero entry: a page whose device address is its guest address has delta 0
        table = PageTable()
        guest = 0x2000_0000
        table.map_page(guest, 0)
        self.assertEqual(table.get(table.entry_word(guest)), 1)
        got, _ = self.load(table, {guest: 0xBEEF}, guest)
        self.assertEqual(got, 0xBEEF)
        self.assertEqual(table.misses, 0)

    def test_an_entry_with_bit_0_clear_is_unmapped_whatever_else_it_holds(self):
        table = PageTable()
        mapped = 0x30_0000_0000
        table.map_page(mapped, 0x1000)
        for what, entry in (("zero", 0), ("a delta without the bit", 0x1000), ("high word only", 7 << 32), ("everything but the bit", M64 - 1)):
            with self.subTest(what):
                table.set_entry(mapped + 0x1000, entry)
                got, mem = self.load(table, {mapped + 0x1000 + (entry & ~1 & M64): 0x99, mapped + 0x1000: 0x99}, mapped + 0x1000)
                self.assertEqual(got, 0)
                self.assertEqual(self.outside_the_table(table, mem), [], "no access beyond the table")
        self.assertEqual(table.misses, 4)
        self.assertEqual({table.miss(n)[2] for n in range(4)}, {A["MissReasonNoPage"]})

    def test_misses_read_zero_and_touch_no_device_memory_but_the_table(self):
        table = PageTable()
        mapped = 0x30_0000_0000
        table.map_page(mapped, 0x1000)
        device = {mapped + 0x1000: 5}
        cases = {
            "unmapped page in a mapped chunk": (mapped + 0x1000, A["MissReasonNoPage"]),
            "unmapped chunk": (mapped + (1 << CHUNK_LOG2), A["MissReasonNoChunk"]),
            "top of the address space": (M64 & ~3, A["MissReasonOutsideSpan"]),
            "beyond 2^53, where the chunk number would wrap": (mapped + (1 << 53), A["MissReasonOutsideSpan"]),
        }
        for n, (what, (guest, reason)) in enumerate(cases.items()):
            with self.subTest(what):
                got, mem = self.load(table, device, guest)
                self.assertEqual(got, 0)
                self.assertEqual(self.outside_the_table(table, mem), [], mem.trace)
                self.assertEqual(table.miss(n), (guest, A["MissKindLoad"], reason))
        self.assertEqual(table.misses, len(cases))

    def test_the_span_bounds_the_addresses_that_are_looked_up(self):
        # The span is the four chunks from base; the last one has a page mapped at offset 0x3000. An address one chunk before the span, one chunk after
        # it or far beyond it, at the same offset in its chunk, would alias that page if the walk clamped the chunk instead of refusing it
        base = 0x30_0000_0000 >> CHUNK_LOG2
        table = PageTable(span_base=base, span_count=4)
        inside = ((base + 3) << CHUNK_LOG2) + 0x3000
        table.map_page(inside, 0x10_0000)
        device = {inside + 0x10_0000: 0xAB}
        self.assertEqual(self.load(table, device, inside)[0], 0xAB)
        first = table.misses
        cases = {"one chunk before": -1, "one chunk after": 4, "far after": 1000, "the first chunk of the 2^40 window": -base}
        for n, (what, chunk_delta) in enumerate(cases.items()):
            with self.subTest(what):
                guest = inside + (chunk_delta - 3) * (1 << CHUNK_LOG2)
                got, mem = self.load(table, device, guest)
                self.assertEqual(got, 0)
                self.assertEqual(self.outside_the_table(table, mem), [])
                self.assertEqual(table.miss(first + n), (guest, A["MissKindLoad"], A["MissReasonOutsideSpan"]))
        self.assertEqual(table.misses, first + len(cases))

    def test_the_span_gives_the_top_level_index(self):
        # The top-level word of a chunk is indexed from the span's first chunk, and the leaf blocks follow the top level, wherever the span lies
        for base, count in ((0, 1), (0x18_0000, 1), (0x18_0000, 300), (0x7_FFFF, 1)):
            with self.subTest(base=base, count=count):
                table = PageTable(span_base=base, span_count=count)
                guest = ((base + count - 1) << CHUNK_LOG2) + 0x7000
                table.map_page(guest, 0x8000)
                self.assertEqual(table.get(A["TopWord"] + count - 1), 1)
                self.assertEqual(self.load(table, {guest + 0x8000: 33}, guest)[0], 33)

    def test_a_store_lands_at_the_delta_and_sets_its_leaf_slots_bit_once(self):
        table = PageTable()
        guest, delta = 0x11_0000_3008, 0x4000_0000
        slot = table.map_page(guest, delta)
        other = table.map_page(guest + (5 << CHUNK_LOG2), 0x8000)
        device: dict[int, int] = {}
        mem = self.store(table, device, guest, 0xABCD)
        self.assertEqual(device, {guest + delta: 0xABCD})
        self.assertTrue(table.stored(slot))
        self.assertFalse(table.stored(other))
        self.assertEqual(table.misses, 0)
        self.assertEqual(table.store_bit_words(), {table.address + 4 * (A["StoreBitsWord"] + (slot >> 5)): 1 << (slot & 31)})
        self.assertEqual(len([t for t in mem.trace if t[0] == "atomic"]), 1)
        mem = self.store(table, device, guest + 4, 0xEF)
        self.assertEqual(device[guest + delta + 4], 0xEF)
        self.assertEqual([t for t in mem.trace if t[0] == "atomic"], [], "a flagged page is read, not or-ed again")

    def test_the_store_bit_is_by_leaf_slot(self):
        # Leaf slot (block - 1) * 512 + page in the chunk: the second block's first page is slot 512, bit 0 of store bit word 16
        table = PageTable()
        first, second = 0x11_0000_0000, 0x12_0000_0000
        table.map_page(first + 0x1000 * 33, 0x1000)
        table.map_page(second + 0x1000 * 3, 0x2000)
        device: dict[int, int] = {}
        self.store(table, device, second + 0x1000 * 3, 1)
        self.assertEqual(table.store_bit_words(), {table.address + 4 * (A["StoreBitsWord"] + 16): 1 << 3})
        self.store(table, device, first + 0x1000 * 33, 1)
        self.assertEqual(table.store_bit_words(), {table.address + 4 * (A["StoreBitsWord"] + 1): 2,
                                                   table.address + 4 * (A["StoreBitsWord"] + 16): 1 << 3})

    def test_a_store_to_an_unmapped_page_is_dropped_and_logged_with_its_address(self):
        table = PageTable()
        table.map_page(0x11_0000_0000, 0x1000)
        device: dict[int, int] = {}
        self.store(table, device, 0x11_0000_2000, 1)
        self.assertEqual(device, {})
        self.assertEqual(table.misses, 1)
        self.assertEqual(table.miss(0), (0x11_0000_2000, A["MissKindStore"], A["MissReasonNoPage"]))
        self.assertEqual(table.store_bit_words(), {}, "a miss flags no page")

    def atomic_module(self, key: str, instruction: bytes, operands: int):
        module, res = self.translate(key, TID + [instruction, ampere(0x388, ra=255, rb=0)], shared=64)
        helpers = module.raw_helpers()
        self.assertEqual(len(helpers), 1)
        self.assertEqual(len(helpers[0]["params"]), 2 + operands)
        self.assertTrue(res["raw_access"]["store_bits"], "an atomic counts as a store for the store bits")
        return module, helpers[0]

    def atomic_ops(self, mem: Memory, name: str) -> list[tuple]:
        return [t for t in mem.trace if t[0] == "atomic" and t[1] == name]

    def test_an_atomic_add_lands_at_the_delta_and_sets_its_leaf_slots_bit_once(self):
        module, fn = self.atomic_module("atomg", ATOMG_ADD, 1)
        table = PageTable()
        guest, delta = 0x11_0000_3008, 0x4000_0000
        slot = table.map_page(guest, delta)
        device = {guest + delta: 5}
        mem = self.memory(table, device)
        self.assertEqual(run(module, fn, [guest & M32, guest >> 32, 3], mem), 5, "the old value comes back")
        self.assertEqual(device, {guest + delta: 8})
        self.assertTrue(table.stored(slot))
        self.assertEqual(table.store_bit_words(), {table.address + 4 * (A["StoreBitsWord"] + (slot >> 5)): 1 << (slot & 31)})
        self.assertEqual(len(self.atomic_ops(mem, "AtomicOr")), 1)
        mem = self.memory(table, device)
        run(module, fn, [guest & M32, guest >> 32, 1], mem)
        self.assertEqual(device[guest + delta], 9)
        self.assertEqual(self.atomic_ops(mem, "AtomicOr"), [], "a flagged page is read, not or-ed again")

    def test_a_compare_and_swap_lands_at_the_delta_and_sets_its_leaf_slots_bit(self):
        module, fn = self.atomic_module("atomcas", ATOMG_CAS, 2)
        table = PageTable()
        guest, delta = 0x12_0000_5000, 0x2000_0000
        slot = table.map_page(guest, delta)
        device = {guest + delta: 7}
        mem = self.memory(table, device)
        self.assertEqual(run(module, fn, [guest & M32, guest >> 32, 7, 99], mem), 7)
        self.assertEqual(device, {guest + delta: 99})
        self.assertTrue(table.stored(slot))
        run(module, fn, [guest & M32, guest >> 32, 7, 1], self.memory(table, device))
        self.assertEqual(device, {guest + delta: 99}, "the compare failed: nothing is swapped")

    def test_an_atomic_on_an_unmapped_page_is_dropped_and_logged_as_an_atomic(self):
        for name, instruction, operands in (("add", ATOMG_ADD, 1), ("cas", ATOMG_CAS, 2)):
            with self.subTest(name):
                module, fn = self.atomic_module(f"atomm_{name}", instruction, operands)
                table = PageTable()
                table.map_page(0x11_0000_0000, 0x1000)
                device: dict[int, int] = {}
                mem = self.memory(table, device)
                got = run(module, fn, [0x2000, 0x11, 1, 2][:2 + operands], mem)
                self.assertEqual((got, device), (0, {}))
                self.assertEqual(self.outside_the_table(table, mem), [])
                self.assertEqual(table.miss(0), (0x11_0000_2000, A["MissKindAtomic"], A["MissReasonNoPage"]))
                self.assertEqual(table.store_bit_words(), {}, "a miss flags no page")

    def test_a_span_count_of_zero_never_reads_outside_the_table(self):
        # A runtime bug (the ABI says at least 1): the top-level index must not follow an address that is in no span
        table = PageTable(span_base=0, span_count=0)
        guest = 0x1FFF_FFFF_F000
        got, mem = self.load(table, {}, guest)
        self.assertEqual(got, 0)
        self.assertEqual(self.outside_the_table(table, mem), [], mem.trace)
        self.assertEqual(table.miss(0), (guest, A["MissKindLoad"], A["MissReasonOutsideSpan"]))

    def test_the_miss_log_keeps_the_first_entries_and_counts_the_rest(self):
        table = PageTable()
        table.map_page(0x11_0000_0000, 0x1000)
        total = A["MissEntries"] + 3
        for n in range(total):
            self.load(table, {}, 0x11_0000_1000 + 0x1000 * n)
        self.assertEqual(table.misses, total, "the counter goes on")
        for n in range(A["MissEntries"]):
            self.assertEqual(table.miss(n)[1:], (A["MissKindLoad"], A["MissReasonNoPage"]))
        log_end = A["MissLogWord"] + A["MissEntries"] * A["MissEntryWords"]
        self.assertEqual(log_end, A["StoreBitsWord"])
        self.assertEqual(table.store_bit_words(), {}, "an entry past the log's end would land in the store bits")


if __name__ == "__main__":
    unittest.main()
