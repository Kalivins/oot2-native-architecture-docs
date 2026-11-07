# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/shaders/spvstats.py, the counter of what a SPIR-V module does to memory.

The modules are assembled here from words, never read from a dump. The checks that guard the
instrument itself: a removed atomic is counted as lost, a read that crossed a write or moved into a
branch is counted as moved, an unchanged module counts as neither, and a read that is only dropped
(dead code) never counts as moved.

Run: python -m unittest discover -s tests -p test_spvstats.py -v
"""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "shaders"))

import spvstats  # noqa: E402

OP = dict(TypeVoid=19, TypeBool=20, TypeInt=21, TypeFunction=33, TypePointer=32, Function=54, FunctionEnd=56,
          FunctionCall=57, Variable=59, Load=61, Store=62, AtomicIAdd=234, ControlBarrier=224, Label=248,
          Return=253, Branch=249, BranchConditional=250, SelectionMerge=247, ImageWrite=99,
          GroupNonUniformShuffle=345, TypeStruct=30, Constant=43)
WORKGROUP, PRIVATE, FUNCTION, STORAGE_BUFFER = 4, 6, 7, 12


class Asm:
    """A tiny SPIR-V assembler: ids are handed out in order, instructions are word lists."""

    def __init__(self):
        self.next_id = 1
        self.head: list[list[int]] = []
        self.funcs: list[list[list[int]]] = []

    def id(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    @staticmethod
    def ins(op: int, *w: int) -> list[int]:
        return [((len(w) + 1) << 16) | op, *w]

    def decl(self, op: int, *operands: int, result: bool = True):
        rid = self.id() if result else None
        self.head.append(self.ins(OP[op] if isinstance(op, str) else op, *(([rid] if result else []) + list(operands))))
        return rid

    def typed(self, op: str, type_id: int, *operands: int) -> int:
        """An instruction whose layout is (result type, result id, operands...)."""
        rid = self.id()
        self.head.append(self.ins(OP[op], type_id, rid, *operands))
        return rid

    def module(self) -> bytes:
        words = [0x07230203, 0x10500, 0, self.next_id, 0]
        for i in self.head:
            words += i
        for f in self.funcs:
            for i in f:
                words += i
        return struct.pack(f"<{len(words)}I", *words)


class Types:
    """The types and variables every test module shares."""

    def __init__(self, a: Asm):
        self.void = a.decl("TypeVoid")
        self.fn = a.decl("TypeFunction", self.void)
        self.uint = a.decl("TypeInt", 32, 0)
        self.p_wg = a.decl("TypePointer", WORKGROUP, self.uint)
        self.p_sb = a.decl("TypePointer", STORAGE_BUFFER, self.uint)
        self.p_fn = a.decl("TypePointer", FUNCTION, self.uint)
        self.wg = a.typed("Variable", self.p_wg, WORKGROUP)
        self.sb = a.typed("Variable", self.p_sb, STORAGE_BUFFER)
        self.local = a.typed("Variable", self.p_fn, FUNCTION)
        self.scope = a.typed("Constant", self.uint, 2)    # a constant; its value is irrelevant here


class Func:
    """One function under construction: its body is a list of instruction word lists."""

    def __init__(self, a: Asm, t: Types):
        self.a, self.t = a, t
        self.fid = a.id()
        self.code: list[list[int]] = [Asm.ins(OP["Function"], t.void, self.fid, 0, t.fn), Asm.ins(OP["Label"], a.id())]

    def add(self, *ins: list[int]) -> "Func":
        self.code += list(ins)
        return self

    def load(self, var: int) -> int:
        rid = self.a.id()
        self.add(Asm.ins(OP["Load"], self.t.uint, rid, var))
        return rid

    def store(self, var: int, value: int) -> "Func":
        return self.add(Asm.ins(OP["Store"], var, value))

    def atomic_add(self, var: int, value: int) -> int:
        rid = self.a.id()
        self.add(Asm.ins(OP["AtomicIAdd"], self.t.uint, rid, var, self.t.scope, self.t.scope, value))
        return rid

    def barrier(self) -> "Func":
        return self.add(Asm.ins(OP["ControlBarrier"], self.t.scope, self.t.scope, self.t.scope))

    def call(self, callee: "Func") -> int:
        rid = self.a.id()
        self.add(Asm.ins(OP["FunctionCall"], self.t.void, rid, callee.fid))
        return rid

    def done(self) -> "Func":
        self.add(Asm.ins(OP["Return"]), Asm.ins(OP["FunctionEnd"]))
        self.a.funcs.append(self.code)
        return self


def module_of(build) -> dict:
    a = Asm()
    t = Types(a)
    build(a, t)
    return spvstats.analyze(a.module())


def read_then_write(order: str):
    """x = load shared[0]; shared[0] = 7; shared[0] = x  (order 'lsu'), or the load moved after the store (order 'slu')."""
    def build(a: Asm, t: Types):
        f = Func(a, t)
        if order == "lsu":
            x = f.load(t.wg)
            f.store(t.wg, t.scope)
        else:
            f.store(t.wg, t.scope)
            x = f.load(t.wg)
        f.store(t.wg, x).done()
    return module_of(build)


class CountsTest(unittest.TestCase):
    def test_counts_by_kind_and_class(self):
        def build(a, t):
            f = Func(a, t)
            f.atomic_add(t.wg, t.scope)
            f.barrier()
            x = f.load(t.sb)
            f.store(t.sb, x)
            f.store(t.local, x)                       # a local variable's store is not memory
            f.add(Asm.ins(OP["GroupNonUniformShuffle"], t.uint, a.id(), t.scope, x, t.scope)).done()
        s = module_of(build)
        self.assertEqual((s["atomics"], s["barriers"], s["subgroup"]), (1, 1, 1))
        self.assertEqual(s["stores"], {"StorageBuffer": 1})
        self.assertEqual(s["loads"], {"StorageBuffer": 1})

    def test_calls_are_split_by_what_the_callee_does(self):
        def build(a, t):
            writer = Func(a, t).store(t.sb, t.scope).done()
            reader = Func(a, t)
            reader.load(t.sb)
            reader.done()
            pure = Func(a, t).done()
            wrapper = Func(a, t)
            wrapper.call(writer)                      # a call that reaches a write through another function
            wrapper.done()
            main = Func(a, t)
            for callee in (writer, reader, pure, wrapper):
                main.call(callee)
            main.done()
        s = module_of(build)
        self.assertEqual((s["calls_write"], s["calls_read"], s["calls_pure"]), (3, 1, 1))   # main calls two writers (one through wrapper), wrapper calls one

    def test_rejects_what_is_not_a_module(self):
        with self.assertRaises(ValueError):
            spvstats.analyze(b"\x00" * 24)
        with self.assertRaises(ValueError):
            spvstats.analyze(b"\x03\x02\x23\x07")


class LostTest(unittest.TestCase):
    def pair(self, with_atomic: bool, with_barrier: bool = True) -> dict:
        def build(a, t):
            f = Func(a, t)
            if with_atomic:
                f.atomic_add(t.sb, t.scope)
            if with_barrier:
                f.barrier()
            f.store(t.sb, t.scope).done()
        return module_of(build)

    def test_a_removed_atomic_is_lost(self):
        verdict = spvstats.lost(self.pair(True), self.pair(False))
        self.assertEqual(verdict, {"atomics": 1})

    def test_a_removed_barrier_and_store_are_lost(self):
        def build(a, t):
            Func(a, t).done()
        empty = module_of(build)
        verdict = spvstats.lost(self.pair(True), empty)
        self.assertEqual(verdict, {"atomics": 1, "barriers": 1, "stores_StorageBuffer": 1})

    def test_the_same_module_loses_nothing(self):
        m = self.pair(True)
        self.assertEqual(spvstats.lost(m, m), {})

    def test_extra_operations_are_not_a_loss(self):
        self.assertEqual(spvstats.lost(self.pair(False), self.pair(True)), {})

    def test_a_dropped_call_that_does_nothing_is_not_a_loss(self):
        def build(with_call):
            def b(a, t):
                pure = Func(a, t).done()
                main = Func(a, t)
                if with_call:
                    main.call(pure)
                main.done()
            return module_of(b)
        self.assertEqual(spvstats.lost(build(True), build(False)), {})


class MovedTest(unittest.TestCase):
    def test_a_read_that_crossed_a_write_is_moved(self):
        v = spvstats.compare(read_then_write("lsu"), read_then_write("slu"))
        self.assertEqual(v["moved"], 1)
        self.assertEqual(v["lost"], {})

    def test_a_flattened_construct_is_not_a_moved_read(self):
        def branch(a, t):
            cond = a.typed("Variable", t.p_fn, FUNCTION)
            f = Func(a, t)
            then, merge = a.id(), a.id()
            f.add(Asm.ins(OP["SelectionMerge"], merge, 0), Asm.ins(OP["BranchConditional"], cond, then, merge))
            f.add(Asm.ins(OP["Label"], then))
            x = f.load(t.wg)
            f.store(t.sb, x)
            f.add(Asm.ins(OP["Branch"], merge), Asm.ins(OP["Label"], merge)).done()

        def straight(a, t):
            f = Func(a, t)
            f.store(t.sb, f.load(t.wg)).done()
        ref, cand = module_of(branch), module_of(straight)
        verdict = spvstats.compare(ref, cand)
        self.assertEqual((verdict["restructured"], verdict["moved"], verdict["moved_convergent"]), (True, 0, None))
        self.assertEqual(spvstats.moved(ref, cand), 1)                  # by depth alone it would look like a move

    def test_a_shuffle_moved_into_a_branch_is_a_moved_convergent_operation(self):
        def build(inside):
            def b(a, t):
                cond = a.typed("Variable", t.p_fn, FUNCTION)
                f = Func(a, t)
                then, merge = a.id(), a.id()
                if not inside:
                    f.add(Asm.ins(OP["GroupNonUniformShuffle"], t.uint, a.id(), t.scope, t.scope, t.scope))
                f.add(Asm.ins(OP["SelectionMerge"], merge, 0), Asm.ins(OP["BranchConditional"], cond, then, merge))
                f.add(Asm.ins(OP["Label"], then))
                if inside:
                    f.add(Asm.ins(OP["GroupNonUniformShuffle"], t.uint, a.id(), t.scope, t.scope, t.scope))
                f.add(Asm.ins(OP["Branch"], merge), Asm.ins(OP["Label"], merge)).done()
            return module_of(b)
        verdict = spvstats.compare(build(False), build(True))
        self.assertEqual((verdict["restructured"], verdict["moved_convergent"]), (False, 1))
        self.assertEqual(spvstats.compare(build(True), build(True))["moved_convergent"], 0)

    def test_a_candidate_that_lost_a_write_says_nothing_about_moved_reads(self):
        def build(atomic):
            def b(a, t):
                f = Func(a, t)
                if atomic:
                    f.atomic_add(t.wg, t.scope)
                x = f.load(t.wg)
                f.store(t.sb, x).done()
            return module_of(b)
        verdict = spvstats.compare(build(True), build(False))
        self.assertEqual((verdict["lost"], verdict["moved"]), ({"atomics": 1}, None))

    def test_the_same_order_moves_nothing(self):
        self.assertEqual(spvstats.moved(read_then_write("lsu"), read_then_write("lsu")), 0)

    def test_the_reverse_direction_is_also_seen(self):
        self.assertEqual(spvstats.moved(read_then_write("slu"), read_then_write("lsu")), 1)

    def test_a_read_moved_into_a_branch_is_moved(self):
        def build(inside):
            def b(a, t):
                cond = a.typed("Variable", t.p_fn, FUNCTION)
                f = Func(a, t)
                then, merge = a.id(), a.id()
                x = None
                if not inside:
                    x = f.load(t.wg)
                f.add(Asm.ins(OP["SelectionMerge"], merge, 0), Asm.ins(OP["BranchConditional"], cond, then, merge))
                f.add(Asm.ins(OP["Label"], then))
                if inside:
                    x = f.load(t.wg)
                f.store(t.sb, x)
                f.add(Asm.ins(OP["Branch"], merge), Asm.ins(OP["Label"], merge)).done()
            return module_of(b)
        self.assertEqual(spvstats.moved(build(False), build(True)), 1)
        self.assertEqual(spvstats.moved(build(True), build(True)), 0)

    def test_a_read_that_is_only_dropped_is_not_moved(self):
        def build(read):
            def b(a, t):
                f = Func(a, t)
                if read:
                    f.load(t.wg)
                f.store(t.wg, t.scope).done()
            return module_of(b)
        self.assertEqual(spvstats.moved(build(True), build(False)), 0)

    def test_other_functions_make_the_comparison_unavailable(self):
        def build(n):
            def b(a, t):
                for _ in range(n):
                    Func(a, t).done()
            return module_of(b)
        self.assertEqual(spvstats.compare(build(1), build(2))["moved"], None)

    def test_a_read_through_a_reader_call_is_ordered_like_a_load(self):
        def build(order):
            def b(a, t):
                reader = Func(a, t)
                reader.load(t.sb)
                reader.done()
                f = Func(a, t)
                if order == "rw":
                    f.call(reader)
                    f.store(t.sb, t.scope)
                else:
                    f.store(t.sb, t.scope)
                    f.call(reader)
                f.done()
            return module_of(b)
        self.assertEqual(spvstats.moved(build("rw"), build("wr")), 1)
        self.assertEqual(spvstats.moved(build("rw"), build("rw")), 0)


if __name__ == "__main__":
    unittest.main()
