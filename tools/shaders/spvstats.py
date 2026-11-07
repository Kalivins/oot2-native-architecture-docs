#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Count what a SPIR-V module does to memory, and compare two translations of one program.

    py -3 tools/shaders/spvstats.py MODULE.spv [MODULE.spv ...]
    py -3 tools/shaders/spvstats.py --compare REFERENCE.spv CANDIDATE.spv

The translator's AST optimizer may be run with or without its debug mode. Without it,
an expression with a single use is moved to that use, and an assignment whose result nobody reads is
dropped. Both are right for arithmetic and wrong for an operation with an effect, and for a read of
memory that a later store changes. This module is the instrument that sees it: for one module it
counts the operations a translation must keep, and for two modules of the same program it says what
the candidate lost and what it moved.

Counted per module: atomics (every OpAtomic*, the float forms included), control and memory
barriers, subgroup operations (OpGroupNonUniform*), function calls split by what the callee does (a
callee that writes memory, one that only reads, one that does neither), stores and loads by storage
class (Workgroup, StorageBuffer, PhysicalStorageBuffer, Private, Output, and images).

Moved reads. A read is the module's OpLoad of a memory storage class, an image read, or a call to a
function that only reads. Inside one function the writes in source order are numbered: a store, an
atomic, a barrier, an image write or a call to a function that writes ends one epoch and starts the
next. A read's place is (function, kind, storage class, nesting depth, epoch), where the depth is
the number of structured constructs (selection or loop) open around it. The same program translated
twice has the same multiset of places unless an expression moved: a read that crossed a write has
another epoch, one moved into an `if` or a loop another depth. A candidate never has fewer reads
than a place needs, so the excess in a place is a lower bound on the reads that moved into it (a
dropped dead read can only lower a count, never raise one). Subgroup operations and the derivative
instructions (implicit-lod samples) need the same convergence they had, so their depth is compared
the same way.

The optimizer also removes empty blocks and flattens `else { if }` into `else if`, which changes the depth of what
the blocks hold without moving it. A module whose number of constructs changed is `restructured`: its reads are
compared by epoch alone and its convergent operations not at all.

A function is compared by its ordinal in the module, so a candidate that defines other functions is
reported as `functions_differ` and not compared further.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter

OP = {
    "TypePointer": 32, "Function": 54, "FunctionParameter": 55, "FunctionEnd": 56, "FunctionCall": 57,
    "Variable": 59, "ImageTexelPointer": 60, "Load": 61, "Store": 62, "CopyMemory": 63, "AccessChain": 65,
    "InBoundsAccessChain": 66, "PtrAccessChain": 67, "InBoundsPtrAccessChain": 70, "CopyObject": 83,
    "ImageRead": 98, "ImageWrite": 99, "ConvertUToPtr": 120, "Bitcast": 124, "Select": 169,
    "ControlBarrier": 224, "MemoryBarrier": 225, "Phi": 245, "LoopMerge": 246, "SelectionMerge": 247,
    "Label": 248,
}
NAMES = {v: k for k, v in OP.items()}

# Opcodes that produce a pointer as their result: layout (result type, result id, ...).
POINTER_PRODUCERS = {OP[n] for n in ("FunctionParameter", "FunctionCall", "ImageTexelPointer", "Load", "AccessChain",
                                     "InBoundsAccessChain", "PtrAccessChain", "InBoundsPtrAccessChain", "CopyObject",
                                     "ConvertUToPtr", "Bitcast", "Select", "Phi")}
ATOMIC_OPS = set(range(227, 243)) | {318, 319, 5614, 5615, 6035}   # OpAtomicLoad..OpAtomicXor, flag ops, Min/Max/Add EXT
SUBGROUP_OPS = set(range(333, 367))                                # OpGroupNonUniformElect..OpGroupNonUniformQuadSwap
DERIVATIVE_OPS = {87, 89, 91, 93, 105} | set(range(207, 219))     # implicit-lod samples, OpImageQueryLod, OpDPdx..OpFwidthCoarse

STORAGE = {0: "UniformConstant", 1: "Input", 2: "Uniform", 3: "Output", 4: "Workgroup", 6: "Private", 7: "Function",
           9: "PushConstant", 12: "StorageBuffer", 5349: "PhysicalStorageBuffer", 5402: "TaskPayload"}
# Classes whose contents a store can change under a later read, and so order reads.
MEMORY = {4, 6, 12, 5349, 5402}
# Stores worth counting; a Function store is a local variable's, not memory.
COUNTED_STORES = MEMORY | {3}

COUNT_FIELDS = ("atomics", "barriers", "memory_barriers", "subgroup", "calls_write", "calls_read", "calls_pure")


def storage_name(cls: int) -> str:
    return STORAGE.get(cls, f"class{cls}")


def check(data: bytes) -> tuple[int, ...]:
    """The module's words, after the size and the magic number are checked."""
    if len(data) % 4 or len(data) < 20:
        raise ValueError("not a SPIR-V module: size")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != 0x07230203:
        raise ValueError("not a SPIR-V module: magic")
    return words


def instructions(data: bytes):
    """(opcode, operand words) for every instruction after the header."""
    words = check(data)
    i = 5
    while i < len(words):
        wc = words[i] >> 16
        if wc == 0 or i + wc > len(words):
            raise ValueError("not a SPIR-V module: instruction length")
        yield words[i] & 0xFFFF, words[i + 1:i + wc]
        i += wc


class Function:
    def __init__(self, fid: int):
        self.id = fid
        self.code: list[tuple[int, list[int]]] = []
        self.callees: list[int] = []
        self.writes = False    # directly: a store to memory, an atomic, a barrier, an image write
        self.reads = False     # directly: a load of memory, an image read


def parse(data: bytes) -> tuple[list[Function], dict[int, int], int]:
    """Functions in order, the storage class of every pointer id, and the id bound."""
    bound = check(data)[3]
    pointer_class: dict[int, int] = {}      # type id of a pointer -> storage class
    pointer_type: dict[int, int] = {}       # value id -> its type id (only for ids that can be pointers)
    functions: list[Function] = []
    cur: Function | None = None
    for op, w in instructions(data):
        if op == OP["TypePointer"]:
            pointer_class[w[0]] = w[1]
        elif op == OP["Variable"]:
            pointer_type[w[1]] = w[0]
        elif op in POINTER_PRODUCERS and len(w) >= 2:
            pointer_type[w[1]] = w[0]
        if op == OP["Function"]:
            cur = Function(w[1])
            functions.append(cur)
        elif op == OP["FunctionEnd"]:
            cur = None
        elif cur is not None:
            cur.code.append((op, list(w)))
    pointer_to_class = {vid: pointer_class[t] for vid, t in pointer_type.items() if t in pointer_class}
    return functions, pointer_to_class, bound


def summarise(functions: list[Function], pclass: dict[int, int]) -> dict[int, tuple[bool, bool]]:
    """(writes, reads) per function id, with callees folded in."""
    by_id = {f.id: f for f in functions}
    for f in functions:
        for op, w in f.code:
            if op == OP["Store"]:
                f.writes |= pclass.get(w[0]) in MEMORY
            elif op == OP["CopyMemory"]:
                f.writes |= pclass.get(w[0]) in MEMORY
                f.reads |= pclass.get(w[1]) in MEMORY
            elif op == OP["Load"]:
                f.reads |= pclass.get(w[2]) in MEMORY
            elif op == OP["ImageRead"]:
                f.reads = True
            elif op == OP["ImageWrite"] or op in ATOMIC_OPS or op in (OP["ControlBarrier"], OP["MemoryBarrier"]):
                f.writes = True
            elif op == OP["FunctionCall"]:
                f.callees.append(w[2])
    result: dict[int, tuple[bool, bool]] = {}

    def fold(fid: int, seen: frozenset) -> tuple[bool, bool]:
        if fid in result:
            return result[fid]
        f = by_id.get(fid)
        if f is None or fid in seen:
            return (True, True)         # unknown or recursive: assume the worst
        writes, reads = f.writes, f.reads
        for c in f.callees:
            cw, cr = fold(c, seen | {fid})
            writes |= cw
            reads |= cr
        result[fid] = (writes, reads)
        return result[fid]

    for f in functions:
        fold(f.id, frozenset())
    return result


def analyze(data: bytes) -> dict:
    """Counts and the read signature of one module (see the module docstring)."""
    functions, pclass, bound = parse(data)
    effect = summarise(functions, pclass)
    stats: dict = {"bytes": len(data), "ids": bound, "functions": len(functions), "instructions": 0, "constructs": 0,
                   **{f: 0 for f in COUNT_FIELDS}, "stores": Counter(), "loads": Counter()}
    sig: Counter = Counter()
    for ordinal, f in enumerate(functions):
        epoch = 0
        stack: list[int] = []               # merge labels of the open structured constructs
        for op, w in f.code:
            stats["instructions"] += 1
            if op == OP["Label"]:
                while stack and stack[-1] == w[0]:
                    stack.pop()
            elif op in (OP["SelectionMerge"], OP["LoopMerge"]):
                stats["constructs"] += 1
                stack.append(w[0])
            elif op == OP["Store"]:
                cls = pclass.get(w[0])
                if cls in COUNTED_STORES:
                    stats["stores"][storage_name(cls)] += 1
                if cls in MEMORY:
                    epoch += 1
            elif op == OP["CopyMemory"]:
                cls = pclass.get(w[0])
                if cls in COUNTED_STORES:
                    stats["stores"][storage_name(cls)] += 1
                if cls in MEMORY:
                    epoch += 1
                src = pclass.get(w[1])
                if src in MEMORY:
                    stats["loads"][storage_name(src)] += 1
                    sig[(ordinal, "load", storage_name(src), len(stack), epoch - (cls in MEMORY))] += 1
            elif op == OP["ImageWrite"]:
                stats["stores"]["Image"] += 1
                epoch += 1
            elif op == OP["Load"]:
                cls = pclass.get(w[2])
                if cls in MEMORY:
                    stats["loads"][storage_name(cls)] += 1
                    sig[(ordinal, "load", storage_name(cls), len(stack), epoch)] += 1
            elif op == OP["ImageRead"]:
                stats["loads"]["Image"] += 1
                sig[(ordinal, "load", "Image", len(stack), epoch)] += 1
            elif op in ATOMIC_OPS:
                stats["atomics"] += 1
                epoch += 1
            elif op == OP["ControlBarrier"]:
                stats["barriers"] += 1
                epoch += 1
            elif op == OP["MemoryBarrier"]:
                stats["memory_barriers"] += 1
                epoch += 1
            elif op in SUBGROUP_OPS:
                stats["subgroup"] += 1
                sig[(ordinal, "subgroup", NAMES.get(op, str(op)), len(stack), 0)] += 1
            elif op in DERIVATIVE_OPS:
                sig[(ordinal, "derivative", str(op), len(stack), 0)] += 1
            elif op == OP["FunctionCall"]:
                writes, reads = effect.get(w[2], (True, True))
                if writes:
                    stats["calls_write"] += 1
                    epoch += 1
                elif reads:
                    stats["calls_read"] += 1
                    sig[(ordinal, "load", "call", len(stack), epoch)] += 1
                else:
                    stats["calls_pure"] += 1
    stats["stores"] = dict(sorted(stats["stores"].items()))
    stats["loads"] = dict(sorted(stats["loads"].items()))
    stats["sig"] = sorted([list(k) + [n] for k, n in sig.items()])
    return stats


def lost(reference: dict, candidate: dict) -> dict[str, int]:
    """What the candidate has fewer of than the reference, per counted operation (only the positive differences)."""
    out: dict[str, int] = {}
    for f in COUNT_FIELDS:
        if f == "calls_pure":
            continue            # dropping a call that does nothing is dead code, not a loss
        d = reference[f] - candidate[f]
        if d > 0:
            out[f] = d
    for cls, n in reference["stores"].items():
        d = n - candidate["stores"].get(cls, 0)
        if d > 0:
            out[f"stores_{cls}"] = d
    return out


def moved(reference: dict, candidate: dict, kinds: tuple[str, ...] = ("load",), depth: bool = True) -> int:
    """A lower bound on the reads (or the convergent operations) the candidate moved: the excess over the reference in any place.

    `depth=False` leaves the nesting depth out of a place, for a candidate whose structure changed (see compare)."""
    if reference["functions"] != candidate["functions"]:
        raise ValueError("functions_differ")

    def places(stats: dict) -> Counter:
        c: Counter = Counter()
        for r in stats["sig"]:
            if r[1] in kinds:
                key = tuple(r[:5]) if depth else (r[0], r[1], r[2], r[4])
                c[key] += r[5]
        return c

    ref = places(reference)
    return sum(max(0, n - ref.get(key, 0)) for key, n in places(candidate).items())


def compare(reference: dict, candidate: dict) -> dict:
    """The verdict of one pair.

    lost: what the candidate has fewer of. moved: reads moved (None when the functions differ or a write was lost: a removed
    write renumbers the epochs after it). moved_convergent: subgroup operations and derivative instructions moved, None as
    above and None when the structure changed. restructured: the candidate has another number of selection and loop
    constructs, because the optimizer's removal of empty blocks and its flattening of `else { if }` change the nesting depth
    of what they hold without moving it; reads are then compared by epoch alone and convergent operations not at all."""
    gone = lost(reference, candidate)
    restructured = reference["constructs"] != candidate["constructs"]
    reads = convergent = None
    if not gone:
        try:
            reads = moved(reference, candidate, ("load",), depth=not restructured)
            convergent = None if restructured else moved(reference, candidate, ("subgroup", "derivative"))
        except ValueError:
            pass
    return {"lost": gone, "moved": reads, "moved_convergent": convergent, "restructured": restructured,
            "bytes": [reference["bytes"], candidate["bytes"]],
            "instructions": [reference["instructions"], candidate["instructions"]]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("modules", nargs="+")
    ap.add_argument("--compare", action="store_true", help="two modules: reference then candidate")
    args = ap.parse_args(argv)
    try:
        stats = [analyze(open(p, "rb").read()) for p in args.modules]
    except (OSError, ValueError) as e:
        print(f"spvstats: {e}", file=sys.stderr)
        return 2
    if args.compare:
        if len(stats) != 2:
            print("spvstats: --compare takes exactly two modules", file=sys.stderr)
            return 2
        print(json.dumps(compare(stats[0], stats[1])))
        return 0
    for p, s in zip(args.modules, stats):
        s = {k: v for k, v in s.items() if k != "sig"}
        print(p, json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
