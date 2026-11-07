#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
#
# Adapted from LAS Recomp tools/gen.py @ da35b83 (MIT, Copyright (c) 2026 Peiikai).
# Taken from DKB Recomp (https://github.com/Lucas-blip-png/dkb-pc-recomp) at 9d8fa02, recomp/gen.py, with the setjmp
# detection of oot-pc-recomp (https://github.com/philneitz/oot-pc-recomp) at 8e09aaa. Changed in oot2-native:
# architecture D3 (per-function trust, RT_SETJMP and RT_JMP_LEAVE, replaced definitions, the lift report), module
# placement by tools/layout.py, no code patch table (the spin gates come from the table tools/lift/spin_waits.py
# computes, read with --spin-waits, instead of from DKB's patch table), no function-entry profiler.
"""Lift the game's modules to C: function discovery, then parallel emission into chunked C files.

    py -3 tools/lift/gen.py --exefs DIR --out DIR [--modules main,subsdk0,sdk] [--jobs N] [--limit N]
                            [--chunk-fns N] [--chunk-bytes B] [--big-fn B] [--reg-locals N] [--no-callee-saved-trust]
                            [--spin-waits FILE]

--exefs holds the player's decoded modules (main, subsdk0..9, sdk), plain or LZ4 NSO files. --out is a build folder
outside the repository (the C is generated from game code and never leaves the player's machine). Output under --out:

  layout.h, layout.json        module bases (tools/layout.py), from the sizes of every module present
  <mod>/<mod>_NNNN.c           one `void f_<addr>(Cpu *c)` per guest function, about --chunk-bytes of C per file
  <mod>/<mod>_table.c          <mod>_fn_addrs[] (strictly ascending), <mod>_fn_ptrs[], <mod>_fn_count (runtime.h),
                               and <mod>_lifted_end(), which marks where the module's lifted code ends
  <mod>/manifest.json          chunk list, function count, statistics and timings
  lift_report.json             architecture D3: setjmp and longjmp sites, replaced definitions, landing-pad functions
                               and the trust setting of each, unsupported instructions, counts and times (and, with
                               --spin-waits, the number of spin gates, in total and per module)

Function starts are the module seeds (.eh_frame_hdr, exported FUNC symbols, RELATIVE relocation targets), every BL
target, and code addresses materialised by ADR or ADRP+ADD (except an ADR that anchors a jump table: one with a `br`
in the next 16 instructions, or one an ADD register form uses as its base right after). A function is
[start, next start). A direct branch whose target lies outside its function makes the target a new start, until no
new start appears. Chunks hold functions in sorted address order and are sized from a per-instruction estimate of the
C they produce, so their names and contents depend only on the input; a chunk whose text did not change is not
rewritten and keeps its mtime, so the build recompiles only what changed (architecture section 13).

Emission runs in worker processes, one chunk per task. Each worker loads the modules itself, lifts one function at a
time and resolves its jump tables (jumptables.py), so memory per worker does not grow with the module.

Architecture D3. Generated functions keep their hottest registers and their flags in C locals and trust AAPCS64's
callee-saved registers across calls (regs.py, flags.py), except where that premise is false in this game:

- a function that calls setjmp keeps every register in the Cpu; its call becomes RT_SETJMP (cpu.h), it makes no tail
  call and runs RT_JMP_LEAVE before it returns or leaves. setjmp is reached through a PLT stub where a module imports
  it (main) and directly where a module defines it (sdk).
- a function that calls longjmp keeps every register in the Cpu, and the call becomes rt_longjmp.
- a function that contains an exception landing pad (.eh_frame LSDAs) keeps every register in the Cpu.
- a module's own definitions of setjmp, longjmp and the C++ throw entry points are not lifted: their bodies become
  rt_longjmp or rt_replaced (cpu.h), so a direct call inside the defining module reaches the runtime's replacement as
  surely as an import does through the loader's binding. Throws are not unwound: they stop the run with their type.

Spin waits. --spin-waits takes the JSON file that tools/lift/spin_waits.py writes during preparation: the loops of the
player's own modules that only another thread can end, as {module: [{head, back_edges, ...}]} with offsets from the
start of each module's image. The table is derived from game code, so it lives in the build folder and is never
checked in. The direct branch at each listed back-edge is lifted with `rt_spin_gate(c, <head>);` in front of its goto,
which keeps every register and only adds time. A back-edge that is not a direct branch (b, b.cond, cbz / cbnz, tbz /
tbnz) to its head, or whose head lies outside its function, is bad input. Without the option nothing changes.

Exit codes: 0 lifted, 1 a file could not be written or removed, 2 bad input.
"""
import argparse
import bisect
import ctypes
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import flags  # noqa: E402
import jumptables  # noqa: E402
import layout  # noqa: E402  (tools/layout.py)
import regs  # noqa: E402
from check_content import GENERATED_MARKER  # noqa: E402  (tools/check_content.py)
from lifter import hx  # noqa: E402
from module import Module  # noqa: E402
from nso import Nso, NsoError  # noqa: E402

MARKER = GENERATED_MARKER.decode()
TERMINATORS = ('b', 'br', 'ret', 'trap')
MUSTTAIL = '__attribute__((musttail)) return'

# Estimated bytes of C per instruction, by class, for chunking (estimate_sizes()). DKB Recomp's least squares over
# every function of another game's three modules; the median error per function was 7 %.
EST_INT = 62
EST_SIMD = 75
EST_SIMD_LDST = 48
EST_FN = 38          # per function: signature, closing brace
EST_BR_LABELS = 10   # per instruction of a function with a `br`: switch cases, or labels if it falls back

# Guest symbols the runtime replaces (architecture D3), by the name the runtime's HOST_OVERRIDE uses. Several names can
# stand at one address (the sdk defines setjmp, _setjmp and __setjmp as one function).
SETJMP = {'setjmp': 'setjmp', '_setjmp': 'setjmp', '__setjmp': 'setjmp'}
LONGJMP = {'longjmp': 'longjmp', '_longjmp': 'longjmp'}
THROW = {n: n for n in ('__cxa_throw', '__cxa_rethrow', '_Unwind_Resume', '_Unwind_RaiseException',
                        '_Unwind_Resume_or_Rethrow')}
SPECIAL = {**{n: ('setjmp', c) for n, c in SETJMP.items()}, **{n: ('longjmp', c) for n, c in LONGJMP.items()},
           **{n: ('throw', c) for n, c in THROW.items()}}

STT_FUNC = 2
PLT_BR_X17 = 0xD61F0220


def fname(a):
    return f'f_{a:x}'


def header(what):
    return f'// {MARKER}: {what}, generated by tools/lift/gen.py from game code. Do not edit or commit.\n'


# ------------------------------------------------------------------------------------------------ discovery

def _sext(v, n):
    v = v.astype(np.int64)
    return np.where(v & (1 << (n - 1)), v - (1 << n), v)


def discover(mod):
    """Function starts of mod as a sorted int64 array of word indices into mod.words, plus discovery statistics."""
    w = mod.words
    n = len(w)

    def idx(mask):
        return np.flatnonzero(mask).astype(np.int64)

    seeds = (np.asarray(mod.seeds(), dtype=np.int64) - mod.lo) >> 2

    # B and BL: imm26. BL targets are function starts; B targets join the fixpoint below.
    b_all = idx((w & 0x7C000000) == 0x14000000)
    imm = _sext(w[b_all] & 0x03FFFFFF, 26)
    is_bl = (w[b_all] >> 31).astype(bool)
    bl_tgt = b_all[is_bl] + imm[is_bl]
    br_src, br_tgt = [b_all[~is_bl]], [b_all[~is_bl] + imm[~is_bl]]
    # B.cond (the lifter accepts o1 = 1 too), CBZ/CBNZ: imm19 at bit 5. TBZ/TBNZ: imm14 at bit 5.
    for mask, val, nb in ((0xFE000010, 0x54000000, 19), (0x7E000000, 0x34000000, 19), (0x7E000000, 0x36000000, 14)):
        s = idx((w & mask) == val)
        br_src.append(s)
        br_tgt.append(s + _sext((w[s] >> 5) & ((1 << nb) - 1), nb))
    br_src, br_tgt = np.concatenate(br_src), np.concatenate(br_tgt)
    ok = (br_tgt >= 0) & (br_tgt < n)
    br_src, br_tgt = br_src[ok], br_tgt[ok]

    # ADR: pc + imm21. ADRP: page + (imm21 << 12), then ADD (imm, 64-bit, unshifted) of the same register within
    # 8 instructions. Targets must be aligned and in .text.
    def imm21(s):
        v = w[s]
        return _sext(((v >> 5) & 0x7FFFF) << 2 | ((v >> 29) & 3), 21)

    pc0 = np.int64(mod.lo)
    adr = idx((w & 0x9F000000) == 0x10000000)
    # Clang's jump tables add a table entry to an ADR anchor, the first case block, then `br`. That anchor is not a
    # function (it would split the switch): an ADR with a `br` in the next 16 instructions is not a seed, nor is one
    # whose register is the base of an ADD (shifted or extended register) in the next 4 instructions
    # (`adr xA, L; ldrb wO, [..]; add xA, xA, wO, lsl #2`), however far the scheduler moved the `br`.
    br = idx((w & 0xFFFFFC1F) == 0xD61F0000)
    nxt = np.searchsorted(br, adr)
    has_br = nxt < len(br)
    near = np.zeros(len(adr), dtype=bool)
    near[has_br] = br[nxt[has_br]] - adr[has_br] <= 16
    rd = w[adr] & 31
    for k in range(1, 5):
        x = w[np.minimum(adr + k, n - 1)]
        near |= (((x & 0xFF200000) == 0x8B000000) | ((x & 0xFFE00000) == 0x8B200000)) & (((x >> 5) & 31) == rd)
    adr = adr[~near]
    adr_bytes = pc0 + adr * 4 + imm21(adr)
    is_adrp = (w & 0x9F000000) == 0x90000000
    add = idx((w & 0xFFC00000) == 0x91000000)
    rn = ((w[add] >> 5) & 31).astype(np.int64)
    found = np.full(len(add), -1, dtype=np.int64)
    for k in range(1, 9):
        p = add - k
        cand = (found < 0) & (p >= 0)
        pc = np.where(cand, p, 0)
        cand &= is_adrp[pc] & ((w[pc] & 31).astype(np.int64) == rn)
        found[cand] = p[cand]
    hit = found >= 0
    p = found[hit]
    page = ((pc0 + p * 4) & ~np.int64(0xFFF)) + (imm21(p) << 12)
    add_bytes = page + ((w[add[hit]] >> 10) & 0xFFF).astype(np.int64)
    mat = np.concatenate([adr_bytes, add_bytes])
    mat = mat[(mat >= mod.lo) & (mat < mod.hi) & (mat % 4 == 0)]
    mat = (mat - pc0) >> 2

    bl_tgt = bl_tgt[(bl_tgt >= 0) & (bl_tgt < n)]
    starts = np.unique(np.concatenate([seeds, bl_tgt, mat]))
    stats = {'seeds': int(len(seeds)), 'bl_targets': int(len(np.unique(bl_tgt))),
             'materialised_targets': int(len(np.unique(mat))), 'initial_starts': int(len(starts))}

    rounds = 0
    while True:
        rounds += 1
        f = np.searchsorted(starts, br_src, side='right') - 1
        live = f >= 0
        f = np.where(live, f, 0)
        lo = starts[f]
        hi = np.where(f + 1 < len(starts), starts[np.minimum(f + 1, len(starts) - 1)], n)
        out = live & ((br_tgt < lo) | (br_tgt >= hi))
        new = np.setdiff1d(br_tgt[out], starts, assume_unique=False)
        if not len(new):
            break
        starts = np.union1d(starts, new)
    stats['rounds'] = rounds
    stats['functions'] = int(len(starts))
    return starts, stats


def estimate_sizes(mod, starts):
    """Estimated bytes of C per function (for chunking only; the manifest records the real sizes)."""
    w = mod.words
    cost = np.full(len(w), EST_INT, dtype=np.int64)
    cost[(w & 0x0E000000) == 0x0E000000] = EST_SIMD                    # FP/AdvSIMD data processing
    cost[((w & 0x0A000000) == 0x08000000) & (((w >> 26) & 1) == 1)] = EST_SIMD_LDST  # FP/SIMD load/store
    has_br = np.zeros(len(w), dtype=np.int64)
    has_br[np.flatnonzero((w & 0xFFFFFC1F) == 0xD61F0000)] = 1           # BR Xn
    csum = np.concatenate([[0], np.cumsum(cost)])
    bsum = np.concatenate([[0], np.cumsum(has_br)])
    ends = np.append(starts[1:], len(w))
    size = csum[ends] - csum[starts] + EST_FN
    with_br = (bsum[ends] - bsum[starts]) > 0
    size[with_br] += (ends[with_br] - starts[with_br]) * EST_BR_LABELS
    return size


def partition(est, max_fns, max_bytes, big_fn):
    """Contiguous [i0, i1) function ranges: at most max_fns functions and about max_bytes of C each; a function
    estimated above big_fn gets a file of its own so large functions compile in parallel."""
    chunks, i0, size = [], 0, 0
    for i, s in enumerate(est.tolist()):
        if s > big_fn:
            if i > i0:
                chunks.append((i0, i))
            chunks.append((i, i + 1))
            i0, size = i + 1, 0
            continue
        if i > i0 and (i - i0 >= max_fns or size + s > max_bytes):
            chunks.append((i0, i))
            i0, size = i, 0
        size += s
    if len(est) > i0:
        chunks.append((i0, len(est)))
    return chunks


# ------------------------------------------------------------------------------------------------ D3 targets

def special_targets(mod):
    """{address: (kind, runtime name, how, guest symbol)} for setjmp, longjmp and the throw entry points: 'plt' for an
    import's PLT stub (adrp x16; ldr x17, [x16, #slot]; add x16; br x17 whose GOT slot relocates to the symbol),
    'definition' for a FUNC symbol the module defines. Sorted by address, then name, so the result does not depend
    on the order of the symbol table."""
    nso = mod.nso
    found = {}
    for name, info, shndx, value, _ in sorted(nso.syms, key=lambda s: (s[3], s[0])):
        if name in SPECIAL and shndx and (info & 0xF) == STT_FUNC and mod.in_text(mod.base + value):
            kind, canon = SPECIAL[name]
            found.setdefault(mod.base + value, (kind, canon, 'definition', name))
    slots = {}
    for off, _, sym, _ in nso.relocs:
        if sym and sym < len(nso.syms) and nso.syms[sym][0] in SPECIAL:
            slots[mod.base + off] = nso.syms[sym][0]
    if slots:
        for idx in np.flatnonzero(mod.words == PLT_BR_X17).tolist():
            if idx < 3:
                continue
            pc = mod.lo + (idx - 3) * 4
            adrp, ldr = mod.word(pc), mod.word(pc + 4)
            if adrp & 0x9F00001F != 0x90000010 or ldr & 0xFFC003FF != 0xF9400211:
                continue
            imm = ((adrp >> 5 & 0x7FFFF) << 2) | (adrp >> 29 & 3)
            if imm & (1 << 20):
                imm -= 1 << 21
            slot = (pc & ~0xFFF) + (imm << 12) + ((ldr >> 10 & 0xFFF) * 8)
            name = slots.get(slot)
            if name:
                kind, canon = SPECIAL[name]
                found.setdefault(pc, (kind, canon, 'plt', name))
    return dict(sorted(found.items()))


# ------------------------------------------------------------------------------------------------ emission

class LiftError(Exception):
    pass


class SpinTableError(LiftError):
    """A spin wait table (--spin-waits) that is malformed or does not fit the module: bad input."""


def load_spin_waits(path):
    """{module: {back-edge offset: head offset}} from the JSON file tools/lift/spin_waits.py writes."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        raise SpinTableError(f'{path}: {e}') from e
    if not isinstance(data, dict):
        raise SpinTableError(f'{path}: expected a JSON object of modules')

    def offset(v):
        return isinstance(v, int) and not isinstance(v, bool) and v >= 0

    out = {}
    for name, regions in data.items():
        if not isinstance(regions, list):
            raise SpinTableError(f'{path}: {name}: expected a list of spin waits')
        mod = out.setdefault(name, {})
        for r in regions:
            head, bes = (r.get('head'), r.get('back_edges')) if isinstance(r, dict) else (None, None)
            if not offset(head) or not isinstance(bes, list) or not bes or not all(offset(b) for b in bes):
                raise SpinTableError(f'{path}: {name}: a spin wait needs an integer head and a list of integer '
                                     f'back_edges')
            for b in bes:
                if b in mod:
                    raise SpinTableError(f'{path}: {name}+{b:#x} is a back-edge twice')
                mod[b] = head
    return out


def check_spins(mod, starts, spins):
    """Stops on a spin wait back-edge (absolute address; spins is {back-edge: head}) that is not a direct branch to
    its head inside the function that holds it. starts: every function start of the module."""
    ordered = sorted(starts)
    for a, head in sorted(spins.items()):
        where = f'{mod.name}+{a - mod.base:#x}'
        ins = mod.lift(a) if mod.in_text(a) else None
        if ins is None or ins.kind not in ('b', 'bcond') or ins.target != head:
            raise SpinTableError(f'{where}: not a direct branch to the spin wait head {mod.name}+{head - mod.base:#x}')
        i = bisect.bisect_right(ordered, a) - 1
        lo = ordered[i] if i >= 0 else mod.lo
        hi = ordered[i + 1] if i + 1 < len(ordered) else mod.hi
        if not lo <= head < hi:
            raise SpinTableError(f'{where}: the spin wait head {mod.name}+{head - mod.base:#x} lies outside the '
                                 f'function that holds the back-edge')


class Emitter:
    """C for the functions of one module. starts is the set of every function start of the module. spins:
    {back-edge address: its spin wait head} (load_spin_waits), checked against the module here."""

    def __init__(self, mod, starts, reg_locals=regs.LOCALS, callee_saved=True, spins=None):
        self.mod = mod
        self.starts = starts
        self.spins = spins or {}
        if self.spins:
            check_spins(mod, starts, self.spins)
        self.reg_locals = reg_locals   # regs.py: registers per function in C locals
        self.kept = regs.CALLEE_SAVED if callee_saved else frozenset()   # not reloaded after a call
        self.special = special_targets(mod)
        self.kinds = {a: k for a, (k, _, _, _) in self.special.items()}
        self.pads = sorted(jumptables.landing_pads(mod))
        self.stats = Counter()
        self.unsupported = Counter()
        self.d3 = []   # one entry per function D3 concerns, for the lift report

    def scan(self, insns, entry, end):
        """Labels and jump tables of one function: {pc: sorted targets} for resolved tables, 'all' for an
        unresolved computed br."""
        mod, labels, tables = self.mod, set(), {}
        for i, ins in enumerate(insns):
            k = ins.kind
            if k in ('b', 'bcond'):
                if entry <= ins.target < end:
                    labels.add(ins.target)
            elif k == 'br':
                a = entry + 4 * i
                jt = jumptables.resolve(mod, a, entry, end)
                if jt:
                    tables[a] = jt
                    labels.update(jt)
                    self.stats['jump_tables'] += 1
                    self.stats['jump_table_targets'] += len(jt)
                elif jumptables.is_computed(mod, a, entry):
                    tables[a] = 'all'
                    self.stats['all_label_fallbacks'] += 1
                else:
                    self.stats['br_dispatch'] += 1
            elif k == 'trap':
                self.stats['traps'] += 1
                c = ins.code
                if c.startswith('rt_udf') and '/*' in c:
                    self.stats['unsupported'] += 1
                    msg = c[c.index('/*') + 2:c.rindex('*/')].strip()
                    self.unsupported[re.sub(r'\b[0-9a-f]{8}\b', '', msg).strip()] += 1
        return labels, tables

    @staticmethod
    def flow(insns, entry, end, tables):
        """Successors of each instruction inside the function [entry, end). Index len(insns) stands for falling
        through into the next function and has none."""
        def idx(t):
            return [(t - entry) // 4] if entry <= t < end else []

        succ = []
        for i, ins in enumerate(insns):
            k = ins.kind
            if k in ('op', 'bl', 'blr'):
                succ.append([i + 1])
            elif k == 'b':
                succ.append(idx(ins.target))
            elif k == 'bcond':
                succ.append(idx(ins.target) + [i + 1])
            elif k == 'br':
                jt = tables.get(entry + 4 * i)
                succ.append([t for x in jt for t in idx(x)] if jt and jt != 'all' else [])
            else:   # ret, trap
                succ.append([])
        succ.append([])
        return succ

    @staticmethod
    def flag_plan(insns, entry, end, succ):
        """flags.py plan for the function: each instruction's flag use over the flow succ (flow())."""
        n = len(insns)
        calls, effects, exits = set(), [], {n}
        for i, ins in enumerate(insns):
            k = ins.kind
            if k in ('bl', 'blr'):
                calls.add(i)
            elif k in ('b', 'bcond') and not entry <= ins.target < end:
                exits.add(i)
            elif k == 'br':
                exits.add(i)   # through rt_dispatch, also a table's default
            effects.append(flags.effect(ins.cond if k == 'bcond' else ins.code or ''))
        effects.append((False, None))
        return flags.plan(effects, succ, calls, exits)

    def replaced(self, entry):
        """Body of a module's own definition of a symbol the runtime replaces (D3)."""
        kind, canon, _, sym = self.special[entry]
        self.stats['replaced_definitions'] += 1
        self.d3.append({'function': hx(entry), 'kind': 'replaced', 'symbol': sym, 'by': canon})
        call = ('rt_longjmp(c, c->x[0], (uint32_t)c->x[1]);' if kind == 'longjmp'
                else f'c->pc = {hx(entry)}; rt_replaced(c, "{canon}");')
        return [f'void {fname(entry)}(Cpu *c) {{   /* {sym}: replaced by the runtime (D3) */', f'    {call}', '}']

    def function(self, entry, end):
        """(lines of C, referenced function starts) for the guest function [entry, end)."""
        mod, starts = self.mod, self.starts
        self.stats['functions'] += 1
        if self.special.get(entry, (None, None, None))[2] == 'definition':
            return self.replaced(entry), set()
        insns = [mod.lift(a) for a in range(entry, end, 4)]
        self.stats['instructions'] += len(insns)
        labels, tables = self.scan(insns, entry, end)
        refs = set()
        all_labels = 'all' in tables.values()
        if all_labels:
            labels = set(range(entry, end, 4))

        kinds = self.kinds
        sj = [entry + 4 * i for i, ins in enumerate(insns)
              if ins.kind in ('bl', 'b') and kinds.get(ins.target) == 'setjmp']
        lj = [entry + 4 * i for i, ins in enumerate(insns)
              if ins.kind in ('bl', 'b') and kinds.get(ins.target) == 'longjmp']
        pads = self.pads[bisect.bisect_left(self.pads, entry):bisect.bisect_left(self.pads, end)]
        for a in sj:
            if insns[(a - entry) // 4].kind == 'b':
                raise LiftError(f'{a:#x}: a tail branch to setjmp; setjmp must return to its caller')
        has_setjmp = bool(sj)
        if has_setjmp and all_labels:
            raise LiftError(f'{entry:#x}: setjmp in a function with an unresolved computed branch')
        # D3: a longjmp lands in a setjmp's frame with the guest registers of the setjmp, and an unwinder would land
        # in a landing pad with the registers of the throw, so neither can rely on locals that held other values.
        # An unresolved computed br re-enters the function through fb_<entry>, a C function of its own, which locals
        # would not survive either.
        local = not (all_labels or sj or lj or pads)
        if sj or lj or pads:
            self.d3.append({'function': hx(entry), 'end': hx(end),
                            'setjmp_sites': [hx(a) for a in sj], 'longjmp_sites': [hx(a) for a in lj],
                            'landing_pads': len(pads), 'trust': 'off: every register in the Cpu'})
            self.stats['setjmp_functions'] += bool(sj)
            self.stats['setjmp_sites'] += len(sj)
            self.stats['longjmp_functions'] += bool(lj)
            self.stats['longjmp_sites'] += len(lj)
            self.stats['landing_pad_functions'] += bool(pads)
            self.stats['landing_pads'] += len(pads)
        spill, reload, reload_call = (regs.SPILL, regs.RELOAD, regs.RELOAD_CALL) if local else ('', '', '')
        succ = self.flow(insns, entry, end, tables) if local else None
        fp = self.flag_plan(insns, entry, end, succ) if local else None
        if fp and fp.used:
            self.stats['local_flag_functions'] += 1
            self.stats['flag_reads_direct'] += sum(1 for k in fp.reader if k and k != 'nz')
            self.stats['flag_reads_packed'] += sum(1 for k in fp.reader if k == 'nz')
            self.stats['flag_entry_functions'] += fp.entry
            self.stats['flag_entry_reads'] += fp.entry_read
            self.stats['flag_reads_after_call'] += fp.after_call
            self.stats['flag_publishing_exits'] += sum(1 for k in fp.exit if k)

        def fx(i, code):
            return fp.rewrite(i, code) if fp and fp.used else code

        def publish(i):
            return fp.publish(i) if fp and fp.used else ''

        def leave(call):
            """Leave the function through call: a tail call, except in a setjmp function, which runs RT_JMP_LEAVE
            first and then calls and returns (a tail call would reuse the frame RT_SETJMP's record names)."""
            if has_setjmp:
                return f'RT_JMP_LEAVE(c); {call}; return;'
            return f'{MUSTTAIL} {call};'

        def longjmp():
            self.stats['longjmp_calls'] += 1
            return 'rt_longjmp(c, c->x[0], (uint32_t)c->x[1]);'

        def jump(t, i):
            if entry <= t < end:
                return f'goto L_{t:x};'
            if kinds.get(t) == 'longjmp':
                return longjmp()
            if t in starts:
                refs.add(t)
                return f'{publish(i)}{spill}{leave(f"{fname(t)}(c)")}'
            self.stats['external_branches'] += 1
            return f'{publish(i)}{spill}c->pc = {hx(t)}; {leave("rt_dispatch(c)")}'

        if all_labels:
            # An unresolved computed br re-enters the function through fb_<entry>, which switches on c->pc to any
            # instruction. A goto to a shared dispatch block inside the body instead turns every instruction into
            # a loop header, and clang's loop passes then take tens of seconds on one function.
            fb = f'fb_{entry:x}'
            cases = ' '.join(f'case {hx(t)}: goto L_{t:x};' for t in range(entry, end, 4))
            redispatch = f'{MUSTTAIL} {fb}(c);'
            out = [f'static void {fb}(Cpu *c);',
                   f'void {fname(entry)}(Cpu *c) {{', f'    c->pc = {hx(entry)};', f'    {redispatch}', '}',
                   f'static void {fb}(Cpu *c) {{', f'    switch (c->pc) {{ {cases} default: break; }}',
                   f'    {MUSTTAIL} rt_dispatch(c);']
        else:
            redispatch = leave('rt_dispatch(c)')
            # A setjmp function is never inlined: RT_SETJMP and RT_JMP_LEAVE name the frame it runs in.
            attr = '__attribute__((noinline)) ' if has_setjmp else ''
            out = [f'{attr}void {fname(entry)}(Cpu *c) {{']
            if fp and fp.used:
                out.append('    ' + fp.decl())

        def call(t, a):
            if kinds.get(t) == 'setjmp':
                self.stats['setjmp_calls'] += 1
                return f'c->x[30] = {hx(a + 4)}; c->x[0] = RT_SETJMP(c, c->x[0]);'
            if kinds.get(t) == 'longjmp':
                return f'c->x[30] = {hx(a + 4)}; {longjmp()}'
            if t in starts:
                refs.add(t)
                return f'c->x[30] = {hx(a + 4)}; {spill}{fname(t)}(c);{reload_call}'
            self.stats['external_calls'] += 1
            return f'c->x[30] = {hx(a + 4)}; {spill}c->pc = {hx(t)}; rt_dispatch(c);{reload_call}'

        def op(i, code):
            code = fx(i, code)
            return f'{spill}{code}{reload}' if regs.ESCAPE.search(code) else code

        body = []   # (instruction index or None for a label, line); a line belongs to the instruction it lifts

        def emit(i, line):
            body.append((i, line))

        k = None
        for i, ins in enumerate(insns):
            a = entry + 4 * i
            if a in labels:
                emit(None, f'L_{a:x}:;')
            k = ins.kind
            if k == 'op':
                emit(i, '    ' + op(i, ins.code))
            elif k == 'trap':
                emit(i, f'    {spill}{fx(i, ins.code)}')
            elif k in ('b', 'bcond'):
                j = jump(ins.target, i)
                if self.spins.get(a) == ins.target and entry <= ins.target < end:
                    j = f'rt_spin_gate(c, {hx(ins.target)}); {j}'   # a pass of a spin wait
                    self.stats['spin_gates'] += 1
                emit(i, '    ' + j if k == 'b' else f'    if ({fx(i, ins.cond)}) {{ {j} }}')
            elif k == 'bl':
                emit(i, '    ' + call(ins.target, a))
            elif k == 'blr':
                emit(i, f'    c->pc = {ins.target}; c->x[30] = {hx(a + 4)}; {spill}rt_dispatch(c);{reload_call}')
            elif k == 'br':
                jt = tables.get(a)
                if jt == 'all':
                    emit(i, f'    c->pc = {ins.target}; {redispatch}')
                elif jt:
                    cases = ' '.join(f'case {hx(t)}: goto L_{t:x};' for t in jt)
                    emit(i, f'    switch ({ins.target}) {{ {cases} default: break; }}')
                    emit(i, f'    {publish(i)}{spill}c->pc = {ins.target}; {redispatch}')
                else:
                    emit(i, f'    {publish(i)}{spill}c->pc = {ins.target}; {leave("rt_dispatch(c)")}')
            elif k == 'ret':
                emit(i, f'    {spill}RT_JMP_LEAVE(c); return;' if has_setjmp else f'    {spill}return;')
            else:
                raise LiftError(f'{a:#x}: unknown instruction kind {k!r}')
        n = len(insns)
        if k not in TERMINATORS:  # falls through into the next function
            if end < mod.hi:
                emit(n, '    ' + jump(end, n))
            else:
                emit(n, f'    {spill}rt_udf(c, {hx(end)});')
        if local:
            texts = [''] * (n + 1)
            for i, line in body:
                if i is not None:
                    texts[i] += line
            clean = {i for i, ins in enumerate(insns)
                     if ins.kind in ('bl', 'blr') or (ins.kind == 'op' and regs.RELOAD in texts[i])}
            rp = regs.plan(texts, succ, clean, self.reg_locals, self.kept)
            if rp.regs:
                out.append('    ' + rp.decl())
            self.stats['local_reg_functions'] += bool(rp.regs)
            self.stats['reg_spills'] += sum(len(s) for s in rp.spill)
            out += [line if i is None else rp.resolve(i, line) for i, line in body]
        else:
            out += [line for _, line in body]
        out.append('}')
        return out, refs


def chunk_text(name, number, emitter, fns):
    """C text of one chunk. fns: [(entry, end)]."""
    body, refs = [], set()
    for entry, end in fns:
        lines, r = emitter.function(entry, end)
        body.append('\n'.join(lines))
        refs |= r
    # Every referenced function gets a prototype, including ones defined later in this chunk.
    protos = ''.join(f'void {fname(r)}(Cpu *c);\n' for r in sorted(refs))
    return (header(f'{name} chunk {number:04d}') + '#include "oot2/lift.h"\n\n' + protos + '\n'
            + '\n\n'.join(body) + '\n')


def write_if_changed(path, text):
    """Returns True if the file was written. An unchanged file keeps its mtime, so the build skips it. A changed one
    is written under a temporary name and renamed into place, so a reader never sees it half written."""
    data = text.encode()
    try:
        if path.stat().st_size == len(data) and path.read_bytes() == data:
            return False
    except FileNotFoundError:
        pass
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_bytes(data)
    try:
        _retry(os.replace, tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return True


def _retry(fn, *args):
    """Windows refuses to rewrite or delete a file another process has open or mapped (clang maps its sources, so a
    build running on the output holds them; so can a scanner or an editor). Waits up to 30 s for it."""
    for i in range(120):
        try:
            return fn(*args)
        except OSError as e:
            if i == 119 or isinstance(e, FileNotFoundError):
                raise
            time.sleep(0.25)


# ------------------------------------------------------------------------------------------------ workers

_W = {}  # worker state: 'mods' {name: (path, base)}, 'starts' {name: sorted array}, 'emitters' {name: Emitter}


def _worker_init(mods, starts, opts, spins=None):
    _W['mods'] = mods
    _W['starts'] = starts
    _W['opts'] = opts   # Emitter keyword arguments
    _W['spins'] = spins or {}   # {module: {back-edge address: head address}}
    _W['emitters'] = {}


def _emitter(name):
    e = _W['emitters'].get(name)
    if e is None:
        path, base = _W['mods'][name]
        e = _W['emitters'][name] = Emitter(Module(name, path, base), set(_W['starts'][name].tolist()),
                                           spins=_W['spins'].get(name), **_W['opts'])
    return e


def _emit_chunk(task):
    """Worker: write one chunk. Returns its statistics."""
    name, number, path, fns = task
    t0 = time.perf_counter()
    e = _emitter(name)
    e.stats.clear()
    e.unsupported.clear()
    e.d3 = []
    text = chunk_text(name, number, e, fns)
    error = None
    try:
        written = write_if_changed(Path(path), text)
    except OSError as ex:
        written, error = False, f'{path}: {ex}'
    return {'module': name, 'number': number, 'file': Path(path).name, 'functions': len(fns), 'bytes': len(text),
            'written': written, 'error': error, 'seconds': time.perf_counter() - t0, 'stats': dict(e.stats),
            'unsupported': dict(e.unsupported), 'd3': e.d3}


# ------------------------------------------------------------------------------------------------ driver

# The end of a module's lifted code (ticket 92). The table is the last object of its module in the link, so this
# function lies after every lifted function and static helper of the module; its address is the first byte of host
# code that is not the module's. The runtime refuses a function at or past it (dispatch.c), so a link order that moved
# the table is an error at start, not a silent misreading of host code as guest code. noinline and used keep it out of
# line and in the link. Every module's marker has the same bytes, so the build must never fold identical code (no
# --icf): a folded marker would move one module's end onto another's.
END_MARKER_TEXT = (
    '\n__attribute__((noinline, used)) void {name}_lifted_end(Cpu *c) {{\n'
    '    __asm__ volatile("int3" : : "r"(c) : "memory");   // the module\'s lifted code ends before this function\n'
    '}}\n')


def end_marker_name(name):
    return f'{name}_lifted_end'


def table_text(name, addrs):
    lines = [header(f'{name} function table'), '#include "oot2/cpu.h"\n\n']
    lines += [f'void {fname(a)}(Cpu *c);\n' for a in addrs]
    lines.append(f'\nconst uint64_t {name}_fn_addrs[] = {{\n')
    lines.append(',\n'.join(hx(a) for a in addrs))
    lines.append(f'\n}};\nconst GuestFn {name}_fn_ptrs[] = {{\n')
    lines.append(',\n'.join(fname(a) for a in addrs))
    lines.append(f'\n}};\nconst uint64_t {name}_fn_count = {len(addrs)};\n')
    lines.append(END_MARKER_TEXT.format(name=name))
    return ''.join(lines)


def remove_stale(moddir, name, nchunks, errors):
    """Delete chunks an earlier run wrote beyond this run's count, and temporary files an interrupted run left."""
    pat = re.compile(re.escape(name) + r'_(\d{4})\.c')
    removed = 0
    for p in sorted(moddir.iterdir()):
        m = pat.fullmatch(p.name)
        if (m and int(m.group(1)) >= nchunks) or (p.name.startswith('.') and p.name.endswith('.tmp')):
            try:
                _retry(p.unlink)
                removed += 1
            except OSError as ex:
                errors.append(f'{p}: {ex}')
    return removed


# Peak memory of one emit worker on DKB Recomp's largest module is 1.3 GiB (1.5 GiB with margin).
WORKER_MEM = 1536 << 20
MEM_SHARE = 0.8  # of the free memory the workers may take, the rest is for the system and other programs


def available_ram():
    """Free physical memory in bytes, or None when it cannot be read."""
    if sys.platform == 'win32':
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        st = MemoryStatusEx()
        st.dwLength = ctypes.sizeof(st)
        return st.ullAvailPhys if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)) else None
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def pick_jobs(requested):
    """Worker count and the reason for it: --jobs if given, else the CPU count capped by the free memory."""
    if requested:
        return requested, '--jobs'
    cpus = os.cpu_count() or 4
    free = available_ram()
    if free is None:
        return cpus, f'{cpus} CPUs, free memory unknown'
    by_mem = max(1, int(free * MEM_SHARE) // WORKER_MEM)
    return min(cpus, by_mem), f'{cpus} CPUs, {free / 2**30:.1f} GiB free memory allows {by_mem}'


def place_modules(exefs):
    """[(name, path, base)] of the modules present in exefs, placed by tools/layout.py from their sizes."""
    sizes, paths = {}, {}
    for name in layout.ORDER:
        path = exefs / name
        if path.is_file():
            sizes[name] = Nso(str(path)).size
            paths[name] = path
    mods = layout.place(sizes)
    return [(m.name, paths[m.name], m.base) for m in mods], mods


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--exefs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--modules', default='main,subsdk0,sdk')
    ap.add_argument('--jobs', type=int, help='emit workers (default: by free memory and CPU count)')
    ap.add_argument('--limit', type=int, default=0, help='emit only the first N functions of each module')
    ap.add_argument('--chunk-fns', type=int, default=3000, help='functions per chunk at most')
    ap.add_argument('--chunk-bytes', type=int, default=3000000, help='estimated bytes of C per chunk')
    ap.add_argument('--big-fn', type=int, default=300000, help='estimated bytes of C from which a function gets '
                                                               'its own chunk')
    ap.add_argument('--reg-locals', type=int, default=regs.LOCALS,
                    help=f'guest registers per function kept in C locals (regs.py; default {regs.LOCALS}, 0: none)')
    ap.add_argument('--no-callee-saved-trust', action='store_true',
                    help='reload X19-X29 and SP after every call too, instead of trusting AAPCS64 (slower; for '
                         'telling a broken callee from a generator bug)')
    ap.add_argument('--spin-waits', type=Path, metavar='FILE',
                    help='the JSON file tools/lift/spin_waits.py wrote: lift the back-edges of the spin waits it lists '
                         'through rt_spin_gate')
    args = ap.parse_args(argv)
    opts = {'reg_locals': args.reg_locals, 'callee_saved': not args.no_callee_saved_trust}

    t0 = time.perf_counter()
    try:
        placed, mods = place_modules(args.exefs)
    except (OSError, NsoError, ValueError) as e:
        print(f'gen: {e}', file=sys.stderr)
        return 2
    want = args.modules.split(',')
    unknown = set(want) - {n for n, _, _ in placed}
    if unknown:
        print(f'gen: not in {args.exefs}: {", ".join(sorted(unknown))}', file=sys.stderr)
        return 2
    spin_table = {}   # {module: {back-edge offset: head offset}}
    if args.spin_waits:
        try:
            spin_table = load_spin_waits(args.spin_waits)
        except SpinTableError as e:
            print(f'gen: {e}', file=sys.stderr)
            return 2
        stray = set(spin_table) - {n for n, _, _ in placed}
        if stray:
            print(f'gen: {args.spin_waits}: not in {args.exefs}: {", ".join(sorted(stray))}', file=sys.stderr)
            return 2
    args.out.mkdir(parents=True, exist_ok=True)
    layout.write(mods, args.out)

    paths, starts_abs, info, tasks, spins_abs = {}, {}, {}, [], {}
    for name, path, base in placed:
        if name not in want:
            continue
        t = time.perf_counter()
        mod = Module(name, path, base)
        starts, dstats = discover(mod)
        est = estimate_sizes(mod, starts)
        abs_starts = mod.lo + starts * 4
        n = len(starts) if not args.limit else min(args.limit, len(starts))
        ends = np.append(abs_starts[1:], mod.hi)
        spins = {base + b: base + h for b, h in spin_table.get(name, {}).items()}
        if spins:
            try:
                check_spins(mod, abs_starts.tolist(), spins)
            except SpinTableError as e:
                print(f'gen: {args.spin_waits}: {e}', file=sys.stderr)
                return 2
        spins_abs[name] = spins
        chunks = partition(est[:n], args.chunk_fns, args.chunk_bytes, args.big_fn)
        moddir = args.out / name
        moddir.mkdir(parents=True, exist_ok=True)
        for number, (i0, i1) in enumerate(chunks):
            fns = list(zip(abs_starts[i0:i1].tolist(), ends[i0:i1].tolist()))
            tasks.append((int(est[i0:i1].sum()), (name, number, str(moddir / f'{name}_{number:04d}.c'), fns)))
        special = special_targets(mod)
        paths[name] = (str(path), base)
        starts_abs[name] = abs_starts
        info[name] = {'base': base, 'lo': mod.lo, 'hi': mod.hi, 'emitted': abs_starts[:n].tolist(),
                      'chunks': len(chunks), 'discover': dstats, 'discover_seconds': time.perf_counter() - t,
                      'moddir': moddir, 'build_id': mod.nso.build_id,
                      'special': [{'address': hx(a), 'kind': k, 'runtime': c, 'how': h, 'symbol': s}
                                  for a, (k, c, h, s) in special.items()]}
        print(f'{name}: {dstats["functions"]} functions ({dstats["seeds"]} seeds, {dstats["rounds"]} rounds), '
              f'{len(chunks)} chunks, discovery {time.perf_counter() - t:.1f}s', file=sys.stderr, flush=True)
        del mod

    # Largest chunks first, so the pool does not end on a long tail.
    tasks.sort(key=lambda t: (-t[0], t[1][0], t[1][1]))
    results = {name: [] for name in info}
    done_at, errors = {}, []
    t_emit = time.perf_counter()
    ctx = mp.get_context('spawn')
    jobs, why = pick_jobs(args.jobs)
    print(f'emitting with {jobs} workers ({why})', file=sys.stderr, flush=True)
    with ctx.Pool(jobs, initializer=_worker_init, initargs=(paths, starts_abs, opts, spins_abs)) as pool:
        for i, r in enumerate(pool.imap_unordered(_emit_chunk, [t for _, t in tasks], chunksize=1)):
            results[r['module']].append(r)
            if r['error']:
                errors.append(r['error'])
            done_at[r['module']] = time.perf_counter() - t_emit
            if (i + 1) % 200 == 0 or i + 1 == len(tasks):
                print(f'emitted {i + 1}/{len(tasks)} chunks, {time.perf_counter() - t_emit:.0f}s', file=sys.stderr,
                      flush=True)
    emit_wall = time.perf_counter() - t_emit

    report = {'marker': MARKER, 'options': {**opts, 'jobs': jobs, 'jobs_reason': why, 'limit': args.limit},
              'modules': {}}
    for name, d in info.items():
        rs = sorted(results[name], key=lambda r: r['number'])
        moddir = d['moddir']
        stats, unsupported = Counter(), Counter()
        d3 = []
        for r in rs:
            stats.update(r['stats'])
            unsupported.update(r['unsupported'])
            d3 += r['d3']
        d3.sort(key=lambda e: int(e['function'][:-3], 16))
        t = time.perf_counter()
        table_written = write_if_changed(moddir / f'{name}_table.c', table_text(name, d['emitted']))
        removed = remove_stale(moddir, name, d['chunks'], errors)
        keys = ('functions', 'instructions', 'jump_tables', 'jump_table_targets', 'all_label_fallbacks', 'br_dispatch',
                'traps', 'unsupported', 'external_branches', 'external_calls', 'local_flag_functions',
                'flag_reads_direct', 'flag_reads_packed', 'flag_entry_functions', 'flag_entry_reads',
                'flag_reads_after_call', 'flag_publishing_exits', 'local_reg_functions', 'reg_spills',
                'setjmp_functions', 'setjmp_sites', 'setjmp_calls', 'longjmp_functions', 'longjmp_sites',
                'longjmp_calls', 'landing_pad_functions', 'landing_pads', 'replaced_definitions')
        if args.spin_waits:
            keys += ('spin_gates',)
        manifest = {
            'module': name,
            'marker': MARKER,
            'base': f'{d["base"]:#x}',
            'text': [f'{d["lo"]:#x}', f'{d["hi"]:#x}'],
            'function_count': len(d['emitted']),
            'chunks': [r['file'] for r in rs],
            'table': f'{name}_table.c',
            'end_marker': end_marker_name(name),
            'chunk_info': [{'file': r['file'], 'functions': r['functions'], 'bytes': r['bytes'],
                            'seconds': round(r['seconds'], 2)} for r in rs],
            'stats': {
                **d['discover'],
                **{('emitted_functions' if k == 'functions' else k): stats[k] for k in keys},
                'c_bytes': sum(r['bytes'] for r in rs),
                'largest_chunk_bytes': max((r['bytes'] for r in rs), default=0),
                'chunks_written': sum(r['written'] for r in rs) + table_written,
                'stale_chunks_removed': removed,
            },
            'unsupported_census': dict(sorted(unsupported.items(), key=lambda kv: (-kv[1], kv[0]))[:50]),
            'timings': {
                'discover_s': round(d['discover_seconds'], 2),
                'emit_wall_s': round(done_at.get(name, 0.0), 2),
                'emit_worker_s': round(sum(r['seconds'] for r in rs), 2),
                'table_s': round(time.perf_counter() - t, 2),
            },
        }
        write_if_changed(moddir / 'manifest.json', json.dumps(manifest, indent=1) + '\n')
        report['modules'][name] = {'build_id': d['build_id'], 'functions': len(d['emitted']),
                                   'stats': manifest['stats'], 'unsupported_census': manifest['unsupported_census'],
                                   'special_targets': d['special'], 'd3_functions': d3,
                                   'timings': manifest['timings']}
        if args.spin_waits:
            report['modules'][name]['spin_gates'] = manifest['stats']['spin_gates']
        s = manifest['stats']
        print(f'{name}: {s["emitted_functions"]} functions, {len(rs)} chunks, {s["c_bytes"] / 1e6:.0f} MB of C, '
              f'{s["jump_tables"]} jump tables, {s["all_label_fallbacks"]} all-label fallbacks, '
              f'{s["traps"]} traps ({s["unsupported"]} unsupported); D3: {s["setjmp_sites"]} setjmp sites in '
              f'{s["setjmp_functions"]} functions, {s["longjmp_sites"]} longjmp sites in {s["longjmp_functions"]}, '
              f'{s["landing_pad_functions"]} landing-pad functions, {s["replaced_definitions"]} replaced definitions',
              file=sys.stderr)
        if args.spin_waits:
            print(f'{name}: {s["spin_gates"]} spin gates ({len(spins_abs[name])} back-edges listed)', file=sys.stderr)
    total = time.perf_counter() - t0
    report['timings'] = {'total_s': round(total, 2), 'emit_wall_s': round(emit_wall, 2)}
    report['total_functions'] = sum(m['functions'] for m in report['modules'].values())
    if args.spin_waits:
        report['spin_gates'] = sum(m['spin_gates'] for m in report['modules'].values())
    write_if_changed(args.out / 'lift_report.json', json.dumps(report, indent=1) + '\n')
    print(f'total {report["total_functions"]} functions, {total:.0f}s', file=sys.stderr)
    if errors:
        print(f'{len(errors)} files could not be written or removed (held open by another process?):',
              file=sys.stderr)
        for e in errors:
            print(f'  {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
