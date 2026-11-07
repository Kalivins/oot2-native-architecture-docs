#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
#
# Adapted from DKB Recomp (https://github.com/Lucas-blip-png/dkb-pc-recomp) at 9d8fa02, tools/spin_waits.py (no licence
# header of its own: the repository's GPL-3.0-or-later), which is adapted from nxnative dm/tools/patch_nso.py @ 6d34984
# (MIT, Copyright (c) 2026 seedds): the spin wait search ("Spin waits": spin_rw, spin_kind, spin_rmw, spin_lock,
# spin_store, spin_quick, spin_region, spin_region1). Changed in oot2-native: the search finds what upstream's does (the
# constants and the Search class are as upstream, but for the memo of decoded words, which is bounded: SPIN_MEMO). The
# patch table is gone, so `touched` is an optional argument that defaults to empty; the modules are read through
# tools/lift/nso.py and placed by name only; find_module() is a library entry point; the command writes a JSON file in
# a build folder instead of printing `spin` lines of a checked-in table.
"""Find the spin waits of the ExeFS modules and write them as JSON.

    py -3 tools/lift/spin_waits.py --exefs DIR --out FILE [--modules main,subsdk0,sdk]

--out is a file in the build folder, outside the repository: the table is derived from the player's own game code, so
it is computed on the player's machine, as a preparation step before the lift, and never checked in. tools/lift/gen.py
reads it with --spin-waits. The file is {"<module>": [{"head": H, "back_edges": [B, ...], "size": N, "kinds": [...]},
...]}: H and B are offsets from the start of the module's image, N counts the instructions of the region, kinds names
what the region polls with ("rmw", "acq", "rel", "tst", "stc"; empty for plain loads). One count line per module goes to
stderr. Needs capstone (pinned in CI: capstone==5.0.7); gen.py itself does not.

A guest loop that polls memory until another thread changes it fills only its own core on the Switch, where a better
thread pinned there preempts it. Here it holds one of the program's virtual cores (runtime/core/kernel.c) and a host core,
and when the poll takes and releases a lock on every pass, two pollers hand the lock's cache line between their cores
so that the thread whose item it is rarely wins its exclusive store (nxnative 6d34984: Breath of the Wild's job queue,
main+0xef62dc, CPU frames of 30-210 ms every 1-10 s). The generator sends the back-edges of such loops through
rt_spin_gate (declared in runtime/include/oot2/cpu.h), which slows the poll down and hands the virtual core to a
waiting thread once the loop has polled for a while.

A loop counts only when it provably cannot leave by itself. From its head T every path back to T (the region R,
within SPIN_WIN bytes after T) runs loads, register and flag arithmetic, direct branches and barriers; no call, SVC,
system register or event wait; nothing live at T is written in R. Allowed besides: an exclusive add of a constant
that every path back to T undoes on the same address, a spin lock taken and released on the same address on every
path, a test loop on a word inside R, and at most SPIN_STORES stores of registers R never writes (fixed values to
fixed addresses, so the passes repeat after a few). Offsets in `touched`, if given, are kept out of regions.

Memory. The search decodes words with capstone and memoises them (Search.ins). A decoded word costs about 3.4 KiB in
the process (a copy of capstone's instruction and of its detail record, and the Python objects around them), and an
unbounded memo grew with every head searched, to several GiB for the game's main module: the peak of a whole prepare.
The memo is bounded to SPIN_MEMO words (about 7 MiB) and is dropped when it fills. It only saves decoding the same
word again, so no answer depends on it; the search of one head reads from 4 bytes before the head to 24 bytes past its
farthest back-edge, which is far fewer than SPIN_MEMO words, so a head is searched from the memo almost entirely.
What is left is the module image, the loader's copies and the table of candidate heads. The search's peak private
bytes are held to 1 GiB (ticket 122): a quarter of the 4 GiB or so that an 8 GB machine has free once the system and
the player's other programs have their share, and under the 1.2 GiB per process of the lift's code generation, so the
search is no longer the peak of the lift step. The lift step records the peak beside the seconds
(tools/prepare/program.py).

Exit codes: 0 written, 1 the output file could not be written, 2 bad input.
"""
import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

from capstone import CS_ARCH_ARM64, CS_MODE_ARM, Cs
from capstone.arm64 import ARM64_OP_IMM, ARM64_OP_MEM, ARM64_OP_REG

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import layout  # noqa: E402  (tools/layout.py)
from nso import Nso, NsoError  # noqa: E402

SPIN_WIN = 512   # bytes from a backward branch back to its target
SPIN_TOPS = {0x16, 0x17, 0x54, 0x34, 0x35, 0xB4, 0xB5, 0x36, 0x37, 0xB6, 0xB7}   # b, b.cond, cbz/cbnz, tbz/tbnz
SPIN_PATHS, SPIN_STEPS, SPIN_STORES = 256, 4096, 4   # complete paths back to T; search steps; stores in a region
SPIN_MEMO = 2048   # decoded words the memo keeps before it is dropped (one head reads at most about 150)
SPIN_LSE = ('ldadd', 'ldclr', 'ldeor', 'ldset', 'ldsmax', 'ldsmin', 'ldumax', 'ldumin')
SPIN_BAD = {'bl', 'blr', 'br', 'ret', 'eret', 'svc', 'hvc', 'smc', 'brk', 'hlt', 'udf', 'mrs', 'msr', 'sys', 'sysl',
            'wfe', 'wfi', 'sev', 'sevl', 'clrex', 'dc', 'ic', 'at', 'tlbi', 'drps', 'dcps1', 'dcps2', 'dcps3', 'blraa',
            'blrab', 'blraaz', 'blrabz', 'braa', 'brab', 'braaz', 'brabz', 'retaa', 'retab', 'eretaa', 'eretab'}
# Words that end every path through them by their encoding alone: bl; br / blr / ret; svc / brk; stores (immediate,
# unscaled, pre / post-index, register offset, vector); stp / stnp; LSE atomics. Exclusive stores and pre / post-index
# loads are left to the full check.
SPIN_STOP = ((0xFC000000, 0x94000000), (0xFE000000, 0xD6000000), (0xFF000000, 0xD4000000), (0x3FC00000, 0x39000000),
             (0x3FC00000, 0x3D000000), (0x3FC00000, 0x3D800000), (0x3FE00000, 0x38000000), (0x3FE00C00, 0x38200800),
             (0x3FE00000, 0x3C000000), (0x3FE00000, 0x3C800000), (0x3A400000, 0x28000000), (0x3F200C00, 0x38200000))


class SpinNo(Exception):
    pass


def norm(n):
    """w / x views as x<n>, vector views as v<n>, zero registers as None."""
    n = {'fp': 'x29', 'lr': 'x30', 'wsp': 'sp'}.get(n, n)
    if n in ('wzr', 'xzr'):
        return None
    m = re.fullmatch(r'[wx](\d+)', n)
    if m:
        return 'x' + m.group(1)
    m = re.fullmatch(r'[bhsdqv](\d+)', n)
    return 'v' + m.group(1) if m else n


class Search:
    """The spin waits of one module image. touched: image offsets no region may contain (none by default). memo: how
    many decoded words are kept at once (see the docstring's Memory paragraph)."""

    def __init__(self, img, text_end, touched=(), memo=SPIN_MEMO):
        self.img, self.text_end, self.touched = img, text_end, set(touched)
        self.md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        self.md.detail = True
        self.dec, self.memo = {}, max(1, memo)

    def ins(self, a):
        """The instruction at a, or None outside the text or when capstone cannot decode it. A pure function of the
        image: self.dec only saves decoding a word again, and holds at most self.memo of them."""
        if a not in self.dec:
            if len(self.dec) >= self.memo:
                self.dec.clear()   # the callers hold what they decoded; a word asked for again is decoded again
            self.dec[a] = next(self.md.disasm(bytes(self.img[a:a + 4]), a), None) if 0 <= a < self.text_end else None
        return self.dec[a]

    def word(self, a):
        return struct.unpack_from('<I', self.img, a)[0]

    def candidates(self):
        """{loop head: [backward branches to it]}, branches at most SPIN_WIN bytes back."""
        cand = {}
        for o in range(0, self.text_end, 4):
            w = self.word(o)
            t = w >> 24
            if t not in SPIN_TOPS:
                continue
            if t in (0x16, 0x17):
                d = ((w & 0x3FFFFFF) - (1 << 26)) * 4
            elif t & 0x7E == 0x36:
                d = (((w >> 5) & 0x3FFF) - (1 << 14)) * 4 if w & 0x40000 else 4
            elif t == 0x54:
                d = (((w >> 5) & 0x7FFFF) - (1 << 19)) * 4 if w & 0x800000 and not w & 0x10 else 4
            else:
                d = (((w >> 5) & 0x7FFFF) - (1 << 19)) * 4 if w & 0x800000 else 4
            if -SPIN_WIN <= d <= 0:
                cand.setdefault(o + d, []).append(o)
        return cand

    @staticmethod
    def rw(ins):
        """(registers read, registers written). Capstone lists a compare's first operand as written and the flags as
        read by cbz / tbz, and leaves out the destination of adds, so every register operand counts as well: operand 0
        written and the others read, or for a load all of them written."""
        m = ins.mnemonic
        regs = [norm(ins.reg_name(o.reg)) for o in ins.operands if o.type == ARM64_OP_REG]
        ops = set(regs) - {None}
        if m in ('cbz', 'cbnz', 'tbz', 'tbnz'):
            return ops, set()
        if m.startswith('b.'):
            return {'nzcv'}, set()
        if m == 'b':
            return set(), set()
        if m in ('cmp', 'cmn', 'tst', 'fcmp', 'fcmpe'):
            return ops, {'nzcv'}
        if m in ('ccmp', 'ccmn', 'fccmp', 'fccmpe'):
            return ops | {'nzcv'}, {'nzcv'}
        r, w = ins.regs_access()
        r = {norm(ins.reg_name(x)) for x in r} - {None}
        w = {norm(ins.reg_name(x)) for x in w} - {None}
        if m.startswith(('ld', 'prfm', 'prfum')):
            return r, w | ops
        return r | (set(regs[1:]) - {None}), w | (set(regs[:1]) - {None})

    @staticmethod
    def kind(ins):
        """'br' direct branch, 'pc' adr / adrp, 'ok' a load or register / flag arithmetic, None anything else."""
        if ins is None:
            return None
        m = ins.mnemonic
        if m in ('b', 'cbz', 'cbnz', 'tbz', 'tbnz') or m.startswith('b.'):
            return 'br'
        if m in ('adr', 'adrp'):
            return 'pc'
        if m in SPIN_BAD or m.startswith(('st', 'cas', 'swp') + SPIN_LSE):
            return None
        mem = any(o.type == ARM64_OP_MEM for o in ins.operands)
        if m.startswith(('ld', 'prfm', 'prfum')):
            return 'ok' if mem else None   # no memory operand: a pc-relative literal load
        return None if mem else 'ok'

    def rmw(self, a):
        """ld(a)xr wA,[xN]; add|sub wB,wA,#k; st(l)xr wS,wB,[xN]; cbnz wS,<the load> at a -> (xN, +-k, registers
        written, the four instructions), or None."""
        l = [self.ins(a + 4 * i) for i in range(4)]
        if (None in l or l[0].mnemonic not in ('ldxr', 'ldaxr') or l[1].mnemonic not in ('add', 'sub')
                or l[2].mnemonic not in ('stxr', 'stlxr') or l[3].mnemonic != 'cbnz'):
            return None
        r = lambda x, i: x.reg_name(x.operands[i].reg)
        if len(l[1].operands) != 3 or l[1].operands[2].type != ARM64_OP_IMM or 'lsl' in l[1].op_str:
            return None
        base = l[0].operands[1].mem
        if base.disp or base.index or l[2].operands[2].mem.base != base.base or l[2].operands[2].mem.disp:
            return None
        if r(l[1], 1) != r(l[0], 0) or r(l[2], 1) != r(l[1], 0) or r(l[3], 0) != r(l[2], 0) or l[3].operands[1].imm != a:
            return None
        k = l[1].operands[2].imm * (1 if l[1].mnemonic == 'add' else -1)
        return norm(l[0].reg_name(base.base)), k, {norm(r(l[0], 0)), norm(r(l[1], 0)), norm(r(l[2], 0))} - {None}, l

    def lock(self, a):
        """An exclusive spin lock sequence or a test loop entered at a -> (kind, xN, its instructions, how many come
        before a, how many from a on, registers read, registers written):
          'acq'  ld(a)xr wA,[xN]; cbz wA,+16; clrex; b <the load>; st(l)xr wS,wK,[xN]; cbnz wS,<the load>
          'rel'  clrex; ld(a)xr wA,[xN]; cmp wA,#k; b.ne <the clrex>; st(l)xr wS,wzr,[xN]; cbnz wS,<the load>
                 (entered at the load), or stlr wzr,[xN]
          'tst'  ldr|ldrb|ldrh wA,[xN{,#imm}]; cmp wA,#k; b.eq|b.ne <the load>"""
        l = [self.ins(a + 4 * i) for i in range(6)]
        if l[0] is None:
            return None
        r = lambda x, i: x.reg_name(x.operands[i].reg)
        plain = lambda x, i: x.operands[i].type == ARM64_OP_MEM and not x.operands[i].mem.index and not x.operands[i].mem.disp
        m0 = l[0].mnemonic
        if m0 in ('ldxr', 'ldaxr') and None not in l and r(l[0], 0).startswith('w') and plain(l[0], 1):
            base = l[0].operands[1].mem.base
            bn = norm(l[0].reg_name(base))
            if (l[1].mnemonic == 'cbz' and r(l[1], 0) == r(l[0], 0) and l[1].operands[1].imm == a + 16
                    and l[2].mnemonic == 'clrex' and l[3].mnemonic == 'b' and l[3].operands[0].imm == a
                    and l[4].mnemonic in ('stxr', 'stlxr') and plain(l[4], 2) and l[4].operands[2].mem.base == base
                    and r(l[4], 1) not in ('wzr', 'xzr')
                    and l[5].mnemonic == 'cbnz' and r(l[5], 0) == r(l[4], 0) and l[5].operands[1].imm == a):
                return 'acq', bn, l, 0, 6, {bn, norm(r(l[4], 1))}, {norm(r(l[0], 0)), norm(r(l[4], 0))}
            p = self.ins(a - 4)
            if (p is not None and p.mnemonic == 'clrex' and l[1].mnemonic == 'cmp' and r(l[1], 0) == r(l[0], 0)
                    and l[1].operands[1].type == ARM64_OP_IMM and l[2].mnemonic == 'b.ne'
                    and l[2].operands[0].imm == a - 4 and l[3].mnemonic in ('stxr', 'stlxr') and r(l[3], 1) == 'wzr'
                    and plain(l[3], 2) and l[3].operands[2].mem.base == base and l[4].mnemonic == 'cbnz'
                    and r(l[4], 0) == r(l[3], 0) and l[4].operands[1].imm == a):
                return 'rel', bn, [p] + l[:5], 1, 5, {bn}, {norm(r(l[0], 0)), norm(r(l[3], 0)), 'nzcv'}
        if m0 == 'stlr' and len(l[0].operands) == 2 and r(l[0], 0) in ('wzr', 'xzr') and plain(l[0], 1):
            return 'rel', norm(l[0].reg_name(l[0].operands[1].mem.base)), l[:1], 0, 1, {norm(l[0].reg_name(l[0].operands[1].mem.base))}, set()
        if (m0 in ('ldr', 'ldrb', 'ldrh') and None not in l[:3] and len(l[0].operands) == 2 and r(l[0], 0).startswith('w')
                and l[0].operands[1].type == ARM64_OP_MEM and not l[0].operands[1].mem.index and not l[0].writeback
                and l[1].mnemonic == 'cmp' and r(l[1], 0) == r(l[0], 0) and l[1].operands[1].type == ARM64_OP_IMM
                and l[2].mnemonic in ('b.eq', 'b.ne') and l[2].operands[0].imm == a):
            bn = norm(l[0].reg_name(l[0].operands[1].mem.base))
            return 'tst', bn, l[:3], 0, 3, {bn}, {norm(r(l[0], 0)), 'nzcv'}
        return None

    @staticmethod
    def store(ins):
        """str / strb / strh / stur / sturb / sturh of a w register or zr to [xN{,#imm}] (no writeback, no register
        offset) -> (value register or None for zr, base register), or None."""
        if (ins is None or ins.mnemonic not in ('str', 'strb', 'strh', 'stur', 'sturb', 'sturh') or ins.writeback
                or len(ins.operands) != 2):
            return None
        v, o = ins.operands
        if v.type != ARM64_OP_REG or o.type != ARM64_OP_MEM or o.mem.index:
            return None
        vn = ins.reg_name(v.reg)
        if not (vn.startswith('w') or vn == 'xzr'):
            return None
        return norm(vn), norm(ins.reg_name(o.mem.base))

    def quick(self, T, hi):
        """A back-edge to T can be reached from T without passing a word of SPIN_STOP (only such heads get the full
        check, which decodes)."""
        seen, todo = set(), [T]
        while todo:
            a = todo.pop()
            if a in seen or not T <= a <= hi:
                continue
            seen.add(a)
            w = self.word(a)
            t = w >> 24
            if 0x14 <= t <= 0x17:
                succ = (a + (((w & 0x3FFFFFF) ^ 0x2000000) - 0x2000000) * 4,)
            elif t == 0x54 or t & 0x7E == 0x34:
                succ = (a + ((((w >> 5) & 0x7FFFF) ^ 0x40000) - 0x40000) * 4, a + 4)
            elif t & 0x7E == 0x36:
                succ = (a + ((((w >> 5) & 0x3FFF) ^ 0x2000) - 0x2000) * 4, a + 4)
            elif any(w & m == v for m, v in SPIN_STOP) and not (
                    w & 0xBFC00000 in (0xB9000000, 0x39000000, 0x79000000)
                    or w & 0xBFE00C00 in (0xB8000000, 0x38000000, 0x78000000)):
                continue   # str / strb / strh (immediate, unscaled) may store the same on every pass
            else:
                succ = (a + 4,)
            if T in succ:
                return True
            todo.extend(succ)
        return False

    def region(self, T, bes):
        """The spin wait with head T (bes: the backward branches to T) -> dict, or None. A store whose value or
        address a pass changes is no part of a region: the paths through it are dropped and the region is found again
        without them. The region without stores is taken instead when the wider one does not cover its back-edges."""
        ban, g = set(), None
        for _ in range(8):
            g = self.region1(T, bes, ban, True)
            if not isinstance(g, set):
                break
            ban |= g
        old = self.region1(T, bes, set(), False)
        return g if isinstance(g, dict) and (not old or old['valid'] <= g['valid']) else old

    def region1(self, T, bes, ban, stores):
        hi = max(bes)
        if not self.quick(T, hi):
            return None
        nodes, back, bad, inner = {}, set(), set(), set()
        todo = [T]
        while todo:
            a = todo.pop()
            if a in nodes or a in bad:
                continue
            if a in ban:
                bad.add(a)
                continue
            rmw = self.rmw(a) if a + 12 <= hi and self.word(a) & 0x3FE00000 == 0x08400000 else None   # ldxr / ldaxr
            lk = None if rmw else self.lock(a)
            if lk and lk[0] == 'tst' and a == T:
                lk = None   # the region's own test loop
            if rmw:
                n = {'kind': 'rmw', 'base': rmw[0], 'k': rmw[1], 'ins': rmw[3], 'uses': {rmw[0]}, 'defs': rmw[2],
                     'succ': [a + 16]}
            elif lk:
                n = {'kind': lk[0], 'base': lk[1], 'k': {'acq': 1, 'rel': -1, 'tst': 0}[lk[0]], 'ins': lk[2],
                     'pre': lk[3], 'uses': lk[5], 'defs': lk[6], 'succ': [a + 4 * lk[4]]}
            elif stores and self.store(self.ins(a)):
                x = self.ins(a)
                st = self.store(x)
                n = {'kind': 'stc', 'st': st, 'ins': [x], 'uses': {st[1]} | ({st[0]} - {None}), 'defs': set(),
                     'succ': [a + 4]}
            else:
                x = self.ins(a)
                k = self.kind(x)
                if k is None:
                    bad.add(a)
                    continue
                if k == 'br':
                    t = x.operands[-1].imm
                    succ = [t] if x.mnemonic == 'b' else [t, a + 4]
                else:
                    succ = [a + 4]
                n = {'kind': k, 'ins': [x], 'succ': succ}
            pre = n.get('pre', 0)
            if any(a + 4 * (i - pre) in self.touched for i in range(len(n['ins']))):
                bad.add(a)
                continue
            nodes[a] = n
            if n['kind'] == 'rmw':
                inner |= {a + 4, a + 8, a + 12}
            if n['kind'] in ('acq', 'rel', 'tst'):
                inner |= {a + 4 * (i - pre) for i in range(len(n['ins']))} - {a}
            for s in n['succ']:
                if s == T:
                    back.add(a)
                elif T < s <= hi:
                    todo.append(s)
        if not back or inner & set(nodes) or any(s in inner for n in nodes.values() for s in n['succ']):
            return None

        def prune(ends):
            keep, more = set(ends), True
            while more:
                more = False
                for a, n in nodes.items():
                    if a not in keep and any(s in keep and s != T for s in n['succ']):
                        keep.add(a)
                        more = True
            return keep

        R = prune(back)
        for a in R:
            if 'uses' not in nodes[a]:
                nodes[a]['uses'], nodes[a]['defs'] = self.rw(nodes[a]['ins'][0])
        # Every path from T to a back-edge: no inner cycle, and the exclusive adds on each address (a register between
        # two writes to it) sum to 0; a lock is taken at most once and released after it.
        bad_back, cnt = set(), [0, 0]

        def walk(a, onpath, ver, delta):
            cnt[1] += 1
            if cnt[1] > SPIN_STEPS:
                raise SpinNo
            n = nodes[a]
            if n['kind'] in ('rmw', 'acq', 'rel'):
                key = (n['kind'] != 'rmw', n['base'], ver.get(n['base'], 0))
                delta = dict(delta)
                delta[key] = delta.get(key, 0) + n['k']
                if n['kind'] != 'rmw' and not 0 <= delta[key] <= 1:
                    raise SpinNo
            if n['defs']:
                ver = dict(ver)
                ver.update({r: ver.get(r, 0) + 1 for r in n['defs']})
            for s in n['succ']:
                if s == T:
                    cnt[0] += 1
                    if cnt[0] > SPIN_PATHS:
                        raise SpinNo
                    if any(delta.values()):
                        bad_back.add(a)
                elif s in R:
                    if s in onpath:
                        raise SpinNo
                    walk(s, onpath | {s}, ver, delta)

        try:
            walk(T, {T}, {}, {})
        except (SpinNo, RecursionError):
            return None
        valid = back - bad_back
        if not valid:
            return None
        R = prune(valid)
        # No state carried from one pass to the next: nothing live at T is written in R.
        live, more = {a: set() for a in R}, True
        while more:
            more = False
            for a in sorted(R, reverse=True):
                n, out = nodes[a], set()
                for s in n['succ']:
                    if s == T:
                        out |= live[T] if a in valid else set()
                    elif s in R:
                        out |= live[s]
                new = n['uses'] | (out - n['defs'])
                if new != live[a]:
                    live[a] = new
                    more = True
        defs = set().union(*(nodes[a]['defs'] for a in R))
        if any(nodes[a]['kind'] == 'stc' for a in R) and not any(
                nodes[a]['kind'] in ('rmw', 'acq', 'rel', 'tst') or nodes[a]['ins'][0].mnemonic.startswith('ld')
                for a in R):
            return None   # stores and nothing polled
        vary = {a for a in R if nodes[a]['kind'] == 'stc' and nodes[a]['uses'] & defs}
        if vary:
            return vary
        if sum(nodes[a]['kind'] == 'stc' for a in R) > SPIN_STORES:
            return None
        if live[T] & defs:
            return None
        kinds = sorted({nodes[a]['kind'] for a in R} & {'rmw', 'acq', 'rel', 'tst', 'stc'})
        return {'T': T, 'R': sorted(R), 'valid': valid, 'size': sum(len(nodes[a]['ins']) for a in R), 'kinds': kinds}

    def all(self):
        return [g for g in (self.region(T, bes) for T, bes in sorted(self.candidates().items())) if g]


def find_nso(nso, touched=()):
    """The spin waits of a loaded Nso, as the dicts the JSON file holds (sorted by head)."""
    found = Search(bytes(nso.image), nso.text[0] + nso.text[1], touched).all()
    return [{'head': g['T'], 'back_edges': sorted(g['valid']), 'size': g['size'], 'kinds': list(g['kinds'])}
            for g in found]


def find_module(path, touched=()):
    """The spin waits of the module file at path: [{"head", "back_edges", "size", "kinds"}], offsets from the start
    of the module's image. Raises nso.NsoError or OSError on a file that cannot be read."""
    return find_nso(Nso(str(path)), touched)


def write_atomic(path, text):
    """Write text to path under a temporary name and rename it into place, so a reader never sees half a file."""
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_text(text, encoding='utf-8', newline='\n')
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--exefs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='the JSON file to write (in the build folder)')
    ap.add_argument('--modules', default='main,subsdk0,sdk')
    args = ap.parse_args(argv)
    sys.setrecursionlimit(20000)
    if not args.exefs.is_dir():
        print(f'spin_waits: {args.exefs} is not a folder', file=sys.stderr)
        return 2
    names = [n for n in args.modules.split(',') if n]
    stray = [n for n in names if n not in layout.ORDER]
    if stray or not names:
        print(f'spin_waits: unknown module name(s): {", ".join(stray) or "(none given)"}; expected '
              f'{", ".join(layout.ORDER)}', file=sys.stderr)
        return 2
    table = {}
    for name in names:
        path = args.exefs / name
        if not path.is_file():
            print(f'spin_waits: not in {args.exefs}: {name}', file=sys.stderr)
            return 2
        try:
            found = find_module(path)
        except (OSError, NsoError, ValueError) as e:
            print(f'spin_waits: {e}', file=sys.stderr)
            return 2
        table[name] = found
        print(f'{name}: {len(found)} spin waits', file=sys.stderr)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(args.out, json.dumps(table, indent=1) + '\n')
    except OSError as e:
        print(f'spin_waits: {args.out}: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
