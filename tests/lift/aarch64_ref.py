# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Random AArch64 programs and a reference model of what they compute, for the lifter's differential test.

Each generated instruction is a pair: its assembly text (encoded by clang's assembler, not by the lifter) and a Python
function giving its effect, written from the architecture's definitions and never from the lifter's decoder or its C.
A program is a set of functions: leaves (no calls, write x0-x15), middles (call leaves, may end in a tail call) and
tops (call middles and leaves, write callee-saved x19-x28 as well). Only tops are called from the test driver, so
every bl target honours AAPCS64: no callee writes x19-x29, and no code reads flags a callee or a caller left.

Conventions the driver shares: x20 is the base of a 1 KiB buffer and x21 a pointer into it that loads and stores move
(both reported as offsets from the buffer), sp starts at the top of a stack the driver owns (reported as an offset),
and x30 holds RT_RET_HOST on entry.
"""

from __future__ import annotations

import random
import struct

import numpy as np

M64 = (1 << 64) - 1
BUF = 0x50000000          # where the model puts the buffer; the driver reports addresses as offsets from it
BUF_SIZE = 1024
STACK_TOP = 0x60000000
STACK_SIZE = 4096
RT_RET_HOST = 0x7FF0000000 + 0x100000 - 16
CONDS = ["eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le"]
DEST = list(range(16))                            # x0-x15
SRC = list(range(18)) + list(range(22, 29))       # x0-x17, x22-x28 (x19 kept for tops, x20/x21 are pointers)
TOP_DEST = DEST + list(range(22, 29))


def mask(sz: int) -> int:
    return (1 << sz) - 1


def sx(v: int, bits: int) -> int:
    v &= mask(bits)
    return v - (1 << bits) if v >> (bits - 1) else v


def reg(r: int, sz: int) -> str:
    return f"{'x' if sz == 64 else 'w'}{r}"


class State:
    def __init__(self, x: list[int], nzcv: int, v: list[int], buf: bytes):
        self.x = list(x) + [0] * (32 - len(x))
        self.x[20], self.x[21], self.x[30] = BUF, BUF + 512, RT_RET_HOST
        self.sp = STACK_TOP
        self.n, self.z, self.c, self.v = (bool(nzcv >> 31 & 1), bool(nzcv >> 30 & 1), bool(nzcv >> 29 & 1),
                                          bool(nzcv >> 28 & 1))
        self.vr = list(v) + [0] * (32 - len(v))
        self.buf = bytearray(buf)
        self.stack = bytearray(STACK_SIZE)

    def r(self, n: int, sz: int) -> int:
        return self.sp & mask(sz) if n == 31 else self.x[n] & mask(sz)

    def w(self, n: int, val: int, sz: int) -> None:
        self.x[n] = val & mask(sz)    # a W write clears the upper half

    def flags(self, n, z, c, v) -> None:
        self.n, self.z, self.c, self.v = bool(n), bool(z), bool(c), bool(v)

    def cond(self, name: str) -> bool:
        n, z, c, v = self.n, self.z, self.c, self.v
        return {"eq": z, "ne": not z, "cs": c, "cc": not c, "mi": n, "pl": not n, "vs": v, "vc": not v,
                "hi": c and not z, "ls": not (c and not z), "ge": n == v, "lt": n != v,
                "gt": not z and n == v, "le": not (not z and n == v)}[name]

    def _mem(self, a: int, size: int) -> tuple[bytearray, int]:
        if BUF <= a and a + size <= BUF + BUF_SIZE:
            return self.buf, a - BUF
        if STACK_TOP - STACK_SIZE <= a and a + size <= STACK_TOP:
            return self.stack, a - (STACK_TOP - STACK_SIZE)
        raise AssertionError(f"generator bug: access at {a:#x}")

    def load(self, a: int, size: int) -> int:
        m, o = self._mem(a, size)
        return int.from_bytes(m[o:o + size], "little")

    def store(self, a: int, size: int, val: int) -> None:
        m, o = self._mem(a, size)
        m[o:o + size] = (val & mask(8 * size)).to_bytes(size, "little")


def add_with_carry(a: int, b: int, cin: int, sz: int) -> tuple[int, tuple]:
    m = mask(sz)
    total = (a & m) + (b & m) + cin
    r = total & m
    v = sx(a, sz) + sx(b, sz) + cin != sx(r, sz)
    return r, (r >> (sz - 1), r == 0, total > m, v)


def bitmask_imm(rng: random.Random, sz: int) -> int:
    """A value the logical-immediate encoding can hold: a rotated run of ones replicated over the register."""
    esize = rng.choice([e for e in (2, 4, 8, 16, 32, 64) if e <= sz])
    ones = rng.randrange(1, esize)
    rot = rng.randrange(esize)
    elem = mask(ones)
    elem = ((elem >> rot) | (elem << (esize - rot))) & mask(esize) if rot else elem
    val = 0
    for i in range(0, sz, esize):
        val |= elem << i
    return val


def shift_val(v: int, kind: str, amt: int, sz: int) -> int:
    v &= mask(sz)
    if kind == "lsl":
        return (v << amt) & mask(sz)
    if kind == "lsr":
        return v >> amt
    if kind == "asr":
        return (sx(v, sz) >> amt) & mask(sz)
    return ((v >> amt) | (v << (sz - amt))) & mask(sz) if amt else v   # ror


def f64(bits: int) -> float:
    return struct.unpack("<d", (bits & M64).to_bytes(8, "little"))[0]


def b64(f: float) -> int:
    return int.from_bytes(struct.pack("<d", f), "little")


def f32(bits: int) -> float:
    return float(np.frombuffer((bits & 0xFFFFFFFF).to_bytes(4, "little"), dtype="<f4")[0])


def b32(f: float) -> int:
    with np.errstate(over="ignore"):
        return int.from_bytes(np.float32(f).tobytes(), "little")


def to_int(f: float, sz: int, signed: bool) -> int:
    """FCVTZS / FCVTZU: round toward zero, saturate, NaN to 0."""
    if f != f:
        return 0
    lo, hi = (-(1 << (sz - 1)), (1 << (sz - 1)) - 1) if signed else (0, mask(sz))
    if f >= hi:
        return hi & mask(sz)
    if f <= lo:
        return lo & mask(sz)
    return int(f) & mask(sz)


class Gen:
    """Random instructions. Each method returns (asm, effect) or None when its preconditions do not hold."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.flags_ok = False    # flags were set in this function on every path
        self.sane_d: set[int] = set()   # V registers holding a finite double of moderate size
        self.sane_s: set[int] = set()   # ... a finite single
        self.dest = DEST
        self.wb_left = 0   # x21 writebacks this function may still make

    def sz(self) -> int:
        return self.rng.choice((32, 64))

    def d(self) -> int:
        return self.rng.choice(self.dest)

    def s(self) -> int:
        return self.rng.choice(SRC)

    # ---- integer
    def add_imm(self):
        r, sz, op, setf = self.rng, self.sz(), self.rng.choice(("add", "sub")), self.rng.random() < 0.4
        dn, nn, imm, sh = self.d(), self.s(), r.randrange(4096), r.random() < 0.3
        cmp_form = setf and r.random() < 0.3
        b = imm << 12 if sh else imm
        name = op + ("s" if setf else "")
        text = (f"{'cmp' if op == 'sub' else 'cmn'} {reg(nn, sz)}, #{imm}" if cmp_form
                else f"{name} {reg(dn, sz)}, {reg(nn, sz)}, #{imm}") + (", lsl #12" if sh else "")
        if setf:
            self.flags_ok = True

        def fx(st):
            a = st.r(nn, sz)
            res, fl = add_with_carry(a, b, 0, sz) if op == "add" else add_with_carry(a, ~b & mask(sz), 1, sz)
            if setf:
                st.flags(*fl)
            if not cmp_form:
                st.w(dn, res, sz)
        return text, fx

    def add_shift(self):
        r, sz = self.rng, self.sz()
        op, setf = r.choice(("add", "sub")), r.random() < 0.4
        dn, nn, mm = self.d(), self.s(), self.s()
        kind, amt = r.choice(("lsl", "lsr", "asr")), r.randrange(sz)
        if setf and r.random() < 0.3:   # equal operands: the edge of every unsigned and signed condition
            mm, amt = nn, 0
        if setf:
            self.flags_ok = True

        def fx(st):
            a, b = st.r(nn, sz), shift_val(st.r(mm, sz), kind, amt, sz)
            res, fl = add_with_carry(a, b, 0, sz) if op == "add" else add_with_carry(a, ~b & mask(sz), 1, sz)
            if setf:
                st.flags(*fl)
            st.w(dn, res, sz)
        return f"{op}{'s' if setf else ''} {reg(dn, sz)}, {reg(nn, sz)}, {reg(mm, sz)}, {kind} #{amt}", fx

    def add_ext(self):
        r, sz = self.rng, self.sz()
        op, setf = r.choice(("add", "sub")), r.random() < 0.3
        dn, nn, mm = self.d(), self.s(), self.s()
        ext = r.choice(("uxtb", "uxth", "uxtw", "sxtb", "sxth", "sxtw") + (("uxtx", "sxtx") if sz == 64 else ()))
        amt = r.randrange(5)
        mreg = reg(mm, 64 if ext.endswith("x") else 32)
        if setf:
            self.flags_ok = True

        def fx(st):
            v = st.r(mm, 64)
            width = {"b": 8, "h": 16, "w": 32, "x": 64}[ext[-1]]
            v = sx(v, width) & M64 if ext[0] == "s" else v & mask(width)
            b = (v << amt) & mask(sz)
            a = st.r(nn, sz)
            res, fl = add_with_carry(a, b, 0, sz) if op == "add" else add_with_carry(a, ~b & mask(sz), 1, sz)
            if setf:
                st.flags(*fl)
            st.w(dn, res, sz)
        return f"{op}{'s' if setf else ''} {reg(dn, sz)}, {reg(nn, sz)}, {mreg}, {ext} #{amt}", fx

    def logic_imm(self):
        r, sz = self.rng, self.sz()
        op = r.choice(("and", "orr", "eor", "ands"))
        dn, nn, imm = self.d(), self.s(), bitmask_imm(r, sz)
        if op == "ands":
            self.flags_ok = True

        def fx(st):
            a = st.r(nn, sz)
            res = {"and": a & imm, "orr": a | imm, "eor": a ^ imm, "ands": a & imm}[op]
            if op == "ands":
                st.flags(res >> (sz - 1), res == 0, 0, 0)
            st.w(dn, res, sz)
        return f"{op} {reg(dn, sz)}, {reg(nn, sz)}, #{imm:#x}", fx

    def logic_shift(self):
        r, sz = self.rng, self.sz()
        op = r.choice(("and", "bic", "orr", "orn", "eor", "eon", "ands", "bics"))
        dn, nn, mm = self.d(), self.s(), self.s()
        kind, amt = r.choice(("lsl", "lsr", "asr", "ror")), r.randrange(sz)
        if op in ("ands", "bics"):
            self.flags_ok = True

        def fx(st):
            a, b = st.r(nn, sz), shift_val(st.r(mm, sz), kind, amt, sz)
            if op in ("bic", "orn", "eon", "bics"):
                b = ~b & mask(sz)
            res = a & b if op in ("and", "bic", "ands", "bics") else a | b if op in ("orr", "orn") else a ^ b
            if op in ("ands", "bics"):
                st.flags(res >> (sz - 1), res == 0, 0, 0)
            st.w(dn, res, sz)
        return f"{op} {reg(dn, sz)}, {reg(nn, sz)}, {reg(mm, sz)}, {kind} #{amt}", fx

    def mov_wide(self):
        r, sz = self.rng, self.sz()
        op, dn, imm, hw = r.choice(("movz", "movn", "movk")), self.d(), r.randrange(65536), r.randrange(sz // 16)
        if r.random() < 0.3:   # 0 and all ones: the carry and overflow edges
            imm, hw = 0, 0

        def fx(st):
            v = imm << (16 * hw)
            if op == "movn":
                v = ~v
            elif op == "movk":
                v = (st.r(dn, sz) & ~(0xFFFF << (16 * hw))) | v
            st.w(dn, v, sz)
        return f"{op} {reg(dn, sz)}, #{imm}, lsl #{16 * hw}", fx

    def bitfield(self):
        r, sz = self.rng, self.sz()
        op = r.choice(("ubfx", "sbfx", "bfi", "bfxil", "lsl", "lsr", "asr", "ror", "sxtb", "sxth", "uxtb", "uxth",
                       "sxtw"))
        dn, nn = self.d(), self.s()
        if op == "sxtw" and sz == 32:
            op = "sxth"
        if op in ("uxtb", "uxth"):   # only the 32-bit form exists
            sz = 32
        lsb = r.randrange(sz)
        width = r.randrange(1, sz - lsb + 1)
        amt = r.randrange(sz)

        def fx(st):
            a = st.r(nn, sz)
            if op == "ubfx":
                res = (a >> lsb) & mask(width)
            elif op == "sbfx":
                res = sx(a >> lsb, width) & mask(sz)
            elif op == "bfi":
                res = (st.r(dn, sz) & ~(mask(width) << lsb)) | ((a & mask(width)) << lsb)
            elif op == "bfxil":
                res = (st.r(dn, sz) & ~mask(width)) | ((a >> lsb) & mask(width))
            elif op in ("lsl", "lsr", "asr", "ror"):
                res = shift_val(a, op, amt, sz)
            else:
                bits = {"b": 8, "h": 16, "w": 32}[op[-1]]
                res = sx(a, bits) & mask(sz) if op[0] == "s" else a & mask(bits)
            st.w(dn, res, sz)
        if op in ("ubfx", "sbfx", "bfi", "bfxil"):
            return f"{op} {reg(dn, sz)}, {reg(nn, sz)}, #{lsb}, #{width}", fx
        if op in ("lsl", "lsr", "asr", "ror"):
            return f"{op} {reg(dn, sz)}, {reg(nn, sz)}, #{amt}", fx
        return f"{op} {reg(dn, sz)}, {reg(nn, 32)}", fx

    def extr(self):
        sz = self.sz()
        dn, nn, mm, lsb = self.d(), self.s(), self.s(), self.rng.randrange(sz)

        def fx(st):
            st.w(dn, ((st.r(nn, sz) << sz | st.r(mm, sz)) >> lsb) & mask(sz), sz)
        return f"extr {reg(dn, sz)}, {reg(nn, sz)}, {reg(mm, sz)}, #{lsb}", fx

    def mul(self):
        r = self.rng
        op = r.choice(("madd", "msub", "smaddl", "smsubl", "umaddl", "umsubl", "smulh", "umulh"))
        sz = self.sz() if op in ("madd", "msub") else 64
        dn, nn, mm, aa = self.d(), self.s(), self.s(), self.s()

        def fx(st):
            if op in ("madd", "msub"):
                p = st.r(nn, sz) * st.r(mm, sz)
                res = st.r(aa, sz) + p if op == "madd" else st.r(aa, sz) - p
            elif op in ("smulh", "umulh"):
                a, b = (sx(st.r(nn, 64), 64), sx(st.r(mm, 64), 64)) if op == "smulh" else (st.r(nn, 64), st.r(mm, 64))
                res = (a * b) >> 64
            else:
                a, b = (sx(st.r(nn, 32), 32), sx(st.r(mm, 32), 32)) if op[0] == "s" else (st.r(nn, 32), st.r(mm, 32))
                res = st.r(aa, 64) + a * b if "add" in op else st.r(aa, 64) - a * b
            st.w(dn, res, sz)
        if op in ("madd", "msub"):
            return f"{op} {reg(dn, sz)}, {reg(nn, sz)}, {reg(mm, sz)}, {reg(aa, sz)}", fx
        if op in ("smulh", "umulh"):
            return f"{op} x{dn}, x{nn}, x{mm}", fx
        return f"{op} x{dn}, w{nn}, w{mm}, x{aa}", fx

    def div(self):
        sz, op = self.sz(), self.rng.choice(("udiv", "sdiv"))
        dn, nn, mm = self.d(), self.s(), self.s()

        def fx(st):
            a, b = st.r(nn, sz), st.r(mm, sz)
            if b == 0:
                res = 0
            elif op == "udiv":
                res = a // b
            else:
                q = abs(sx(a, sz)) // abs(sx(b, sz))
                res = q if (sx(a, sz) < 0) == (sx(b, sz) < 0) else -q
            st.w(dn, res, sz)
        return f"{op} {reg(dn, sz)}, {reg(nn, sz)}, {reg(mm, sz)}", fx

    def shiftv(self):
        sz, op = self.sz(), self.rng.choice(("lsl", "lsr", "asr", "ror"))
        dn, nn, mm = self.d(), self.s(), self.s()

        def fx(st):
            st.w(dn, shift_val(st.r(nn, sz), op, st.r(mm, sz) % sz, sz), sz)
        return f"{op}v {reg(dn, sz)}, {reg(nn, sz)}, {reg(mm, sz)}", fx

    def adc(self):
        if not self.flags_ok:
            return None
        sz, op, setf = self.sz(), self.rng.choice(("adc", "sbc")), self.rng.random() < 0.5
        dn, nn, mm = self.d(), self.s(), self.s()

        def fx(st):
            b = st.r(mm, sz) if op == "adc" else ~st.r(mm, sz) & mask(sz)
            res, fl = add_with_carry(st.r(nn, sz), b, int(st.c), sz)
            if setf:
                st.flags(*fl)
            st.w(dn, res, sz)
        return f"{op}{'s' if setf else ''} {reg(dn, sz)}, {reg(nn, sz)}, {reg(mm, sz)}", fx

    def ccmp(self):
        if not self.flags_ok:
            return None
        r, sz = self.rng, self.sz()
        op, cond, nzcv, nn = r.choice(("ccmp", "ccmn")), r.choice(CONDS), r.randrange(16), self.s()
        imm_form = r.random() < 0.5
        imm, mm = r.randrange(32), self.s()

        def fx(st):
            if st.cond(cond):
                b = imm if imm_form else st.r(mm, sz)
                a = st.r(nn, sz)
                _, fl = add_with_carry(a, b, 0, sz) if op == "ccmn" else add_with_carry(a, ~b & mask(sz), 1, sz)
                st.flags(*fl)
            else:
                st.flags(nzcv >> 3 & 1, nzcv >> 2 & 1, nzcv >> 1 & 1, nzcv & 1)
        return f"{op} {reg(nn, sz)}, {('#' + str(imm)) if imm_form else reg(mm, sz)}, #{nzcv}, {cond}", fx

    def csel(self):
        if not self.flags_ok:
            return None
        r, sz = self.rng, self.sz()
        op, cond, dn, nn, mm = r.choice(("csel", "csinc", "csinv", "csneg", "cset", "csetm")), r.choice(CONDS), \
            self.d(), self.s(), self.s()

        def fx(st):
            if op in ("cset", "csetm"):
                res = (mask(sz) if op == "csetm" else 1) if st.cond(cond) else 0
            elif st.cond(cond):
                res = st.r(nn, sz)
            else:
                b = st.r(mm, sz)
                res = {"csel": b, "csinc": b + 1, "csinv": ~b, "csneg": -b}[op]
            st.w(dn, res, sz)
        if op in ("cset", "csetm"):
            return f"{op} {reg(dn, sz)}, {cond}", fx
        return f"{op} {reg(dn, sz)}, {reg(nn, sz)}, {reg(mm, sz)}, {cond}", fx

    def bits1(self):
        sz, op = self.sz(), self.rng.choice(("clz", "cls", "rbit", "rev", "rev16", "rev32"))
        if op == "rev32" and sz == 32:
            op = "rev"
        dn, nn = self.d(), self.s()

        def fx(st):
            a = st.r(nn, sz)
            if op == "clz":
                res = sz - a.bit_length()
            elif op == "cls":   # CountLeadingZeroBits(x<N-1:1> EOR x<N-2:0>)
                res = sz - 1 - ((a ^ (a >> 1)) & mask(sz - 1)).bit_length()
            elif op == "rbit":
                res = int(f"{a:0{sz}b}"[::-1], 2)
            elif op == "rev":
                res = int.from_bytes(a.to_bytes(sz // 8, "little"), "big")
            else:
                chunk = 2 if op == "rev16" else 4
                bs = a.to_bytes(sz // 8, "little")
                res = int.from_bytes(b"".join(bs[i:i + chunk][::-1] for i in range(0, len(bs), chunk)), "little")
            st.w(dn, res, sz)
        return f"{op} {reg(dn, sz)}, {reg(nn, sz)}", fx

    # ---- memory
    def ldst(self):
        r = self.rng
        size = r.choice((1, 2, 4, 8))
        op = r.choice(("ldr", "str", "lds") if size < 8 else ("ldr", "str"))
        tn, off = self.d() if op != "str" else self.s(), r.randrange(0, 448 // size) * size
        dsz = 64 if size == 8 or (op == "lds" and r.random() < 0.5) else 32
        if op == "lds" and size == 4:
            dsz = 64
        suffix = {1: "b", 2: "h", 4: "", 8: ""}[size]
        mnem = {"ldr": "ldr", "str": "str", "lds": "ldrs"}[op] + (suffix if op != "lds" or size != 4 else "w")
        rt = reg(tn, 64 if size == 8 or (op == "lds" and dsz == 64) else 32)

        def fx(st):
            a = st.x[20] + off
            if op == "str":
                st.store(a, size, st.r(tn, 64))
            elif op == "ldr":
                st.w(tn, st.load(a, size), 64)
            else:
                st.w(tn, sx(st.load(a, size), 8 * size), dsz)
        return f"{mnem} {rt}, [x20, #{off}]", fx

    def ldst_wb(self):
        """Pre- or post-indexed access through x21, which it moves. x21 is callee-saved, so only tops use it, and
        at most 8 times each, which keeps it inside the buffer whatever the branches do."""
        if self.wb_left <= 0:
            return None
        self.wb_left -= 1
        r = self.rng
        op, pre, step = r.choice(("ldr", "str")), r.random() < 0.5, r.choice((-16, -8, 8, 16))
        tn = self.d() if op == "ldr" else self.s()

        def fx(st):
            base = st.x[21]
            a = base + step if pre else base
            if op == "str":
                st.store(a, 8, st.r(tn, 64))
            else:
                st.w(tn, st.load(a, 8), 64)
            st.x[21] = base + step
        addr = f"[x21, #{step}]!" if pre else f"[x21], #{step}"
        return f"{op} x{tn}, {addr}", fx

    def ldst_pair(self):
        r = self.rng
        op, size = r.choice(("ldp", "stp", "ldpsw")), r.choice((4, 8))
        if op == "ldpsw":
            size = 4
        t1, t2 = (self.d(), self.d()) if op != "stp" else (self.s(), self.s())
        if op != "stp" and t1 == t2:
            return None
        off = r.randrange(0, 56) * size
        sz = 64 if size == 8 or op == "ldpsw" else 32

        def fx(st):
            a = st.x[20] + off
            if op == "stp":
                st.store(a, size, st.r(t1, 64))
                st.store(a + size, size, st.r(t2, 64))
            else:
                v1, v2 = st.load(a, size), st.load(a + size, size)
                if op == "ldpsw":
                    v1, v2 = sx(v1, 32), sx(v2, 32)
                st.w(t1, v1, sz)
                st.w(t2, v2, sz)
        return f"{op} {reg(t1, sz)}, {reg(t2, sz)}, [x20, #{off}]", fx

    def ldst_reg(self):
        op, tn, idx = self.rng.choice(("ldr", "str")), self.d(), self.s()
        if op == "str":
            tn = self.s()

        def fx(st):
            st.x[16] = st.r(idx, 64) & 0x3F
            a = st.x[20] + st.x[16] * 8
            if op == "str":
                st.store(a, 8, st.r(tn, 64))
            else:
                st.w(tn, st.load(a, 8), 64)
        return f"and x16, x{idx}, #0x3f\n    {op} x{tn}, [x20, x16, lsl #3]", fx

    # ---- floating point and vectors (only on values the model can follow exactly)
    def fp(self):
        r = self.rng
        kind = r.choice(("scvtf", "scvtf", "arith", "arith", "fcvtzs", "fmov", "fneg", "fcmp", "fcmp", "fcsel", "vec"))
        if len(self.sane_d) < 2 and r.random() < 0.6:   # comparisons need two doubles
            kind = "scvtf"
        vd, vn, vm = r.randrange(8), r.randrange(8), r.randrange(8)
        if kind == "scvtf":
            double, nn = r.random() < (0.8 if len(self.sane_d) < 2 else 0.5), self.s()

            def fx(st):
                if double:
                    st.vr[vd] = b64(float(sx(st.r(nn, 64), 64)))
                else:
                    st.vr[vd] = b32(float(sx(st.r(nn, 32), 32)))
            self.sane_d.discard(vd), self.sane_s.discard(vd)
            (self.sane_d if double else self.sane_s).add(vd)
            return (f"scvtf d{vd}, x{nn}" if double else f"scvtf s{vd}, w{nn}"), fx
        if kind == "arith":
            double = r.random() < 0.5
            sane = self.sane_d if double else self.sane_s
            if vn not in sane or vm not in sane:
                return None
            op = r.choice(("fadd", "fsub", "fmul"))

            def fx(st):
                a, b = (f64(st.vr[vn]), f64(st.vr[vm])) if double else (f32(st.vr[vn]), f32(st.vr[vm]))
                res = a + b if op == "fadd" else a - b if op == "fsub" else a * b
                st.vr[vd] = b64(res) if double else b32(res)
            self.sane_d.discard(vd), self.sane_s.discard(vd)
            if op != "fmul":   # products can grow out of the range the model keeps exact; keep them out of reach
                sane.add(vd)
            p = "d" if double else "s"
            return f"{op} {p}{vd}, {p}{vn}, {p}{vm}", fx
        if kind == "fcvtzs":
            double, dn = r.random() < 0.5, self.d()
            if vn not in (self.sane_d if double else self.sane_s):
                return None
            sz = 64 if r.random() < 0.5 else 32

            def fx(st):
                st.w(dn, to_int(f64(st.vr[vn]) if double else f32(st.vr[vn]), sz, True), sz)
            return f"fcvtzs {reg(dn, sz)}, {'d' if double else 's'}{vn}", fx
        if kind == "fmov":
            dn = self.d()

            def fx(st):
                st.w(dn, st.vr[vn], 64)
            return f"fmov x{dn}, d{vn}", fx
        if kind == "fneg":
            op = r.choice(("fneg", "fabs"))
            double = vn in self.sane_d
            if not double and vn not in self.sane_s:
                return None

            def fx(st):
                sign = 1 << (63 if double else 31)
                v = st.vr[vn] & (M64 if double else 0xFFFFFFFF)
                st.vr[vd] = v ^ sign if op == "fneg" else v & ~sign
            self.sane_d.discard(vd), self.sane_s.discard(vd)
            (self.sane_d if double else self.sane_s).add(vd)
            p = "d" if double else "s"
            return f"{op} {p}{vd}, {p}{vn}", fx
        if kind == "fcmp":
            if vn not in self.sane_d or vm not in self.sane_d:
                return None
            self.flags_ok = True

            def fx(st):
                a, b = f64(st.vr[vn]), f64(st.vr[vm])
                st.flags(*((0, 1, 1, 0) if a == b else (1, 0, 0, 0) if a < b else (0, 0, 1, 0)))
            return f"fcmp d{vn}, d{vm}", fx
        if kind == "fcsel":
            if not self.flags_ok or vn not in self.sane_d or vm not in self.sane_d:
                return None
            cond = r.choice(CONDS)

            def fx(st):
                st.vr[vd] = (st.vr[vn] if st.cond(cond) else st.vr[vm]) & M64
            self.sane_s.discard(vd)
            self.sane_d.add(vd)
            return f"fcsel d{vd}, d{vn}, d{vm}, {cond}", fx
        op, arr = r.choice(("add", "sub", "eor")), r.choice(("4s", "2d", "16b", "8h"))
        if op == "eor":
            arr = "16b"
        lane = {"4s": 32, "2d": 64, "16b": 8, "8h": 16}[arr]

        def fx(st):
            a, b = st.vr[vn], st.vr[vm]
            if op == "eor":
                st.vr[vd] = a ^ b
                return
            res = 0
            for i in range(0, 128, lane):
                x, y = a >> i & mask(lane), b >> i & mask(lane)
                res |= ((x + y if op == "add" else x - y) & mask(lane)) << i
            st.vr[vd] = res
        self.sane_d.discard(vd), self.sane_s.discard(vd)
        return f"{op} v{vd}.{arr}, v{vn}.{arr}, v{vm}.{arr}", fx

    def any_op(self):
        r = self.rng
        pick = r.choices(
            (self.add_imm, self.add_shift, self.add_ext, self.logic_imm, self.logic_shift, self.mov_wide,
             self.bitfield, self.extr, self.mul, self.div, self.shiftv, self.adc, self.ccmp, self.csel, self.bits1,
             self.ldst, self.ldst_wb, self.ldst_pair, self.ldst_reg, self.fp),
            weights=(6, 5, 3, 4, 4, 4, 5, 2, 3, 2, 2, 4, 3, 5, 3, 4, 2, 2, 2, 12))[0]
        for _ in range(20):
            got = pick()
            if got:
                return got
            pick = self.add_imm
        return self.add_imm()


class Program:
    """Functions as lists of items: ('op', asm, fx), ('skip', asm, test, k), ('call', j), ('tail', j), and the
    prologue and epilogue of functions that call."""

    def __init__(self, seed: int, leaves: int = 10, middles: int = 6, tops: int = 6, length: tuple = (12, 40)):
        self.prefix = f"s{seed}_"   # labels, so that several programs share one source
        self.rng = rng = random.Random(seed)
        self.funcs: list[dict] = []
        self.label = 0
        for i in range(leaves):
            self.funcs.append(self.function("leaf", rng.randrange(*length), []))
        leaf_ids = list(range(leaves))
        for i in range(middles):
            self.funcs.append(self.function("middle", rng.randrange(*length), leaf_ids))
        mid_ids = list(range(leaves + middles))
        for i in range(tops):
            self.funcs.append(self.function("top", rng.randrange(*length), mid_ids))
        self.tops = list(range(leaves + middles, len(self.funcs)))

    def function(self, role: str, n: int, callees: list[int]) -> dict:
        rng = self.rng
        g = Gen(rng)
        if role == "top":
            g.dest, g.wb_left = TOP_DEST, 8
        items = []
        while len(items) < n:
            roll = rng.random()
            if callees and roll < 0.12:
                items.append(("call", rng.choice(callees)))
                g.flags_ok = False   # AAPCS64: a callee's flags are not the caller's to read
                g.sane_d.clear(), g.sane_s.clear()   # nor are the caller-saved V registers
                continue
            if roll < 0.25:
                k = rng.randrange(1, 4)
                kind = rng.choice(("bcond", "cbz", "tbz") if g.flags_ok else ("cbz", "tbz"))
                rn, sz, bit, cond = rng.choice(SRC), rng.choice((32, 64)), rng.randrange(64), rng.choice(CONDS)
                self.label += 1
                lbl = f".L{self.prefix}skip{self.label}"
                if kind == "bcond":
                    asm, test = f"b.{cond} {lbl}", (lambda st, c=cond: st.cond(c))
                elif kind == "cbz":
                    nz = rng.random() < 0.5
                    asm = f"{'cbnz' if nz else 'cbz'} {reg(rn, sz)}, {lbl}"
                    test = (lambda st, r_=rn, s_=sz, nz_=nz: (st.r(r_, s_) != 0) == nz_)
                else:
                    nz = rng.random() < 0.5
                    asm = f"{'tbnz' if nz else 'tbz'} x{rn}, #{bit}, {lbl}"
                    test = (lambda st, r_=rn, b_=bit, nz_=nz: bool(st.r(r_, 64) >> b_ & 1) == nz_)
                flags_before, d_before, s_before = g.flags_ok, set(g.sane_d), set(g.sane_s)
                body = [g.any_op() for _ in range(k)]
                items.append(("skip", asm, test, body, lbl))
                g.flags_ok = flags_before and g.flags_ok
                g.sane_d &= d_before
                g.sane_s &= s_before
                continue
            asm, fx = g.any_op()
            items.append(("op", asm, fx))
        tail = None
        if role == "middle" and rng.random() < 0.5:
            tail = rng.choice(callees)
        return {"role": role, "items": items, "tail": tail, "frame": role != "leaf"}

    def name(self, i: int) -> str:
        return f"{self.prefix}fn{i}"

    def assembly(self) -> str:
        """The functions, without the section header: several programs can follow one another."""
        out = [".p2align 4"]
        for i, f in enumerate(self.funcs):
            out.append(f"{self.name(i)}:")
            if f["frame"]:
                out.append("    stp x29, x30, [sp, #-32]!")
            for it in f["items"]:
                if it[0] == "op":
                    out.append(f"    {it[1]}")
                elif it[0] == "call":
                    out.append(f"    bl {self.name(it[1])}")
                else:
                    _, asm, _, body, lbl = it
                    out.append(f"    {asm}")
                    out += [f"    {a}" for a, _ in body]
                    out.append(f"{lbl}:")
            if f["frame"]:
                out.append("    ldp x29, x30, [sp], #32")
            out.append(f"    b {self.name(f['tail'])}" if f["tail"] is not None else "    ret")
        return "\n".join(out) + "\n"

    def run(self, i: int, st: State) -> None:
        f = self.funcs[i]
        if f["frame"]:
            st.sp -= 32
            st.store(st.sp, 8, st.x[29])
            st.store(st.sp + 8, 8, st.x[30])
        for it in f["items"]:
            if it[0] == "op":
                it[2](st)
            elif it[0] == "call":
                # The return address is never observed: no generated code reads x30, and every function that
                # calls restores it from its frame before it returns.
                st.x[30] = 0
                self.run(it[1], st)
            else:
                _, _, test, body, _ = it
                if not test(st):
                    for _, fx in body:
                        fx(st)
        if f["frame"]:
            st.x[29] = st.load(st.sp, 8)
            st.x[30] = st.load(st.sp + 8, 8)
            st.sp += 32
        if f["tail"] is not None:
            self.run(f["tail"], st)
