#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Run an Ampere (SM86) graphics program on the CPU, from nvdisasm's listing.

Spike S2 needs a reference for a translated draw that does not come from the
translator. The program's own instructions are decoded by NVIDIA's disassembler
(`nvdisasm -b SM86 -hex`, CUDA 11.8 or later), and this module gives the
instructions it prints their meaning, one thread at a time:

    prog = parse_listing(text)
    m = Machine(prog, env)
    m.run()

The environment (`Env`) answers what a program reads from outside: constant
banks, global memory, input attributes (ALD), interpolated attributes (IPA)
and texture samples (TEX). The result holds the attribute stores (AST), the
registers at EXIT and whether the thread was killed.

One thread runs alone, so what only matters between the threads of a warp
does nothing here: the convergence barriers (BSSY, BSYNC) and scoreboard waits
(DEPBAR). CALL.REL.NOINC keeps its return address for a RET; a program that
never returns uses it as a branch. Texture instructions read their operands in
the order the compiler itself places them (see `Machine._tex`).

Only the instructions and modifiers listed in `Machine.run` are known. Anything
else raises `Unsupported`, never a guess: a reference that silently skips an
instruction would agree with a translator that skips it too.

Float arithmetic is exact to binary32: each result is the real result of its
binary32 operands rounded once (fused for FFMA), with flush to zero where the
instruction says FTZ and rounding toward zero where it says RZ. MUFU is an
approximation on the hardware (within about an ulp); here it is the correctly
rounded function, so a comparison against hardware or a translation needs a
tolerance of a few ulps there.

The shader program header (SPH, the 0x80 bytes in front of a graphics
program's code) is read by `parse_sph` from NVIDIA's documented layout
(open-gpu-doc, "Shader Program Header"), for the pixel shader's input
interpolation modes and its output map.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

QNAN = 0x7FFFFFFF  # what the hardware's float units return for an invalid operation
MAX_STEPS = 1 << 20


class Unsupported(Exception):
    """An instruction, modifier or operand form this evaluator does not model."""


# ---- binary32 helpers -------------------------------------------------------

def bits_to_float(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def float_to_bits(x: float) -> int:
    """x rounded to nearest even binary32 (overflow gives infinity)."""
    if math.isnan(x):
        return QNAN
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0xFF800000 if x < 0 else 0x7F800000


def _is_denormal(b: int) -> bool:
    return (b & 0x7F800000) == 0 and (b & 0x007FFFFF) != 0


def _flush(b: int) -> int:
    return b & 0x80000000 if _is_denormal(b) else b


def round_f32(x: Fraction | float, rz: bool = False, ftz: bool = False) -> int:
    """Round an exact value (a Fraction or a float) once to binary32 bits.

    Ties go to even; rz rounds toward zero instead. An exact zero comes back as
    +0: the caller decides the sign of a zero result.
    """
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return float_to_bits(x)
        x = Fraction(x)
    if x == 0:
        return 0
    sign = 0x80000000 if x < 0 else 0
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if _pow2(e) > a:
        e -= 1
    e = max(e, -126)  # below that the quantum stays 2^-149 (subnormals)
    m = a / _pow2(e - 23)
    n = m.numerator // m.denominator
    rem = m - n
    if not rz and (rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n & 1)):
        n += 1
    if n == 1 << 24:
        n, e = 1 << 23, e + 1
    if e > 127:
        return sign | (0x7F7FFFFF if rz else 0x7F800000)
    b = n if n < (1 << 23) else ((e + 127) << 23) | (n - (1 << 23))
    b |= sign
    return _flush(b) if ftz else b


def _pow2(e: int) -> Fraction:
    return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)


def f32_value(b: int, ftz: bool) -> float:
    """The value of operand bits, flushed to signed zero when the instruction says FTZ."""
    return bits_to_float(_flush(b) if ftz else b)


# ---- shader program header -----------------------------------------------------

IMAP_UNUSED, IMAP_CONSTANT, IMAP_PERSPECTIVE, IMAP_SCREEN_LINEAR = 0, 1, 2, 3


def parse_sph(sph: bytes) -> dict:
    """The fields of a shader program header that the first draw needs.

    Layout (open-gpu-doc, Shader Program Header, words of 32 bits):
    CommonWord0 bits 0-4 SphType (1 vertex/tessellation/geometry, 2 pixel),
    bits 5-9 Version, bits 10-13 ShaderType (1 vertex .. 5 pixel), bit 15
    KillsPixels. For a pixel shader (type 2): word 5 ImapSystemValuesA/B,
    words 6-13 ImapGenericVector[32], one byte per vector holding the X, Y, Z
    and W components' interpolation in two bits each (0 unused, 1 constant,
    2 perspective, 3 screen linear), word 18 OmapTarget[8], four component bits
    per colour target, and word 19 bit 0 OmapSampleMask, bit 1 OmapDepth.
    """
    if len(sph) < 0x50:
        raise ValueError(f"a shader program header has at least 0x50 bytes, got {len(sph)}")
    w = struct.unpack_from("<20I", sph)
    out = {
        "sph_type": w[0] & 0x1F,
        "version": (w[0] >> 5) & 0x1F,
        "shader_type": (w[0] >> 10) & 0xF,
        "kills_pixels": bool((w[0] >> 15) & 1),
    }
    if out["sph_type"] == 2:
        imap = []
        for i in range(32):
            byte = (w[6 + (i >> 2)] >> ((i & 3) * 8)) & 0xFF
            imap.append(tuple((byte >> (2 * c)) & 3 for c in range(4)))
        out["imap_generic"] = imap
        out["omap_target"] = [(w[18] >> (4 * t)) & 0xF for t in range(8)]
        out["omap_sample_mask"] = bool(w[19] & 1)
        out["omap_depth"] = bool(w[19] & 2)
    return out


def pixel_output_registers(sph: dict) -> dict[tuple[int, int], int]:
    """(target, component) to the register that holds it at EXIT.

    The enabled components of the enabled targets take consecutive registers
    from R0 in target then component order (the omap packing).
    """
    regs = {}
    r = 0
    for t, mask in enumerate(sph["omap_target"]):
        for c in range(4):
            if mask & (1 << c):
                regs[(t, c)] = r
                r += 1
    return regs


# ---- listing ---------------------------------------------------------------

@dataclass
class Instr:
    addr: int
    guard: tuple[str, bool] | None  # (predicate name, negated)
    op: str
    mods: tuple[str, ...]
    operands: list[str]
    lo: int  # the instruction's two 64-bit words, as nvdisasm prints them
    hi: int
    text: str


_LINE = re.compile(r"^\s*/\*([0-9a-fA-F]{4,})\*/\s*(.*?)\s*;\s*/\*\s*(0x[0-9a-fA-F]{16})\s*\*/\s*$")
_HI = re.compile(r"^\s*/\*\s*(0x[0-9a-fA-F]{16})\s*\*/\s*$")


def parse_listing(text: str) -> list[Instr]:
    """Instructions from `nvdisasm -hex` output, in address order."""
    lines = text.splitlines()
    out: list[Instr] = []
    i = 0
    while i < len(lines):
        m = _LINE.match(lines[i])
        i += 1
        if not m:
            continue
        if i >= len(lines) or not _HI.match(lines[i]):
            raise ValueError(f"no second word after {lines[i - 1].strip()!r}")
        hi = int(_HI.match(lines[i]).group(1), 16)
        i += 1
        addr = int(m.group(1), 16)
        body = m.group(2).strip()
        guard = None
        gm = re.match(r"^@(!?)(U?P[0-6T])\s+", body)
        if gm:
            guard = (gm.group(2), gm.group(1) == "!")
            body = body[gm.end():]
        head, _, rest = body.partition(" ")
        parts = head.split(".")
        operands = [o.strip() for o in rest.split(",")] if rest.strip() else []
        out.append(Instr(addr, guard, parts[0], tuple(parts[1:]), operands, int(m.group(3), 16), hi, body))
    if not out:
        raise ValueError("no instruction in the listing")
    return out


# ---- machine ---------------------------------------------------------------

@dataclass
class Env:
    """What a program reads from outside. Every callback returns 32-bit words.

    tex(handle, dim, coords, layer, lod) samples the texture whose handle sits
    at word `handle` of c[2]: dim is nvdisasm's dimension (2D, 3D, CUBE,
    ARRAY_2D), coords the floats it reads (2, 3 or the cube direction), layer
    the integer array index (None when not an array), lod None for the
    implicit level of detail, else the explicit one (0.0 for LZ)."""
    cbank: Callable[[int, int], int] = lambda bank, offset: 0
    gload: Callable[[int, int], bytes] | None = None  # (address, size) -> bytes
    ald: Callable[[int], int] | None = None  # input attribute word at an address
    ipa: Callable[[int], int] | None = None  # IPA.PASS value (bits) of an attribute address
    ipa_flat: Callable[[int], int] | None = None  # IPA.CONSTANT value (bits): the provoking vertex's word
    tex: Callable[[int, str, list, int | None, float | None], tuple[float, float, float, float]] | None = None


@dataclass
class Result:
    stores: dict[int, int] = field(default_factory=dict)  # attribute address -> word (AST)
    regs: dict[int, int] = field(default_factory=dict)
    killed: bool = False
    steps: int = 0
    exit_addr: int = -1


_CMP = {
    "LT": lambda a, b: a < b, "GT": lambda a, b: a > b, "LE": lambda a, b: a <= b,
    "GE": lambda a, b: a >= b, "EQ": lambda a, b: a == b, "NE": lambda a, b: a != b,
}
_BOOL = {"AND": lambda a, b: a and b, "OR": lambda a, b: a or b, "XOR": lambda a, b: a != b}


_PRED = re.compile(r"!?U?P[0-6T]")
_M32 = 0xFFFFFFFF
_M64 = (1 << 64) - 1


def _s32(v: int) -> int:
    v &= _M32
    return v - (1 << 32) if v & 0x80000000 else v


class Machine:
    def __init__(self, prog: list[Instr], env: Env, tex_sources_from_ra: bool = False):
        # tex_sources_from_ra reads every TEX source word from Ra, Ra+1, ... (the order of earlier
        # architectures): a deliberate error, for a counter-run of the operand split in `_tex`.
        self.tex_sources_from_ra = tex_sources_from_ra
        self.prog = prog
        self.by_addr = {ins.addr: k for k, ins in enumerate(prog)}
        self.env = env
        self.r = [0] * 256
        self.ur = [0] * 64
        self.p = [False] * 7
        self.calls: list[int] = []  # return addresses of CALL, for RET
        self.res = Result()

    # -- operands --

    def reg(self, name: str) -> int:
        name = name.split(".")[0]
        if name in ("RZ", "SRZ"):
            return 0
        if name == "URZ":
            return 0
        if name.startswith("UR"):
            return self.ur[int(name[2:])]
        if name.startswith("R"):
            return self.r[int(name[1:])]
        raise Unsupported(f"register {name!r}")

    def set_reg(self, name: str, value: int) -> None:
        name = name.split(".")[0]
        if name == "RZ":
            return
        if name.startswith("UR"):
            self.ur[int(name[2:])] = value & 0xFFFFFFFF
        elif name.startswith("R"):
            self.r[int(name[1:])] = value & 0xFFFFFFFF
        else:
            raise Unsupported(f"destination {name!r}")

    def reg_index(self, name: str) -> int:
        name = name.split(".")[0]
        if name == "RZ":
            return 255
        if not re.fullmatch(r"R\d+", name):
            raise Unsupported(f"vector register {name!r}")
        return int(name[1:])

    def pred(self, name: str) -> bool:
        neg = name.startswith("!")
        name = name.lstrip("!")
        if name == "PT":
            v = True
        elif re.fullmatch(r"P[0-6]", name):
            v = self.p[int(name[1:])]
        else:
            raise Unsupported(f"predicate {name!r}")
        return v != neg

    def set_pred(self, name: str, v: bool) -> None:
        if name == "PT":
            return
        if not re.fullmatch(r"P[0-6]", name):
            raise Unsupported(f"predicate destination {name!r}")
        self.p[int(name[1:])] = v

    def word(self, opnd: str) -> int:
        """An integer or raw-bits source: register, constant bank or immediate."""
        cm = re.fullmatch(r"c\[(0x[0-9a-fA-F]+)\]\[(0x[0-9a-fA-F]+)\]", opnd)
        if cm:
            return self.env.cbank(int(cm.group(1), 16), int(cm.group(2), 16)) & 0xFFFFFFFF
        if re.fullmatch(r"-?0x[0-9a-fA-F]+", opnd):
            return int(opnd, 16) & 0xFFFFFFFF
        if re.fullmatch(r"U?R(Z|\d+)(\.reuse)?|SRZ", opnd):
            return self.reg(opnd)
        raise Unsupported(f"integer operand {opnd!r}")

    def fsrc(self, opnd: str, ftz: bool) -> float:
        """A float source with its negate and absolute-value modifiers."""
        neg = opnd.startswith("-")
        if neg or opnd.startswith("+"):
            opnd = opnd[1:]
        absv = opnd.startswith("|") and opnd.endswith("|")
        if absv:
            opnd = opnd[1:-1]
        if re.fullmatch(r"[0-9.]+(e[-+]?\d+)?|INF|NAN|QNAN", opnd, re.IGNORECASE):
            v = float(opnd) if opnd.upper() not in ("QNAN", "NAN") else math.nan
            v = f32_value(float_to_bits(v), ftz)  # float immediates are binary32 values
        else:
            v = f32_value(self.word(opnd), ftz)
        if absv:
            v = abs(v)
        if neg:
            v = -v
        return v

    def fbits(self, opnd: str) -> int:
        """A float source's bits with its negate and absolute-value modifiers applied
        to the sign bit, nothing flushed (the select instructions move bits)."""
        neg = opnd.startswith("-")
        if neg or opnd.startswith("+"):
            opnd = opnd[1:]
        absv = opnd.startswith("|") and opnd.endswith("|")
        if absv:
            opnd = opnd[1:-1]
        if re.fullmatch(r"[0-9.]+(e[-+]?\d+)?|INF", opnd, re.IGNORECASE):
            b = float_to_bits(float(opnd))
        else:
            b = self.word(opnd)
        if absv:
            b &= 0x7FFFFFFF
        if neg:
            b ^= 0x80000000
        return b

    def ival(self, opnd: str) -> int:
        """An integer source with a two's complement negate (-) or a bitwise not (~)."""
        if opnd.startswith("-") and not re.fullmatch(r"-0x[0-9a-fA-F]+", opnd):
            return -self.word(opnd[1:]) & _M32
        if opnd.startswith("~"):
            return ~self.word(opnd[1:]) & _M32
        return self.word(opnd)

    def pair(self, name: str) -> int:
        """A 64-bit value from a register pair (Rn, Rn+1) or a uniform pair."""
        name = name.split(".")[0]
        if name in ("RZ", "URZ"):
            return 0
        if name.startswith("UR"):
            k = int(name[2:])
            return self.ur[k] | (self.ur[k + 1] << 32)
        k = self.reg_index(name)
        return self.r[k] | (self.r[k + 1] << 32)

    def address(self, text: str) -> int:
        """A memory operand [a+b+...]: Rn is a 64-bit pair, Rn.U32 a 32-bit offset,
        URn a 64-bit uniform pair, then an immediate."""
        m = re.fullmatch(r"\[(.*)\]", text)
        if not m:
            raise Unsupported(f"address {text!r}")
        total = 0
        for term in m.group(1).split("+"):
            term = term.strip()
            if re.fullmatch(r"-?0x[0-9a-fA-F]+", term):
                total += int(term, 16)
            elif re.fullmatch(r"R\d+\.U32", term):
                total += self.r[self.reg_index(term)]
            elif re.fullmatch(r"U?R\d+|U?RZ", term):
                total += self.pair(term)
            else:
                raise Unsupported(f"address term {term!r} in {text!r}")
        return total & _M64

    # -- execution --

    def run(self) -> Result:
        pc = 0
        while pc < len(self.prog):
            self.res.steps += 1
            if self.res.steps > MAX_STEPS:
                raise RuntimeError("the program did not exit")
            ins = self.prog[pc]
            pc += 1
            if ins.guard is not None:
                name, neg = ins.guard
                if self.pred(("!" if neg else "") + name) is False:
                    continue
            target = self.step(ins)
            if target == "exit":
                self.res.exit_addr = ins.addr
                self.res.regs = {i: v for i, v in enumerate(self.r)}
                return self.res
            if target is not None:
                if target not in self.by_addr:
                    raise Unsupported(f"branch to {target:#x}, not an instruction")
                pc = self.by_addr[target]
        raise RuntimeError("ran past the end of the program")

    def step(self, ins: Instr):
        op, mods, o = ins.op, set(ins.mods), ins.operands
        ftz = "FTZ" in mods
        if op == "NOP":
            return None
        if op == "EXIT":
            if mods:
                raise Unsupported(ins.text)
            if o and not self.pred(o[0]):
                return None
            return "exit"
        if op == "BRA":
            # BRA [!]Pp, target is taken where both the guard and Pp hold.
            if mods or len(o) not in (1, 2) or (len(o) == 2 and not _PRED.fullmatch(o[0])):
                raise Unsupported(ins.text)
            if len(o) == 2 and not self.pred(o[0]):
                return None
            return int(o[-1], 16)
        if op in ("BSSY", "BSYNC", "DEPBAR"):
            # Convergence barriers and scoreboard waits: nothing to do for one thread.
            if (op == "DEPBAR" and mods != {"LE"}) or (op != "DEPBAR" and mods):
                raise Unsupported(ins.text)
            return None
        if op == "CALL":
            if mods != {"REL", "NOINC"} or len(o) != 1:
                raise Unsupported(ins.text)
            self.calls.append(ins.addr + 16)
            return int(o[0], 16)
        if op == "RET":
            if mods != {"REL", "NODEC"} or not self.calls:
                raise Unsupported(ins.text)
            return self.calls.pop()
        if op == "BRX":
            # BRX Ra imm: to the 64-bit value of Ra, Ra+1 plus imm, relative to the next instruction.
            m = re.fullmatch(r"(R\d+) (-?0x[0-9a-fA-F]+)", " ".join(o))
            if mods or not m:
                raise Unsupported(ins.text)
            return (self.pair(m.group(1)) + int(m.group(2), 16) + ins.addr + 16) & _M64
        if op == "PLOP3":
            # PLOP3.LUT Pd0, Pd1, Pa, Pb, Pc, lut0, lut1: Pd0 = lut0 of (Pa, Pb, Pc).
            if ins.mods != ("LUT",) or len(o) != 7 or o[1] != "PT":
                raise Unsupported(ins.text)
            idx = (int(self.pred(o[2])) << 2) | (int(self.pred(o[3])) << 1) | int(self.pred(o[4]))
            self.set_pred(o[0], bool((int(o[5], 16) >> idx) & 1))
            return None
        if op in ("IADD3", "UIADD3"):
            return self._iadd3(ins)
        if op == "LEA":
            # LEA Rd, [Pd,] Ra, Rb, s: Rd = (Ra << s) + Rb, Pd the carry out.
            dst, rest = o[0], o[1:]
            pd = rest.pop(0) if rest and _PRED.fullmatch(rest[0]) else None
            if mods or len(rest) != 3:
                raise Unsupported(ins.text)
            total = ((self.ival(rest[0]) << int(rest[2], 16)) & _M32) + self.ival(rest[1])
            self.set_reg(dst, total)
            if pd:
                self.set_pred(pd, total >> 32 != 0)
            return None
        if op == "SHF":
            return self._shf(ins)
        if op in ("IMAD",):
            # The low 32 bits of a * b + c (.MOV, .SHL and .IADD are the same operation with a constant operand).
            if mods - {"U32", "MOV", "SHL", "IADD"} or len(o) != 4:
                raise Unsupported(ins.text)
            self.set_reg(o[0], self.ival(o[1]) * self.ival(o[2]) + self.ival(o[3]))
            return None
        if op == "IMNMX":
            # IMNMX Rd, Ra, Rb, Pp: the minimum where Pp holds, else the maximum; signed unless .U32.
            if mods - {"U32"} or len(o) != 4:
                raise Unsupported(ins.text)
            conv = (lambda v: v) if "U32" in mods else _s32
            a, b = conv(self.ival(o[1])), conv(self.ival(o[2]))
            self.set_reg(o[0], min(a, b) if self.pred(o[3]) else max(a, b))
            return None
        if op == "ISETP":
            cmp = [m for m in ins.mods if m in _CMP]
            boolop = [m for m in ins.mods if m in _BOOL]
            if len(cmp) != 1 or len(boolop) != 1 or set(ins.mods) - {cmp[0], boolop[0], "U32"} or len(o) != 5:
                raise Unsupported(ins.text)
            conv = (lambda v: v) if "U32" in mods else _s32
            r = _CMP[cmp[0]](conv(self.ival(o[2])), conv(self.ival(o[3])))
            p = self.pred(o[4])
            self.set_pred(o[0], _BOOL[boolop[0]](r, p))
            self.set_pred(o[1], _BOOL[boolop[0]](not r, p))
            return None
        if op == "LDC":
            # LDC Rd, c[bank][Ra+imm]: a constant bank word at a register offset.
            width = 64 if "64" in mods else 32
            m = re.fullmatch(r"c\[(0x[0-9a-fA-F]+)\]\[(R\d+|RZ)(?:\+(-?0x[0-9a-fA-F]+))?\]", o[1]) if len(o) == 2 else None
            if mods - {"64"} or not m:
                raise Unsupported(ins.text)
            off = (self.word(m.group(2)) + (int(m.group(3), 16) if m.group(3) else 0)) & _M32
            if off % (width // 8):
                raise RuntimeError(f"misaligned constant load at {off:#x}")
            d = self.reg_index(o[0])
            for k in range(width // 32):
                self.r[d + k] = self.env.cbank(int(m.group(1), 16), off + 4 * k) & _M32
            return None
        if op == "FMNMX":
            # FMNMX Rd, Ra, Rb, Pp: the minimum where Pp holds, else the maximum; a NaN loses to a number, -0 < +0.
            if mods - {"FTZ"} or len(o) != 4:
                raise Unsupported(ins.text)
            self.set_reg(o[0], _fmnmx(self.fsrc(o[1], ftz), self.fsrc(o[2], ftz), self.pred(o[3])))
            return None
        if op == "FSEL":
            # FSEL Rd, Ra, Rb, Pp: Ra where Pp holds, else Rb, as bits.
            if mods or len(o) != 4:
                raise Unsupported(ins.text)
            self.set_reg(o[0], self.fbits(o[1]) if self.pred(o[3]) else self.fbits(o[2]))
            return None
        if op == "FSET":
            # FSET.BF.cmp.bool Rd, Ra, Rb, Pp: 1.0 where the comparison (combined with Pp) holds, else 0.0.
            cmp = [m for m in ins.mods if m.rstrip("U") in _CMP]
            boolop = [m for m in ins.mods if m in _BOOL]
            if (len(cmp) != 1 or len(boolop) != 1 or "BF" not in mods
                    or set(ins.mods) - {cmp[0], boolop[0], "FTZ", "BF"} or len(o) != 4):
                raise Unsupported(ins.text)
            r = _fcmp(cmp[0], self.fsrc(o[1], ftz), self.fsrc(o[2], ftz))
            self.set_reg(o[0], 0x3F800000 if _BOOL[boolop[0]](r, self.pred(o[3])) else 0)
            return None
        if op == "FRND":
            # Round to an integral value: to nearest even by default, else .FLOOR, .CEIL or .TRUNC.
            modes = [m for m in ins.mods if m in ("FLOOR", "CEIL", "TRUNC")]
            if set(ins.mods) - {"FLOOR", "CEIL", "TRUNC", "FTZ"} or len(modes) > 1 or len(o) != 2:
                raise Unsupported(ins.text)
            self.set_reg(o[0], _frnd(self.fsrc(o[1], ftz), modes[0] if modes else "RN"))
            return None
        if op == "F2I":
            return self._f2i(ins)
        if op == "I2FP":
            # I2FP.F32.S32|U32 Rd, Ra: to the nearest binary32.
            if ins.mods not in (("F32", "S32"), ("F32", "U32")) or len(o) != 2:
                raise Unsupported(ins.text)
            v = self.ival(o[1])
            self.set_reg(o[0], round_f32(Fraction(_s32(v) if ins.mods[1] == "S32" else v)))
            return None
        if op == "HADD2":
            # HADD2.F32 Rd, -RZ, Rb.H0_H0|H1_H1: one half of Rb widened to binary32 (adding -0 changes no value).
            m = re.fullmatch(r"(R\d+)(?:\.reuse)?\.(H0_H0|H1_H1)", o[2]) if len(o) == 3 else None
            if ins.mods != ("F32",) or o[1] != "-RZ" or not m:
                raise Unsupported(ins.text)
            word = self.reg(m.group(1))
            half = word & 0xFFFF if m.group(2) == "H0_H0" else word >> 16
            self.set_reg(o[0], float_to_bits(struct.unpack("<e", struct.pack("<H", half))[0]))
            return None
        if op == "KILL":
            if mods or o:
                raise Unsupported(ins.text)
            self.res.killed = True
            return "exit"
        if op == "MOV":
            if mods or len(o) not in (2, 3) or (len(o) == 3 and o[2] != "0xf"):
                raise Unsupported(ins.text)
            self.set_reg(o[0], self.word(o[1]))
            return None
        if op == "CS2R":
            if mods or o[1] != "SRZ":
                raise Unsupported(ins.text)
            d = self.reg_index(o[0])
            self.r[d] = self.r[d + 1] = 0
            return None
        if op in ("FMUL", "FADD", "FFMA"):
            allowed = {"FTZ", "RZ", "SAT", "D2", "D4", "D8", "M2", "M4", "M8"}
            if mods - allowed:
                raise Unsupported(ins.text)
            n = {"FMUL": 2, "FADD": 2, "FFMA": 3}[op]
            vals = [self.fsrc(x, ftz) for x in o[1:1 + n]]
            if len(o) != 1 + n:
                raise Unsupported(ins.text)
            if any(math.isnan(v) for v in vals):
                b = QNAN
            else:
                exact = _exact(op, vals)
                if exact is None or exact == 0:  # an infinity, or a zero whose sign IEEE decides
                    b = float_to_bits(_float_op(op, vals))
                else:
                    for m, s in (("D2", Fraction(1, 2)), ("D4", Fraction(1, 4)), ("D8", Fraction(1, 8)),
                                 ("M2", 2), ("M4", 4), ("M8", 8)):
                        if m in mods:
                            exact *= s
                    b = round_f32(exact, rz="RZ" in mods, ftz=ftz)
            if "SAT" in mods:
                b = _saturate(b)
            self.set_reg(o[0], b)
            return None
        if op == "MUFU":
            fn = ins.mods[0] if ins.mods else ""
            if len(ins.mods) != 1 or len(o) != 2:
                raise Unsupported(ins.text)
            x = self.fsrc(o[1], True)
            self.set_reg(o[0], _mufu(fn, x))
            return None
        if op == "FSETP":
            cmp = [m for m in ins.mods if m.rstrip("U") in _CMP]
            boolop = [m for m in ins.mods if m in _BOOL]
            if len(cmp) != 1 or len(boolop) != 1 or set(ins.mods) - {cmp[0], boolop[0], "FTZ"} or len(o) != 5:
                raise Unsupported(ins.text)
            r = _fcmp(cmp[0], self.fsrc(o[2], ftz), self.fsrc(o[3], ftz))
            p = self.pred(o[4])
            self.set_pred(o[0], _BOOL[boolop[0]](r, p))
            self.set_pred(o[1], _BOOL[boolop[0]](not r, p))
            return None
        if op == "LOP3":
            if ins.mods != ("LUT",) or len(o) != 7:
                raise Unsupported(ins.text)
            a, b, c = self.word(o[2]), self.word(o[3]), self.word(o[4])
            lut = int(o[5], 16)
            r = _lop3(a, b, c, lut)
            if o[6] != "!PT":
                # The extra predicate source; NAK always encodes it as false.
                raise Unsupported(ins.text)
            self.set_pred(o[0], r != 0)
            self.set_reg(o[1], r)
            return None
        if op == "ULDC":
            width = 64 if "64" in mods else 32
            if mods - {"64"}:
                raise Unsupported(ins.text)
            d = int(o[0][2:])
            cm = re.fullmatch(r"c\[(0x[0-9a-fA-F]+)\]\[(0x[0-9a-fA-F]+)\]", o[1])
            if not cm:
                raise Unsupported(ins.text)
            bank, off = int(cm.group(1), 16), int(cm.group(2), 16)
            for k in range(width // 32):
                self.ur[d + k] = self.env.cbank(bank, off + 4 * k) & 0xFFFFFFFF
            return None
        if op == "LDG":
            # .CONSTANT and .STRONG.SM name the cache and the memory ordering, which one thread cannot observe.
            width = 128 if "128" in mods else 64 if "64" in mods else 32
            if mods - {"E", "64", "128", "CONSTANT", "STRONG", "SM"} or "E" not in mods or len(o) != 2:
                raise Unsupported(ins.text)
            addr = self.address(o[1])
            if addr % (width // 8):
                raise RuntimeError(f"misaligned {width}-bit load at {addr:#x}")
            data = self.env.gload(addr, width // 8)
            d = self.reg_index(o[0])
            for k, (v,) in enumerate(struct.iter_unpack("<I", data)):
                self.r[d + k] = v
            return None
        if op == "ALD":
            width = int(ins.mods[0]) if ins.mods else 32
            am = re.fullmatch(r"a\[(0x[0-9a-fA-F]+)\]", o[1])
            if not am or len(ins.mods) > 1 or len(o) != 2:
                raise Unsupported(ins.text)
            d = self.reg_index(o[0])
            base = int(am.group(1), 16)
            for k in range(width // 32):
                self.r[d + k] = self.env.ald(base + 4 * k) & 0xFFFFFFFF
            return None
        if op == "AST":
            width = int(ins.mods[0]) if ins.mods else 32
            am = re.fullmatch(r"a\[(0x[0-9a-fA-F]+)\]", o[0])
            if not am or len(ins.mods) > 1 or len(o) != 2:
                raise Unsupported(ins.text)
            s = self.reg_index(o[1])
            base = int(am.group(1), 16)
            for k in range(width // 32):
                self.res.stores[base + 4 * k] = self.r[s + k]
            return None
        if op == "IPA":
            # PASS: the hardware's screen-linear interpolation (see reference.py); CONSTANT: flat, the provoking vertex's word.
            mode = ins.mods[0] if ins.mods else ""
            if mode not in ("PASS", "CONSTANT") or set(ins.mods) - {mode, "SAT"} or len(o) != 2:
                raise Unsupported(ins.text)
            am = re.fullmatch(r"a\[(0x[0-9a-fA-F]+)\]", o[1])
            if not am:
                raise Unsupported(ins.text)
            b = (self.env.ipa if mode == "PASS" else self.env.ipa_flat)(int(am.group(1), 16))
            self.set_reg(o[0], _saturate(b) if "SAT" in mods else b)
            return None
        if op == "TEX":
            return self._tex(ins)
        raise Unsupported(ins.text)

    def _tex(self, ins: Instr):
        # TEX[.SCR][.LZ|.LL][.NODEP] Rd2, Rd, Ra, Rb, 0x2, handle, dim[, mask].
        # The fields are checked against the instruction words: Rd in bits 16-23,
        # Ra 24-31, Rb 32-39, the handle in bits 40-53, Rd2 in bits 64-71 and the
        # write mask in bits 72-75 (Mesa NAK, the SM70 encoding of OpTex); nvdisasm
        # prints no mask when all four components are written.
        # Sources: n words, the array index (an integer) first, then the
        # coordinates, then an explicit level of detail; the first ceil(n / 2)
        # come from Ra, Ra+1 and the rest from Rb, Rb+1. That split is read from
        # the compiler's own placement in the programs drawn: a 2D sample whose x
        # and y are computed separately reads x from Ra and y from Rb, a 3D one x
        # and y from Ra, Ra+1 and z from Rb.
        # Results: the components the mask enables, in order, packed into Rd,
        # Rd+1, Rd2, Rd2+1 (a TEX that keeps only alpha writes it to Rd).
        mods = set(ins.mods)
        o = ins.operands
        lod_mode = mods & {"LZ", "LL"}
        if mods - {"SCR", "NODEP", "LZ", "LL"} or len(lod_mode) > 1 or len(o) not in (7, 8) or o[4] != "0x2":
            raise Unsupported(ins.text)
        dim = o[6]
        if dim not in _TEX_COORDS:
            raise Unsupported(ins.text)
        mask = int(o[7], 16) if len(o) == 8 else 0xF
        rd2, rd, ra, rb = (self.reg_index(x) for x in o[:4])
        handle = int(o[5], 16)
        lo, hi = ins.lo, ins.hi
        fields = ((lo >> 16) & 0xFF, (lo >> 24) & 0xFF, (lo >> 32) & 0xFF, (lo >> 40) & 0x3FFF, hi & 0xFF, (hi >> 8) & 0xF)
        if fields != (rd, ra, rb, handle, rd2, mask):
            raise Unsupported(f"{ins.text}: operand order does not match the encoding {fields}")
        array = dim.startswith("ARRAY")
        n = int(array) + _TEX_COORDS[dim] + int("LL" in mods)
        if n > 4:
            raise Unsupported(f"{ins.text}: {n} source words")
        na = n if self.tex_sources_from_ra else (n + 1) // 2
        regs = [ra + k if ra != 255 else 255 for k in range(na)] + [rb + k if rb != 255 else 255 for k in range(n - na)]
        words = [self.r[r] if r != 255 else 0 for r in regs]
        layer = words.pop(0) & 0xFFFF if array else None
        lod = f32_value(words.pop(), False) if "LL" in mods else (0.0 if "LZ" in mods else None)
        coords = [f32_value(w, False) for w in words]
        rgba = self.env.tex(handle, dim, coords, layer, lod)
        dests = ([rd, rd + 1] if rd != 255 else [255, 255]) + ([rd2, rd2 + 1] if rd2 != 255 else [255, 255])
        k = 0
        for c in range(4):
            if mask & (1 << c):
                if dests[k] != 255:
                    self.r[dests[k]] = float_to_bits(rgba[c])
                k += 1
        return None

    def _iadd3(self, ins: Instr):
        # IADD3 Rd, [Pd,] a, b, c: a + b + c, Pd the carry out; a negated source
        # adds its complement plus one (so a subtraction carries where it does not
        # borrow). IADD3.X Rd, a, b, c, Pc0, Pc1 adds the two carries in instead.
        o, mods = ins.operands, set(ins.mods)
        dst, i, pds = o[0], 1, []
        while i < len(o) and _PRED.fullmatch(o[i]):
            pds.append(o[i])
            i += 1
        srcs, cins = o[i:i + 3], o[i + 3:]
        if mods - {"X"} or len(srcs) != 3 or len(pds) > 1:
            raise Unsupported(ins.text)
        total = 0
        if "X" in mods:
            if len(cins) != 2 or any(s[:1] in ("-", "~") and not s.startswith("-0x") for s in srcs):
                raise Unsupported(ins.text)
            total = sum(self.word(s) for s in srcs) + sum(int(self.pred(c)) for c in cins)
        else:
            if cins:
                raise Unsupported(ins.text)
            for s in srcs:
                if s.startswith("-") and not s.startswith("-0x"):
                    total += (~self.word(s[1:]) & _M32) + 1
                else:
                    total += self.ival(s)
        if total >> 32 > 1 and pds:
            raise Unsupported(f"{ins.text}: a carry of {total >> 32}")
        self.set_reg(dst, total)
        if pds:
            self.set_pred(pds[0], total >> 32 != 0)
        return None

    def _shf(self, ins: Instr):
        # SHF.L|R.U32|S32|U64|S64[.HI][.W] Rd, Ra, Rb, Rc: the 64-bit Rc:Ra shifted
        # left or right (arithmetic for a signed type) by Rb, the low word, or the
        # high one with .HI. The shift clamps to the type's width, or wraps with .W.
        mods = set(ins.mods)
        kinds = [m for m in ins.mods if m in ("U32", "S32", "U64", "S64")]
        dirs = [m for m in ins.mods if m in ("L", "R")]
        if len(kinds) != 1 or len(dirs) != 1 or mods - {kinds[0], dirs[0], "HI", "W"} or len(ins.operands) != 4:
            raise Unsupported(ins.text)
        o = ins.operands
        a, sh, c = self.ival(o[1]), self.ival(o[2]), self.ival(o[3])
        width = 32 if kinds[0].endswith("32") else 64
        sh = sh & (width - 1) if "W" in mods else min(sh, width)
        val = (c << 32) | a
        if kinds[0][0] == "S" and c & 0x80000000:
            val -= 1 << 64
        res = val << sh if dirs[0] == "L" else val >> sh
        self.set_reg(o[0], res >> 32 if "HI" in mods else res)
        return None

    def _f2i(self, ins: Instr):
        # F2I[.FTZ][.TRUNC|.FLOOR|.CEIL][.NTZ][.S32|.U32] Rd, Ra: to the nearest
        # integer by default; NaN gives 0, out of range saturates; signed unless .U32.
        mods = set(ins.mods)
        modes = [m for m in ins.mods if m in ("TRUNC", "FLOOR", "CEIL")]
        if mods - {"FTZ", "TRUNC", "FLOOR", "CEIL", "NTZ", "S32", "U32"} or len(modes) > 1 or len(ins.operands) != 2:
            raise Unsupported(ins.text)
        x = self.fsrc(ins.operands[1], "FTZ" in mods)
        lo, hi = (0, _M32) if "U32" in mods else (-(1 << 31), (1 << 31) - 1)
        if math.isnan(x):
            v = 0
        elif math.isinf(x):
            v = hi if x > 0 else lo
        else:
            v = min(max(_round_int(Fraction(x), modes[0] if modes else "RN"), lo), hi)
        self.set_reg(ins.operands[0], v & _M32)
        return None


_TEX_COORDS = {"2D": 2, "3D": 3, "CUBE": 3, "ARRAY_2D": 2}


def _round_int(x: Fraction, mode: str) -> int:
    if mode == "FLOOR":
        return math.floor(x)
    if mode == "CEIL":
        return math.ceil(x)
    if mode == "TRUNC":
        return math.trunc(x)
    return round(x)  # Fraction rounds half to even


def _frnd(x: float, mode: str) -> int:
    if math.isnan(x):
        return QNAN
    if math.isinf(x):
        return float_to_bits(x)
    v = float(_round_int(Fraction(x), mode))
    return float_to_bits(math.copysign(v, x) if v == 0 else v)


def _fcmp(c: str, a: float, b: float) -> bool:
    """A float comparison: an ordered one (LT, EQ ...) is false on NaN, an unordered one (LTU, EQU ...) true."""
    if math.isnan(a) or math.isnan(b):
        return c.endswith("U")
    return _CMP[c.rstrip("U")](a, b)


def _fmnmx(a: float, b: float, take_min: bool) -> int:
    if math.isnan(a) and math.isnan(b):
        return QNAN
    if math.isnan(a) or math.isnan(b):
        return float_to_bits(b if math.isnan(a) else a)
    if a == b == 0:
        neg = math.copysign(1, a) < 0, math.copysign(1, b) < 0
        return float_to_bits(-0.0 if (any(neg) if take_min else all(neg)) else 0.0)
    return float_to_bits(min(a, b) if take_min else max(a, b))


def _exact(op: str, vals: list[float]) -> Fraction | None:
    if any(math.isinf(v) for v in vals):
        return None
    f = [Fraction(v) for v in vals]
    if op == "FMUL":
        return f[0] * f[1]
    if op == "FADD":
        return f[0] + f[1]
    return f[0] * f[1] + f[2]


def _float_op(op: str, vals: list[float]) -> float:
    try:
        if op == "FMUL":
            return vals[0] * vals[1]
        if op == "FADD":
            return vals[0] + vals[1]
        return vals[0] * vals[1] + vals[2]
    except (OverflowError, ValueError):
        return math.nan


def _saturate(b: int) -> int:
    v = bits_to_float(b)
    if math.isnan(v) or v <= 0:
        return 0
    if v >= 1:
        return 0x3F800000
    return b


def _mufu(fn: str, x: float) -> int:
    if fn == "RCP":
        if x == 0:
            return 0xFF800000 if math.copysign(1, x) < 0 else 0x7F800000
        if math.isinf(x):
            return 0x80000000 if x < 0 else 0
        return round_f32(Fraction(1) / Fraction(x), ftz=True) if not math.isnan(x) else QNAN
    if fn == "RSQ":
        if math.isnan(x) or x < 0:
            return QNAN
        if x == 0:
            return 0x7F800000
        return float_to_bits(1 / math.sqrt(x))
    if fn == "SQRT":
        return QNAN if math.isnan(x) or x < 0 else float_to_bits(math.sqrt(x))
    if fn == "EX2":
        try:
            return float_to_bits(2.0 ** x)
        except OverflowError:
            return 0x7F800000
    if fn == "LG2":
        if math.isnan(x) or x < 0:
            return QNAN
        return 0xFF800000 if x == 0 else float_to_bits(math.log2(x))
    if fn in ("SIN", "COS"):
        # Since Volta the operand is in turns: the compiler multiplies radians by 1 / (2 pi) first (FMUL.RZ).
        if math.isnan(x) or math.isinf(x):
            return QNAN
        return float_to_bits((math.sin if fn == "SIN" else math.cos)(2 * math.pi * x))
    raise Unsupported(f"MUFU.{fn}")


def _lop3(a: int, b: int, c: int, lut: int) -> int:
    r = 0
    for bit in range(32):
        idx = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
        r |= ((lut >> idx) & 1) << bit
    return r
