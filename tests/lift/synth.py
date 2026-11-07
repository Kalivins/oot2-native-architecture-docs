# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Synthetic modules for the lifter's tests: no game bytes, only code these tests write.

assemble() turns AArch64 assembly into words with clang's own assembler (an encoder independent of the lifter's
decoder), and build_nso() wraps them in a plain NSO module with the parts the generator reads: MOD0, a dynamic
section, exported and imported symbols, relocations, a GOT, an empty .eh_frame and PLT stubs for imports.
"""

from __future__ import annotations

import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

PAGE = 0x1000
DT_NULL, DT_PLTRELSZ, DT_HASH, DT_STRTAB, DT_SYMTAB, DT_RELA, DT_RELASZ, DT_STRSZ, DT_JMPREL = 0, 2, 4, 5, 6, 7, 8, 10, 23
R_AARCH64_JUMP_SLOT = 1026
STT_FUNC, STB_GLOBAL = 2, 1


def align(v: int, a: int = PAGE) -> int:
    return (v + a - 1) & ~(a - 1)


def assemble(cc: str, source: str, workdir: Path) -> tuple[bytes, dict[str, int]]:
    """(.text bytes, {label: offset}) of an AArch64 assembly source, assembled by clang."""
    src, obj = workdir / "synth.s", workdir / "synth.o"
    src.write_text(source, encoding="utf-8", newline="\n")
    done = subprocess.run([cc, "--target=aarch64-none-elf", "-c", str(src), "-o", str(obj)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"assembling failed:\n{done.stderr}")
    return elf_text(obj.read_bytes())


def elf_text(elf: bytes) -> tuple[bytes, dict[str, int]]:
    """.text and the offsets of its symbols, from a relocatable ELF64 little-endian object without relocations."""
    if elf[:4] != b"\x7fELF" or elf[4] != 2 or elf[5] != 1:
        raise ValueError("not an ELF64 little-endian object")
    shoff, = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", elf, shoff + i * shentsize) for i in range(shnum)]
    names = secs[shstrndx]

    def cstr(off: int) -> str:
        return elf[off:elf.index(b"\0", off)].decode()

    by_name = {cstr(names[4] + s[0]): (i, s) for i, s in enumerate(secs)}
    if ".rela.text" in by_name:
        raise ValueError("the test source left relocations in .text: branch to local labels only")
    ti, text = by_name[".text"]
    _, symtab = by_name[".symtab"]
    strtab = secs[symtab[6]]
    labels = {}
    for o in range(symtab[4], symtab[4] + symtab[5], 24):
        name, info, _, shndx, value, _ = struct.unpack_from("<IBBHQQ", elf, o)
        if shndx == ti and name and (info & 0xF) in (0, STT_FUNC):
            labels[cstr(strtab[4] + name)] = value
    return elf[text[4]:text[4] + text[5]], labels


@dataclass
class Symbol:
    name: str
    offset: int | None = None   # text offset of a definition; None for an import


@dataclass
class Module:
    """A plain NSO built from code: text (assembled), exports and imports, and the PLT stubs of the imports.

    plt maps an import's name to the text offset of four placeholder words the source reserved for its stub
    (`adrp x16; ldr x17, [x16, #slot]; add x16, x16, #slot; br x17`); build() writes the stub and the GOT slot."""
    text: bytes
    exports: dict[str, int] = field(default_factory=dict)
    plt: dict[str, int] = field(default_factory=dict)
    data: bytes = b""
    bss: int = PAGE

    def build(self) -> bytes:
        text = bytearray(self.text)
        if len(text) < 8 or len(text) % 4:
            raise ValueError("text must start with two reserved words and hold whole words")
        syms = [Symbol("")] + [Symbol(n, o) for n, o in sorted(self.exports.items())]
        syms += [Symbol(n) for n in sorted(self.plt)]
        ro = align(len(text))
        # rodata: MOD0, dynamic, dynsym, dynstr, hash, rela.plt, eh_frame_hdr, eh_frame
        strtab = bytearray(b"\0")
        name_off = {}
        for s in syms[1:]:
            name_off[s.name] = len(strtab)
            strtab += s.name.encode() + b"\0"
        mod0 = ro
        dyn = mod0 + 0x20
        ndyn = 10
        dynsym = align(dyn + 16 * ndyn, 8)
        dynstr = dynsym + 24 * len(syms)
        hashtab = align(dynstr + len(strtab), 8)
        rela = hashtab + 4 * (2 + 1 + len(syms))
        rela = align(rela, 8)
        imports = [s for s in syms if s.name and s.offset is None]
        eh_hdr = rela + 24 * len(imports)
        eh_frame = eh_hdr + 12
        ro_end = eh_frame + 4
        data_off = align(ro_end)
        got = data_off
        data = bytearray(8 * len(imports)) + self.data
        size = len(data)
        bss_start = data_off + size
        image_end = align(bss_start + self.bss)

        img = bytearray(data_off + size)
        img[0:len(text)] = text
        struct.pack_into("<I", img, 4, mod0)
        struct.pack_into("<4s6i", img, mod0, b"MOD0", dyn - mod0, bss_start - mod0, image_end - mod0,
                         eh_hdr - mod0, ro_end - mod0, 0)
        tags = [(DT_STRTAB, dynstr), (DT_SYMTAB, dynsym), (DT_STRSZ, len(strtab)), (DT_HASH, hashtab),
                (DT_RELA, rela), (DT_RELASZ, 0), (DT_JMPREL, rela), (DT_PLTRELSZ, 24 * len(imports)), (DT_NULL, 0)]
        for i, (tag, val) in enumerate(tags):
            struct.pack_into("<QQ", img, dyn + 16 * i, tag, val)
        for i, s in enumerate(syms):
            if not s.name:
                continue
            shndx, value = (1, s.offset) if s.offset is not None else (0, 0)
            struct.pack_into("<IBBHQQ", img, dynsym + 24 * i, name_off[s.name], STB_GLOBAL << 4 | STT_FUNC, 0,
                             shndx, value, 4)
        img[dynstr:dynstr + len(strtab)] = strtab
        struct.pack_into("<III", img, hashtab, 1, len(syms), 0)   # one bucket, nchain = symbol count
        for i, s in enumerate(imports):
            slot = got + 8 * i
            struct.pack_into("<QQq", img, rela + 24 * i, slot, syms.index(s) << 32 | R_AARCH64_JUMP_SLOT, 0)
            stub = self.plt[s.name]
            for k, word in enumerate(plt_stub(stub, slot)):
                struct.pack_into("<I", img, stub + 4 * k, word)
        # .eh_frame_hdr: version, eh_frame_ptr pcrel sdata4, count udata4 = 0, table datarel sdata4; empty .eh_frame
        struct.pack_into("<BBBBiI", img, eh_hdr, 1, 0x1B, 0x03, 0x3B, eh_frame - (eh_hdr + 4), 0)
        struct.pack_into("<I", img, eh_frame, 0)
        img[data_off:data_off + size] = data

        segs = [(0, bytes(img[0:ro])), (ro, bytes(img[ro:data_off])), (data_off, bytes(img[data_off:]))]
        head = bytearray(0x100)
        struct.pack_into("<4sIII", head, 0, b"NSO" + b"0", 0, 0, 0)
        off = 0x100
        for i, (mem, seg) in enumerate(segs):
            struct.pack_into("<III", head, 0x10 + 0x10 * i, off, mem, len(seg))
            struct.pack_into("<I", head, 0x60 + 4 * i, len(seg))
            off += len(seg)
        struct.pack_into("<I", head, 0x3C, image_end - (data_off + size))
        # The extents of .dynstr and .dynsym relative to rodata, where the runtime's loader reads the symbol count.
        struct.pack_into("<IIII", head, 0x90, dynstr - ro, len(strtab), dynsym - ro, 24 * len(syms))
        return bytes(head) + b"".join(s for _, s in segs)


def plt_stub(pc: int, slot: int) -> list[int]:
    """adrp x16, slot; ldr x17, [x16, #lo12]; add x16, x16, #lo12; br x17 (the shape the generator recognises)."""
    page = ((slot & ~0xFFF) - (pc & ~0xFFF)) >> 12
    imm = page & 0x1FFFFF
    lo12 = slot & 0xFFF
    return [0x90000010 | (imm & 3) << 29 | (imm >> 2) << 5, 0xF9400211 | (lo12 >> 3) << 10,
            0x91000210 | lo12 << 10, 0xD61F0220]


def lz4_literals(data: bytes) -> bytes:
    """A valid LZ4 block holding data as one literal run (no match): enough to test a decoder."""
    n = len(data)
    if n < 15:
        return bytes([n << 4]) + data
    out = bytearray([0xF0])
    n -= 15
    while n >= 255:
        out.append(255)
        n -= 255
    out.append(n)
    return bytes(out) + data
