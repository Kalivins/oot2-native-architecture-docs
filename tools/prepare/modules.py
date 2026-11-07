# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The modules step of `oot2 prepare`: the game's NSO modules to plain modules and a segment table.

Input: the extract step's ExeFS files in exefs-dump/ (main, subsdk0 to subsdk9, sdk; rtld is not loaded, the host
links the modules) and, optionally, the player's own plain modules (--plain-modules DIR).
Output, under the output folder, in exefs/, the folder the lift, compile and link steps (program.py) and oot2.exe read:

    exefs/<name>           each module as a plain NSO: segments stored uncompressed, header flag bits 0 to 2 and 7
                           clear, the hash bits and hashes kept; what tools/lift/gen.py --exefs, tools/layout.py,
                           tools/build_game.py and runtime/core/loader.c read
    exefs/main.npdm        copied from the dump's ExeFS, for the link step and the program's start
    exefs/modules.json     per module, in load order: the file, its size and SHA-256, the build id, text, rodata,
                           data and bss (offset and size in the image), the image size (the size tools/layout.py
                           places), MOD0, the dynamic section and .eh_frame_hdr, and where the module came from

A segment stored LZ4-compressed (flag bits 0 to 2) is decoded in Python (tools/lift/nso.py). A module whose flag
bit 7 is set stores its compressed segments as ZBIC frames instead; those are decoded by the third-party decoder
that zbic.py fetches and builds, unless the player gave that module with --plain-modules, in which case nothing is
fetched. Every segment whose hash bit is set is checked against the SHA-256 in the module's header, which is the
console loader's own check, so a wrong decode cannot pass; a given plain module must also have the same build id and
segment layout as the module in the dump.

The plain layout is the one oot-pc-recomp's tools/nso_plain.py writes (at f4656d3, GPL-3.0-or-later): the 0x100-byte
header, then text, rodata and data back to back, each segment's file offset and both its sizes rewritten. The bytes
between the header and the first segment (the module name, which no reader here uses) are not kept, and the name
offset and size fields are left as they were. So a module decodes to the same file as there, which is what lets the
decode be compared with the oracle's by one hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import bundle_env
import layout
from lift.nso import NsoError, lz4_block

from . import zbic
from .check import TEMP, compiler
from .extract import DUMP_EXEFS
from .runner import Context, PrepareError, Step, file_sha256, fingerprint

FORMAT = 1
OUT, TABLE, NPDM = "exefs", "modules.json", "main.npdm"
HDR = 0x100
LZ4_BITS = 0x7
ZBIC_BIT = 0x80
SEGMENTS = ("text", "rodata", "data")
PAGE = 0x1000
HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent


class Header:
    """The fields of an NSO0 header this step reads or rewrites."""

    def __init__(self, raw: bytes, what: str):
        if len(raw) < HDR or raw[:4] != b"NSO0":
            raise PrepareError(f"{what} is not an NSO0 module")
        self.raw = raw[:HDR]
        self.flags = struct.unpack_from("<I", raw, 0xC)[0]
        self.segs = [struct.unpack_from("<III", raw, 0x10 + i * 0x10) for i in range(3)]   # file offset, mem, size
        self.stored = [struct.unpack_from("<I", raw, 0x60 + i * 4)[0] for i in range(3)]
        self.bss_size = struct.unpack_from("<I", raw, 0x3C)[0]
        self.build_id = raw[0x40:0x60]
        self.hashes = [raw[0xA0 + 32 * i:0xC0 + 32 * i] for i in range(3)]

    @property
    def codec(self) -> str:
        if not self.flags & LZ4_BITS:
            return "plain"
        return "zbic" if self.flags & ZBIC_BIT else "lz4"

    def hashed(self, i: int) -> bool:
        return bool(self.flags & (8 << i))


def read_segments(raw: bytes, hdr: Header, what: str, zbic_decode=None) -> list[bytes]:
    """The three segments of a module, decoded and checked against the header's hashes."""
    out = []
    for i, name in enumerate(SEGMENTS):
        file_off, _, size = hdr.segs[i]
        stored = hdr.stored[i] if hdr.flags & (1 << i) else size
        if file_off < HDR or file_off + stored > len(raw):
            raise PrepareError(f"{what}: its {name} segment lies outside the file")
        data = raw[file_off:file_off + stored]
        if hdr.flags & (1 << i):
            try:
                if hdr.flags & ZBIC_BIT:
                    if zbic_decode is None:
                        raise PrepareError(f"{what} needs the ZBIC decoder")
                    data = zbic_decode(data, size)
                else:
                    data = lz4_block(data, size)
            except (ValueError, NsoError) as e:
                raise PrepareError(f"{what}: its {name} segment does not decode ({e})") from None
        if len(data) != size:
            raise PrepareError(f"{what}: its {name} segment holds {len(data):,} bytes, the header says {size:,}")
        if hdr.hashed(i) and hashlib.sha256(data).digest() != hdr.hashes[i]:
            raise PrepareError(f"{what}: its {name} segment does not match the SHA-256 in its header")
        out.append(data)
    return out


def plain_bytes(hdr: Header, segs: list[bytes]) -> bytes:
    """The plain module for a header and its decoded segments (the layout of oot-pc-recomp tools/nso_plain.py)."""
    out = bytearray(hdr.raw)
    pos = HDR
    for i, data in enumerate(segs):
        struct.pack_into("<III", out, 0x10 + i * 0x10, pos, hdr.segs[i][1], len(data))
        struct.pack_into("<I", out, 0x60 + i * 4, len(data))
        pos += len(data)
    struct.pack_into("<I", out, 0xC, hdr.flags & ~(LZ4_BITS | ZBIC_BIT))
    return bytes(out) + b"".join(segs)


def image_read(hdr: Header, segs: list[bytes], addr: int, n: int) -> bytes:
    """n bytes at an image offset, from the segment that holds them (zeros in bss)."""
    for (_, mem, size), data in zip(hdr.segs, segs):
        if mem <= addr and addr + n <= mem + size:
            return data[addr - mem:addr - mem + n]
    data_end = hdr.segs[2][1] + hdr.segs[2][2]
    if data_end <= addr and addr + n <= data_end + hdr.bss_size:
        return bytes(n)
    raise PrepareError(f"image offset {addr:#x} lies outside the module's segments")


def table_entry(name: str, hdr: Header, segs: list[bytes]) -> dict:
    """The segment table row of one module."""
    entry = {"name": name, "build_id": hdr.build_id.rstrip(b"\0").hex()}
    for i, seg in enumerate(SEGMENTS):
        entry[seg] = {"offset": hdr.segs[i][1], "size": hdr.segs[i][2]}
    bss_off = hdr.segs[2][1] + hdr.segs[2][2]
    entry["bss"] = {"offset": bss_off, "size": hdr.bss_size}
    entry["image_size"] = (bss_off + hdr.bss_size + PAGE - 1) & ~(PAGE - 1)
    mod0 = struct.unpack("<I", image_read(hdr, segs, 4, 4))[0]
    words = image_read(hdr, segs, mod0, 28)
    if words[:4] != b"MOD0":
        raise PrepareError(f"module {name}: no MOD0 header at the offset its text gives ({mod0:#x})")
    rel = struct.unpack_from("<6i", words, 4)
    entry["mod0"] = mod0
    entry["dynamic"] = mod0 + rel[0]
    entry["eh_frame_hdr"] = {"offset": mod0 + rel[3], "size": rel[4] - rel[3]}
    return entry


def check_supplied(name: str, given: Path, hdr: Header, raw: bytes) -> list[bytes]:
    """The segments of a module the player gave, checked against the module in the dump."""
    data = given.read_bytes()
    ghdr = Header(data, f"the given module {given}")
    if ghdr.codec == "zbic":
        raise PrepareError(f"the given module {given} is ZBIC-compressed; give a plain (or LZ4) module")
    if ghdr.build_id != hdr.build_id:
        raise PrepareError(f"the given module {given} has build id {ghdr.build_id.rstrip(bytes(1)).hex()}, the dump's "
                           f"{name} has {hdr.build_id.rstrip(bytes(1)).hex()}: it is another version of the game")
    if [s[1:] for s in ghdr.segs] != [s[1:] for s in hdr.segs] or ghdr.bss_size != hdr.bss_size:
        raise PrepareError(f"the given module {given} lays out its segments unlike the dump's {name}")
    segs = read_segments(data, ghdr, f"the given module {given}")
    if hdr.codec != "zbic":
        theirs = read_segments(raw, hdr, f"{DUMP_EXEFS}/{name}")
    for i, seg in enumerate(SEGMENTS):
        if hdr.hashed(i):
            ok = hashlib.sha256(segs[i]).digest() == hdr.hashes[i]
        elif hdr.codec != "zbic":
            ok = segs[i] == theirs[i]
        else:
            raise PrepareError(f"the dump's {name} carries no hash of its {seg} segment, so the given module cannot be "
                               "checked against it")
        if not ok:
            raise PrepareError(f"the {seg} segment of the given module {given} differs from the dump's {name}")
    return segs


class Modules(Step):
    id = "modules"
    needs = ("extract",)

    def __init__(self, decoder_factory=None):
        self.decoder_factory = decoder_factory    # tests replace the fetch and build of the ZBIC decoder

    def key(self, ctx: Context) -> str:
        given = getattr(ctx.options, "plain_modules", None)
        return fingerprint(f"format={FORMAT}", f"zbic={zbic.COMMIT}:{zbic.CONTENTS_SHA256}",
                           HERE / "modules.py", TOOLS / "lift" / "nso.py",
                           "given:", Path(given) if given else "none")

    def current(self, ctx: Context, record: dict) -> bool:
        try:
            for name, size in record["files"].items():
                if (ctx.dest / OUT / name).stat().st_size != size:
                    return False
            return (ctx.dest / OUT / TABLE).is_file()
        except (OSError, KeyError, AttributeError):
            return False

    def verify(self, ctx: Context, record: dict) -> bool:
        try:
            return all(file_sha256(ctx.dest / OUT / n) == s for n, s in record["sha256"].items())
        except (OSError, KeyError, AttributeError):
            return False

    def decoder(self, ctx: Context):
        if self.decoder_factory is not None:
            return self.decoder_factory(ctx)
        archive = getattr(ctx.options, "zbic_archive", None) or bundle_env.zbic_archive()
        return zbic.prepare_decoder(compiler(ctx), ctx.dest / TEMP / "zbic", ctx.reporter,
                                    archive=Path(archive) if archive else None)

    def run(self, ctx: Context) -> dict:
        rep = ctx.reporter
        exefs, out = ctx.dest / DUMP_EXEFS, ctx.dest / OUT
        given_dir = getattr(ctx.options, "plain_modules", None)
        names = [n for n in layout.ORDER if (exefs / n).is_file()]
        if "main" not in names:
            raise PrepareError(f"the extracted ExeFS ({exefs}) has no main module")
        out.mkdir(parents=True, exist_ok=True)
        decode = None
        table, files, sums = [], {}, {}
        for name in names:
            raw = (exefs / name).read_bytes()
            hdr = Header(raw, f"{DUMP_EXEFS}/{name}")
            given = Path(given_dir) / name if given_dir else None
            if given is not None and given.is_file():
                segs = check_supplied(name, given, hdr, raw)
                source = "given"
            else:
                if hdr.codec == "zbic" and decode is None:
                    decode = self.decoder(ctx)
                segs = read_segments(raw, hdr, f"{DUMP_EXEFS}/{name}", decode)
                source = "decoded"
            data = plain_bytes(hdr, segs)
            tmp = out / (name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, out / name)
            sha = hashlib.sha256(data).hexdigest()
            entry = table_entry(name, hdr, segs)
            entry.update({"file": name, "bytes": len(data), "sha256": sha,
                          "source": {"file": f"{DUMP_EXEFS}/{name}", "codec": hdr.codec, "from": source,
                                     "sha256": hashlib.sha256(raw).hexdigest()}})
            table.append(entry)
            files[name], sums[name] = len(data), sha
            rep.say(f"modules: {name}: {hdr.codec} {source}, build id {entry['build_id']}, "
                    f"{len(data):,} bytes, SHA-256 {sha}")
        if (exefs / NPDM).is_file():
            data = (exefs / NPDM).read_bytes()
            tmp = out / (NPDM + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, out / NPDM)
            files[NPDM], sums[NPDM] = len(data), hashlib.sha256(data).hexdigest()
        tmp = out / (TABLE + ".part")
        tmp.write_text(json.dumps({"format": FORMAT, "modules": table}, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, out / TABLE)
        return {"bytes": sum(files.values()), "files": files, "sha256": sums}
