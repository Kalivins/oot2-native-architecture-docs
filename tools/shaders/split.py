# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Split the player's shader archives into translator jobs.

The game ships its shader programs inside several containers: zstd frames that
need a dictionary, SARC archives, BNSH program tables (alone or inside an
effect or a BFSHA file), BAHS arrays, NDSR packs and one NDVE movie file. This
tool walks them, cuts every program into its control section and its code
area, derives the program's key, and writes one translator job per distinct
program (job contract v1, tools/shaders/pool.py is the consumer).

Outputs, all under --out and all derived from the player's own files, so none
of them belongs in the repository:

  programs.bin    each distinct code slice once, 16-byte aligned
  jobs.jsonl      one job object per line, contract fields only
  programs.tsv    one row per job, with where it was first seen
  pairing.tsv     one row per BNSH variation: the keys of its vertex, pixel, task and mesh stages, and
                  whether it sits in a superseded Product.100 file (summary.json pairing.outside_product100
                  holds the counts to set against research 13, which walked the 110 files only)
  summary.json    counts, errors and the cost of the run

Usage:
  py -3.14 -I tools/shaders/split.py scan --romfs <folder> --out <folder> [--bahs walk|scan]
  py -3.14 -I tools/shaders/split.py compare --programs <programs.tsv> --map <map.tsv>

Exit codes: 0 ok, 1 errors found (or a difference, for compare), 2 cannot run.

The key (key16) and the container readers follow the nxnative project's
shader_split.py, so a map made by that project can be compared with ours.
Adapted from nxnative dm/gfx/tools/shader_split.py (MIT): the key definition,
the BNSH stage table walk, the BAHS address scan (--bahs scan), the NDSR record
table and the NDVE raw and table readers. Adapted from nxnative
dm/gfx/tools/dmsh_meshstage.py and dmsh_build.py job_for (MIT): the stage memory
of task and mesh shaders and the slice each stage gets.

Two ways to enumerate a BAHS exist. "scan" is nxnative's: every aligned
16-byte window that looks like a code entry. It also matches a few windows that
are not entries, so a map made that way holds a handful of pairings that no
program uses. "walk" reads the binary array the file declares and yields only
the real records. Walk is the default; scan exists to reproduce the map.

Python 3.14 provides compression.zstd. It is imported only when a zstd frame
is met, so the container readers and their tests work on older interpreters.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Sequence

# --- formats --------------------------------------------------------------------

CTL_MAGIC = b"\x34\x12\x76\x98"          # 0x98761234, little endian
CTL_VERSION = b"\xc9\x00\x00\x00"        # NVN2 control section version
KEY_CTL_BYTES = 0x48                     # the part of the control section the key hashes
REPO_ROOT = Path(__file__).resolve().parents[2]
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
ZSTD_DICT_MAGIC = b"\x37\xa4\x30\xec"
DICT_PACK = "Pack/ZsDic.pack.zs"

# Trees that hold shaders. The scan order is the sorted order of all their paths.
TREES = ("Effect/", "Layout2Shader/", "Lib/", "Movie/", "Shader/")

# The first entry whose text occurs in a label names the group (nxnative's shader_groups).
GROUPS = (("/Shader/", "shader"), ("/Lib/", "lib"), ("/Effect/", "effect"),
          ("/Layout2Shader/", "layout"), ("/Movie/", "movie"))

# The base game's shader files that the update replaced. They stay in the
# player's merged image but the nxnative map was made without them.
PRODUCT_100 = re.compile(r"Shader/[^/]*\.Product\.100\.")

SLOT_STAGE = {0: "vs", 1: "hs", 2: "ds", 3: "gs", 4: "ps", 5: "cs"}   # BNSH stage slots
CTL_STAGE = {5: "cs", 6: "ts", 7: "ms"}                                # control byte +0x56
KIND_STAGE = {0: "vs", 1: "ps", 3: "cs", 4: "ts", 5: "ms"}             # BAHS record kinds
SPH_STAGE = {1: "vs", 2: "hs", 3: "ds", 4: "gs", 5: "ps"}              # SPH type field
JOB_STAGES = ("vs", "ps", "cs", "ts", "ms")

MAX_DEPTH = 8                    # nested SARC levels
MAX_VARIATIONS = 1 << 20         # sanity bound on a BNSH variation count
SPH_BYTES = 0x80                 # the shader program header that graphics stages start with
BLOCK = 32                       # vertices and primitives per attribute block of a mesh shader
INPUT_OFFSET = 0x10              # where a mesh shader finds the payload its task wrote
PIECE = 16 << 20                 # bytes of output taken from the zstd decoder at a time

# Ampere opcodes (low 12 bits of the first instruction word) that touch shared memory:
# LDS, STS, ATOMS and their variants. A task or mesh shader keeps its stage memory
# there, so a program that also uses shared memory itself needs care downstream.
SHARED_MEMORY_OPCODES = frozenset((0x984, 0x388, 0x988, 0x38c, 0x98c, 0xf8c))

_BNSH_MAGIC = re.compile(b"BNSH\0\0\0\0")
_NUL = re.compile(b"\0")
_CTL_SCAN = re.compile(re.escape(CTL_MAGIC))

_PARSE_ERRORS = (struct.error, IndexError, ValueError, OverflowError)


class ContainerError(ValueError):
    """A container is truncated or inconsistent."""


class InputError(Exception):
    """The inputs cannot be used; the tool exits with code 2."""


class JobError(Exception):
    """A program was found but cannot become a job."""


@dataclass
class Problem:
    file: str
    label: str
    reason: str


# --- the file image -------------------------------------------------------------

class Image(Protocol):
    """Where the game's files come from. Paths use '/' and are relative to the RomFS root."""

    def list_paths(self, prefixes: Sequence[str]) -> list[str]:
        """Every file path that starts with one of the prefixes, sorted."""

    def read(self, path: str) -> bytes: ...

    def close(self) -> None: ...


class PackedImage:
    """One packed file (romfs.bin) plus an index.json listing {path, offset, size}.

    Files are read with seek and read, never the whole image.
    """

    def __init__(self, folder: Path):
        index_path = folder / "index.json"
        self.data_path = folder / "romfs.bin"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise InputError(f"cannot read {index_path}: {exc.strerror or exc}") from exc
        except ValueError as exc:
            raise InputError(f"{index_path} is not valid JSON: {exc}") from exc
        files = index.get("files") if isinstance(index, dict) else None
        if not isinstance(files, list):
            raise InputError(f"{index_path}: expected an object with a 'files' list")
        try:
            total = self.data_path.stat().st_size
        except OSError as exc:
            raise InputError(f"cannot read {self.data_path}: {exc.strerror or exc}") from exc
        self.entries: dict[str, tuple[int, int]] = {}
        for n, item in enumerate(files):
            ok = (isinstance(item, dict) and isinstance(item.get("path"), str)
                  and isinstance(item.get("offset"), int) and isinstance(item.get("size"), int))
            if not ok:
                raise InputError(f"{index_path}: files[{n}] needs a string path and integer offset and size")
            offset, size = item["offset"], item["size"]
            if offset < 0 or size < 0 or offset + size > total:
                raise InputError(f"{index_path}: files[{n}] lies outside {self.data_path.name} ({total} bytes)")
            self.entries[item["path"].replace("\\", "/").lstrip("/")] = (offset, size)
        self._fh = None

    def list_paths(self, prefixes: Sequence[str]) -> list[str]:
        return sorted(p for p in self.entries if p.startswith(tuple(prefixes)))

    def read(self, path: str) -> bytes:
        try:
            offset, size = self.entries[path]
        except KeyError:
            raise InputError(f"{path} is not in the index") from None
        if self._fh is None:
            self._fh = open(self.data_path, "rb")
        self._fh.seek(offset)
        data = self._fh.read(size)
        if len(data) != size:
            raise InputError(f"{self.data_path.name} ends inside {path}")
        return data

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class TreeImage:
    """A plain extracted folder tree."""

    def __init__(self, folder: Path):
        self.root = folder.resolve()

    def list_paths(self, prefixes: Sequence[str]) -> list[str]:
        out: list[str] = []
        for prefix in prefixes:
            top = prefix.strip("/").split("/")[0]
            start = self.root / top
            if not start.is_dir():
                if start.is_file() and top == prefix.strip("/"):
                    out.append(top)
                continue
            for base, _dirs, names in os.walk(start):
                for name in names:
                    rel = (Path(base) / name).relative_to(self.root).as_posix()
                    if rel.startswith(prefix):
                        out.append(rel)
        return sorted(set(out))

    def read(self, path: str) -> bytes:
        target = (self.root / path).resolve()
        if self.root != target and self.root not in target.parents:
            raise InputError(f"{path} leaves the folder")
        try:
            return target.read_bytes()
        except OSError as exc:
            raise InputError(f"cannot read {target}: {exc.strerror or exc}") from exc

    def close(self) -> None:
        pass


def open_image(folder: Path) -> Image:
    """A packed image when the folder holds romfs.bin, else a plain tree."""
    if not folder.is_dir():
        raise InputError(f"{folder} is not a folder")
    if (folder / "romfs.bin").exists() or (folder / "index.json").exists():
        return PackedImage(folder)
    if not any((folder / t.strip("/")).exists() for t in TREES):
        raise InputError(f"{folder} holds neither romfs.bin with index.json nor any of {', '.join(TREES)}")
    return TreeImage(folder)


# --- byte helpers ---------------------------------------------------------------

def u32(buf, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def u64(buf, off: int) -> int:
    return struct.unpack_from("<Q", buf, off)[0]


def cut(buf, start: int, length: int, what: str):
    """buf[start:start+length], or a ContainerError when it does not fit."""
    if start < 0 or length < 0 or start + length > len(buf):
        raise ContainerError(f"{what} lies outside its container")
    return buf[start:start + length]


def nvn2_ctl(ctl) -> bool:
    """A control section of the NVN2 flavour: magic and version 0xC9."""
    return len(ctl) >= KEY_CTL_BYTES and bytes(ctl[:4]) == CTL_MAGIC and bytes(ctl[4:8]) == CTL_VERSION


def ctl_key(ctl, code) -> str:
    """nxnative's dm_ctl_sha16: sha256 of the first 0x48 control bytes (the field at +0x30,
    the section's own length in some containers and zero in others, cleared) and the code area."""
    head = bytes(ctl[:0x30]) + b"\0\0\0\0" + bytes(ctl[0x34:KEY_CTL_BYTES])
    digest = hashlib.sha256(head)
    digest.update(code)
    return digest.hexdigest()[:16]


def stage_of(ctl, code) -> str | None:
    """Stage name from the control byte (compute, task, mesh), else from the SPH word at the
    start of the code area (NVN2 code has no other header)."""
    byte = ctl[0x56] if len(ctl) > 0x56 else 0
    if byte in CTL_STAGE:
        return CTL_STAGE[byte]
    if len(code) < 4:
        return None
    word = u32(code, 0)
    kind = (word >> 10) & 0xF
    if kind == 0 and (word & 0x1F) == 1:
        return "vs"
    return SPH_STAGE.get(kind)


# --- zstd and dictionaries ------------------------------------------------------

def frame_dict_id(frame) -> int:
    """The dictionary ID a zstd frame header names, 0 when it names none."""
    if bytes(frame[:4]) != ZSTD_MAGIC or len(frame) < 6:
        raise ContainerError("not a zstd frame")
    fhd = frame[4]
    size = (0, 1, 2, 4)[fhd & 3]
    offset = 5 + (0 if (fhd >> 5) & 1 else 1)      # a window byte precedes it unless single segment
    return int.from_bytes(bytes(frame[offset:offset + size]), "little")


def parse_sarc(buf) -> list[tuple[str, Any]]:
    """Members of a little-endian SARC as (name, bytes view)."""
    if len(buf) < 0x14:
        raise ContainerError("SARC header is truncated")
    header_len, bom, _size, data_off = struct.unpack_from("<HHII", buf, 4)
    if bom != 0xFEFF:
        raise ContainerError("SARC is not little endian")
    if bytes(cut(buf, header_len, 12, "SFAT header")[:4]) != b"SFAT":
        raise ContainerError("SARC has no SFAT table")
    count = struct.unpack_from("<H", buf, header_len + 6)[0]
    table = header_len + 12
    names_at = table + 16 * count
    if bytes(cut(buf, names_at, 8, "SFNT header")[:4]) != b"SFNT":
        raise ContainerError("SARC has no SFNT table")
    names_at += 8
    members = []
    for i in range(count):
        name_hash, attr, start, end = struct.unpack_from("<IIII", buf, table + 16 * i)
        if attr >> 24:
            at = names_at + (attr & 0xFFFFFF) * 4
            stop = _NUL.search(buf, at)
            if stop is None:
                raise ContainerError("SARC member name is not terminated")
            name = bytes(buf[at:stop.start()]).decode("utf-8", "replace")
        else:
            name = f"{name_hash:08x}"
        if end < start:
            raise ContainerError(f"SARC member {name} ends before it starts")
        members.append((name, cut(buf, data_off + start, end - start, f"SARC member {name}")))
    return members


class Decompressor:
    """zstd frames, with the dictionaries of Pack/ZsDic.pack.zs found on first use."""

    def __init__(self, image: Image):
        self.image = image
        self._zstd = None
        self._dict_bytes: dict[int, bytes] | None = None
        self._dicts: dict[int, Any] = {}

    def zstd(self):
        if self._zstd is None:
            try:
                from compression import zstd           # Python 3.14
            except ImportError as exc:
                raise InputError("zstd frames need Python 3.14 (compression.zstd); "
                                 "run with py -3.14") from exc
            self._zstd = zstd
        return self._zstd

    def _load_dictionaries(self) -> dict[int, bytes]:
        if self._dict_bytes is None:
            packed = self.image.read(DICT_PACK)
            raw = self.zstd().decompress(packed)
            found: dict[int, bytes] = {}
            for _name, member in parse_sarc(memoryview(raw)):
                if bytes(member[:4]) == ZSTD_DICT_MAGIC:
                    found[u32(member, 4)] = bytes(member)
            self._dict_bytes = found
        return self._dict_bytes

    def decompress(self, frame: bytes) -> bytes | bytearray:
        """Expand one frame. The result is written piece by piece into a buffer of the
        announced size, so the largest file (about 320 MB) is not held twice."""
        zstd = self.zstd()
        did = frame_dict_id(frame)
        zdict = None
        if did:
            if did not in self._dicts:
                blobs = self._load_dictionaries()
                if did not in blobs:
                    raise ContainerError(f"zstd frame names dictionary {did}, which {DICT_PACK} does not hold")
                self._dicts[did] = zstd.ZstdDict(blobs[did])
            zdict = self._dicts[did]
        size = zstd.get_frame_info(frame).decompressed_size
        if size is None:
            return zstd.decompress(frame, zstd_dict=zdict)
        stream = zstd.ZstdDecompressor(zstd_dict=zdict)
        out = bytearray(size)
        pos = 0
        piece = stream.decompress(frame, max_length=PIECE)
        while True:
            if pos + len(piece) > size:
                raise ContainerError("zstd frame is longer than it announces")
            out[pos:pos + len(piece)] = piece
            pos += len(piece)
            if stream.eof:
                break
            if not piece and stream.needs_input:
                raise ContainerError("zstd frame is truncated")
            piece = stream.decompress(b"", max_length=PIECE)
        if pos != size or stream.unused_data:
            return zstd.decompress(frame, zstd_dict=zdict)      # several frames, or a wrong size field
        return out


# --- programs -------------------------------------------------------------------

@dataclass
class Program:
    """A control section and a code area found in a container."""
    label: str
    stage: str
    container: str                   # BNSH, BAHS, NDSR or NDVE
    ctl: bytes
    code: Any                        # a view of the code area
    variation: str = ""              # the label shared by the stages of one BNSH variation ("" elsewhere)


class Scanner:
    """Finds programs in a blob of bytes. Anything that does not parse becomes a Problem."""

    def __init__(self, problems: list[Problem], bahs: str = "walk"):
        if bahs not in ("walk", "scan"):
            raise ValueError("bahs must be 'walk' or 'scan'")
        self.problems = problems
        self.bahs = bahs
        self.file = ""
        # (variation label, stage) of every BNSH stage record that did not parse, for the pairing file
        self.failed_stages: list[tuple[str, str]] = []

    def problem(self, label: str, reason: str) -> None:
        self.problems.append(Problem(self.file, label, reason))

    def scan(self, buf, label: str, depth: int = 0) -> Iterator[Program]:
        """Dispatch on the magic: SARC recurses, BAHS, NDSR and NDVE have readers, anything
        else is searched for BNSH tables."""
        magic = bytes(buf[:4])
        try:
            if magic == b"SARC":
                if depth >= MAX_DEPTH:
                    raise ContainerError("SARC nesting is too deep")
                for name, member in parse_sarc(buf):
                    yield from self.scan(member, f"{label}/{name}", depth + 1)
            elif magic == b"BAHS":
                yield from self._bahs(buf, label)
            elif magic == b"NDSR":
                yield from self._ndsr(buf, label)
            elif magic == b"NDVE":
                yield from self._ndve(buf, label)
            else:
                yield from self._bnsh_all(buf, label)
        except _PARSE_ERRORS as exc:
            self.problem(label, f"{type(exc).__name__}: {exc}")

    # -- BNSH ----

    def _bnsh_all(self, buf, label: str) -> Iterator[Program]:
        for match in _BNSH_MAGIC.finditer(buf):
            base = match.start()
            if base + 0x64 > len(buf) or bytes(buf[base + 0x60:base + 0x64]) != b"grsc":
                continue
            if base + u32(buf, base + 0x1C) > len(buf):
                self.problem(f"{label}#BNSH@{base:#x}", "BNSH is cut off by the end of its container")
                continue
            try:
                found = list(self._bnsh(buf, base, label))
            except _PARSE_ERRORS as exc:
                self.problem(f"{label}#BNSH@{base:#x}", f"{type(exc).__name__}: {exc}")
                continue
            yield from found

    def _bnsh(self, buf, base: int, label: str) -> Iterator[Program]:
        count, table = struct.unpack_from("<IQ", buf, base + 0x7C)
        if count > MAX_VARIATIONS:
            raise ContainerError(f"BNSH claims {count} variations")
        cut(buf, base + table, 0x40 * count, "variation table")
        for var in range(count):
            program = u64(buf, base + table + 0x40 * var + 0x10)
            if not program:
                continue
            for slot in range(9):
                record = u64(buf, base + program + 8 + 8 * slot)
                if not record:
                    continue
                where = f"{label}#BNSH@{base:#x}#var{var}"
                try:
                    yield self._bnsh_stage(buf, base, record, slot, where)
                except _PARSE_ERRORS as exc:
                    self.problem(where, f"slot {slot}: {type(exc).__name__}: {exc}")
                    stage = self._failed_slot_stage(buf, base, record, slot)
                    if stage is not None:
                        self.failed_stages.append((where, stage))

    @staticmethod
    def _failed_slot_stage(buf, base: int, record: int, slot: int) -> str | None:
        """The stage a record that did not parse stands for: its slot, or for the task and mesh
        slots the control byte when that much can still be read (None when it cannot)."""
        if slot in SLOT_STAGE:
            return SLOT_STAGE[slot]
        try:
            ctl_at, _code_at, _code_len, ctl_len = struct.unpack_from("<QQII", buf, base + record + 8)
            ctl = cut(buf, base + ctl_at, ctl_len, "control section")
            return CTL_STAGE.get(ctl[0x56]) if len(ctl) > 0x56 else None
        except _PARSE_ERRORS:
            return None

    def _bnsh_stage(self, buf, base: int, record: int, slot: int, where: str) -> Program:
        ctl_at, code_at, code_len, ctl_len = struct.unpack_from("<QQII", buf, base + record + 8)
        ctl = bytes(cut(buf, base + ctl_at, ctl_len, "control section"))
        code = cut(buf, base + code_at, code_len, "code area")
        if not nvn2_ctl(ctl):
            raise ContainerError("stage record does not hold an NVN2 control section")
        if u32(ctl, 0x44) != len(code):
            raise ContainerError("code length differs from the control section's")
        if slot in SLOT_STAGE:
            stage = SLOT_STAGE[slot]
            seen = stage_of(ctl, code)
            if seen is not None and seen != stage:
                raise ContainerError(f"program header says {seen}")
        else:
            stage = CTL_STAGE.get(ctl[0x56] if len(ctl) > 0x56 else 0)
            if stage not in ("ts", "ms"):
                raise ContainerError("slot is not a task or mesh stage")
        return Program(where, stage, "BNSH", ctl, code, where)

    # -- BAHS ----

    def _bahs(self, buf, label: str) -> Iterator[Program]:
        if len(buf) < 0x20:
            raise ContainerError("BAHS header is truncated")
        region_len, region = struct.unpack_from("<II", buf, 0x18)
        if region + region_len > len(buf):
            raise ContainerError("BAHS code region lies outside the file")
        entries = self._bahs_scan(buf, region_len, region) if self.bahs == "scan" \
            else self._bahs_walk(buf, region, label)
        for addr, kind, ctl, code in entries:
            where = f"{label}#BAHS@{addr:#x}"
            stage = stage_of(ctl, code)
            if self.bahs == "walk":
                expected = KIND_STAGE.get(kind)
                if expected is None or stage != expected:
                    self.problem(where, f"record kind {kind} does not match stage {stage}")
                    continue
            elif stage is None:
                continue                                   # the address scan drops what has no stage
            yield Program(where, stage, "BAHS", ctl, code)

    @staticmethod
    def _bahs_scan(buf, region_len: int, region: int) -> Iterator[tuple[int, int | None, bytes, Any]]:
        """nxnative's scan: every 4-byte-aligned window {code offset, 0, control size, relative
        control offset} whose control section is NVN2 and whose code fits the region."""
        words = array.array("I")
        if words.itemsize != 4:
            raise ContainerError("this platform has no 4-byte array type")
        words.frombytes(bytes(buf[:region // 4 * 4]))
        if sys.byteorder == "big":
            words.byteswap()
        for i in range(0, min((region - 16 + 3) // 4, len(words) - 3)):
            off = words[i]
            if off & 0xFF or words[i + 1]:
                continue
            size = words[i + 2]
            if not 0x48 <= size < 0x10000:
                continue
            rel = words[i + 3]
            addr = i * 4
            at = addr + (rel - (1 << 32) if rel >= 1 << 31 else rel)
            if at < 0 or at + size > region:
                continue
            ctl = bytes(buf[at:at + size])
            if not nvn2_ctl(ctl):
                continue
            length = u32(ctl, 0x44)
            if off + length > region_len:
                continue
            yield addr, None, ctl, buf[region + off:region + off + length]

    @staticmethod
    def _bahs_walk(buf, region: int, label: str) -> Iterator[tuple[int, int | None, bytes, Any]]:
        """The structured walk of the binary array, which sits at u32[+0x14] + 0x150."""
        version = u32(buf, 4)
        if version != 11:
            raise ContainerError(f"BAHS version {version} is not the one this reader knows (11)")
        region_len = u32(buf, 0x18)
        array_at = u32(buf, 0x14) + 0x150
        size, count = struct.unpack_from("<II", buf, array_at)
        cut(buf, array_at, size, "BAHS binary array")
        at = array_at + 8
        records = []
        for _ in range(count):
            rec_size, kind = struct.unpack_from("<II", buf, at)
            if rec_size < 0x20:
                raise ContainerError("BAHS record is too small")
            records.append((at + 0x10, kind))
            at += rec_size
        if at != array_at + size:
            raise ContainerError("BAHS records do not add up to the array size")
        for addr, kind in records:
            off, zero, size, rel = struct.unpack_from("<IIIi", buf, addr)
            ctl = bytes(cut(buf, addr + rel, size, "control section"))
            if zero or not nvn2_ctl(ctl):
                raise ContainerError(f"BAHS entry at {addr:#x} is not a code entry")
            length = u32(ctl, 0x44)
            if off + length > region_len:
                raise ContainerError(f"BAHS entry at {addr:#x} has code past the region")
            yield addr, kind, ctl, buf[region + off:region + off + length]

    # -- NDSR ----

    def _ndsr(self, buf, label: str) -> Iterator[Program]:
        def points_at_control(c: int) -> bool:
            return c + KEY_CTL_BYTES <= len(buf) and nvn2_ctl(buf[c:c + KEY_CTL_BYTES])

        def table_score(start: int) -> int:
            window = bytes(buf[start:start + 1024])
            window = window[:len(window) // 16 * 16]
            return sum(1 for c, _cl, _k, _kl in struct.iter_unpack("<4I", window) if points_at_control(c))

        at = max((0x10, 0x14), key=table_score)
        while at + 16 <= len(buf):
            c, ctl_word, k, _code_word = struct.unpack_from("<4I", buf, at)
            if not c and not ctl_word:
                break
            at += 16
            if not points_at_control(c):
                continue                                   # a record of some other kind
            where = f"{label}#NDSR@{c:#x}"
            try:
                ctl = bytes(cut(buf, c, max(ctl_word >> 4, KEY_CTL_BYTES), "control section"))
                code = cut(buf, k, u32(ctl, 0x44), "code area")
                stage = stage_of(ctl, code)
            except _PARSE_ERRORS as exc:
                self.problem(where, f"{type(exc).__name__}: {exc}")
                continue
            if stage is None:
                self.problem(where, "no stage")
                continue
            yield Program(where, stage, "NDSR", ctl, code)

    # -- NDVE ----

    def _ndve(self, buf, label: str) -> Iterator[Program]:
        tag = _find(buf, b"NDSR", 0, 0x100)
        if tag >= 0:
            # Programs named by a record table that follows the NDSR tag, offsets from the tag.
            for at in range(tag + 4, min(len(buf) - 16, tag + 0x1000), 4):
                c, cl, k, kl = struct.unpack_from("<4I", buf, at)
                if cl & 0xF not in (0, 3, 6) or kl & 0xFF != 7 or not 0x48 <= cl >> 4 < 0x10000:
                    continue
                c, k, n = tag + c, tag + k, (kl >> 8) * 16
                if c + (cl >> 4) > len(buf) or k + n > len(buf):
                    continue
                if not nvn2_ctl(buf[c:c + KEY_CTL_BYTES]) or u32(buf, c + 0x44) != n:
                    continue
                ctl, code = bytes(buf[c:c + (cl >> 4)]), buf[k:k + n]
                stage = stage_of(ctl, code)
                if stage:
                    yield Program(f"{label}#NDVE#NDSR@{c:#x}", stage, "NDVE", ctl, code)
        # Programs stored raw: a control section whose length is at +0x30, code right after it.
        for match in _CTL_SCAN.finditer(buf):
            c = match.start()
            if c + KEY_CTL_BYTES > len(buf) or not nvn2_ctl(buf[c:c + KEY_CTL_BYTES]) or u32(buf, c + 8) != 0x14:
                continue
            rel, n = u64(buf, c + 0x30), u32(buf, c + 0x44)
            if not rel or not n or rel < 0x68 or c + rel + n > len(buf):
                continue
            ctl, code = bytes(buf[c:c + rel]), buf[c + rel:c + rel + n]
            stage = stage_of(ctl, code)
            if stage:
                yield Program(f"{label}#NDVE@{c:#x}", stage, "NDVE", ctl, code)


def _find(buf, needle: bytes, start: int, end: int) -> int:
    found = re.compile(re.escape(needle)).search(buf, start, end)
    return found.start() if found else -1


# --- jobs -----------------------------------------------------------------------

@dataclass
class JobSpec:
    """The translator-facing facts of one program."""
    stage: str
    start: int                              # slice of the code area the job carries
    length: int
    c1: str
    nvn2_cs: list[int] | None = None
    local: list[int] | None = None
    shared: int | None = None
    stage_memory: list[int] | None = None
    mesh: dict[str, Any] | None = None
    # The components a task or mesh program writes: {"vertex": [...], "primitive": [...]}, each entry
    # [location, component] with location -1 for the position. Not part of variant_hash (see there).
    outputs: dict[str, list[list[int]]] | None = None

    def variant_hash(self) -> str:
        """First 8 hex digits of sha256 over the canonical JSON of the fields that separate
        control variants of one key16. The mesh object is included although the translator
        does not read it, so that task and mesh variants stay apart for the mesh emission.
        The output map is left out on purpose: it is read from the program headers, which lie
        inside the code area that key16 hashes, so it cannot tell two variants of one key16
        apart, and leaving it out keeps every job key as it was (nvn_program.c builds the same
        JSON for the front end)."""
        fields: dict[str, Any] = {"stage": self.stage}
        for name in ("nvn2_cs", "local", "shared", "stage_memory", "mesh"):
            value = getattr(self, name)
            if value is not None:
                fields[name] = value
        text = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("ascii")).hexdigest()[:8]


def sph_outputs(sph) -> list[tuple[int | None, int]]:
    """Components a shader program header's output map enables, in attribute-area order:
    the position first (byte 0x35, bits 4..7), then generic attributes 0..31 (a nibble each
    from byte 0x36). Adapted from nxnative dmsh_meshstage.py sph_outputs (MIT)."""
    out: list[tuple[int | None, int]] = [(None, c) for c in range(4) if sph[0x35] >> (4 + c) & 1]
    for n in range(32):
        out += [(n, c) for c in range(4) if sph[0x36 + n // 2] >> (4 * (n % 2) + c) & 1]
    return out


def output_map(sph) -> list[list[int]]:
    """sph_outputs as JSON lists: [location, component], location -1 for the position."""
    return [[-1 if n is None else n, c] for n, c in sph_outputs(sph)]


def own_shared_memory(code) -> bool:
    """Whether instructions use shared memory of their own (adapted from nxnative
    dmsh_meshstage.py own_shared, MIT)."""
    return any(struct.unpack_from("<H", code, o)[0] & 0xFFF in SHARED_MEMORY_OPCODES
               for o in range(0, len(code) - 15, 16))


def stage_memory(kind: str, payload: int, max_v: int, max_p: int, nv: int, np_: int) -> tuple[list[int], int]:
    """Stage memory layout of a task or mesh shader, which the translator keeps in shared
    memory: [workgroup number, input, outputs, primitive bytes] and the bytes in all.
    Adapted from nxnative dmsh_meshstage.py memory (MIT)."""
    out = INPUT_OFFSET + ((payload + 15) & ~15)
    if kind == "ts":
        return [0, INPUT_OFFSET, out, out + payload], out + ((payload + 3) & ~3)
    blocks = (max(max_v, max_p, 1) + BLOCK - 1) // BLOCK
    prim = out + blocks * max(nv + np_, 1) * BLOCK * 4
    return [0, INPUT_OFFSET, out, prim], prim + ((4 + 3 * max(max_p, 1) + 3) & ~3)


def build_job(stage: str, ctl: bytes, code) -> JobSpec:
    """Cut the slice and derive the fields the contract asks for. Adapted from nxnative
    dmsh_build.py job_for (MIT)."""
    if stage not in JOB_STAGES:
        raise JobError(f"stage {stage} has no job form")
    inst = u32(ctl, 0x38)
    if inst <= 0 or inst % 16 or inst > len(code):
        raise JobError(f"instruction length {inst} is not a positive multiple of 16 inside the code area")
    c1_at, c1_end = struct.unpack_from("<II", ctl, 0x40)
    c1 = bytes(code[c1_at:c1_end]).hex() if c1_at <= c1_end <= len(code) else ""
    spec = JobSpec(stage, 0, inst, c1)
    if stage == "cs":
        if len(ctl) < 0xBC:
            raise JobError("control section is too short for a compute shader")
        spec.nvn2_cs = [ctl[0xB1], ctl[0xB2], ctl[0xB3], struct.unpack_from("<H", ctl, 0xBA)[0],
                        ctl[0xA9] * 4, ctl[0xB4], ctl[0xB5]]
        local = list(struct.unpack_from("<3I", ctl, 0x58))
        if min(local) < 1 or local[0] * local[1] * local[2] > 1024:
            raise JobError(f"thread group {local} is outside 1..1024 threads")
        spec.local = local
        spec.shared = (u32(ctl, 0x64) + 15) // 16 * 16
    elif stage in ("ts", "ms"):
        if len(ctl) < 0x6C:
            raise JobError("control section is too short for a task or mesh shader")
        if inst <= SPH_BYTES:
            raise JobError("no instructions after the program header")
        threads, payload = struct.unpack_from("<2H", ctl, 0x58)
        if threads < 1:
            raise JobError("workgroup has no threads")
        spec.start, spec.length = SPH_BYTES, inst - SPH_BYTES
        mesh: dict[str, Any] = {"kind": stage, "threads": threads, "payload": payload}
        # The task map is whatever its own header says: nothing in the dump shows yet that a task
        # program's output bits mean anything (research 13 finding 10 calls its header a type-1
        # vertex header), so it may come out empty; ticket 71 checks it on the real scan.
        outputs: dict[str, list[list[int]]] = {"vertex": output_map(code[:SPH_BYTES]), "primitive": []}
        max_v = max_p = nv = np_ = 0
        if stage == "ms":
            max_v, max_p = struct.unpack_from("<2H", ctl, 0x60)
            if len(code) < SPH_BYTES:
                raise JobError("code area is shorter than a program header")
            vertex = sph_outputs(code[:SPH_BYTES])
            primitive: list[tuple[int | None, int]] = []
            second = u32(ctl, 0x64) if ctl[0x68] & 0x80 else 0
            if second:
                if second + SPH_BYTES > len(code):
                    raise JobError("per-primitive program header lies outside the code area")
                primitive = [c for c in sph_outputs(code[second:second + SPH_BYTES]) if c[0] is not None]
            nv, np_ = len(vertex), len(primitive)
            outputs["primitive"] = [[n, c] for n, c in primitive]
            mesh.update(has_task=bool(ctl[0x5E]), max_vertices=max_v, max_primitives=max_p,
                        vertex_components=nv, primitive_components=np_)
        mesh["own_shared"] = own_shared_memory(code[SPH_BYTES:inst])
        layout, size = stage_memory(stage, payload, max_v, max_p, nv, np_)
        spec.stage_memory, spec.shared, spec.local, spec.mesh = layout, size, [threads, 1, 1], mesh
        spec.outputs = outputs
    return spec


# --- collecting -----------------------------------------------------------------

def group_of(label: str) -> str:
    for text, name in GROUPS:
        if text in label:
            return name
    return "other"


@dataclass
class JobEntry:
    key: str
    key16: str
    stage: str
    container: str
    label: str
    from_100: bool            # the recorded label comes from a Product.100 file
    records: int = 1
    only_100: bool = True     # every occurrence is in a Product.100 file
    # Digests of the control bytes the job does not carry (see ctl_tail_digest), one per
    # variant merged into this job. More than one means the first occurrence's bytes won.
    tails: set[str] = field(default_factory=set)
    outputs: dict[str, list[list[int]]] | None = None     # the output map of the first occurrence


# Control-section words that echo the section's own length, which differs by container.
CTL_ECHOES = (0x98, 0xBC, 0xC8)
CTL_TAIL_END = 0x100


def ctl_tail_digest(ctl: bytes) -> str:
    """sha256 of control bytes +0x48..+0x100 with the length echoes cleared: the part of a
    control section that neither the key nor the job fields cover (register count, flag
    words). Counted per job so a merge of differing sections is visible, never silent."""
    tail = bytearray(ctl[KEY_CTL_BYTES:CTL_TAIL_END].ljust(CTL_TAIL_END - KEY_CTL_BYTES, b"\0"))
    for at in CTL_ECHOES:
        tail[at - KEY_CTL_BYTES:at - KEY_CTL_BYTES + 4] = bytes(4)
    return hashlib.sha256(tail).hexdigest()


@dataclass
class Collector:
    """Turns programs into distinct jobs and streams the slices and job lines to disk."""
    out_dir: Path
    problems: list[Problem]
    jobs: dict[str, JobEntry] = field(default_factory=dict)
    records: int = 0
    by_stage_records: dict[str, int] = field(default_factory=dict)
    # variation label -> stage -> job key ("?" when the stage's job could not be built)
    variations: dict[str, dict[str, str]] = field(default_factory=dict)
    # labels of the variations that sit in a superseded Product.100 file (research 13 walked the 110 files only)
    variations_100: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bin_path = self.out_dir / "programs.bin"
        self._bin = open(self.bin_path, "wb")
        self._jsonl = open(self.out_dir / "jobs.jsonl", "w", encoding="utf-8", newline="\n")
        self.bin_size = 0
        self._slices: dict[tuple[str, int], tuple[int, int]] = {}
        self._code_path = str(self.bin_path.resolve())

    def close(self) -> None:
        self._bin.close()
        self._jsonl.close()

    def note_variation(self, variation: str, stage: str, key: str, path: str) -> None:
        """Records the job key of one stage of a BNSH variation ("?" when it could not be built)."""
        if not variation:
            return
        self.variations.setdefault(variation, {})[stage] = key
        if PRODUCT_100.match(path):
            self.variations_100.add(variation)

    def add(self, prog: Program, path: str) -> None:
        self.records += 1
        self.by_stage_records[prog.stage] = self.by_stage_records.get(prog.stage, 0) + 1
        try:
            spec = build_job(prog.stage, prog.ctl, prog.code)
        except (JobError, *_PARSE_ERRORS) as exc:
            self.problems.append(Problem(path, prog.label, f"job: {exc}"))
            self.note_variation(prog.variation, prog.stage, "?", path)
            return
        key16 = ctl_key(prog.ctl, prog.code)
        key = f"{key16}.{spec.variant_hash()}"
        self.note_variation(prog.variation, prog.stage, key, path)
        is_100 = PRODUCT_100.match(path) is not None
        entry = self.jobs.get(key)
        if entry is None:
            self.jobs[key] = JobEntry(key, key16, prog.stage, prog.container, prog.label, is_100,
                                      only_100=is_100, tails={ctl_tail_digest(prog.ctl)}, outputs=spec.outputs)
            self._emit(key, key16, spec, prog.code)
            return
        entry.records += 1
        if spec.outputs != entry.outputs:
            # the output map is not in the key: two variants of one key16 whose per-primitive
            # header offsets differ would be merged with the first one's map
            self.problems.append(Problem(path, prog.label, "job: output map differs from the first variant's"))
        entry.tails.add(ctl_tail_digest(prog.ctl))
        if not is_100:
            entry.only_100 = False
            if entry.from_100:                 # the first occurrence outside the superseded files
                entry.stage, entry.container, entry.label, entry.from_100 = \
                    prog.stage, prog.container, prog.label, False

    def _emit(self, key: str, key16: str, spec: JobSpec, code) -> None:
        slot = (key16, spec.start)
        if slot not in self._slices:
            self._bin.write(b"\0" * (-self.bin_size % 16))
            self.bin_size += -self.bin_size % 16
            self._bin.write(code[spec.start:spec.start + spec.length])
            self._slices[slot] = (self.bin_size, spec.length)
            self.bin_size += spec.length
        offset, length = self._slices[slot]
        job: dict[str, Any] = {"key": key, "stage": spec.stage, "arch": "sm86", "code": self._code_path,
                               "code_offset": offset, "code_length": length, "c1": spec.c1}
        for name in ("nvn2_cs", "local", "shared", "stage_memory"):
            value = getattr(spec, name)
            if value is not None:
                job[name] = value
        if spec.mesh is not None:
            job["mesh"] = spec.mesh
        if spec.outputs is not None:
            job["outputs"] = spec.outputs
        self._jsonl.write(json.dumps(job, separators=(",", ":")) + "\n")


TSV_COLUMNS = ("job_key", "key16", "stage", "container", "group", "first_label", "records",
               "only_product100", "variants", "ctl_variants")


def write_programs_tsv(path: Path, jobs: dict[str, JobEntry]) -> None:
    variants: dict[str, int] = {}
    for entry in jobs.values():
        variants[entry.key16] = variants.get(entry.key16, 0) + 1
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(TSV_COLUMNS) + "\n")
        for entry in jobs.values():
            fh.write("\t".join((entry.key, entry.key16, entry.stage, entry.container, group_of(entry.label),
                                entry.label, str(entry.records), str(int(entry.only_100)),
                                str(variants[entry.key16]), str(len(entry.tails)))) + "\n")


PAIRING_COLUMNS = ("variation", "vertex", "pixel", "task", "mesh", "product100")
PAIRING_STAGES = ("vs", "ps", "ts", "ms")


def pairing_counts(rows: list[list[str]]) -> dict[str, int]:
    """The counts research 13 made by hand, from rows of [variation, vertex, pixel, task, mesh]:
    variations, variations with a mesh stage, distinct (task, mesh) pairs and distinct
    (task, mesh, pixel) triples (only variations whose keys are all built)."""
    pairs: set[tuple[str, str]] = set()
    triples: set[tuple[str, str, str]] = set()
    with_mesh = 0
    for _label, _vs, ps, ts, ms in rows:
        with_mesh += bool(ms)
        if ts and ms and "?" not in (ts, ms):
            pairs.add((ts, ms))
            if ps and ps != "?":
                triples.add((ts, ms, ps))
    return {"variations": len(rows), "variations_with_mesh": with_mesh, "task_mesh_pairs": len(pairs),
            "task_mesh_pixel_triples": len(triples)}


def write_pairing_tsv(path: Path, variations: dict[str, dict[str, str]],
                      variations_100: frozenset[str] | set[str] = frozenset()) -> dict[str, Any]:
    """One row per BNSH variation that holds a vertex, pixel, task or mesh stage: the job key of
    each ("" when the variation has none, "?" when its job could not be built), then 1 when the
    variation sits in a superseded Product.100 file. Returns the counts over all rows plus the
    same counts over the rows outside Product.100 (under "outside_product100"), which are the
    ones to set against research 13: it walked the 110 files only."""
    rows: list[list[str]] = []
    flags: list[bool] = []
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(PAIRING_COLUMNS) + "\n")
        for label, stages in variations.items():
            keys = [stages.get(stage, "") for stage in PAIRING_STAGES]
            if not any(keys):
                continue
            old = label in variations_100
            fh.write("\t".join((label, *keys, str(int(old)))) + "\n")
            rows.append([label, *keys])
            flags.append(old)
    counts: dict[str, Any] = pairing_counts(rows)
    counts["outside_product100"] = pairing_counts([r for r, old in zip(rows, flags) if not old])
    return counts


def peak_working_set() -> int:
    """Peak resident memory of this process in bytes (0 when unknown)."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return 0
    try:
        import resource
    except ImportError:
        return 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def count_by(items: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items()))


def scan_image(image: Image, out_dir: Path, bahs: str = "walk",
               progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Scan every shader file of the image and write the outputs. Returns the summary."""
    started = time.perf_counter()
    problems: list[Problem] = []
    scanner = Scanner(problems, bahs)
    decompressor = Decompressor(image)
    collector = Collector(out_dir, problems)
    paths = image.list_paths(TREES)
    files_with_programs = 0
    try:
        for n, path in enumerate(paths, 1):
            scanner.file = path
            before = collector.records
            scanner.failed_stages.clear()
            try:
                data = image.read(path)
                if data[:4] == ZSTD_MAGIC:
                    data = decompressor.decompress(data)
                view = memoryview(data)
                prog = None
                for prog in scanner.scan(view, "/" + path):
                    collector.add(prog, path)
                for variation, stage in scanner.failed_stages:
                    collector.note_variation(variation, stage, "?", path)
                scanner.failed_stages.clear()
                prog = view = None             # views pin the file's bytes until the next file is read
            except InputError:
                raise
            except (ContainerError, *_PARSE_ERRORS) as exc:
                problems.append(Problem(path, "/" + path, f"{type(exc).__name__}: {exc}"))
            except Exception as exc:                       # a zstd failure or similar: keep going
                problems.append(Problem(path, "/" + path, f"{type(exc).__name__}: {exc}"))
            data = None
            files_with_programs += collector.records > before
            if progress and n % 100 == 0:
                progress(f"{n}/{len(paths)} files, {collector.records} records, {len(collector.jobs)} jobs")
    finally:
        collector.close()
    write_programs_tsv(out_dir / "programs.tsv", collector.jobs)
    pairing = write_pairing_tsv(out_dir / "pairing.tsv", collector.variations, collector.variations_100)

    jobs = list(collector.jobs.values())
    outside_100 = {}
    for entry in jobs:
        if not entry.only_100:
            outside_100.setdefault(entry.key16, entry.stage)
    summary: dict[str, Any] = {
        "bahs_mode": bahs,
        "files_scanned": len(paths),
        "files_with_programs": files_with_programs,
        "records": collector.records,
        "records_by_stage": dict(sorted(collector.by_stage_records.items())),
        "distinct_key16": len({e.key16 for e in jobs}),
        "jobs": len(jobs),
        "key16_with_several_jobs": sum(1 for n in count_by([e.key16 for e in jobs]).values() if n > 1),
        "jobs_with_merged_ctl_variants": sum(1 for e in jobs if len(e.tails) > 1),
        "only_product100_jobs": sum(1 for e in jobs if e.only_100),
        "pairing": pairing,
        "jobs_by_stage": count_by([e.stage for e in jobs]),
        "jobs_by_container": count_by([e.container for e in jobs]),
        "jobs_by_group": count_by([group_of(e.label) for e in jobs]),
        "key16_outside_product100": len(outside_100),
        "key16_outside_product100_by_stage": count_by(list(outside_100.values())),
        "programs_bin_bytes": collector.bin_size,
        "error_count": len(problems),
        "errors": [{"file": p.file, "label": p.label, "reason": p.reason} for p in problems[:1000]],
        "run": {"seconds": round(time.perf_counter() - started, 1), "peak_working_set_bytes": peak_working_set()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=False) + "\n",
                                          encoding="utf-8", newline="\n")
    return summary


# --- compare --------------------------------------------------------------------

MAP_STAGE = {"vertex": "vs", "pixel": "ps", "compute": "cs", "mesh": "ms", "task": "ts"}


def read_tsv_rows(path: Path, columns: int) -> list[list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc.strerror or exc}") from exc
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if len(cells) < columns:
            raise InputError(f"{path}:{number}: expected {columns} tab-separated columns")
        rows.append(cells)
    return rows


def compare_files(programs: Path, map_path: Path) -> dict[str, int]:
    """Compare our non-superseded keys with a map. Counts only."""
    ours: dict[str, tuple[str, str]] = {}
    rows = read_tsv_rows(programs, len(TSV_COLUMNS))
    if not rows or tuple(rows[0][:len(TSV_COLUMNS)]) != TSV_COLUMNS:
        raise InputError(f"{programs} does not start with the programs.tsv header")
    for cells in rows[1:]:
        if cells[7] == "0":
            ours.setdefault(cells[1], (cells[2], cells[5]))
    theirs: dict[str, tuple[str, str]] = {}
    rows = read_tsv_rows(map_path, 5)
    if rows and rows[0][0] == "package":
        rows = rows[1:]
    for cells in rows:
        theirs.setdefault(cells[3], (MAP_STAGE.get(cells[2], cells[2]), cells[4]))
    both = [k for k in theirs if k in ours]
    return {
        "map keys": len(theirs),
        "our keys": len(ours),
        "both": len(both),
        "only map": len(theirs) - len(both),
        "only ours": len(ours) - len(both),
        "stage disagreements": sum(1 for k in both if theirs[k][0] != ours[k][0]),
        "source disagreements": sum(1 for k in both if theirs[k][1] != ours[k][1]),
    }


# --- command line ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="split the shader archives into jobs")
    scan.add_argument("--romfs", required=True, type=Path,
                      help="folder with romfs.bin and index.json, or a plain extracted tree")
    scan.add_argument("--out", required=True, type=Path)
    scan.add_argument("--bahs", choices=("walk", "scan"), default="walk",
                      help="walk the declared record array (default) or reproduce nxnative's address scan")
    cmp_ = sub.add_parser("compare", help="compare programs.tsv with a map")
    cmp_.add_argument("--programs", required=True, type=Path)
    cmp_.add_argument("--map", required=True, type=Path, dest="map_path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "compare":
            counts = compare_files(args.programs, args.map_path)
            for name, value in counts.items():
                print(f"{name}: {value}")
            return 1 if any(counts[k] for k in ("only map", "only ours", "stage disagreements",
                                                "source disagreements")) else 0
        out = args.out.resolve()
        if out == REPO_ROOT or REPO_ROOT in out.parents:
            raise InputError(f"--out {out} lies inside the repository; the outputs are derived from the game")
        image = open_image(args.romfs)
        try:
            summary = scan_image(image, out, args.bahs, lambda m: print(m, file=sys.stderr, flush=True))
        finally:
            image.close()
    except InputError as exc:
        print(f"split: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"split: {exc.strerror or exc}: {exc.filename}", file=sys.stderr)
        return 2
    for name in ("files_scanned", "records", "distinct_key16", "jobs", "key16_with_several_jobs",
                 "jobs_with_merged_ctl_variants", "only_product100_jobs", "key16_outside_product100", "error_count"):
        print(f"{name}: {summary[name]}")
    for name in ("jobs_by_stage", "jobs_by_container", "jobs_by_group", "key16_outside_product100_by_stage"):
        print(f"{name}: {json.dumps(summary[name])}")
    print(f"run: {json.dumps(summary['run'])}")
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
