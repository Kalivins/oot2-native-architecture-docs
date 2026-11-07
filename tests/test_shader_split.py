# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/shaders/split.py.

Every container here is assembled inside the test from synthetic bytes: no game
data, no archive and no key. The zstd tests need Python 3.14 (compression.zstd)
and are skipped, with that reason, on older interpreters.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "tools" / "shaders" / "split.py"
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import split  # noqa: E402

try:
    from compression import zstd  # type: ignore[import-not-found]
except ImportError:
    zstd = None

NEEDS_ZSTD = unittest.skipUnless(zstd is not None, "compression.zstd needs Python 3.14")

CTL_MAGIC = bytes.fromhex("34127698")
VS_WORD = (1 << 10) | 1      # SPH word 0 of a vertex shader: type 1
PS_WORD = (5 << 10) | 2      # SPH word 0 of a pixel shader: type 5


# --- builders -------------------------------------------------------------------

def pad_to(data: bytes | bytearray, align: int) -> bytearray:
    out = bytearray(data)
    out.extend(b"\0" * (-len(out) % align))
    return out


def make_code(first_word: int, length: int, seed: int) -> bytes:
    """A code area: zero bytes, a tag in the second half of every 16-byte slot so that slices
    can be told apart, and the first word set."""
    code = bytearray(length)
    for o in range(0, length, 16):
        struct.pack_into("<Q", code, o + 8, (seed << 32) | o)
    struct.pack_into("<I", code, 0, first_word)
    return bytes(code)


def make_ctl(*, stage: int = 0, inst: int, code_len: int, c1_at: int, own: int = 0xE0,
             size: int = 0xE0, section8: int = 0x14, patch: tuple = ()) -> bytes:
    """An NVN2 control section with the fields the splitter reads."""
    ctl = bytearray(size)
    ctl[0:4] = CTL_MAGIC
    struct.pack_into("<I", ctl, 4, 0xC9)
    struct.pack_into("<I", ctl, 8, section8)
    struct.pack_into("<I", ctl, 0x30, own)
    struct.pack_into("<I", ctl, 0x38, inst)
    struct.pack_into("<I", ctl, 0x40, c1_at)
    struct.pack_into("<I", ctl, 0x44, code_len)
    ctl[0x56] = stage
    for offset, fmt, value in patch:
        struct.pack_into(fmt, ctl, offset, value)
    return bytes(ctl)


def make_vs(seed: int = 1, **kw) -> tuple[bytes, bytes]:
    """Vertex shader: SPH and two instructions (0x90 bytes), a c1 segment 0x100..0x200."""
    return make_ctl(inst=0x90, code_len=0x200, c1_at=0x100, **kw), make_code(VS_WORD, 0x200, seed)


def make_ps(seed: int = 2, **kw) -> tuple[bytes, bytes]:
    return make_ctl(inst=0xA0, code_len=0x200, c1_at=0x100, **kw), make_code(PS_WORD, 0x200, seed)


COMPUTE_PATCH = ((0xB1, "<B", 3), (0xB2, "<B", 4), (0xB3, "<B", 5), (0xBA, "<H", 0x1234), (0xA9, "<B", 6),
                 (0xB4, "<B", 7), (0xB5, "<B", 8), (0x58, "<I", 8), (0x5C, "<I", 4), (0x60, "<I", 2),
                 (0x64, "<I", 21))


def make_cs(seed: int = 3, patch: tuple = COMPUTE_PATCH, **kw) -> tuple[bytes, bytes]:
    """Compute shader: no SPH, 0x60 bytes of instructions, no c1 segment."""
    return make_ctl(stage=5, inst=0x60, code_len=0x100, c1_at=0x100, patch=patch, **kw), \
        make_code(0, 0x100, seed)


def make_ts(seed: int = 4, payload: int = 20, **kw) -> tuple[bytes, bytes]:
    """Task shader: SPH, two instructions, 32 threads."""
    patch = ((0x58, "<H", 32), (0x5A, "<H", payload))
    return make_ctl(stage=6, inst=0xA0, code_len=0x100, c1_at=0x100, patch=patch, **kw), \
        make_code(VS_WORD, 0x100, seed)


def make_ms(seed: int = 5, **kw) -> tuple[bytes, bytes]:
    """Mesh shader with a second, per-primitive SPH at 0x200: 4 position components and
    2 generic ones in the first, 1 generic one in the second."""
    patch = ((0x58, "<H", 32), (0x5A, "<H", 8), (0x5E, "<B", 1), (0x60, "<H", 4), (0x62, "<H", 2),
             (0x64, "<I", 0x200), (0x68, "<B", 0x80))
    code = bytearray(make_code(VS_WORD, 0x280, seed))
    code[4:0x80] = bytes(0x7C)       # the output maps must hold only what is set below
    code[0x200:0x280] = bytes(0x80)
    code[0x35] = 0xF0                 # position x y z w
    code[0x36] = 0x03                 # attribute 0: two components
    code[0x200 + 0x36] = 0x10        # second SPH, attribute 1: one component
    code[0x200 + 0x35] = 0xF0        # its position is not counted
    return make_ctl(stage=7, inst=0xA0, code_len=0x280, c1_at=0x100, patch=patch, **kw), bytes(code)


def written(sph: bytes) -> list[list[int]]:
    """The components a program header enables, read again from the bits (position: byte 0x35 high nibble;
    attribute n: nibble n of the 16 bytes from 0x36), as [location, component] with -1 for the position."""
    out = [[-1, c] for c in range(4) if sph[0x35] & (0x10 << c)]
    for n in range(32):
        nibble = (sph[0x36 + n // 2] >> (4 * (n % 2))) & 15
        out += [[n, c] for c in range(4) if nibble & (1 << c)]
    return out


def make_bnsh(variations: list[dict[int, tuple[bytes, bytes]]]) -> bytes:
    """A BNSH table: variations[v][slot] = (control, code). Offsets are from the start."""
    count = len(variations)
    out = bytearray(0xC0 + 0x40 * count)
    out[0:8] = b"BNSH\0\0\0\0"
    out[0x60:0x64] = b"grsc"
    struct.pack_into("<I", out, 0x7C, count)
    struct.pack_into("<Q", out, 0x80, 0xC0)
    for v, stages in enumerate(variations):
        out = pad_to(out, 0x10)
        program = len(out)
        struct.pack_into("<Q", out, 0xC0 + 0x40 * v + 0x10, program)
        out.extend(b"\0" * 0x50)
        for slot, (ctl, code) in sorted(stages.items()):
            out = pad_to(out, 0x10)
            record = len(out)
            struct.pack_into("<Q", out, program + 8 + 8 * slot, record)
            out.extend(b"\0" * 0x20)
            out = pad_to(out, 0x10)
            ctl_at = len(out)
            out.extend(ctl)
            out = pad_to(out, 0x100)
            code_at = len(out)
            out.extend(code)
            struct.pack_into("<QQII", out, record + 8, ctl_at, code_at, len(code), len(ctl))
    struct.pack_into("<I", out, 0x1C, len(out))
    return bytes(out)


def in_container(blob: bytes, at: int = 0x1000) -> bytes:
    """A blob with unrelated bytes in front, so the BNSH sits at a nonzero offset."""
    return b"\x11" * at + blob


def make_sarc(members: list[tuple[str, bytes]]) -> bytes:
    count = len(members)
    names = bytearray()
    entries = bytearray()
    data = bytearray()
    for i, (name, blob) in enumerate(members):
        start = len(data)
        data.extend(blob)
        data.extend(b"\0" * (-len(data) % 16))
        entries.extend(struct.pack("<IIII", 0x1000 + i, (1 << 24) | (len(names) // 4), start, start + len(blob)))
        names.extend(name.encode() + b"\0")
        names.extend(b"\0" * (-len(names) % 4))
    head = 0x14 + 12 + len(entries) + 8 + len(names)
    data_at = head + (-head % 32)
    out = bytearray(b"SARC")
    out += struct.pack("<HHII", 0x14, 0xFEFF, data_at + len(data), data_at) + b"\0\0\0\0"
    out += b"SFAT" + struct.pack("<HHI", 0xC, count, 0x65) + entries
    out += b"SFNT" + struct.pack("<HH", 8, 0) + names
    out = pad_to(out, 32)
    assert len(out) == data_at
    return bytes(out + data)


def make_bahs(records: list[tuple[int, bytes, bytes, int | None]], version: int = 11) -> bytes:
    """A BAHS with its binary array at u32[+0x14] + 0x150. records = (kind, control, code, share)
    where share is the index of an earlier record whose control section this one points at."""
    x = 0x10
    array_at = x + 0x150
    sizes = [0x78 if share is not None else 0x78 + len(ctl) + (-len(ctl) % 8) for _k, ctl, _c, share in records]
    region_at = array_at + 8 + sum(sizes)
    region_at += -region_at % 0x100
    region = bytearray()
    code_offsets = []
    for _k, _ctl, code, _s in records:
        region = pad_to(region, 0x100)
        code_offsets.append(len(region))
        region.extend(code)
    out = bytearray(region_at)
    out[0:4] = b"BAHS"
    struct.pack_into("<IIIIII", out, 4, version, region_at + len(region), 0, 0, x, len(region))
    struct.pack_into("<I", out, 0x1C, region_at)
    struct.pack_into("<II", out, array_at, 8 + sum(sizes), len(records))
    at = array_at + 8
    ctl_at = {}
    for i, ((kind, ctl, _code, share), size) in enumerate(zip(records, sizes)):
        struct.pack_into("<II", out, at, size, kind)
        entry = at + 0x10
        if share is None:
            ctl_at[i] = at + 0x78
            out[at + 0x78:at + 0x78 + len(ctl)] = ctl
            struct.pack_into("<IIIi", out, entry, code_offsets[i], 0, len(ctl), 0x78 - 0x10)
        else:
            ctl_at[i] = ctl_at[share]
            struct.pack_into("<IIIi", out, entry, code_offsets[i], 0, len(ctl), ctl_at[share] - entry)
        at += size
    return bytes(out + region)


def make_ndsr(records: list[tuple[bytes, bytes, int, int]]) -> bytes:
    """An NDSR pack with 16-byte records from +0x14: (control, code, control type, code type)."""
    table = 0x14
    data_at = table + 16 * (len(records) + 1)
    out = bytearray(data_at)
    out[0:4] = b"NDSR"
    struct.pack_into("<II", out, 4, 0x10100, 0x20302)
    for i, (ctl, code, t_ctl, t_code) in enumerate(records):
        out = pad_to(out, 16)
        ctl_at = len(out)
        out.extend(ctl)
        out = pad_to(out, 16)
        code_at = len(out)
        out.extend(code)
        struct.pack_into("<IIII", out, table + 16 * i, ctl_at, (len(ctl) << 4) | t_ctl, code_at,
                         ((len(code) // 16) << 8) | t_code)
    return bytes(out)


def make_ndve() -> tuple[bytes, dict[str, tuple[bytes, bytes]]]:
    """An NDVE file with one raw program and one program named by a record table."""
    out = bytearray(0x600)
    out[0:4] = b"NDVE"
    tag = 0x28
    out[tag:tag + 4] = b"NDSR"
    code_raw = make_code(0, 0x100, 7)
    ctl_raw = make_ctl(stage=5, inst=0x60, code_len=len(code_raw), c1_at=0x100, own=0,
                       patch=((0x30, "<Q", 0xE0), (0x58, "<I", 16), (0x5C, "<I", 8), (0x60, "<I", 1)))
    ctl_tab = make_ctl(stage=5, inst=0x40, code_len=0x80, c1_at=0x80, own=0, section8=0x11,
                       patch=((0x58, "<I", 32), (0x5C, "<I", 2), (0x60, "<I", 1)))
    code_tab = make_code(0, 0x80, 8)
    raw_at, ctl_at, code_at = 0x100, 0x400, 0x500
    out[raw_at:raw_at + len(ctl_raw) + len(code_raw)] = ctl_raw + code_raw
    out[ctl_at:ctl_at + len(ctl_tab)] = ctl_tab
    out[code_at:code_at + len(code_tab)] = code_tab
    # the record table: offsets from the tag, found at tag + 0x24
    struct.pack_into("<IIII", out, tag + 0x24, ctl_at - tag, (len(ctl_tab) << 4) | 6, code_at - tag,
                     ((len(code_tab) // 16) << 8) | 7)
    return bytes(out), {f"#NDVE@{raw_at:#x}": (ctl_raw, code_raw), f"#NDVE#NDSR@{ctl_at:#x}": (ctl_tab, code_tab)}


def key16(ctl: bytes, code: bytes) -> str:
    """The key, written again from its definition: the first 0x48 control bytes with the u32
    at +0x30 cleared, then the whole code area."""
    head = bytearray(ctl[:0x48])
    head[0x30:0x34] = b"\0\0\0\0"
    return hashlib.sha256(bytes(head) + code).hexdigest()[:16]


def scan_blob(blob: bytes, label: str = "/t", bahs: str = "walk") -> tuple[list[split.Program], list[split.Problem]]:
    problems: list[split.Problem] = []
    found = list(split.Scanner(problems, bahs).scan(memoryview(blob), label))
    return found, problems


def write_tree(root: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


def write_packed(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    index = []
    with open(root / "romfs.bin", "wb") as fh:
        for rel, data in files.items():
            index.append({"path": rel, "offset": fh.tell(), "size": len(data)})
            fh.write(data)
            fh.write(b"\0" * 5)           # the packer leaves gaps
    (root / "index.json").write_text(json.dumps({"files": index}), encoding="utf-8")
    return root


def run_scan(files: dict[str, bytes], out: Path, bahs: str = "walk") -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        image = split.open_image(write_tree(Path(tmp), files))
        try:
            return split.scan_image(image, out, bahs)
        finally:
            image.close()


def read_jobs(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "jobs.jsonl").read_text(encoding="utf-8").splitlines()]


def read_rows(out: Path) -> list[dict[str, str]]:
    lines = (out / "programs.tsv").read_text(encoding="utf-8").splitlines()
    return [dict(zip(lines[0].split("\t"), line.split("\t"))) for line in lines[1:]]


# --- containers -----------------------------------------------------------------

class BnshTest(unittest.TestCase):
    def test_vertex_pixel_and_task_mesh_variations(self):
        vs, ps0, ps1, ts, ms = make_vs(), make_ps(), make_ps(9), make_ts(), make_ms()
        blob = in_container(make_bnsh([{0: vs, 4: ps0}, {4: ps1, 7: ts, 8: ms}]))
        found, problems = scan_blob(blob)
        self.assertEqual(problems, [])
        got = [(p.label, p.stage, p.container) for p in found]
        self.assertEqual(got, [
            ("/t#BNSH@0x1000#var0", "vs", "BNSH"), ("/t#BNSH@0x1000#var0", "ps", "BNSH"),
            ("/t#BNSH@0x1000#var1", "ps", "BNSH"), ("/t#BNSH@0x1000#var1", "ts", "BNSH"),
            ("/t#BNSH@0x1000#var1", "ms", "BNSH")])
        self.assertEqual(found[0].ctl, vs[0])
        self.assertEqual(bytes(found[0].code), vs[1])
        self.assertEqual(bytes(found[4].code), ms[1])

    def test_two_tables_in_one_blob(self):
        a = make_bnsh([{0: make_vs()}])
        b = make_bnsh([{5: make_cs()}])
        blob = bytes(pad_to(b"\x22" * 0x40, 0x1000)) + a
        blob = bytes(pad_to(blob, 0x1000)) + b
        found, problems = scan_blob(blob)
        self.assertEqual(problems, [])
        self.assertEqual([(p.label, p.stage) for p in found],
                         [("/t#BNSH@0x1000#var0", "vs"), ("/t#BNSH@0x2000#var0", "cs")])

    def test_a_graphics_slot_must_agree_with_the_program_header(self):
        blob = in_container(make_bnsh([{4: make_vs()}]))        # a vertex program in the pixel slot
        found, problems = scan_blob(blob)
        self.assertEqual(found, [])
        self.assertEqual(len(problems), 1)
        self.assertIn("program header says vs", problems[0].reason)

    def test_a_magic_without_grsc_is_not_a_table(self):
        found, problems = scan_blob(b"\0" * 64 + b"BNSH\0\0\0\0" + b"\0" * 200)
        self.assertEqual((found, problems), ([], []))


class KeyTest(unittest.TestCase):
    def test_key_matches_its_definition_and_ignores_the_own_length_field(self):
        ctl, code = make_vs()
        other = make_ctl(inst=0x90, code_len=0x200, c1_at=0x100, own=0)     # the BAHS flavour: field is 0
        self.assertEqual(split.ctl_key(ctl, code), key16(ctl, code))
        self.assertEqual(split.ctl_key(ctl, code), split.ctl_key(other, code))

    def test_key_covers_the_first_0x48_bytes_and_the_code_but_not_the_rest(self):
        ctl, code = make_vs()
        tail = bytearray(ctl)
        tail[0x4C] ^= 0xFF
        head = bytearray(ctl)
        head[0x39] ^= 0x01
        self.assertEqual(split.ctl_key(ctl, code), split.ctl_key(bytes(tail), code))
        self.assertNotEqual(split.ctl_key(ctl, code), split.ctl_key(bytes(head), code))
        self.assertNotEqual(split.ctl_key(ctl, code), split.ctl_key(ctl, code[:-16] + bytes(16)))

    def test_stage_rules(self):
        self.assertEqual(split.stage_of(make_vs()[0], make_vs()[1]), "vs")
        self.assertEqual(split.stage_of(make_ps()[0], make_ps()[1]), "ps")
        self.assertEqual(split.stage_of(make_cs()[0], make_cs()[1]), "cs")
        # a task or mesh shader's program header reads as vertex; the control byte wins
        self.assertEqual(split.stage_of(make_ts()[0], make_ts()[1]), "ts")
        self.assertEqual(split.stage_of(make_ms()[0], make_ms()[1]), "ms")
        self.assertIsNone(split.stage_of(make_ctl(inst=16, code_len=16, c1_at=16), bytes(16)))


class SarcTest(unittest.TestCase):
    def test_members_and_nesting(self):
        inner = make_sarc([("deep.bfsha", in_container(make_bnsh([{0: make_vs()}])))])
        sarc = make_sarc([("note.raw", b"nothing here"), ("a.bfsha", in_container(make_bnsh([{4: make_ps()}]))),
                          ("inner.sarc", inner)])
        found, problems = scan_blob(sarc, "/Lib/x.sarc")
        self.assertEqual(problems, [])
        self.assertEqual([(p.label, p.stage) for p in found], [
            ("/Lib/x.sarc/a.bfsha#BNSH@0x1000#var0", "ps"),
            ("/Lib/x.sarc/inner.sarc/deep.bfsha#BNSH@0x1000#var0", "vs")])

    def test_offsets_are_relative_to_the_member(self):
        # the second member starts at a nonzero archive offset but its table sits at its own 0x1000
        first = b"x" * 100
        sarc = make_sarc([("first.raw", first), ("second.bfsha", in_container(make_bnsh([{0: make_vs()}])))])
        found, _ = scan_blob(sarc)
        self.assertEqual([p.label for p in found], ["/t/second.bfsha#BNSH@0x1000#var0"])

    def test_nesting_depth_is_bounded(self):
        blob = make_bnsh([{0: make_vs()}])
        for _ in range(split.MAX_DEPTH + 2):
            blob = make_sarc([("n.sarc", blob)])
        found, problems = scan_blob(blob)
        self.assertEqual(found, [])
        self.assertIn("too deep", problems[0].reason)


class BahsTest(unittest.TestCase):
    RECORDS = [(0, *make_vs(), None), (1, *make_ps(), None), (0, *make_vs(), 0), (3, *make_cs(), None)]

    def labels(self, bahs: str):
        found, problems = scan_blob(make_bahs(self.RECORDS), "/b", bahs)
        return [(p.label, p.stage) for p in found], problems

    def test_walk_reads_the_declared_records(self):
        got, problems = self.labels("walk")
        self.assertEqual(problems, [])
        self.assertEqual([stage for _l, stage in got], ["vs", "ps", "vs", "cs"])
        self.assertTrue(all(label.startswith("/b#BAHS@0x") for label, _s in got))
        found, _ = scan_blob(make_bahs(self.RECORDS), "/b")
        # the third record shares the first one's control section and code: the same key
        self.assertEqual(split.ctl_key(found[0].ctl, found[0].code), split.ctl_key(found[2].ctl, found[2].code))
        self.assertEqual(found[2].ctl, found[0].ctl)

    def test_scan_and_walk_agree_on_a_clean_file(self):
        self.assertEqual(self.labels("scan"), self.labels("walk"))

    def test_walk_checks_kind_against_stage(self):
        records = [(1, *make_vs(), None)]                       # kind 1 is a pixel record
        found, problems = scan_blob(make_bahs(records))
        self.assertEqual(found, [])
        self.assertIn("record kind 1", problems[0].reason)

    def test_walk_refuses_a_version_it_does_not_know(self):
        found, problems = scan_blob(make_bahs(self.RECORDS, version=12))
        self.assertEqual(found, [])
        self.assertIn("version 12", problems[0].reason)

    def test_truncated_file(self):
        blob = make_bahs(self.RECORDS)
        found, problems = scan_blob(blob[:0x180])
        self.assertEqual(found, [])
        self.assertEqual(len(problems), 1)


class NdsrNdveTest(unittest.TestCase):
    def test_ndsr_records(self):
        vs, ps, cs = make_vs(own=0), make_ps(own=0), make_cs(own=0)
        blob = make_ndsr([(*vs, 0, 1), (*ps, 3, 4), (*cs, 6, 7)])
        found, problems = scan_blob(blob, "/p.pck")
        self.assertEqual(problems, [])
        self.assertEqual([(p.stage, p.container) for p in found], [("vs", "NDSR"), ("ps", "NDSR"), ("cs", "NDSR")])
        self.assertTrue(all(p.label.startswith("/p.pck#NDSR@0x") for p in found))
        self.assertEqual(bytes(found[2].code), cs[1])

    def test_ndsr_record_out_of_range_is_reported(self):
        blob = make_ndsr([(*make_vs(own=0), 0, 1), (*make_ps(own=0), 3, 4)])
        found, problems = scan_blob(blob[:-0x40], "/p.pck")
        self.assertEqual(len(found), 1)
        self.assertEqual(len(problems), 1)

    def test_ndve_raw_and_table_programs(self):
        blob, expected = make_ndve()
        found, problems = scan_blob(blob, "/m.dat")
        self.assertEqual(problems, [])
        got = {p.label[len("/m.dat"):]: (p.ctl, bytes(p.code)) for p in found}
        self.assertEqual(got, expected)
        self.assertEqual({p.stage for p in found}, {"cs"})
        self.assertEqual([p.container for p in found], ["NDVE", "NDVE"])


# --- jobs -----------------------------------------------------------------------

class JobTest(unittest.TestCase):
    def job(self, prog: tuple[bytes, bytes], stage: str) -> split.JobSpec:
        return split.build_job(stage, prog[0], memoryview(prog[1]))

    def test_vertex_slice_and_c1(self):
        ctl, code = make_vs()
        spec = self.job((ctl, code), "vs")
        self.assertEqual((spec.start, spec.length), (0, 0x90))
        self.assertEqual(spec.c1, code[0x100:0x200].hex())
        self.assertIsNone(spec.local)
        self.assertIsNone(spec.nvn2_cs)

    def test_compute_fields(self):
        spec = self.job(make_cs(), "cs")
        self.assertEqual((spec.start, spec.length), (0, 0x60))
        self.assertEqual(spec.nvn2_cs, [3, 4, 5, 0x1234, 24, 7, 8])      # +0xB1 +0xB2 +0xB3 +0xBA 4*+0xA9 +0xB4 +0xB5
        self.assertEqual(spec.local, [8, 4, 2])
        self.assertEqual(spec.shared, 32)                                  # 21 rounded up to a multiple of 16
        self.assertEqual(spec.c1, "")

    def test_compute_rejects_a_bad_thread_group(self):
        zero = make_cs(patch=COMPUTE_PATCH + ((0x5C, "<I", 0),))
        big = make_cs(patch=COMPUTE_PATCH + ((0x58, "<I", 1024), (0x5C, "<I", 2)))
        for prog in (zero, big):
            with self.assertRaises(split.JobError):
                self.job(prog, "cs")

    def test_small_task_shader(self):
        spec = self.job(make_ts(payload=20), "ts")
        self.assertEqual((spec.start, spec.length), (0x80, 0x20))          # the program header is cut off
        # outputs at 0x10 + 32; primitive bytes after the 20-byte payload; shared = outputs + 20
        self.assertEqual(spec.stage_memory, [0, 16, 48, 68])
        self.assertEqual(spec.shared, 68)
        self.assertEqual(spec.local, [32, 1, 1])
        self.assertEqual(spec.mesh, {"kind": "ts", "threads": 32, "payload": 20, "own_shared": False})

    def test_small_mesh_shader(self):
        ctl, code = make_ms()
        spec = self.job((ctl, code), "ms")
        self.assertEqual((spec.start, spec.length), (0x80, 0x20))
        # 6 vertex components (position 4 + attribute 0 with 2), 1 per-primitive component;
        # outputs at 0x10 + 16; one block of 7 components * 32 words * 4 bytes
        self.assertEqual(spec.stage_memory, [0, 16, 32, 32 + 7 * 128])
        self.assertEqual(spec.shared, 32 + 7 * 128 + 12)                    # 4 + 3 bytes per primitive, padded to 4
        self.assertEqual(spec.mesh, {"kind": "ms", "threads": 32, "payload": 8, "has_task": True,
                                     "max_vertices": 4, "max_primitives": 2, "vertex_components": 6,
                                     "primitive_components": 1, "own_shared": False})

    def test_shared_memory_use_is_flagged(self):
        ctl, code = make_ts()
        code = bytearray(code)
        struct.pack_into("<H", code, 0x90, 0x984)                           # LDS in the second instruction
        self.assertTrue(self.job((ctl, bytes(code)), "ts").mesh["own_shared"])

    def test_bad_lengths(self):
        for inst in (0, 0x88, 0x300):
            ctl, code = make_vs()
            ctl = bytearray(ctl)
            struct.pack_into("<I", ctl, 0x38, inst)
            with self.assertRaises(split.JobError):
                split.build_job("vs", bytes(ctl), memoryview(code))

    def test_variant_hash_depends_on_the_contract_fields_only(self):
        a = self.job(make_cs(), "cs")
        b = self.job(make_cs(patch=COMPUTE_PATCH + ((0x58, "<I", 16),)), "cs")
        c = self.job(make_cs(patch=COMPUTE_PATCH + ((0x4C, "<I", 99),)), "cs")      # register count: not a job field
        self.assertNotEqual(a.variant_hash(), b.variant_hash())
        self.assertEqual(a.variant_hash(), c.variant_hash())
        canonical = json.dumps({"stage": "cs", "nvn2_cs": a.nvn2_cs, "local": [8, 4, 2], "shared": 32},
                               sort_keys=True, separators=(",", ":"))
        self.assertEqual(a.variant_hash(), hashlib.sha256(canonical.encode()).hexdigest()[:8])
        vs = self.job(make_vs(), "vs")
        self.assertEqual(vs.variant_hash(), hashlib.sha256(b'{"stage":"vs"}').hexdigest()[:8])

    def test_mesh_facts_separate_variants_the_translator_would_merge(self):
        ctl, code = make_ms()
        other = bytearray(ctl)
        other[0x5E] = 0                                    # no task: a mesh fact, not a translator input
        a, b = self.job((ctl, code), "ms"), self.job((bytes(other), code), "ms")
        self.assertEqual((a.stage_memory, a.shared, a.local), (b.stage_memory, b.shared, b.local))
        self.assertNotEqual(a.variant_hash(), b.variant_hash())

    # Job keys written down before the output map existed: the key16 and the variant hash of each
    # synthetic program. The map rides in the job beside the hash, so none of these may move.
    PINNED = {"vs": ("d2f2eea11d8dfd86", "d5786c25"), "ps": ("0640e280b36267f1", "dec1f6cf"),
              "cs": ("550ae4fbaaf1e623", "de8ca3f7"), "ts": ("a25758d13c3cfed8", "4716ae0c"),
              "ms": ("4b3c65d49b814a0f", "eb478558")}

    def test_job_keys_do_not_move_when_a_job_gains_its_output_map(self):
        makers = {"vs": make_vs, "ps": make_ps, "cs": make_cs, "ts": make_ts, "ms": make_ms}
        for stage, make in makers.items():
            with self.subTest(stage=stage):
                ctl, code = make()
                spec = self.job((ctl, code), stage)
                self.assertEqual((split.ctl_key(ctl, code), spec.variant_hash()), self.PINNED[stage])

    def test_only_task_and_mesh_jobs_carry_an_output_map(self):
        for stage, make in (("vs", make_vs), ("ps", make_ps), ("cs", make_cs)):
            self.assertIsNone(self.job(make(), stage).outputs, stage)
        self.assertIsNotNone(self.job(make_ts(), "ts").outputs)
        self.assertIsNotNone(self.job(make_ms(), "ms").outputs)

    def test_a_mesh_shaders_output_map_lists_what_its_headers_write(self):
        spec = self.job(make_ms(), "ms")
        # position x y z w (location -1), attribute 0 components 0 and 1; the second header: attribute 1 component 0
        self.assertEqual(spec.outputs, {"vertex": [[-1, 0], [-1, 1], [-1, 2], [-1, 3], [0, 0], [0, 1]],
                                        "primitive": [[1, 0]]})
        self.assertEqual(len(spec.outputs["vertex"]), spec.mesh["vertex_components"])
        self.assertEqual(len(spec.outputs["primitive"]), spec.mesh["primitive_components"])

    def test_the_map_lists_a_component_only_when_its_bit_is_set(self):
        ctl, code = make_ms()
        code = bytearray(code)
        code[0x35] = 0x50                       # position x and z
        code[0x36:0x36 + 16] = bytes(16)
        code[0x36 + 3] = 0xA4                   # attribute 6: component 2; attribute 7: components 1 and 3
        code[0x200 + 0x35] = 0xF0               # the second header's position is not an output
        code[0x200 + 0x36:0x200 + 0x36 + 16] = bytes(16)
        code[0x200 + 0x36 + 16] = 0x08          # attribute 32 does not exist: bits past the last nibble
        code[0x200 + 0x36 + 15] = 0x80          # attribute 31: component 3
        spec = self.job((ctl, bytes(code)), "ms")
        self.assertEqual(spec.outputs["vertex"], [[-1, 0], [-1, 2], [6, 2], [7, 1], [7, 3]])
        self.assertEqual(spec.outputs["primitive"], [[31, 3]])

    def test_a_task_shader_lists_the_components_of_its_header_too(self):
        ctl, code = make_ts()
        self.assertEqual(self.job((ctl, code), "ts").outputs, {"vertex": written(code[:0x80]), "primitive": []})
        code = bytearray(code)
        code[4:0x80] = bytes(0x7C)
        code[0x35] = 0x30
        code[0x37] = 0x40                       # attribute 3: component 2
        spec = self.job((ctl, bytes(code)), "ts")
        self.assertEqual(spec.outputs, {"vertex": [[-1, 0], [-1, 1], [3, 2]], "primitive": []})

    def test_the_map_is_not_in_the_variant_hash_because_the_key_already_sees_it(self):
        ctl, code = make_ms()
        moved = bytearray(code)
        moved[0x36] = 0x06                      # attribute 0: components 1 and 2 instead of 0 and 1
        a, b = self.job((ctl, code), "ms"), self.job((ctl, bytes(moved)), "ms")
        self.assertNotEqual(a.outputs, b.outputs)
        self.assertEqual(a.variant_hash(), b.variant_hash())               # the counts are the same
        self.assertNotEqual(split.ctl_key(ctl, code), split.ctl_key(ctl, bytes(moved)))   # the code area is hashed
        again = self.job((ctl, code), "ms")
        again.outputs = {"vertex": [[0, 0]], "primitive": []}
        self.assertEqual(a.variant_hash(), again.variant_hash())

    def test_control_tail_digest_ignores_the_length_echoes_only(self):
        ctl, _ = make_vs()
        echo, regs = bytearray(ctl), bytearray(ctl)
        struct.pack_into("<I", echo, 0x98, 0x1234)
        struct.pack_into("<I", regs, 0x4C, 77)
        self.assertEqual(split.ctl_tail_digest(ctl), split.ctl_tail_digest(bytes(echo)))
        self.assertNotEqual(split.ctl_tail_digest(ctl), split.ctl_tail_digest(bytes(regs)))
        self.assertEqual(split.ctl_tail_digest(ctl[:0x60]), split.ctl_tail_digest(ctl[:0x60] + bytes(0x40)))


class ScanImageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "out"

    def files(self):
        cs_a = make_cs()
        cs_b = make_cs(patch=COMPUTE_PATCH + ((0x58, "<I", 16),))          # same code, other thread group
        return {
            "Effect/a.bfsha": in_container(make_bnsh([{0: make_vs(), 4: make_ps()}, {5: cs_a}])),
            "Effect/b.bfsha": in_container(make_bnsh([{0: make_vs()}, {5: cs_b}, {7: make_ts(), 8: make_ms()}])),
            "Pack/other.bin": b"not scanned",
        }

    def test_outputs(self):
        summary = run_scan(self.files(), self.out)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["files_scanned"], 2)
        self.assertEqual(summary["records"], 7)
        self.assertEqual(summary["jobs"], 6)
        self.assertEqual(summary["distinct_key16"], 5)
        self.assertEqual(summary["key16_with_several_jobs"], 1)
        self.assertEqual(summary["jobs_by_stage"], {"cs": 2, "ms": 1, "ps": 1, "ts": 1, "vs": 1})
        self.assertEqual(summary["only_product100_jobs"], 0)
        jobs = read_jobs(self.out)
        self.assertEqual(len(jobs), 6)
        blob = (self.out / "programs.bin").read_bytes()
        by_key = {j["key"]: j for j in jobs}
        vs_ctl, vs_code = make_vs()
        job = by_key[f"{key16(vs_ctl, vs_code)}.{hashlib.sha256(b'{\"stage\":\"vs\"}').hexdigest()[:8]}"]
        self.assertEqual(job["stage"], "vs")
        self.assertEqual(job["arch"], "sm86")
        self.assertEqual(Path(job["code"]), (self.out / "programs.bin").resolve())
        self.assertEqual(job["code_offset"] % 16, 0)
        self.assertEqual(blob[job["code_offset"]:job["code_offset"] + job["code_length"]], vs_code[:0x90])
        self.assertEqual(job["c1"], vs_code[0x100:0x200].hex())
        self.assertEqual(set(job), {"key", "stage", "arch", "code", "code_offset", "code_length", "c1"})
        ts_ctl, ts_code = make_ts()
        ts_job = next(j for j in jobs if j["stage"] == "ts")
        self.assertEqual(blob[ts_job["code_offset"]:ts_job["code_offset"] + ts_job["code_length"]], ts_code[0x80:0xA0])
        self.assertEqual(ts_job["stage_memory"], [0, 16, 48, 68])
        self.assertEqual(ts_job["mesh"]["kind"], "ts")
        self.assertEqual(ts_job["outputs"], {"vertex": written(ts_code[:0x80]), "primitive": []})
        ms_job = next(j for j in jobs if j["stage"] == "ms")
        self.assertEqual(ms_job["outputs"]["vertex"][:2], [[-1, 0], [-1, 1]])
        self.assertEqual(ms_job["outputs"]["primitive"], [[1, 0]])
        self.assertEqual({j["stage"] for j in jobs if "outputs" in j}, {"ts", "ms"})
        pinned = {"vs": "d2f2eea11d8dfd86.d5786c25", "ts": "a25758d13c3cfed8.4716ae0c",
                  "ms": "4b3c65d49b814a0f.eb478558", "ps": "0640e280b36267f1.dec1f6cf"}
        for stage, key in pinned.items():
            self.assertEqual(next(j for j in jobs if j["stage"] == stage)["key"], key)
        # the two compute variants share one slice
        cs_jobs = [j for j in jobs if j["stage"] == "cs"]
        self.assertEqual(len(cs_jobs), 2)
        self.assertEqual((cs_jobs[0]["code_offset"], cs_jobs[0]["code_length"]),
                         (cs_jobs[1]["code_offset"], cs_jobs[1]["code_length"]))
        self.assertEqual([j["local"] for j in cs_jobs], [[8, 4, 2], [16, 4, 2]])

    def job_key(self, stage: str, prog: tuple[bytes, bytes]) -> str:
        spec = split.build_job(stage, prog[0], memoryview(prog[1]))
        return f"{split.ctl_key(prog[0], prog[1])}.{spec.variant_hash()}"

    def pairing(self) -> list[list[str]]:
        lines = (self.out / "pairing.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0].split("\t"), ["variation", "vertex", "pixel", "task", "mesh", "product100"])
        return [line.split("\t") for line in lines[1:]]

    def test_pairing_file_lists_the_stage_keys_of_each_variation(self):
        vs, ps1, ps2, ts, ms = make_vs(), make_ps(), make_ps(9), make_ts(), make_ms()
        compute = make_cs()
        blob = in_container(make_bnsh([{0: vs, 4: ps1}, {4: ps1, 7: ts, 8: ms}, {4: ps2, 7: ts, 8: ms},
                                       {8: ms}, {5: compute}, {7: ts, 8: ms}]))
        summary = run_scan({"Effect/a.bfsha": blob}, self.out)
        self.assertEqual(summary["error_count"], 0)
        key = self.job_key
        label = "/Effect/a.bfsha#BNSH@0x1000#var"
        self.assertEqual(self.pairing(), [
            [label + "0", key("vs", vs), key("ps", ps1), "", "", "0"],
            [label + "1", "", key("ps", ps1), key("ts", ts), key("ms", ms), "0"],
            [label + "2", "", key("ps", ps2), key("ts", ts), key("ms", ms), "0"],
            [label + "3", "", "", "", key("ms", ms), "0"],
            [label + "5", "", "", key("ts", ts), key("ms", ms), "0"]])  # the compute-only variation has no row
        counts = {"variations": 5, "variations_with_mesh": 4, "task_mesh_pairs": 1, "task_mesh_pixel_triples": 2}
        self.assertEqual(summary["pairing"], {**counts, "outside_product100": counts})

    def test_pairing_counts_are_also_given_outside_the_superseded_files(self):
        ts, ms, ms2, ps = make_ts(), make_ms(), make_ms(9), make_ps()
        new = in_container(make_bnsh([{4: ps, 7: ts, 8: ms}]))
        old = in_container(make_bnsh([{4: ps, 7: ts, 8: ms}, {7: ts, 8: ms2}]))
        summary = run_scan({"Shader/M.Product.110.bfsha": new, "Shader/M.Product.100.bfsha": old}, self.out)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual([row[-1] for row in self.pairing()], ["1", "1", "0"])
        self.assertEqual(summary["pairing"]["task_mesh_pairs"], 2)           # the old file adds a pair
        self.assertEqual(summary["pairing"]["outside_product100"],
                         {"variations": 1, "variations_with_mesh": 1, "task_mesh_pairs": 1,
                          "task_mesh_pixel_triples": 1})

    def test_pairing_marks_a_stage_whose_record_did_not_parse(self):
        ts, ms, ps, vs = make_ts(), make_ms(), make_ps(), make_vs()
        for slot, column in ((0, 1), (4, 2), (7, 3), (8, 4)):
            with self.subTest(slot=slot):
                stages = {0: vs, 4: ps, 7: ts, 8: ms}
                bad = bytearray(stages[slot][0])
                struct.pack_into("<I", bad, 0x44, len(stages[slot][1]) + 16)     # code length differs
                stages[slot] = (bytes(bad), stages[slot][1])
                summary = run_scan({"Effect/a.bfsha": in_container(make_bnsh([stages]))}, self.out)
                self.assertEqual(summary["error_count"], 1)
                row = self.pairing()[0]
                self.assertEqual(row[column], "?")
                self.assertEqual(sum(1 for c in row[1:5] if c), 4)
                self.assertEqual((summary["pairing"]["task_mesh_pairs"], summary["pairing"]["task_mesh_pixel_triples"]),
                                 {0: (1, 1), 4: (1, 0)}.get(slot, (0, 0)))

    def test_pairing_marks_a_stage_whose_job_could_not_be_built(self):
        ts, ms, ps = make_ts(), make_ms(), make_ps()
        bad = bytearray(ts[0])
        struct.pack_into("<I", bad, 0x38, 0x1000)
        summary = run_scan({"Effect/a.bfsha": in_container(make_bnsh([{4: ps, 7: (bytes(bad), ts[1]), 8: ms}]))},
                           self.out)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(self.pairing()[0][1:], ["", self.job_key("ps", ps), "?", self.job_key("ms", ms), "0"])
        self.assertEqual((summary["pairing"]["task_mesh_pairs"], summary["pairing"]["task_mesh_pixel_triples"]), (0, 0))

    def test_pairing_file_is_only_for_bnsh_variations(self):
        summary = run_scan({"Effect/a.bfsha": in_container(make_bnsh([{0: make_vs()}])),
                            "Lib/b.bahs": make_bahs([(0, *make_vs(7), None)])}, self.out)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["jobs_by_container"], {"BAHS": 1, "BNSH": 1})
        self.assertEqual(len(self.pairing()), 1)

    def test_variants_of_one_key_with_other_output_maps_are_reported(self):
        # Same code area, so the same key16; the second header sits at 0x200 in one control section and at 0x280 in
        # the other, with one component each, so the counts and the variant hash agree and only the map differs.
        code = bytearray(make_code(VS_WORD, 0x300, 5))
        code[4:0x80] = bytes(0x7C)
        for at in (0x200, 0x280):
            code[at:at + 0x80] = bytes(0x80)
        code[0x35] = 0xF0
        code[0x200 + 0x36] = 0x01
        code[0x280 + 0x36] = 0x10
        patch = ((0x58, "<H", 32), (0x5A, "<H", 8), (0x5E, "<B", 1), (0x60, "<H", 4), (0x62, "<H", 2),
                 (0x68, "<B", 0x80))
        near = make_ctl(stage=7, inst=0xA0, code_len=0x300, c1_at=0x100, patch=patch + ((0x64, "<I", 0x200),))
        far = make_ctl(stage=7, inst=0xA0, code_len=0x300, c1_at=0x100, patch=patch + ((0x64, "<I", 0x280),))
        self.assertEqual(self.job_key("ms", (near, bytes(code))), self.job_key("ms", (far, bytes(code))))
        summary = run_scan({"Effect/a.bfsha": in_container(make_bnsh([{8: (near, bytes(code))},
                                                                      {8: (far, bytes(code))}]))}, self.out)
        self.assertEqual(summary["jobs"], 1)
        self.assertEqual(summary["error_count"], 1)
        self.assertIn("output map differs", summary["errors"][0]["reason"])
        self.assertEqual(read_jobs(self.out)[0]["outputs"]["primitive"], [[0, 0]])      # the first one's map

    def test_rows(self):
        run_scan(self.files(), self.out)
        rows = read_rows(self.out)
        self.assertEqual(len(rows), 6)
        vs = next(r for r in rows if r["stage"] == "vs")
        self.assertEqual(vs["records"], "2")
        self.assertEqual(vs["group"], "effect")
        self.assertEqual(vs["container"], "BNSH")
        self.assertEqual(vs["first_label"], "/Effect/a.bfsha#BNSH@0x1000#var0")
        self.assertEqual(vs["only_product100"], "0")
        cs = [r for r in rows if r["stage"] == "cs"]
        self.assertEqual([r["variants"] for r in cs], ["2", "2"])
        self.assertEqual([r["records"] for r in cs], ["1", "1"])
        self.assertEqual(len({r["key16"] for r in cs}), 1)
        self.assertEqual({r["ctl_variants"] for r in rows}, {"1"})

    def test_merged_control_variants_are_counted(self):
        ctl, code = make_vs()
        regs = bytearray(ctl)
        struct.pack_into("<I", regs, 0x4C, 77)            # same key and job fields, other register count
        echo = bytearray(ctl)
        struct.pack_into("<I", echo, 0x98, 0x40)          # a length echo only
        files = {"Effect/a.bfsha": in_container(make_bnsh([{0: (ctl, code)}, {0: (bytes(echo), code)}])),
                 "Effect/b.bfsha": in_container(make_bnsh([{0: (bytes(regs), code)}]))}
        summary = run_scan(files, self.out)
        self.assertEqual((summary["jobs"], summary["jobs_with_merged_ctl_variants"]), (1, 1))
        (row,) = read_rows(self.out)
        self.assertEqual((row["records"], row["ctl_variants"]), ("3", "2"))

    def test_superseded_product_files_are_marked_and_do_not_claim_first_place(self):
        shared, only_old, only_new = make_vs(11), make_vs(12), make_vs(13)
        files = {
            "Shader/M.Product.100.bfsha": in_container(make_bnsh([{0: shared}, {0: only_old}])),
            "Shader/M.Product.110.bfsha": in_container(make_bnsh([{0: shared}, {0: only_new}])),
            "Layout2Shader/M.Product.100.bfsha": in_container(make_bnsh([{0: make_vs(14)}])),
        }
        summary = run_scan(files, self.out)
        rows = {r["key16"]: r for r in read_rows(self.out)}
        self.assertEqual(summary["only_product100_jobs"], 1)
        self.assertEqual(rows[key16(*shared)]["only_product100"], "0")
        self.assertEqual(rows[key16(*shared)]["first_label"], "/Shader/M.Product.110.bfsha#BNSH@0x1000#var0")
        self.assertEqual(rows[key16(*shared)]["records"], "2")
        self.assertEqual(rows[key16(*only_old)]["only_product100"], "1")
        self.assertEqual(rows[key16(*only_old)]["first_label"], "/Shader/M.Product.100.bfsha#BNSH@0x1000#var1")
        self.assertEqual(rows[key16(*only_new)]["only_product100"], "0")
        # the layout file is not one of the superseded set
        self.assertEqual(rows[key16(*make_vs(14))]["only_product100"], "0")
        self.assertEqual(rows[key16(*make_vs(14))]["group"], "layout")
        self.assertEqual(summary["key16_outside_product100"], 3)

    def test_group_names(self):
        self.assertEqual(split.group_of("/Shader/x"), "shader")
        self.assertEqual(split.group_of("/Lib/agl/x.sarc/y"), "lib")
        self.assertEqual(split.group_of("/Layout2Shader/x"), "layout")
        self.assertEqual(split.group_of("/Movie/x"), "movie")

    def test_files_are_scanned_in_sorted_order(self):
        a, b = make_vs(21), make_vs(22)
        files = {"Lib/z.bfsha": in_container(make_bnsh([{0: a}])),
                 "Effect/y.bfsha": in_container(make_bnsh([{0: a}, {0: b}]))}
        run_scan(files, self.out)
        rows = read_rows(self.out)
        self.assertEqual([r["key16"] for r in rows], [key16(*a), key16(*b)])
        self.assertEqual(rows[0]["first_label"], "/Effect/y.bfsha#BNSH@0x1000#var0")

    def test_all_containers_in_one_run_and_both_bahs_modes(self):
        ndve, _ = make_ndve()
        files = {
            "Lib/agl.sarc": make_sarc([("c.sharcb", make_bahs(BahsTest.RECORDS)),
                                       ("p.pck", make_ndsr([(*make_vs(31, own=0), 0, 1)]))]),
            "Movie/v.dat": ndve,
        }
        walk = run_scan(files, self.out / "walk", "walk")
        scan = run_scan(files, self.out / "scan", "scan")
        self.assertEqual(walk["jobs_by_container"], {"BAHS": 3, "NDSR": 1, "NDVE": 2})
        self.assertEqual(walk["jobs_by_container"], scan["jobs_by_container"])
        self.assertEqual(walk["error_count"], 0)
        self.assertEqual(walk["jobs_by_group"], {"lib": 4, "movie": 2})


class ImageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.files = {"Effect/a.bfsha": in_container(make_bnsh([{0: make_vs()}])), "Lib/b.bin": b"abc",
                      "Model/x.bin": b"skipped"}

    def test_packed_and_tree_forms_give_the_same_result(self):
        packed = write_packed(self.root / "packed", self.files)
        tree = write_tree(self.root / "tree", self.files)
        results = []
        for name, folder in (("a", packed), ("b", tree)):
            image = split.open_image(folder)
            self.addCleanup(image.close)
            self.assertEqual(image.list_paths(split.TREES), ["Effect/a.bfsha", "Lib/b.bin"])
            results.append(split.scan_image(image, self.root / name)["jobs"])
        self.assertEqual(results, [1, 1])
        self.assertEqual((self.root / "a" / "programs.bin").read_bytes(), (self.root / "b" / "programs.bin").read_bytes())

    def test_bad_inputs(self):
        with self.assertRaises(split.InputError):
            split.open_image(self.root / "missing")
        (self.root / "empty").mkdir()
        with self.assertRaises(split.InputError):
            split.open_image(self.root / "empty")
        bad = self.root / "bad"
        bad.mkdir()
        (bad / "romfs.bin").write_bytes(b"0123456789")
        for text in ("{not json", '{"files": 3}', '{"files": [{"path": "a", "offset": 0}]}',
                     '{"files": [{"path": "a", "offset": 8, "size": 9}]}'):
            (bad / "index.json").write_text(text, encoding="utf-8")
            with self.assertRaises(split.InputError, msg=text):
                split.PackedImage(bad)
        (bad / "index.json").unlink()
        with self.assertRaises(split.InputError):
            split.PackedImage(bad)

    def test_cli_exit_codes(self):
        packed = write_packed(self.root / "packed", self.files)
        out = self.root / "out"
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(split.main(["scan", "--romfs", str(packed), "--out", str(out)]), 0)
        self.assertIn("jobs: 1", stdout.getvalue())
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(split.main(["scan", "--romfs", str(self.root / "nope"), "--out", str(out)]), 2)
        self.assertIn("is not a folder", stderr.getvalue())
        broken = write_tree(self.root / "broken", {"Effect/a.bfsha": in_container(make_bnsh([{0: make_vs()}]))[:0x1100]})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(split.main(["scan", "--romfs", str(broken), "--out", str(self.root / "o2")]), 1)
        inside = ROOT / "split-test-output"
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(split.main(["scan", "--romfs", str(packed), "--out", str(inside)]), 2)
        self.assertIn("inside the repository", stderr.getvalue())
        self.assertFalse(inside.exists())


class CorruptTest(unittest.TestCase):
    """Damage must become a Problem record, never an exception."""

    def test_truncated_and_damaged_containers(self):
        bnsh = in_container(make_bnsh([{0: make_vs(), 4: make_ps()}]))
        sarc = make_sarc([("a.bfsha", bnsh), ("b.bfsha", bnsh)])
        bahs = make_bahs(BahsTest.RECORDS)
        ndsr = make_ndsr([(*make_vs(own=0), 0, 1)])
        ndve, _ = make_ndve()
        damaged = bytearray(bnsh)
        struct.pack_into("<I", damaged, 0x1000 + 0x7C, 0xFFFFFFF)            # absurd variation count
        wild = bytearray(bnsh)
        struct.pack_into("<Q", wild, 0x1000 + 0xC0 + 0x10, 0x7FFFFFFF00)     # program pointer far away
        cases = {"bnsh cut": bnsh[:0x1200], "bnsh variations": bytes(damaged), "bnsh pointer": bytes(wild),
                 "sarc cut": sarc[:len(sarc) // 2], "sarc head": sarc[:30], "sarc table": sarc[:0x30],
                 "bahs cut": bahs[:0x300], "bahs head": bahs[:0x10], "ndsr cut": ndsr[:0x60], "ndsr head": ndsr[:6],
                 "ndve cut": ndve[:0x120], "ndve head": ndve[:8]}
        for name, blob in cases.items():
            for mode in ("walk", "scan"):
                with self.subTest(case=name, bahs=mode):
                    found, problems = scan_blob(blob, "/d", mode)         # must not raise
                    self.assertIsInstance(found, list)
                    self.assertIsInstance(problems, list)
        found, problems = scan_blob(bnsh[:0x1200])
        self.assertEqual((found, len(problems)), ([], 1))
        found, problems = scan_blob(bytes(wild))
        self.assertEqual(found, [])
        self.assertEqual(len(problems), 1)
        found, problems = scan_blob(sarc[:len(sarc) // 2])
        self.assertTrue(problems)

    def test_a_damaged_stage_record_leaves_the_others(self):
        blob = bytearray(in_container(make_bnsh([{0: make_vs(), 4: make_ps()}])))
        program = struct.unpack_from("<Q", blob, 0x1000 + 0xC0 + 0x10)[0]
        record = struct.unpack_from("<Q", blob, 0x1000 + program + 8)[0]       # slot 0
        struct.pack_into("<Q", blob, 0x1000 + record + 8, 0x7FFFFFFF00)        # its control pointer
        found, problems = scan_blob(bytes(blob))
        self.assertEqual([p.stage for p in found], ["ps"])
        self.assertEqual(len(problems), 1)
        self.assertIn("slot 0", problems[0].reason)

    def test_a_job_that_cannot_be_built_is_a_problem_not_a_job(self):
        ctl, code = make_vs()
        bad = bytearray(ctl)
        struct.pack_into("<I", bad, 0x38, 0x1000)                              # more instructions than code
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "o"
            summary = run_scan({"Effect/a.bfsha": in_container(make_bnsh([{0: (bytes(bad), code)}, {0: make_vs(2)}]))}, out)
        self.assertEqual(summary["jobs"], 1)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["errors"][0]["file"], "Effect/a.bfsha")
        self.assertIn("instruction length", summary["errors"][0]["reason"])


class DeterminismTest(unittest.TestCase):
    def test_output_does_not_depend_on_the_hash_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ndve, _ = make_ndve()
            tree = write_tree(root / "tree", {
                "Effect/a.bfsha": in_container(make_bnsh([{0: make_vs(), 4: make_ps()}, {5: make_cs()}])),
                "Effect/b.bfsha": in_container(make_bnsh([{5: make_cs(patch=COMPUTE_PATCH + ((0x58, "<I", 16),))},
                                                          {7: make_ts(), 8: make_ms()}])),
                "Lib/agl.sarc": make_sarc([("c.sharcb", make_bahs(BahsTest.RECORDS))]),
                "Movie/v.dat": ndve,
            })
            outputs = []
            for seed in ("1", "2", "31337"):
                env = dict(os.environ, PYTHONHASHSEED=seed)
                done = subprocess.run([sys.executable, "-P", "-s", str(SPLIT), "scan", "--romfs", str(tree),
                                       "--out", str(root / "out")], env=env, capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)
                outputs.append({name: (root / "out" / name).read_bytes()
                                for name in ("jobs.jsonl", "programs.tsv", "programs.bin")})
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[0], outputs[2])
            self.assertGreater(len(outputs[0]["jobs.jsonl"]), 0)


class CompareTest(unittest.TestCase):
    def test_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, c = make_vs(41), make_ps(42), make_cs(43)
            old = make_vs(44)
            run_scan({"Effect/a.bfsha": in_container(make_bnsh([{0: a}, {4: b}, {5: c}])),
                      "Shader/M.Product.100.bfsha": in_container(make_bnsh([{0: old}]))}, root / "out")
            label = "/Effect/a.bfsha#BNSH@0x1000#var"
            header = "package\tindex\tstage\tdm_ctl_sha16\tsource\n"
            rows = [("vs_effect.bin", 0, "vertex", key16(*a), label + "0"),
                    ("ps_effect.bin", 0, "pixel", key16(*b), label + "1"),
                    ("cs_effect.bin", 0, "compute", key16(*c), label + "2")]

            def write_map(entries):
                path = root / "map.tsv"
                path.write_text("# a comment\n" + header + "".join("\t".join(map(str, r)) + "\n" for r in entries),
                                encoding="utf-8")
                return path

            programs = root / "out" / "programs.tsv"
            self.assertEqual(split.compare_files(programs, write_map(rows)), {
                "map keys": 3, "our keys": 3, "both": 3, "only map": 0, "only ours": 0,
                "stage disagreements": 0, "source disagreements": 0})
            # counter-run: a missing key, an extra key, a wrong stage and a wrong source
            rows[0] = rows[0][:2] + ("pixel",) + rows[0][3:]
            rows[1] = rows[1][:4] + ("/elsewhere",)
            rows[2] = rows[2][:3] + ("0123456789abcdef", rows[2][4])
            self.assertEqual(split.compare_files(programs, write_map(rows)), {
                "map keys": 3, "our keys": 3, "both": 2, "only map": 1, "only ours": 1,
                "stage disagreements": 1, "source disagreements": 1})
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = split.main(["compare", "--programs", str(programs), "--map", str(root / "map.tsv")])
            self.assertEqual(code, 1)
            self.assertNotIn(key16(*a), stdout.getvalue())          # counts only, never keys

    def test_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "p.tsv").write_text("not\ta\tprograms\n", encoding="utf-8")
            (root / "m.tsv").write_text("a\tb\n", encoding="utf-8")
            with self.assertRaises(split.InputError):
                split.compare_files(root / "p.tsv", root / "m.tsv")
            with self.assertRaises(split.InputError):
                split.compare_files(root / "none.tsv", root / "m.tsv")


# --- zstd and dictionaries ------------------------------------------------------

@NEEDS_ZSTD
class ZstdTest(unittest.TestCase):
    def trained_dictionary(self):
        samples = [(b"program table %d " % i) * 40 + bytes((i * j) & 0xFF for j in range(64)) for i in range(400)]
        try:
            return zstd.train_dict(samples, 2048)
        except zstd.ZstdError as exc:
            self.skipTest(f"cannot train a dictionary here: {exc}")

    def test_frame_header_names_the_dictionary(self):
        zd = self.trained_dictionary()
        frame = zstd.compress(b"payload" * 50, zstd_dict=zd)
        if split.frame_dict_id(frame) != zd.dict_id:
            self.skipTest("this zstd build does not write dictionary IDs")
        self.assertEqual(split.frame_dict_id(zstd.compress(b"payload")), 0)

    def test_dictionary_pack_and_piecewise_expansion(self):
        zd = self.trained_dictionary()
        content = zd.dict_content
        self.assertEqual(content[:4], split.ZSTD_DICT_MAGIC)
        pack = zstd.compress(make_sarc([("zs.zsdic", content), ("other.bin", b"x" * 100)]))
        blob = in_container(make_bnsh([{0: make_vs()}, {5: make_cs()}]))
        frame = zstd.compress(blob, zstd_dict=zd)
        if split.frame_dict_id(frame) != zd.dict_id:
            self.skipTest("this zstd build does not write dictionary IDs")
        files = {"Pack/ZsDic.pack.zs": pack, "Effect/a.vsetb.byml.zs": frame, "Effect/plain.zs": zstd.compress(blob)}
        with tempfile.TemporaryDirectory() as tmp:
            packed = write_packed(Path(tmp) / "img", files)
            for piece in (split.PIECE, 97):                              # 97: many pieces for a small file
                with self.subTest(piece=piece), mock.patch.object(split, "PIECE", piece):
                    image = split.open_image(packed)
                    try:
                        summary = split.scan_image(image, Path(tmp) / f"out{piece}")
                    finally:
                        image.close()
                    self.assertEqual(summary["error_count"], 0, summary["errors"])
                    self.assertEqual(summary["files_scanned"], 2)
                    self.assertEqual(summary["records"], 4)
                    self.assertEqual(summary["jobs"], 2)

    def test_unknown_dictionary_and_junk_frames_are_reported(self):
        zd = self.trained_dictionary()
        frame = zstd.compress(b"payload" * 50, zstd_dict=zd)
        if split.frame_dict_id(frame) != zd.dict_id:
            self.skipTest("this zstd build does not write dictionary IDs")
        pack = zstd.compress(make_sarc([("other.bin", b"x" * 100)]))          # a pack without that dictionary
        junk = zstd.compress(b"abcdef" * 100)[:20]
        with tempfile.TemporaryDirectory() as tmp:
            packed = write_packed(Path(tmp) / "img", {"Pack/ZsDic.pack.zs": pack, "Effect/a.zs": frame,
                                                       "Effect/b.zs": junk})
            image = split.open_image(packed)
            try:
                summary = split.scan_image(image, Path(tmp) / "out")
            finally:
                image.close()
        self.assertEqual(summary["error_count"], 2)
        self.assertEqual([e["file"] for e in summary["errors"]], ["Effect/a.zs", "Effect/b.zs"])


class ZstdMissingTest(unittest.TestCase):
    def test_frame_without_zstd_support_stops_the_run_cleanly(self):
        frame = bytes.fromhex("28b52ffd") + b"\x00" * 20
        with tempfile.TemporaryDirectory() as tmp:
            tree = write_tree(Path(tmp) / "t", {"Effect/a.zs": frame})
            image = split.open_image(tree)
            try:
                with mock.patch.dict(sys.modules, {"compression": None, "compression.zstd": None}):
                    with self.assertRaises(split.InputError) as raised:
                        split.scan_image(image, Path(tmp) / "out")
            finally:
                image.close()
        self.assertIn("Python 3.14", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
