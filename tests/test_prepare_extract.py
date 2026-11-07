# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the extract and modules steps of `oot2 prepare` and the ZBIC decoder fetch (tools/prepare/nup.py,
extract.py, modules.py, zbic.py).

Every input is synthetic and built inside the test: containers from a small writer of the .nup layout, modules from
made-up segments (LZ4 blocks of literals, or a stand-in for ZBIC frames), and the decoder's source archive from a
few lines of C that only copy bytes. No game file is read and nothing is downloaded. The build of the stand-in
decoder needs clang (OOT2_CC, else clang on PATH) and skips without one unless OOT2_REQUIRE_CLANG is set.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import struct
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from prepare import extract, modules, nup, runner, zbic  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, PrepareError, Reporter, Step, run_steps  # noqa: E402


def zstd_compress():
    try:
        from compression import zstd
        return zstd.compress
    except ImportError:
        pass
    try:
        import zstandard
        return zstandard.ZstdCompressor().compress
    except ImportError:
        return None


# ---------------------------------------------------------------------------------------------------------------
# A writer of the .nup layout (tools/prepare/nup.py describes it).

class NupWriter:
    """Packages of ExeFS and RomFS trees to container bytes. Identical stored bytes share one extent, as in a real
    container whose update points its unchanged files at the base's."""

    def __init__(self):
        self.packages = []

    def package(self, kind, version, exefs=None, romfs=None, title_id=0x0100000000010000):
        self.packages.append((kind, version, title_id, exefs, romfs))
        return self

    def build(self, sha_override=None) -> bytes:
        sha_override = sha_override or {}
        dirs, files = [], []          # dirs: [name, [("d", i) | ("f", i)]]; files: [name, data, stored, codec, sha]

        def add_dir(name, tree):
            idx = len(dirs)
            dirs.append([name, []])
            sub: dict[str, dict] = {}
            for path, spec in sorted(tree.items()):
                head, _, rest = path.partition("/")
                if rest:
                    sub.setdefault(head, {})[rest] = spec
                else:
                    data, codec = spec if isinstance(spec, tuple) else (spec, nup.RAW)
                    stored = zstd_compress()(data) if codec == nup.ZSTD else data
                    sha = sha_override.get(path, hashlib.sha256(data).digest())
                    files.append([head, data, stored, codec, sha])
                    dirs[idx][1].append(("f", len(files) - 1))
            for head, t in sorted(sub.items()):
                dirs[idx][1].append(("d", add_dir(head, t)))
            return idx

        pkgs = []
        for kind, version, title_id, exefs, romfs in self.packages:
            root = len(dirs)
            dirs.append(["", []])
            ex = add_dir("exefs", exefs) if exefs is not None else nup.NO_DIR
            ro = add_dir("romfs", romfs) if romfs is not None else nup.NO_DIR
            dirs[root][1] = [("d", d) for d in (ex, ro) if d != nup.NO_DIR]
            pkgs.append((title_id, root, ex, ro, version, kind))

        pool = bytearray()
        names = {}

        def name_ref(s):
            if s not in names:
                raw = s.encode("utf-8")
                names[s] = (len(pool), len(raw))
                pool.extend(raw)
            return names[s]

        children = []
        dir_recs = []
        for name, kids in dirs:
            dir_recs.append((*name_ref(name), len(children), len(kids)))
            children.extend(i if t == "d" else len(dirs) + i for t, i in kids)
        file_names = [name_ref(f[0]) for f in files]

        needed = nup.TOC_HEADER.size + len(pkgs) * nup.PACKAGE_SIZE + len(dirs) * nup.DIRECTORY.size +             len(files) * nup.FILE.size
        pool_off = needed + 4 * len(children)
        toc_off = nup.SUPER_SIZE
        toc_size = pool_off + len(pool)
        blob_off = (toc_off + toc_size + 15) & ~15
        blobs = bytearray()
        extents = {}
        file_recs = []
        for (off, ln), (name, data, stored, codec, sha) in zip(file_names, files):
            if not data:
                file_recs.append((off, ln, 0, 0, 0, sha, codec))
                continue
            key = (stored, codec)
            if key not in extents:
                blobs.extend(bytes(-len(blobs) % 16))
                extents[key] = blob_off + len(blobs)
                blobs.extend(stored)
            file_recs.append((off, ln, len(data), extents[key], len(stored), sha, codec))
        toc = bytearray(nup.TOC_HEADER.pack(nup.TOC_MAGIC, len(pkgs), len(dirs), len(files), pool_off, len(pool), 0))
        for title_id, root, ex, ro, version, kind in pkgs:
            toc += nup.PACKAGE.pack(title_id, root, ex, ro, version, kind, 0, 0, 0, 0, 0).ljust(nup.PACKAGE_SIZE, b"\0")
        for rec in dir_recs:
            toc += nup.DIRECTORY.pack(*rec, 0)
        for rec in file_recs:
            toc += nup.FILE.pack(*rec, 0, 0, bytes(12))
        toc += struct.pack(f"<{len(children)}I", *children)
        toc += pool
        assert len(toc) == toc_size
        body = bytes(toc).ljust(blob_off - toc_off, b"\0") + bytes(blobs)
        total = toc_off + len(body)
        sb = nup.SUPER.pack(nup.MAGIC, nup.SUPER_SIZE, 1, toc_off, toc_size, hashlib.sha256(toc).digest(),
                            blob_off, total, len(pkgs), 0).ljust(nup.SUPER_SIZE, b"\0")
        return sb + body


# ---------------------------------------------------------------------------------------------------------------
# Synthetic modules.

def lz4_literals(data: bytes) -> bytes:
    """An LZ4 block holding data as one run of literals (valid: the last sequence of a block has no match)."""
    n = len(data)
    out = bytearray([min(n, 15) << 4])
    if n >= 15:
        rest = n - 15
        while rest >= 255:
            out.append(255)
            rest -= 255
        out.append(rest)
    return bytes(out) + data


def fake_zbic(data: bytes) -> bytes:
    """A stand-in for a ZBIC frame: the magic, then the bytes reversed. fake_unzbic undoes it."""
    return b"ZBIC" + data[::-1]


def fake_unzbic(frame: bytes, size: int) -> bytes:
    if frame[:4] != b"ZBIC":
        raise ValueError("not a stand-in frame")
    return frame[4:][::-1]


def segments(seed: int = 1):
    """text with MOD0 at 0x10 pointing at a dynamic section in data, rodata and data, as (mem offset, bytes)."""
    text = bytearray(0x1000)
    struct.pack_into("<I", text, 4, 0x10)                       # the MOD0 offset, at text + 4
    dynamic, bss_start, bss_end, eh0, eh1 = 0x3000, 0x3800, 0x3900, 0x2000, 0x2010
    struct.pack_into("<4s6i", text, 0x10, b"MOD0", dynamic - 0x10, bss_start - 0x10, bss_end - 0x10, eh0 - 0x10,
                     eh1 - 0x10, 0)
    text[0x100:0x110] = bytes(range(seed, seed + 16))
    rodata = bytes((i * seed) & 0xFF for i in range(0x800))
    data = bytes((i + seed) & 0xFF for i in range(0x800))
    return [(0, bytes(text)), (0x2000, rodata), (0x3000, data)]


def nso(segs, codec="lz4", build_id=b"\x11" * 20, bss=0x100, name=b"", hashes=True, bad_hash=None) -> bytes:
    """An NSO0 file. codec: plain, lz4 (flags 0..2) or zbic (flags 0..2 and 7, stand-in frames)."""
    hdr = bytearray(modules.HDR)
    hdr[:4] = b"NSO0"
    flags = {"plain": 0, "lz4": 0x7, "zbic": 0x87}[codec] | (0x38 if hashes else 0)
    struct.pack_into("<I", hdr, 0xC, flags)
    body = bytearray(name)
    for i, (mem, data) in enumerate(segs):
        stored = {"plain": data, "lz4": lz4_literals(data), "zbic": fake_zbic(data)}[codec]
        struct.pack_into("<III", hdr, 0x10 + i * 0x10, modules.HDR + len(body), mem, len(data))
        struct.pack_into("<I", hdr, 0x60 + i * 4, len(stored))
        digest = hashlib.sha256(data).digest() if bad_hash != i else bytes(32)
        hdr[0xA0 + 32 * i:0xC0 + 32 * i] = digest
        body += stored
    struct.pack_into("<I", hdr, 0x3C, bss)
    hdr[0x40:0x40 + len(build_id)] = build_id
    return bytes(hdr) + bytes(body)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2-extract-", ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.dest = self.root / "out"
        self.out = io.StringIO()

    def tearDown(self):
        self.tmp.cleanup()

    def ctx(self, **options):
        opts = types.SimpleNamespace(**{"cc": None, "plain_modules": None, "zbic_archive": None, **options})
        state = statefile.load(self.dest)
        return Context(self.dest, opts, Reporter(self.out, self.out), state=state)

    def write_dump(self, data: bytes, name="game.nup") -> Path:
        p = self.root / name
        p.write_bytes(data)
        return p


def base_and_update():
    return (NupWriter()
            .package(0, 0x10000, exefs={"main": b"base main", "sdk": b"base sdk"},
                     romfs={"a.txt": b"old a", "old.txt": b"gone in the update", "dir/b.bin": b"B" * 100})
            .package(1, 0x20000, exefs={"main": b"update main", "sdk": b"base sdk"},
                     romfs={"a.txt": b"new a", "dir/b.bin": b"B" * 100, "new.txt": b"added"}))


class ContainerTest(Base):
    def test_update_tree_replaces_the_base_tree(self):
        path = self.write_dump(base_and_update().build())
        with nup.Nup(path) as n:
            pkg, tree = n.program_tree("romfs")
            self.assertEqual(pkg.kind, 1)
            self.assertEqual(sorted(tree), ["a.txt", "dir/b.bin", "new.txt"])
            self.assertEqual(n.dropped("romfs"), ["old.txt"])
            base, update = n.program_packages()
            # the unchanged file's two records point at one extent
            self.assertEqual(n.tree(base, "romfs")["dir/b.bin"].offset, tree["dir/b.bin"].offset)

    def test_base_alone_is_the_tree(self):
        path = self.write_dump(NupWriter().package(0, 1, exefs={"main": b"m"}, romfs={"x": b"x"}).build())
        with nup.Nup(path) as n:
            self.assertEqual(sorted(n.program_tree("romfs")[1]), ["x"])
            self.assertEqual(n.dropped("romfs"), [])

    def test_damaged_table_of_contents_is_refused(self):
        data = bytearray(base_and_update().build())
        data[nup.SUPER_SIZE + nup.TOC_HEADER.size + 3] ^= 1
        with self.assertRaisesRegex(nup.NupError, "does not match its SHA-256"):
            nup.Nup(self.write_dump(bytes(data)))

    def test_truncated_container_is_refused(self):
        data = base_and_update().build()
        with self.assertRaisesRegex(nup.NupError, "truncated"):
            nup.Nup(self.write_dump(data[:-1]))

    def test_not_a_container_is_refused(self):
        with self.assertRaisesRegex(nup.NupError, "not a .nup"):
            nup.Nup(self.write_dump(b"PK\x03\x04" + bytes(200)))

    def test_names_that_leave_the_folder_are_refused(self):
        for bad in ("..", "a\\b", "c:d", "x\0y"):
            with self.subTest(name=bad):
                path = self.write_dump(NupWriter().package(0, 1, exefs={"main": b"m"}, romfs={bad: b"x"}).build())
                with nup.Nup(path) as n, self.assertRaisesRegex(nup.NupError, "unsafe name"):
                    n.program_tree("romfs")


class ExtractTest(Base):
    """The copy form (romfs.bin), asked for, since these containers are raw and would be read in place."""

    def extract(self, data: bytes):
        return extract.extract(self.write_dump(data), self.dest, Reporter(self.out, self.out), copy=True)

    def pack(self):
        header, rows = extract.read_index(self.dest / "romfs" / extract.INDEX)
        image = (self.dest / "romfs" / extract.IMAGE).read_bytes()
        return header, rows, {path: image[off:off + size] for path, off, size, _ in rows}

    def test_exefs_is_loose_and_romfs_is_packed_with_its_index(self):
        rec = self.extract(base_and_update().build())
        self.assertEqual((self.dest / extract.DUMP_EXEFS / "main").read_bytes(), b"update main")
        self.assertEqual(sorted(p.name for p in (self.dest / extract.DUMP_EXEFS).iterdir()), ["main", "sdk"])
        header, rows, files = self.pack()
        self.assertEqual(files, {"a.txt": b"new a", "dir/b.bin": b"B" * 100, "new.txt": b"added"})
        self.assertEqual(header["image_bytes"], rec["image_bytes"])
        self.assertEqual((rec["romfs_files"], rec["romfs_stored_files"], rec["romfs_dropped"]), (3, 3, 1))
        for path, off, size, sha in rows:
            self.assertEqual(off % extract.ALIGN, 0)
            self.assertEqual(sha, hashlib.sha256(files[path]).digest())
        self.assertEqual(rec["index_sha256"],
                         hashlib.sha256((self.dest / "romfs" / extract.INDEX).read_bytes()).hexdigest())
        self.assertEqual(extract.check_pack(self.dest / "romfs"), [])
        self.assertFalse(list(self.dest.rglob("*.part")))

    def test_one_extent_named_twice_is_stored_once(self):
        rec = self.extract(NupWriter().package(0, 1, exefs={"main": b"m"},
                                               romfs={"x/one": b"same" * 50, "y/two": b"same" * 50, "z": b"z"})
                           .build())
        _, rows, files = self.pack()
        offs = {p: off for p, off, _, _ in rows}
        self.assertEqual(offs["x/one"], offs["y/two"])
        self.assertEqual(files["y/two"], b"same" * 50)
        self.assertEqual((rec["romfs_files"], rec["romfs_stored_files"]), (3, 2))

    def test_index_is_sorted_by_utf8_bytes(self):
        names = ["b", "B", "a/z", "a-z", "été", "Z", "a.z"]
        self.extract(NupWriter().package(0, 1, exefs={"main": b"m"},
                                         romfs={n: n.encode() * 3 for n in names}).build())
        _, rows, _ = self.pack()
        got = [r[0] for r in rows]
        self.assertEqual(got, sorted(names, key=lambda s: s.encode("utf-8")))

    def test_small_window_streams_large_files_and_skips_gaps(self):
        romfs = {f"f{i:03}": bytes([i]) * (i * 7) for i in range(60)}
        romfs["big"] = bytes(range(256)) * 20
        data = (NupWriter().package(0, 1, exefs={"main": b"m"}, romfs={"f001": b"old", "gap": b"G" * 999})
                .package(1, 2, exefs={"main": b"m"}, romfs=romfs).build())
        with mock.patch.object(extract, "WINDOW", 64):
            self.extract(data)
        _, _, files = self.pack()
        self.assertEqual(files, romfs)

    @unittest.skipIf(zstd_compress() is None, "no zstd compressor in this Python (3.14's compression.zstd or "
                                              "zstandard)")
    def test_zstd_stored_files_are_decompressed(self):
        text = b"compressible " * 400
        self.extract(NupWriter().package(0, 1, exefs={"main": (b"main " * 99, nup.ZSTD)},
                                         romfs={"z.txt": (text, nup.ZSTD), "r.txt": b"raw"}).build())
        _, _, files = self.pack()
        self.assertEqual(files, {"z.txt": text, "r.txt": b"raw"})
        self.assertEqual((self.dest / extract.DUMP_EXEFS / "main").read_bytes(), b"main " * 99)

    def test_damaged_file_stops_the_step_and_leaves_no_output(self):
        data = NupWriter().package(0, 1, exefs={"main": b"m"},
                                   romfs={"a": b"a" * 40, "b": b"b" * 40}).build(sha_override={"b": bytes(32)})
        with self.assertRaisesRegex(nup.NupError, "b: its contents do not match the SHA-256"):
            self.extract(data)
        self.assertFalse((self.dest / "romfs" / extract.IMAGE).exists())
        self.assertFalse(list(self.dest.rglob("*.part")))

    def test_damaged_streamed_file_is_refused(self):
        """A file too large for one window, and an ExeFS file, go through the streaming path and its own check."""
        big = NupWriter().package(0, 1, exefs={"main": b"m"}, romfs={"big": bytes(500)}).build(
            sha_override={"big": bytes(32)})
        with mock.patch.object(extract, "WINDOW", 64), \
                self.assertRaisesRegex(nup.NupError, "big: its contents do not match the SHA-256"):
            self.extract(big)
        self.assertFalse(list(self.dest.rglob("*.part")))
        shutil.rmtree(self.dest)
        exe = NupWriter().package(0, 1, exefs={"main": b"m"}, romfs={"a": b"a"}).build(
            sha_override={"main": bytes(32)})
        with self.assertRaisesRegex(nup.NupError, "main: its contents do not match the SHA-256"):
            self.extract(exe)
        self.assertFalse((self.dest / extract.DUMP_EXEFS / "main").exists())
        self.assertFalse(list(self.dest.rglob("*.part")))

    def test_two_paths_on_one_extent_must_agree(self):
        data = NupWriter().package(0, 1, exefs={"main": b"m"},
                                   romfs={"a": b"same" * 9, "b": b"same" * 9}).build(sha_override={"b": bytes(32)})
        with self.assertRaisesRegex(nup.NupError, "shares its extent"):
            self.extract(data)

    def test_exefs_names_differing_by_case_are_refused(self):
        data = NupWriter().package(0, 1, exefs={"main": b"m", "MAIN": b"n"}, romfs={"a": b"a"}).build()
        with self.assertRaisesRegex(nup.NupError, "only by case"):
            self.extract(data)

    def test_check_pack_finds_a_damaged_file(self):
        self.extract(base_and_update().build())
        image = self.dest / "romfs" / extract.IMAGE
        _, rows, _ = self.pack()
        off = dict((p, o) for p, o, _, _ in rows)["new.txt"]
        data = bytearray(image.read_bytes())
        data[off] ^= 0xFF
        image.write_bytes(bytes(data))
        self.assertEqual(extract.check_pack(self.dest / "romfs"), ["new.txt"])

    def test_malformed_index_is_refused(self):
        self.extract(base_and_update().build())
        index = self.dest / "romfs" / extract.INDEX
        data = bytearray(index.read_bytes())
        data[0] ^= 1
        index.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "bad header"):
            extract.read_index(index)


class InPlaceTest(Base):
    """Which form the step chooses, and the in-place index: container offsets, the .nup as the image, no copy."""

    def extract(self, data: bytes, **kw):
        self.dump = self.write_dump(data)
        return extract.extract(self.dump, self.dest, Reporter(self.out, self.out), **kw)

    def test_all_raw_records_give_an_in_place_index_and_no_image(self):
        rec = self.extract(base_and_update().build())
        self.assertEqual(rec["romfs_form"], extract.IN_PLACE)
        self.assertFalse((self.dest / "romfs" / extract.IMAGE).exists())
        self.assertFalse(list(self.dest.rglob("*.part")))
        self.assertEqual((self.dest / extract.DUMP_EXEFS / "main").read_bytes(), b"update main")
        header, rows = extract.read_index(self.dest / "romfs" / extract.INDEX)
        dump = self.dump.read_bytes()
        self.assertEqual(header["image_bytes"], len(dump))                  # the image is the .nup
        self.assertEqual(rec["image_bytes"], len(dump))
        files = {path: dump[off:off + size] for path, off, size, _ in rows}
        self.assertEqual(files, {"a.txt": b"new a", "dir/b.bin": b"B" * 100, "new.txt": b"added"})
        with nup.Nup(self.dump) as n:
            tree = n.program_tree("romfs")[1]
        for path, off, size, sha in rows:
            self.assertEqual((off, size), (tree[path].offset, tree[path].size))   # container offsets
            self.assertEqual(sha, hashlib.sha256(files[path]).digest())
        self.assertEqual(rec["bytes"], len(b"update main") + len(b"base sdk") +
                         (self.dest / "romfs" / extract.INDEX).stat().st_size)  # nothing of the RomFS copied
        self.assertEqual(extract.check_pack(self.dest / "romfs", image=self.dump), [])
        self.assertIn("RomFS read in place from game.nup: every one of its 3 files is stored raw", self.out.getvalue())

    def test_shared_and_empty_files_in_place(self):
        self.extract(NupWriter().package(0, 1, exefs={"main": b"m"},
                                         romfs={"x/one": b"same" * 50, "y/two": b"same" * 50, "e": b""}).build())
        header, rows = extract.read_index(self.dest / "romfs" / extract.INDEX)
        offs = {p: (off, size) for p, off, size, _ in rows}
        self.assertEqual(offs["x/one"], offs["y/two"])
        self.assertEqual(offs["e"], (0, 0))
        self.assertEqual(header["stored"], 2)
        self.assertEqual(extract.check_pack(self.dest / "romfs", image=self.dump), [])

    @unittest.skipIf(zstd_compress() is None, "no zstd compressor in this Python (3.14's compression.zstd or "
                                              "zstandard)")
    def test_one_compressed_record_gives_the_copy(self):
        text = b"compressible " * 400
        rec = self.extract(NupWriter().package(0, 1, exefs={"main": b"m"},
                                               romfs={"r.txt": b"raw", "s/z.txt": (text, nup.ZSTD)}).build())
        self.assertEqual(rec["romfs_form"], extract.COPY)
        image = (self.dest / "romfs" / extract.IMAGE).read_bytes()
        self.assertEqual(rec["image_bytes"], len(image))
        _, rows = extract.read_index(self.dest / "romfs" / extract.INDEX)
        self.assertEqual({p: image[off:off + size] for p, off, size, _ in rows}, {"r.txt": b"raw", "s/z.txt": text})
        self.assertIn("RomFS copied to romfs/romfs.bin: 1 of its 2 files are stored zstd-compressed (s/z.txt first)",
                      self.out.getvalue())

    @unittest.skipIf(zstd_compress() is None, "no zstd compressor in this Python (3.14's compression.zstd or "
                                              "zstandard)")
    def test_a_compressed_exefs_file_leaves_the_romfs_in_place(self):
        rec = self.extract(NupWriter().package(0, 1, exefs={"main": (b"main " * 99, nup.ZSTD)},
                                               romfs={"r.txt": b"raw"}).build())
        self.assertEqual(rec["romfs_form"], extract.IN_PLACE)
        self.assertEqual((self.dest / extract.DUMP_EXEFS / "main").read_bytes(), b"main " * 99)

    def test_in_place_deletes_a_copy_an_earlier_run_wrote(self):
        data = base_and_update().build()
        self.assertEqual(self.extract(data, copy=True)["romfs_form"], extract.COPY)
        self.assertTrue((self.dest / "romfs" / extract.IMAGE).is_file())
        self.extract(data)
        self.assertFalse((self.dest / "romfs" / extract.IMAGE).exists())
        self.assertIn("deleted the copy romfs/romfs.bin an earlier run wrote", self.out.getvalue())

    def test_check_pack_finds_a_file_changed_in_the_dump(self):
        self.extract(base_and_update().build())
        _, rows = extract.read_index(self.dest / "romfs" / extract.INDEX)
        off = {p: o for p, o, _, _ in rows}["new.txt"]
        data = bytearray(self.dump.read_bytes())
        data[off] ^= 0xFF
        self.dump.write_bytes(bytes(data))
        self.assertEqual(extract.check_pack(self.dest / "romfs", image=self.dump), ["new.txt"])


class ModulesTest(Base):
    def setUp(self):
        super().setUp()
        (self.dest / extract.DUMP_EXEFS).mkdir(parents=True)
        self.decoded = []

    def factory(self, ctx):
        self.decoded.append("built")
        return fake_unzbic

    def put(self, name, data):
        (self.dest / extract.DUMP_EXEFS / name).write_bytes(data)

    def run_modules(self, **options):
        return modules.Modules(self.factory).run(self.ctx(**options))

    def table(self):
        return {m["name"]: m for m in json.loads((self.dest / "exefs" / "modules.json").read_text())["modules"]}

    def test_lz4_and_zbic_modules_decode_to_plain_ones(self):
        segs = segments()
        self.put("main", nso(segs, "zbic", name=b"Owl\0"))
        self.put("sdk", nso(segments(3), "lz4", build_id=b"\x22" * 20, name=b"nnSdk\0"))
        self.put("rtld", nso(segments(5), "lz4"))
        rec = self.run_modules()
        self.assertEqual(self.decoded, ["built"])
        self.assertEqual(sorted(rec["files"]), ["main", "sdk"])                  # rtld is not loaded
        plain = (self.dest / "exefs" / "main").read_bytes()
        self.assertEqual(plain, nso(segs, "plain"))                               # the name before text is dropped
        self.assertEqual(struct.unpack_from("<I", plain, 0xC)[0], 0x38)            # hash bits kept
        t = self.table()
        self.assertEqual([m["name"] for m in json.loads((self.dest / "exefs" / "modules.json").read_text())
                          ["modules"]], ["main", "sdk"])                           # load order
        m = t["main"]
        self.assertEqual((m["text"], m["rodata"], m["data"]),
                         ({"offset": 0, "size": 0x1000}, {"offset": 0x2000, "size": 0x800},
                          {"offset": 0x3000, "size": 0x800}))
        self.assertEqual((m["bss"], m["image_size"]), ({"offset": 0x3800, "size": 0x100}, 0x4000))
        self.assertEqual((m["mod0"], m["dynamic"], m["eh_frame_hdr"]), (0x10, 0x3000, {"offset": 0x2000, "size": 16}))
        self.assertEqual((m["build_id"], m["source"]["codec"], m["source"]["from"]), ("11" * 20, "zbic", "decoded"))
        self.assertEqual(m["sha256"], hashlib.sha256(plain).hexdigest())
        self.assertEqual(t["sdk"]["source"]["codec"], "lz4")
        self.assertEqual((self.dest / "exefs" / "sdk").read_bytes()[modules.HDR:modules.HDR + 0x1000],
                         segments(3)[0][1])

    def test_no_zbic_module_means_no_decoder(self):
        self.put("main", nso(segments(), "lz4"))
        self.run_modules()
        self.assertEqual(self.decoded, [])

    def test_a_segment_that_misses_its_header_hash_is_refused(self):
        for codec in ("lz4", "zbic"):
            with self.subTest(codec=codec):
                self.put("main", nso(segments(), codec, bad_hash=1))
                with self.assertRaisesRegex(PrepareError, "rodata segment does not match the SHA-256 in its header"):
                    self.run_modules()

    def test_a_frame_the_decoder_rejects_is_refused(self):
        data = bytearray(nso(segments(), "zbic"))
        off = struct.unpack_from("<I", data, 0x30)[0]
        data[off:off + 4] = b"XXXX"
        self.put("main", bytes(data))
        with self.assertRaisesRegex(PrepareError, "data segment does not decode"):
            self.run_modules()

    def test_given_plain_module_is_used_without_the_decoder(self):
        segs = segments()
        self.put("main", nso(segs, "zbic"))
        given = self.root / "given"
        given.mkdir()
        (given / "main").write_bytes(nso(segs, "plain", name=b"x" * 7))
        self.run_modules(plain_modules=given)
        self.assertEqual(self.decoded, [])
        self.assertEqual((self.dest / "exefs" / "main").read_bytes(), nso(segs, "plain"))
        self.assertEqual(self.table()["main"]["source"]["from"], "given")

    def test_given_module_of_another_version_is_refused(self):
        segs = segments()
        self.put("main", nso(segs, "zbic"))
        given = self.root / "given"
        given.mkdir()
        cases = (("another version of the game", nso(segs, "plain", build_id=b"\x99" * 20)),
                 ("text segment of the given module", nso([(0, b"\x01" + segs[0][1][1:])] + segs[1:], "plain",
                                                          hashes=False)),
                 ("lays out its segments unlike", nso(segs, "plain", bss=0x200)),
                 ("ZBIC-compressed", nso(segs, "zbic")))
        for why, data in cases:
            with self.subTest(why=why):
                (given / "main").write_bytes(data)
                with self.assertRaisesRegex(PrepareError, why):
                    self.run_modules(plain_modules=given)

    def test_missing_main_is_refused(self):
        self.put("sdk", nso(segments(), "lz4"))
        with self.assertRaisesRegex(PrepareError, "no main module"):
            self.run_modules()


class FakeCheck(Step):
    """The check step's one duty that the later steps rely on: ctx.facts["dump"]."""
    id = "check"
    always = True

    def __init__(self, dump: Path):
        self.dump = dump

    def run(self, ctx):
        data = self.dump.read_bytes()
        ctx.facts["dump"] = {"path": str(self.dump), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return {}


class StepsTest(Base):
    """extract then modules through the runner: recorded, kept, rerun on a changed input."""

    def setUp(self):
        super().setUp()
        segs = segments()
        self.dump = self.write_dump(NupWriter().package(
            0, 1, exefs={"main": nso(segs, "zbic"), "sdk": nso(segments(2), "lz4", build_id=b"\x33" * 20),
                         "main.npdm": b"meta"},
            romfs={"System/RegionLangMask.txt": b"1\n2\n111\n", "Data/x.bin": bytes(300)}).build())
        self.builds = []

    def run_all(self, **options):
        out = io.StringIO()
        opts = types.SimpleNamespace(**{"cc": None, "plain_modules": None, "zbic_archive": None, **options})
        ctx = Context(self.dest, opts, Reporter(out, out), state=statefile.load(self.dest))
        steps = [FakeCheck(self.dump), extract.Extract(),
                 modules.Modules(lambda c: self.builds.append(1) or fake_unzbic)]
        result = run_steps(steps, ctx)
        return result, out.getvalue()

    def test_first_run_then_kept_then_rerun_when_an_output_goes_missing(self):
        result, text = self.run_all()
        self.assertEqual(result, "ok", text)
        state = statefile.load(self.dest)["steps"]
        self.assertEqual(set(state), {"extract", "modules"})
        self.assertEqual(state["extract"]["romfs_files"], 2)
        self.assertEqual(state["extract"]["romfs_form"], extract.IN_PLACE)
        self.assertEqual(state["extract"]["nup"], {"path": str(self.dump.resolve()), "size": self.dump.stat().st_size,
                                                   "sha256": hashlib.sha256(self.dump.read_bytes()).hexdigest()})
        self.assertEqual(sorted(state["modules"]["files"]), ["main", "main.npdm", "sdk"])
        self.assertEqual((self.dest / "exefs" / "main.npdm").read_bytes(), b"meta")
        result, text = self.run_all()
        self.assertIn("extract: kept", text)
        self.assertIn("modules: kept", text)
        (self.dest / "romfs" / extract.INDEX).unlink()
        result, text = self.run_all()
        self.assertIn("extract: started (its output is missing)", text)
        self.assertIn("modules: kept", text)                     # same inputs: its key did not change
        self.assertEqual(result, "ok")

    def test_a_dump_found_elsewhere_is_indexed_again_at_its_new_place(self):
        """The same dump given at another place, the old copy still there: the index and the record follow it."""
        self.run_all()
        moved = self.dump.with_name("moved.nup")
        shutil.copyfile(self.dump, moved)
        self.dump = moved
        result, text = self.run_all()
        self.assertEqual(result, "ok", text)
        self.assertIn("extract: started (its output is missing)", text)
        self.assertEqual(statefile.load(self.dest)["steps"]["extract"]["nup"]["path"], str(moved.resolve()))
        result, text = self.run_all()
        self.assertIn("extract: kept", text)

    def test_the_copy_form_is_kept_and_rerun_on_its_image(self):
        with mock.patch.object(extract, "romfs_form", lambda tree: (extract.COPY, "a test asks for it")):
            result, text = self.run_all()
            self.assertEqual(result, "ok", text)
            state = statefile.load(self.dest)["steps"]["extract"]
            self.assertEqual(state["romfs_form"], extract.COPY)
            self.assertNotIn("nup", state)
            self.assertIn("RomFS copied to romfs/romfs.bin: a test asks for it", text)
            self.assertIn("extract: kept", self.run_all()[1])
            (self.dest / "romfs" / extract.IMAGE).unlink()
            self.assertIn("extract: started (its output is missing)", self.run_all()[1])

    def test_another_dump_reruns_both_and_given_modules_rerun_only_the_decode(self):
        self.run_all()
        given = self.root / "given"
        given.mkdir()
        (given / "main").write_bytes(nso(segments(), "plain"))
        result, text = self.run_all(plain_modules=given)
        self.assertIn("extract: kept", text)
        self.assertIn("modules: started (its inputs changed)", text)
        self.assertEqual(len(self.builds), 1)                    # the given main needs no decoder
        self.dump.write_bytes(self.dump.read_bytes()[:-1] + b"\0")    # same size, other contents
        result, text = self.run_all(plain_modules=given)
        self.assertIn("extract: started (its inputs changed)", text)

    def test_verify_catches_a_damaged_pack(self):
        with mock.patch.object(extract, "romfs_form", lambda tree: (extract.COPY, "a test asks for it")):
            self.run_all()
        image = self.dest / "romfs" / extract.IMAGE
        data = bytearray(image.read_bytes())
        data[-1] ^= 1
        image.write_bytes(bytes(data))
        ctx = self.ctx()
        rec = statefile.load(self.dest)["steps"]["extract"]
        self.assertTrue(extract.Extract().current(ctx, rec))   # same size: the cheap check cannot see it
        self.assertFalse(extract.Extract().verify(ctx, rec))

    def test_verify_reads_every_file_from_the_dump_in_place(self):
        self.run_all()
        ctx = self.ctx()
        rec = statefile.load(self.dest)["steps"]["extract"]
        self.assertTrue(extract.Extract().verify(ctx, rec))
        _, rows = extract.read_index(self.dest / "romfs" / extract.INDEX)
        off = {p: o for p, o, _, _ in rows}["Data/x.bin"]
        data = bytearray(self.dump.read_bytes())
        data[off + 7] ^= 1
        self.dump.write_bytes(bytes(data))
        self.assertTrue(extract.Extract().current(ctx, rec))   # same size and place: the cheap check cannot see it
        self.assertFalse(extract.Extract().verify(ctx, rec))

    def test_romfs_index_and_image_are_checked_at_start_and_key_the_link(self):
        """The RomFS reaches oot2.exe's start-up checks and the link step's key (the build and link step's program.py): the copy."""
        from prepare import program
        ctx = self.ctx()
        self.assertEqual(program.game_files(ctx), [])
        self.assertEqual(program.romfs_key(ctx), ["romfs=none"])
        with mock.patch.object(extract, "romfs_form", lambda tree: (extract.COPY, "a test asks for it")):
            self.assertEqual(self.run_all()[0], "ok")
        ctx = self.ctx()
        files = program.game_files(ctx)
        self.assertEqual([(f.what, f.path.name, f.hashed, f.sampled) for f in files],
                         [("RomFS index", extract.INDEX, True, False), ("RomFS image", extract.IMAGE, False, False)])
        before = runner.fingerprint(*program.romfs_key(ctx))
        image = self.dest / "romfs" / extract.IMAGE
        image.write_bytes(image.read_bytes() + b"x")
        self.assertNotEqual(runner.fingerprint(*program.romfs_key(ctx)), before)       # a longer image
        image.write_bytes(image.read_bytes()[:-1])
        self.assertEqual(runner.fingerprint(*program.romfs_key(ctx)), before)
        index = self.dest / "romfs" / extract.INDEX
        data = bytearray(index.read_bytes())
        data[-2] ^= 1                                                                    # a name in the index
        index.write_bytes(bytes(data))
        self.assertNotEqual(runner.fingerprint(*program.romfs_key(ctx)), before)

    def test_the_dump_read_in_place_is_checked_at_start_and_keys_the_link(self):
        """In place, the image the program checks is the dump at its recorded path, by size and sampled SHA-256."""
        from prepare import program
        self.assertEqual(self.run_all()[0], "ok")
        ctx = self.ctx()
        files = program.game_files(ctx)
        self.assertEqual([(f.what, f.path, f.hashed, f.sampled) for f in files],
                         [("RomFS index", (self.dest / "romfs" / extract.INDEX).resolve(), True, False),
                          ("RomFS image", self.dump.resolve(), False, True)])
        self.assertFalse((self.dest / "romfs" / extract.IMAGE).exists())
        before = runner.fingerprint(*program.romfs_key(ctx))
        original = self.dump.read_bytes()
        changed = bytearray(original)
        changed[len(changed) // 2] ^= 1                          # same size; this small file is sampled whole
        self.dump.write_bytes(bytes(changed))
        self.assertNotEqual(runner.fingerprint(*program.romfs_key(ctx)), before)
        self.dump.write_bytes(original)
        self.assertEqual(runner.fingerprint(*program.romfs_key(ctx)), before)
        self.dump.rename(self.dump.with_name("away.nup"))
        self.assertNotEqual(runner.fingerprint(*program.romfs_key(ctx)), before)       # the program would not start


# ---------------------------------------------------------------------------------------------------------------
# The decoder fetch: contents hash, refusal before writing, and a stand-in library built and called.

STAND_IN = {
    "lib/zstd.h": b"#include <stddef.h>\n"
                  b"size_t ZSTD_decompress(void *dst, size_t cap, const void *src, size_t n);\n"
                  b"unsigned ZSTD_isError(size_t code);\nconst char *ZSTD_getErrorName(size_t code);\n",
    "lib/zstd_errors.h": b"/* stand-in */\n",
    "LICENSE": b"stand-in licence\n",
    "README.md": b"not unpacked\n",
    "lib/decompress/zstd_decompress.c":
        b"#include <string.h>\n#include \"zstd.h\"\n"
        b"size_t ZSTD_decompress(void *dst, size_t cap, const void *src, size_t n)\n"
        b"{ if (n > cap) return (size_t)-1; memcpy(dst, src, n); return n; }\n"
        b"unsigned ZSTD_isError(size_t code) { return code == (size_t)-1; }\n"
        b"const char *ZSTD_getErrorName(size_t code) { (void)code; return \"Destination buffer is too small\"; }\n",
}
for _s in zbic.SOURCES:
    STAND_IN.setdefault(_s, b"typedef int stand_in_unused;\n")


def archive(files: dict[str, bytes], top="zstd-0000", mtime=0) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for path, data in files.items():
            info = tarfile.TarInfo(f"{top}/{path}")
            info.size, info.mtime = len(data), mtime
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo(f"{top}/tests/link")
        link.type, link.linkname = tarfile.SYMTYPE, "../../outside"
        tar.addfile(link)
    return gzip.compress(raw.getvalue(), mtime=mtime)


def find_compiler():
    found = shutil.which(os.environ.get("OOT2_CC", "clang"))
    if not found and os.environ.get("OOT2_REQUIRE_CLANG"):
        raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
    return found


class ZbicFetchTest(Base):
    def test_contents_hash_ignores_the_archive_bytes_and_top_folder(self):
        a, _ = zbic.contents_sha256(archive(STAND_IN, "zstd-02a8f10", mtime=1))
        b, _ = zbic.contents_sha256(archive(STAND_IN, f"zstd-{zbic.COMMIT}", mtime=2))
        self.assertEqual(a, b)
        changed = dict(STAND_IN, LICENSE=b"other\n")
        self.assertNotEqual(zbic.contents_sha256(archive(changed))[0], a)

    def test_the_pinned_hash_is_a_sha256(self):
        self.assertRegex(zbic.CONTENTS_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(zbic.COMMIT, r"^[0-9a-f]{40}$")
        self.assertIn(zbic.COMMIT, zbic.URL)

    def test_an_archive_that_is_not_the_pinned_one_writes_nothing(self):
        path = self.root / "zstd.tar.gz"
        path.write_bytes(archive(STAND_IN))
        work = self.root / "zbic"
        with self.assertRaisesRegex(PrepareError, "is not the pinned one"):
            zbic.prepare_decoder("cc", work, Reporter(self.out, self.out), archive=path)
        self.assertEqual(list(work.iterdir()), [])

    def test_not_an_archive_is_refused(self):
        with self.assertRaisesRegex(PrepareError, "cannot be read"):
            zbic.contents_sha256(b"<html>rate limited</html>")

    def test_unpack_writes_only_the_kept_paths_and_refuses_unsafe_ones(self):
        out = self.root / "src"
        zbic.unpack(zbic.contents_sha256(archive(STAND_IN))[1], out)
        self.assertTrue((out / "lib" / "zstd.h").is_file())
        self.assertFalse((out / "README.md").exists())
        self.assertFalse((out / "tests").exists())
        with self.assertRaisesRegex(PrepareError, "unsafe path"):
            zbic.unpack({"lib/common/../../evil.c": b""}, out)

    def test_download_failure_says_what_to_do(self):
        with self.assertRaisesRegex(PrepareError, "--zbic-archive FILE"):
            zbic.download("http://127.0.0.1:9/none.tar.gz")

    def test_stand_in_library_builds_loads_and_decodes(self):
        cc = find_compiler()
        if not cc:
            self.skipTest("no clang on PATH (set OOT2_CC)")
        data = archive(STAND_IN)
        path = self.root / "zstd.tar.gz"
        path.write_bytes(data)
        work = self.root / "zbic"
        expected = zbic.contents_sha256(data)[0]
        dec = zbic.prepare_decoder(cc, work, Reporter(self.out, self.out), archive=path, expected=expected)
        self.assertEqual(dec(b"segment bytes", 13), b"segment bytes")
        with self.assertRaisesRegex(ValueError, "too small"):
            dec(b"segment bytes", 5)
        with self.assertRaisesRegex(ValueError, "the header says 20"):
            dec(b"segment bytes", 20)
        again = zbic.prepare_decoder(cc, work, Reporter(self.out, self.out), archive=path, expected=expected)
        self.assertIn("already built", self.out.getvalue())
        self.assertEqual(again(b"abc", 3), b"abc")


if __name__ == "__main__":
    unittest.main()
