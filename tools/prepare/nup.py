# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Read the player's .nup container: its superblock, table of contents and packages (architecture D2).

A .nup holds the ExeFS and RomFS trees of a base program and its updates, each file stored raw or as one zstd frame and
listed with its size and SHA-256. Only the superblock, the table of contents and the meta blocks are read here; the
extraction step (extract.py) streams the file contents. Nothing here decrypts anything: the container holds plain
files.

    superblock (0x80)  "NUP\\x01", header size, version, TOC offset and size, TOC SHA-256, blob offset, file size,
                       package count, flags
    TOC header (0x30)  magic "TOC\\x01", package count, directory count, file count, string pool offset and size
    packages   (0x40)  title id, root, ExeFS and RomFS directory, version, kind (base, update, add-on), flags, meta
    directories (0x18) name, first child slot, child count
    files      (0x50)  name, size, extent offset and size, SHA-256, codec (raw or zstd)
    child slots (u32)  a directory's children: an index below the directory count is a directory, else a file
    string pool        UTF-8 names

The tree the game reads (Nup.program_tree) is the newest update's whole tree when an update is present, else the
base's. An update package lists every file of the patched tree, the unchanged ones pointing at the base's extents, so
it is complete on its own; a file only the base lists is one the update removed or renamed, and the console does not
see it either.

Adapted from oot-pc-recomp tools/extract_nup.py at f4656d3 (no licence header: that repository's GPL-3.0-or-later),
an independent reader of the layout of Eglantine's core/file_sys/nup_container.cpp. Changed: the reader is separate
from the extraction; every check raises NupError with the record named; the program's tree is chosen across
packages (the newest update's) instead of extracting one package at a time.
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path

SUPER = struct.Struct("<4sHHQQ32sQQII")      # 0x50 used of the 0x80 superblock
SUPER_SIZE = 0x80
TOC_HEADER = struct.Struct("<IIQQQQQ")       # 0x30
PACKAGE = struct.Struct("<QIIIIBBHIQI")      # 0x2c used of 0x40
PACKAGE_SIZE = 0x40
DIRECTORY = struct.Struct("<IIIIQ")          # 0x18
FILE = struct.Struct("<IIQQQ32sBBH12s")      # 0x50
MAGIC = b"NUP\x01"
TOC_MAGIC = 0x01434F54                       # "TOC\x01"
NO_DIR = 0xFFFFFFFF
RAW, ZSTD = 0, 1
KINDS = {0: "base", 1: "update", 2: "add-on"}
META = struct.Struct("<HHI")
META_DISPLAY_VERSION, META_TITLE_NAME = 4, 5


class NupError(Exception):
    """The container is not one this reader accepts, or is damaged. The message says which record."""


@dataclass(frozen=True)
class Entry:
    """One file record of the table of contents."""
    index: int
    name: str
    size: int          # bytes once decoded
    offset: int        # extent in the container
    stored: int        # extent size: equal to size when raw
    sha256: bytes      # of the decoded bytes
    codec: int         # RAW or ZSTD


@dataclass
class Package:
    index: int
    title_id: int
    root: int
    exefs: int
    romfs: int
    version: int
    kind: int
    flags: int
    meta_offset: int
    meta_size: int
    title: str = ""
    display_version: str = ""

    @property
    def kind_name(self) -> str:
        return KINDS.get(self.kind, f"kind {self.kind}")


# Names Windows keeps for devices: a file called any of these, with or without an extension (CON.txt), is the device.
# COM and LPT also take the superscript digits. A path with one would write to a device or fail in a way that names no
# file, on the player's machine, from a container someone handed over.
DEVICE_NAMES = frozenset({"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} |
                         {f"{base}{d}" for base in ("COM", "LPT") for d in "123456789" + chr(0xb9) + chr(0xb2) + chr(0xb3)})
FORBIDDEN_CHARS = set("/" + chr(92) + ":" + chr(0) + '*?"<>|')


def safe_name(name: str) -> str:
    """A container is a file someone handed over: refuse a name that could leave the output folder or that Windows
    would not take as written. Refused: an empty name, "." and "..", the separators, a drive colon, the wildcards
    * and ?, the characters " < > | that Windows file names cannot hold, control characters, a trailing space or dot
    (Windows drops them, so two names become one), and a device name with or without an extension (CON, NUL, COM1,
    LPT1 and the rest of DEVICE_NAMES)."""
    if not name or name in (".", ".."):
        raise NupError(f"unsafe name in the container: {name!r}")
    bad = sorted({c for c in name if c in FORBIDDEN_CHARS or ord(c) < 32})
    if bad:
        raise NupError(f"unsafe name in the container: {name!r} holds {', '.join(repr(c) for c in bad)}")
    if name[-1] in " .":
        raise NupError(f"unsafe name in the container: {name!r} ends with a space or a dot, which Windows removes")
    if name.split(".", 1)[0].rstrip(" ").upper() in DEVICE_NAMES:
        raise NupError(f"unsafe name in the container: {name!r} is the name of a Windows device")
    return name


def link_in_path(path: Path, base: Path) -> Path | None:
    """The first link or reparse point (symbolic link, junction, mount point) on the way from base down to path, or None.
    base itself is not asked: the player may keep the whole folder on a junction they made. What is below it was made
    by prepare, so a link there came from somewhere else and would carry a write out of the folder."""
    base, path = Path(base), Path(path)
    try:
        parts = path.relative_to(base).parts
    except ValueError:
        return path
    here = base
    for part in parts:
        here = here / part
        try:
            st = os.lstat(here)
        except FileNotFoundError:
            return None
        except OSError:
            return here
        if os.path.islink(here) or getattr(st, "st_file_attributes", 0) & 0x400 or                 (hasattr(os.path, "isjunction") and os.path.isjunction(here)):
            return here
    return None


def refuse_links(path: Path, base: Path) -> None:
    """NupError naming the link when one lies between base and path (link_in_path)."""
    found = link_in_path(path, base)
    if found is not None:
        raise NupError(f"{found} is a link or a reparse point on the way to {path}; prepare writes only into real "
                       f"folders of its own output folder. Remove it, or choose another output folder.")


class Nup:
    """A parsed container. Open with Nup(path); close() releases the file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.f = open(self.path, "rb")
        try:
            self._read()
        except BaseException:
            self.f.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self.f.close()

    def _read(self) -> None:
        size = os.fstat(self.f.fileno()).st_size
        sb = self.f.read(SUPER_SIZE)
        if len(sb) < SUPER_SIZE or sb[:4] != MAGIC:
            raise NupError("not a .nup container (no \"NUP\\x01\" magic at the start)")
        (_, header_size, self.version, self.toc_offset, self.toc_size, toc_sha, self.blob_offset, self.file_size,
         package_count, flags) = SUPER.unpack_from(sb)
        if header_size < SUPER_SIZE or flags != 0:
            raise NupError(f"unsupported superblock (header size {header_size}, flags {flags:#x})")
        if (self.file_size != size or self.toc_offset < SUPER_SIZE or
                self.blob_offset < self.toc_offset + self.toc_size or self.blob_offset % 16 or
                self.blob_offset > self.file_size):
            raise NupError(f"the superblock disagrees with the file ({size:,} bytes; it says {self.file_size:,}): "
                           "the file is truncated or was modified")
        self.f.seek(self.toc_offset)
        toc = self.f.read(self.toc_size)
        if len(toc) != self.toc_size or hashlib.sha256(toc).digest() != toc_sha:
            raise NupError("the table of contents does not match its SHA-256: the file is damaged or was modified")
        self._parse_toc(toc, package_count)
        for pkg in self.packages:
            self._parse_meta(pkg)

    def _parse_toc(self, toc: bytes, package_count: int) -> None:
        if len(toc) < TOC_HEADER.size:
            raise NupError("the table of contents is too short")
        magic, count, ndirs, nfiles, pool_off, pool_size, _ = TOC_HEADER.unpack_from(toc)
        if magic != TOC_MAGIC or count == 0 or count != package_count:
            raise NupError("bad table of contents header")
        needed = TOC_HEADER.size + count * PACKAGE_SIZE + ndirs * DIRECTORY.size + nfiles * FILE.size
        if pool_off < needed or pool_off + pool_size > len(toc) or (pool_off - needed) % 4:
            raise NupError("bad table of contents layout")
        pool = toc[pool_off:pool_off + pool_size]

        def name_at(off: int, length: int, what: str) -> str:
            if off + length > pool_size:
                raise NupError(f"the name of {what} lies outside the string pool")
            try:
                return pool[off:off + length].decode("utf-8")
            except UnicodeDecodeError:
                raise NupError(f"the name of {what} is not UTF-8") from None

        pos = TOC_HEADER.size
        self.packages: list[Package] = []
        for i in range(count):
            (title_id, root, exefs, romfs, version, kind, pflags, _, _, meta_off, meta_size) = \
                PACKAGE.unpack_from(toc, pos)
            pos += PACKAGE_SIZE
            if kind > 2 or (pflags & ~1) or root >= ndirs or \
                    (exefs != NO_DIR and exefs >= ndirs) or (romfs != NO_DIR and romfs >= ndirs):
                raise NupError(f"bad package record {i}")
            if not (meta_size and meta_off and meta_off + meta_size <= self.file_size):
                meta_off = meta_size = 0
            self.packages.append(Package(i, title_id, root, exefs, romfs, version, kind, pflags, meta_off, meta_size))

        slots = (pool_off - needed) // 4
        self.dirs: list[tuple[str, int, int]] = []          # (name, first child slot, child count)
        for i in range(ndirs):
            name_off, name_len, first, n, _ = DIRECTORY.unpack_from(toc, pos)
            pos += DIRECTORY.size
            if first + n > slots:
                raise NupError(f"bad directory record {i}")
            self.dirs.append((name_at(name_off, name_len, f"directory {i}"), first, n))

        self.files: list[Entry] = []
        for i in range(nfiles):
            name_off, name_len, size, ext_off, ext_size, sha, codec, codec_flags, _, _ = FILE.unpack_from(toc, pos)
            pos += FILE.size
            if codec > ZSTD or codec_flags != 0 or (size == 0 and (ext_off or ext_size)) or \
                    (size != 0 and (ext_size == 0 or ext_off < self.blob_offset or ext_off % 16 or
                                    ext_off + ext_size > self.file_size)) or \
                    (codec == RAW and ext_size != size):
                raise NupError(f"bad file record {i}")
            self.files.append(Entry(i, name_at(name_off, name_len, f"file {i}"), size, ext_off, ext_size, sha, codec))

        self.children = struct.unpack_from(f"<{slots}I", toc, needed)
        limit = ndirs + nfiles
        for c in self.children:
            if c >= limit:
                raise NupError("a directory names a child that does not exist")

    def _parse_meta(self, pkg: Package) -> None:
        if not pkg.meta_offset:
            return
        self.f.seek(pkg.meta_offset)
        block = self.f.read(pkg.meta_size)
        pos = 0
        while pos + META.size <= len(block):
            typ, _, size = META.unpack_from(block, pos)
            pos += META.size
            if pos + size > len(block):
                raise NupError(f"bad meta block in package {pkg.index}")
            payload = block[pos:pos + size]
            pos += size
            if typ == META_TITLE_NAME:
                pkg.title = payload.decode("utf-8", "replace")
            elif typ == META_DISPLAY_VERSION:
                pkg.display_version = payload.decode("utf-8", "replace")

    def walk(self, dir_index: int, prefix: tuple[str, ...] = (), depth: int = 0):
        """Yield (path parts, Entry) for every file under a directory."""
        if depth > 64:
            raise NupError("directories nest more than 64 deep (a loop in the table of contents?)")
        _, first, n = self.dirs[dir_index]
        for slot in range(first, first + n):
            child = self.children[slot]
            if child >= len(self.dirs):
                entry = self.files[child - len(self.dirs)]
                yield prefix + (safe_name(entry.name),), entry
            else:
                yield from self.walk(child, prefix + (safe_name(self.dirs[child][0]),), depth + 1)

    def tree(self, pkg: Package, which: str) -> dict[str, Entry]:
        """The files of one package's "exefs" or "romfs" tree by path ("a/b/c", no leading slash)."""
        root = pkg.exefs if which == "exefs" else pkg.romfs
        out: dict[str, Entry] = {}
        if root == NO_DIR:
            return out
        for parts, entry in self.walk(root):
            path = "/".join(parts)
            if path in out:
                raise NupError(f"package {pkg.index} lists {which}/{path} twice")
            out[path] = entry
        return out

    def program_packages(self) -> list[Package]:
        """The base and its updates, oldest first: the packages whose files the game reads. Add-ons are left out."""
        base = [p for p in self.packages if p.kind == 0]
        if len(base) != 1:
            raise NupError(f"the container holds {len(base)} base packages; one is needed")
        updates = sorted((p for p in self.packages if p.kind == 1), key=lambda p: p.version)
        return base + updates

    def program_tree(self, which: str) -> tuple[Package, dict[str, Entry]]:
        """The "exefs" or "romfs" tree the game reads, and the package it comes from: the newest update that has the
        tree, else the base. An update's tree is the whole patched tree (it lists the unchanged files too, pointing
        at the base's extents), so it replaces the base's tree rather than being laid over it: a file only the base
        lists was removed or renamed by the update."""
        for pkg in reversed(self.program_packages()):
            if (pkg.exefs if which == "exefs" else pkg.romfs) != NO_DIR:
                return pkg, self.tree(pkg, which)
        return self.program_packages()[0], {}

    def dropped(self, which: str) -> list[str]:
        """Paths an older package lists that the tree of program_tree does not: removed or renamed by the update."""
        winner, tree = self.program_tree(which)
        old: set[str] = set()
        for pkg in self.program_packages():
            if pkg.index != winner.index:
                old.update(self.tree(pkg, which))
        return sorted(old - set(tree))
