# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The extract step of `oot2 prepare` (architecture D2): the player's .nup to loose ExeFS files and a RomFS index, in
one pass over the container.

Input: the dump the check step identified (ctx.facts["dump"]: path, size, SHA-256).
Output, under the output folder:

    exefs-dump/<name>        each ExeFS file of the game's tree as the dump stores it (main, sdk and subsdk0
                             compressed, rtld, main.npdm), loose; the modules step reads them and writes the
                             plain modules the program loads to exefs/
    romfs/romfs-index.bin    the index: every RomFS path with its offset and size in the image and its SHA-256
    romfs/romfs.bin          the image, in the copy form only: the RomFS files, each stored once, aligned to 16 bytes

The RomFS takes one of two forms (the step's record says which, in "romfs_form"):

- In place (IN_PLACE), when every file of the tree is stored raw in the container: the image is the player's .nup
  itself, and the index holds container offsets. Nothing is copied: the step reads the table of contents and writes
  the index (seconds, and about 20 MiB for the whole game), and records the .nup's path, size and SHA-256. oot2.exe
  reads the files from the .nup where the player keeps it, and checks it at each start (tools/build_game.py
  sample_sha256): a .nup moved, replaced or changed stops the start and asks for prepare again. A copy an earlier run
  left in romfs/ is deleted. The files are not hashed here, since that means reading the whole dump; --verify hashes
  every file through the index.
- Copy (COPY), the fallback, when any file of the tree is stored zstd-compressed, so it cannot be read in place:
  romfs.bin holds every file decoded. Every file is checked against the size and SHA-256 its record gives, and a
  zstd-stored file is decompressed first; a file that fails stops the step, which writes nothing under its final
  names until every file has passed. A container that lists two paths with the same extent (the update's unchanged
  files point at the base's) gets one copy in romfs.bin and two index rows. The extents are read in container order,
  in large windows, so the container is read once from start to end, and the image is written in the same order on a
  second thread.

The tree is the newest update's, else the base's (nup.Nup.program_tree). The ExeFS files are written loose in both
forms, each checked against its record.

romfs-index.bin, little-endian, all offsets from the start of the file, the same layout in both forms:

    header (64 bytes)  "OOT2RFS1", version (u32, 1), row size (u32, 56), row count (u64), rows offset (u64, 64),
                       names offset (u64), names size (u64), image size (u64), stored files (u64)
    rows   (56 bytes)  image offset (u64), size (u64), name offset (u32, from the names offset), name length
                       (u32, bytes, without the NUL), SHA-256 (32 bytes); sorted by the UTF-8 bytes of the path,
                       which is the order strcmp sees, so the runtime finds a path by binary search
    names              each path as UTF-8 ("dir/file", no leading slash) followed by a NUL

The image is romfs.bin in the copy form and the .nup in place, so its size and every offset are those of that file:
in place, an offset is where the file's extent starts in the container (an empty file has offset 0), the image size
is the container's size, and the stored files are its distinct extents.

The row layout follows MacNativeGames' RomFS index as oot-pc-recomp tools/romfs_index.py describes it (offset, size,
name offset, name length), with the file's SHA-256 added so the pack can be checked without the dump.
"""

from __future__ import annotations

import hashlib
import os
import queue
import struct
import threading
from dataclasses import dataclass
from pathlib import Path

from . import nup as nupfile
from .nup import Entry, Nup, NupError, refuse_links
from .runner import Context, PrepareError, Step, file_sha256, fingerprint

FORMAT = 2                                  # changes when the outputs change shape: the step runs again
EXEFS, ROMFS = "exefs", "romfs"            # the container's trees; romfs/ is also the output folder
DUMP_EXEFS = "exefs-dump"                  # the ExeFS files as the dump stores them (exefs/ is the program's)
IMAGE, INDEX = "romfs.bin", "romfs-index.bin"
IN_PLACE, COPY = "in-place", "copy"         # the RomFS forms (module docstring)
ALIGN = 16
WINDOW = 32 << 20                           # bytes read from the container at a time
INDEX_MAGIC = b"OOT2RFS1"
INDEX_HEADER = struct.Struct("<8sIIQQQQQQ")
INDEX_ROW = struct.Struct("<QQII32s")
EMPTY_SHA256 = hashlib.sha256(b"").digest()
HERE = Path(__file__).resolve().parent


class Extract(Step):
    id = "extract"
    needs = ("check",)

    def key(self, ctx: Context) -> str:
        dump = ctx.facts.get("dump") or {}
        return fingerprint(f"format={FORMAT}", f"dump={dump.get('sha256')}:{dump.get('size')}",
                           HERE / "extract.py", HERE / "nup.py")

    def current(self, ctx: Context, record: dict) -> bool:
        try:
            for name, size in record["exefs"].items():
                if (ctx.dest / DUMP_EXEFS / name).stat().st_size != size:
                    return False
            if file_sha256(ctx.dest / ROMFS / INDEX) != record["index_sha256"]:
                return False
            if record["romfs_form"] == IN_PLACE:
                # The image is the dump where it was: a dump the check step now finds elsewhere is indexed again, so
                # the record, and the program's start-up check, name its new place.
                dump = ctx.facts.get("dump")
                if dump and Path(dump["path"]).resolve() != Path(record["nup"]["path"]).resolve():
                    return False
                return Path(record["nup"]["path"]).stat().st_size == record["image_bytes"]
            return (ctx.dest / ROMFS / IMAGE).stat().st_size == record["image_bytes"]
        except (OSError, KeyError, TypeError):
            return False

    def verify(self, ctx: Context, record: dict) -> bool:
        try:
            for name, sha in record["exefs_sha256"].items():
                if file_sha256(ctx.dest / DUMP_EXEFS / name) != sha:
                    return False
            image = Path(record["nup"]["path"]) if record["romfs_form"] == IN_PLACE else None
            return not check_pack(ctx.dest / ROMFS, ctx.reporter, image)
        except (OSError, KeyError, ValueError):
            return False

    def run(self, ctx: Context) -> dict:
        dump = ctx.facts.get("dump")
        if not dump:
            raise PrepareError("the check step did not identify the dump")
        try:
            rec = extract(Path(dump["path"]), ctx.dest, ctx.reporter)
        except NupError as e:
            raise PrepareError(f"the dump {Path(dump['path']).name} cannot be used: {e}") from None
        if rec["romfs_form"] == IN_PLACE:
            rec["nup"] = {"path": str(Path(dump["path"]).resolve()), "size": dump["size"], "sha256": dump["sha256"]}
        return rec


@dataclass
class Blob:
    """One extent of the container, stored once in romfs.bin however many paths name it."""
    entry: Entry
    image_offset: int = -1


def romfs_form(tree: dict[str, Entry]) -> tuple[str, str]:
    """IN_PLACE when every file of the tree is stored raw, so the container can serve it as it is; else COPY. And why,
    in the words the step prints."""
    packed = sorted(p for p, e in tree.items() if e.size and e.codec != nupfile.RAW)
    if not packed:
        return IN_PLACE, f"every one of its {len(tree):,} files is stored raw"
    return COPY, (f"{len(packed):,} of its {len(tree):,} files are stored zstd-compressed ({packed[0]} first), and only "
                  "a raw file can be read in place")


def extent(e: Entry) -> tuple[int, int, int]:
    """What identifies a stored file: its extent and codec; every empty file is the same one."""
    return (e.offset, e.stored, e.codec) if e.size else (-1, 0, nupfile.RAW)


def extract(path: Path, dest: Path, rep, copy: bool = False) -> dict:
    """Extract the game's ExeFS and RomFS from the container at path into dest. Returns the step's record fields. The
    RomFS is indexed in place when it can be (romfs_form), copied when not or when copy is set (the tests' way to
    the copy form with raw containers)."""
    dest = Path(dest)
    with Nup(path) as nup:
        exefs_pkg, exefs = nup.program_tree(EXEFS)
        romfs_pkg, romfs = nup.program_tree(ROMFS)
        dropped = nup.dropped(ROMFS)
        rep.say(f"extract: {path.name}: {len(nup.packages)} packages; ExeFS from the {exefs_pkg.kind_name} "
                f"(version {exefs_pkg.version}, {len(exefs)} files), RomFS from the {romfs_pkg.kind_name} "
                f"(version {romfs_pkg.version}, {len(romfs):,} files; {len(dropped)} older files it removed or "
                "renamed are left out)")
        if not exefs:
            raise NupError("the game has no ExeFS")
        folded = {}
        for name in exefs:
            if "/" in name or name.lower() in folded:
                raise NupError(f"ExeFS file {name!r} is in a subfolder or differs from {folded.get(name.lower())!r} "
                               "only by case")
            folded[name.lower()] = name
        blobs = plan(romfs)
        form, why = romfs_form(romfs)
        if copy and form == IN_PLACE:
            form, why = COPY, "a copy was asked for"
        for folder in (DUMP_EXEFS, ROMFS):
            refuse_links(dest / folder, dest)
        written = write_exefs(nup, exefs, dest / DUMP_EXEFS)
        total = sum(e.size for e in romfs.values())
        if form == IN_PLACE:
            rep.say(f"extract: RomFS read in place from {path.name}: {why}, so no image is written")
            image_bytes, index_sha = index_in_place(nup, romfs, blobs, dest / ROMFS, rep)
            stored_bytes = 0
            distinct = sum(b.entry.size for b in blobs)
            rep.say(f"extract: RomFS {len(romfs):,} files, {total / (1 << 30):.2f} GiB, in {len(blobs):,} distinct "
                    f"extents of the {image_bytes / (1 << 30):.2f} GiB dump: {distinct / (1 << 30):.2f} GiB not copied")
        else:
            rep.say(f"extract: RomFS copied to {ROMFS}/{IMAGE}: {why}")
            image_bytes, index_sha = write_romfs(nup, romfs, blobs, dest / ROMFS, rep)
            stored_bytes = image_bytes
            rep.say(f"extract: RomFS {len(romfs):,} files, {total / (1 << 30):.2f} GiB, stored as {len(blobs):,} "
                    f"distinct files in {image_bytes / (1 << 30):.2f} GiB")
    stored = len(blobs)
    return {
        "bytes": sum(written.values()) + stored_bytes + (dest / ROMFS / INDEX).stat().st_size,
        "romfs_form": form,
        "exefs": written,
        "exefs_sha256": {name: exefs[name].sha256.hex() for name in written},
        "exefs_package": exefs_pkg.version,
        "romfs_package": romfs_pkg.version,
        "romfs_files": len(romfs),
        "romfs_bytes": total,
        "romfs_stored_files": stored,
        "romfs_dropped": len(dropped),
        "image_bytes": image_bytes,
        "index_sha256": index_sha,
    }


def plan(tree: dict[str, Entry]) -> list[Blob]:
    """The distinct extents of a tree in container order. Two paths that share an extent must agree on its contents."""
    by_extent: dict[tuple[int, int, int], Blob] = {}
    for path, e in tree.items():
        key = extent(e)
        blob = by_extent.get(key)
        if blob is None:
            by_extent[key] = Blob(e)
        elif (blob.entry.size, blob.entry.sha256) != (e.size, e.sha256):
            raise NupError(f"{path} shares its extent with {blob.entry.name} but not its size or SHA-256")
    return sorted(by_extent.values(), key=lambda b: b.entry.offset)


def zstd_decompressor():
    """A zstd decompressor object (decompress(bytes) -> bytes), or None when this Python has none."""
    try:
        from compression import zstd                # Python 3.14
        return zstd.ZstdDecompressor()
    except ImportError:
        pass
    try:
        import zstandard                            # the optional zstandard package
        return zstandard.ZstdDecompressor().decompressobj()
    except ImportError:
        return None


def decode(f, e: Entry, sink) -> None:
    """Stream the decoded bytes of one record to sink(memoryview), checking its size and SHA-256."""
    h = hashlib.sha256()
    done = 0
    f.seek(e.offset)
    left = e.stored
    dec = None
    if e.codec == nupfile.ZSTD:
        dec = zstd_decompressor()
        if dec is None:
            raise PrepareError(f"{e.name} is stored zstd-compressed and this Python cannot decompress zstd: use "
                               "Python 3.14 or later, or install the zstandard package")
    while left:
        chunk = f.read(min(WINDOW, left))
        if not chunk:
            raise NupError(f"{e.name}: the container ends inside the file")
        left -= len(chunk)
        if dec is not None:
            try:
                chunk = dec.decompress(chunk)
            except Exception as ex:                  # each library raises its own error type
                raise NupError(f"{e.name}: its zstd data is damaged ({ex})") from None
        if done + len(chunk) > e.size:
            raise NupError(f"{e.name}: decodes to more than the {e.size:,} bytes its record gives")
        h.update(chunk)
        sink(memoryview(chunk))
        done += len(chunk)
    if done != e.size:
        raise NupError(f"{e.name}: decodes to {done:,} bytes, its record gives {e.size:,}")
    if h.digest() != e.sha256:
        raise NupError(f"{e.name}: its contents do not match the SHA-256 of its record (the dump is damaged)")


def write_exefs(nup: Nup, tree: dict[str, Entry], out: Path) -> dict[str, int]:
    """Write each ExeFS file under out, through a temporary name, once it has passed its checks."""
    out.mkdir(parents=True, exist_ok=True)
    root = out.resolve()
    written = {}
    for name in sorted(tree):
        e = tree[name]
        target = (out / name).resolve()
        if target.parent != root:
            raise NupError(f"refusing to write {name!r} outside {out}")
        refuse_links(out / name, out.parent)
        tmp = target.with_name(target.name + ".part")
        try:
            with open(tmp, "wb") as f:
                decode(nup.f, e, f.write)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        written[name] = e.size
    return written


class Writer:
    """Writes buffers to a file on a second thread, so reading, hashing and writing overlap."""

    def __init__(self, path: Path):
        self.f = open(path, "wb", buffering=0)
        self.q: queue.Queue = queue.Queue(maxsize=4)
        self.error: BaseException | None = None
        self.t = threading.Thread(target=self._loop, name="romfs-writer", daemon=True)
        self.t.start()

    def _loop(self) -> None:
        while (buf := self.q.get()) is not None:
            if self.error is None:
                try:
                    view = memoryview(buf)
                    while view:
                        view = view[self.f.write(view):]
                except BaseException as e:   # reported on the reading thread
                    self.error = e

    def put(self, buf) -> None:
        if self.error is not None:
            raise self.error
        self.q.put(buf)

    def close(self) -> None:
        self.q.put(None)
        self.t.join()
        try:
            self.f.flush()
            os.fsync(self.f.fileno())
        finally:
            self.f.close()
        if self.error is not None:
            raise self.error


def write_romfs(nup: Nup, tree: dict[str, Entry], blobs: list[Blob], out: Path, rep) -> tuple[int, str]:
    """Write romfs.bin and romfs-index.bin under out. Returns the image size and the index's SHA-256."""
    out.mkdir(parents=True, exist_ok=True)
    image_tmp, index_tmp = out / (IMAGE + ".part"), out / (INDEX + ".part")
    total = sum(b.entry.size for b in blobs)
    writer = Writer(image_tmp)
    pending = bytearray()
    pos = 0

    def emit(view) -> None:
        nonlocal pending, pos
        pending += view
        pos += len(view)
        if len(pending) >= WINDOW:
            writer.put(pending)
            pending = bytearray()

    src = open(nup.path, "rb", buffering=0)
    try:
        window, win_start = b"", 0
        for n, blob in enumerate(blobs, 1):
            e = blob.entry
            pad = -pos % ALIGN
            if pad:
                emit(bytes(pad))
            blob.image_offset = pos
            if e.size == 0:
                if e.sha256 != EMPTY_SHA256:
                    raise NupError(f"{e.name}: an empty file whose record gives the SHA-256 of something else")
            elif e.codec == nupfile.RAW and e.size <= WINDOW:
                # Small raw files come out of one large read of the container, in container order.
                if not (win_start <= e.offset and e.offset + e.size <= win_start + len(window)):
                    src.seek(e.offset)
                    window = read_full(src, max(WINDOW, e.size))
                    win_start = e.offset
                    if len(window) < e.size:
                        raise NupError(f"{e.name}: the container ends inside the file")
                view = memoryview(window)[e.offset - win_start:e.offset - win_start + e.size]
                if hashlib.sha256(view).digest() != e.sha256:
                    raise NupError(f"{e.name}: its contents do not match the SHA-256 of its record (the dump is "
                                   "damaged)")
                emit(view)
            else:
                decode(src, e, emit)
            rep.progress("romfs", f"extract: RomFS {n:,} of {len(blobs):,} files, {pos / (1 << 30):.2f} of "
                         f"{total / (1 << 30):.2f} GiB", final=n == len(blobs))
        if pending:
            writer.put(pending)
    except BaseException:
        try:
            writer.close()
        except BaseException:
            pass
        image_tmp.unlink(missing_ok=True)
        raise
    finally:
        src.close()
    writer.close()
    index = build_index(tree, {extent(b.entry): b.image_offset for b in blobs}, pos, len(blobs))
    write_synced(index_tmp, index)
    os.replace(image_tmp, out / IMAGE)
    os.replace(index_tmp, out / INDEX)
    return pos, hashlib.sha256(index).hexdigest()


def index_in_place(nup: Nup, tree: dict[str, Entry], blobs: list[Blob], out: Path, rep) -> tuple[int, str]:
    """Write romfs-index.bin with the container's own offsets: the .nup is the image. A romfs.bin an earlier run wrote
    is deleted once the index is in place. Returns the image size (the container's) and the index's SHA-256."""
    out.mkdir(parents=True, exist_ok=True)
    index = build_index(tree, {extent(b.entry): b.entry.offset if b.entry.size else 0 for b in blobs},
                        nup.file_size, len(blobs))
    index_tmp = out / (INDEX + ".part")
    try:
        write_synced(index_tmp, index)
        os.replace(index_tmp, out / INDEX)
    finally:
        index_tmp.unlink(missing_ok=True)
    stale = out / IMAGE
    if stale.exists():
        size = stale.stat().st_size
        stale.unlink()
        rep.say(f"extract: deleted the copy {ROMFS}/{IMAGE} an earlier run wrote ({size / (1 << 30):.2f} GiB): the dump "
                "serves the RomFS now")
    return nup.file_size, hashlib.sha256(index).hexdigest()


def write_synced(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def read_full(f, n: int) -> bytes:
    """Up to n bytes from f; fewer only at the end of the file."""
    parts, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            break
        parts.append(b)
        got += len(b)
    return b"".join(parts)


def build_index(tree: dict[str, Entry], image_offsets: dict, image_bytes: int, stored: int) -> bytes:
    """The bytes of romfs-index.bin for a tree whose extents were stored at image_offsets."""
    names = bytearray()
    rows = bytearray()
    for path in sorted(tree, key=lambda p: p.encode("utf-8")):
        e = tree[path]
        raw = path.encode("utf-8")
        rows += INDEX_ROW.pack(image_offsets[extent(e)], e.size, len(names), len(raw), e.sha256)
        names += raw + b"\0"
    rows_off = INDEX_HEADER.size
    names_off = rows_off + len(rows)
    header = INDEX_HEADER.pack(INDEX_MAGIC, 1, INDEX_ROW.size, len(tree), rows_off, names_off, len(names),
                               image_bytes, stored)
    return header + bytes(rows) + bytes(names)


def read_index(path: Path) -> tuple[dict, list[tuple[str, int, int, bytes]]]:
    """The header fields and the rows (path, offset, size, SHA-256) of a romfs-index.bin. ValueError if malformed."""
    data = Path(path).read_bytes()
    if len(data) < INDEX_HEADER.size:
        raise ValueError("the RomFS index is too short")
    magic, version, row_size, count, rows_off, names_off, names_size, image_bytes, stored = \
        INDEX_HEADER.unpack_from(data)
    if magic != INDEX_MAGIC or version != 1 or row_size != INDEX_ROW.size or rows_off != INDEX_HEADER.size or \
            names_off != rows_off + count * row_size or names_off + names_size != len(data):
        raise ValueError("the RomFS index has a bad header")
    rows = []
    for i in range(count):
        off, size, name_off, name_len, sha = INDEX_ROW.unpack_from(data, rows_off + i * row_size)
        start = names_off + name_off
        if name_off + name_len >= names_size or data[start + name_len] != 0:
            raise ValueError(f"row {i} of the RomFS index names bytes outside its names")
        rows.append((data[start:start + name_len].decode("utf-8"), off, size, sha))
    header = {"count": count, "image_bytes": image_bytes, "stored": stored}
    return header, rows


def check_pack(romfs: Path, rep=None, image: Path | None = None) -> list[str]:
    """Hash every file of a RomFS against its index, read from image: romfs.bin by default, the .nup in place. Returns
    the paths that fail (empty when all pass)."""
    header, rows = read_index(romfs / INDEX)
    image = image or romfs / IMAGE
    if image.stat().st_size != header["image_bytes"]:
        return [f"({image.name} has another size than its index gives)"]
    bad = []
    seen: dict[tuple[int, int, bytes], bool] = {}
    with open(image, "rb") as f:
        for n, (path, off, size, sha) in enumerate(sorted(rows, key=lambda r: r[1]), 1):
            ok = seen.get((off, size, sha))
            if ok is None:
                f.seek(off)
                h = hashlib.sha256()
                left = size
                while left:
                    b = f.read(min(WINDOW, left))
                    if not b:
                        break
                    h.update(b)
                    left -= len(b)
                ok = seen[(off, size, sha)] = not left and h.digest() == sha
            if not ok:
                bad.append(path)
            if rep is not None:
                rep.progress("romfs-check", f"extract: checked {n:,} of {len(rows):,} RomFS files",
                             final=n == len(rows))
    return bad
