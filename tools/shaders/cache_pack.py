#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The shader cache pack (architecture D21): the translator's modules and their resource descriptions in one file,
which the runtime maps and never reads whole. This module owns the layout; runtime/gpu/shader_cache.c reads it.

    py -3 tools/shaders/cache_pack.py write --spv-dir DIR (--translator SHA256 | --translator-exe FILE) --out FILE
                                            [--results results.tsv] [--schema N] [--align N]
    py -3 tools/shaders/cache_pack.py info FILE
    py -3 tools/shaders/cache_pack.py list FILE
    py -3 tools/shaders/cache_pack.py verify FILE

`write` packs a folder of the translator's loose outputs (<key>.spv and <key>.res.json, xlate/host/ResultWriter.cs)
as a base pack; --results adds a failure record for every job a corpus results.tsv (tools/shaders/corpus.py) reports
as not translated. `list` prints one line per row: key, answers digest ("0" for the default answers), status, module
bytes, description bytes; its first two columns are the query lines tests/runtime/shader_cache_test.c reads. Packs
and listings name the game's programs: they stay in the output or work folder, never in the repository.

Two packs, one layout. The base pack (role 1) holds every job translated with the default answers, each row with the
answers digest zero; prepare's translate step writes it (BASE_FILE). The variants pack (role 2) holds modules
translated again with answers a ledger row recorded, none with the digest zero; the ledger step writes it
(VARIANTS_FILE). A reader opens the base and, when the variants pack exists, that one too, and asks for (key, answers
digest): the variants pack first, for a digest other than zero, then the base pack's row of that key with the digest
zero. Both packs must come from one translator and one description schema.

Write-once: write_pack never replaces an existing file, so no step rewrites a pack another step recorded; a step that
runs again deletes its own pack first. The pack is written under a temporary name and renamed once complete.

Layout, little-endian, every offset from the start of the file:

    header (128 bytes)
        0    magic "OOT2SHC1"
        8    format version (u32, 1)
        12   header size (u32, 128)
        16   row size (u32, 128)
        20   description schema (u32): the "schema" of every description, xlate/host/ResultWriter.cs Schema
        24   role (u32): 1 base, 2 variants
        28   zero (u32)
        32   translator identity: SHA-256 of the translator executable (tools/build_xlate.py), 32 bytes
        64   row count (u64)
        72   index offset (u64, 128)
        80   data offset (u64): index offset + row count * 128
        88   zero (u64)
        96   SHA-256 of the index bytes, 32 bytes
    index (128 bytes a row), sorted by the first 64 bytes of each row as unsigned bytes, each (key, digest) once, so a
    reader finds a row by binary search without reading any entry:
        0    key: the job key, 1 to 31 ASCII letters, digits, '.', '_' or '-', padded with NUL to 32 bytes (the front
             end's "<key16>.<variant>", research 35, has 25)
        32   answers digest, 32 bytes: zero for the default answers, else answers_digest of the job
        64   entry offset (u64): a multiple of 16, at or after the data offset
        72   module size (u32): bytes of SPIR-V, a positive multiple of 4; 0 in a failure record
        76   description size (u32): bytes of the description, or of the failure message; never 0
        80   status (u32): 0 translated, 1 not translated (a failure record)
        84   zero (u32)
        88   SHA-256 of the entry's bytes (module, then description), 32 bytes
        120  zero (u64)
    data: each entry is its module followed by its description (the translator's res.json as it wrote it), or for a
        job that did not translate the message "<status>: <reason>" in UTF-8 and no module. Entries follow index
        order, each at a multiple of 16 (a writer may align them further), with zero bytes between them.

Fields marked zero are written as zero and ignored by readers of version 1.

The index's SHA-256 lets a reader refuse a damaged index at open for the cost of hashing the index alone. An entry's
own SHA-256 is checked by `verify` and on request by the C reader, never on every lookup.

A reader refuses, by name: a file shorter than a header (SHORT); another magic (MAGIC) or format version (VERSION);
sizes and offsets that version 1 does not give, or an unknown role (HEADER); another description schema than the one
it reads (SCHEMA); a pack of the other role (ROLE); an index that ends past the end of the file (INDEX_TRUNCATED); an
index whose SHA-256 differs (INDEX_HASH); a malformed row: key, digest for the role, status, sizes, alignment (ROW);
rows not strictly ascending (ORDER); an entry that ends past the end of the file (ENTRY_PAST_END). The writer adds
EXISTS (the pack is already there) and INPUT (an entry it cannot pack). The C reader adds OPEN (no such file) and
TRANSLATOR (a variants pack from another translator than the base's).

Exit codes: 0 done, 1 the pack or its input was refused (verify: an entry failed), 2 usage.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

MAGIC = b"OOT2SHC1"
VERSION = 1
HEADER = struct.Struct("<8sIIIII4x32sQQQ8x32s")      # magic .. reserved, translator, count, index, data, index sha
ROW = struct.Struct("<32s32sQIII4x32s8x")             # key, answers, offset, module, description, status, sha256
HEADER_BYTES = 128
ROW_BYTES = 128
KEY_BYTES = 32
DIGEST_BYTES = 32
ENTRY_ALIGN = 16
DESC_SCHEMA = 2                 # the description schema this writer packs: xlate/host/ResultWriter.cs Schema
BASE, VARIANTS = 1, 2
ROLES = {BASE: "base", VARIANTS: "variants"}
TRANSLATED, FAILED = 0, 1
ZERO_DIGEST = bytes(DIGEST_BYTES)
SPIRV_MAGIC = 0x07230203
KEY_RE = re.compile(r"[A-Za-z0-9._-]{1,31}")
# D21's answers: texture and image types, vertex attribute types, fragment output types (the job's fields).
ANSWER_FIELDS = ("attr", "fout", "img", "tex")
BASE_FILE = "shader-base.bin"
VARIANTS_FILE = "shader-variants.bin"

assert HEADER.size == HEADER_BYTES and ROW.size == ROW_BYTES


class PackError(Exception):
    """A pack or an entry refused; code is one of the names in the module docstring."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# --- answers ------------------------------------------------------------------

def answers_digest(job: dict) -> bytes:
    """The answers digest of a translator job: zero when the job sets none of ANSWER_FIELDS, else the SHA-256 of the
    canonical JSON text (sorted keys, no spaces, ASCII) of the fields it sets, as research 35 builds a variant's text.
    The format takes any 32 bytes; the ledger's writer must compute the same digest for the same answers."""
    chosen = {name: job[name] for name in ANSWER_FIELDS if job.get(name)}
    if not chosen:
        return ZERO_DIGEST
    text = json.dumps(chosen, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).digest()


def digest_text(digest: bytes) -> str:
    """"0" for the default answers, else 64 hexadecimal digits (the listing's and the C test's form)."""
    return "0" if digest == ZERO_DIGEST else digest.hex()


def parse_digest(text: str) -> bytes:
    if text == "0":
        return ZERO_DIGEST
    raw = bytes.fromhex(text)
    if len(raw) != DIGEST_BYTES:
        raise ValueError(f"an answers digest is 64 hexadecimal digits or 0, not {text!r}")
    return raw


# --- writing ------------------------------------------------------------------

@dataclass
class Entry:
    """One job for the pack: a module and its description, or, when failed, the reason in `description`. Bytes, the
    path of a file to read them from, or a function that returns them (prepare's translate step reads them out of one
    scratch file), read only while the entry is written."""
    key: str
    module: bytes | Path | Callable[[], bytes] | None = None
    description: bytes | Path | str | Callable[[], bytes] = b""
    answers: bytes = ZERO_DIGEST
    failed: bool = False

    def sort_bytes(self) -> bytes:
        return self.key.encode("ascii").ljust(KEY_BYTES, b"\0") + self.answers


def _load(value: bytes | Path | str | Callable[[], bytes] | None) -> bytes:
    if value is None:
        return b""
    if callable(value):
        return bytes(value())
    if isinstance(value, Path):
        return value.read_bytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def _translator_bytes(translator: str | bytes) -> bytes:
    raw = bytes.fromhex(translator) if isinstance(translator, str) else bytes(translator)
    if len(raw) != 32:
        raise PackError("INPUT", "the translator identity is a SHA-256 (64 hexadecimal digits)")
    return raw


def _check_module(key: str, module: bytes) -> None:
    if not module or len(module) % 4 or struct.unpack_from("<I", module)[0] != SPIRV_MAGIC:
        raise PackError("INPUT", f"{key}: the module is not SPIR-V (empty, not whole words, or no SPIR-V magic)")


def _check_description(e: Entry, module: bytes, desc: bytes, schema: int, translator: str, role: int) -> None:
    """The description must be the translator's res.json of this very module, for this pack."""
    try:
        d = json.loads(desc)
    except (ValueError, UnicodeDecodeError):
        raise PackError("INPUT", f"{e.key}: the description is not JSON") from None
    if not isinstance(d, dict):
        raise PackError("INPUT", f"{e.key}: the description is not a JSON object")
    wrong = []
    if d.get("schema") != schema:
        wrong.append(f"schema {d.get('schema')!r}, the pack's is {schema}")
    if d.get("key") != e.key:
        wrong.append(f"key {d.get('key')!r}")
    if d.get("translator_id") != translator:
        wrong.append("another translator identity")
    if d.get("spv_bytes") != len(module) or d.get("spv_sha256") != hashlib.sha256(module).hexdigest():
        wrong.append("another module's size or SHA-256")
    answers = d.get("answers")
    if not isinstance(answers, dict):
        wrong.append("no answers object")
    else:
        set_here = [f for f in ANSWER_FIELDS if answers.get(f)]
        if role == BASE and set_here:
            wrong.append(f"answers set ({', '.join(set_here)}) in a base pack, which holds the default answers only")
        if role == VARIANTS and not set_here:
            wrong.append("no answer set in a variants pack")
    if wrong:
        raise PackError("INPUT", f"{e.key}: the description gives " + "; ".join(wrong))


def write_pack(path: Path, entries: Iterable[Entry], *, translator: str | bytes, schema: int = DESC_SCHEMA,
               role: int = BASE, align: int = ENTRY_ALIGN) -> dict:
    """Write a new pack at path from entries and return its sizes. Refuses (PackError) an existing path (EXISTS) and
    any entry that does not belong in this pack (INPUT): a bad key, a repeated (key, digest), a digest against the
    role, a module that is not SPIR-V, a description of another module, schema, translator or role, an empty failure
    message. Entries are read one at a time; only the index is held in memory."""
    path = Path(path)
    if role not in ROLES:
        raise PackError("INPUT", f"role {role} is neither base ({BASE}) nor variants ({VARIANTS})")
    if align < ENTRY_ALIGN or align & (align - 1):
        raise PackError("INPUT", f"the alignment {align} is not a power of two of at least {ENTRY_ALIGN}")
    tid = _translator_bytes(translator)
    if path.exists():
        raise PackError("EXISTS", f"{path} exists: a pack is written once (delete your own pack to write it again)")
    todo = list(entries)
    # Keys and digests are checked before the sort, which encodes them: a bad one is INPUT, never a bare exception.
    for e in todo:
        if not isinstance(e.key, str) or not KEY_RE.fullmatch(e.key):
            raise PackError("INPUT", f"the key {e.key!r} is not 1 to 31 letters, digits, '.', '_' or '-'")
        if not isinstance(e.answers, bytes) or len(e.answers) != DIGEST_BYTES:
            raise PackError("INPUT", f"{e.key}: the answers digest is not 32 bytes")
    todo.sort(key=Entry.sort_bytes)
    seen: set[bytes] = set()
    for e in todo:
        if role == BASE and e.answers != ZERO_DIGEST:
            raise PackError("INPUT", f"{e.key}: an answers digest in a base pack, which holds the default answers only")
        if role == VARIANTS and e.answers == ZERO_DIGEST:
            raise PackError("INPUT", f"{e.key}: the default answers in a variants pack")
        if e.sort_bytes() in seen:
            raise PackError("INPUT", f"{e.key} {digest_text(e.answers)}: given twice")
        seen.add(e.sort_bytes())

    data_off = HEADER_BYTES + ROW_BYTES * len(todo)
    tmp = path.with_name(path.name + ".tmp")
    rows = bytearray()
    stats = {"rows": len(todo), "failures": 0, "module_bytes": 0, "description_bytes": 0}
    try:
        with open(tmp, "wb") as f:
            f.write(bytes(data_off))
            at = data_off
            for e in todo:
                at = -(-at // align) * align
                module = b"" if e.failed else _load(e.module)
                desc = _load(e.description)
                if e.failed:
                    if not desc:
                        raise PackError("INPUT", f"{e.key}: a failure record needs its reason")
                    try:
                        desc.decode("utf-8")
                    except UnicodeDecodeError:
                        raise PackError("INPUT", f"{e.key}: the failure reason is not UTF-8") from None
                    stats["failures"] += 1
                else:
                    _check_module(e.key, module)
                    _check_description(e, module, desc, schema, tid.hex(), role)
                f.seek(at)
                f.write(module)
                f.write(desc)
                rows += ROW.pack(e.key.encode("ascii"), e.answers, at, len(module), len(desc),
                                 FAILED if e.failed else TRANSLATED, hashlib.sha256(module + desc).digest())
                stats["module_bytes"] += len(module)
                stats["description_bytes"] += len(desc)
                at += len(module) + len(desc)
            f.truncate(at)
            header = HEADER.pack(MAGIC, VERSION, HEADER_BYTES, ROW_BYTES, schema, role, tid, len(todo), HEADER_BYTES,
                                 data_off, hashlib.sha256(rows).digest())
            f.seek(0)
            f.write(header)
            f.write(rows)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            raise PackError("EXISTS", f"{path} appeared while the pack was written")
        os.rename(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    stats["bytes"] = path.stat().st_size
    stats["index_bytes"] = len(rows)
    return stats


def entries_from_folder(folder: Path, results: Path | None = None) -> list[Entry]:
    """Base-pack entries for the translator's loose outputs in folder, plus a failure record for every job that
    results.tsv (corpus.py's columns key, status, message) reports as not translated."""
    folder = Path(folder)
    spv, res = set(), set()
    with os.scandir(folder) as it:
        for d in it:
            if d.name.endswith(".res.json"):
                res.add(d.name[:-len(".res.json")])
            elif d.name.endswith(".spv"):
                spv.add(d.name[:-len(".spv")])
    if spv != res:
        odd = sorted(spv ^ res)
        raise PackError("INPUT", f"{len(odd)} job(s) have one file of the two, the first {odd[0]!r}")
    entries = [Entry(k, folder / f"{k}.spv", folder / f"{k}.res.json") for k in sorted(spv)]
    if results:
        with open(results, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                status = row.get("status")
                if status == "ok" or not status:
                    continue
                key = row.get("key") or ""
                if key in spv:
                    raise PackError("INPUT", f"{key}: reported {status} and translated in the same folder")
                reason = (row.get("message") or "").strip() or "no reason given"
                entries.append(Entry(key, None, f"{status}: {reason}", failed=True))
    return entries


# --- reading ------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    key: str
    answers: bytes
    offset: int
    module_size: int
    description_size: int
    status: int
    sha256: bytes

    @property
    def failed(self) -> bool:
        return self.status == FAILED

    def sort_bytes(self) -> bytes:
        return self.key.encode("ascii").ljust(KEY_BYTES, b"\0") + self.answers


@dataclass
class Pack:
    path: Path
    size: int
    schema: int
    role: int
    translator: str
    data_offset: int
    rows: list[Row]
    _keys: list[bytes] | None = field(default=None, repr=False, compare=False)

    def find(self, key: str, answers: bytes = ZERO_DIGEST) -> Row | None:
        """The row of (key, answers), by binary search over the index."""
        probe = key.encode("ascii", "replace").ljust(KEY_BYTES, b"\0") + answers
        if self._keys is None:
            self._keys = [r.sort_bytes() for r in self.rows]
        i = bisect.bisect_left(self._keys, probe)
        return self.rows[i] if i < len(self._keys) and self._keys[i] == probe else None

    def read(self, row: Row) -> tuple[bytes, bytes]:
        """The entry of one row: (module, description or failure message). Reads that entry alone."""
        with open(self.path, "rb") as f:
            f.seek(row.offset)
            data = f.read(row.module_size + row.description_size)
        if len(data) != row.module_size + row.description_size:
            raise PackError("ENTRY_PAST_END", f"{self.path.name}: {row.key}'s entry is cut short")
        return data[:row.module_size], data[row.module_size:]


def open_pack(path: Path, *, role: int | None = None, schema: int | None = DESC_SCHEMA) -> Pack:
    """Read and check the header and the index of a pack (never an entry). role and schema None accept any."""
    path = Path(path)
    name = path.name
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
        if len(head) < HEADER_BYTES:
            raise PackError("SHORT", f"{name} has {size} bytes, less than a header")
        magic, version, header_size, row_size, sch, rol, tid, count, index_off, data_off, index_sha = \
            HEADER.unpack(head)
        if magic != MAGIC:
            raise PackError("MAGIC", f"{name} is not a shader cache pack (no {MAGIC.decode()} header)")
        if version != VERSION:
            raise PackError("VERSION", f"{name} has format version {version}, this reader reads {VERSION}")
        if header_size != HEADER_BYTES or row_size != ROW_BYTES or index_off != HEADER_BYTES:
            raise PackError("HEADER", f"{name}'s header gives the sizes or offsets of another layout")
        if schema is not None and sch != schema:
            raise PackError("SCHEMA", f"{name} holds descriptions of schema {sch}, this reader reads schema {schema}")
        if rol not in ROLES:
            raise PackError("HEADER", f"{name} has role {rol}, neither base ({BASE}) nor variants ({VARIANTS})")
        if role is not None and rol != role:
            raise PackError("ROLE", f"{name} is a {ROLES[rol]} pack, a {ROLES[role]} pack was expected")
        if count > (size - HEADER_BYTES) // ROW_BYTES:
            raise PackError("INDEX_TRUNCATED", f"{name}: the index of {count} rows ends past the end of the "
                                               f"{size}-byte file")
        if data_off != HEADER_BYTES + count * ROW_BYTES:
            raise PackError("HEADER", f"{name}: the data offset {data_off} is not where the index ends")
        index = f.read(count * ROW_BYTES)
    if hashlib.sha256(index).digest() != index_sha:
        raise PackError("INDEX_HASH", f"{name}: the index does not match its SHA-256 (the file is damaged)")
    rows: list[Row] = []
    prev = b""
    for i in range(count):
        key_raw, answers, off, ms, ds, status, sha = ROW.unpack_from(index, i * ROW_BYTES)
        key_text = key_raw.rstrip(b"\0")
        try:
            key = key_text.decode("ascii")
        except UnicodeDecodeError:
            key = ""
        if not KEY_RE.fullmatch(key) or b"\0" in key_text:
            raise PackError("ROW", f"{name}: row {i}'s key is not 1 to 31 key characters padded with NUL")
        if rol == BASE and answers != ZERO_DIGEST:
            raise PackError("ROW", f"{name}: row {i} ({key}) has an answers digest in a base pack")
        if rol == VARIANTS and answers == ZERO_DIGEST:
            raise PackError("ROW", f"{name}: row {i} ({key}) has the default answers in a variants pack")
        if status == TRANSLATED:
            if not ms or ms % 4 or not ds:
                raise PackError("ROW", f"{name}: row {i} ({key}) is translated with a module of {ms} bytes and a "
                                       f"description of {ds}")
        elif status == FAILED:
            if ms or not ds:
                raise PackError("ROW", f"{name}: row {i} ({key}) is a failure record with a module or no reason")
        else:
            raise PackError("ROW", f"{name}: row {i} ({key}) has the unknown status {status}")
        if off % ENTRY_ALIGN or off < data_off:
            raise PackError("ROW", f"{name}: row {i} ({key})'s entry offset {off} is not a multiple of "
                                   f"{ENTRY_ALIGN} after the index")
        if off > size or ms + ds > size - off:
            raise PackError("ENTRY_PAST_END", f"{name}: row {i} ({key})'s entry ends at byte {off + ms + ds}, "
                                              f"past the end of the {size}-byte file")
        row = Row(key, answers, off, ms, ds, status, sha)
        if i and prev >= row.sort_bytes():
            raise PackError("ORDER", f"{name}: row {i} ({key}) does not follow row {i - 1} in ascending key order")
        prev = row.sort_bytes()
        rows.append(row)
    return Pack(path, size, sch, rol, tid.hex(), data_off, rows)


def verify_pack(path: Path, *, role: int | None = None, schema: int | None = DESC_SCHEMA) -> list[str]:
    """Check every entry of a pack against its row (SHA-256) and every description against its module, schema,
    translator and role. Returns one line per entry that fails (empty when all pass); PackError when the header or
    the index is refused."""
    pack = open_pack(path, role=role, schema=schema)
    bad = []
    with open(pack.path, "rb") as f:
        for row in pack.rows:
            f.seek(row.offset)
            data = f.read(row.module_size + row.description_size)
            if hashlib.sha256(data).digest() != row.sha256:
                bad.append(f"{row.key} {digest_text(row.answers)}: the entry does not match its SHA-256")
                continue
            if row.failed:
                continue
            module, desc = data[:row.module_size], data[row.module_size:]
            try:
                _check_module(row.key, module)
                _check_description(Entry(row.key, answers=row.answers), module, desc, pack.schema, pack.translator,
                                   pack.role)
            except PackError as e:
                bad.append(f"{row.key} {digest_text(row.answers)}: {e.message}")
    return bad


# --- command line -------------------------------------------------------------

def _translator_arg(args: argparse.Namespace) -> str:
    if args.translator_exe:
        return hashlib.sha256(Path(args.translator_exe).read_bytes()).hexdigest()
    return args.translator


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="pack a folder of loose translator outputs as a base pack")
    w.add_argument("--spv-dir", type=Path, required=True)
    who = w.add_mutually_exclusive_group(required=True)
    who.add_argument("--translator", help="the translator's SHA-256 (64 hexadecimal digits)")
    who.add_argument("--translator-exe", type=Path, help="the translator executable, hashed for its identity")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--results", type=Path, help="a corpus results.tsv: jobs not translated become failure records")
    w.add_argument("--schema", type=int, default=DESC_SCHEMA)
    w.add_argument("--align", type=int, default=ENTRY_ALIGN)
    for name in ("info", "list", "verify"):
        p = sub.add_parser(name)
        p.add_argument("pack", type=Path)
        p.add_argument("--schema", type=int, default=None, help="refuse another description schema (default: any)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "write":
            t0 = time.perf_counter()
            st = write_pack(args.out, entries_from_folder(args.spv_dir, args.results),
                            translator=_translator_arg(args), schema=args.schema, align=args.align)
            print(f"cache_pack: wrote {args.out.name}: {st['rows']} rows ({st['failures']} failure records), "
                  f"{st['bytes']} bytes, index {st['index_bytes']} bytes, modules {st['module_bytes']} bytes, "
                  f"descriptions {st['description_bytes']} bytes, {time.perf_counter() - t0:.1f} s")
            return 0
        pack = open_pack(args.pack, schema=args.schema)
        if args.cmd == "info":
            failures = sum(r.failed for r in pack.rows)
            print(f"cache_pack: {pack.path.name}: {ROLES[pack.role]} pack, schema {pack.schema}, translator "
                  f"{pack.translator}, {len(pack.rows)} rows ({failures} failure records), {pack.size} bytes, index "
                  f"{len(pack.rows) * ROW_BYTES} bytes")
        elif args.cmd == "list":
            for r in pack.rows:
                print(f"{r.key}\t{digest_text(r.answers)}\t{r.status}\t{r.module_size}\t{r.description_size}")
        else:
            bad = verify_pack(args.pack, schema=args.schema)
            for line in bad:
                print(line)
            print(f"cache_pack: {len(pack.rows) - len(bad)} of {len(pack.rows)} entries pass")
            return 1 if bad else 0
        return 0
    except PackError as e:
        print(f"cache_pack: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"cache_pack: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
