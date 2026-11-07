#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Refuse anything that looks like game content, keys or generated game code.

This repository ships tools and a runtime. It never ships anything that came
out of the game: no executables, containers, shaders, textures, models, audio,
text, keys, firmware or frame captures. This script is the guard, run as the
pre-commit hook (.githooks/pre-commit) and in CI (.github/workflows).

Modes (exactly one):
  --staged         the files staged for the next commit (index content)
  --range REV      every commit in A..B, not only the tip; a single REV checks
                   all history reachable from it (new branch, forced push)
  --all            the tracked tree plus untracked files that are not ignored
  --paths P [P..]  files, or directories searched recursively, on disk

Exit codes: 0 clean, 1 at least one problem, 2 the check itself could not run
(a failing git command never counts as a pass).

Standard library only, so it runs before any environment is set up.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple

MAX_BYTES = 512 * 1024

# Reviewed exceptions, by repository-relative POSIX path, each with its reason.
# ALLOW_BINARY exempts a file from the binary-content rule and from the image,
# audio and video rules (a logo, say; compiled SPIR-V built from the project's
# own shader sources would need an entry too, so prefer building it in CI).
# It never exempts game formats, keys or the generated-code marker.
# ALLOW_BLOBS exempts a file from the hex, base64 and decimal-array heuristics,
# from the generated-code heuristics and from the key-assignment rule. It is the
# route for known tables (a CRC-32 table, an S-box) and for a reviewed list of
# hashes whose labels are file names. A PEM private key is never exempt.
ALLOW_BINARY: set[str] = set()
ALLOW_BLOBS: set[str] = {
    # The lifter's FP and SIMD helpers hold four architectural tables, none from
    # the game: the FRECPE and FRSQRTE estimate tables of the Arm architecture's
    # pseudocode (RecipEstimate, RecipSqrtEstimate) and the CRC-32 and CRC-32C
    # tables of their public polynomials. Taken with the file from LAS Recomp
    # through DKB Recomp (THIRD_PARTY_NOTICES.md).
    "runtime/include/oot2/lift_simd.h",
}

# Switch containers, executables, game data, archives and shader archives.
BLOCKED_SUFFIXES = {
    ".nso", ".nro", ".nrr", ".nca", ".ncz", ".nsp", ".nsz", ".xci", ".xcz",
    ".nup", ".tik", ".cert", ".npdm", ".keys", ".bin", ".zbic",
    ".bfres", ".bntx", ".bnsh", ".bfsha", ".sarc", ".szs", ".zs", ".zst",
    ".pack", ".byml", ".bgyml", ".msbt", ".bars", ".bwav",
    ".bfwav", ".bfstm", ".bfttf", ".bfotf", ".sharcb", ".astc",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".xz", ".bz2",
}
# Pictures, audio and video: game-derived by default, exemptable per file.
MEDIA_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tga", ".tif", ".tiff",
    ".ico", ".dds", ".ktx", ".ktx2", ".wav", ".ogg", ".opus", ".mp3", ".flac",
    ".mp4", ".mkv", ".webm", ".bik", ".mov", ".avi",
}
# ExeFS modules and key files carry fixed names, often without an extension.
# Joined from pieces so that this file does not contain the literal names.
MODULE_NAMES = {
    "main", "sdk", "rtld", "subsdk0", "subsdk1", "subsdk2", "subsdk3",
    "subsdk4", "subsdk5", "subsdk6", "subsdk7", "subsdk8", "subsdk9", "main.npdm",
}
BLOCKED_NAMES = MODULE_NAMES | {
    "prod." + "keys", "title." + "keys", "dev." + "keys",
    "console." + "keys", "header" + "_key", "header" + "_key.bin",
}
# Matched at any depth. Keep in step with the directory patterns of .gitignore;
# a unit test fails when they drift apart, so the guard and the ignore file agree.
BLOCKED_DIRS = {
    "exefs", "romfs", "dump", "dumps", "extracted",
    "build", "out", "generated", "captures", "frame-dumps",
    "shader-cache", "pipeline-cache",
    "bin", "obj",  # .NET build output (xlate/)
}
BLOCKED_DIR_PREFIXES = ("cmake-build-",)

# Source files are small when a person wrote them. Recompiled game code is not:
# one file is hundreds of kilobytes, or full of functions named by address.
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl", ".s", ".asm"}
MAX_SOURCE_BYTES = 256 * 1024
# Code addresses from 0x70_0000_0000 (this project's modules, tools/layout.py,
# named f_<address> by the lifter) or 0x71_0000_0000 (the usual Switch 1 base).
ADDRESS_NAME = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*?_(?:0[xX])?7[01][0-9A-Fa-f]{8}"
    r"|(?:[Ss][Uu][Bb]|[Ff][Uu][Nn][Cc]?|[Ll][Oo][Cc]|[Pp][Rr][Oo][Cc])_(?:0[xX])?[0-9A-Fa-f]{6,16})\b"
)
ADDRESS_NAMES_LIMIT = 16  # distinct names built from an address

# Written by code generators into every file they emit.
GENERATED_MARKER = b"GENERATED_FROM_" + b"GAME_CODE"


class Finding(NamedTuple):
    rule: str
    message: str


class Magic(NamedTuple):
    name: str
    offset: int
    sig: bytes
    kind: str  # "game" (never exempt) or "media" (exemptable per file)
    binary_only: bool = False  # only when the first bytes look like binary data


# Fixed-offset signatures. Four-letter ASCII magics at offset 0 are the Switch
# formats; the entries with binary_only sit at inner offsets where a text file
# could collide by chance.
MAGICS = (
    Magic("NSO executable (NSO0)", 0, b"NSO0", "game"),
    Magic("NRO executable (NRO0)", 0x10, b"NRO0", "game"),
    Magic("NCA content archive", 0x200, b"NCA3", "game", True),
    Magic("NCA content archive", 0x200, b"NCA2", "game", True),
    Magic("NCA content archive", 0x200, b"NCA0", "game", True),
    Magic("XCI cartridge image", 0x100, b"HEAD", "game", True),
    Magic("PFS0 partition (NSP)", 0, b"PFS0", "game"),
    Magic("HFS0 partition (XCI)", 0, b"HFS0", "game"),
    Magic("BNTX texture archive", 0, b"BNTX", "game"),
    Magic("BNSH shader binary", 0, b"BNSH", "game"),
    Magic("BFSHA shader archive", 0, b"BFSHA", "game"),
    Magic("FSHA shader archive", 0, b"FSHA", "game"),
    Magic("SARC archive", 0, b"SARC", "game"),
    Magic("Yaz0 compressed data", 0, b"Yaz0", "game"),
    Magic("zstd frame", 0, bytes.fromhex("28b52ffd"), "game"),
    Magic("ZBIC container", 0, b"ZBIC", "game"),
    Magic("BFRES model archive", 0, b"FRES", "game"),
    Magic("BWAV audio", 0, b"BWAV", "game"),
    Magic("BARS audio archive", 0, b"BARS", "game"),
    Magic("BFSTM audio", 0, b"FSTM", "game"),
    Magic("BFWAV audio", 0, b"FWAV", "game"),
    Magic("MSBT text archive", 0, b"MsgStdBn", "game"),
    Magic("zip archive", 0, b"PK\x03\x04", "game"),
    Magic("gzip data", 0, b"\x1f\x8b\x08", "game"),
    Magic("xz data", 0, bytes.fromhex("fd377a585a00"), "game"),
    Magic("7z archive", 0, bytes.fromhex("377abcaf271c"), "game"),
    Magic("ASTC texture", 0, bytes.fromhex("13aba15c"), "media"),
    Magic("DDS texture", 0, b"DDS ", "media"),
    Magic("PNG image", 0, b"\x89PNG\r\n\x1a\n", "media"),
    Magic("JPEG image", 0, b"\xff\xd8\xff", "media"),
    Magic("GIF image", 0, b"GIF87a", "media"),
    Magic("GIF image", 0, b"GIF89a", "media"),
    Magic("KTX texture", 0, b"\xabKTX 11\xbb\r\n\x1a\n", "media"),
    Magic("KTX2 texture", 0, b"\xabKTX 20\xbb\r\n\x1a\n", "media"),
    Magic("Ogg audio", 0, b"OggS", "media"),
    Magic("FLAC audio", 0, b"fLaC", "media"),
)
HEADER_AREA = 0x400  # an NSO or NRO keeps its MOD0 block this close to the start


def detect_magic(data: bytes) -> list[Magic]:
    """Signatures present in the data, including the few that need logic."""
    hits = []
    binaryish = b"\0" in data[:0x400]
    for magic in MAGICS:
        if magic.binary_only and not binaryish:
            continue
        if data[magic.offset:magic.offset + len(magic.sig)] == magic.sig:
            hits.append(magic)
    head = data[:HEADER_AREA]
    if binaryish and b"MOD0" in head:
        hits.append(Magic("MOD0 module header", head.index(b"MOD0"), b"MOD0", "game"))
    if len(data) >= 4 and 0x50 <= data[0] <= 0x5F and data[1:4] == b"\x2a\x4d\x18":
        hits.append(Magic("zstd skippable frame", 0, data[:4], "game"))
    if data[:2] == b"YB" and len(data) >= 4 and data[2] in range(2, 8) and data[3] == 0:
        hits.append(Magic("BYML data (little endian)", 0, data[:4], "game"))
    if data[:3] == b"BY\0" and len(data) >= 4 and data[3] in range(2, 8):
        hits.append(Magic("BYML data (big endian)", 0, data[:4], "game"))
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        hits.append(Magic("WebP image", 0, b"RIFF", "media"))
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        hits.append(Magic("WAVE audio", 0, b"RIFF", "media"))
    return hits


# A key assignment, as in a keys file: "<name> = <32 or 64 hex digits>". It also
# reads C declarations and defines, an optional 0x prefix, quoted values, CRLF
# line ends and a trailing comment. Written with spaces only ("name <hex>") it
# counts only for names that say key, kek or source.
# Hashes and build ids have the same shape and are evidence, so a name made of
# hash-like words is exempt. Everything else is treated as a key.
KEY_ASSIGN = re.compile(
    r"""^[ \t{,]*(?:\#[ \t]*define[ \t]+)?(?:[A-Za-z_][\w:]*[ \t*&]+){0,4}?"""
    r"""["']?([A-Za-z0-9_.\-]+)["']?(?:\[\])?([ \t]*[=:][ \t]*|[ \t]+)["']?(?:0[xX])?"""
    r"""(?:[0-9A-Fa-f]{32}|[0-9A-Fa-f]{64})["']?[ \t,;})\r]*(?:(?:\#|//|/\*).*)?$""",
    re.MULTILINE,
)
KEY_WORDS = re.compile(r"key|kek|source", re.IGNORECASE)
HASH_WORDS = ("sha", "md5", "crc", "hash", "digest", "checksum", "build",
              "commit", "fingerprint", "etag", "uuid", "guid")
HASH_EXACT = {"id"}  # a bare "id" is an identifier; "idle" and "identity" are not
PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")

HEX_RUN = re.compile(r"[0-9A-Fa-f]{192,}")
HEX_ONLY = re.compile(r"[0-9A-Fa-f]+")
B64_RUN = re.compile(r"[A-Za-z0-9+/]{256,}={0,2}")
B64_LINE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
HEX_LINE_DIGITS = 16  # a line of hex tokens needs this many digits
HEX_TABLE_LINES = 32  # consecutive hex-dump lines
HEX_DENSE_LINES = 16  # ...or this many, when they are most of the file
HEX_DENSE_PERCENT = 60
BYTE_TOKEN_LIMIT = 2048  # 0xNN or \xNN tokens in one file
B64_BLOCK_LINES = 12  # consecutive wrapped base64 lines

# One hex-dump line: an optional offset column, then hex tokens (0xNN, \xNN, bare
# NN or NNNN...), then nothing or a gutter (an ASCII column, a comment).
_TOKEN = r"(?:(?:0[xX]|\\x)[0-9A-Fa-f]{2,16}|[0-9A-Fa-f]{2,64})(?![0-9A-Fa-f])"
HEX_LEAD = re.compile(
    r"""^[ \t{(\["']*(?:[0-9A-Fa-f]{4,16}:[ \t]+|0[xX][0-9A-Fa-f]{2,16}:[ \t]+|[0-9A-Fa-f]{6,16}[ \t]{2,})?"""
    r"(" + _TOKEN + r"(?:[ \t,;]*" + _TOKEN + r")*)"
)
HEX_TAIL = re.compile(r"""[ \t,;})\]"']*(?:[ \t]*\|.*\||[ \t]{2,}\S.*|//.*|\#.*|/\*.*)?\r?""")
BYTE_TOKEN = re.compile(r"(?:0[xX]|\\x)[0-9A-Fa-f]{2}(?![0-9A-Fa-f])")
_DEC = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
DEC_ARRAY = re.compile(r"\b" + _DEC + r"\b(?:[ \t]*,\s*" + _DEC + r"\b){255,}")
DEC_ARRAY_DISTINCT = 16  # a run of 256+ small numbers with at least this many different values


def is_hash_name(name: str) -> bool:
    """Hash-like names, and ExeFS module names, which key tables of build ids."""
    if name.lower() in MODULE_NAMES:
        return True
    words = re.split(r"[^a-z0-9]+", name.lower())
    return any(word.startswith(HASH_WORDS) or word in HASH_EXACT for word in words)


def is_hex_table_line(line: str) -> bool:
    """A hex-dump line: 16+ hex digits as tokens, with an offset column or an ASCII gutter allowed."""
    match = HEX_LEAD.match(line)
    if match is None or HEX_TAIL.fullmatch(line, match.end()) is None:
        return False
    tokens = re.findall(_TOKEN, match.group(1))
    gutter = line[match.end():].strip(" \t,;})]\"'\r")
    if gutter and len(tokens) < 4:
        return False  # "<hash>  file name" is a labelled hash, not a dump
    digits = re.sub(r"[^0-9A-Fa-f]", "", re.sub(r"0[xX]|\\x", "", match.group(1)))
    prefixed = re.search(r"0[xX]|\\x", match.group(1)) is not None
    return len(digits) >= HEX_LINE_DIGITS and (prefixed or re.search(r"[A-Fa-f]", digits) is not None)


def longest_streak(lines: Iterable[str], pred: Callable[[str], bool]) -> int:
    best = cur = 0
    for line in lines:
        cur = cur + 1 if pred(line) else 0
        best = max(best, cur)
    return best


def key_findings(text: str) -> list[Finding]:
    found = []
    for match in KEY_ASSIGN.finditer(text):
        name, separator = match.group(1), match.group(2)
        if separator.strip() == "" and not KEY_WORDS.search(name):
            continue  # "Build ID <hex>" and friends: spaces alone are weak evidence
        if not is_hash_name(name):
            found.append(Finding("key-line", f"line assigns a key-sized hex value to '{name}'"))
            break
    if PRIVATE_KEY.search(text):
        found.append(Finding("private-key", "contains a PEM private key block"))
    return found


def blob_findings(text: str) -> list[Finding]:
    found = []
    for match in HEX_RUN.finditer(text):
        if len(set(match.group().lower())) >= 6:
            found.append(Finding("hex-blob", f"hex run of {len(match.group())} digits looks like a dumped table"))
            break
    for match in B64_RUN.finditer(text):
        run = match.group()
        if len(set(run)) >= 16 and not HEX_ONLY.fullmatch(run):
            found.append(Finding("base64-blob", f"base64 run of {len(run)} characters looks like embedded data"))
            break
    lines = text.splitlines()
    hex_lines = [is_hex_table_line(ln) for ln in lines]
    hex_count, nonblank = sum(hex_lines), sum(1 for ln in lines if ln.strip())
    if longest_streak(hex_lines, bool) >= HEX_TABLE_LINES:
        found.append(Finding("hex-blob", f"{HEX_TABLE_LINES}+ consecutive lines of hex dump"))
    elif hex_count >= HEX_DENSE_LINES and hex_count * 100 >= HEX_DENSE_PERCENT * nonblank:
        found.append(Finding("hex-blob", f"{hex_count} of {nonblank} lines are hex dump"))
    if len(BYTE_TOKEN.findall(text)) > BYTE_TOKEN_LIMIT:
        found.append(Finding("hex-blob", f"over {BYTE_TOKEN_LIMIT} byte values written as 0xNN or \\xNN tokens"))
    for match in DEC_ARRAY.finditer(text):
        if len(set(re.findall(r"\d+", match.group()))) >= DEC_ARRAY_DISTINCT:
            found.append(Finding("hex-blob", "a decimal array of 256+ byte values"))
            break
    if longest_streak(lines, is_base64_line) >= B64_BLOCK_LINES:
        found.append(Finding("base64-blob", f"{B64_BLOCK_LINES}+ consecutive lines of wrapped base64"))
    return found


def is_base64_line(line: str) -> bool:
    """24+ base64 characters with upper case, lower case and a digit or +/ : not a path or a word."""
    body = line.strip()
    return bool(
        B64_LINE.fullmatch(body) and not HEX_ONLY.fullmatch(body)
        and re.search(r"[A-Z]", body) and re.search(r"[a-z]", body) and re.search(r"[0-9+/]", body)
    )


def source_findings(rel: str, size: int, text: str) -> list[Finding]:
    """Recompiled game code is huge and names its functions by address."""
    if suffix_of(rel.rsplit("/", 1)[-1].lower()) not in SOURCE_SUFFIXES:
        return []
    found = []
    if size > MAX_SOURCE_BYTES:
        found.append(Finding("generated-code", f"source file of {size // 1024} KiB, over {MAX_SOURCE_BYTES // 1024} KiB (looks generated)"))
    names = {m.group() for m in ADDRESS_NAME.finditer(text)}
    if len(names) >= ADDRESS_NAMES_LIMIT:
        found.append(Finding("generated-code", f"{len(names)} names built from code addresses (looks like recompiled code)"))
    return found


def suffix_of(name: str) -> str:
    return "." + name.rsplit(".", 1)[-1] if "." in name else ""


def path_findings(rel: str, allow_media: bool) -> list[Finding]:
    """Rules that need only the path. rel is POSIX, relative to the repository."""
    found = []
    parts = rel.split("/")
    name = parts[-1].lower()
    for part in parts[:-1]:
        if part.lower() in BLOCKED_DIRS or part.lower().startswith(BLOCKED_DIR_PREFIXES):
            found.append(Finding("blocked-dir", f"directory '{part}' is where extracted game data lives"))
    if name in BLOCKED_NAMES:
        found.append(Finding("blocked-name", f"'{parts[-1]}' is the name of a Switch executable module or key file"))
    suffix = suffix_of(name)
    if suffix in BLOCKED_SUFFIXES:
        found.append(Finding("blocked-suffix", f"file type '{suffix}' is a game, key, archive or blob format"))
    if suffix in MEDIA_SUFFIXES and not allow_media:
        found.append(Finding("media-suffix", f"'{suffix}' pictures, audio and video are not allowed (reviewed exceptions: ALLOW_BINARY)"))
    return found


def check_file(rel: str, size: int, read: Callable[[], bytes]) -> list[Finding]:
    """Every rule, for one file. read() is only called when the size is acceptable."""
    allow_binary = rel in ALLOW_BINARY
    found = path_findings(rel, allow_binary)
    if size > MAX_BYTES:
        found.append(Finding("size", f"{size // 1024} KiB, over the {MAX_BYTES // 1024} KiB limit"))
        return found
    data = read()
    for magic in detect_magic(data):
        if magic.kind == "media" and allow_binary:
            continue
        found.append(Finding("magic", f"starts like {magic.name} (signature at offset 0x{magic.offset:x})"))
    if GENERATED_MARKER in data:
        found.append(Finding("generated-marker", "marked as generated from game code"))
    text = None
    try:
        if b"\0" in data:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "NUL byte")
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        if not allow_binary:
            found.append(Finding("binary", "binary data (NUL bytes or not UTF-8 text); only reviewed exceptions in ALLOW_BINARY"))
        text = data.decode("utf-8", errors="ignore").replace("\0", "")
    found.extend(key_findings(text))
    if rel not in ALLOW_BLOBS:
        found.extend(blob_findings(text))
        found.extend(source_findings(rel, size, text))
    else:  # a reviewed table of hashes may have arbitrary labels; a PEM block is never exempt
        found = [f for f in found if f.rule != "key-line"]
    return found


class Entry(NamedTuple):
    label: str  # what to print
    rel: str  # path used by the path rules
    size: int
    read: Callable[[], bytes]
    cache_key: str | None = None  # blob id: the same content is judged once


class GitError(RuntimeError):
    pass


def git(root: str, *args: str) -> bytes:
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    except OSError as exc:
        raise GitError(f"cannot run git: {exc}") from exc
    if done.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {done.stderr.decode('utf-8', 'replace').strip()}")
    return done.stdout


def repo_root() -> str:
    return git(os.getcwd(), "rev-parse", "--show-toplevel").decode("utf-8").strip()


def decode_path(raw: bytes) -> str:
    return raw.decode("utf-8", errors="surrogateescape")


def disk_entry(root: str, rel: str) -> Entry | None:
    path = Path(root) / rel
    if not path.is_file():
        return None
    return Entry(rel, rel, path.stat().st_size, path.read_bytes)


def blob_entry(root: str, label: str, rel: str, oid: str) -> Entry:
    size = int(git(root, "cat-file", "-s", oid))
    return Entry(label, rel, size, lambda: git(root, "cat-file", "blob", oid), f"{oid}:{rel}")


def raw_changes(raw: bytes) -> Iterator[tuple[str, str]]:
    """(path, new blob id) pairs from 'git diff --raw -z --no-renames' output."""
    tokens = raw.split(b"\0")
    i = 0
    while i + 1 < len(tokens):
        if not tokens[i].startswith(b":"):
            i += 1
            continue
        fields = tokens[i][1:].split()
        path = decode_path(tokens[i + 1])
        i += 2
        if len(fields) >= 5 and fields[1] != b"160000":  # skip submodule pointers
            yield path, fields[3].decode()


# --diff-filter=d keeps every status except deletions. A type change (T, a
# symlink that becomes a file) and a copy would be lost with AM.
def staged_entries(root: str) -> list[Entry]:
    raw = git(root, "diff", "--cached", "--raw", "-z", "--no-renames", "--diff-filter=d")
    return [blob_entry(root, rel, rel, oid) for rel, oid in raw_changes(raw)]


def range_entries(root: str, spec: str) -> list[Entry]:
    commits = git(root, "rev-list", "--reverse", spec, "--").decode().split()
    entries = []
    for commit in commits:
        raw = git(root, "diff-tree", "-r", "-m", "--root", "--no-commit-id", "--raw", "-z",
                  "--no-renames", "--diff-filter=d", commit)
        seen = set()
        for rel, oid in raw_changes(raw):
            if (rel, oid) not in seen:
                seen.add((rel, oid))
                entries.append(blob_entry(root, f"{commit[:10]}:{rel}", rel, oid))
    return entries


def all_entries(root: str) -> list[Entry]:
    raw = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    paths = sorted({decode_path(p) for p in raw.split(b"\0") if p})
    return [e for e in (disk_entry(root, rel) for rel in paths) if e is not None]


def path_entries(paths: list[str]) -> list[Entry]:
    entries = []
    for arg in paths:
        base = Path(arg)
        if base.is_dir():
            files = sorted(p for p in base.rglob("*") if p.is_file())
            rels = [(p, p.relative_to(base).as_posix()) for p in files]
        elif base.is_file():
            rels = [(base, base.as_posix() if not base.is_absolute() else base.name)]
        else:
            raise GitError(f"no such file or directory: {arg}")
        entries.extend(Entry(str(p), rel, p.stat().st_size, p.read_bytes) for p, rel in rels)
    return entries


def run_checks(entries: list[Entry]) -> list[tuple[str, Finding]]:
    problems = []
    cache: dict[str, list[Finding]] = {}
    for entry in entries:
        if entry.cache_key is not None and entry.cache_key in cache:
            findings = cache[entry.cache_key]
        else:
            findings = check_file(entry.rel, entry.size, entry.read)
            if entry.cache_key is not None:
                cache[entry.cache_key] = findings
        problems.extend((entry.label, finding) for finding in findings)
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="files staged for the next commit")
    mode.add_argument("--range", metavar="REV", help="every commit in A..B (or all history up to REV)")
    mode.add_argument("--all", action="store_true", help="the tracked tree and untracked, unignored files")
    mode.add_argument("--paths", nargs="+", metavar="PATH", help="files or directories on disk")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.paths:
            entries = path_entries(args.paths)
        else:
            root = repo_root()
            if args.staged:
                entries = staged_entries(root)
            elif args.range:
                entries = range_entries(root, args.range)
            else:
                entries = all_entries(root)
        problems = run_checks(entries)
    except GitError as exc:
        print(f"check_content: cannot run the check: {exc}")
        return 2
    for label, finding in problems:
        print(f"{label}: [{finding.rule}] {finding.message}")
    if problems:
        print(f"\ncheck_content: {len(problems)} problem(s) in {len({p[0] for p in problems})} file(s).")
        print("Nothing that came out of the game, and no key, ever goes into this repository.")
        print("See CONTRIBUTING.md. Never bypass this check with --no-verify.")
        return 1
    print(f"check_content: OK, {len(entries)} file(s) checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
