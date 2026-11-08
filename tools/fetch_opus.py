#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Fetch libopus's release source and build it with the compiler the runtime is built with, laid out as an "Opus kit".

    py -3 tools/fetch_opus.py --dest DIR [--cc CC] [--ar AR] [--ninja NINJA]

The game's hardware Opus decoder (runtime/hos/ov_opus.c) decodes with libopus behind the seam of runtime/audio/opus.h.
libopus is BSD-3-Clause and none of it is kept in this repository (THIRD_PARTY_NOTICES.md): this script downloads the
release archive from Xiph, checks it against a pinned SHA-256 before it unpacks anything (a file that fails the check is
deleted), and compiles it with a generated Ninja file, so the build needs no autotools, no CMake and no package manager.
The digest is the one Xiph publishes in SHA256SUMS.txt beside the archive.

What is built. The floating-point library in plain C (no SIMD, no run-time CPU detection: the decoder's cost at the
game's rates is small), with the encoder as well, because the runtime's test makes its own Opus stream with libopus's
encoder. The deep-learning extensions (DRED, OSCE, deep PLC) are off, as in the release's default configuration. The
list of source files is read from the release's own *.mk files, so a release that renames or adds a file fails here
instead of at the link.

DIR receives include/opus/*.h, lib/libopus.a and licenses/ (libopus's COPYING). OPUS_DIR=DIR (or --opus-dir on
tools/build_game.py) then serves the build. DIR/fetch_opus.json records the pins and the SHA-256 of every file written; a
DIR whose record matches the pins and whose files still match their hashes is kept as it is, so a cached DIR costs no
download and no compile.

Needs network access (first run). Exit codes: 0 done, 1 a download, a check or the build failed, 2 bad command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RECORD = "fetch_opus.json"
VERSION = "1.5.2"
DEFAULT_ENV = "OPUS_DIR"   # the variable tools/build_game.py reads when it is given no --opus-dir


@dataclass(frozen=True)
class Archive:
    url: str
    sha256: str


# Xiph's release archive; the same digest is in SHA256SUMS.txt in the same folder of downloads.xiph.org.
OPUS = Archive(f"https://downloads.xiph.org/releases/opus/opus-{VERSION}.tar.gz",
               "65c1d2f78b9f2fb20082c38cbe47c951ad5839345876e46941612ee87f9a7ce1")

# The variables of the release's *.mk files that name what is compiled: the decoder and encoder of CELT and SILK, the
# floating-point parts of SILK's encoder, and the Opus layer with its floating-point analysis.
SOURCE_LISTS = (("celt_sources.mk", "CELT_SOURCES"), ("silk_sources.mk", "SILK_SOURCES"),
                ("silk_sources.mk", "SILK_SOURCES_FLOAT"), ("opus_sources.mk", "OPUS_SOURCES"),
                ("opus_sources.mk", "OPUS_SOURCES_FLOAT"))
HEADERS = ("opus.h", "opus_defines.h", "opus_types.h", "opus_multistream.h")
# What every file is compiled with (the release's own defaults for a C99 float build with no SIMD): VAR_ARRAYS for the
# temporary buffers, lrint from C99, and the version string that opus_get_version_string reports.
DEFINES = ("OPUS_BUILD", "VAR_ARRAYS", "HAVE_LRINT", "HAVE_LRINTF")


class FetchError(Exception):
    """A download, a check or the build failed. The message says which."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pins() -> dict:
    return {"opus": [OPUS.url, OPUS.sha256], "defines": list(DEFINES),
            "recipe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def tree_hashes(dest: Path) -> dict[str, str]:
    return {p.relative_to(dest).as_posix(): file_sha256(p) for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name != RECORD}


def is_current(dest: Path) -> bool:
    """dest holds this script's pins, and every file it wrote is still there, unchanged, with nothing added."""
    try:
        record = json.loads((dest / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return record.get("pins") == pins() and record.get("files") == tree_hashes(dest)


def download(archive: Archive, path: Path) -> None:
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(archive.url, timeout=120) as r, open(path, "wb") as f:
            for block in iter(lambda: r.read(1 << 20), b""):
                h.update(block)
                f.write(block)
    except OSError as e:
        raise FetchError(f"{archive.url}: download failed: {e}") from None
    if h.hexdigest() != archive.sha256:
        path.unlink(missing_ok=True)   # what was downloaded is not kept
        raise FetchError(f"{archive.url}: SHA-256 {h.hexdigest()}, expected {archive.sha256}")


def mk_list(text: str, name: str) -> list[str]:
    """The files a make variable names: `NAME = \\` and then one file a line, each but the last ending in a backslash, as
    the release's *.mk files write them. FetchError when the variable is not there or names nothing."""
    m = re.search(rf"^{re.escape(name)}[ \t]*=[ \t]*((?:[^\n]*\\\n)*[^\n]*)", text, re.M)
    files = re.findall(r"[A-Za-z0-9_./-]+\.c\b", m.group(1)) if m else []
    if not files:
        raise FetchError(f"the release's source lists name no files for {name}: the recipe has to be read again for this release")
    return files


def source_files(src: Path) -> list[str]:
    """Every source file of the build, relative to the release's folder, in the order of SOURCE_LISTS; each must exist."""
    files: list[str] = []
    for mk, name in SOURCE_LISTS:
        try:
            text = (src / mk).read_text(encoding="utf-8")
        except OSError as e:
            raise FetchError(f"the release has no {mk}: {e}") from None
        for f in mk_list(text, name):
            if f in files:
                raise FetchError(f"{f} is named twice by the release's source lists")
            if not (src / f).is_file():
                raise FetchError(f"{mk} names {f}, which the release does not hold")
            files.append(f)
    return files


def object_name(rel: str) -> str:
    return "obj/" + rel.replace("/", "_")[:-2] + ".o"


def ninja_text(cc: str, ar: str, src: Path, files: list[str]) -> str:
    S = src.as_posix().replace("$", "$$").replace(":", "$:").replace(" ", "$ ")   # a path in a build line
    inc = " ".join(f"-I{src.as_posix()}/{d}" for d in ("include", "celt", "silk", "silk/float", "src"))
    defs = " ".join(f"-D{d}" for d in DEFINES) + f' -DPACKAGE_VERSION=\\"{VERSION}\\"'
    lines = [f"cc = {cc}", f"ar = {ar}", f"cflags = -std=c99 -O2 -w {defs} {inc}", "builddir = obj", "",
             "rule cc", "  command = $cc $cflags -MD -MF $out.d -c $in -o $out", "  deps = gcc", "  depfile = $out.d",
             "  description = cc $in", "",
             "rule ar", "  command = $ar rcs $out @$out.rsp", "  rspfile = $out.rsp", "  rspfile_content = $in",
             "  description = ar $out", ""]
    objs = []
    for f in files:
        lines.append(f"build {object_name(f)}: cc {S}/{f}")
        objs.append(object_name(f))
    lines += ["", f"build libopus.a: ar {' '.join(objs)}", "default libopus.a", ""]
    return chr(10).join(lines)


def build(dest: Path, cc: str, ar: str, ninja: str) -> list[str]:
    """Lay out dest from the pins; return what was done, one line per step. dest must be absent, empty, or a folder this
    script laid out before (it holds the record), which is then replaced."""
    if dest.exists():
        if not dest.is_dir() or (any(dest.iterdir()) and not (dest / RECORD).is_file()):
            raise FetchError(f"{dest} exists and is not a folder this script laid out; give an empty or new folder")
        shutil.rmtree(dest)
    machine = subprocess.run([cc, "-dumpmachine"], capture_output=True, text=True).stdout.strip()
    lines = []
    with tempfile.TemporaryDirectory(prefix="oot2-fetch-opus-") as tmp:
        work = Path(tmp)
        archive = work / f"opus-{VERSION}.tar.gz"
        download(OPUS, archive)   # checked against the pin before anything is unpacked
        try:
            with tarfile.open(archive) as t:
                t.extractall(work / "src", filter="data")
        except tarfile.TarError as e:   # a member that would leave the folder, a link out of it, a damaged archive
            raise FetchError(f"{OPUS.url}: the archive cannot be unpacked safely: {e}") from None
        src = work / "src" / f"opus-{VERSION}"
        lines.append(f"libopus {VERSION} ({OPUS.sha256})")
        files = source_files(src)
        lines.append(f"{len(files)} source files, plain C float build; target {machine}")
        b = work / "build"
        b.mkdir()
        (b / "build.ninja").write_text(ninja_text(cc, ar, src, files), encoding="utf-8", newline=chr(10))
        done = subprocess.run([ninja, "-C", str(b)], capture_output=True, text=True)
        if done.returncode:
            raise FetchError(f"building libopus failed (exit {done.returncode}):\n{(done.stdout + done.stderr)[-2400:]}")
        (dest / "lib").mkdir(parents=True)
        shutil.copyfile(b / "libopus.a", dest / "lib" / "libopus.a")
        (dest / "include" / "opus").mkdir(parents=True)
        for h in HEADERS:
            shutil.copyfile(src / "include" / h, dest / "include" / "opus" / h)
        (dest / "licenses").mkdir()
        shutil.copyfile(src / "COPYING", dest / "licenses" / "opus-COPYING.txt")
        lines.append(f"libopus.a built: {(dest / 'lib' / 'libopus.a').stat().st_size} bytes")
    (dest / RECORD).write_text(json.dumps({"pins": pins(), "files": tree_hashes(dest)}, indent=1) + "\n", encoding="utf-8")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, required=True, help="the folder to lay out (OPUS_DIR then names it)")
    ap.add_argument("--cc", default=os.environ.get("OOT2_CC") or "clang")
    ap.add_argument("--ar", default=None)
    ap.add_argument("--ninja", default=os.environ.get("OOT2_NINJA") or "ninja")
    args = ap.parse_args(argv)
    dest = args.dest.resolve()
    if is_current(dest):
        print(f"fetch_opus: {dest} already holds the pinned build, unchanged")
        return 0
    cc, ninja = shutil.which(args.cc), shutil.which(args.ninja)
    ar = shutil.which(args.ar) if args.ar else None
    if cc and not ar:
        ext = ".exe" if os.name == "nt" else ""
        for d in dict.fromkeys((Path(cc).parent, Path(cc).resolve().parent)):
            if (d / f"llvm-ar{ext}").is_file():
                ar = str(d / f"llvm-ar{ext}")
                break
    ar = ar or shutil.which("llvm-ar") or shutil.which("ar")
    for what, path in (("C compiler", cc), ("Ninja", ninja), ("archiver", ar)):
        if not path:
            print(f"fetch_opus: no {what} found (--cc, --ninja, --ar, or PATH)", file=sys.stderr)
            return 2
    try:
        for line in build(dest, cc, ar, ninja):
            print(f"fetch_opus: {line}")
    except FetchError as e:
        print(f"fetch_opus: {e}", file=sys.stderr)
        return 1
    print(f"fetch_opus: {dest} is ready; set OPUS_DIR to it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
