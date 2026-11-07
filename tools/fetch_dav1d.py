#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Fetch dav1d's release source and build it with the compiler the runtime is built with, laid out as a "dav1d kit".

    py -3 tools/fetch_dav1d.py --dest DIR [--cc CC] [--ar AR] [--ninja NINJA] [--no-asm]

The runtime decodes the AV1 of the game's movies with dav1d (architecture D24b), behind its own video decoder service
(runtime/video). dav1d is BSD-2-Clause and none of it is kept in this repository (THIRD_PARTY_NOTICES.md): this script
downloads the release archive, checks it against a pinned SHA-256 before it unpacks anything (a file that fails the
check is deleted), and compiles it with a
generated Ninja file, so the build needs no Meson, no MSYS2 and no package manager.

DIR receives include/dav1d/*.h, lib/libdav1d.a and licenses/ (dav1d's COPYING and PATENTS, and the ISC text of the
x86inc.asm macros the assembly is written with). DAV1D_DIR=DIR (or --dav1d-dir on tools/build_game.py) then serves the
build. DIR/fetch_dav1d.json records the pins and the SHA-256 of every file written; a DIR whose record matches the pins
and whose files still match their hashes is kept as it is, so a cached DIR costs no download and no compile.

Assembly: on x86-64 dav1d's hand-written assembly is about two and a half times faster than its C. It needs NASM
(2.14 or later): on Windows the script fetches NASM 2.16.03's own release archive (pinned SHA-256) into a temporary
folder, on other systems it uses a nasm on PATH when there is one. With none, or with --no-asm, the C-only library is
built; dav1d gives the same pictures either way.

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
import zipfile
from dataclasses import dataclass
from pathlib import Path

RECORD = "fetch_dav1d.json"
VERSION = "1.5.4"


@dataclass(frozen=True)
class Archive:
    url: str
    sha256: str


# VideoLAN's release archive; the .sha256 file beside it on downloads.videolan.org holds the same digest, and so does the
# source of MSYS2's dav1d 1.5.4-1 package.
DAV1D = Archive(f"https://downloads.videolan.org/pub/videolan/dav1d/{VERSION}/dav1d-{VERSION}.tar.xz",
                "686616b7c69eb88d44459391ab25cac13b6647a3b288835c5784e71c1514a5c5")
# NASM's release build for Windows. nasm.us publishes no checksum file: this digest is of the file downloaded on
# 2026-11-07 from the address below (progress notes of the movie player), and every later download is held to it.
NASM_VERSION = "2.16.03"
NASM_WIN = Archive(f"https://www.nasm.us/pub/nasm/releasebuilds/{NASM_VERSION}/win64/nasm-{NASM_VERSION}-win64.zip",
                   "3ee4782247bcb874378d02f7eab4e294a84d3d15f3f6ee2de2f47a46aa7226e6")

# The files of src/ the library is made of (src/meson.build, 1.5.4): compiled once, and the template files once for each
# bit depth. The assembly files are those of x86-64.
CORE = ("cdf cpu ctx data decode dequant_tables getbits intra_edge itx_1d lf_mask lib log mem msac obu pal picture qm ref "
        "refmvs scan tables thread_task warpmv wedge").split()
TEMPLATES = ("cdef_apply cdef fg_apply filmgrain ipred_prepare ipred itx lf_apply loopfilter looprestoration lr_apply mc "
             "recon").split()
ASM_BOTH = "cpuid msac pal refmvs itx_avx512 cdef_avx2 itx_avx2 cdef_sse itx_sse".split()
ASM_8 = ("cdef_avx512 filmgrain_avx512 ipred_avx512 loopfilter_avx512 looprestoration_avx512 mc_avx512 filmgrain_avx2 "
         "ipred_avx2 loopfilter_avx2 looprestoration_avx2 mc_avx2 filmgrain_sse ipred_sse loopfilter_sse "
         "looprestoration_sse mc_sse").split()
ASM_16 = ("cdef16_avx512 filmgrain16_avx512 ipred16_avx512 itx16_avx512 loopfilter16_avx512 looprestoration16_avx512 "
          "mc16_avx512 cdef16_avx2 filmgrain16_avx2 ipred16_avx2 itx16_avx2 loopfilter16_avx2 looprestoration16_avx2 "
          "mc16_avx2 cdef16_sse filmgrain16_sse ipred16_sse itx16_sse loopfilter16_sse looprestoration16_sse "
          "mc16_sse").split()
HEADERS = "common data dav1d headers picture version".split()


class FetchError(Exception):
    """A download, a check or the build failed. The message says which."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pins(asm: bool) -> dict:
    return {"dav1d": [DAV1D.url, DAV1D.sha256], "asm": [NASM_WIN.url, NASM_WIN.sha256] if asm else "none",
            "recipe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def tree_hashes(dest: Path) -> dict[str, str]:
    return {p.relative_to(dest).as_posix(): file_sha256(p) for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name != RECORD}


def is_current(dest: Path, asm: bool) -> bool:
    """dest holds this script's pins (with the assembly or without it as asked), and every file it wrote is still there,
    unchanged, with nothing added."""
    try:
        record = json.loads((dest / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return record.get("pins") == pins(asm) and record.get("files") == tree_hashes(dest)


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


def find_nasm(work: Path, allow: bool, windows: bool | None = None) -> str | None:
    """A NASM of 2.14 or later: the pinned release on Windows, else one on PATH; None for a C-only build."""
    if not allow:
        return None
    if os.name == "nt" if windows is None else windows:
        archive = work / "nasm.zip"
        download(NASM_WIN, archive)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not (work / name).resolve().is_relative_to(work.resolve()):
                    raise FetchError(f"nasm archive: {name} would leave the extraction folder")
            z.extractall(work)
        exe = work / f"nasm-{NASM_VERSION}" / "nasm.exe"
        if not exe.is_file():
            raise FetchError("nasm archive: no nasm.exe where the release puts it")
        return str(exe)
    path = shutil.which("nasm")
    if not path:
        return None
    out = subprocess.run([path, "-v"], capture_output=True, text=True).stdout
    m = re.search(r"version (\d+)\.(\d+)", out)
    return path if m and (int(m[1]), int(m[2])) >= (2, 14) else None


def config_h(windows: bool, asm: bool, x86_64: bool) -> str:
    """config.h as dav1d's meson.build writes it for this target (the meson.build of 1.5.4, read for it)."""
    d = {
        "CONFIG_8BPC": 1, "CONFIG_16BPC": 1, "HAVE_ASM": int(asm), "TRIM_DSP_FUNCTIONS": 0, "CONFIG_LOG": 1,
        "HAVE_CLOCK_GETTIME": int(not windows), "HAVE_SIGACTION": int(not windows),
        "HAVE_POSIX_MEMALIGN": int(not windows), "HAVE_MEMALIGN": int(not windows), "HAVE_ALIGNED_ALLOC": int(not windows),
        "HAVE_DLSYM": 0, "HAVE_SYS_TYPES_H": 1, "HAVE_UNISTD_H": 1, "HAVE_IO_H": int(windows), "HAVE_PTHREAD_NP_H": 0,
        "HAVE_GETAUXVAL": 0, "HAVE_ELF_AUX_INFO": 0,
        "HAVE_PTHREAD_GETAFFINITY_NP": int(not windows), "HAVE_PTHREAD_SETAFFINITY_NP": int(not windows),
        "HAVE_PTHREAD_SETNAME_NP": int(not windows), "HAVE_PTHREAD_SET_NAME_NP": 0, "HAVE_C11_GENERIC": 1,
        "ENDIANNESS_BIG": 0, "ARCH_X86": int(x86_64), "ARCH_X86_64": int(x86_64), "ARCH_X86_32": 0, "ARCH_AARCH64": 0,
        "ARCH_ARM": 0, "ARCH_PPC64LE": 0, "ARCH_RISCV": 0, "ARCH_RV32": 0, "ARCH_RV64": 0, "ARCH_LOONGARCH": 0,
        "ARCH_LOONGARCH32": 0, "ARCH_LOONGARCH64": 0, "HAVE_AS_FUNC": 0, "HAVE_AS_ARCH_DIRECTIVE": 0,
        "HAVE_AS_ARCHEXT_DOTPROD_DIRECTIVE": 0, "HAVE_AS_ARCHEXT_I8MM_DIRECTIVE": 0, "HAVE_AS_ARCHEXT_SVE_DIRECTIVE": 0,
        "HAVE_AS_ARCHEXT_SVE2_DIRECTIVE": 0, "HAVE_DOTPROD": 0, "HAVE_I8MM": 0, "HAVE_SVE": 0, "HAVE_SVE2": 0,
    }
    if windows:   # the Windows block of meson.build
        d.update({"_WIN32_WINNT": "0x0601", "UNICODE": 1, "_UNICODE": 1, "__USE_MINGW_ANSI_STDIO": 1,
                  "_CRT_DECLARE_NONSTDC_NAMES": 1, "_FILE_OFFSET_BITS": 64})
    return "".join(f"#define {k} {v}\n" for k, v in sorted(d.items()))


def config_asm() -> str:
    """config.asm (NASM syntax), as meson writes it for x86-64 (private_prefix dav1d, a 16-byte stack alignment)."""
    return "".join(f"%define {k} {v}\n" for k, v in (("STACK_ALIGNMENT", 16), ("private_prefix", "dav1d"),
                                                      ("ARCH_X86_64", 1), ("ARCH_X86_32", 0), ("PIC", 1),
                                                      ("FORCE_VEX_ENCODING", 0)))


def ninja_text(cc: str, ar: str, nasm: str | None, nasm_format: str, windows: bool, src: Path, out: Path) -> str:
    n = chr(10)
    S = src.as_posix().replace("$", "$$").replace(":", "$:").replace(" ", "$ ")   # a path in a build line
    cflags = (f"-std=c99 -O2 -fvisibility=hidden -fomit-frame-pointer -ffast-math -w -I{src.as_posix()} "
              f"-I{out.as_posix()} -I{src.as_posix()}/include/dav1d -I{src.as_posix()}/include")
    if not windows:
        cflags += " -D_GNU_SOURCE -fPIC"
    lines = [f"cc = {cc}", f"ar = {ar}", f"cflags = {cflags}", "builddir = obj", "",
             "rule cc", "  command = $cc $cflags $extra -MD -MF $out.d -c $in -o $out", "  deps = gcc", "  depfile = $out.d",
             "  description = cc $in", "",
             "rule ar", "  command = $ar rcs $out @$out.rsp", "  rspfile = $out.rsp", "  rspfile_content = $in",
             "  description = ar $out", ""]
    objs = []
    for name in CORE:
        lines += [f"build obj/c_{name}.o: cc {S}/src/{name}.c"]
        objs.append(f"obj/c_{name}.o")
    if windows:
        lines += [f"build obj/win32_thread.o: cc {S}/src/win32/thread.c"]
        objs.append("obj/win32_thread.o")
    for bd in (8, 16):
        for name in TEMPLATES:
            lines += [f"build obj/{name}_{bd}.o: cc {S}/src/{name}_tmpl.c", f"  extra = -DBITDEPTH={bd}"]
            objs.append(f"obj/{name}_{bd}.o")
    if nasm:
        lines += ["", f"nasm = {nasm}", "rule asm",
                  f"  command = $nasm -f {nasm_format} -I {src.as_posix()}/src/ -I {out.as_posix()}/ -MD $out.d -MQ $out $in -o $out",
                  "  deps = gcc", "  depfile = $out.d", "  description = nasm $in", ""]
        lines += [f"build obj/x86_cpu.o: cc {S}/src/x86/cpu.c"]
        objs.append("obj/x86_cpu.o")
        for name in ASM_BOTH + ASM_8 + ASM_16:
            lines += [f"build obj/a_{name}.o: asm {S}/src/x86/{name}.asm | config.asm"]
            objs.append(f"obj/a_{name}.o")
    lines += ["", f"build libdav1d.a: ar {' '.join(objs)}", "default libdav1d.a", ""]
    return n.join(lines)


def build(dest: Path, cc: str, ar: str, ninja: str, asm: bool) -> list[str]:
    """Lay out dest from the pins; return what was done, one line per step. dest must be absent, empty, or a folder this
    script laid out before (it holds the record), which is then replaced."""
    if dest.exists():
        if not dest.is_dir() or (any(dest.iterdir()) and not (dest / RECORD).is_file()):
            raise FetchError(f"{dest} exists and is not a folder this script laid out; give an empty or new folder")
        shutil.rmtree(dest)
    machine = subprocess.run([cc, "-dumpmachine"], capture_output=True, text=True).stdout.strip()
    windows = "mingw" in machine or "windows" in machine
    x86_64 = machine.startswith(("x86_64", "amd64"))
    lines = []
    with tempfile.TemporaryDirectory(prefix="oot2-fetch-dav1d-") as tmp:
        work = Path(tmp)
        archive = work / f"dav1d-{VERSION}.tar.xz"
        download(DAV1D, archive)   # checked against the pin before anything is unpacked
        with tarfile.open(archive) as t:
            t.extractall(work / "src", filter="data")
        src = work / "src" / f"dav1d-{VERSION}"
        lines.append(f"dav1d {VERSION} ({DAV1D.sha256})")
        nasm = find_nasm(work, asm and x86_64)
        if asm and x86_64 and not nasm:
            raise FetchError("no NASM 2.14 or later on PATH: install one, or pass --no-asm for the C-only library")
        use_asm = bool(nasm)
        lines.append(f"assembly: {'NASM ' + NASM_VERSION if use_asm else 'none (C only)'}; target {machine}")
        b = work / "build"
        b.mkdir()
        (b / "config.h").write_text(config_h(windows, use_asm, x86_64), encoding="utf-8", newline="\n")
        (b / "vcs_version.h").write_text(f'#define DAV1D_VERSION "{VERSION}"\n', encoding="utf-8", newline="\n")
        if use_asm:
            (b / "config.asm").write_text(config_asm(), encoding="utf-8", newline="\n")
        fmt = "win64" if windows else "elf64"
        (b / "build.ninja").write_text(ninja_text(cc, ar, nasm, fmt, windows, src, b), encoding="utf-8", newline=chr(10))
        done = subprocess.run([ninja, "-C", str(b)], capture_output=True, text=True)
        if done.returncode:
            raise FetchError(f"building dav1d failed (exit {done.returncode}):\n{(done.stdout + done.stderr)[-2400:]}")
        (dest / "lib").mkdir(parents=True)
        shutil.copyfile(b / "libdav1d.a", dest / "lib" / "libdav1d.a")
        (dest / "include" / "dav1d").mkdir(parents=True)
        for h in HEADERS:
            shutil.copyfile(src / "include" / "dav1d" / f"{h}.h", dest / "include" / "dav1d" / f"{h}.h")
        (dest / "licenses").mkdir()
        shutil.copyfile(src / "COPYING", dest / "licenses" / "dav1d-COPYING.txt")
        patents = src / "doc" / "PATENTS"
        shutil.copyfile(patents, dest / "licenses" / "dav1d-PATENTS.txt")
        if use_asm:   # the ISC licence of x86inc.asm: the header comment of the file, up to the first blank line after it
            head = (src / "src" / "ext" / "x86" / "x86inc.asm").read_text(encoding="utf-8").split("\n\n", 1)[0]
            (dest / "licenses" / "x86inc-ISC.txt").write_text(head + "\n", encoding="utf-8", newline="\n")
        lines.append(f"libdav1d.a built: {(dest / 'lib' / 'libdav1d.a').stat().st_size} bytes")
    (dest / RECORD).write_text(json.dumps({"pins": pins(asm), "asm": use_asm, "files": tree_hashes(dest)}, indent=1) + "\n",
                               encoding="utf-8")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, required=True, help="the folder to lay out (DAV1D_DIR then names it)")
    ap.add_argument("--cc", default=os.environ.get("OOT2_CC") or "clang")
    ap.add_argument("--ar", default=None)
    ap.add_argument("--ninja", default=os.environ.get("OOT2_NINJA") or "ninja")
    ap.add_argument("--no-asm", action="store_true", help="build the C-only library")
    args = ap.parse_args(argv)
    dest = args.dest.resolve()
    asm = not args.no_asm
    if is_current(dest, asm):
        print(f"fetch_dav1d: {dest} already holds the pinned build, unchanged")
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
            print(f"fetch_dav1d: no {what} found (--cc, --ninja, --ar, or PATH)", file=sys.stderr)
            return 2
    try:
        for line in build(dest, cc, ar, ninja, asm):
            print(f"fetch_dav1d: {line}")
    except FetchError as e:
        print(f"fetch_dav1d: {e}", file=sys.stderr)
        return 1
    print(f"fetch_dav1d: {dest} is ready; set DAV1D_DIR to it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
