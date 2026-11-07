#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Build the release bundle (architecture D24c): one folder, and one zip of it, that a player's Windows machine prepares
the game from without installing a toolchain.

    py -3.14 -I tools/bundle.py --out DIR [--cache DIR] [--xlate FILE] [--runtime-lib FILE --dav1d-dir DIR]
                                [--no-zip] [--allow-dirty]

What goes in, by kind (every piece has a row in THIRD_PARTY_NOTICES.md whose first cell names `bundle/<component>/`):

  llvm-mingw   the LLVM-MinGW subset the build uses: clang, clang++, lld, llvm-ar, llvm-nm and llvm-objdump, their
               libraries, the C and C++ headers, libc++ and libunwind (static), compiler-rt's builtins
  python       Python's embeddable distribution, tkinter with Tcl/Tk (from the official installer), numpy, capstone
  ninja        Ninja
  oot2-xlate   the shader translator, built from xlate/ by tools/build_xlate.py (or --xlate FILE), self-contained
  oot2-runtime the runtime library, built here with the bundle's own compiler (tools/build_game.py --make-runtime-lib):
               the runtime, Dear ImGui and Volk inside, the shaders embedded
  dav1d        the AV1 decoder (libdav1d.a with its headers), linked into the program as in a normal build
  sdl3         the window library (libSDL3.a with its headers and notices, the kit tools/fetch_sdl3.py lays out),
               linked into the program as in a normal build
  vulkan-headers  the Vulkan headers the host probe compiles against
  zbic-source  the ZBIC decoder's source archive (hexkyz/zstd at its pinned commit), so that prepare needs no download
  oot2         this repository's tools and runtime headers, the launchers, the licence texts, bundle.json

Nothing from the game goes in: no game file, key, firmware, shader, texture or generated C. The content rules of
tools/check_content.py that name the game (formats, key files, the generated-code marker, private keys) run on every
file of the bundle (guard_file), and the bundle is refused with the file named when one trips. Its size and kinds are
printed and written to CONTENTS.tsv (path, bytes, SHA-256, component); a file whose component has no notices row, or no
licence text in licenses/<component>/, is a named failure and no zip is written.

Every input that is downloaded is checked against a SHA-256 pinned below before it is read; a file that fails is
deleted. The zip is deterministic (sorted entries, fixed dates and modes), so the same inputs give the same file.

Needs Windows (expand.exe and msiexec unpack the installer's Tcl/Tk), network on the first run, and, unless --xlate and
--runtime-lib are given, git, the .NET 10 SDK and Visual Studio's build tools (the translator) and the network for the
kits that tools/fetch_vulkan.py, tools/fetch_imgui.py, tools/fetch_dav1d.py and tools/fetch_sdl3.py make (the last
needs CMake on PATH). Exit codes: 0 done, 1 a download,
a check or a build failed, 2 bad command line.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import bundle_env  # noqa: E402
import check_content  # noqa: E402

BUNDLE_FORMAT = 1
NOTICES = "THIRD_PARTY_NOTICES.md"
DOWNLOAD_TIMEOUT = 120
ZIP_DATE = (2026, 1, 1, 0, 0, 0)


class BundleError(Exception):
    """A failure the maintainer can act on. The message names the file, the component or the input."""


# ------------------------------------------------------------------------------------------------ pinned inputs

@dataclass(frozen=True)
class Pin:
    name: str
    url: str
    sha256: str
    note: str = ""


# Each digest is the one the publisher records where it records one (llvm-mingw: the release asset; PyPI: the file's
# digest) and otherwise the digest of the file downloaded on 2026-11-07 from the address below (the Python installer
# and its embeddable zip: python.org publishes MD5 only; the installer's Authenticode signature, Python Software
# Foundation, was valid). Every later download is held to it.
LLVM_MINGW = Pin("llvm-mingw-20260616-ucrt-x86_64.zip",
                 "https://github.com/mstorsjo/llvm-mingw/releases/download/20260616/"
                 "llvm-mingw-20260616-ucrt-x86_64.zip",
                 "b9b68a4d276e16fa25802aaba458e4638f64b3884c290aaccdc2d87083b6ca35", "clang 22.1.8")
PYTHON_EMBED = Pin("python-3.14.2-embed-amd64.zip",
                   "https://www.python.org/ftp/python/3.14.2/python-3.14.2-embed-amd64.zip",
                   "f05e28d161c6b15af64a7cb7f08b4a22b3a6b03eee71baee24ea557b3bdd5798")
PYTHON_INSTALLER = Pin("python-3.14.2-amd64.exe",
                       "https://www.python.org/ftp/python/3.14.2/python-3.14.2-amd64.exe",
                       "9db919cefe30a0051658c600a9912acb0cd2b872aaf35842c9ec2bf401efa848",
                       "only for its Tcl/Tk package")
NUMPY = Pin("numpy-2.5.4-cp314-cp314-win_amd64.whl",
            "https://files.pythonhosted.org/packages/52/9a/e1e512ebc948d5b9dd33b08736760f0ebbed2848fd4eda1f553088a6dcee/"
            "numpy-2.5.4-cp314-cp314-win_amd64.whl",
            "2ca144f15135b6212a5c47b1e2aeca6e412f102f95a2d5d88d8aec77eb255de3")
CAPSTONE = Pin("capstone-5.0.7-py3-none-win_amd64.whl",
               "https://files.pythonhosted.org/packages/70/39/2138d890a8e827636b9de9924fbc8527fe83e38b6d26605b30ac55e30ebe/"
               "capstone-5.0.7-py3-none-win_amd64.whl",
               "4ab8bcb7da8f221ff45926ca168ca33e76f7237d06fbf3c10780002faa2670e1")
NINJA = Pin("ninja-1.13.0-py3-none-win_amd64.whl",
            "https://files.pythonhosted.org/packages/29/45/c0adfbfb0b5895aa18cec400c535b4f7ff3e52536e0403602fc1a23f7de9/"
            "ninja-1.13.0-py3-none-win_amd64.whl",
            "fb8ee8719f8af47fed145cced4a85f0755dd55d45b2bddaf7431fa89803c5f3e")
# The Tcl/Tk package inside the installer, as the installer's own cabinet holds it: the SHA-256 of that package.
TCLTK_MSI_SHA256 = "e35622a6fa1bd4f1e210133615ca45b1bca3a27db688a42bcd26050d8c9f9466"
PINS = (LLVM_MINGW, PYTHON_EMBED, PYTHON_INSTALLER, NUMPY, CAPSTONE, NINJA)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def fetch(pin: Pin, cache: Path, say: Callable[[str], None] = print) -> Path:
    """The pinned file in cache, downloaded when it is not there; checked against its SHA-256 before it is returned,
    and deleted when it does not match."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / pin.name
    if not path.is_file():
        say(f"bundle: downloading {pin.url}")
        part = path.with_name(path.name + ".part")
        try:
            req = urllib.request.Request(pin.url, headers={"User-Agent": "oot2-native-bundle"})
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r, open(part, "wb") as out:
                shutil.copyfileobj(r, out, 1 << 20)
            os.replace(part, path)
        except OSError as e:
            part.unlink(missing_ok=True)
            raise BundleError(f"cannot download {pin.url}: {e}") from None
    got = sha256_file(path)
    if got != pin.sha256:
        path.unlink(missing_ok=True)
        raise BundleError(f"{pin.name} has SHA-256 {got}, the pin is {pin.sha256}. The file was deleted; nothing was "
                          "read from it.")
    return path


# ------------------------------------------------------------------------------------------------ components

@dataclass(frozen=True)
class Component:
    name: str          # the notices key: a row of THIRD_PARTY_NOTICES.md names `bundle/<name>/`
    what: str


COMPONENTS = {c.name: c for c in (
    Component("llvm-mingw", "the LLVM-MinGW subset"),
    Component("python", "Python's embeddable distribution"),
    Component("tcltk", "tkinter with Tcl/Tk"),
    Component("numpy", "numpy"),
    Component("capstone", "capstone"),
    Component("ninja", "Ninja"),
    Component("oot2-xlate", "the shader translator"),
    Component("oot2-runtime", "the runtime library"),
    Component("dav1d", "the AV1 decoder"),
    Component("sdl3", "the window library"),
    Component("vulkan-headers", "the Vulkan headers"),
    Component("zbic-source", "the ZBIC decoder's source archive"),
    Component("oot2", "this repository's tools, headers and launchers"),
)}


def notice_components(text: str) -> set[str]:
    """The components that have a row: a table row whose first cell names `bundle/<component>/`."""
    found = set()
    for line in text.splitlines():
        if line.startswith("|"):
            first = line.split("|")[1] if line.count("|") > 1 else ""
            found.update(re.findall(r"`bundle/([a-z0-9-]+)/`", first))
    return found


class Stage:
    """The bundle's folder while it is built: every file it receives has a component, and it is hashed at the end."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.owner: dict[str, str] = {}
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, component: str, rel: str, data: bytes | Path, mode: int | None = None) -> None:
        if component not in COMPONENTS:
            raise BundleError(f"{rel}: the component {component!r} is not one of {', '.join(COMPONENTS)}")
        if rel.startswith("/") or ".." in rel.split("/") or rel in self.owner:
            raise BundleError(f"{rel}: not a path of its own inside the bundle")
        dest = self.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, Path):
            shutil.copyfile(data, dest)
        else:
            dest.write_bytes(data)
        self.owner[rel] = component

    def listing(self) -> list[dict]:
        """Every file now in the folder with its size, SHA-256 and component; a file nobody put there is a failure."""
        rows, stray = [], []
        for rel in sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()):
            path = self.root / rel
            if rel not in self.owner:
                stray.append(rel)
                continue
            rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path),
                         "component": self.owner[rel]})
        if stray:
            raise BundleError(f"{len(stray)} file(s) belong to no component: {', '.join(stray[:8])}"
                              f"{', ...' if len(stray) > 8 else ''}")
        return rows


def check_notices(rows: list[dict], notices_text: str) -> None:
    """Refuse the bundle when a file's component has no row in the notices, or no licence text of its own in
    licenses/<component>/ (the component oot2 has the repository's LICENSE). Every such file is named."""
    have = notice_components(notices_text)
    missing_row = [r["path"] for r in rows if r["component"] not in have]
    if missing_row:
        names = sorted({r["component"] for r in rows if r["component"] not in have})
        raise BundleError(f"{len(missing_row)} file(s) have no row in {NOTICES}: " + ", ".join(missing_row[:10]) +
                          (", ..." if len(missing_row) > 10 else "") +
                          f". Add a row whose first cell names `bundle/{names[0]}/`"
                          + (f" (and one each for {', '.join(names[1:])})" if len(names) > 1 else "") + ".")
    with_licence = {r["path"].split("/")[1] for r in rows
                    if r["path"].startswith("licenses/") and r["path"].count("/") >= 2}
    lacking = sorted({r["component"] for r in rows} - with_licence - {"oot2"})
    if lacking:
        raise BundleError(f"no licence text in licenses/<component>/ for: {', '.join(lacking)}")


# ------------------------------------------------------------------------------------------------ the content rules

ARCHIVE_SUFFIXES = {".zip", ".7z", ".rar", ".tar", ".gz", ".xz", ".bz2"}
GAME_SUFFIXES = check_content.BLOCKED_SUFFIXES - ARCHIVE_SUFFIXES - {".bin", ".pack"}
GAME_DIRS = {"exefs", "romfs", "dump", "dumps", "extracted", "captures", "frame-dumps", "shader-cache", "pipeline-cache",
             "fetched"}
ARCHIVE_MAGICS = {"zip archive", "gzip data", "xz data", "7z archive"}
# Archives the bundle carries on purpose, by path; each is pinned by its input's hash.
ALLOWED_ARCHIVES = {"python/python314.zip", "third_party/zbic-source.tar.gz"}
# Reviewed exceptions to the key-assignment rule, by path, each with its reason: the record that tools/fetch_sdl3.py
# writes into its kit lists the pins and the SHA-256 of every file it laid out, one of them labelled "recipe", and
# build_game.find_sdl3 refuses a kit without it. Nothing in it is a key.
HASH_RECORDS = {"runtime/sdl3/fetch_sdl3.json"}
HEAD = 0x400
MARKER = check_content.GENERATED_MARKER
TEXT_LIMIT = check_content.MAX_BYTES


def guard_file(rel: str, path: Path) -> list[str]:
    """The rules of tools/check_content.py that name the game, applied to one bundle file (the rules about size, binary
    content and source shape are for a repository of text and would flag every executable): a game or key file name or
    suffix, a game folder in the path, a game container signature at the start of the file, the generated-code marker
    anywhere in it, and key assignments and private keys in a text file. Returns what tripped, empty when nothing."""
    found = []
    parts = rel.split("/")
    name = parts[-1].lower()
    if name in check_content.BLOCKED_NAMES:
        found.append(f"{rel}: the name of a Switch executable module or key file")
    suffix = check_content.suffix_of(name)
    if suffix in GAME_SUFFIXES:
        found.append(f"{rel}: the file type {suffix} is a game, key or blob format")
    elif suffix in ARCHIVE_SUFFIXES and rel not in ALLOWED_ARCHIVES:
        found.append(f"{rel}: an archive the bundle does not carry on purpose")
    for part in parts[:-1]:
        if part.lower() in GAME_DIRS:
            found.append(f"{rel}: the folder {part!r} is where game data lives")
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(HEAD + 64)
        for magic in check_content.detect_magic(head):
            if magic.kind == "game" and not (magic.name in ARCHIVE_MAGICS and rel in ALLOWED_ARCHIVES):
                found.append(f"{rel}: starts like {magic.name}")
        f.seek(0)
        tail = b""
        while block := f.read(8 << 20):
            if MARKER in tail + block:
                found.append(f"{rel}: marked as generated from game code")
                break
            tail = block[-len(MARKER):]
    if size <= TEXT_LIMIT:
        data = path.read_bytes()
        if b"\0" not in data:
            text = data.decode("utf-8", errors="ignore")
            if rel not in HASH_RECORDS:
                found.extend(f"{rel}: {f.rule}: {f.message}" for f in check_content.key_findings(text))
            if check_content.PRIVATE_KEY.search(text):
                found.append(f"{rel}: holds a private key")
    return found


def guard(root: Path, rows: Iterable[dict]) -> None:
    found = []
    for r in rows:
        found.extend(guard_file(r["path"], root / r["path"]))
    if found:
        raise BundleError("the content rules refuse the bundle:\n  " + "\n  ".join(found[:20]))


# ------------------------------------------------------------------------------------------------ unpacking the inputs

def zip_members(zf: zipfile.ZipFile, strip: str = "") -> Iterable[tuple[str, zipfile.ZipInfo]]:
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if strip:
            if not name.startswith(strip):
                continue
            name = name[len(strip):]
        yield name, info


LLVM_FILES = (
    "LICENSE.TXT",
    "bin/clang.exe", "bin/clang++.exe", "bin/clang-22.exe", "bin/x86_64-w64-mingw32-clang.exe",
    "bin/x86_64-w64-mingw32-clang++.exe", "bin/ld.lld.exe", "bin/llvm-ar.exe", "bin/llvm-ranlib.exe",
    "bin/llvm-nm.exe", "bin/llvm-objdump.exe", "bin/libLLVM-22.dll", "bin/libclang-cpp.dll", "bin/libc++.dll", "bin/libunwind.dll", "bin/libwinpthread-1.dll",
    "bin/libffi-8.dll",
    "bin/mingw32-common.cfg", "bin/x86_64-w64-windows-gnu.cfg", "bin/x86_64-pc-windows-gnu.cfg",
)
LLVM_TREES = (
    "include/",                           # the C library and Windows headers, and libc++'s headers (include/c++/v1)
    "lib/clang/22/include/",              # the compiler's own headers
    "x86_64-w64-mingw32/lib/",            # the C runtime, libc++.a, libunwind.a, the import libraries
    "x86_64-w64-mingw32/bin/",            # libc++.dll, libunwind.dll and the like, beside the tools that start them
    "x86_64-w64-mingw32/share/",          # the licence texts of mingw-w64
)
LLVM_ALSO = ("lib/clang/22/lib/windows/libclang_rt.builtins-x86_64.a",)
# What those trees hold that no build needs: Windows driver headers, the GL and Direct3D headers a game build never
# reads are kept (clang's own headers include <windows.h>, which includes them); only the DDK is dropped.
LLVM_DROP = ("include/ddk/", "x86_64-w64-mingw32/lib/ldscripts/", "x86_64-w64-mingw32/bin/libclang_rt.asan",
             "x86_64-w64-mingw32/bin/libomp.dll")
LLVM_DROP_SUFFIXES = (".idl",)           # interface definitions: the compiler reads the .h files, not these
LLVM_LICENCES = {
    "LICENSE.TXT": "LICENSE.TXT",
}


def llvm_wanted(rel: str) -> bool:
    if rel in LLVM_FILES or rel in LLVM_ALSO:
        return True
    if any(rel.startswith(d) for d in LLVM_DROP) or rel.endswith(LLVM_DROP_SUFFIXES):
        return False
    return any(rel.startswith(t) for t in LLVM_TREES)


def stage_llvm(stage: Stage, archive: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        top = names[0].split("/", 1)[0] + "/"
        seen = set()
        for rel, info in zip_members(zf, top):
            if not llvm_wanted(rel):
                continue
            seen.add(rel)
            stage.put("llvm-mingw", f"llvm/{rel}", zf.read(info))
            if rel.startswith("x86_64-w64-mingw32/share/") and "COPYING" in rel:
                stage.put("llvm-mingw", f"licenses/llvm-mingw/{rel.rsplit('/', 1)[-1]}", zf.read(info))
        stage.put("llvm-mingw", "licenses/llvm-mingw/LICENSE.TXT", zf.read(top + "LICENSE.TXT"))
        lacking = [f for f in LLVM_FILES + LLVM_ALSO if f not in seen]
        if lacking:
            raise BundleError(f"{archive.name} does not hold {', '.join(lacking)}: the subset's list is out of date")


PYTHON_PTH = "python314.zip\r\n.\r\nLib\r\nLib\\site-packages\r\n"
PYTHON_DROP = {"python314._pth", "python.cat"}
TK_DLLS = ("_tkinter.pyd", "tcl86t.dll", "tk86t.dll", "zlib1.dll")
TK_TREES = ("tcl/tcl8/", "tcl/tcl8.6/", "tcl/tk8.6/", "tcl/dde1.4/", "tcl/reg1.3/")
TK_DROP = ("tcl/tk8.6/demos/",)


def stage_python(stage: Stage, embed: Path) -> None:
    with zipfile.ZipFile(embed) as zf:
        for rel, info in zip_members(zf):
            if rel in PYTHON_DROP:
                continue
            stage.put("python", f"python/{rel}", zf.read(info))
            if rel == "LICENSE.txt":
                stage.put("python", "licenses/python/LICENSE.txt", zf.read(info))
    stage.put("python", "python/python314._pth", PYTHON_PTH.encode("ascii"))


def carve_installer_cab(installer: Path) -> bytes:
    """The payload cabinet inside the Python installer (a Burn bundle): the last cabinet in the file that runs to near
    its end. The installer's other cabinet is the bootstrapper's own."""
    data = installer.read_bytes()
    best = None
    for m in re.finditer(b"MSCF", data):
        start = m.start()
        size = struct.unpack_from("<I", data, start + 8)[0]
        if start + size <= len(data) and (best is None or size > best[1]):
            best = (start, size)
    if best is None:
        raise BundleError(f"{installer.name} holds no cabinet: this is not the installer the bundle expects")
    return data[best[0]:best[0] + best[1]]


def tcltk_tree(installer: Path, work: Path) -> Path:
    """The Tcl/Tk package of the Python installer, unpacked without installing anything (expand.exe on the cabinet,
    msiexec /a on the package), into work. The package is found by its contents and checked against TCLTK_MSI_SHA256."""
    done = work / "tcltk"
    if (done / "DLLs" / "_tkinter.pyd").is_file():
        return done
    if os.name != "nt":
        raise BundleError("unpacking the installer's Tcl/Tk needs Windows (expand.exe and msiexec)")
    shutil.rmtree(work, ignore_errors=True)
    out = work / "cab"
    out.mkdir(parents=True)
    cab = work / "payload.cab"
    cab.write_bytes(carve_installer_cab(installer))
    r = subprocess.run(["expand.exe", "-F:*", str(cab), str(out)], capture_output=True, text=True, errors="replace")
    if r.returncode:
        raise BundleError(f"expand.exe failed on the installer's cabinet: {(r.stderr or r.stdout)[-300:]}")
    candidates = [p for p in sorted(out.iterdir()) if p.is_file() and sha256_file(p) == TCLTK_MSI_SHA256]
    if not candidates:
        raise BundleError(f"no package of the installer has the Tcl/Tk SHA-256 {TCLTK_MSI_SHA256}")
    msi = work / "tcltk.msi"
    shutil.copyfile(candidates[0], msi)
    r = subprocess.run(["msiexec.exe", "/a", str(msi), "/qn", f"TARGETDIR={done}"], capture_output=True, text=True,
                       errors="replace")
    if r.returncode or not (done / "DLLs" / "_tkinter.pyd").is_file():
        raise BundleError(f"msiexec could not unpack the Tcl/Tk package (exit code {r.returncode})")
    return done


def stage_tcltk(stage: Stage, tree: Path) -> None:
    for name in TK_DLLS:
        stage.put("tcltk", f"python/{name}", tree / "DLLs" / name)
    for path in sorted((tree / "Lib" / "tkinter").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            stage.put("tcltk", "python/Lib/tkinter/" + path.relative_to(tree / "Lib" / "tkinter").as_posix(), path)
    for path in sorted((tree / "tcl").rglob("*")):
        if not path.is_file():
            continue
        rel = "tcl/" + path.relative_to(tree / "tcl").as_posix()
        if any(rel.startswith(t) for t in TK_TREES) and not any(rel.startswith(d) for d in TK_DROP):
            stage.put("tcltk", f"python/{rel}", path)
    # Tk's own terms (the same text covers Tcl, which carries none of its own in this package) and zlib's licence.
    stage.put("tcltk", "licenses/tcltk/license.terms", tree / "tcl" / "tk8.6" / "license.terms")
    stage.put("tcltk", "licenses/tcltk/zlib-license.txt", ROOT / "licenses" / "zlib-license.txt")


def stage_wheel(stage: Stage, component: str, wheel: Path, dest: str, skip: Callable[[str], bool],
                licences: Callable[[str], str | None]) -> None:
    with zipfile.ZipFile(wheel) as zf:
        for rel, info in zip_members(zf):
            if skip(rel):
                continue
            stage.put(component, f"{dest}/{rel}", zf.read(info))
            lic = licences(rel)
            if lic:
                stage.put(component, f"licenses/{component}/{lic}", zf.read(info))


def stage_numpy(stage: Stage, wheel: Path) -> None:
    prefix = "numpy-2.5.4.dist-info/licenses/"
    stage_wheel(stage, "numpy", wheel, "python/Lib/site-packages",
                lambda r: "/tests/" in r or r.startswith("numpy/_pyinstaller/") or r.endswith(".pyc"),
                lambda r: r[len(prefix):].replace("/", "_") if r.startswith(prefix) else None)


def stage_capstone(stage: Stage, wheel: Path) -> None:
    stage_wheel(stage, "capstone", wheel, "python/Lib/site-packages", lambda r: r.endswith(".pyc"),
                lambda r: "LICENSE.TXT" if r.endswith("dist-info/LICENSE.TXT") else None)


def stage_ninja(stage: Stage, wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        stage.put("ninja", "bin/ninja.exe", zf.read("ninja-1.13.0.data/scripts/ninja.exe"))
        stage.put("ninja", "licenses/ninja/LICENSE_Apache_20", zf.read("ninja-1.13.0.dist-info/licenses/LICENSE_Apache_20"))
        stage.put("ninja", "licenses/ninja/AUTHORS.rst", zf.read("ninja-1.13.0.dist-info/licenses/AUTHORS.rst"))


# ------------------------------------------------------------------------------------------------ this repository's part

TOOLS_FILES = ("oot2.py", "oot2_setup.pyw", "bundle_env.py", "build_game.py", "layout.py", "fetch_imgui.py",
               "check_content.py", "fetch_sdl3.py")
TOOLS_TREES = ("installer", "prepare", "lift", "shaders")
SKIP_PARTS = {"__pycache__"}
LAUNCHERS = "tools/bundle_files"


def git_files(root: Path, *paths: str) -> list[str]:
    r = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", *paths], capture_output=True)
    if r.returncode:
        raise BundleError(f"git ls-files failed: {r.stderr.decode(errors='replace')[-200:]}")
    return [p for p in r.stdout.decode("utf-8").split("\0") if p]


def stage_oot2(stage: Stage, root: Path) -> None:
    """The tools prepare runs, the runtime headers, the launchers and the repository's licence texts, taken from git's
    own list so that nothing untracked gets in."""
    tracked = set(git_files(root))
    wanted = [f"tools/{f}" for f in TOOLS_FILES]
    wanted += sorted(p for p in tracked if p.split("/")[0] == "tools" and len(p.split("/")) > 2 and
                     p.split("/")[1] in TOOLS_TREES and not SKIP_PARTS & set(p.split("/")))
    wanted += sorted(p for p in tracked if p.startswith("runtime/include/"))
    wanted += ["LICENSE", NOTICES] + sorted(p for p in tracked if p.startswith("licenses/"))
    for rel in wanted:
        if rel not in tracked:
            raise BundleError(f"{rel} is not a tracked file of the repository (git ls-files)")
        stage.put("oot2", rel, root / rel)
    for p in sorted(tracked):
        if p.startswith(LAUNCHERS + "/"):
            name = p[len(LAUNCHERS) + 1:]
            text = (root / p).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            stage.put("oot2", name, text)


# ------------------------------------------------------------------------------------------------ the built pieces

def run(cmd: list[str], say: Callable[[str], None], env: dict | None = None, cwd: Path | None = None) -> None:
    say("bundle: " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd)
    if r.returncode:
        raise BundleError(f"{Path(str(cmd[1] if len(cmd) > 1 else cmd[0])).name} failed (exit code {r.returncode})")


def build_runtime_lib(stage_root: Path, cache: Path, say: Callable[[str], None]) -> tuple[Path, Path, Path, Path]:
    """The runtime library, built with the bundle's own compiler and Ninja (so that the toolchain subset is proven by
    building the runtime), from the kits tools/fetch_vulkan.py, tools/fetch_imgui.py and tools/fetch_dav1d.py make.
    Returns the library, the dav1d kit, the SDL3 kit and the Vulkan SDK folder."""
    cc, ninja = stage_root / "llvm" / "bin" / "clang.exe", stage_root / "bin" / "ninja.exe"
    env = {k: v for k, v in os.environ.items() if k not in ("VULKAN_SDK", "VK_SDK_PATH", "DAV1D_DIR", "PYTHONPATH",
                                                            "PYTHONHOME") and not k.startswith("OOT2_")}
    env["PATH"] = os.pathsep.join([str(stage_root / "llvm" / "bin"), str(stage_root / "bin"), env.get("PATH", "")])
    py = [sys.executable, "-I"]
    vulkan, imgui, dav1d, out = cache / "vulkan-kit", cache / "imgui", cache / "dav1d-kit", cache / "runtime-lib"
    sdl3 = cache / "sdl3-kit"
    run([*py, ROOT / "tools" / "fetch_vulkan.py", "--dest", vulkan], say)
    run([*py, ROOT / "tools" / "fetch_imgui.py", "--dest", imgui], say)
    run([*py, ROOT / "tools" / "fetch_dav1d.py", "--dest", dav1d, "--cc", cc, "--ninja", ninja], say, env)
    run([*py, ROOT / "tools" / "fetch_sdl3.py", "--dest", sdl3, "--cc", cc, "--ninja", ninja], say, env)
    shutil.rmtree(out, ignore_errors=True)
    run([*py, ROOT / "tools" / "build_game.py", "--make-runtime-lib", "--out", out, "--cc", cc, "--ninja", ninja,
         "--vulkan-sdk", vulkan, "--imgui", imgui, "--dav1d-dir", dav1d, "--sdl3-dir", sdl3], say, env)
    return out / "liboot2rt.a", dav1d, sdl3, vulkan


def build_translator(cache: Path, say: Callable[[str], None]) -> Path:
    out = cache / "xlate"
    run([sys.executable, "-I", ROOT / "tools" / "build_xlate.py", "--out", out], say)
    hits = sorted((out / "publish").rglob("oot2-xlate.exe"))
    if not hits:
        raise BundleError(f"build_xlate.py left no oot2-xlate.exe under {out / 'publish'}")
    return hits[0]


def stage_runtime(stage: Stage, lib: Path, dav1d: Path, sdl3: Path, vulkan: Path) -> None:
    stage.put("oot2-runtime", bundle_env.RUNTIME_LIB, lib)
    for rel in ("lib/libdav1d.a", "include/dav1d/dav1d.h", "include/dav1d/common.h", "include/dav1d/data.h",
                "include/dav1d/headers.h", "include/dav1d/picture.h", "include/dav1d/version.h"):
        if (dav1d / rel).is_file():
            stage.put("dav1d", f"{bundle_env.DAV1D_KIT}/{rel}", dav1d / rel)
    for path in sorted((dav1d / "licenses").rglob("*")):
        if path.is_file():
            stage.put("dav1d", f"licenses/dav1d/{path.name}", path)
    # The SDL3 kit whole: build_game.find_sdl3 refuses a folder that is not exactly what fetch_sdl3.py laid out.
    for path in sorted(p for p in sdl3.rglob("*") if p.is_file()):
        rel = path.relative_to(sdl3).as_posix()
        stage.put("sdl3", f"{bundle_env.SDL3_KIT}/{rel}", path)
        if rel.startswith("licenses/"):
            stage.put("sdl3", f"licenses/sdl3/{rel[len('licenses/'):]}", path)
    inc = vulkan / "Include"
    for sub in ("vulkan", "vk_video"):
        for path in sorted((inc / sub).rglob("*")):
            if path.is_file():
                stage.put("vulkan-headers", f"{bundle_env.VULKAN_SDK}/Include/{sub}/" +
                          path.relative_to(inc / sub).as_posix(), path)
    for path in sorted((vulkan / "Licenses").rglob("*")) if (vulkan / "Licenses").is_dir() else []:
        if path.is_file():
            stage.put("vulkan-headers", f"licenses/vulkan-headers/{path.name}", path)
    for name in ("imgui-MIT.txt", "imgui-stb-MIT.txt", "proggyclean-MIT.txt", "proggyforever-MIT.txt"):
        stage.put("oot2-runtime", f"licenses/oot2-runtime/{name}", ROOT / "licenses" / name)
    stage.put("oot2-runtime", "licenses/oot2-runtime/LICENSE", ROOT / "LICENSE")


def stage_translator(stage: Stage, exe: Path) -> None:
    stage.put("oot2-xlate", bundle_env.XLATE, exe)
    stage.put("oot2-xlate", "licenses/oot2-xlate/LICENSE", ROOT / "LICENSE")
    stage.put("oot2-xlate", "licenses/oot2-xlate/ryujinx-LICENSE.txt",
              ROOT / "xlate" / "third_party" / "ryujinx" / "LICENSE.txt")


def stage_zbic(stage: Stage, cache: Path, say: Callable[[str], None]) -> None:
    """The decoder's source archive, fetched and checked by the code prepare uses (tools/prepare/zbic.py), kept as the
    tar.gz it checked, with the licence files of the fork."""
    sys.path.insert(0, str(ROOT / "tools"))
    from prepare import zbic
    path = cache / "zbic-source.tar.gz"
    if not path.is_file():
        say(f"bundle: downloading {zbic.URL}")
        path.write_bytes(zbic.download())
    data = path.read_bytes()
    got, files = zbic.contents_sha256(data)
    if got != zbic.CONTENTS_SHA256:
        path.unlink(missing_ok=True)
        raise BundleError(f"the ZBIC source archive has contents hash {got}, the pin is {zbic.CONTENTS_SHA256}")
    stage.put("zbic-source", bundle_env.ZBIC_ARCHIVE, data)
    for name in ("LICENSE", "COPYING"):
        if name in files:
            stage.put("zbic-source", f"licenses/zbic-source/{name}", files[name])


# ------------------------------------------------------------------------------------------------ the bundle

def readme_and_manifest(stage: Stage, version: str, commit: str, rows: list[dict]) -> dict:
    return {"bundle": BUNDLE_FORMAT, "version": version, "source_commit": commit,
            "pins": {p.name: {"url": p.url, "sha256": p.sha256} for p in PINS},
            "files": rows, "total_bytes": sum(r["bytes"] for r in rows)}


def contents_tsv(rows: list[dict], notices_text: str) -> str:
    have = notice_components(notices_text)
    out = ["path\tbytes\tsha256\tcomponent\tnotices_row"]
    out += [f"{r['path']}\t{r['bytes']}\t{r['sha256']}\t{r['component']}\t"
            f"{'bundle/' + r['component'] + '/' if r['component'] in have else 'NONE'}" for r in rows]
    return "\n".join(out) + "\n"


def summary(rows: list[dict]) -> list[str]:
    by: dict[str, list[int]] = {}
    for r in rows:
        n = by.setdefault(r["component"], [0, 0])
        n[0] += 1
        n[1] += r["bytes"]
    lines = [f"  {c:16s} {n:6d} files {b / (1 << 20):9.1f} MiB  {COMPONENTS[c].what}" for c, (n, b) in sorted(by.items())]
    lines.append(f"  {'total':16s} {sum(n for n, _ in by.values()):6d} files "
                 f"{sum(b for _, b in by.values()) / (1 << 20):9.1f} MiB")
    return lines


def write_zip(root: Path, dest: Path) -> None:
    """A deterministic zip of root: entries sorted, dates and modes fixed, one compression level."""
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            info = zipfile.ZipInfo(f"{root.name}/{path.relative_to(root).as_posix()}", ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.create_system = 3
            zf.writestr(info, path.read_bytes(), compresslevel=9)


def finish(stage: Stage, version: str, commit: str, notices_text: str, say: Callable[[str], None] = print) -> list[dict]:
    """Write bundle.json and CONTENTS.tsv last, then refuse the bundle when a file has no component, no notices row or
    no licence text, or trips the content rules. Returns the listing."""
    rows = stage.listing()
    check_notices(rows, notices_text)
    guard(stage.root, rows)
    stage.put("oot2", "CONTENTS.tsv", contents_tsv(rows, notices_text).encode("utf-8"))
    manifest = readme_and_manifest(stage, version, commit, rows)
    stage.put("oot2", bundle_env.MANIFEST, (json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    for line in summary(rows):
        say(line)
    return rows


def build(root: Path, out: Path, cache: Path, xlate: Path | None, runtime_lib: Path | None,
          dav1d_dir: Path | None, sdl3_dir: Path | None, vulkan_dir: Path | None, make_zip: bool, say: Callable[[str], None] = print) -> Path:
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or "unknown"
    version = f"oot2-native-bundle-{commit}"
    stage_dir = out / version
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage = Stage(stage_dir)
    notices_text = (root / NOTICES).read_text(encoding="utf-8")
    inputs = {p.name: fetch(p, cache / "inputs", say) for p in PINS}
    say("bundle: the LLVM-MinGW subset")
    stage_llvm(stage, inputs[LLVM_MINGW.name])
    say("bundle: Python, tkinter, numpy, capstone, Ninja")
    stage_python(stage, inputs[PYTHON_EMBED.name])
    stage_tcltk(stage, tcltk_tree(inputs[PYTHON_INSTALLER.name], cache / "tcltk-work"))
    stage_numpy(stage, inputs[NUMPY.name])
    stage_capstone(stage, inputs[CAPSTONE.name])
    stage_ninja(stage, inputs[NINJA.name])
    if runtime_lib is None:
        say("bundle: the runtime library, built with the bundle's own compiler")
        runtime_lib, dav1d_dir, sdl3_dir, vulkan_dir = build_runtime_lib(stage_dir, cache, say)
    elif dav1d_dir is None or sdl3_dir is None or vulkan_dir is None:
        raise BundleError("--runtime-lib needs --dav1d-dir, --sdl3-dir and --vulkan-sdk as well (the kits it was built from)")
    stage_runtime(stage, runtime_lib, dav1d_dir, sdl3_dir, vulkan_dir)
    say("bundle: the shader translator")
    stage_translator(stage, xlate or build_translator(cache, say))
    stage_zbic(stage, cache, say)
    stage_oot2(stage, root)
    rows = finish(stage, version, commit, notices_text, say)
    say(f"bundle: {len(rows)} files in {stage_dir}")
    if make_zip:
        dest = out / f"{version}-windows-x64.zip"
        write_zip(stage_dir, dest)
        say(f"bundle: {dest} ({dest.stat().st_size / (1 << 20):.1f} MiB, SHA-256 {sha256_file(dest)})")
        return dest
    return stage_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="where the bundle folder and its zip go")
    ap.add_argument("--cache", type=Path, help="downloaded inputs and built pieces (default: OUT/cache)")
    ap.add_argument("--xlate", type=Path, help="a built oot2-xlate.exe (default: build it with tools/build_xlate.py)")
    ap.add_argument("--runtime-lib", type=Path, help="a runtime library built before (default: build it)")
    ap.add_argument("--dav1d-dir", type=Path, help="the dav1d kit that library was built with")
    ap.add_argument("--sdl3-dir", type=Path, help="the SDL3 kit that library was built with")
    ap.add_argument("--vulkan-sdk", type=Path, help="the Vulkan SDK folder that library was built with")
    ap.add_argument("--no-zip", action="store_true", help="leave the folder, write no zip")
    ap.add_argument("--allow-dirty", action="store_true", help="build from a working tree with uncommitted changes")
    args = ap.parse_args(argv)
    try:
        status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip()
        if status and not args.allow_dirty:
            raise BundleError("the working tree has uncommitted changes; the bundle is built from a commit "
                              "(--allow-dirty builds from the tree as it is)")
        cache = (args.cache or args.out / "cache").resolve()
        build(ROOT, args.out.resolve(), cache, args.xlate, args.runtime_lib, args.dav1d_dir, args.sdl3_dir, args.vulkan_sdk,
              not args.no_zip)
    except BundleError as e:
        print(f"bundle: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
