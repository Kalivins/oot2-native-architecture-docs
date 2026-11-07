#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Fetch SDL's release source and build it with the compiler the runtime is built with, laid out as an "SDL3 kit".

    py -3 tools/fetch_sdl3.py --dest DIR [--cc CC] [--cxx CXX] [--ar AR] [--ninja NINJA] [--cmake CMAKE]

The runtime opens its window, creates the Vulkan surface and reads the keyboard, the mouse and the game controllers with
SDL3 (runtime/platform, runtime/gpu). SDL is zlib-licensed and none of it is kept in this repository
(THIRD_PARTY_NOTICES.md): this script downloads the release archive, checks it against a pinned SHA-256 before it unpacks
anything (a file that fails the check is deleted), and builds the static library with CMake and Ninja and the given
compiler. The target is the compiler's own, and it must be the machine the script runs on (Windows with a mingw-w64
compiler, or Linux): there are no cross builds.

DIR receives include/SDL3/*.h, lib/libSDL3.a and licenses/ (SDL's own zlib text and the notices of the other code that
the static library holds, named in licenses/README-sdl3.txt). SDL3_DIR=DIR (or --sdl3-dir on tools/build_game.py and
tools/oot2.py) then serves the build. DIR/fetch_sdl3.json records the pins and the SHA-256 of every file written; a DIR
whose record matches the pins and whose files still match their hashes is kept as it is, so a cached DIR costs no
download and no build.

What is built. Static library only; tests, examples and the SDL test library are off. The subsystems the game uses stay
on: video (Windows' own window driver, plus the offscreen and dummy drivers, and the Vulkan support), events, audio,
joysticks and game controllers (XInput, DirectInput, Windows.Gaming.Input, raw input and HIDAPI, which is what lets a
Switch or a DualSense pad speak its own protocol). Off, because the game never asks for them and each is code and system
libraries to link: the render API (the game draws with Vulkan itself), the GPU API, camera, haptic, sensor and system
tray, and the OpenGL and OpenGL ES context support. The same command builds the same kit on every Windows machine; the
Linux path (X11 and Wayland through the usual development packages) follows the same recipe and is NOT verified.

Linking. The program links DIR/lib/libSDL3.a followed by LINK_LIBS, the system libraries the static library needs. The
script reads them from the sdl3.pc that the build installs and refuses the kit when they differ from the constant below
(on Windows exactly; on Linux the .pc must name nothing the constant lacks), so a new SDL release that wants another
library fails here rather than at the link. -mwindows, which the .pc also carries, is not among them: the choice of
console or window subsystem belongs to the program's link, not to this kit.

Needs network access (first run), CMake and Ninja. Exit codes: 0 done, 1 a download, a check or the build failed, 2 bad
command line.
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

RECORD = "fetch_sdl3.json"
VERSION = "3.4.20"
DEFAULT_ENV = "SDL3_DIR"   # the variable tools/build_game.py reads when it is given no --sdl3-dir


@dataclass(frozen=True)
class Archive:
    url: str
    sha256: str


# libsdl.org's release archive on GitHub. The digest is GitHub's own record of the asset, and it was checked against the
# file downloaded on 2026-11-07 from the address below (progress notes of ticket 127): every later download is held to it.
SDL3 = Archive(f"https://github.com/libsdl-org/SDL/releases/download/release-{VERSION}/SDL3-{VERSION}.tar.gz",
               "f3bb06f55bed9557227a700af2ea62e49497ebdf58037732a731a5fb3ef8581f")

# The system libraries a program that links libSDL3.a statically must add, in the order sdl3.pc lists them (read from the
# 3.4.20 build of this recipe on Windows with llvm-mingw; the .pc is checked against this tuple at build time).
WINDOWS_LINK_LIBS = ("-lm", "-lkernel32", "-luser32", "-lgdi32", "-lwinmm", "-limm32", "-lole32", "-loleaut32",
                     "-lversion", "-luuid", "-ladvapi32", "-lsetupapi", "-lshell32", "-ldinput8")
# Linux: not verified here. SDL loads X11, Wayland, ALSA, PulseAudio and udev with dlopen when built this way, so what is
# left is the C library's own pieces; a .pc that names less than this is accepted.
LINUX_LINK_LIBS = ("-lm", "-ldl", "-lpthread", "-lrt")
LINK_LIBS = WINDOWS_LINK_LIBS if os.name == "nt" else LINUX_LINK_LIBS

# CMake options, beyond the compiler and the build type. See the module's text for why each subsystem is off.
OPTIONS = (
    "SDL_STATIC=ON", "SDL_SHARED=OFF", "SDL_TESTS=OFF", "SDL_TEST_LIBRARY=OFF", "SDL_EXAMPLES=OFF",
    "SDL_INSTALL_TESTS=OFF", "SDL_INSTALL=ON", "SDL_UNINSTALL=OFF", "SDL_CCACHE=OFF", "SDL_WERROR=OFF",
    "SDL_VIDEO=ON", "SDL_VULKAN=ON", "SDL_AUDIO=ON", "SDL_JOYSTICK=ON", "SDL_HIDAPI=ON", "SDL_POWER=ON", "SDL_DIALOG=ON",
    "SDL_RENDER=OFF", "SDL_GPU=OFF", "SDL_CAMERA=OFF", "SDL_HAPTIC=OFF", "SDL_SENSOR=OFF", "SDL_TRAY=OFF",
    "SDL_OPENGL=OFF", "SDL_OPENGLES=OFF",
)

# The licence texts of what libSDL3.a holds besides SDL's own code (zlib), and where each is taken from in the source
# tree. Each is code the Windows build really compiles (checked on 3.4.20 by the size of its object in the library):
# fdlibm in src/libm, the stb image loader in src/video/SDL_stb.c, yuv2rgb in src/video/yuv2rgb, HIDAPI in
# src/hidapi. SDL's own qsort, hash and iconv are its own code under its own licence.
LICENCE_FILES = {
    "SDL3-zlib.txt": "LICENSE.txt",
    "hidapi-BSD-3-Clause.txt": "src/hidapi/LICENSE-bsd.txt",
    "yuv2rgb-BSD-3-Clause.txt": "src/video/yuv2rgb/LICENSE",
}
README_NAME = "README-sdl3.txt"
README = f"""\
SDL {VERSION} in this kit, and the licences of the code compiled into libSDL3.a.

SDL3-zlib.txt             SDL itself (Simple DirectMedia Layer), the zlib licence.
hidapi-BSD-3-Clause.txt   HIDAPI (Alan Ott, Signal 11 Software), src/hidapi: game controller access over USB and
                          Bluetooth. HIDAPI is offered under three licences; the BSD-style one is relied on.
yuv2rgb-BSD-3-Clause.txt  yuv2rgb (Adrien Descamps), src/video/yuv2rgb: colour conversion of video surfaces.
stb-MIT-or-Unlicense.txt  stb_image and stb_image_write (Sean Barrett), src/video/SDL_stb.c: the PNG and JPEG loader and
                          the PNG writer. Offered under the MIT licence (alternative A) or as public domain
                          (alternative B); the MIT text is relied on.
fdlibm-SunPro.txt         fdlibm (Sun Microsystems), src/libm: the math functions SDL carries for itself.

Public domain and needing no notice: the MurmurHash3 of src/stdlib/SDL_murmur3.c. SDL's own qsort (stable on every
platform), iconv, CRC and string code are SDL's, under the zlib licence. The Khronos Vulkan headers that SDL builds
against (src/video/khronos, Apache-2.0 OR MIT) give declarations and constants only, no code.

Other targets compile less of this: a Linux build uses the C library's maths, for one. The Windows set is the full one.
"""

STB_START = "This software is available under 2 licenses"


class FetchError(Exception):
    """A download, a check or the build failed. The message says which."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pins() -> dict:
    """What the record must hold for a kit to be this script's: the archive, the options and the libraries to link."""
    return {"sdl3": [SDL3.url, SDL3.sha256], "options": list(OPTIONS), "link_libs": list(LINK_LIBS)}


def recipe_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def tree_hashes(dest: Path) -> dict[str, str]:
    return {p.relative_to(dest).as_posix(): file_sha256(p) for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name != RECORD}


def problems(dest: Path) -> list[str]:
    """Why dest is not the layout this script lays out, one line each; an empty list when it is: its record names this
    script's pins and link libraries, the files of the layout are there, and every file is what was recorded, with
    nothing added. (A kit of an older recipe with the same pins is still a kit: is_current says whether to rebuild.)"""
    if not dest.is_dir():
        return [f"{dest} is not a folder"]
    try:
        record = json.loads((dest / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [f"{dest / RECORD} is missing or unreadable: fetch with tools/fetch_sdl3.py"]
    if record.get("pins") != pins():
        return [f"{dest} was made for other pins than this tree's (SDL {VERSION}, its link libraries, its options): "
                "fetch it again"]
    layout = ("include/SDL3/SDL.h", "include/SDL3/SDL_vulkan.h", "lib/libSDL3.a",
              *(f"licenses/{n}" for n in (*LICENCE_FILES, "fdlibm-SunPro.txt", "stb-MIT-or-Unlicense.txt", README_NAME)))
    for name in layout:
        if not (dest / name).is_file():
            return [f"{dest / name} is missing from the kit"]
    have = tree_hashes(dest)
    want = record.get("files")
    if not isinstance(want, dict):
        return [f"{dest / RECORD} lists no files"]
    for name in sorted(want.keys() | have.keys()):
        if name not in have:
            return [f"{name} is recorded and missing from {dest}"]
        if name not in want:
            return [f"{name} is in {dest} and is not one of the files the kit was laid out with"]
        if have[name] != want[name]:
            return [f"{name} in {dest} differs from what was laid out (SHA-256 {have[name]}, recorded {want[name]})"]
    return []


def is_current(dest: Path) -> bool:
    """dest is a good kit (problems) made by this very recipe, so there is nothing to download or build."""
    if problems(dest):
        return False
    try:
        record = json.loads((dest / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return record.get("recipe") == recipe_hash()


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


def posix(path: str) -> str:
    """A tool's path in the form CMake takes on its command line on every system (forward slashes)."""
    return Path(path).as_posix()


def run(cmd: list[str], what: str) -> str:
    done = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    out = done.stdout + done.stderr
    if done.returncode:
        raise FetchError(f"{what} failed (exit {done.returncode}):\n{out[-2400:]}")
    return out


def link_libs_of(pc: str) -> tuple[str, ...]:
    """The -l libraries of an sdl3.pc, in order, without SDL itself."""
    libs: list[str] = []
    for line in pc.splitlines():
        if line.startswith(("Libs:", "Libs.private:")):
            libs += [t for t in line.split(":", 1)[1].split() if t.startswith("-l") and t != "-lSDL3"]
    return tuple(libs)


def check_link_libs(libs: tuple[str, ...], windows: bool) -> None:
    """The libraries sdl3.pc asks for are the constant's: exactly on Windows; on Linux nothing the constant lacks."""
    if windows:
        ok = libs == WINDOWS_LINK_LIBS
    else:
        ok = set(libs) <= set(LINUX_LINK_LIBS)
    if not ok:
        raise FetchError(f"sdl3.pc asks for the libraries {' '.join(libs)}, and tools/fetch_sdl3.py lists "
                         f"{' '.join(WINDOWS_LINK_LIBS if windows else LINUX_LINK_LIBS)}: update the list there")


def check_config(config: str, windows: bool) -> None:
    """The build configured a window driver and Vulkan support: SDL builds without either when a development package is
    missing (X11 and Wayland on Linux), and that kit would give a game with no window, so it is refused."""
    drivers = ("SDL_VIDEO_DRIVER_WINDOWS",) if windows else ("SDL_VIDEO_DRIVER_X11", "SDL_VIDEO_DRIVER_WAYLAND")
    on = lambda name: re.search(rf"^#define {name} 1\s*$", config, re.M) is not None   # noqa: E731
    if not any(on(d) for d in drivers):
        raise FetchError(f"SDL configured none of {', '.join(drivers)}: no window driver (on Linux, install the X11 or "
                         "Wayland development packages)")
    if not on("SDL_VIDEO_VULKAN"):
        raise FetchError("SDL configured no Vulkan support (SDL_VIDEO_VULKAN)")


def fdlibm_notice(source: str) -> str:
    """The Sun Microsystems notice that heads the files of src/libm, as plain text."""
    m = re.search(r"/\*\s*\n \* =+\n((?: \*.*\n)+?) \* =+\n \*/", source)
    if not m:
        raise FetchError("src/libm/e_pow.c holds no notice of the form the recipe knows")
    return "\n".join(line[3:] if line.startswith(" * ") else "" for line in m.group(1).splitlines()) + "\n"


def stb_notice(source: str) -> str:
    """The two licences that close stb_image.h, as plain text."""
    start = source.find(STB_START)
    end = source.find("*/", start)
    if start < 0 or end < 0:
        raise FetchError("src/video/stb_image.h holds no licence of the form the recipe knows")
    head = source.rfind("\n-----", 0, start)
    return source[head + 1:end].strip("\n") + "\n"


def sibling(cc: str, name: str) -> str | None:
    """The tool called name beside the compiler cc (or beside what cc resolves to), else on PATH."""
    ext = ".exe" if os.name == "nt" else ""
    for d in dict.fromkeys((Path(cc).parent, Path(cc).resolve().parent)):
        if (d / f"{name}{ext}").is_file():
            return str(d / f"{name}{ext}")
    return shutil.which(name)


def build(dest: Path, cc: str, cxx: str, ar: str, ninja: str, cmake: str) -> list[str]:
    """Lay out dest from the pins; return what was done, one line per step. dest must be absent, empty, or a folder this
    script laid out before (it holds the record), which is then replaced."""
    if dest.exists() and (not dest.is_dir() or (any(dest.iterdir()) and not (dest / RECORD).is_file())):
        raise FetchError(f"{dest} exists and is not a folder this script laid out; give an empty or new folder")
    machine = subprocess.run([cc, "-dumpmachine"], capture_output=True, text=True).stdout.strip()
    windows = "mingw" in machine or "windows" in machine
    if windows != (os.name == "nt") or not machine.startswith(("x86_64", "amd64")):
        raise FetchError(f"the compiler targets {machine or 'an unknown machine'}: this script builds for x86-64 and for "
                         "the machine it runs on (Windows with mingw-w64, or Linux)")
    lines = []
    with tempfile.TemporaryDirectory(prefix="oot2-fetch-sdl3-") as tmp:
        work = Path(tmp)
        archive = work / f"SDL3-{VERSION}.tar.gz"
        download(SDL3, archive)   # checked against the pin before anything is unpacked
        with tarfile.open(archive) as t:
            t.extractall(work / "src", filter="data")
        src = work / "src" / f"SDL3-{VERSION}"
        lines.append(f"SDL {VERSION} ({SDL3.sha256})")
        build_dir, stage = work / "build", work / "stage"
        configure = [cmake, "-S", str(src), "-B", str(build_dir), "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={posix(ninja)}",
                     f"-DCMAKE_C_COMPILER={posix(cc)}", f"-DCMAKE_CXX_COMPILER={posix(cxx)}", f"-DCMAKE_AR={posix(ar)}",
                     "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={stage}", *(f"-D{o}" for o in OPTIONS)]
        run(configure, "configuring SDL")
        configs = sorted(build_dir.glob("include-config-*/build_config/SDL_build_config.h"))
        if not configs:
            raise FetchError("the SDL build wrote no SDL_build_config.h where the recipe looks for it")
        check_config(configs[0].read_text(encoding="utf-8", errors="replace"), windows)
        run([cmake, "--build", str(build_dir)], "building SDL")
        run([cmake, "--install", str(build_dir)], "installing SDL")
        libs = link_libs_of((stage / "lib" / "pkgconfig" / "sdl3.pc").read_text(encoding="utf-8"))
        check_link_libs(libs, windows)
        if dest.exists():
            shutil.rmtree(dest)
        (dest / "lib").mkdir(parents=True)
        shutil.copyfile(stage / "lib" / "libSDL3.a", dest / "lib" / "libSDL3.a")
        (dest / "include" / "SDL3").mkdir(parents=True)
        for h in sorted((stage / "include" / "SDL3").glob("*.h")):
            shutil.copyfile(h, dest / "include" / "SDL3" / h.name)
        if not (dest / "include" / "SDL3" / "SDL_vulkan.h").is_file():
            raise FetchError("the SDL install holds no SDL_vulkan.h")
        lic = dest / "licenses"
        lic.mkdir()
        for name, rel in LICENCE_FILES.items():
            shutil.copyfile(src / rel, lic / name)
        text = (src / "src" / "libm" / "e_pow.c").read_text(encoding="utf-8")
        (lic / "fdlibm-SunPro.txt").write_text(fdlibm_notice(text), encoding="utf-8", newline="\n")
        text = (src / "src" / "video" / "stb_image.h").read_text(encoding="utf-8")
        (lic / "stb-MIT-or-Unlicense.txt").write_text(stb_notice(text), encoding="utf-8", newline="\n")
        (lic / README_NAME).write_text(README, encoding="utf-8", newline="\n")
        lines.append(f"libSDL3.a built for {machine}: {(dest / 'lib' / 'libSDL3.a').stat().st_size} bytes; "
                     f"link with {' '.join(libs)}")
    (dest / RECORD).write_text(json.dumps({"pins": pins(), "recipe": recipe_hash(), "target": machine,
                                           "files": tree_hashes(dest)}, indent=1) + "\n", encoding="utf-8")
    bad = problems(dest)
    if bad:   # the kit just laid out must pass the check the build makes of it
        raise FetchError("the kit just laid out is refused: " + bad[0])
    return lines


def find_tools(args: argparse.Namespace) -> tuple[str, str, str, str, str]:
    """(cc, cxx, ar, ninja, cmake) as full paths; SystemExit(2) naming the first one missing."""
    cc = shutil.which(args.cc)
    ninja, cmake = shutil.which(args.ninja), shutil.which(args.cmake)
    cxx = shutil.which(args.cxx) if args.cxx else (sibling(cc, "clang++") if cc else None)
    ar = shutil.which(args.ar) if args.ar else (sibling(cc, "llvm-ar") or shutil.which("ar") if cc else None)
    for what, path in (("C compiler", cc), ("C++ compiler (clang++ beside the C compiler)", cxx), ("archiver", ar),
                       ("Ninja", ninja), ("CMake", cmake)):
        if not path:
            print(f"fetch_sdl3: no {what} found (--cc, --cxx, --ar, --ninja, --cmake, or PATH)", file=sys.stderr)
            raise SystemExit(2)
    return cc, cxx, ar, ninja, cmake


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, required=True, help=f"the folder to lay out ({DEFAULT_ENV} then names it)")
    ap.add_argument("--cc", default=os.environ.get("OOT2_CC") or "clang")
    ap.add_argument("--cxx", default=os.environ.get("OOT2_CXX"), help="default: clang++ beside the C compiler")
    ap.add_argument("--ar", default=None)
    ap.add_argument("--ninja", default=os.environ.get("OOT2_NINJA") or "ninja")
    ap.add_argument("--cmake", default=os.environ.get("OOT2_CMAKE") or "cmake")
    args = ap.parse_args(argv)
    dest = args.dest.resolve()
    if is_current(dest):
        print(f"fetch_sdl3: {dest} already holds the pinned build, unchanged")
        return 0
    cc, cxx, ar, ninja, cmake = find_tools(args)
    try:
        for line in build(dest, cc, cxx, ar, ninja, cmake):
            print(f"fetch_sdl3: {line}")
    except FetchError as e:
        print(f"fetch_sdl3: {e}", file=sys.stderr)
        return 1
    print(f"fetch_sdl3: {dest} is ready; set {DEFAULT_ENV} to it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
