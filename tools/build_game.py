#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
#
# Adapted from TotK Recomp tools/build_game.py at 0ae47a5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh,
# licenses/totk-MIT.txt), as DKB Recomp (https://github.com/Lucas-blip-png/dkb-pc-recomp) carries it at 9d8fa02: the
# Ninja file and its escaping, the module glue, the object manifest checked before the link, the job-object measure of
# peak memory and the PE section report. Changed in oot2-native: one build folder per player, with Ninja run inside it and
# every object path relative to it; the runtime always goes through its static library, linked whole; the glue carries
# the module files to load and their SHA-256, the other files the program checks at each start and main.npdm's values;
# no RomFS index writer, overlays, shader check, MSYS2 defaults or reuse of another build's objects.
"""Compile the lifted C and link it with the runtime library into oot2.exe (architecture D4 and section 13).

    py -3 tools/build_game.py --gen DIR --exefs DIR --out DIR [--exe FILE] [--jobs N] [--cc CC] [--cxx CXX]
                              [--ninja FILE] [--vulkan-sdk DIR] [--runtime-lib FILE] [--compile-only]
                              [--file WHAT=PATH] [--file-size WHAT=PATH]
    py -3 tools/build_game.py --make-runtime-lib --out DIR [--cc CC] [--cxx CXX] [--ninja FILE] [--vulkan-sdk DIR]

--gen is the lifter's output (tools/lift/gen.py: layout.json, layout.h, <mod>/manifest.json and the chunks it lists).
--exefs holds the player's plain modules and main.npdm, which oot2.exe loads at each start. --out is the build folder
(build.ninja, obj/, the runtime library, the program glue), outside the repository: the glue and the objects come from
the player's dump. --exe defaults to oot2.exe in --out.

Ninja runs in --out, so an object is rebuilt only when its source, a header it includes or its flags change: a new lift
rewrites only the chunks whose C changed (tools/lift/gen.py), and an edit of the runtime recompiles that file and
relinks, without touching the 1.2 GB of generated C.

Generated C is compiled at -O1 -march=x86-64-v3 (cpu.h refuses anything below x86-64-v3); the runtime at -O2. The
runtime (every runtime/*/*.c and runtime/*/*.cpp) goes into a static library, liboot2rt.a, which holds no game data;
--make-runtime-lib builds only that, as a release would prebuild it, and --runtime-lib links a prebuilt one instead. The
library is linked whole, because HOST_OVERRIDE entries (runtime.h) are found through a linker section, not through a
reference that would pull their object out of an archive. On Windows the program is linked with
--disable-high-entropy-va: with high-entropy ASLR the main thread's stack can land inside the fixed module region
(research 28).

C++. A .cpp file of the runtime is compiled as C++17 without exceptions and without run-time type information
(CXX_FLAGS: the runtime does not unwind C++ exceptions, research 05), by the compiler's C++ driver (--cxx, else
OOT2_CXX, else clang++ beside the C compiler). The program is always linked by that driver, so a C++ object in the
library finds the C++ standard library, and on Windows with -static, so libc++ and libunwind are inside oot2.exe and no
DLL of the toolchain sits beside it (research 104). Only a .cpp file is C++ here: the sources of runtime/ui stay behind
the C header oot2/ui.h.

The Vulkan back end. The runtime compiles against the Vulkan headers and Volk (runtime/gpu/vk_volk.c includes volk.c),
which come from a Vulkan SDK: --vulkan-sdk, else the VULKAN_SDK variable, whose Include folder holds vulkan/ and Volk/
(tools/fetch_vulkan.py lays one out at the pins continuous integration uses). Neither is in this repository. The
runtime's own shaders (runtime/gpu/shaders/*.comp, GLSL) are compiled for Vulkan 1.3 by glslang (OOT2_GLSLANG, else the
SDK's Bin folder, else PATH), validated by it (--spirv-val), and embedded as spirv/gpu_shaders.c in --out, which goes
into the runtime library (oot2/gpu.h). They are compiled at every build of the library, so an embedded shader always
matches its source; no compiled shader is kept in the repository.

program_glue.c, written into --out, carries what the build knows of the player's dump: the layout (checked against
layout.h at compile time), the lifted modules' function tables, each module file to load with its whole-file SHA-256,
the other files oot2.exe checks at each start (main.npdm and --file by SHA-256, --file-size by size only, and the
player's dump, when the RomFS is read from it in place, by size and by the SHA-256 of fixed ranges: sample_ranges), and
the values of main.npdm. Paths under the folder of the executable are stored relative to it, so the folder can move.

Object manifest. Ninja rebuilds an object only when its inputs change, so a bit flipped in an object an earlier build
wrote would stay forever. obj/objects.sha256 holds the SHA-256, size and modification time of every object. Before the
link, and after the compile, each object whose size and time match its entry is hashed and must equal it; an object
Ninja just rebuilt is recorded. A mismatch deletes the object, Ninja compiles it again, and the build says so.

Compile jobs default to the CPU count, capped by the free memory at JOB_MEM per job. Each phase reports its wall time
and its peak memory: on Windows from a job object (committed private bytes, of the largest process and of all of them
at once), elsewhere the largest resident set of one child.

Exit codes: 0 built, 1 the build failed (the message says why), 2 bad command line.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402
from check_content import GENERATED_MARKER  # noqa: E402

WIN = os.name == "nt"
RUNTIME = ROOT / "runtime"
EXE_NAME = "oot2.exe" if WIN else "oot2"
LIB_NAME = "liboot2rt.a"
GLUE_NAME = "program_glue.c"
MANIFEST = "objects.sha256"
RUNTIME_LIST = "runtime_objects.txt"
MARKER = GENERATED_MARKER.decode()

GEN_FLAGS = ("-std=gnu11", "-O1", "-march=x86-64-v3", "-fno-strict-aliasing", "-w")
RT_FLAGS = ("-std=gnu11", "-O2", "-march=x86-64-v3", "-fno-strict-aliasing", "-Wall", "-Wextra") + \
    (() if WIN else ("-pthread",))
# C++17 is what RmlUi asks for and Dear ImGui's C++11 code builds under; exceptions and run-time type information are
# off because nothing unwinds a C++ exception (research 05). The other flags are the runtime's own.
CXX_FLAGS = ("-std=c++17", "-O2", "-march=x86-64-v3", "-fno-strict-aliasing", "-fno-exceptions", "-fno-rtti",
             "-Wall", "-Wextra") + (() if WIN else ("-pthread",))
# -static on Windows: llvm-mingw's clang++ links libc++ as libc++.dll and libunwind as libunwind.dll unless told
# otherwise, and a DLL of the toolchain must not sit beside oot2.exe (research 104). Linux links the system's C++
# library, which every machine that builds there has.
LINK_FLAGS = ("-Wl,--disable-high-entropy-va", "-static") if WIN else ("-pthread",)
LINK_LIBS = () if WIN else ("-lm", "-ldl")   # Volk opens the Vulkan loader with dlopen

# The runtime's own shaders: GLSL compute shaders in runtime/gpu/shaders, compiled for Vulkan 1.3 and validated by
# glslang's own SPIR-V validator, then embedded as one C file of the build folder (oot2/gpu.h, gpu_shaders).
SHADER_DIR = Path("gpu") / "shaders"
SHADER_SUFFIX = ".comp"
SPIRV_DIR = "spirv"
SHADER_GLUE = "gpu_shaders.c"
GLSLANG_FLAGS = ("-V", "--target-env", "vulkan1.3", "--spirv-val", "--quiet")
SPIRV_MAGIC = 0x07230203
SHADER_NAME = re.compile(r"[a-z][a-z0-9_]*")

GiB = 1 << 30
# Peak commit of one compile in DKB Recomp's build: 120 to 180 MiB per chunk of about 3 MB, 568 MiB for a table of
# 295,000 functions. 768 MiB leaves room for the largest file.
JOB_MEM = 768 << 20
MEM_SHARE = 0.8   # of the free memory the compile jobs may take
NINJA_LINE = re.compile(r"^\[(\d+)/(\d+)\] ")


class BuildError(Exception):
    """A failure the player can act on. The message says what is wrong."""


# ------------------------------------------------------------------------------------------------ toolchain

@dataclass(frozen=True)
class Toolchain:
    cc: str
    ar: str
    ninja: str
    vulkan: Path | None = None    # a Vulkan SDK's Include folder: vulkan/ and Volk/ (vulkan_include)
    glslang: str | None = None    # compiles the runtime's own shaders (find_glslang)
    cxx: str | None = None        # the C++ driver: compiles the runtime's .cpp files and links the program (find_cxx)


VOLK_FILES = ("volk.c", "volk.h")


def vulkan_include(sdk: str | os.PathLike | None = None) -> Path | None:
    """The Include folder of the Vulkan SDK sdk (else the VULKAN_SDK variable) when it holds the Vulkan headers
    (vulkan/vulkan_core.h) and Volk (Volk/volk.c and volk.h), else None. Windows SDKs name it Include, others include."""
    sdk = sdk or os.environ.get("VULKAN_SDK")
    if not sdk:
        return None
    for name in ("Include", "include"):
        inc = Path(sdk) / name
        if (inc / "vulkan" / "vulkan_core.h").is_file() and all((inc / "Volk" / f).is_file() for f in VOLK_FILES):
            return inc.resolve()
    return None


def vulkan_header_version(include: Path) -> str:
    """The version of the Vulkan headers in include, as 1.4.357, or "unknown"."""
    try:
        text = (include / "vulkan" / "vulkan_core.h").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unknown"
    patch = re.search(r"^#define VK_HEADER_VERSION (\d+)", text, re.M)
    whole = re.search(r"^#define VK_HEADER_VERSION_COMPLETE VK_MAKE_API_VERSION\(\d+, (\d+), (\d+), ", text, re.M)
    return f"{whole.group(1)}.{whole.group(2)}.{patch.group(1)}" if patch and whole else "unknown"


def find_glslang(include: Path | None) -> str | None:
    """glslang: OOT2_GLSLANG, else glslang or glslangValidator in the Bin folder beside the SDK's include folder, else
    either on PATH. None when there is none."""
    name = os.environ.get("OOT2_GLSLANG")
    if name:
        return shutil.which(name)
    exe = ".exe" if WIN else ""
    if include is not None:
        for tool in ("glslang", "glslangValidator"):
            for folder in ("Bin", "bin"):
                path = include.parent / folder / f"{tool}{exe}"
                if path.is_file():
                    return str(path)
    return shutil.which("glslang") or shutil.which("glslangValidator")


def find_cxx(cc_path: str, cxx: str | None = None) -> str | None:
    """The C++ driver: cxx, else OOT2_CXX, else the C compiler's own (clang is followed by clang++, clang-18 by
    clang++-18, x86_64-w64-mingw32-clang by x86_64-w64-mingw32-clang++) in its folder or the folder its link leads to,
    else on PATH, else clang++ on PATH. None when there is none."""
    name = cxx or os.environ.get("OOT2_CXX")
    if name:
        return shutil.which(name)
    cc = Path(cc_path)
    sibling = re.sub(r"clang(?!\+)", "clang++", cc.name, count=1)
    if sibling == cc.name:
        return shutil.which("clang++")
    for d in dict.fromkeys((cc.parent, cc.resolve().parent)):
        if (d / sibling).is_file():
            return str(d / sibling)
    return shutil.which(sibling)


def find_toolchain(cc: str | None = None, ninja: str | None = None, ar: str | None = None,
                   vulkan_sdk: str | os.PathLike | None = None, cxx: str | None = None) -> Toolchain:
    """The C compiler (cc, else OOT2_CC, else clang), its C++ driver (find_cxx), Ninja (ninja, else OOT2_NINJA, else
    ninja on PATH), the archiver (ar, else llvm-ar beside the compiler, else llvm-ar or ar on PATH), the Vulkan headers
    and Volk (the Vulkan SDK vulkan_sdk, else VULKAN_SDK) and glslang (find_glslang). BuildError names what is
    missing."""
    name = cc or os.environ.get("OOT2_CC") or "clang"
    cc_path = shutil.which(name)
    if not cc_path:
        raise BuildError(f"no C compiler named {name!r} was found (give one with --cc or the OOT2_CC variable)")
    cxx_path = find_cxx(cc_path, cxx)
    if not cxx_path:
        raise BuildError(f"no C++ compiler was found for {cc_path}: clang++ comes with clang and links the program "
                         "(give one with --cxx or the OOT2_CXX variable)")
    name = ninja or os.environ.get("OOT2_NINJA") or "ninja"
    ninja_path = shutil.which(name)
    if not ninja_path:
        raise BuildError(f"Ninja ({name!r}) was not found: install it (ninja-build.org) or name it with OOT2_NINJA")
    ar_path = shutil.which(ar) if ar else None
    if ar and not ar_path:
        raise BuildError(f"no archiver named {ar!r} was found")
    if not ar_path:
        exe = ".exe" if WIN else ""
        # Ubuntu's /usr/bin/clang links into /usr/lib/llvm-N/bin, where llvm-ar has no version suffix.
        for d in dict.fromkeys((Path(cc_path).parent, Path(cc_path).resolve().parent)):
            if (d / f"llvm-ar{exe}").is_file():
                ar_path = str(d / f"llvm-ar{exe}")
                break
    ar_path = ar_path or shutil.which("llvm-ar") or shutil.which("ar")
    if not ar_path:
        raise BuildError("no archiver (llvm-ar or ar) was found beside the compiler or on PATH")
    include = vulkan_include(vulkan_sdk)
    if include is None:
        where = vulkan_sdk or os.environ.get("VULKAN_SDK")
        raise BuildError(
            "the Vulkan headers and Volk were not found" + (f" in {where}" if where else " (VULKAN_SDK is not set)") +
            ": the runtime's Vulkan back end compiles against them. Install the Vulkan SDK (vulkan.lunarg.com) and set "
            "VULKAN_SDK to it, or run tools/fetch_vulkan.py --dest DIR and set VULKAN_SDK to DIR")
    glslang = find_glslang(include)
    if not glslang:
        raise BuildError("glslang was not found (OOT2_GLSLANG, the Vulkan SDK's Bin folder or PATH): it compiles the "
                         "runtime's own shaders (runtime/gpu/shaders)")
    return Toolchain(cc_path, ar_path, ninja_path, include, glslang, cxx_path)


def vulkan_inputs(include: Path | None, glslang: str | None) -> list:
    """What the runtime library takes from the Vulkan SDK, for the link step's key: the headers' and Volk's files and
    glslang's version."""
    if include is None or glslang is None:
        return ["no Vulkan SDK"]
    return [include / "vulkan" / "vulkan_core.h", *(include / "Volk" / f for f in VOLK_FILES), tool_id(glslang)]


_IDS: dict[str, str] = {}


def tool_id(path: str) -> str:
    """The first line of `path --version`: which compiler or archiver built the objects, for the steps' keys."""
    if path not in _IDS:
        done = subprocess.run([path, "--version"], capture_output=True, text=True, errors="replace")
        _IDS[path] = ((done.stdout or done.stderr).strip().splitlines() or ["unknown"])[0]
    return _IDS[path]


def default_jobs(free: int | None = None) -> tuple[int, str]:
    """Compile jobs and why: the CPU count, capped by the free memory at JOB_MEM per job."""
    cpus = os.cpu_count() or 4
    free = available_ram() if free is None else free
    if not free:
        return cpus, f"{cpus} CPUs, free memory unknown"
    by_mem = max(1, int(free * MEM_SHARE) // JOB_MEM)
    return min(cpus, by_mem), f"{cpus} CPUs, {free / GiB:.1f} GiB free memory allows {by_mem}"


def available_ram() -> int | None:
    if WIN:
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "aev")]
        st = MemoryStatusEx()
        st.dwLength = ctypes.sizeof(st)
        return st.avail if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)) else None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


# ------------------------------------------------------------------------------------------------ inputs

@dataclass
class Lifted:
    name: str
    sources: list[Path]   # the chunks in manifest order, then the function table
    functions: int


def end_marker(name: str) -> str:
    """The function the lifter emits at the end of a module's function table: the first byte past its lifted code."""
    return f"{name}_lifted_end"


def check_end_marker(name: str, man: dict, table: Path) -> None:
    """The lifted code's end marker must be named by the manifest and defined by the function table, which the link
    places after the module's chunks. Without it the runtime cannot tell the last lifted function from the host code
    linked after it (ticket 92), so the build stops here, before any compile."""
    symbol = end_marker(name)
    if man.get("end_marker") != symbol:
        raise BuildError(f"end marker missing: the manifest of {name} does not name {symbol}: lift again with the "
                         f"current lifter")
    with open(table, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 4096))
        tail = f.read()
    if f"void {symbol}(Cpu *c)".encode() not in tail:
        raise BuildError(f"end marker missing: {table} does not define {symbol} at its end: lift again with the "
                         f"current lifter")


def lifted_modules(gen: Path) -> tuple[list[layout.Module], list[Lifted]]:
    """The layout of gen and the modules lifted into it. A module of the layout without a manifest is loaded at run
    time but not lifted (a partial build); one whose manifest disagrees with the layout, or lists a missing file, is a
    BuildError."""
    if not (gen / "layout.json").is_file():
        raise BuildError(f"{gen / 'layout.json'} is missing: lift the modules first")
    mods = layout.read(gen)
    lifted = []
    for m in mods:
        path = gen / m.name / "manifest.json"
        if not path.is_file():
            continue
        man = json.loads(path.read_text(encoding="utf-8"))
        if man.get("marker") != MARKER or man.get("module") != m.name:
            raise BuildError(f"{path} is not a manifest the lifter wrote for {m.name}")
        if int(man["base"], 16) != m.base:
            raise BuildError(f"{path} places {m.name} at {man['base']}, layout.json at {m.base:#x}: lift again")
        sources = [gen / m.name / f for f in man["chunks"]] + [gen / m.name / man["table"]]
        missing = [s for s in sources if not s.is_file()]
        if missing:
            raise BuildError(f"{m.name}: {len(missing)} files listed in {path} are missing, first {missing[0]}")
        check_end_marker(m.name, man, sources[-1])
        lifted.append(Lifted(m.name, sources, int(man["function_count"])))
    if not lifted:
        raise BuildError(f"no module of {gen / 'layout.json'} was lifted into {gen}")
    return mods, lifted


def read_npdm(path: Path) -> dict:
    """The values of a main.npdm the runtime starts the program with (its META header)."""
    raw = path.read_bytes()
    if len(raw) < 0x80 or raw[:4] != b"META":
        raise BuildError(f"{path} is not a program description (no META header)")
    sysres, _, stack = struct.unpack_from("<III", raw, 0x14)
    return {"flags": raw[0xC], "main_thread_priority": raw[0xE], "main_thread_core": raw[0xF],
            "system_resource_size": sysres, "main_thread_stack_size": stack}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        buf = bytearray(1 << 20)
        view = memoryview(buf)
        while n := f.readinto(buf):
            h.update(view[:n])
    return h.hexdigest()


SAMPLE_COUNT, SAMPLE_BYTES = 16, 64 << 10


def sample_ranges(size: int) -> list[tuple[int, int]]:
    """The (offset, length) ranges of a file of size bytes that sample_sha256 hashes: SAMPLE_COUNT blocks of
    SAMPLE_BYTES (the whole file when it is smaller), the first at the start, the last ending at the end, the others
    evenly between. runtime/program/main.c picks the same ranges."""
    n = min(SAMPLE_BYTES, size)
    span = size - n
    return [(span * i // (SAMPLE_COUNT - 1), n) for i in range(SAMPLE_COUNT)]


def sample_sha256(path: Path) -> str:
    """SHA-256 of the bytes of sample_ranges, in order. Cheap at any size: 1 MiB read from 16 places. On a .nup the
    first range holds the superblock, whose SHA-256 of the table of contents names every file's size and SHA-256, so
    another dump of the same size differs there; the other ranges catch most changes made in place."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        for off, n in sample_ranges(size):
            f.seek(off)
            data = f.read(n)
            if len(data) != n:
                raise OSError(f"{path} ended while it was read")
            h.update(data)
    return h.hexdigest()


@dataclass
class GameFile:
    what: str        # what the player is told when it fails: "main.npdm", "RomFS index"
    path: Path
    hashed: bool     # SHA-256 at each start; else the size only (a file too large to hash at each start)
    sampled: bool = False   # with hashed False: also the SHA-256 of sample_ranges, for a file the program reads in
                            # place that the player owns and could move, replace or change (the dump)


# ------------------------------------------------------------------------------------------------ the build

@dataclass
class Build:
    out: Path                                  # build folder
    tool: Toolchain
    gen: Path | None = None                    # lifter output; None for --make-runtime-lib
    exefs: Path | None = None                  # plain modules and main.npdm
    exe: Path | None = None                    # default out/oot2.exe
    runtime: Path = RUNTIME                    # runtime sources (tests build a copy)
    runtime_lib: Path | None = None            # a prebuilt library instead of building one
    files: list[GameFile] = field(default_factory=list)   # checked at each start besides the modules and main.npdm
    jobs: int = 0

    def __post_init__(self) -> None:
        # Every path resolved, so that relative paths between them hold (a Windows temporary folder can be named by
        # its 8.3 short name, RUNNER~1, which resolve() expands).
        self.out = Path(self.out).resolve()
        self.exe = Path(self.exe).resolve() if self.exe else self.out / EXE_NAME
        self.runtime = Path(self.runtime).resolve()
        for name in ("gen", "exefs", "runtime_lib"):
            if getattr(self, name) is not None:
                setattr(self, name, Path(getattr(self, name)).resolve())
        if not self.jobs:
            self.jobs = default_jobs()[0]

    @property
    def lib(self) -> Path:
        return (self.runtime_lib or self.out / LIB_NAME).resolve()


def rel(path: Path, base: Path) -> str:
    """path relative to base where it can be (POSIX), else absolute. Both are resolved first (Build.__post_init__)."""
    path, base = Path(path).resolve(), Path(base).resolve()
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:   # another drive
        return path.as_posix()


def ninja_path(path: Path, base: Path) -> str:
    """A path on a build line: relative to the build folder where it can be, Ninja-escaped."""
    return rel(path, base).replace("$", "$$").replace(" ", "$ ").replace(":", "$:")


def cmd_arg(s: str) -> str:
    """s on a Ninja command line: Ninja-escaped, and quoted when it holds a space (clang and lld split their Windows
    command line at unquoted spaces)."""
    s = s.replace("$", "$$")
    return f'"{s}"' if " " in s else s


CXX_SUFFIX = ".cpp"


def runtime_sources(runtime: Path) -> list[Path]:
    """The runtime's C and C++ sources, runtime/<module>/*.c and *.cpp, in name order."""
    return sorted(p for suffix in (".c", CXX_SUFFIX) for p in runtime.glob(f"*/*{suffix}"))


# ------------------------------------------------------------------------------------------------ the runtime's shaders

def runtime_shaders(runtime: Path) -> list[Path]:
    """The runtime's own shaders, runtime/gpu/shaders/*.comp, in name order."""
    return sorted((runtime / SHADER_DIR).glob(f"*{SHADER_SUFFIX}"))


def shader_glue_path(out: Path) -> Path:
    return Path(out) / SPIRV_DIR / SHADER_GLUE


def compile_shaders(runtime: Path, out: Path, glslang: str) -> list[tuple[str, bytes]]:
    """Compile every shader of runtime/gpu/shaders with glslang for Vulkan 1.3, validated, into out/spirv/<name>.spv,
    each time from its source; return (name, SPIR-V) in name order. BuildError names a shader whose name is not a C
    identifier in lower case or which does not compile or validate."""
    folder = Path(out) / SPIRV_DIR
    folder.mkdir(parents=True, exist_ok=True)
    shaders = []
    for src in runtime_shaders(runtime):
        name = src.name[:-len(SHADER_SUFFIX)]
        if not SHADER_NAME.fullmatch(name):
            raise BuildError(f"shader {src}: its name must be lower-case letters, digits and underscores, starting "
                             "with a letter (it names a C array)")
        spv = folder / f"{name}.spv"
        spv.unlink(missing_ok=True)   # a failed compile must not leave the last build's code to embed
        done = subprocess.run([glslang, *GLSLANG_FLAGS, "-o", str(spv), str(src)], capture_output=True, text=True,
                              errors="replace")
        if done.returncode or not spv.is_file():
            detail = (done.stdout + done.stderr).strip()[-1500:]
            raise BuildError(f"glslang could not compile or validate shader {src} (exit {done.returncode}):\n{detail}")
        data = spv.read_bytes()
        if len(data) < 20 or len(data) % 4 or struct.unpack_from("<I", data)[0] != SPIRV_MAGIC:
            raise BuildError(f"glslang wrote {len(data)} bytes for shader {src}, which is not SPIR-V")
        shaders.append((name, data))
    return shaders


def shader_glue(shaders: list[tuple[str, bytes]]) -> str:
    """The C file that embeds the shaders as gpu_shaders (oot2/gpu.h), in the order given."""
    lines = ["// Written by tools/build_game.py from runtime/gpu/shaders; do not edit.", "#include <stdint.h>", "",
             '#include "oot2/gpu.h"', ""]
    for name, data in shaders:
        words = struct.unpack(f"<{len(data) // 4}I", data)
        lines.append(f"static const uint32_t spirv_{name}[{len(words)}] = {{")
        lines += ["    " + ", ".join(f"0x{w:08x}u" for w in words[i:i + 8]) + "," for i in range(0, len(words), 8)]
        lines += ["};", ""]
    if shaders:
        lines.append("const GpuShader gpu_shaders[] = {")
        lines += [f'    {{"{name}", spirv_{name}, sizeof spirv_{name}}},' for name, _ in shaders]
        lines.append("};")
    else:
        lines.append("const GpuShader gpu_shaders[1];")
    lines.append(f"const int gpu_shader_count = {len(shaders)};")
    return "\n".join(lines) + "\n"


def write_shader_glue(runtime: Path, out: Path, glslang: str) -> Path:
    """Compile the runtime's shaders and write out/spirv/gpu_shaders.c, touched only when its text changed, so Ninja
    compiles it again exactly when a shader's code did."""
    path = shader_glue_path(out)
    write_if_changed(path, shader_glue(compile_shaders(runtime, out, glslang)))
    return path


def object_of(b: Build, src: Path, module: str | None) -> Path:
    if module:
        return b.out / "obj" / module / (src.stem + ".o")
    # An archive names its members by file name and `ar r` replaces a member of the same name, so runtime/core/x.c and
    # runtime/hos/x.c must not both make x.o.
    return b.out / "obj" / "runtime" / f"{src.parent.name}_{src.stem}.o"


@dataclass
class Plan:
    gen_objects: dict[str, list[tuple[Path, Path]]]   # module: [(source, object)]
    rt_objects: list[tuple[Path, Path]]
    glue: Path
    glue_object: Path
    text: str                                          # build.ninja


def plan(b: Build, lifted: list[Lifted]) -> Plan:
    o = b.out
    inc = f"-I {cmd_arg(rel(b.runtime / 'include', o))}"
    gen_objects = {m.name: [(s, object_of(b, s, m.name)) for s in m.sources] for m in lifted}
    # The runtime's sources and the file that embeds its shaders (written by runtime_library before Ninja runs).
    rt_objects = [] if b.runtime_lib else [(s, object_of(b, s, None))
                                           for s in runtime_sources(b.runtime) + [shader_glue_path(o)]]
    if rt_objects and b.tool.vulkan is None:
        raise BuildError("the runtime cannot be built without the Vulkan headers and Volk (find_toolchain)")
    if b.tool.cxx is None:
        raise BuildError("the program is linked by the C++ driver, and there is none (find_toolchain)")
    # x.c and x.cpp of one module would make the same object, and an archive keeps only one member of a name.
    made: dict[Path, Path] = {}
    for src, obj in rt_objects:
        if obj in made:
            raise BuildError(f"{made[obj]} and {src} both make {obj.name}: rename one of them")
        made[obj] = src
    # The Vulkan headers and Volk, for the runtime only: generated code and the program glue never include them.
    vk = f" -I {cmd_arg(rel(b.tool.vulkan, o))} -I {cmd_arg(rel(b.tool.vulkan / 'Volk', o))}" if b.tool.vulkan else ""
    glue, glue_object = o / GLUE_NAME, o / "obj" / "program_glue.o"
    lines = [
        "# Written by tools/build_game.py; do not edit.",
        "ninja_required_version = 1.10",
        f"cc = {cmd_arg(b.tool.cc)}",
        f"cxx = {cmd_arg(b.tool.cxx)}",
        f"ar = {cmd_arg(b.tool.ar)}",
        f"genflags = {' '.join(GEN_FLAGS)} {inc}",
        f"rtflags = {' '.join(RT_FLAGS)} {inc} -I {cmd_arg(rel(b.runtime / 'core', o))}{vk}",
        f"cxxflags = {' '.join(CXX_FLAGS)} {inc} -I {cmd_arg(rel(b.runtime / 'core', o))}{vk}",
        f"glueflags = {' '.join(RT_FLAGS)} {inc}" + (f" -I {cmd_arg(rel(b.gen, o))}" if b.gen else ""),
        f"ldflags = {' '.join(LINK_FLAGS)}",
        f"lib = {cmd_arg(rel(b.lib, o))}",
        f"libs = {' '.join(LINK_LIBS)}",
        "",
        "rule cc_gen",
        "  command = $cc $genflags -MD -MF $out.d -c $in -o $out",
        "  deps = gcc",
        "  depfile = $out.d",
        "  description = CC $in",
        "",
        "rule cc_rt",
        "  command = $cc $rtflags -MD -MF $out.d -c $in -o $out",
        "  deps = gcc",
        "  depfile = $out.d",
        "  description = CC $in",
        "",
        "rule cc_glue",
        "  command = $cc $glueflags -MD -MF $out.d -c $in -o $out",
        "  deps = gcc",
        "  depfile = $out.d",
        "  description = CC $in",
        "",
        "rule cxx_rt",
        "  command = $cxx $cxxflags -MD -MF $out.d -c $in -o $out",
        "  deps = gcc",
        "  depfile = $out.d",
        "  description = CXX $in",
        "",
        "rule ar",
        "  command = $ar rcs $out $in",
        "  description = AR $out",
        "",
        # Whole archive: nothing references the objects that only hold HOST_OVERRIDE entries. Linked by the C++
        # driver, so that the C++ objects of the library find the C++ standard library (CXX_FLAGS, LINK_FLAGS).
        "rule link",
        "  command = $cxx $ldflags @$out.rsp -Wl,--whole-archive $lib -Wl,--no-whole-archive $libs -o $out",
        "  rspfile = $out.rsp",
        "  rspfile_content = $in_newline",
        "  description = LINK $out",
        "",
    ]
    gen_list = [obj for pairs in gen_objects.values() for _, obj in pairs]
    for pairs in gen_objects.values():
        for src, obj in pairs:
            lines.append(f"build {ninja_path(obj, o)}: cc_gen {ninja_path(src, o)}")
    for src, obj in rt_objects:
        rule = "cxx_rt" if src.suffix.lower() == CXX_SUFFIX else "cc_rt"
        lines.append(f"build {ninja_path(obj, o)}: {rule} {ninja_path(src, o)}")
    if rt_objects:
        lines.append(f"build {ninja_path(b.lib, o)}: ar " + " ".join(ninja_path(obj, o) for _, obj in rt_objects))
    lines.append(f"build runtime_lib: phony {ninja_path(b.lib, o)}")
    if gen_list:
        lines.append(f"build {ninja_path(glue_object, o)}: cc_glue {ninja_path(glue, o)}")
        lines.append("build gen_objects: phony " + " ".join(ninja_path(x, o) for x in gen_list))
        lines.append(f"build {ninja_path(b.exe, o)}: link " + " ".join(ninja_path(x, o) for x in gen_list) +
                     f" {ninja_path(glue_object, o)} | {ninja_path(b.lib, o)}")
        lines.append(f"default {ninja_path(b.exe, o)}")
    else:
        lines.append(f"default {ninja_path(b.lib, o)}")
    return Plan(gen_objects, rt_objects, glue, glue_object, "\n".join(lines) + "\n")


def write_if_changed(path: Path, text: str) -> bool:
    """Keeps the modification time of an unchanged file, so Ninja does not rebuild from it."""
    data = text.encode("utf-8")
    try:
        if path.read_bytes() == data:
            return False
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


# ------------------------------------------------------------------------------------------------ program glue

def c_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def game_path(path: Path, exe_dir: Path) -> str:
    """How oot2.exe finds a file: relative to its own folder when the file is inside it, else absolute."""
    path, exe_dir = Path(path).resolve(), Path(exe_dir).resolve()
    return path.relative_to(exe_dir).as_posix() if path.is_relative_to(exe_dir) else path.as_posix()


def glue_text(mods: list[layout.Module], lifted: list[str], modules: dict[str, tuple[str, str]],
              files: list[tuple[str, str, int, str | None, str | None]], npdm: dict | None) -> str:
    """C of program_glue.c. modules: {name: (path, sha256)} for every module of the layout; files: (what, path, size,
    sha256 or None, sample_sha256 or None); npdm: read_npdm() or None when the dump has none."""
    out = [f"// Program glue written by tools/build_game.py for one player's dump; do not edit. {MARKER}",
           '#include "layout.h"', '#include "oot2/program.h"', ""]
    for m in mods:
        u = m.name.upper()
        out.append(f"_Static_assert(MOD_{u}_BASE == {m.base:#x}ull && MOD_{u}_SIZE == {m.size:#x}ull, "
                   f'"layout.h and layout.json disagree on {m.name}");')
    out.append("")
    for name in lifted:
        out += [f"extern const uint64_t {name}_fn_addrs[];", f"extern const GuestFn {name}_fn_ptrs[];",
                f"extern const uint64_t {name}_fn_count;", f"extern void {end_marker(name)}(Cpu *c);"]
    out += ["", "static const RtModuleFile modules[] = {"]
    for m in mods:
        path, sha = modules[m.name]
        u = m.name.upper()
        out.append(f"    {{{c_string(m.name)}, {c_string(path)}, MOD_{u}_BASE, MOD_{u}_SIZE, {c_string(sha)}}},")
    out.append("};")
    if files:
        out += ["", "static const RtGameFile files[] = {"]
        for what, path, size, sha, sample in files:
            out.append(f"    {{{c_string(what)}, {c_string(path)}, {size}ull, {c_string(sha) if sha else 'NULL'}, "
                       f"{c_string(sample) if sample else 'NULL'}}},")
        out.append("};")
    out += ["", "int rt_program_lifted(RtModule *out, int max) {", "    int n = 0;"]
    for name in lifted:
        u = name.upper()
        out.append(f"    if (n < max) out[n++] = (RtModule){{{c_string(name)}, MOD_{u}_BASE, MOD_{u}_SIZE, "
                   f"{name}_fn_addrs, {name}_fn_ptrs, {name}_fn_count, (const void *){end_marker(name)}}};")
    out += ["    return n;", "}", ""]
    n = npdm or {}
    out += ["const RtProgram rt_program = {",
            f"    .modules = modules, .module_count = {len(mods)},",
            f"    .files = {'files' if files else 'NULL'}, .file_count = {len(files)},",
            f"    .has_npdm = {1 if npdm else 0},",
            f"    .system_resource_size = {n.get('system_resource_size', 0):#x}ull,",
            f"    .main_thread_priority = {n.get('main_thread_priority', 0)},",
            f"    .main_thread_core = {n.get('main_thread_core', 0)},",
            f"    .main_thread_stack_size = {n.get('main_thread_stack_size', 0):#x},",
            "};", ""]
    return "\n".join(out)


def program_glue(b: Build, mods: list[layout.Module], lifted: list[Lifted]) -> tuple[str, dict]:
    """The glue's text, and what it baked: module hashes, the files checked at start, main.npdm's values."""
    exe_dir = b.exe.parent
    modules, files, facts = {}, [], {"modules": {}, "files": {}}
    for m in mods:
        path = b.exefs / m.name
        if not path.is_file():
            raise BuildError(f"the module {path} is missing: the build bakes the SHA-256 of every module it loads")
        sha = file_sha256(path)
        modules[m.name] = (game_path(path, exe_dir), sha)
        facts["modules"][m.name] = sha
    npdm = None
    npdm_path = b.exefs / "main.npdm"
    checked = list(b.files)
    if npdm_path.is_file():
        npdm = read_npdm(npdm_path)
        checked.insert(0, GameFile("main.npdm", npdm_path, True))
    for f in checked:
        if not f.path.is_file():
            raise BuildError(f"{f.what}: {f.path} is missing")
        sha = file_sha256(f.path) if f.hashed else None
        sample = sample_sha256(f.path) if f.sampled and not f.hashed else None
        size = f.path.stat().st_size
        files.append((f.what, game_path(f.path, exe_dir), size, sha, sample))
        facts["files"][f.what] = {"size": size, **({"sha256": sha} if sha else {}),
                                  **({"sample_sha256": sample} if sample else {})}
    facts["npdm"] = npdm
    return glue_text(mods, [m.name for m in lifted], modules, files, npdm), facts


# ------------------------------------------------------------------------------------------------ running Ninja

@dataclass
class Measured:
    code: int
    seconds: float
    peak_process: int   # bytes, the largest process
    peak_all: int       # bytes, every process of the run at once (Windows only, else 0)
    tail: list[str]     # the last lines of the output


def run_measured(cmd: list[str], cwd: Path, on_line: Callable[[str], None] | None = None) -> Measured:
    """Run cmd, its output read line by line, and measure its wall time and peak memory (see the module docstring)."""
    # A Python child (the lifter) writes UTF-8 to the pipe, as clang and Ninja do.
    env = dict(os.environ, NINJA_STATUS="[%f/%t] ", PYTHONUTF8="1")
    tail: deque[str] = deque(maxlen=60)
    t0 = time.monotonic()
    job = _Job() if WIN else None
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace", creationflags=_Job.CREATE_SUSPENDED if job else 0)
    try:
        if job:
            job.add(proc)
        with proc.stdout:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                tail.append(line)
                if on_line:
                    on_line(line)
        code = proc.wait()
    except BaseException:
        if job:
            job.close()   # kills every compiler of the run
        else:
            proc.kill()
        proc.wait()
        raise
    seconds = time.monotonic() - t0
    if job:
        peak_one, peak_all = job.peaks()
        job.close()
    else:
        import resource
        peak_one, peak_all = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024, 0
    return Measured(code, seconds, peak_one, peak_all, list(tail))


class _Job:
    """A Windows job object around one process tree: peak memory of its processes, and every one killed on close."""
    CREATE_SUSPENDED = 0x4
    EXTENDED_LIMIT_INFORMATION = 9
    KILL_ON_JOB_CLOSE = 0x2000

    def __init__(self) -> None:
        import ctypes.wintypes as wt

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wt.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        k32.QueryInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p]
        k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        k32.CloseHandle.argtypes = [wt.HANDLE]
        self.ntdll = ctypes.WinDLL("ntdll")
        self.ntdll.NtResumeProcess.argtypes = [wt.HANDLE]
        self.k32, self.info = k32, ExtendedLimits()
        self.handle = k32.CreateJobObjectW(None, None)
        self.info.BasicLimitInformation.LimitFlags = self.KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(self.handle, self.EXTENDED_LIMIT_INFORMATION, ctypes.byref(self.info),
                                    ctypes.sizeof(self.info))

    def add(self, proc: subprocess.Popen) -> None:
        # Started suspended, so every compiler it starts is counted from the first.
        if not self.k32.AssignProcessToJobObject(self.handle, int(proc._handle)):
            print(f"build_game: cannot measure memory (error {ctypes.get_last_error()})", file=sys.stderr)
        self.ntdll.NtResumeProcess(int(proc._handle))

    def peaks(self) -> tuple[int, int]:
        self.k32.QueryInformationJobObject(self.handle, self.EXTENDED_LIMIT_INFORMATION, ctypes.byref(self.info),
                                           ctypes.sizeof(self.info), None)
        return self.info.PeakProcessMemoryUsed, self.info.PeakJobMemoryUsed

    def close(self) -> None:
        if self.handle:
            self.k32.CloseHandle(self.handle)
            self.handle = None


def ninja(b: Build, targets: list[str], on_line: Callable[[str], None] | None = None) -> Measured:
    m = run_measured([b.tool.ninja, "-f", "build.ninja", "-j", str(b.jobs), *targets], b.out, on_line)
    if m.code:
        detail = "\n".join(m.tail[-25:])
        raise BuildError(f"Ninja stopped with exit code {m.code} building {' '.join(targets)}:\n{detail}")
    return m


# ------------------------------------------------------------------------------------------------ object manifest

def read_manifest(path: Path) -> dict[str, tuple[str, int, int]]:
    """{object path below obj/: (sha256, size, mtime_ns)}; empty when the file is missing."""
    man = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            sha, size, mtime, name = line.split(" ", 3)
            man[name] = (sha, int(size), int(mtime))
    return man


def write_manifest(path: Path, man: dict[str, tuple[str, int, int]]) -> None:
    write_if_changed(path, "".join(f"{sha} {size} {mtime} {name}\n" for name, (sha, size, mtime) in sorted(man.items())))


def check_objects(obj_dir: Path, objects: list[Path], man: dict) -> tuple[int, list[Path]]:
    """Objects whose size and time match their entry are hashed and must equal it; any other was just built and is
    recorded in man. Returns (objects verified, objects whose hash is wrong)."""
    verified, bad = 0, []
    for o in objects:
        name = o.relative_to(obj_dir).as_posix()
        st = o.stat()
        entry = man.get(name)
        if entry and entry[1] == st.st_size and entry[2] == st.st_mtime_ns:
            if file_sha256(o) == entry[0]:
                verified += 1
            else:
                bad.append(o)
        else:
            man[name] = (file_sha256(o), st.st_size, st.st_mtime_ns)
            verified += 1
    return verified, bad


def verify_objects(b: Build, objects: list[Path], on_line=None, prune: bool = False) -> dict:
    """Check objects against the manifest; recompile the ones that fail (module docstring). prune: objects is the
    whole build, so entries of objects it no longer has are dropped."""
    t0 = time.monotonic()
    obj_dir = b.out / "obj"
    path = obj_dir / MANIFEST
    man = read_manifest(path)
    if prune:
        names = {o.relative_to(obj_dir).as_posix() for o in objects}
        for name in [n for n in man if n not in names]:
            del man[name]
    verified, bad = check_objects(obj_dir, objects, man)
    for o in bad:
        print(f"build_game: object {o} does not match its SHA-256; deleting it to compile it again", flush=True)
        o.unlink()
    if bad:
        ninja(b, [rel(o, b.out) for o in bad], on_line)
        more, still = check_objects(obj_dir, bad, man)
        verified += more
        if still:
            raise BuildError(f"{still[0]} was compiled again and still does not match: the disk may be failing")
    write_manifest(path, man)
    return {"verified": verified, "recompiled": len(bad), "seconds": round(time.monotonic() - t0, 1)}


# ------------------------------------------------------------------------------------------------ phases

def prepare(b: Build) -> tuple[list[layout.Module], list[Lifted], Plan]:
    mods, lifted = lifted_modules(b.gen)
    p = plan(b, lifted)
    b.out.mkdir(parents=True, exist_ok=True)
    write_if_changed(b.out / "build.ninja", p.text)
    return mods, lifted, p


def compile_generated(b: Build, on_line=None) -> dict:
    """Compile the lifted C (Ninja target gen_objects) and check every object against the manifest."""
    _, lifted, p = prepare(b)
    m = ninja(b, ["gen_objects"], on_line)
    objects = [obj for pairs in p.gen_objects.values() for _, obj in pairs]
    check = verify_objects(b, objects, on_line)
    return {"objects": len(objects), "functions": sum(x.functions for x in lifted), "seconds": round(m.seconds, 1),
            "peak_process": m.peak_process, "peak_all": m.peak_all, "jobs": b.jobs,
            "bytes": sum(o.stat().st_size for o in objects), "manifest": check}


def runtime_library(b: Build, p: Plan, on_line=None) -> None:
    """Build the runtime library unless a prebuilt one was given: the runtime's shaders compiled and their C file
    written first, then Ninja. An archive keeps members whose source is gone, so it is deleted first when the list of
    runtime objects changed."""
    if b.runtime_lib:
        if not b.runtime_lib.is_file():
            raise BuildError(f"the runtime library {b.runtime_lib} is missing")
        return
    if b.tool.glslang is None:
        raise BuildError("the runtime's shaders cannot be compiled without glslang (find_toolchain)")
    write_shader_glue(b.runtime, b.out, b.tool.glslang)
    listing = "\n".join(rel(o, b.out) for _, o in p.rt_objects) + "\n"
    if write_if_changed(b.out / "obj" / RUNTIME_LIST, listing):
        b.lib.unlink(missing_ok=True)
    ninja(b, ["runtime_lib"], on_line)


def link_program(b: Build, on_line=None) -> dict:
    """Write the program glue, check every object against the manifest, then build the library and link."""
    mods, lifted, p = prepare(b)
    text, baked = program_glue(b, mods, lifted)
    write_if_changed(p.glue, text)
    ninja(b, ["gen_objects", rel(p.glue_object, b.out)], on_line)
    runtime_library(b, p, on_line)
    objects = [obj for pairs in p.gen_objects.values() for _, obj in pairs]
    objects += [obj for _, obj in p.rt_objects] + [p.glue_object]
    check = verify_objects(b, objects, on_line, prune=True)
    try:
        m = ninja(b, [rel(b.exe, b.out)], on_line)
    except BuildError as e:
        raise BuildError(f"{e}\n(a running {b.exe.name} locks the file: close it and try again)") from None
    st = b.exe.stat()
    image, sections = executable_layout(b.exe)
    return {"exe": str(b.exe), "exe_bytes": st.st_size, "exe_mtime_ns": st.st_mtime_ns, "image_bytes": image,
            "sections": sections, "seconds": round(m.seconds, 1), "peak_process": m.peak_process,
            "peak_all": m.peak_all, "manifest": check, "baked": baked}


def make_runtime_lib(b: Build, on_line=None) -> dict:
    p = plan(b, [])
    b.out.mkdir(parents=True, exist_ok=True)
    write_if_changed(b.out / "build.ninja", p.text)
    runtime_library(b, p, on_line)
    check = verify_objects(b, [obj for _, obj in p.rt_objects], on_line)
    return {"lib": str(b.lib), "bytes": b.lib.stat().st_size, "objects": len(p.rt_objects), "manifest": check}


def executable_layout(path: Path) -> tuple[int, dict[str, int]]:
    """(image size, {section: virtual size}) of a PE32+ program; (file size, {}) for any other file."""
    with open(path, "rb") as f:
        data = f.read(4096)
    if data[:2] != b"MZ":
        return path.stat().st_size, {}
    pe = int.from_bytes(data[0x3C:0x40], "little")
    count = int.from_bytes(data[pe + 6:pe + 8], "little")
    opt = pe + 24
    image = int.from_bytes(data[opt + 56:opt + 60], "little")
    table = opt + int.from_bytes(data[pe + 20:pe + 22], "little")
    sections = {}
    for i in range(count):
        s = data[table + 40 * i:table + 40 * (i + 1)]
        name = s[:8].rstrip(b"\0").decode(errors="replace")
        sections[name] = sections.get(name, 0) + int.from_bytes(s[8:12], "little")
    return image, sections


# ------------------------------------------------------------------------------------------------ command line

def mib(n: int) -> str:
    return f"{n / (1 << 20):,.1f} MiB"


def parse_file(text: str) -> tuple[str, Path]:
    what, sep, path = text.partition("=")
    if not sep or not what or not path:
        raise argparse.ArgumentTypeError(f"expected WHAT=PATH, got {text!r}")
    return what, Path(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen", type=Path, help="the lifter's output folder")
    ap.add_argument("--exefs", type=Path, help="the plain modules and main.npdm")
    ap.add_argument("--out", type=Path, required=True, help="build folder")
    ap.add_argument("--exe", type=Path, help=f"the program to write (default {EXE_NAME} in --out)")
    ap.add_argument("--jobs", type=int, help="compile jobs (default: by CPU count and free memory)")
    ap.add_argument("--cc", help="C compiler (default: OOT2_CC, else clang)")
    ap.add_argument("--cxx", help="C++ compiler and linker (default: OOT2_CXX, else clang++ beside the C compiler)")
    ap.add_argument("--ninja", help="Ninja (default: OOT2_NINJA, else ninja on PATH)")
    ap.add_argument("--vulkan-sdk", type=Path,
                    help="Vulkan SDK with the headers, Volk and glslang (default: the VULKAN_SDK variable)")
    ap.add_argument("--runtime", type=Path, default=RUNTIME, help="runtime source folder (default runtime/)")
    ap.add_argument("--runtime-lib", type=Path, help=f"a prebuilt {LIB_NAME} to link instead of building one")
    ap.add_argument("--make-runtime-lib", action="store_true", help=f"build only {LIB_NAME} into --out")
    ap.add_argument("--compile-only", action="store_true", help="compile the lifted C, do not link")
    ap.add_argument("--file", type=parse_file, action="append", default=[], metavar="WHAT=PATH",
                    help="another file the program checks by SHA-256 at each start")
    ap.add_argument("--file-size", type=parse_file, action="append", default=[], metavar="WHAT=PATH",
                    help="another file the program checks by size at each start")
    args = ap.parse_args(argv)
    if not args.make_runtime_lib and not (args.gen and args.exefs):
        ap.error("--gen and --exefs are required unless --make-runtime-lib")
    jobs, why = (args.jobs, "--jobs") if args.jobs else default_jobs()

    def show(line: str) -> None:
        if not NINJA_LINE.match(line):
            print(line, flush=True)

    try:
        tool = find_toolchain(args.cc, args.ninja, vulkan_sdk=args.vulkan_sdk, cxx=args.cxx)
        files = [GameFile(w, p.resolve(), True) for w, p in args.file]
        files += [GameFile(w, p.resolve(), False) for w, p in args.file_size]
        b = Build(args.out, tool, gen=args.gen.resolve() if args.gen else None,
                  exefs=args.exefs.resolve() if args.exefs else None, exe=args.exe, runtime=args.runtime.resolve(),
                  runtime_lib=args.runtime_lib.resolve() if args.runtime_lib else None, files=files, jobs=jobs)
        print(f"build_game: {tool.cc} ({tool_id(tool.cc)}), C++ {tool.cxx}, {jobs} jobs ({why})", flush=True)
        if args.make_runtime_lib:
            r = make_runtime_lib(b, show)
            print(f"{r['lib']}: {mib(r['bytes'])}, {r['objects']} objects")
            return 0
        c = compile_generated(b, show)
        print(f"compile: {c['objects']} objects ({c['functions']:,} functions), {c['seconds']:.1f} s, peak "
              f"{mib(c['peak_process'])} per process" + (f", {mib(c['peak_all'])} all processes" if WIN else "") +
              f"; objects verified {c['manifest']['verified']}, recompiled {c['manifest']['recompiled']}")
        if args.compile_only:
            return 0
        k = link_program(b, show)
        print(f"link: {k['seconds']:.1f} s, peak {mib(k['peak_process'])}; objects verified "
              f"{k['manifest']['verified']}, recompiled {k['manifest']['recompiled']}")
        print(f"{k['exe']}: file {mib(k['exe_bytes'])}, image {mib(k['image_bytes'])}; " +
              ", ".join(f"{n} {mib(v)}" for n, v in k["sections"].items() if v))
    except BuildError as e:
        print(f"build_game: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
