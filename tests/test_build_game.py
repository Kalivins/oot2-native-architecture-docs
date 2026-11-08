# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the build and link (tools/build_game.py, runtime/program/main.c) and of the lift, compile and link steps of
`oot2 prepare` (tools/prepare/program.py).

The pieces (Ninja file, program glue, main.npdm reader, object manifest) are tested without a compiler. The rest
builds a program end to end from two synthetic modules written here (tests/lift/synth.py; no game bytes): lifted by
tools/lift/gen.py, compiled with Ninja, linked with the runtime library into a program that is then run. It loads the
modules, builds its dispatch table and starts the main thread in the boot driver (runtime/hos/boot.c), which calls the
synthetic sdk's nn::init::Start; that calls the three callbacks in turn, and the third ends the run as the game ending
does (a guest stop), with census.tsv written. Each test that changes a file works on a copy, or leaves the build as
valid as it found it.

Needs clang for x86-64, Ninja and an archiver (OOT2_CC, OOT2_NINJA, else PATH). Without them those tests skip, unless
OOT2_REQUIRE_CLANG is set, as continuous integration sets it. The steps also need capstone, which the lift step's
spin-wait search imports: without it they skip, unless OOT2_REQUIRE_CAPSTONE is set.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "lift"))
sys.path.insert(0, str(ROOT / "tests"))

import build_game  # noqa: E402
import fetch_imgui  # noqa: E402
import fetch_sdl3  # noqa: E402
import layout  # noqa: E402
import synth  # noqa: E402
from test_fetch_sdl3 import fake_sdl3_kit  # noqa: E402
from prepare import program as steps  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, Reporter, Step, run_steps  # noqa: E402

GEN = ROOT / "tools" / "lift" / "gen.py"
EXIT_FATAL, EXIT_UNIMPLEMENTED, EXIT_GUEST_STOP = 70, 71, 72
RET, NOP, ADD_X0_1 = 0xD65F03C0, 0xD503201F, 0x91000400
POLL = [0xB9400001, 0x34FFFFE1, RET]   # ldr w1, [x0]; cbz w1, .-4; ret: a spin wait (tests/test_spin_waits.py)
INIT_START = "_ZN2nn4init5StartEmmPFvvES2_S2_"
# nn::init::Start(handle, arguments, loader_ready, module_inits, module_finis) reduced to its callbacks: stp x29, x30,
# [sp, #-32]!; str x19, [sp, #16]; mov x19, x4; blr x2; blr x3; blr x19; ldr x19, [sp, #16]; ldp x29, x30, [sp], #32;
# ret. The third callback does not return (the game ended).
START = [0xA9BE7BFD, 0xF9000BF3, 0xAA0403F3, 0xD63F0040, 0xD63F0060, 0xD63F0260, 0xF9400BF3, 0xA8C27BFD, RET]
BOOT_STOP = ("FATAL: guest stop: nnMain returned and nn::init::Start asked for the module finalizers: the game ended "
             "(pc 0x7ff")
WIN = os.name == "nt"


def module(functions: list[list[int]], names: list[str]) -> bytes:
    """A plain module: two reserved words, then each function at a 16-byte boundary, exported under names."""
    text, exports = bytearray(8), {}
    for name, fn in zip(names, functions):
        text += bytes(-len(text) % 16)
        exports[name] = len(text)
        text += struct.pack(f"<{len(fn)}I", *fn)
    return synth.Module(bytes(text), exports).build()


# The kernel capabilities of the synthetic ACI0, as tests/runtime/kernel_test.c's npdm case writes them: ThreadInfo
# (priorities 28 to 59, cores 0 to 5), two EnableSystemCalls words, HandleTableSize 512, MiscParams (an application).
KERNEL_CAPS = (0x050073B7, 0x1FFFFFCF, 0xA000200F, 0x00005FFF, 0x02007FFF, 0xFFFFFFFF)


def npdm(priority=49, core=1, system_resource=0x200000, stack=0x80000) -> bytes:
    """A META header with the fields the build reads, and an ACI0 with the kernel capabilities the runtime reads
    (rt_npdm_parse); nothing else of a real description."""
    raw = bytearray(0x180)
    raw[0:4] = b"META"
    raw[0xC], raw[0xE], raw[0xF] = 0x37, priority, core
    struct.pack_into("<III", raw, 0x14, system_resource, 0, stack)
    raw[0x20:0x30] = b"Synthetic".ljust(16, b"\0")
    struct.pack_into("<II", raw, 0x70, 0x80, 0x100)              # ACI0 offset and size
    raw[0x80:0x84] = b"ACI0"
    struct.pack_into("<II", raw, 0x80 + 0x30, 0x40, 4 * len(KERNEL_CAPS))   # kernel capabilities, inside ACI0
    struct.pack_into(f"<{len(KERNEL_CAPS)}I", raw, 0x80 + 0x40, *KERNEL_CAPS)
    return bytes(raw)


def write_exefs(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "main").write_bytes(module([[ADD_X0_1, RET], [NOP, RET], [ADD_X0_1, ADD_X0_1, RET]], ["f0", "f1", "f2"]))
    # The sdk defines a function the runtime replaces (runtime/core/jmp.c): the loader reports it replaced only when
    # jmp.c is in the program, and nothing here references jmp.c, so only the whole-archive link brings it in.
    (folder / "sdk").write_bytes(module([[RET], [RET], START], ["f0", "_Unwind_RaiseException", INIT_START]))
    (folder / "main.npdm").write_bytes(npdm())


def toolchain_or_skip() -> build_game.Toolchain:
    try:
        return build_game.find_toolchain()
    except build_game.BuildError as e:
        if os.environ.get("OOT2_REQUIRE_CLANG"):
            raise RuntimeError(f"OOT2_REQUIRE_CLANG is set but the build tools are missing: {e}") from None
        raise unittest.SkipTest(str(e))


def capstone_or_skip() -> None:
    try:
        import capstone  # noqa: F401
    except ImportError as e:
        if os.environ.get("OOT2_REQUIRE_CAPSTONE"):
            raise RuntimeError(f"OOT2_REQUIRE_CAPSTONE is set but capstone cannot be imported: {e}") from None
        raise unittest.SkipTest(f"capstone is not installed (pip install capstone==5.0.7): {e}")


def lift(exefs: Path, out: Path) -> None:
    done = subprocess.run([sys.executable, "-I", str(GEN), "--exefs", str(exefs), "--out", str(out),
                           "--modules", "main,sdk", "--jobs", "2"], capture_output=True, text=True, timeout=300)
    if done.returncode:
        raise RuntimeError(f"gen.py failed:\n{done.stderr}")


def copy_runtime(dst: Path) -> Path:
    shutil.copytree(ROOT / "runtime", dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def run_program(exe: Path, *args: str, cwd: Path | None = None,
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run the program with this process's environment, less OOT2_GFX, OOT2_TIMER_RESOLUTION (a shell that turned the
    timer request off would silently change the log of every run) and census mode (OOT2_CENSUS*, a run for a gap list,
    never inherited from the shell that runs the suite), plus env."""
    full = {k: v for k, v in os.environ.items()
            if k not in ("OOT2_GFX", "OOT2_TIMER_RESOLUTION") and not k.startswith("OOT2_CENSUS")}
    full.update(env or {})
    return subprocess.run([str(exe), *args], cwd=cwd or exe.parent.parent, capture_output=True, text=True, timeout=120,
                          env=full)


def flip_byte(path: Path, offset: int, keep_time: bool = False) -> None:
    st = path.stat()
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x40
    path.write_bytes(bytes(data))
    if keep_time:
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def pe_imports(path: Path) -> list[str]:
    """The DLL names a PE32+ program imports, in table order (the import directory, the second data directory)."""
    data = path.read_bytes()
    pe = int.from_bytes(data[0x3C:0x40], "little")
    count = int.from_bytes(data[pe + 6:pe + 8], "little")
    opt = pe + 24
    table = opt + int.from_bytes(data[pe + 20:pe + 22], "little")
    directory = int.from_bytes(data[opt + 120:opt + 124], "little")
    sections = [struct.unpack_from("<IIII", data, table + 40 * i + 8) for i in range(count)]   # size, rva, raw size, raw

    def offset(rva: int) -> int:
        for size, base, raw_size, raw in sections:
            if base <= rva < base + max(size, raw_size):
                return rva - base + raw
        raise ValueError(f"RVA {rva:#x} is in no section")

    names, at = [], offset(directory) if directory else None
    while at is not None:
        name_rva = struct.unpack_from("<5I", data, at)[3]
        if not name_rva:
            break
        start = offset(name_rva)
        names.append(data[start:data.index(b"\0", start)].decode("ascii"))
        at += 20
    return names


def lifted_objects(build: Path) -> int:
    """Objects of the lifted modules (obj/main, obj/sdk); the runtime's are in obj/runtime."""
    return sum(len(list((build / "obj" / m).glob("*.o"))) for m in ("main", "sdk"))


def fake_tool(folder: Path) -> build_game.Toolchain:
    return build_game.Toolchain(str(folder / "my tools" / "clang.exe"), str(folder / "llvm-ar.exe"),
                                str(folder / "ninja.exe"), folder / "Vulkan SDK" / "Include",
                                str(folder / "Vulkan SDK" / "Bin" / "glslang.exe"),
                                str(folder / "my tools" / "clang++.exe"))


def fake_gen(gen: Path, chunks: dict[str, int], base_of: dict[str, int] | None = None,
             marker: bool = True) -> None:
    """A lifter output without C: layout.json and per module a manifest listing empty files (and, unless marker is
    False, the end marker the lifter writes at the end of the function table)."""
    mods = layout.place({name: 0x10000 for name in ("main", "sdk")})
    layout.write(mods, gen)
    for m in mods:
        if m.name not in chunks:
            continue
        (gen / m.name).mkdir(parents=True, exist_ok=True)
        files = [f"{m.name}_{i:04d}.c" for i in range(chunks[m.name])]
        for f in files:
            (gen / m.name / f).write_text("", encoding="utf-8")
        table = f"void {m.name}_lifted_end(Cpu *c) {{}}\n" if marker else ""
        (gen / m.name / f"{m.name}_table.c").write_text(table, encoding="utf-8")
        base = (base_of or {}).get(m.name, m.base)
        (gen / m.name / "manifest.json").write_text(
            json.dumps({"module": m.name, "marker": build_game.MARKER, "base": f"{base:#x}",
                                   "chunks": files, "table": f"{m.name}_table.c", "function_count": 7,
                                   **({"end_marker": f"{m.name}_lifted_end"} if marker else {})}),
            encoding="utf-8")


class PiecesTest(unittest.TestCase):
    """The build's parts that need no compiler."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2 build-")
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_ninja_file_compiles_generated_c_at_o1_and_links_the_library_whole(self) -> None:
        gen = self.dir / "game folder" / "gen"
        fake_gen(gen, {"main": 2, "sdk": 1})
        b = build_game.Build(self.dir / "game folder" / "build", fake_tool(self.dir), gen=gen, exefs=self.dir, jobs=3)
        _, lifted = build_game.lifted_modules(gen)
        text = build_game.plan(b, lifted).text
        genflags = next(line for line in text.splitlines() if line.startswith("genflags = "))
        self.assertIn("-O1 -march=x86-64-v3", genflags)
        self.assertIn("-Wl,--whole-archive $lib -Wl,--no-whole-archive", text)
        self.assertEqual("--disable-high-entropy-va" in text, WIN)
        # Objects and sources relative to the build folder, so no path on a build line holds the space of the folder.
        self.assertIn("build obj/main/main_0000.o: cc_gen ../gen/main/main_0000.c", text)
        self.assertIn("build obj/sdk/sdk_table.o: cc_gen ../gen/sdk/sdk_table.c", text)
        self.assertIn('cc = "', text)   # the compiler's folder has a space: quoted on the command line
        self.assertIn(f"build {build_game.EXE_NAME}: link ", text)
        self.assertIn(f" obj/program_glue.o | {build_game.LIB_NAME}", text)
        self.assertNotIn(" -I ../gen", next(line for line in text.splitlines() if line.startswith("rtflags = ")))

    def test_ninja_escapes_spaces_and_drive_colons_on_build_lines(self) -> None:
        self.assertEqual(build_game.ninja_path(self.dir / "a b" / "c.o", self.dir), "a$ b/c.o")
        if WIN:   # another drive: no relative path, so the drive's colon is escaped
            other = "D:/x" if not str(self.dir).upper().startswith("D:") else "E:/x"
            self.assertEqual(build_game.ninja_path(Path(other) / "a b.o", self.dir), other[0] + "$:/x/a$ b.o")
        self.assertEqual(build_game.cmd_arg("a b$c"), '"a b$$c"')

    def test_cpp_sources_have_their_own_rule_and_the_program_is_linked_by_the_cxx_driver(self) -> None:
        # The C++ build check: runtime/ui is C++ behind a C header. A .cpp file is compiled as C++17 without exceptions and
        # without run-time type information, and the C++ driver links, with -static on Windows so that no DLL of the
        # toolchain sits beside oot2.exe.
        gen = self.dir / "game folder" / "gen"
        fake_gen(gen, {"main": 1})
        b = build_game.Build(self.dir / "game folder" / "build", fake_tool(self.dir), gen=gen, exefs=self.dir, jobs=3)
        _, lifted = build_game.lifted_modules(gen)
        text = build_game.plan(b, lifted).text
        lines = text.splitlines()
        self.assertIn("build obj/runtime/ui_ui_standin.o: cxx_rt ", text)
        self.assertIn("build obj/runtime/gpu_gpu.o: cc_rt ", text)   # a C file keeps the C rule
        cxxflags = next(line for line in lines if line.startswith("cxxflags = ")).split()
        for flag in ("-std=c++17", "-O2", "-march=x86-64-v3", "-fno-exceptions", "-fno-rtti"):
            self.assertIn(flag, cxxflags)
        self.assertIn("command = $cxx $cxxflags ", lines[lines.index("rule cxx_rt") + 1])
        self.assertIn("command = $cxx $ldflags ", lines[lines.index("rule link") + 1])
        self.assertEqual("-static" in next(line for line in lines if line.startswith("ldflags = ")).split(), WIN)

    def test_runtime_sources_take_c_and_cpp_and_one_object_name_is_not_made_twice(self) -> None:
        rt = self.dir / "rt"
        for name in ("a/x.c", "a/y.cpp", "a/z.h", "b/w.cpp", "b/v.cc", "c/deep/q.cpp"):
            (rt / name).parent.mkdir(parents=True, exist_ok=True)
            (rt / name).write_text("", encoding="utf-8")
        got = [p.relative_to(rt).as_posix() for p in build_game.runtime_sources(rt)]
        self.assertEqual(got, ["a/x.c", "a/y.cpp", "b/w.cpp"])
        (rt / "a" / "x.cpp").write_text("", encoding="utf-8")   # x.c and x.cpp would both make obj/runtime/a_x.o
        b = build_game.Build(self.dir / "build", fake_tool(self.dir), runtime=rt, jobs=1)
        with self.assertRaisesRegex(build_game.BuildError, r"x\.c and .*x\.cpp both make a_x\.o"):
            build_game.plan(b, [])

    def test_the_cxx_driver_is_the_c_compilers_sibling_unless_named(self) -> None:
        suffix = ".exe" if WIN else ""
        with mock.patch.dict(os.environ):
            os.environ.pop("OOT2_CXX", None)
            for cc, cxx in (("clang", "clang++"), ("clang-18", "clang++-18"),
                            ("x86_64-w64-mingw32-clang", "x86_64-w64-mingw32-clang++")):
                folder = self.dir / cc
                folder.mkdir()
                for name in (cc, cxx):
                    (folder / (name + suffix)).write_text("", encoding="utf-8")
                found = build_game.find_cxx(str(folder / (cc + suffix)))
                self.assertEqual(Path(found).name, cxx + suffix)
            # a C compiler whose sibling is nowhere: None, and the toolchain says so
            lone = self.dir / "lone"
            lone.mkdir()
            (lone / ("clang-no-such-version" + suffix)).write_text("", encoding="utf-8")
            self.assertIsNone(build_game.find_cxx(str(lone / ("clang-no-such-version" + suffix))))
            # naming one wins over the sibling, and a name that is nowhere is None
            self.assertIsNone(build_game.find_cxx(str(lone / ("clang-no-such-version" + suffix)), "no-such-cxx-driver"))
            os.environ["OOT2_CXX"] = "no-such-cxx-driver"
            self.assertIsNone(build_game.find_cxx(str(self.dir / "clang" / ("clang" + suffix))))

    def test_a_module_without_an_end_marker_is_refused_by_name(self) -> None:
        # The lifted-code end marker: the marker says where a module's lifted code ends; a lift made without it cannot be built.
        gen = self.dir / "gen"
        fake_gen(gen, {"main": 1}, marker=False)
        with self.assertRaisesRegex(build_game.BuildError, "end marker missing: the manifest of main does not name "
                                                           "main_lifted_end"):
            build_game.lifted_modules(gen)
        fake_gen(gen, {"main": 1})   # the manifest names it, but the table does not define it
        (gen / "main" / "main_table.c").write_text("const unsigned long long main_fn_count = 1;\n", encoding="utf-8")
        with self.assertRaisesRegex(build_game.BuildError, "end marker missing: .*main_table.c does not define "
                                                           "main_lifted_end"):
            build_game.lifted_modules(gen)
        fake_gen(gen, {"main": 1})
        _, lifted = build_game.lifted_modules(gen)   # and with the marker, it builds
        self.assertEqual(lifted[0].sources[-1].name, "main_table.c")   # the table, which holds the marker, links last

    def test_glue_hands_the_runtime_each_modules_end_marker(self) -> None:
        exefs = self.dir / "exefs"
        write_exefs(exefs)
        gen = self.dir / "gen"
        fake_gen(gen, {"main": 1})
        b = build_game.Build(self.dir / "build", fake_tool(self.dir), gen=gen, exefs=exefs, exe=self.dir / "oot2.exe",
                             jobs=1)
        mods, lifted = build_game.lifted_modules(gen)
        text, _ = build_game.program_glue(b, mods, lifted)
        self.assertIn("extern void main_lifted_end(Cpu *c);", text)
        self.assertIn("main_fn_count, (const void *)main_lifted_end};", text)

    def test_a_manifest_that_disagrees_with_the_layout_is_refused(self) -> None:
        gen = self.dir / "gen"
        fake_gen(gen, {"main": 1}, base_of={"main": 0x7100000000})
        with self.assertRaisesRegex(build_game.BuildError, "places main at 0x7100000000"):
            build_game.lifted_modules(gen)

    def test_a_missing_chunk_is_refused_and_an_unlifted_module_is_only_loaded(self) -> None:
        gen = self.dir / "gen"
        fake_gen(gen, {"main": 2})
        mods, lifted = build_game.lifted_modules(gen)
        self.assertEqual([m.name for m in mods], ["main", "sdk"])
        self.assertEqual([m.name for m in lifted], ["main"])
        (gen / "main" / "main_0001.c").unlink()
        with self.assertRaisesRegex(build_game.BuildError, "1 files listed"):
            build_game.lifted_modules(gen)

    def test_glue_bakes_hashes_paths_and_npdm_values(self) -> None:
        exefs = self.dir / "exefs"
        write_exefs(exefs)
        gen = self.dir / "gen"
        fake_gen(gen, {"main": 1})
        b = build_game.Build(self.dir / "build", fake_tool(self.dir), gen=gen, exefs=exefs, exe=self.dir / "oot2.exe",
                             files=[build_game.GameFile("Big file", exefs / "sdk", False)], jobs=1)
        mods, lifted = build_game.lifted_modules(gen)
        text, baked = build_game.program_glue(b, mods, lifted)
        self.assertIn(build_game.MARKER, text)
        self.assertIn("_Static_assert(MOD_MAIN_BASE == 0x7000000000ull", text)
        self.assertIn(f'{{"main", "exefs/main", MOD_MAIN_BASE, MOD_MAIN_SIZE, '
                      f'"{build_game.file_sha256(exefs / "main")}"}}', text)
        self.assertIn(f'{{"main.npdm", "exefs/main.npdm", {len(npdm())}ull, '
                      f'"{build_game.file_sha256(exefs / "main.npdm")}", NULL}}', text)
        self.assertIn(f'{{"Big file", "exefs/sdk", {(exefs / "sdk").stat().st_size}ull, NULL, NULL}}', text)
        self.assertIn("out[n++] = (RtModule){\"main\"", text)
        self.assertNotIn("sdk_fn_addrs", text)     # sdk is loaded, not lifted
        self.assertIn(".system_resource_size = 0x200000ull", text)
        self.assertIn(".main_thread_priority = 49", text)
        self.assertEqual(baked["npdm"]["main_thread_stack_size"], 0x80000)

    def test_sample_ranges_cover_the_start_the_end_and_between(self) -> None:
        n = build_game.SAMPLE_BYTES
        self.assertEqual(build_game.sample_ranges(10), [(0, 10)] * build_game.SAMPLE_COUNT)   # a small file whole
        for size in (n, n + 1, 3 * n + 7, 25_622_267_860):
            with self.subTest(size=size):
                ranges = build_game.sample_ranges(size)
                self.assertEqual(len(ranges), build_game.SAMPLE_COUNT)
                self.assertEqual(ranges[0], (0, n))
                self.assertEqual(ranges[-1], (size - n, n))
                self.assertEqual([o for o, _ in ranges], sorted(o for o, _ in ranges))
                self.assertEqual(ranges[1][0], (size - n) // (build_game.SAMPLE_COUNT - 1))

    def test_glue_bakes_the_sampled_hash_of_a_dump_read_in_place(self) -> None:
        exefs = self.dir / "exefs"
        write_exefs(exefs)
        gen = self.dir / "gen"
        fake_gen(gen, {"main": 1})
        dump = self.dir / "dumps" / "game.nup"
        dump.parent.mkdir()
        dump.write_bytes(bytes(range(256)) * 1000)
        b = build_game.Build(self.dir / "build" / "here", fake_tool(self.dir), gen=gen, exefs=exefs,
                             exe=self.dir / "build" / "here" / "oot2.exe",
                             files=[build_game.GameFile("RomFS image", dump, False, sampled=True)], jobs=1)
        text, baked = build_game.program_glue(b, *build_game.lifted_modules(gen))
        sample = build_game.sample_sha256(dump)
        self.assertIn(f'{{"RomFS image", "{dump.resolve().as_posix()}", 256000ull, NULL, "{sample}"}}', text)
        self.assertEqual(baked["files"]["RomFS image"], {"size": 256000, "sample_sha256": sample})
        data = bytearray(dump.read_bytes())
        data[build_game.sample_ranges(len(data))[7][0] + 5] ^= 1
        dump.write_bytes(bytes(data))
        self.assertNotEqual(build_game.sample_sha256(dump), sample)

    def test_files_outside_the_program_folder_keep_absolute_paths(self) -> None:
        self.assertEqual(build_game.game_path(self.dir / "a" / "b", self.dir), "a/b")
        outside = self.dir.parent / "elsewhere"
        self.assertEqual(build_game.game_path(outside, self.dir), outside.resolve().as_posix())

    def test_npdm_needs_its_header(self) -> None:
        path = self.dir / "main.npdm"
        path.write_bytes(npdm(priority=44, core=0, system_resource=0, stack=0x100000))
        self.assertEqual(build_game.read_npdm(path), {"flags": 0x37, "main_thread_priority": 44, "main_thread_core": 0,
                                                     "system_resource_size": 0, "main_thread_stack_size": 0x100000})
        path.write_bytes(b"NPDM" + bytes(0x7C))
        with self.assertRaisesRegex(build_game.BuildError, "no META header"):
            build_game.read_npdm(path)

    def test_manifest_catches_a_byte_changed_in_place_and_records_a_rebuilt_object(self) -> None:
        obj = self.dir / "obj"
        a, c = obj / "main" / "a.o", obj / "runtime" / "core" / "c.o"
        for p in (a, c):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(bytes(range(64)))
        man: dict = {}
        self.assertEqual(build_game.check_objects(obj, [a, c], man), (2, []))
        self.assertEqual(set(man), {"main/a.o", "runtime/core/c.o"})
        flip_byte(a, 3, keep_time=True)
        self.assertEqual(build_game.check_objects(obj, [a, c], man), (1, [a]))
        a.write_bytes(bytes(64))                   # rebuilt: a new time, recorded
        st = a.stat()
        os.utime(a, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        self.assertEqual(build_game.check_objects(obj, [a], man), (1, []))
        self.assertEqual(man["main/a.o"][0], build_game.file_sha256(a))

    def test_check_step_needs_ninja(self) -> None:
        from prepare import check
        from prepare.runner import PrepareError
        if not shutil.which(os.environ.get("OOT2_CC", "clang")):
            self.skipTest("no clang on PATH (set OOT2_CC)")
        ctx = argparse.Namespace(options=argparse.Namespace(cc=None))
        with mock.patch.dict(os.environ, {"OOT2_NINJA": "no-such-ninja-here"}):
            with self.assertRaisesRegex(PrepareError, r"Ninja \('no-such-ninja-here'\) was not found"):
                check.build_tools(ctx)

    def test_runtime_flags_reach_the_vulkan_headers_and_volk(self) -> None:
        gen = self.dir / "gen"
        fake_gen(gen, {"main": 1})
        tool = fake_tool(self.dir)
        b = build_game.Build(self.dir / "build", tool, gen=gen, exefs=self.dir, jobs=1)
        _, lifted = build_game.lifted_modules(gen)
        text = build_game.plan(b, lifted).text

        def flags(name: str) -> str:
            return next(line for line in text.splitlines() if line.startswith(f"{name} = "))
        vk = build_game.rel(tool.vulkan, b.out)
        self.assertTrue(flags("rtflags").endswith(f" -I {build_game.cmd_arg(vk)} -I {build_game.cmd_arg(vk + '/Volk')}"),
                        flags("rtflags"))
        for name in ("genflags", "glueflags"):   # generated code and the program glue never include them
            self.assertNotIn("Vulkan SDK", flags(name))
        self.assertIn("build obj/runtime/gpu_vk_volk.o: cc_rt ", text)
        self.assertIn("build obj/runtime/gpu_gpu.o: cc_rt ", text)
        self.assertIn("build obj/runtime/spirv_gpu_shaders.o: cc_rt spirv/gpu_shaders.c", text)
        without = dataclasses.replace(tool, vulkan=None)
        with self.assertRaisesRegex(build_game.BuildError, "without the Vulkan headers and Volk"):
            build_game.plan(build_game.Build(self.dir / "build", without, gen=gen, exefs=self.dir, jobs=1), lifted)

    def make_kit(self, name: str = "dav1d kit") -> Path:
        """A folder laid out as tools/fetch_dav1d.py lays out a kit (the files are empty: nothing here links them)."""
        kit = self.dir / name
        (kit / "include" / "dav1d").mkdir(parents=True)
        (kit / "include" / "dav1d" / "dav1d.h").write_text("", encoding="utf-8")
        (kit / "lib").mkdir()
        (kit / "lib" / "libdav1d.a").write_bytes(b"!<arch>\n")
        return kit.resolve()

    def test_a_dav1d_kit_reaches_the_runtime_flags_the_libraries_and_the_link(self) -> None:
        gen = self.dir / "gen"
        fake_gen(gen, {"main": 1})
        tool = fake_tool(self.dir)
        kit = self.make_kit()

        def plan_text(t: build_game.Toolchain) -> tuple[str, Path]:
            b = build_game.Build(self.dir / "build", t, gen=gen, exefs=self.dir, jobs=1)
            _, lifted = build_game.lifted_modules(gen)
            return build_game.plan(b, lifted).text, b.out

        def flags(text: str, name: str) -> str:
            return next(line for line in text.splitlines() if line.startswith(f"{name} = "))

        text, out = plan_text(dataclasses.replace(tool, dav1d=kit))
        include = build_game.cmd_arg(build_game.rel(kit / "include", out))
        lib = build_game.rel(kit / "lib" / "libdav1d.a", out)
        self.assertIn(f" -DOOT2_DAV1D=1 -I {include}", flags(text, "rtflags"))   # the runtime's AV1 backend, not the lifted code
        for name in ("genflags", "glueflags"):
            self.assertNotIn("DAV1D", flags(text, name))
        self.assertIn(build_game.cmd_arg(lib), flags(text, "libs").split(" ", 2)[2])   # the program links the library
        link = next(line for line in text.splitlines() if line.startswith(f"build {build_game.EXE_NAME}: link "))
        # an implicit input of the link: a new kit relinks the program
        self.assertTrue(link.endswith(f" | {build_game.LIB_NAME} {build_game.ninja_path(kit / 'lib' / 'libdav1d.a', out)}"), link)
        if not WIN:   # dav1d's own threads
            self.assertIn("-pthread", flags(text, "libs"))
        text, _ = plan_text(tool)
        self.assertNotIn("libdav1d", text)   # (runtime/video/av1_dav1d.c is in the library either way: the stub)
        self.assertNotIn("DAV1D", text)

    def test_a_dav1d_kit_is_found_by_its_header_and_library_and_keys_the_link(self) -> None:
        kit = self.make_kit()
        self.assertEqual(build_game.find_dav1d(kit), kit)
        with mock.patch.dict(os.environ, {"DAV1D_DIR": str(kit)}):
            self.assertEqual(build_game.find_dav1d(None), kit)   # else the DAV1D_DIR variable
        with mock.patch.dict(os.environ):
            os.environ.pop("DAV1D_DIR", None)
            self.assertIsNone(build_game.find_dav1d(None))
        self.assertIsNone(build_game.find_dav1d(self.dir / "no kit here"))
        self.assertEqual(build_game.dav1d_inputs(kit), [kit / "lib" / "libdav1d.a", kit / "include" / "dav1d" / "dav1d.h"])
        self.assertEqual(build_game.dav1d_inputs(None), ["no dav1d kit"])
        (kit / "lib" / "libdav1d.a").unlink()
        self.assertIsNone(build_game.find_dav1d(kit))   # a kit with no library is no kit
        (kit / "lib" / "libdav1d.a").write_bytes(b"x")
        (kit / "include" / "dav1d" / "dav1d.h").unlink()
        self.assertIsNone(build_game.find_dav1d(kit))   # nor one with no header

    def test_the_prepare_check_says_whether_there_is_a_kit_and_the_link_key_holds_the_library(self) -> None:
        from prepare import check
        tool = fake_tool(self.dir)
        kit = self.make_kit()
        self.assertEqual(check.av1_note(dataclasses.replace(tool, dav1d=kit)), f"AV1 decoder dav1d kit {kit}")
        none = check.av1_note(tool)
        self.assertIn("no dav1d kit", none)
        self.assertIn("tools/fetch_dav1d.py", none)
        self.assertIn("DAV1D_DIR", none)
        ctx = argparse.Namespace(options=argparse.Namespace(vulkan_sdk=self.dir / "no SDK", dav1d_dir=kit))
        key = steps.vulkan_key(ctx)
        self.assertIn(kit / "lib" / "libdav1d.a", key)
        self.assertIn(kit / "include" / "dav1d" / "dav1d.h", key)
        ctx.options.dav1d_dir = None
        with mock.patch.dict(os.environ):
            os.environ.pop("DAV1D_DIR", None)
            self.assertEqual(steps.vulkan_key(ctx), ["no Vulkan SDK", "no dav1d kit"])

    def test_the_toolchain_and_the_check_step_carry_the_kit(self) -> None:
        """find_toolchain and prepare's build check take the kit from the option or the variable; every tool is this
        interpreter (they are only looked up), the Vulkan SDK a folder of empty headers."""
        from prepare import check
        exe = sys.executable
        sdk = self.dir / "SDK"
        for rel_name in ("Include/vulkan/vulkan_core.h", "Include/Volk/volk.c", "Include/Volk/volk.h"):
            (sdk / rel_name).parent.mkdir(parents=True, exist_ok=True)
            (sdk / rel_name).write_text("", encoding="utf-8")
        kit = self.make_kit()
        with mock.patch.dict(os.environ, {"OOT2_GLSLANG": exe, "OOT2_NINJA": exe}):
            os.environ.pop("DAV1D_DIR", None)
            self.assertEqual(build_game.find_toolchain(exe, exe, exe, vulkan_sdk=sdk, dav1d_dir=kit).dav1d, kit)
            self.assertIsNone(build_game.find_toolchain(exe, exe, exe, vulkan_sdk=sdk).dav1d)
            os.environ["DAV1D_DIR"] = str(kit)
            self.assertEqual(build_game.find_toolchain(exe, exe, exe, vulkan_sdk=sdk).dav1d, kit)
            os.environ.pop("DAV1D_DIR")
            ctx = argparse.Namespace(options=argparse.Namespace(cc=exe, vulkan_sdk=sdk, dav1d_dir=kit))
            self.assertTrue(check.build_tools(ctx).endswith(f", AV1 decoder dav1d kit {kit}"))
            ctx.options.dav1d_dir = None
            self.assertTrue(check.build_tools(ctx).endswith(", no dav1d kit (the movies cannot play: run "
                                                            "tools/fetch_dav1d.py and set DAV1D_DIR)"))
            self.assertEqual(steps.toolchain(argparse.Namespace(options=argparse.Namespace(cc=exe, vulkan_sdk=sdk,
                                                                                        dav1d_dir=kit))).dav1d, kit)

    def make_sdl3_kit(self, name: str = "sdl3 kit") -> Path:
        """A folder laid out as tools/fetch_sdl3.py lays out a kit and records it (synthetic files: nothing links them)."""
        return fake_sdl3_kit(self.dir / name)

    def plan_text(self, tool: build_game.Toolchain, with_modules: bool = True) -> tuple[str, Path]:
        gen = self.dir / "gen"
        if not gen.exists():
            fake_gen(gen, {"main": 1})
        b = build_game.Build(self.dir / "build", tool, gen=gen if with_modules else None, exefs=self.dir, jobs=1)
        _, lifted = build_game.lifted_modules(gen) if with_modules else (None, [])
        return build_game.plan(b, lifted).text, b.out

    @staticmethod
    def flags(text: str, name: str) -> str:
        return next(line for line in text.splitlines() if line.startswith(f"{name} = "))

    def test_an_sdl3_kit_reaches_the_runtime_flags_the_libraries_and_the_link(self) -> None:
        tool = fake_tool(self.dir)
        kit = self.make_sdl3_kit()
        text, out = self.plan_text(dataclasses.replace(tool, sdl3=kit))
        include = build_game.cmd_arg(build_game.rel(kit / "include", out))
        lib = build_game.cmd_arg(build_game.rel(kit / "lib" / "libSDL3.a", out))
        for name in ("rtflags", "cxxflags"):   # every file of the runtime, C and C++, as the platform layer and the menu need
            self.assertIn(f" -DOOT2_WITH_SDL3=1 -DOOT2_AUDIO_SDL3=1 -I {include}", self.flags(text, name))
        for name in ("genflags", "glueflags", "imguiflags"):   # generated code, the glue and Dear ImGui's core never see it
            self.assertNotIn("SDL3", self.flags(text, name))
        libs = shlex.split(self.flags(text, "libs")[len("libs = "):], posix=False)
        at = libs.index(lib)   # the program links the library, then the system libraries it needs, last of all
        self.assertEqual(libs[at + 1:], list(fetch_sdl3.LINK_LIBS))
        self.assertEqual(libs[:at], [*build_game.LINK_LIBS, *([] if WIN else ["-pthread"])])
        link = next(line for line in text.splitlines() if line.startswith(f"build {build_game.EXE_NAME}: link "))
        # an implicit input of the link: a new kit relinks the program
        self.assertTrue(link.endswith(f" | {build_game.LIB_NAME} {build_game.ninja_path(kit / 'lib' / 'libSDL3.a', out)}"), link)
        # the runtime library comes after the objects and before the libraries, which are where the kit's library goes
        rule = text[text.index("rule link"):].splitlines()[1]
        self.assertLess(rule.index("$lib "), rule.index("$libs"))

    def make_opus_kit(self) -> Path:
        """A folder laid out as tools/fetch_opus.py lays out a kit (the files are empty: nothing here links them)."""
        kit = self.dir / "opus kit"
        (kit / "include" / "opus").mkdir(parents=True)
        (kit / "include" / "opus" / "opus.h").write_text("", encoding="utf-8")
        (kit / "lib").mkdir()
        (kit / "lib" / "libopus.a").write_bytes(b"!<arch>\n")
        return kit.resolve()

    def test_a_libopus_kit_reaches_the_runtime_flags_the_libraries_and_the_link(self) -> None:
        tool = dataclasses.replace(fake_tool(self.dir), opus=self.make_opus_kit(), dav1d=self.make_kit(),
                                   sdl3=self.make_sdl3_kit())
        text, out = self.plan_text(tool)
        include = build_game.cmd_arg(build_game.rel(tool.opus / "include", out))
        lib = build_game.cmd_arg(build_game.rel(tool.opus / "lib" / "libopus.a", out))
        self.assertIn(f" -DOOT2_OPUS=1 -I {include}", self.flags(text, "rtflags"))   # the decoder's file, not the lifted code
        for name in ("genflags", "glueflags", "imguiflags"):
            self.assertNotIn("OPUS", self.flags(text, name))
        libs = shlex.split(self.flags(text, "libs")[len("libs = "):], posix=False)
        dav1d = libs.index(build_game.cmd_arg(build_game.rel(tool.dav1d / "lib" / "libdav1d.a", out)))
        sdl3 = libs.index(build_game.cmd_arg(build_game.rel(tool.sdl3 / "lib" / "libSDL3.a", out)))
        self.assertLess(dav1d, libs.index(lib))
        self.assertLess(libs.index(lib), sdl3)   # dav1d's library, libopus's, SDL3's, then the system libraries
        link = next(line for line in text.splitlines() if line.startswith(f"build {build_game.EXE_NAME}: link "))
        self.assertIn(f" {build_game.ninja_path(tool.opus / 'lib' / 'libopus.a', out)}", link.split(" | ", 1)[1])   # a new kit relinks
        only, _ = self.plan_text(dataclasses.replace(fake_tool(self.dir), opus=tool.opus))
        self.assertEqual(libs.count("-pthread"), 0 if WIN else 1)
        self.assertNotIn("-pthread", self.flags(only, "libs"))   # libopus needs no threads of its own

    def test_without_a_libopus_kit_nothing_of_it_is_in_the_plan(self) -> None:
        text, _ = self.plan_text(fake_tool(self.dir))
        self.assertNotIn("OPUS", text)
        self.assertNotIn("libopus.a", text)
        self.assertIn("obj/runtime/audio_opus_libopus.o: cc_rt ", text)   # the decoder's file is in the library either way: empty

    def test_without_an_sdl3_kit_nothing_of_it_is_in_the_plan(self) -> None:
        text, _ = self.plan_text(fake_tool(self.dir))
        self.assertNotIn("SDL3", text)
        self.assertNotIn("libSDL", text)
        lib_only, _ = self.plan_text(fake_tool(self.dir), with_modules=False)   # what --make-runtime-lib plans
        self.assertNotIn("SDL3", lib_only)

    def test_the_runtime_library_plan_takes_the_same_flags_as_the_program_plan(self) -> None:
        tool = dataclasses.replace(fake_tool(self.dir), sdl3=self.make_sdl3_kit())
        text, out = self.plan_text(tool, with_modules=False)   # make_runtime_lib plans with no module
        include = build_game.cmd_arg(build_game.rel(tool.sdl3 / "include", out))
        self.assertIn(f" -DOOT2_WITH_SDL3=1 -DOOT2_AUDIO_SDL3=1 -I {include}", self.flags(text, "rtflags"))
        self.assertIn(f" -DOOT2_WITH_SDL3=1 -DOOT2_AUDIO_SDL3=1 -I {include}", self.flags(text, "cxxflags"))
        self.assertIn("build obj/runtime/gpu_vk_volk.o: cc_rt ", text)

    def test_both_kits_link_dav1d_first_then_sdl3_then_the_system_libraries(self) -> None:
        tool = dataclasses.replace(fake_tool(self.dir), sdl3=self.make_sdl3_kit(), dav1d=self.make_kit())
        text, out = self.plan_text(tool)
        libs = shlex.split(self.flags(text, "libs")[len("libs = "):], posix=False)
        dav1d = libs.index(build_game.cmd_arg(build_game.rel(tool.dav1d / "lib" / "libdav1d.a", out)))
        sdl3 = libs.index(build_game.cmd_arg(build_game.rel(tool.sdl3 / "lib" / "libSDL3.a", out)))
        self.assertLess(dav1d, sdl3)
        self.assertEqual(libs[sdl3 + 1:], list(fetch_sdl3.LINK_LIBS))
        self.assertEqual(libs.count("-pthread"), 0 if WIN else 1)

    def test_an_sdl3_kit_is_found_by_option_or_variable_and_refused_by_name_when_it_is_not_the_layout(self) -> None:
        kit = self.make_sdl3_kit()
        self.assertEqual(build_game.find_sdl3(kit), kit)
        with mock.patch.dict(os.environ, {"SDL3_DIR": str(kit)}):
            self.assertEqual(build_game.find_sdl3(None), kit)   # else the SDL3_DIR variable
        with mock.patch.dict(os.environ):
            os.environ.pop("SDL3_DIR", None)
            self.assertIsNone(build_game.find_sdl3(None))   # a build with no window
        with self.assertRaisesRegex(build_game.BuildError, r"SDL3 kit .*no kit here.* cannot be used: .* is not a folder"):
            build_game.find_sdl3(self.dir / "no kit here")
        (kit / "lib" / "libSDL3.a").write_bytes(b"another library")
        with self.assertRaisesRegex(build_game.BuildError, r"(?s)cannot be used: lib/libSDL3\.a in .* differs.*tools/fetch_sdl3\.py"):
            build_game.find_sdl3(kit)   # a library that is not the one laid out is no kit
        (kit / "lib" / "libSDL3.a").write_bytes(b"!<arch>\n")
        (kit / "include" / "SDL3" / "SDL_vulkan.h").unlink()
        with self.assertRaisesRegex(build_game.BuildError, r"SDL_vulkan\.h is missing"):
            build_game.find_sdl3(kit)
        bare = self.dir / "bare"
        (bare / "include" / "SDL3").mkdir(parents=True)   # headers and a library, and no record: not made by the script
        (bare / "include" / "SDL3" / "SDL.h").write_text("", encoding="utf-8")
        (bare / "lib").mkdir()
        (bare / "lib" / "libSDL3.a").write_bytes(b"!<arch>\n")
        with mock.patch.dict(os.environ, {"SDL3_DIR": str(bare)}):
            with self.assertRaisesRegex(build_game.BuildError, r"fetch_sdl3\.json is missing or unreadable"):
                build_game.find_sdl3(None)

    def test_the_sdl3_inputs_of_the_link_key_are_the_library_and_the_two_headers(self) -> None:
        kit = self.make_sdl3_kit()
        self.assertEqual(build_game.sdl3_inputs(kit), [kit / "lib" / "libSDL3.a", kit / "include" / "SDL3" / "SDL.h",
                                                       kit / "include" / "SDL3" / "SDL_vulkan.h"])
        self.assertEqual(build_game.sdl3_inputs(None), ["no SDL3 kit"])

    def test_the_prepare_link_key_follows_the_kit_and_the_libraries_it_adds(self) -> None:
        from prepare.runner import PrepareError, fingerprint
        kit = self.make_sdl3_kit()
        ctx = argparse.Namespace(options=argparse.Namespace(sdl3_dir=kit))
        with mock.patch.dict(os.environ):
            os.environ.pop("SDL3_DIR", None)
            with_kit = steps.sdl3_key(ctx)
            self.assertEqual(with_kit[:3], build_game.sdl3_inputs(kit))
            without = steps.sdl3_key(argparse.Namespace(options=argparse.Namespace(sdl3_dir=None)))
            self.assertEqual(without, ["no SDL3 kit"])
            self.assertNotEqual(fingerprint(*with_kit), fingerprint(*without))
            self.assertEqual(fingerprint(*steps.sdl3_key(ctx)), fingerprint(*with_kit))   # the same kit, the same key
            with mock.patch.object(fetch_sdl3, "LINK_LIBS", (*fetch_sdl3.LINK_LIBS, "-lextra")),                     mock.patch.object(build_game, "find_sdl3", return_value=kit):   # (a real kit would refuse the new list)
                self.assertNotEqual(fingerprint(*steps.sdl3_key(ctx)), fingerprint(*with_kit))
            other = self.make_sdl3_kit("other kit")   # another SDL.h, recorded as laid out: a different kit
            (other / "include" / "SDL3" / "SDL.h").write_text("// another header\n", encoding="utf-8")
            record = json.loads((other / fetch_sdl3.RECORD).read_text(encoding="utf-8"))
            record["files"] = fetch_sdl3.tree_hashes(other)
            (other / fetch_sdl3.RECORD).write_text(json.dumps(record), encoding="utf-8")
            other_key = steps.sdl3_key(argparse.Namespace(options=argparse.Namespace(sdl3_dir=other)))
            self.assertNotEqual(fingerprint(*other_key), fingerprint(*with_kit))
        bad = argparse.Namespace(options=argparse.Namespace(sdl3_dir=self.dir / "not a kit"))
        with self.assertRaisesRegex(PrepareError, "cannot be used"):
            steps.sdl3_key(bad)

    def test_the_prepare_check_says_whether_there_is_an_sdl3_kit_and_refuses_a_wrong_one(self) -> None:
        from prepare import check
        from prepare.runner import PrepareError
        kit = self.make_sdl3_kit()
        with mock.patch.dict(os.environ):
            os.environ.pop("SDL3_DIR", None)
            self.assertEqual(check.sdl3_note(argparse.Namespace(options=argparse.Namespace(sdl3_dir=kit))),
                             f"SDL3 kit {kit}")
            none = check.sdl3_note(argparse.Namespace(options=argparse.Namespace()))   # no option at all: no kit
            self.assertEqual(none, "no SDL3 kit (the game shows no window: run tools/fetch_sdl3.py and set SDL3_DIR)")
            os.environ["SDL3_DIR"] = str(kit)
            self.assertEqual(check.sdl3_note(argparse.Namespace(options=argparse.Namespace(sdl3_dir=None))),
                             f"SDL3 kit {kit}")   # else the variable
            with self.assertRaisesRegex(PrepareError, r"SDL3 kit .*gone.* cannot be used"):
                check.sdl3_note(argparse.Namespace(options=argparse.Namespace(sdl3_dir=self.dir / "gone")))

    def test_the_toolchain_carries_the_sdl3_kit_from_the_option_or_the_variable(self) -> None:
        from prepare.runner import PrepareError
        exe = sys.executable
        sdk = self.dir / "SDK"
        for rel_name in ("Include/vulkan/vulkan_core.h", "Include/Volk/volk.c", "Include/Volk/volk.h"):
            (sdk / rel_name).parent.mkdir(parents=True, exist_ok=True)
            (sdk / rel_name).write_text("", encoding="utf-8")
        kit = self.make_sdl3_kit()
        with mock.patch.dict(os.environ, {"OOT2_GLSLANG": exe, "OOT2_NINJA": exe}):
            os.environ.pop("SDL3_DIR", None)
            self.assertEqual(build_game.find_toolchain(exe, exe, exe, vulkan_sdk=sdk, sdl3_dir=kit).sdl3, kit)
            self.assertIsNone(build_game.find_toolchain(exe, exe, exe, vulkan_sdk=sdk).sdl3)
            os.environ["SDL3_DIR"] = str(kit)
            self.assertEqual(build_game.find_toolchain(exe, exe, exe, vulkan_sdk=sdk).sdl3, kit)
            os.environ.pop("SDL3_DIR")
            with self.assertRaisesRegex(build_game.BuildError, "cannot be used"):
                build_game.find_toolchain(exe, exe, exe, vulkan_sdk=sdk, sdl3_dir=self.dir / "gone")
            ctx = argparse.Namespace(options=argparse.Namespace(cc=exe, vulkan_sdk=sdk, dav1d_dir=None, sdl3_dir=kit))
            self.assertEqual(steps.toolchain(ctx).sdl3, kit)
            ctx.options.sdl3_dir = self.dir / "gone"
            with self.assertRaisesRegex(PrepareError, "cannot be used"):
                steps.toolchain(ctx)

    def test_check_step_needs_the_vulkan_headers_and_volk(self) -> None:
        from prepare import check
        from prepare.runner import PrepareError
        if not shutil.which(os.environ.get("OOT2_CC", "clang")) or not shutil.which(
                os.environ.get("OOT2_NINJA", "ninja")):
            self.skipTest("no clang or Ninja on PATH")
        ctx = argparse.Namespace(options=argparse.Namespace(cc=None, vulkan_sdk=self.dir / "no SDK here"))
        with self.assertRaisesRegex(PrepareError, r"the Vulkan headers and Volk were not found in .*no SDK here"):
            check.build_tools(ctx)

    def test_check_step_needs_capstone(self) -> None:
        from prepare import check
        from prepare.runner import PrepareError
        with mock.patch.dict(sys.modules, {"capstone": None}):
            with self.assertRaisesRegex(PrepareError, r"capstone cannot be imported .*spin-wait search needs it"):
                check.lift_tools()

    def test_jobs_follow_free_memory(self) -> None:
        jobs, why = build_game.default_jobs(free=4 * build_game.GiB)
        self.assertEqual(jobs, min(os.cpu_count() or 4, 4))
        self.assertIn("4.0 GiB free memory allows 4", why)
        self.assertEqual(build_game.default_jobs(free=1)[0], 1)


class ProgramTest(unittest.TestCase):
    """A program built from synthetic modules, run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = toolchain_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-program-", ignore_cleanup_errors=True)
        top = Path(cls.tmp.name)
        cls.game = top / "game"
        write_exefs(cls.game / "exefs")
        lift(cls.game / "exefs", cls.game / "gen")
        cls.runtime = copy_runtime(top / "runtime")
        cls.build = build_game.Build(cls.game / "build", cls.tool, gen=cls.game / "gen", exefs=cls.game / "exefs",
                                     exe=cls.game / build_game.EXE_NAME, runtime=cls.runtime, jobs=4)
        cls.compiled = build_game.compile_generated(cls.build)
        cls.linked = build_game.link_program(cls.build)
        cls.exe = cls.game / build_game.EXE_NAME

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def copy_game(self, name: str) -> Path:
        dst = Path(self.tmp.name) / name
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir()
        shutil.copytree(self.game / "exefs", dst / "exefs")
        shutil.copy2(self.exe, dst / self.exe.name)
        return dst

    def assert_boots_the_sdk(self, done: subprocess.CompletedProcess, folder: Path) -> None:
        self.assertEqual(done.returncode, EXIT_GUEST_STOP, done.stdout + done.stderr)
        self.assertIn(BOOT_STOP, done.stderr)
        self.assertIn("(host <rtld module_finis>)", done.stderr)
        self.assertEqual(len([line for line in done.stderr.splitlines() if line.startswith("FATAL")]), 1, done.stderr)
        self.assertRegex(done.stderr, r"boot: nn::init::Start at 0x[0-9a-f]+ \(sdk\+0x[0-9a-f]+\), main thread 0x")
        self.assertIn("boot: rtld callback 1: the modules are linked", done.stderr)
        self.assertIn("boot: sdk constructors (init_array only, 0 init_array entries)", done.stderr)
        census = (folder / "census.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(census[0].split("\t"), ["kind", "key", "status", "calls", "first_at", "why"])
        rows = {tuple(line.split("\t")[:3]) for line in census[1:]}
        for name in ("<rtld>", "<rtld loader_ready>", "<rtld module_inits>", "<rtld module_finis>"):
            self.assertIn(("boot", name, "implemented"), rows)
        self.assertIn(("stop", "exit 72", "named-fatal"), rows)

    def test_program_loads_the_modules_builds_its_table_and_boots_the_sdk(self) -> None:
        done = run_program(self.exe, cwd=Path(self.tmp.name))
        self.assert_boots_the_sdk(done, self.game)
        self.assertIn(f"dispatch table over 2 lifted modules ({self.compiled['functions']} functions", done.stderr)
        self.assertIn("2 modules, 1 other files to check", done.stderr)
        # main() says where it is, so that a host pc in a fault line can be read against the program's symbols
        self.assertRegex(done.stderr, r"(?m)^\[rt\] oot2: main\(\) is at (?:0x)?[0-9A-Fa-f]+ \(a host pc in a log line is read "
                                      r"against this and the program's symbols\)$")
        self.assertEqual(self.compiled["objects"], lifted_objects(self.game / "build"))

    def test_the_system_timer_is_requested_at_start_unless_the_setting_is_off(self) -> None:
        # The timer resolution hold: main calls rt_timer_hold_start right after the settings load, before the first thread and before
        # main()'s own log line. The request itself is read back by tests/test_runtime_timer.py; here the real main is
        # shown to make the call, once, and to honour the setting.
        done = run_program(self.exe, cwd=Path(self.tmp.name))
        self.assert_boots_the_sdk(done, self.game)
        timer = [line for line in done.stderr.splitlines() if line.startswith("[rt] timer: ")]
        self.assertEqual(len(timer), 1, done.stderr)
        self.assertLess(done.stderr.index("[rt] timer: "), done.stderr.index("main() is at"))
        if os.name == "nt":
            self.assertRegex(timer[0], r"^\[rt\] timer: holding 0\.500 ms \(the system granted \d+\.\d{3} ms, "
                                       r"\d+\.\d{3} ms in force now; timer_resolution = 1, the default\)$")
        else:
            self.assertIn("this platform has no system timer resolution to hold", timer[0])
        for args, env, source in ((("--set", "timer_resolution=0"), None, "--set"),
                                  ((), {"OOT2_TIMER_RESOLUTION": "0"}, "OOT2_TIMER_RESOLUTION")):
            with self.subTest(source=source):
                done = run_program(self.exe, *args, env=env, cwd=Path(self.tmp.name))
                self.assert_boots_the_sdk(done, self.game)
                self.assertIn(f"[rt] timer: the system's resolution is left as it is, not requested "
                              f"(timer_resolution = 0, {source})", done.stderr)
                self.assertNotIn("timer: holding", done.stderr)
                self.assertIn(f"not at their default: timer_resolution=0 ({source})", done.stderr)
                self.assertLess(done.stderr.index("[rt] timer: "), done.stderr.index("main() is at"))

    def test_the_library_is_linked_whole(self) -> None:
        done = run_program(self.exe)
        self.assertIn("replaced _Unwind_RaiseException at 0x", done.stderr)
        overrides = int(done.stderr.split(" host overrides linked")[0].rsplit(" ", 1)[1])
        self.assertGreaterEqual(overrides, 7)   # runtime/core/jmp.c's seven

    def test_a_moved_folder_still_starts(self) -> None:
        moved = self.copy_game("moved")
        self.assert_boots_the_sdk(run_program(moved / self.exe.name, cwd=Path(self.tmp.name)), moved)

    def test_game_option_names_the_folder(self) -> None:
        alone = self.copy_game("alone")
        shutil.rmtree(alone / "exefs")
        done = run_program(alone / self.exe.name)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stderr)
        self.assertIn("main.npdm: ", done.stderr)
        self.assertIn("is missing or cannot be read: run oot2 prepare again", done.stderr)
        self.assert_boots_the_sdk(run_program(alone / self.exe.name, "--game", str(self.game)), self.game)

    def test_bad_command_line_exits_2(self) -> None:
        done = run_program(self.exe, "--no-such-option")
        self.assertEqual(done.returncode, 2)
        self.assertIn("usage: oot2 [--game DIR]", done.stderr)

    def test_the_back_end_is_chosen_by_gfx_then_oot2_gfx_then_the_default(self) -> None:
        done = run_program(self.exe)
        self.assert_boots_the_sdk(done, self.game)
        self.assertIn("[rt] gfx: the null back end (the default)", done.stderr)
        done = run_program(self.exe, "--gfx", "null", env={"OOT2_GFX": "vulkan"})
        self.assert_boots_the_sdk(done, self.game)
        self.assertIn("[rt] gfx: the null back end (--gfx)", done.stderr)
        done = run_program(self.exe, env={"OOT2_GFX": "null"})
        self.assertIn("[rt] gfx: the null back end (OOT2_GFX)", done.stderr)

    def test_the_vulkan_back_end_opens_its_device_or_stops_the_start_by_name(self) -> None:
        # Without a device that has every item (continuous integration), the start stops by name before anything else;
        # with one, the device opens, the game folder's shader packs are opened (a folder without them stops by name)
        # and the program boots the SDK as on the null back end.
        sys.path.insert(0, str(ROOT / "tools" / "shaders"))
        import cache_pack
        no_vulkan = ("there is no Vulkan loader", "there is no Vulkan device", "no Vulkan device has every item",
                     "the Vulkan loader is version")
        packed = self.copy_game("vulkan-packs")
        (packed / "shaders").mkdir()
        cache_pack.write_pack(packed / "shaders" / "shader-base.bin", [], translator="0" * 64)
        for args, env, source in ((("--gfx", "vulkan"), None, "--gfx"), ((), {"OOT2_GFX": "vulkan"}, "OOT2_GFX")):
            for game in (self.game, packed):
                with self.subTest(source=source, packs=game == packed):
                    full = {"VK_LOADER_LAYERS_DISABLE": "~implicit~", **(env or {})}
                    done = run_program(game / self.exe.name, *args, env=full)
                    self.assertIn(f"[rt] gfx: the vulkan back end ({source})", done.stderr)
                    fatal = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
                    if fatal and any(why in fatal[0] for why in no_vulkan):
                        self.assertEqual(done.returncode, EXIT_FATAL, done.stderr)
                        self.assertEqual(len(fatal), 1, done.stderr)
                        continue
                    self.assertRegex(done.stderr, r"gfx: vulkan: .* device made with the \d+ items' features")
                    if game == packed:
                        self.assert_boots_the_sdk(done, packed)
                    else:
                        self.assertEqual(done.returncode, EXIT_FATAL, done.stderr)
                        self.assertEqual(len(fatal), 1, done.stderr)
                        self.assertTrue(fatal[0].startswith("FATAL: gfx: vulkan: the shader packs of the game folder "
                                                            "cannot be opened (run oot2 prepare): "), fatal[0])

    def test_an_unknown_back_end_is_refused(self) -> None:
        done = run_program(self.exe, "--gfx", "metal")
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("oot2: --gfx: no back end has that name", done.stderr)
        # OOT2_GFX is the setting gfx's variable: a name it does not take stops the start before anything runs
        done = run_program(self.exe, env={"OOT2_GFX": "metal"})
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("oot2: OOT2_GFX=metal: gfx takes null or vulkan", done.stderr)
        self.assertNotIn("FATAL", done.stderr)

    def test_settings_come_from_set_then_the_environment_then_the_game_folders_file(self) -> None:
        game = self.copy_game("settings")
        ini = game / "oot2.ini"
        ini.write_text("; written by the test\n[pacing]\nhold_ticks = 4\n[debug]\ncensus_frames = 9\n", encoding="utf-8")
        done = run_program(game / self.exe.name, cwd=Path(self.tmp.name))
        self.assert_boots_the_sdk(done, game)
        self.assertRegex(done.stderr, r"\[rt\] oot2: settings file .*oot2\.ini; not at their default: hold_ticks=4 "
                                      r"\(.*oot2\.ini\), census_frames=9 \(.*oot2\.ini\)\n")
        self.assertIn("nvn: experiment E2: every present is held for 4 ticks, whatever the game asks for (", done.stderr)
        done = run_program(game / self.exe.name, "--set", "hold_ticks=6", env={"OOT2_HOLD_TICKS": "5"})
        self.assertIn("not at their default: hold_ticks=6 (--set), census_frames=9 (", done.stderr)
        done = run_program(game / self.exe.name, env={"OOT2_HOLD_TICKS": "5"})
        self.assertIn("not at their default: hold_ticks=5 (OOT2_HOLD_TICKS), census_frames=9 (", done.stderr)
        # --config, then OOT2_CONFIG, name another file; the game folder's is not read then
        other = Path(self.tmp.name) / "other.ini"
        other.write_text("[graphics]\ngfx = null\n", encoding="utf-8")
        for args, env in ((("--config", str(other)), None), ((), {"OOT2_CONFIG": str(other)})):
            with self.subTest(args=args, env=env):
                done = run_program(game / self.exe.name, *args, env=env)
                self.assert_boots_the_sdk(done, game)
                self.assertIn("other.ini; every setting at its default\n", done.stderr)
        # the file chooses the back end too, and gfx_select names it: the settings are read before gfx_select
        ini.write_text("[graphics]\ngfx = null\n", encoding="utf-8")
        done = run_program(game / self.exe.name)
        self.assert_boots_the_sdk(done, game)
        self.assertRegex(done.stderr, r"\[rt\] gfx: the null back end \(.*oot2\.ini\)")

    def test_a_setting_the_program_does_not_take_exits_2_by_name(self) -> None:
        for args, env, line in ((("--set", "nosuch=1"), None, "oot2: --set nosuch=1: no setting is named nosuch"),
                                (("--set", "hold_ticks=-1"), None,
                                 "oot2: --set hold_ticks=-1: hold_ticks takes an integer from 0 to 999999"),
                                ((), {"OOT2_CENSUS": "yes"}, "oot2: OOT2_CENSUS=yes: census takes 0 or 1"),
                                (("--set",), None, "oot2: unknown argument")):
            with self.subTest(args=args, env=env):
                done = run_program(self.exe, *args, env=env)
                self.assertEqual(done.returncode, 2, done.stderr)
                self.assertIn(line, done.stderr)
                self.assertIn("usage: oot2 [--game DIR]", done.stderr)
                self.assertNotIn("[rt]", done.stderr)   # refused before the first log line

    def test_the_program_and_the_check_probe_print_the_same_vulkan_items(self) -> None:
        from prepare import check
        done = run_program(self.exe, "--vulkan-items")
        self.assertEqual(done.returncode, 0, done.stderr)
        items = [line.split("\t") for line in done.stdout.splitlines()]
        probe = check.run_probe(check.build_probe(self.tool.cc, Path(self.tmp.name) / "probe", self.tool.vulkan))
        self.assertEqual(items, probe["vulkan_items"])
        self.assertGreaterEqual(len(items), 19)   # architecture section 1's items; the back end's tickets add more
        self.assertEqual(len({ident for ident, _ in items}), len(items))
        self.assertEqual(len({what for _, what in items}), len(items))
        for device in (probe["vulkan"] or {}).get("devices", []):
            self.assertEqual(list(device["items"]), [ident for ident, _ in items])

    def test_a_changed_module_stops_the_start(self) -> None:
        g = self.copy_game("changed-module")
        flip_byte(g / "exefs" / "main", 0x120)
        done = run_program(g / self.exe.name)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stderr)
        self.assertIn("the build expects", done.stderr)
        self.assertIn("main has SHA-256", done.stderr)

    def test_a_changed_npdm_stops_the_start(self) -> None:
        g = self.copy_game("changed-npdm")
        flip_byte(g / "exefs" / "main.npdm", 0x30)
        done = run_program(g / self.exe.name)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stderr)
        self.assertIn("main.npdm: ", done.stderr)
        self.assertIn("has SHA-256", done.stderr)
        (g / "exefs" / "main.npdm").write_bytes(npdm() + b"\0")
        done = run_program(g / self.exe.name)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stderr)
        self.assertIn(f"holds {len(npdm()) + 1} bytes, the build expects {len(npdm())}", done.stderr)

    def in_place_game(self) -> tuple[Path, Path]:
        """A copy of the game whose program reads a stand-in for the player's dump in place, outside its folder, as a
        RomFS in place does: (program, dump). Linked once for the class; the shared build is then linked back."""
        cls = type(self)
        if not hasattr(cls, "in_place"):
            g = self.copy_game("in-place")
            dump = Path(self.tmp.name) / "dumps" / "game.nup"
            dump.parent.mkdir(exist_ok=True)
            # 2 MiB and a few bytes, so the 16 sampled ranges are apart (build_game.sample_ranges)
            dump.write_bytes(bytes((i * 7 + (i >> 11)) & 0xFF for i in range(2 * 1024 * 1024 + 13)))
            b = dataclasses.replace(self.build, exe=g / self.exe.name,
                                    files=[build_game.GameFile("RomFS image", dump, False, sampled=True)])
            try:
                build_game.link_program(b)
            finally:
                build_game.link_program(self.build)
            cls.in_place = (g / self.exe.name, dump)
        return cls.in_place

    def test_a_dump_read_in_place_that_moved_stops_the_start(self) -> None:
        exe, dump = self.in_place_game()
        self.assert_boots_the_sdk(run_program(exe), exe.parent)   # in place, unchanged: the C and Python samples agree
        moved = dump.rename(dump.with_name("moved.nup"))
        try:
            done = run_program(exe)
        finally:
            moved.rename(dump)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stderr)
        self.assertIn(f"RomFS image: {dump.resolve().as_posix()} is missing or cannot be read (the RomFS is read from "
                      "this dump in place): run oot2 prepare again, with --dump naming where the dump is now",
                      done.stderr)

    def test_a_dump_read_in_place_that_changed_stops_the_start(self) -> None:
        exe, dump = self.in_place_game()
        original = dump.read_bytes()
        middle = build_game.sample_ranges(len(original))[7][0] + 100
        other = bytearray(original)
        other[0x18:0x38] = bytes(32)                           # another dump of the same size: on a .nup, its
                                                               # superblock names another table of contents here
        flipped = bytearray(original)
        flipped[middle] ^= 0x40                                # one byte, inside a sampled range
        try:
            for why, data in (("another file of the same size", other), ("one byte changed in place", flipped)):
                with self.subTest(why=why):
                    st = dump.stat()
                    dump.write_bytes(bytes(data))
                    os.utime(dump, ns=(st.st_atime_ns, st.st_mtime_ns))
                    done = run_program(exe)
                    self.assertEqual(done.returncode, EXIT_FATAL, done.stderr)
                    self.assertIn(f"RomFS image: {dump.resolve().as_posix()} changed since oot2 prepare built this "
                                  "program (the RomFS is read from this dump in place): the SHA-256 of its 16 sampled "
                                  "ranges is", done.stderr)
                    self.assertIn("run oot2 prepare again", done.stderr)
        finally:
            dump.write_bytes(original)
        self.assert_boots_the_sdk(run_program(exe), exe.parent)

    @unittest.skipUnless(WIN, "a PE header")
    def test_program_is_linked_without_high_entropy_aslr(self) -> None:
        data = self.exe.read_bytes()[:4096]
        pe = int.from_bytes(data[0x3C:0x40], "little")
        characteristics = int.from_bytes(data[pe + 24 + 70:pe + 24 + 72], "little")
        self.assertEqual(characteristics & 0x20, 0, "IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA is set")

    def test_a_runtime_edit_compiles_that_file_and_relinks(self) -> None:
        objs = sorted((self.game / "build" / "obj").glob("main/*.o")) + \
            sorted((self.game / "build" / "obj").glob("sdk/*.o"))
        before = {o: o.stat().st_mtime_ns for o in objs}
        edited = self.runtime / "program" / "main.c"
        edited.write_text(edited.read_text(encoding="utf-8") + "\n// edited by a test\n", encoding="utf-8")
        lines: list[str] = []
        build_game.link_program(self.build, lines.append)
        compiled = [line for line in lines if build_game.NINJA_LINE.match(line) and "] CC " in line]
        self.assertEqual(len(compiled), 1, lines)
        self.assertIn("program/main.c", compiled[0])
        self.assertTrue(any("] AR " in line for line in lines), lines)
        self.assertTrue(any("] LINK " in line for line in lines), lines)
        self.assertEqual(before, {o: o.stat().st_mtime_ns for o in objs})
        self.assert_boots_the_sdk(run_program(self.exe), self.game)

    def test_an_object_changed_in_place_is_compiled_again_before_the_link(self) -> None:
        victim = sorted((self.game / "build" / "obj" / "main").glob("*.o"))[0]
        flip_byte(victim, victim.stat().st_size // 2, keep_time=True)
        dry = subprocess.run([self.tool.ninja, "-n", "gen_objects"], cwd=self.game / "build", capture_output=True,
                             text=True)
        self.assertIn("no work to do", dry.stdout)   # Ninja alone would link the changed object
        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            r = build_game.link_program(self.build)
        self.assertEqual(r["manifest"]["recompiled"], 1)
        self.assertIn(f"object {victim.resolve()} does not match its SHA-256", out.getvalue())
        self.assert_boots_the_sdk(run_program(self.exe), self.game)

    def test_a_runtime_file_removed_leaves_the_library(self) -> None:
        extra = self.runtime / "program" / "extra_for_test.c"
        extra.write_text('#include "oot2/runtime.h"\nstatic void extra(Cpu *c) { (void)c; }\n'
                         'HOST_OVERRIDE("oot2_test_extra", extra);\n', encoding="utf-8")
        try:
            build_game.link_program(self.build)
            with_extra = run_program(self.exe).stderr
        finally:
            extra.unlink()
        build_game.link_program(self.build)
        without = run_program(self.exe).stderr

        def count(text: str) -> int:
            return int(text.split(" host overrides linked")[0].rsplit(" ", 1)[1])
        self.assertEqual(count(with_extra), count(without) + 1)

    def test_a_cpp_object_is_in_the_library_the_manifest_and_the_program(self) -> None:
        # The C++ build check: the stand-in of runtime/ui goes through the C++ rule, is listed and hashed in the object
        # manifest like any other object, and is linked whole into a program that still starts and stops as before.
        build = self.game / "build"
        listed = (build / "obj" / build_game.RUNTIME_LIST).read_text(encoding="utf-8").split()
        self.assertIn("obj/runtime/ui_ui_standin.o", listed)
        self.assertIn("runtime/ui_ui_standin.o", build_game.read_manifest(build / "obj" / build_game.MANIFEST))
        self.assertIn(b"ui_standin_list", (build / build_game.LIB_NAME).read_bytes())
        self.assert_boots_the_sdk(run_program(self.exe), self.game)

    def test_a_cpp_object_changed_in_place_is_compiled_again_before_the_link(self) -> None:
        victim = self.game / "build" / "obj" / "runtime" / "ui_ui_standin.o"
        flip_byte(victim, victim.stat().st_size // 2, keep_time=True)
        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            r = build_game.link_program(self.build)
        self.assertEqual(r["manifest"]["recompiled"], 1)
        self.assertIn(f"object {victim.resolve()} does not match its SHA-256", out.getvalue())
        self.assert_boots_the_sdk(run_program(self.exe), self.game)

    @unittest.skipUnless(WIN, "a PE import table")
    def test_program_imports_no_dll_of_the_toolchain(self) -> None:
        # The C++ build check: libc++ and libunwind are linked in, so no libc++.dll, libunwind.dll or libwinpthread-1.dll has
        # to sit beside oot2.exe.
        imports = [name.lower() for name in pe_imports(self.exe)]
        self.assertIn("kernel32.dll", imports)
        self.assertEqual([name for name in imports if name.startswith("lib")], [])


class RuntimeLibTest(unittest.TestCase):
    """--make-runtime-lib, as a release would prebuild the library: the Vulkan back end's module, Volk and the
    runtime's own shaders are in it."""

    def test_make_runtime_lib_holds_the_back_end_volk_and_the_shaders(self) -> None:
        toolchain_or_skip()
        with tempfile.TemporaryDirectory(prefix="oot2-runtime-lib-") as tmp:
            out = io.StringIO()
            # a build without Dear ImGui, whatever folder this machine has fetched (the next test builds with one)
            with mock.patch("sys.stdout", out), mock.patch.dict(os.environ), \
                    mock.patch.object(build_game.fetch_imgui, "DEFAULT_DEST", Path(tmp) / "no-dear-imgui"):
                os.environ.pop(build_game.IMGUI_ENV, None)
                code = build_game.main(["--make-runtime-lib", "--out", tmp, "--jobs", "4"])
            self.assertEqual(code, 0, out.getvalue())
            self.assertIn("no Dear ImGui", out.getvalue())
            listed = (Path(tmp) / "obj" / build_game.RUNTIME_LIST).read_text(encoding="utf-8").split()
            for name in ("gpu_gpu.o", "gpu_vk_volk.o", "spirv_gpu_shaders.o", "ui_ui_standin.o"):
                self.assertIn(f"obj/runtime/{name}", listed)
            lib = (Path(tmp) / build_game.LIB_NAME).read_bytes()
            for symbol in (b"volkInitialize", b"gfx_select", b"gpu_shaders", b"ui_standin_list"):
                self.assertIn(symbol, lib)

    def test_make_runtime_lib_with_the_fetched_dear_imgui_holds_its_four_core_objects(self) -> None:
        toolchain_or_skip()
        from test_fetch_imgui import fetched_or_skip
        folder = fetched_or_skip()
        with tempfile.TemporaryDirectory(prefix="oot2-runtime-lib-imgui-") as tmp:
            out = io.StringIO()
            with mock.patch("sys.stdout", out):
                code = build_game.main(["--make-runtime-lib", "--out", tmp, "--jobs", "4", "--imgui", str(folder)])
            self.assertEqual(code, 0, out.getvalue())
            self.assertIn(f"Dear ImGui {build_game.fetch_imgui.VERSION} from {folder}", out.getvalue())
            listed = (Path(tmp) / "obj" / build_game.RUNTIME_LIST).read_text(encoding="utf-8").split()
            for name in build_game.fetch_imgui.CORE_SOURCES:
                self.assertIn(f"obj/runtime/imgui-{Path(name).stem}.o", listed)
            self.assertIn("obj/runtime/ui_ui_standin.o", listed)
            lib = (Path(tmp) / build_game.LIB_NAME).read_bytes()
            for symbol in (b"DebugCheckVersionAndDataLayout", b"ui_standin_list", b"volkInitialize"):
                self.assertIn(symbol, lib)


class MeasuredTest(unittest.TestCase):
    """run_measured reports the peak memory of what it ran: private bytes of the process, and its working set."""

    def child(self, megabytes: int) -> build_game.Measured:
        # bytearray(b"x") * n writes every page (bytearray(n) alone would leave them untouched).
        code = f"b = bytearray(b'x') * ({megabytes} << 20); print(len(b))"
        lines: list[str] = []
        m = build_game.run_measured([sys.executable, "-I", "-c", code], ROOT, lines.append)
        self.assertEqual((m.code, lines), (0, [str(megabytes << 20)]))
        return m

    def test_a_process_that_holds_a_buffer_has_it_in_both_peaks(self) -> None:
        m = self.child(200)
        self.assertGreaterEqual(m.peak_process, 200 << 20)
        self.assertGreaterEqual(m.peak_working_set, 200 << 20)
        if WIN:   # elsewhere the figure is the largest of every child the test process ever ran
            self.assertLess(m.peak_working_set, 400 << 20)

    @unittest.skipUnless(WIN, "the handle is a Windows one")
    def test_a_working_set_the_system_will_not_give_is_zero_and_stops_nothing(self) -> None:
        class NoHandle:   # a Popen whose handle is not there
            pass

        self.assertEqual(build_game._peak_working_set(NoHandle()), 0)   # type: ignore[arg-type]
        self.assertEqual(build_game._peak_working_set(types.SimpleNamespace(_handle=0)), 0)  # type: ignore[arg-type]

    @unittest.skipUnless(WIN, "elsewhere the child's resident set is the system's cumulative figure for all children")
    def test_the_peaks_are_those_of_the_run_and_not_of_the_machine(self) -> None:
        small = self.child(1)
        self.assertLess(small.peak_working_set, 150 << 20)
        self.assertLess(small.peak_process, 150 << 20)
        self.assertGreater(small.peak_working_set, 1 << 20)


class GivenModules(Step):
    """Stands for the modules step, which lift needs: the test writes exefs/ itself."""
    id = "modules"
    always = True


class StepsTest(unittest.TestCase):
    """Lift, compile and link through the runner: what runs, what is kept, and why."""

    @classmethod
    def setUpClass(cls) -> None:
        toolchain_or_skip()
        capstone_or_skip()
        # prepare lays out the fetched Dear ImGui when it is absent (git and network); a test names the folder, which
        # this checkout, or CI's fetch step, has made (tools/fetch_imgui.py).
        cls.imgui = build_game.find_imgui()
        if cls.imgui is None:
            if os.environ.get("OOT2_REQUIRE_IMGUI"):
                raise RuntimeError("OOT2_REQUIRE_IMGUI is set but no Dear ImGui folder was fetched (tools/fetch_imgui.py)")
            raise unittest.SkipTest("no fetched Dear ImGui (tools/fetch_imgui.py lays it out)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-steps-")
        cls.top = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def run_steps(self, dest: Path, upto: str = "link") -> tuple[str, str, dict]:
        out, err = io.StringIO(), io.StringIO()
        ctx = Context(dest, argparse.Namespace(cc=None, jobs=4, imgui=self.imgui), Reporter(out, err),
                      state=statefile.load(dest))
        chain = [GivenModules(), steps.Lift(), steps.Compile(), steps.Link()]
        result = run_steps(chain[:[s.id for s in chain].index(upto) + 1], ctx)
        return result, out.getvalue() + err.getvalue(), ctx.state

    def test_steps_run_keep_and_follow_what_changed(self) -> None:
        dest = self.top / "dest"
        write_exefs(dest / "exefs")
        runtime = copy_runtime(self.top / "runtime")
        with mock.patch.object(build_game, "RUNTIME", runtime):
            result, text, state = self.run_steps(dest)
            self.assertEqual(result, "ok", text)
            for step in ("lift", "compile", "link"):
                self.assertIn(f"{step}: started (not done yet)", text)
            self.assertEqual(state["steps"]["compile"]["objects"], lifted_objects(dest / "build"))
            self.assertGreater(state["steps"]["link"]["exe_bytes"], 0)
            done = run_program(dest / build_game.EXE_NAME)
            self.assertEqual(done.returncode, EXIT_GUEST_STOP, done.stderr)
            # prepare passed the fetched Dear ImGui to the link: its core is in the runtime library the program
            # links, and the program's own menu says so at start-up
            for name in fetch_imgui.CORE_SOURCES:
                self.assertTrue((dest / "build" / "obj" / "runtime" / f"imgui-{Path(name).stem}.o").is_file(), name)
            self.assertRegex(done.stderr, rf"oot2: menu: \d+ rows, 2 presets, Dear ImGui {re.escape(fetch_imgui.VERSION)}")

            result, text, _ = self.run_steps(dest)
            self.assertEqual(result, "ok", text)
            for step in ("lift", "compile", "link"):
                self.assertIn(f"{step}: kept", text)

            source = runtime / "core" / "clock.c"
            source.write_text(source.read_text(encoding="utf-8") + "\n// edited by a test\n", encoding="utf-8")
            result, text, _ = self.run_steps(dest)
            self.assertEqual(result, "ok", text)
            self.assertIn("lift: kept", text)
            self.assertIn("compile: kept", text)
            self.assertIn("link: started (its inputs changed)", text)

            header = runtime / "include" / "oot2" / "program.h"
            header.write_text(header.read_text(encoding="utf-8") + "\n// edited by a test\n", encoding="utf-8")
            result, text, _ = self.run_steps(dest)
            self.assertEqual(result, "ok", text)
            self.assertIn("lift: kept", text)
            self.assertIn("compile: started (its inputs changed)", text)
            self.assertIn("link: started (its inputs changed)", text)

    def test_lift_gates_the_spin_waits_it_finds(self) -> None:
        dest = self.top / "spin"
        write_exefs(dest / "exefs")
        (dest / "exefs" / "main").write_bytes(module([POLL, [ADD_X0_1, RET]], ["poll", "f1"]))
        result, text, state = self.run_steps(dest, upto="lift")
        self.assertEqual(result, "ok", text)
        table = json.loads((dest / "gen" / steps.SPIN_TABLE).read_text(encoding="utf-8"))
        self.assertEqual(len(table["main"]), 1, table)
        self.assertEqual(table["sdk"], [])
        self.assertEqual((state["steps"]["lift"]["spin_waits"], state["steps"]["lift"]["spin_gates"]), (1, 1))
        self.assertIn("lift: 1 spin waits found", text)
        chunks = "".join(p.read_text(encoding="utf-8") for p in sorted((dest / "gen" / "main").glob("main_*.c")))
        self.assertRegex(chunks, r"rt_spin_gate\(c, 0x[0-9a-f]+ull\); goto L_[0-9a-f]+;")

    def test_lift_without_modules_says_where_they_go(self) -> None:
        dest = self.top / "empty"
        dest.mkdir()
        result, text, _ = self.run_steps(dest)
        self.assertEqual(result, "failed")
        self.assertIn("the plain modules are not in", text)


if __name__ == "__main__":
    unittest.main()
