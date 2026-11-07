# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of runtime/ui, the runtime's one C++ directory (runtime/include/oot2/ui.h, the C++ build check): a stand-in that uses
the C++ standard library behind a C header, built and linked the way the program is.

tools/build_game.py builds a runtime folder holding the headers and runtime/ui into the runtime library, with the Ninja
rules and flags the program's own build uses. tests/runtime/ui_test.c is compiled as C, linked by the C++ driver with
that library, and run once per case. Checked: C reaches the stand-in through the header and gets its answers; the C++
flags reached the compiler (C++17, no exceptions, no run-time type information); the stand-in's object names things the
C++ standard library defines, so that linking it proves the library is there and not only named (a stand-in that never
touches the library is told apart by the same check); and on Windows the program imports no DLL of the toolchain.

Needs clang and its C++ driver (OOT2_CC and OOT2_CXX, else PATH), Ninja, an archiver, and a Vulkan SDK with the headers,
Volk and glslang (tools/build_game.py, find_toolchain). Without them the tests skip, unless OOT2_REQUIRE_CLANG is set,
as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_game  # noqa: E402
import fetch_imgui  # noqa: E402
from test_build_game import pe_imports, toolchain_or_skip  # noqa: E402
from test_fetch_imgui import fetched_or_skip  # noqa: E402

RUNTIME = ROOT / "runtime"
TEST_SOURCE = ROOT / "tests" / "runtime" / "ui_test.c"
WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
CASES = ("list", "sorted", "cut", "empty", "many")
CXX17 = "201703"   # __cplusplus under -std=c++17 (build_game.CXX_FLAGS)


def nm_of(tool: build_game.Toolchain) -> str:
    """llvm-nm beside the archiver, else on PATH, else nm."""
    for d in dict.fromkeys((Path(tool.ar).parent, Path(tool.ar).resolve().parent)):
        if (d / f"llvm-nm{EXE}").is_file():
            return str(d / f"llvm-nm{EXE}")
    found = shutil.which("llvm-nm") or shutil.which("nm")
    if not found:
        raise RuntimeError("no llvm-nm or nm was found beside the archiver or on PATH")
    return found


def cxx_names(nm: str, obj: Path) -> list[str]:
    """The undefined C++ (mangled, so starting with _Z) names of an object. runtime/ui's C++ is all extern "C" and
    defines whatever else it calls, so an undefined C++ name is something the C++ standard library defines."""
    done = subprocess.run([nm, "-u", str(obj)], capture_output=True, text=True, errors="replace")
    if done.returncode:
        raise RuntimeError(f"{nm} -u {obj} failed:\n{done.stdout}{done.stderr}")
    return [words[-1] for words in (line.split() for line in done.stdout.splitlines()) if words and
            words[-1].startswith("_Z")]


def run(cmd: list[str], what: str) -> None:
    done = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if done.returncode:
        raise RuntimeError(f"{what} failed:\n{' '.join(cmd)}\n{done.stdout}{done.stderr}")


class UiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = toolchain_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-ui-", ignore_cleanup_errors=True)
        top = Path(cls.tmp.name)
        # The runtime's headers and the stand-in of runtime/ui only (the menu has its own tests, test_runtime_menu.py, with
        # the settings store it reads): the library of this folder is the stand-in and nothing else.
        cls.runtime = top / "runtime"
        shutil.copytree(RUNTIME / "include", cls.runtime / "include")
        shutil.copytree(RUNTIME / "ui", cls.runtime / "ui", ignore=shutil.ignore_patterns("__pycache__", "menu*"))
        cls.build = build_game.Build(top / "build", cls.tool, runtime=cls.runtime, jobs=2)
        build_game.make_runtime_lib(cls.build)
        cls.nm = nm_of(cls.tool)
        obj = top / "ui_test.o"
        run([cls.tool.cc, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(cls.runtime / "include"),
             "-c", str(TEST_SOURCE), "-o", str(obj)], "compiling ui_test.c")
        cls.exe = top / f"ui_test{EXE}"
        run([cls.tool.cxx, *build_game.LINK_FLAGS, str(obj), str(cls.build.lib), *build_game.LINK_LIBS,
             "-o", str(cls.exe)], "linking ui_test")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exe), *args], capture_output=True, text=True, timeout=60)

    def test_c_reaches_the_stand_in_through_the_header(self) -> None:
        for case in CASES:
            with self.subTest(case=case):
                done = self.run_case(case)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_the_cxx_flags_reach_the_compiler(self) -> None:
        done = self.run_case("flags", CXX17, "0", "0")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        # the case can tell: a wrong expectation fails
        for wrong in (("201402", "0", "0"), (CXX17, "1", "0"), (CXX17, "0", "1")):
            with self.subTest(wrong=wrong):
                self.assertEqual(self.run_case("flags", *wrong).returncode, 1)

    def test_a_build_without_dear_imgui_says_so(self) -> None:
        done = self.run_case("imgui", "none")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(self.run_case("imgui", fetch_imgui.VERSION).returncode, 1)   # the case can tell

    def test_the_stand_in_takes_its_names_from_the_cxx_library(self) -> None:
        names = cxx_names(self.nm, self.build.out / "obj" / "runtime" / "ui_ui_standin.o")
        self.assertTrue(names, "the stand-in's object names nothing the C++ standard library defines")
        self.assertTrue(any(n.startswith("_Znw") for n in names), f"no operator new among {names}")

    def test_a_stand_in_that_never_touches_the_library_is_told_apart(self) -> None:
        source = Path(self.tmp.name) / "plain.cpp"
        source.write_text('extern "C" int plain(int x) { return x + 1; }\n', encoding="utf-8")
        obj = Path(self.tmp.name) / "plain.o"
        run([self.tool.cxx, *build_game.CXX_FLAGS, "-c", str(source), "-o", str(obj)], "compiling plain.cpp")
        self.assertEqual(cxx_names(self.nm, obj), [])

    @unittest.skipUnless(WIN, "a PE import table")
    def test_the_program_imports_no_dll_of_the_toolchain(self) -> None:
        # libc++ and libunwind are linked in (-static), so none of libc++.dll, libunwind.dll or libwinpthread-1.dll
        # has to sit beside the program.
        imports = [name.lower() for name in pe_imports(self.exe)]
        self.assertIn("kernel32.dll", imports)
        self.assertEqual([name for name in imports if name.startswith("lib")], [])

    def test_the_toolchain_carries_what_the_static_link_needs(self) -> None:
        # The pieces an installer bundle takes from an llvm-mingw tree for C++ (the C++ build check): the driver, the static
        # standard library and unwinder it links, the headers a .cpp file includes. The sizes go to the log.
        root = Path(self.tool.cxx).resolve().parent.parent
        lib = root / "x86_64-w64-mingw32" / "lib"
        if not (lib / "libc++.a").is_file():
            self.skipTest(f"{root} is not an llvm-mingw tree")
        headers = root / "include" / "c++" / "v1"
        files = [p for p in headers.rglob("*") if p.is_file()]
        self.assertIn(headers / "vector", files)
        pieces = {"clang++": Path(self.tool.cxx).stat().st_size, "libc++.a": (lib / "libc++.a").stat().st_size,
                  "libunwind.a": (lib / "libunwind.a").stat().st_size,
                  f"include/c++/v1 ({len(files)} files)": sum(p.stat().st_size for p in files)}
        print("\nllvm-mingw C++ pieces:", ", ".join(f"{name} {size:,} bytes" for name, size in pieces.items()),
              flush=True)


def undefined_names(nm: str, obj: Path) -> list[str]:
    done = subprocess.run([nm, "-u", str(obj)], capture_output=True, text=True, errors="replace")
    if done.returncode:
        raise RuntimeError(f"{nm} -u {obj} failed:\n{done.stdout}{done.stderr}")
    return [words[-1] for words in (line.split() for line in done.stdout.splitlines()) if words]


class UiImguiTest(unittest.TestCase):
    """The stand-in built with the fetched Dear ImGui (tools/fetch_imgui.py, the ImGui fetch study): its four core sources
    are compiled into the runtime library by the plan's own rule, the stand-in includes the header and asks the core
    whether the two agree, and the library links, whole, into a program that runs. No window, no device, no drawing.

    Needs a fetched folder (OOT2_IMGUI, else fetched/imgui); without one the tests skip, unless OOT2_REQUIRE_IMGUI is
    set, as continuous integration sets it."""

    @classmethod
    def setUpClass(cls) -> None:
        tool = toolchain_or_skip()
        cls.imgui = fetched_or_skip()
        cls.tool = dataclasses.replace(tool, imgui=cls.imgui)
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-ui-imgui-", ignore_cleanup_errors=True)
        top = Path(cls.tmp.name)
        cls.runtime = top / "runtime"
        shutil.copytree(RUNTIME / "include", cls.runtime / "include")
        shutil.copytree(RUNTIME / "ui", cls.runtime / "ui", ignore=shutil.ignore_patterns("__pycache__", "menu*"))
        cls.build = build_game.Build(top / "build", cls.tool, runtime=cls.runtime, jobs=2)
        build_game.make_runtime_lib(cls.build)
        cls.nm = nm_of(cls.tool)
        obj = top / "ui_test.o"
        run([cls.tool.cc, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(cls.runtime / "include"),
             "-c", str(TEST_SOURCE), "-o", str(obj)], "compiling ui_test.c")
        cls.exe = top / f"ui_test{EXE}"
        run([cls.tool.cxx, *build_game.LINK_FLAGS, str(obj), str(cls.build.lib), *build_game.LINK_LIBS,
             "-o", str(cls.exe)], "linking ui_test")
        # The program links the library whole (HOST_OVERRIDE entries are found by section): every object of Dear
        # ImGui goes in, so a name one of them needs that another does not define stops the link.
        cls.whole = top / f"ui_test_whole{EXE}"
        run([cls.tool.cxx, *build_game.LINK_FLAGS, str(obj), "-Wl,--whole-archive", str(cls.build.lib),
             "-Wl,--no-whole-archive", *build_game.LINK_LIBS, "-o", str(cls.whole)], "linking ui_test whole")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def objects(self) -> list[Path]:
        return [self.build.out / "obj" / "runtime" / f"imgui-{Path(n).stem}.o" for n in fetch_imgui.CORE_SOURCES]

    def test_the_runtime_reports_the_pinned_version_and_its_core_agrees_with_the_headers(self) -> None:
        for exe in (self.exe, self.whole):
            with self.subTest(exe=exe.name):
                done = subprocess.run([str(exe), "imgui", fetch_imgui.VERSION], capture_output=True, text=True,
                                      timeout=60)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                for wrong in ("1.0.0", "none"):
                    done = subprocess.run([str(exe), "imgui", wrong], capture_output=True, text=True, timeout=60)
                    self.assertEqual(done.returncode, 1, wrong)

    def test_the_stand_in_cases_still_pass_beside_dear_imgui(self) -> None:
        for case in CASES:
            with self.subTest(case=case):
                done = subprocess.run([str(self.exe), case], capture_output=True, text=True, timeout=60)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_the_four_core_sources_are_compiled_into_the_library(self) -> None:
        members = subprocess.run([self.tool.ar, "t", str(self.build.lib)], capture_output=True, text=True,
                                 errors="replace").stdout.split()
        for obj in self.objects():
            with self.subTest(obj=obj.name):
                self.assertTrue(obj.is_file())
                self.assertIn(obj.name, members)
        defined = subprocess.run([self.nm, "-g", "--defined-only", str(self.objects()[0])], capture_output=True,
                                 text=True, errors="replace").stdout
        self.assertIn("DebugCheckVersionAndDataLayout", defined)

    # What an object compiled with exceptions on names and one compiled without them does not: measured on
    # imgui_draw.cpp with the plan's flags, the second has none of these (__cxa_guard_acquire and _release are the
    # thread-safe local statics, present either way).
    EXCEPTION_NAMES = {"_Unwind_Resume", "_ZSt9terminatev", "__cxa_begin_catch", "__cxa_end_catch", "__cxa_throw",
                       "__cxa_rethrow"}

    def test_the_core_is_built_without_exceptions_and_without_run_time_type_information(self) -> None:
        for obj in self.objects():
            with self.subTest(obj=obj.name):
                bad = [n for n in undefined_names(self.nm, obj)
                       if n in self.EXCEPTION_NAMES or "personality" in n or "dynamic_cast" in n]
                self.assertEqual(bad, [])

    @unittest.skipUnless(WIN, "a PE import table")
    def test_the_program_imports_nothing_dear_imgui_would_add_with_the_windows_functions_on(self) -> None:
        # IMGUI_DISABLE_WIN32_FUNCTIONS keeps windows.h out of imgui.cpp: no USER32 (clipboard), SHELL32 (open a link)
        # or IMM32 (input method). The imports stay the system's C library and KERNEL32, and no toolchain DLL.
        imports = [name.lower() for name in pe_imports(self.whole)]
        self.assertIn("kernel32.dll", imports)
        odd = [n for n in imports if n != "kernel32.dll" and not n.startswith("api-ms-win-crt-")]
        self.assertEqual(odd, [], f"imports: {imports}")


if __name__ == "__main__":
    unittest.main()
