# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of runtime/ui, the runtime's one C++ directory (runtime/include/oot2/ui.h, research 104): a stand-in that uses
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
from test_build_game import pe_imports, toolchain_or_skip  # noqa: E402

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
        # The runtime's headers and runtime/ui only: the library of this folder is the stand-in and nothing else.
        cls.runtime = top / "runtime"
        shutil.copytree(RUNTIME / "include", cls.runtime / "include")
        shutil.copytree(RUNTIME / "ui", cls.runtime / "ui", ignore=shutil.ignore_patterns("__pycache__"))
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
        # The pieces an installer bundle takes from an llvm-mingw tree for C++ (research 104): the driver, the static
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


if __name__ == "__main__":
    unittest.main()
