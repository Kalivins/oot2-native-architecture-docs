# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the guest address space and the module loader.

Builds tests/runtime/memory_test.c with runtime/core against a synthetic
layout (two modules of 0x8000 bytes, main and sdk), at -O1 and -O2 with
-march=x86-64-v3, then runs each case in its own process. The modules are
assembled inside the C test from synthetic bytes; nothing is read from a dump.
A passing case must print "ok <case>"; a failing case must end in exactly one
FATAL line with the expected exit code and text. The writewatch case needs the
system's write watch: elsewhere than Windows it prints "skip writewatch" and
its subtest is skipped; on Windows a skip is a failure.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402

CORE = ROOT / "runtime" / "core"
TEST_SOURCE = ROOT / "tests" / "runtime" / "memory_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000

EXIT_FATAL, EXIT_UNIMPLEMENTED = 70, 71

PASSING = ("sha256", "heap", "pool", "alias", "stack", "load", "io", "writewatch")

# case: (exit code, text the FATAL line must contain)
FAILING = {
    "touch_heap": (EXIT_FATAL, "guest memory: read of 0x"),
    "touch_alias": (EXIT_FATAL, "(alias+0x5000): an alias page MapPhysicalMemory has not mapped"),
    "touch_unmapped_again": (EXIT_FATAL, "(alias+0x5000): an alias page MapPhysicalMemory has not mapped"),
    "touch_stack": (EXIT_FATAL, "(stack+0x1000): no stack was moved there with MapMemory"),
    "touch_moved": (EXIT_FATAL, "(heap+0x100): MapMemory moved this memory to the stack region"),
    "touch_free": (EXIT_FATAL, "read of 0x7f00000000: no guest memory is mapped there"),
    "touch_hostfn": (EXIT_FATAL, "host functions have no memory"),
    "write_text": (EXIT_FATAL, "write of 0x7000000010 (main+0x10): module text and rodata are read-only"),
    "write_relro": (EXIT_FATAL, "(main+0x2008): SetMemoryPermission protects it"),
    "missing_import_call": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented import: _ZN2nn2os12NotThereYetEv"),
    "load_compressed": (EXIT_FATAL, "is compressed (flags 0x39)"),
    "load_corrupt": (EXIT_FATAL, "the data segment has SHA-256"),
    "load_size": (EXIT_FATAL, "is a 0x8000-byte image, the layout says 0x9000"),
    "load_sha256": (EXIT_FATAL, "the build expects 00"),
    "load_reloc_outside": (EXIT_FATAL, "main DT_RELA entry 0 writes +0x10, outside the data segment"),
    "load_reloc_type": (EXIT_FATAL, "main DT_RELA entry 0 has relocation type 1024"),
    "load_before_init": (EXIT_FATAL, "rt_load_modules before rt_mem_init"),
}
# The heap case names the exact place too.
HEAP_PLACE = "(heap+0x300000): past the heap size SetHeapSize set"


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


def sha256_inputs() -> list[bytes]:
    buf = bytes((i * 7) & 0xFF for i in range(1000))
    return [b"", b"abc", buf[:55], buf[:56], buf[:64], buf]


class MemoryTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-memory-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE, "sdk": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        # Windows: without high-entropy ASLR the host's own stacks stay below 4 GiB, out of the fixed module region
        # (runtime/core/memory.c, rt_mem_init); the program is linked the same way.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"memory_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(ROOT / "runtime" / "include"), "-I", str(CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory(prefix="oot2-modules-") as scratch:
            return subprocess.run([str(self.exes[level]), case, scratch], capture_output=True, text=True, timeout=120)

    def test_passing_cases(self) -> None:
        for level in LEVELS:
            for case in PASSING:
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    if f"skip {case}" in done.stdout:   # no write watch on this platform
                        self.assertNotEqual(os.name, "nt", done.stdout)
                        self.skipTest(done.stdout.strip())
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)

    def test_sha256_matches_hashlib(self) -> None:
        done = self.run_case(LEVELS[0], "sha256")
        got = [line.split()[2] for line in done.stdout.splitlines() if line.startswith("sha256 ")]
        self.assertEqual(got, [hashlib.sha256(b).hexdigest() for b in sha256_inputs()])

    def test_load_report_names_the_missing_import(self) -> None:
        done = self.run_case(LEVELS[0], "load")
        self.assertIn("main: import _ZN2nn2os12NotThereYetEv is defined nowhere", done.stderr)
        self.assertIn("relocations 2 RELATIVE, 5 RELR, 1 ABS64, 4 GLOB_DAT, 2 JUMP_SLOT", done.stderr)
        self.assertIn("5 imports: main 0, sdk 2, host 1, weak 1, missing 1", done.stderr)
        self.assertIn("replaced longjmp at 0x", done.stderr)

    def test_fatal_cases(self) -> None:
        for level in LEVELS:
            for case, (code, text) in FAILING.items():
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    if case == "touch_heap":
                        self.assertIn(HEAP_PLACE, done.stderr)
                    self.assertNotIn(f"ok {case}", done.stdout)
                    fatal_lines = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
                    self.assertEqual(len(fatal_lines), 1, done.stderr)

    def test_unknown_case_is_refused(self) -> None:
        done = self.run_case(LEVELS[0], "no_such_case")
        self.assertEqual(done.returncode, 2)

    def test_module_base_matches_layout(self) -> None:
        header = (ROOT / "runtime" / "include" / "oot2" / "memory.h").read_text(encoding="utf-8")
        self.assertIn(f"#define RT_MODULE_BASE {layout.FIRST_BASE:#x}ull", header)


if __name__ == "__main__":
    unittest.main()
