# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the page ranges of runtime/core/memory.c (lever L1,
architecture D19 and D27): counting and releasing 4 KiB pages inside the
first-touch handler's 64 KiB chunks.

Builds tests/runtime/pages_test.c with runtime/core at -O1 and -O2 and runs each
case in its own process, with the opt-in switch OOT2_PAGE_RELEASE set as the case
needs. The pool is synthetic (a heap and an alias region with numbers written by
the test); nothing is read from a dump. What the cases show:

- off: no switch, no range; first-touch commit as it was;
- count: a read of an armed page counted and the run goes on, guest code told
  from runtime code, the page restored, never a first touch;
- count_neighbour: a first touch beside armed pages leaves them armed;
- move: MapMemory copies armed pages without faulting under its lock;
- release: a write recommits one page and calls the owner, committed bytes
  follow a release and a recommit, the neighbours keep their bytes;
- busy: a range the owner holds busy is never released;
- neighbour: a released page beside committed ones is not taken for a first touch;
- writewatch: the write watch after a decommit and a recommit (Windows);
- io: host I/O into and out of released pages;
- unsealed: a range before rt_dispatch_init is a fatal error, not a lazy index build in a fault;
- read_released_*: a read of a released page is the named fault, counted, and says
  whether guest code or runtime code read it.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

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

import layout  # noqa: E402

CORE = ROOT / "runtime" / "core"
HOST_SOURCE = ROOT / "tests" / "runtime" / "pages_host.c"
TEST_SOURCE = ROOT / "tests" / "runtime" / "pages_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000
EXIT_FATAL = 70
SWITCH = "OOT2_PAGE_RELEASE"

# case: value of the switch (None: not set)
PASSING = {
    "off": None,
    "count": "count",
    "count_neighbour": "count",
    "move": "count",
    "release": "release",
    "busy": "release",
    "neighbour": "release",
    "writewatch": "release",
    "io": "release",
}
# case: (switch value, text the FATAL line must contain, text the census line must contain)
FAILING = {
    "read_released_guest": (
        "release",
        '(heap+0x102008): a page of range "pool" was released and a read of it is a fault; read by guest code in '
        "0x7000000200 (main+0x200)",
        "released reads guest 1 host 0, released now 1",
    ),
    "read_released_after_end": (
        "release",
        '(heap+0x102008): a page of range "pool" was released and a read of it is a fault; read by runtime or '
        "back-end code",
        "released reads guest 0 host 1, released now 1",
    ),
    "read_released_host": (
        "release",
        '(heap+0x102008): a page of range "pool" was released and a read of it is a fault; read by runtime or '
        "back-end code",
        "released reads guest 0 host 1, released now 1",
    ),
}


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class PagesTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-pages-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE, "sdk": MODULE_SIZE}), out)
        sources = [str(HOST_SOURCE), str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        # Windows: the program links without high-entropy ASLR, as tests/test_runtime_memory.py explains.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"pages_test{level}{'.exe' if os.name == 'nt' else ''}"
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

    def run_case(self, level: str, case: str, switch: str | None) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != SWITCH}
        if switch is not None:
            env[SWITCH] = switch
        return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=30, env=env)

    def test_passing_cases(self) -> None:
        for level in LEVELS:
            for case, switch in PASSING.items():
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case, switch)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    if f"skip {case}" in done.stdout:   # the platform cannot show it (no write watch on POSIX)
                        self.skipTest(done.stdout.strip())
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)

    def test_a_read_of_a_released_page_is_the_named_fault(self) -> None:
        for level in LEVELS:
            for case, (switch, text, census) in FAILING.items():
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case, switch)
                    self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    self.assertIn(census, done.stderr)
                    self.assertNotIn(f"ok {case}", done.stdout)
                    fatal_lines = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
                    self.assertEqual(len(fatal_lines), 1, done.stderr)
                    self.assertIn("guest memory: read of 0x", fatal_lines[0])

    def test_a_range_before_the_tables_are_sealed_is_fatal(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level, case="unsealed"):
                done = self.run_case(level, "unsealed", "count")
                self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                self.assertIn("rt_pages_watch before rt_dispatch_init: the function tables are not sealed", done.stderr)
                self.assertNotIn("ok unsealed", done.stdout)

    def test_the_switch_is_never_the_default(self) -> None:
        # An empty switch is no switch: every range is refused. Any other value the setting page_release does not take
        # (off, count, release) stops the program by name before memory is set up (oot2/settings.h), so it can never
        # turn the lever on.
        done = self.run_case(LEVELS[0], "off", "")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertNotIn("page ranges:", done.stderr)
        for value in ("0", "1", "yes", "RELEASE"):
            with self.subTest(value=value):
                done = self.run_case(LEVELS[0], "off", value)
                self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
                self.assertIn(f"oot2: OOT2_PAGE_RELEASE={value}: page_release takes off, count or release", done.stderr)
                self.assertNotIn("page ranges:", done.stderr)
                self.assertNotIn("ok off", done.stdout)

    def test_the_switch_is_logged_when_set(self) -> None:
        self.assertIn("page ranges: count allowed", self.run_case(LEVELS[0], "count", "count").stderr)
        self.assertIn("page ranges: release allowed", self.run_case(LEVELS[0], "release", "release").stderr)

    def test_unknown_case_is_refused(self) -> None:
        self.assertEqual(self.run_case(LEVELS[0], "no_such_case", None).returncode, 2)


if __name__ == "__main__":
    unittest.main()
