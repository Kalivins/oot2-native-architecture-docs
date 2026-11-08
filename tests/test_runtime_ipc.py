# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the services' side of the IPC layer (runtime/hos/ipc.c).

Builds tests/runtime/ipc_test.c with runtime/core, runtime/nvn and runtime/hos
against a synthetic layout (one module of 0x8000 bytes, main), at -O1 and -O2
with -march=x86-64-v3, then runs each case in its own process with a fresh
game folder. A test service made of host functions answers four names; the
client (tests/runtime/cmif_client.h) builds the messages in a guest thread's
message buffer; nothing is read from a dump.

A passing case must print "ok <case>"; a failing case must end in exactly one
FATAL line with the expected exit code and text, and leave census.tsv with its
stop row in the game folder.

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

RUNTIME = ROOT / "runtime"
TEST_SOURCE = ROOT / "tests" / "runtime" / "ipc_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000
TIMEOUT = 60

EXIT_FATAL, EXIT_UNIMPLEMENTED = 70, 71

PASSING = ("basic", "domain")

# case: (exit code, text the FATAL line must contain, a row census.tsv must hold)
FAILING = {
    "fifth": (EXIT_UNIMPLEMENTED, 'FATAL: unimplemented service: sm: GetServiceHandle("tste:u"): no service of that '
                                  'name is served', "service\ttste:u\tnamed-fatal\t1\t"),
    "unknown": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented service command: tsta:u cmif command 99 (0 bytes of "
                                    "arguments, buffers x0 a0 b0 w0 c0, handles copy 0 move 0)",
                "command\ttsta:u cmif 99\tnamed-fatal\t1\t"),
    "unknown_tipc": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented service command: tstb:u tipc command 99 (0 bytes of "
                                         "arguments", "command\ttstb:u tipc 99\tnamed-fatal\t1\t"),
    "unknown_domain": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented service command: tchild cmif command 99 (0 bytes of "
                                           "arguments", "command\ttchild cmif 99\tnamed-fatal\t1\t"),
    "arg_beyond": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented ipc: tsta:u command 5: argument bytes 0+64 beyond the 0 "
                                       "the request carries", "service\ttsta:u\timplemented\t1\t"),
    "domain_twice": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented service command: tsta:u control 0 "
                                         "ConvertCurrentObjectToDomain: the session is a domain already",
                     "command\ttsta:u control 0\tnamed-fatal\t2\t"),   # met twice: the first conversion was served
    "domain_gone": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented ipc: tsta:u: domain object 9 does not exist",
                    "command\ttsta:u control 0\timplemented\t1\t"),
    "domain_closed": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented ipc: tsta:u: domain object 2 does not exist",
                      "command\ttsta:u control 0\timplemented\t1\t"),
    "domain_nin": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented ipc: tsta:u: domain request kind 1 with 1 objects in "
                                       "and 16 bytes of data", "command\ttsta:u control 0\timplemented\t1\t"),
    "domain_kind": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented ipc: tsta:u: domain request kind 7 with 0 objects in "
                                        "and 16 bytes of data", "command\ttsta:u control 0\timplemented\t1\t"),
    "domain_full": (EXIT_FATAL, "FATAL: ipc: tstb:u: the domain holds 62 objects already",
                    "command\ttstb:u control 0\timplemented\t1\t"),
    "inline_receive": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented ipc: a receive list in the inline mode (1), which no "
                                           "service of this runtime uses", "service\ttsta:u\timplemented\t1\t"),
}


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class IpcTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-ipc-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)]
        for part in ("core", "nvn", "hos", "audio"):
            sources += sorted(str(p) for p in (RUNTIME / part).glob("*.c"))
        # Windows: without high-entropy ASLR the host's own stacks stay below 4 GiB, out of the fixed module region
        # (runtime/core/memory.c, rt_mem_init); the program is linked the same way.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        libs = [] if os.name == "nt" else ["-lm"]   # lrintf in the sink and the tests is not always inlined
        for level in LEVELS:
            exe = out / f"ipc_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(out), "-I",
                   str(ROOT / "tests" / "runtime"), *sources, "-o", str(exe), *libs]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, **env_extra: str) -> tuple[subprocess.CompletedProcess, Path]:
        folder = Path(tempfile.mkdtemp(prefix=f"{case}{level}-", dir=self.tmp.name))
        env = {k: v for k, v in os.environ.items() if not k.startswith("OOT2_CENSUS")}   # a default run unless asked
        env.update(env_extra)
        try:
            done = subprocess.run([str(self.exes[level]), case, str(folder)], capture_output=True, text=True,
                                  timeout=TIMEOUT, env=env)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{case} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")
        return done, folder

    def test_passing_cases(self) -> None:
        for level in LEVELS:
            for case in PASSING:
                with self.subTest(level=level, case=case):
                    done, _ = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)

    def test_fatal_cases(self) -> None:
        for level in LEVELS:
            for case, (code, text, row) in FAILING.items():
                with self.subTest(level=level, case=case):
                    done, folder = self.run_case(level, case)
                    self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    self.assertNotIn(f"ok {case}", done.stdout)
                    fatal_lines = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
                    self.assertEqual(len(fatal_lines), 1, done.stderr)
                    census = (folder / "census.tsv").read_text(encoding="utf-8")
                    self.assertIn("\n" + row, census)
                    stop = [line for line in census.splitlines() if line.startswith("stop\t")]
                    self.assertEqual(len(stop), 1, census)
                    self.assertTrue(stop[0].startswith(f"stop\texit {code}\tnamed-fatal\t1\t\t{fatal_lines[0]}"), stop[0])

    def test_census_mode_refuses_a_fifth_name_and_serves_the_four(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, folder = self.run_case(level, "fifth_census", OOT2_CENSUS="1")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok fifth_census", done.stdout)
                self.assertIn('unserved service: sm: GetServiceHandle("tste:u") refused with 0x415', done.stderr)
                self.assertNotIn("FATAL", done.stderr)


if __name__ == "__main__":
    unittest.main()
