# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the guest threads and the kernel objects.

Builds tests/runtime/kernel_test.c with runtime/core against a synthetic
layout (one module of 0x8000 bytes, main), at -O1 and -O2 with
-march=x86-64-v3, then runs each case in its own process. The guest programs
are C functions written by hand in the lifter's shape; nothing is read from a
dump. A passing case must print "ok <case>"; a failing case must end in
exactly one FATAL line with the expected exit code and text.

Some cases measure time (a 30 ms busy thread, a 5 ms sleep that preempts a
spinning thread); their bounds are wide, so a loaded machine slows them down
without failing them.

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
TEST_SOURCE = ROOT / "tests" / "runtime" / "kernel_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000
TIMEOUT = 60

EXIT_FATAL, EXIT_UNIMPLEMENTED, EXIT_GUEST_STOP = 70, 71, 72
EXIT_HUNG = 3   # kernel_test.c's watchdog: the case had not ended after 20 s; it logs the kernel's state first

PASSING = ("threads", "priority", "mutex", "mutex_order", "condvar", "key_order", "arbiter", "events", "spin_gate",
           "async", "activity", "block", "times", "info", "npdm")

# case: (exit code, text the FATAL line must contain)
FAILING = {
    "thread_returns": (EXIT_GUEST_STOP, "returned from its entry"),
    "svc_outside_mask": (EXIT_GUEST_STOP, "FATAL: guest stop: svc 0x45 CreateEvent: the program's SVC mask does not "
                                          "allow it, so the console's kernel refuses it"),
    "svc_masked": (EXIT_GUEST_STOP, "FATAL: guest stop: svc 0x27 OutputDebugString: the program's SVC mask does not "
                                    "allow it"),
    "svc_deferred": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented svc: 0x21 SendSyncRequest: the service layer serves "
                                         "it, and it is not written yet"),
    "getinfo_unknown": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented GetInfo: id 9 (handle 0, sub id 0)"),
    "arbiter_type": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented WaitForAddress: arbitration type 7"),
    "break": (EXIT_GUEST_STOP, "FATAL: guest stop: Break reason 0 (result 0): the SDK aborted"),
    "touch_kernel": (EXIT_FATAL, "no thread-local page is allocated there"),
    "host_svc": (EXIT_FATAL, "FATAL: kernel: svc 0xb on a host thread, which is no guest thread"),
    "init_order": (EXIT_FATAL, "FATAL: rt_kernel_init before rt_mem_init"),
}
# The kernel touch names its region too.
KERNEL_PLACE = "(kernel+0x"

# An application's 49 SVCs (the mask kernel_test.c sets): 38 the runtime implements, 11 named fatals with a reason.
CENSUS = (
    "svc 0x08 CreateThread: implemented",
    "svc 0x29 GetInfo: implemented",
    "svc 0x13 MapSharedMemory: named fatal, the service layer serves it",
    "svc 0x33 GetThreadContext3: named fatal, the registers of a paused thread",
    "svc census: 38 implemented, 11 named fatals with a reason, 0 without one",
)


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class KernelTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-kernel-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        # Windows: without high-entropy ASLR the host's own stacks stay below 4 GiB, out of the fixed module region
        # (runtime/core/memory.c, rt_mem_init); the program is linked the same way.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"kernel_test{level}{'.exe' if os.name == 'nt' else ''}"
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
        try:
            done = subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{case} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")
        if done.returncode == EXIT_HUNG:
            self.fail(f"{case} at {level} hung (the case's watchdog):\n{done.stdout}{done.stderr}")
        return done

    def test_passing_cases(self) -> None:
        for level in LEVELS:
            for case in PASSING:
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)

    def test_async_preemption_is_in_use(self) -> None:
        # The case checks it too; this names the reason when the build put a runtime function among the guest ones.
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "async")
                off = [line for line in done.stderr.splitlines() if "asynchronous preemption off" in line]
                self.assertEqual(off, [], done.stderr)
                self.assertIn("async sent ", done.stdout)

    def test_census(self) -> None:
        done = self.run_case(LEVELS[0], "info")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        for line in CENSUS:
            self.assertIn(line, done.stderr)
        self.assertIn("guest: hello from the guest", done.stderr)
        self.assertIn("guest Break notification 0x80000000", done.stderr)

    def test_fatal_cases(self) -> None:
        for level in LEVELS:
            for case, (code, text) in FAILING.items():
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    if case == "touch_kernel":
                        self.assertIn(KERNEL_PLACE, done.stderr)
                    self.assertNotIn(f"ok {case}", done.stdout)
                    fatal_lines = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
                    self.assertEqual(len(fatal_lines), 1, done.stderr)

    def test_unknown_case_is_refused(self) -> None:
        done = self.run_case(LEVELS[0], "no_such_case")
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
