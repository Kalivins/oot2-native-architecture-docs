# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Build of tools/loadprobe/load_probe.c by the probe's own build (load_probe.build: -Wall -Wextra -Werror, the probe,
its end marker and runtime/core), then a run without modules.

The probe describes its modules with RtModule literals. When RtModule gained the end marker (ticket 92) the probe was
missed because nothing built it. A literal that leaves a field out is a build error here (-Wmissing-field-
initializers). A marker linked below the probe's stand-in function, which rt_dispatch_init would refuse only after the
modules are loaded, is the probe's exit 3 before it reads its arguments; run without arguments, a correct probe stops
at its usage line (exit 2). The layout.h is synthetic and no module is read. Needs clang (OOT2_CC, else clang on PATH);
skips without one unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.
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
sys.path.insert(0, str(ROOT / "tools" / "loadprobe"))

import load_probe  # noqa: E402

LAYOUT = (
    "#define MOD_COUNT 1\n#define MOD_LIST(X) X(a, A)\n"
    "#define MOD_A_BASE 0x100000ull\n#define MOD_A_SIZE 0x10000ull\n"
)


class LoadProbeBuildTest(unittest.TestCase):
    def test_the_probe_builds_and_its_end_marker_lies_past_its_stand_in(self) -> None:
        cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                self.fail("OOT2_REQUIRE_CLANG is set but there is no clang")
            self.skipTest("no clang")
        with tempfile.TemporaryDirectory(prefix="oot2-loadprobe-", ignore_cleanup_errors=True) as tmp:
            (Path(tmp) / "layout.h").write_text(LAYOUT, encoding="utf-8", newline="\n")
            exe = load_probe.build(Path(tmp), cc)
            done = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        self.assertNotIn("end marker", done.stderr)
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn("usage: load_probe", done.stderr)


if __name__ == "__main__":
    unittest.main()
