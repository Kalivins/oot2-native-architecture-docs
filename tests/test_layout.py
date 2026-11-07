# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/layout.py: module placement and the layout header.

Sizes are invented; no module of the game is read.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402


class PlaceTest(unittest.TestCase):
    def test_load_order_and_alignment(self) -> None:
        mods = layout.place({"sdk": 0x3000, "main": 0x201000, "subsdk0": 0x5000})
        self.assertEqual([m.name for m in mods], ["main", "subsdk0", "sdk"])
        self.assertEqual(mods[0].base, 0x70_0000_0000)
        self.assertEqual(mods[1].base, 0x70_0040_0000)   # main's end rounded up to 2 MiB
        self.assertEqual(mods[2].base, 0x70_0060_0000)
        for m in mods:
            self.assertEqual(m.base % layout.ALIGN, 0)

    def test_refusals(self) -> None:
        bad = [
            {"sdk": 0x1000},                       # no main
            {"main": 0x1000, "rtld": 0x1000},      # rtld is not loaded; unknown names are refused
            {"main": 0},
            {"main": 0x1800},                      # not whole pages
            {"main": layout.MODULE_LIMIT - layout.FIRST_BASE + 0x1000},
        ]
        for sizes in bad:
            with self.subTest(sizes=sizes):
                with self.assertRaises(ValueError):
                    layout.place(sizes)

    def test_largest_module_that_fits(self) -> None:
        mods = layout.place({"main": layout.MODULE_LIMIT - layout.FIRST_BASE})
        self.assertEqual(mods[0].base + mods[0].size, layout.MODULE_LIMIT)

    def test_written_header_and_json(self) -> None:
        mods = layout.place({"main": 0x100000, "sdk": 0x40000})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            layout.write(mods, out)
            self.assertEqual(layout.read(out), mods)
            header = (out / "layout.h").read_text(encoding="utf-8")
            self.assertIn("#define MOD_MAIN_BASE 0x7000000000ull", header)
            self.assertIn("#define MOD_SDK_BASE 0x7000200000ull", header)
            self.assertIn("#define MOD_SDK_SIZE 0x40000ull", header)
            self.assertIn("#define MOD_COUNT 2", header)
            self.assertIn("#define MOD_LIST(X) X(main, MAIN) X(sdk, SDK)", header)
            self.assertNotIn(b"\r", (out / "layout.h").read_bytes())

    def test_command_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(layout.main(["--module", "main=0x2000", "--module", "sdk=4096", "--out", tmp]), 0)
            self.assertEqual([m.size for m in layout.read(Path(tmp))], [0x2000, 0x1000])
            self.assertEqual(layout.main(["--module", "main=0x2000", "--module", "main=0x2000", "--out", tmp]), 1)
            self.assertEqual(layout.main(["--module", "main=0x1001", "--out", tmp]), 1)


if __name__ == "__main__":
    unittest.main()
