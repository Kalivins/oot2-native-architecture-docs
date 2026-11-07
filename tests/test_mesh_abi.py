# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The mesh draw's first-task ABI is one contract written in two places: the translator's MeshAbi.cs and the runtime's
mesh_abi.h. These tests keep the two in step, and keep the numbers consistent with each other and with the push constant
layout the uni descriptor layout reserves for the first task (UniLayout.cs), which the task and mesh stages read.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import raw_abi

ROOT = Path(__file__).resolve().parent.parent
CS = ROOT / "xlate/third_party/ryujinx/src/Ryujinx.Graphics.Shader/MeshAbi.cs"
HEADER = ROOT / "runtime/include/oot2/mesh_abi.h"


def camel(c_name: str) -> str:
    return "".join(part.capitalize() for part in c_name.removeprefix("MESH_").split("_"))


def header_constants() -> dict[str, int]:
    text = HEADER.read_text(encoding="utf-8")
    return {camel(m.group(1)): int(m.group(2)) for m in re.finditer(r"^#define\s+(MESH_\w+)\s+(\d+)\s*$", text, re.M)}


def cs_constants() -> dict[str, int]:
    text = CS.read_text(encoding="utf-8")
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"^\s*public const int (\w+) = (\d+);", text, re.M)}


def uni_constant(name: str) -> int:
    m = re.search(rf"public const int {name} = (\d+);", raw_abi.UNI_LAYOUT.read_text(encoding="utf-8"))
    assert m, f"UniLayout.{name} not found"
    return int(m.group(1))


class MeshAbiTest(unittest.TestCase):
    def test_the_two_sides_name_the_same_constants_and_agree(self):
        c, cs = header_constants(), cs_constants()
        self.assertGreaterEqual(len(c), 7, "the header was not read")
        self.assertEqual(sorted(set(c) ^ set(cs)), [], "a constant exists on one side only")
        self.assertEqual({k: (c[k], cs[k]) for k in c if c[k] != cs[k]}, {})

    def test_the_first_task_words_lie_where_the_uni_layout_keeps_them(self):
        a = header_constants()
        self.assertEqual(a["FirstTaskPushOffset"], 4 * a["FirstTaskPushWord"])
        self.assertEqual(a["FirstTaskPushOffset"], uni_constant("FirstTaskOffset"))
        self.assertLessEqual(a["FirstTaskAddressBytes"], uni_constant("FirstTaskBytes"))
        # The vertex stage's texture slots 2 to 4: a mesh pipeline has no vertex stage
        slots = raw_abi.uni_slots_offset()
        self.assertEqual(a["FirstTaskPushOffset"], slots + 4 * 2)
        self.assertEqual(uni_constant("FirstTaskBytes"), 4 * 3)
        # Clear of the page table words
        self.assertGreaterEqual(a["FirstTaskPushOffset"], raw_abi.A["PushBytes"])

    def test_the_address_and_the_word_are_aligned(self):
        a = header_constants()
        self.assertEqual(a["FirstTaskPushOffset"] % a["FirstTaskAddressBytes"], 0, "a 64-bit address at an 8-byte aligned offset")
        self.assertEqual(a["FirstTaskWordBytes"], 4)
        self.assertEqual(a["IndirectFirstOffset"] % a["FirstTaskWordBytes"], 0, "a physical load of a word is 4-byte aligned")
        self.assertEqual(a["IndirectRecordBytes"] % a["FirstTaskWordBytes"], 0)

    def test_the_indirect_record_is_the_extensions_command_then_the_first_task(self):
        a = header_constants()
        self.assertEqual(a["IndirectCommandBytes"], 12, "VkDrawMeshTasksIndirectCommandEXT is three 32-bit group counts")
        self.assertEqual(a["IndirectFirstOffset"], a["IndirectCommandBytes"])
        self.assertEqual(a["IndirectRecordBytes"], a["IndirectFirstOffset"] + a["FirstTaskWordBytes"])


if __name__ == "__main__":
    unittest.main()
