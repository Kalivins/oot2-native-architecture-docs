# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the NVN to Vulkan tables and the constant bank fillers of the Vulkan back end (runtime/include/oot2/vk_nvn.h,
runtime/gpu/vk_nvn.c; the Vulkan replay), through tests/runtime/vk_nvn_test.c, built like tests/test_runtime_gpu.py's programs.

No device is needed (continuous integration runs these). Two groups:

- The converters. The test program converts every integer from -8 to 0x200 and the extremes with each converter and holds
  it to a table of rows written with both enumerations' names: a row must give its Vulkan value, anything else must be
  refused with the output untouched. This driver then compares what the program accepted with what nvn.h defines, so an
  enumerator that a table misses (or that nvn.h gains) fails here; the only values NVN defines and the converter refuses are
  the four draw primitives that Vulkan has no topology for. Vertex formats are checked against what their names say (the
  Vulkan format, its bytes against the front end's own size of the format, the scalar type) over every NVN format.
- The banks. The program writes each bank filler's output from a bank that was full of 0xFF and requires every byte it
  names to hold its value at its offset (literals from the layout of nxnative's dmsh_abi.h) and every other byte to be zero:
  the storage tables of the four graphics stages, the largest point size and the base instance; the compute bank (group
  counts at byte 4, uniform rows at 0x210, storage rows at 0x310) and its control section's counts and limits, which this
  driver also compares with what tools/shaders/split.py reads from the same bytes; c[2] with the rows that combined
  handles from slot 32 share with the images; the support buffer.

Run: python -m unittest discover -s tests -p test_runtime_vk_nvn.py -v
"""

from __future__ import annotations

import atexit
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import ROOT, build_program, case_env, clang_or_skip, vulkan_or_skip  # sets the path of tools/

sys.path.insert(0, str(ROOT / "tools" / "shaders"))
import split  # noqa: E402

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_nvn_test.c"
NVN_HEADER = ROOT / "runtime" / "include" / "oot2" / "nvn.h"
LEVEL = "-O2"

_build: dict[str, object] = {}


def program() -> Path:
    """The test program, built once per run."""
    if "exe" not in _build:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        tmp = tempfile.TemporaryDirectory(prefix="oot2-vknvn-", ignore_cleanup_errors=True)
        atexit.register(tmp.cleanup)
        _build["tmp"] = tmp
        _build["include"] = include
        _build["exe"] = build_program(cc, include, glslang, TEST_SOURCE, Path(tmp.name), levels=(LEVEL,))[LEVEL]
    return _build["exe"]  # type: ignore[return-value]


def run(case: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(program()), case, *args], capture_output=True, text=True, timeout=120, env=case_env())


def passes(case: str, *args: str) -> str:
    """The case's stdout; it must exit 0, print its ok line and say nothing on stderr."""
    done = run(case, *args)
    assert done.returncode == 0, f"{case}:\n{done.stdout}{done.stderr}"
    assert f"ok {case}\n" in done.stdout, done.stdout
    assert done.stderr == "", done.stderr
    return done.stdout


def nvn_values(*prefixes: str) -> dict[int, str]:
    """The enumerators NVN_<prefix>_<NAME> = value of nvn.h, by value (the *_LARGE sentinels left out)."""
    text = NVN_HEADER.read_text(encoding="utf-8")
    out: dict[int, str] = {}
    for prefix in prefixes:
        for m in re.finditer(rf"^\s*NVN_{prefix}_(\w+) = (0x[0-9A-Fa-f]+|\d+),", text, re.M):
            if m.group(1) != "LARGE":
                out[int(m.group(2), 0)] = m.group(1)
    return out


def nvn_format_names() -> dict[int, str]:
    out: dict[int, str] = {}
    for m in re.finditer(r"^\s*NVN_FORMAT_(\w+) = (0x[0-9A-Fa-f]+),", NVN_HEADER.read_text(encoding="utf-8"), re.M):
        if m.group(1) not in ("NONE", "LARGE"):
            out[int(m.group(2), 16)] = m.group(1)
    return out


def vk_format_names() -> dict[int, str]:
    """VkFormat number to its name without VK_FORMAT_, from vulkan_core.h (the first name of a number)."""
    program()
    text = (Path(_build["include"]) / "vulkan" / "vulkan_core.h").read_text(encoding="utf-8", errors="replace")  # type: ignore[arg-type]
    names: dict[int, str] = {}
    for m in re.finditer(r"^\s*VK_FORMAT_(\w+) = (\d+),", text, re.M):
        n = int(m.group(2))
        if n not in names and not m.group(1).endswith(("_KHR", "_EXT")):
            names[n] = m.group(1)
    return names


class ConverterTest(unittest.TestCase):
    def accepted(self, case: str) -> set[int]:
        line = [x for x in passes(case).splitlines() if x.startswith("accepts")][0]
        return {int(x, 16) for x in line.split()[1:]}

    def check_enum(self, case: str, prefixes: tuple[str, ...], refused: set[str]) -> None:
        """What the program accepted is what nvn.h defines for the enumerations, less the named values."""
        defined = nvn_values(*prefixes)
        self.assertGreater(len(defined), 1, prefixes)
        accepted = self.accepted(case)
        self.assertEqual({defined[v] for v in set(defined) - accepted}, refused, f"{case}: the values refused")
        self.assertEqual(accepted, set(defined) - {v for v, n in defined.items() if n in refused}, case)
        self.assertLessEqual(accepted, set(defined), f"{case} accepted a value that nvn.h does not define")

    def test_draw_primitives_convert_and_the_four_vulkan_lacks_are_refused(self) -> None:
        self.check_enum("topology", ("DRAW_PRIMITIVE",), {"LINE_LOOP", "QUADS", "QUAD_STRIP", "POLYGON"})

    def test_index_types_convert_with_their_sizes(self) -> None:
        self.check_enum("index_type", ("INDEX_TYPE",), set())

    def test_depth_stencil_compare_and_alpha_functions_are_one_list_and_convert(self) -> None:
        lists = [nvn_values(p) for p in ("DEPTH_FUNC", "STENCIL_FUNC", "COMPARE_FUNC", "ALPHA_FUNC")]
        for other in lists[1:]:
            self.assertEqual(other, lists[0])   # the same values under the same names
        self.check_enum("compare", ("DEPTH_FUNC",), set())

    def test_stencil_operations_convert(self) -> None:
        self.check_enum("stencil_op", ("STENCIL_OP",), set())

    def test_blend_factors_convert(self) -> None:
        defined = nvn_values("BLEND_FUNC")
        self.assertIn(0x61, defined)   # the constant factors lie far from the others
        self.check_enum("blend_factor", ("BLEND_FUNC",), set())

    def test_blend_equations_convert(self) -> None:
        self.check_enum("blend_op", ("BLEND_EQUATION",), set())

    def test_faces_convert(self) -> None:
        self.check_enum("cull", ("FACE",), set())

    def test_front_faces_convert(self) -> None:
        self.check_enum("front_face", ("FRONT_FACE",), set())

    def test_polygon_modes_convert(self) -> None:
        self.check_enum("polygon_mode", ("POLYGON_MODE",), set())

    def test_logic_operations_convert(self) -> None:
        self.check_enum("logic_op", ("LOGIC_OP",), set())


# NVN formats that are not R, RG, RGB or RGBA of 8, 16 or 32 bits and that are vertex formats, with the Vulkan format
# (by name) and the scalar type a shader sees: 0 float, 1 signed, 2 unsigned.
PACKED_VERTEX = {
    "RGB10A2": ("A2B10G10R10_UNORM_PACK32", 0), "RGB10A2UI": ("A2B10G10R10_UINT_PACK32", 2),
    "RGB10A2SN": ("A2B10G10R10_SNORM_PACK32", 0), "RGB10A2I": ("A2B10G10R10_SINT_PACK32", 1),
    "BGRA8": ("B8G8R8A8_UNORM", 0),
}
SUFFIX = {"": "UNORM", "SN": "SNORM", "UI": "UINT", "I": "SINT", "F": "SFLOAT"}
TYPE = {"": 0, "SN": 0, "UI": 2, "I": 1, "F": 0}


def expected_vertex(name: str) -> tuple[str, int, int] | None:
    """The Vulkan format name, bytes and scalar type that an NVN format's name says, or None when it is not a vertex format."""
    m = re.fullmatch(r"(R|RG|RGB|RGBA)(8|16|32)(SN|UI|I|F)?", name)
    if m:
        bits, suffix = int(m.group(2)), m.group(3) or ""
        if bits == 32 and suffix in ("", "SN"):
            return None   # NVN has no unsigned or signed normalized 32-bit format
        return ("".join(f"{c}{bits}" for c in m.group(1)) + "_" + SUFFIX[suffix], len(m.group(1)) * bits // 8, TYPE[suffix])
    if name in PACKED_VERTEX:
        return (PACKED_VERTEX[name][0], 4, PACKED_VERTEX[name][1])
    return None


class VertexFormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        out = passes("vertex")
        cls.rows: dict[int, dict[str, int]] = {}
        for line in out.splitlines():
            m = re.match(r"vfmt ([0-9a-f]{2}) (.*)$", line)
            if m and int(m.group(1), 16) != 0xFF:   # 0xFF stands for the probe of -1
                cls.rows[int(m.group(1), 16)] = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", m.group(2))}
        cls.nvn = nvn_format_names()
        cls.vk = vk_format_names()

    def test_every_format_from_0_to_0xa0_was_asked(self) -> None:
        self.assertEqual(sorted(self.rows), list(range(0, 0xA1)))

    def test_the_vertex_formats_are_those_the_names_say_and_nothing_else_is_one(self) -> None:
        accepted = {f for f, r in self.rows.items() if r["ok"]}
        wanted = {f for f, n in self.nvn.items() if expected_vertex(n)}
        self.assertEqual({self.nvn[f] for f in accepted - wanted}, set())
        self.assertEqual({self.nvn[f] for f in wanted - accepted}, set())
        self.assertGreater(len(wanted), 40)

    def test_each_has_the_vulkan_format_its_name_says_the_bytes_the_front_end_sizes_and_the_scalar_type(self) -> None:
        for f, n in self.nvn.items():
            want = expected_vertex(n)
            if not want:
                continue
            r = self.rows[f]
            who = f"NVN {n} (0x{f:02x})"
            self.assertEqual(self.vk[r["vk"]], want[0], who)
            self.assertEqual(r["bytes"], want[1], who)
            self.assertEqual(r["bytes"], r["nvn_bpb"], who)   # the size the game stores the attribute in
            self.assertEqual(r["type"], want[2], who)
            self.assertEqual(r["vk"], r["table"], who)        # what the format table gives, unchanged

    def test_three_component_formats_stay_three_components(self) -> None:
        seen = 0
        for f, n in self.nvn.items():
            m = re.fullmatch(r"RGB(8|16|32)(SN|UI|I|F)?", n)
            if m and expected_vertex(n):
                seen += 1
                self.assertRegex(self.vk[self.rows[f]["vk"]], rf"^R{m.group(1)}G{m.group(1)}B{m.group(1)}_[A-Z]+$", n)
                self.assertEqual(self.rows[f]["bytes"], 3 * int(m.group(1)) // 8, n)
        self.assertEqual(seen, 12)   # RGB8 has no float form: 4 of 8 bits, 5 of 16 and 3 of 32

    def test_the_table_has_a_vertex_format_for_the_two_srgb_formats_and_the_converter_refuses_them(self) -> None:
        # The format table keeps an sRGB format in its vertex column for the formats that have one as an image; Vulkan
        # reads no sRGB format as a vertex. Any other format the table gives and the converter refuses is a hole.
        held_back = {f for f, r in self.rows.items() if r["table"] and not r["ok"]}
        self.assertEqual({self.nvn[f] for f in held_back}, {"RGBA8_SRGB", "BGRA8_SRGB"})
        for f in held_back:
            self.assertIn("SRGB", self.vk[self.rows[f]["table"]])


class BankTest(unittest.TestCase):
    def test_graphics_c0_has_the_stages_storage_table_the_largest_point_size_and_the_base_instance(self) -> None:
        passes("c0_graphics")

    def test_the_compute_control_sections_counts_and_limits(self) -> None:
        passes("c0_layout")

    def test_compute_c0_has_the_group_counts_the_uniform_rows_and_the_storage_rows(self) -> None:
        passes("c0_compute")

    def test_c2_has_each_handle_at_its_row_and_a_bound_image_keeps_the_row_it_shares(self) -> None:
        passes("c2")

    def test_the_support_buffer_has_the_viewport_and_the_render_scales(self) -> None:
        passes("support")


class LayoutAgreesWithSplitTest(unittest.TestCase):
    """split.py is what the translator's job is made from (QueryAmpereComputeC0Layout): the layout the host reads from the
    same control bytes must give the same counts."""

    def control(self, roots: list[int], counts: list[int]) -> bytes:
        ctl = bytearray(0xBC)
        struct.pack_into("<I", ctl, 0x38, 16)            # instruction bytes
        struct.pack_into("<3I", ctl, 0x58, 8, 4, 2)      # thread group
        ctl[0xA9:0xB1] = bytes(roots)
        ctl[0xB1:0xB7] = bytes(counts)                   # ubos, ssbos, textures, separate textures, samplers, images
        struct.pack_into("<H", ctl, 0xBA, 0x1234)
        return bytes(ctl)

    def test_the_counts_come_from_the_bytes_split_reads(self) -> None:
        cases = [([8, 0, 0, 0, 0, 0, 0, 0], [3, 4, 5, 6, 7, 8]), ([1, 2, 3, 4, 5, 6, 7, 9], [16, 15, 14, 13, 12, 11]),
                 ([0] * 8, [0, 0, 0, 0, 0, 0]), ([64, 0, 0, 0, 0, 0, 0, 64], [1, 16, 128, 128, 128, 32])]
        for roots, counts in cases:
            ctl = self.control(roots, counts)
            job = split.build_job("cs", ctl, memoryview(bytes(32)))
            ubos, ssbos, textures, _size, push, septex, sepsamp = job.nvn2_cs
            done = run("layout", ctl.hex())
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            line = done.stdout.strip()
            self.assertEqual(line, f"layout ok=1 roots={','.join(map(str, roots))} ubos={ubos} ssbos={ssbos} "
                                   f"textures={textures} sep_textures={septex} sep_samplers={sepsamp} images={counts[5]}")
            self.assertEqual(roots[0] * 4, push)


if __name__ == "__main__":
    unittest.main()
