# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the NVN2 front end's textures, samplers, their pools and
handles (architecture D10, D13, D19).

Builds tests/runtime/nvn_tex_test.c with runtime/core and runtime/nvn the way
tests/test_runtime_nvn.py builds its harness, at -O1 and -O2, then runs each
case in its own process. Checks:

- storage sizes, alignments, block dimensions and level offsets, through the
  texture builder's entry points, against the values each source states (the
  table below cites one per row);
- the bytes per block of every format of NVN's enumeration, against what the
  format's name says;
- one registration reaching the back end once, with the view's parameters, and
  a sampler's whole state;
- handles that unpack, by the translator's own decoding, to the IDs packed into
  them;
- two textures copied by value that stay distinct;
- the named stops: a format no source sizes, a registration outside its pool or
  of a texture that is not initialised, IDs past their bits, packaged data for
  storage the CPU cannot see, a wrap mode past NVN's.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402
from test_runtime_nvn import (  # noqa: E402
    CORE, EXIT_FATAL, EXIT_UNIMPLEMENTED, INCLUDE, LEVELS, MODULE_SIZE, NVN, compile_flags, find_compiler,
)

TEST_SOURCE = ROOT / "tests" / "runtime" / "nvn_tex_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
NVN_H = INCLUDE / "oot2" / "nvn.h"
SHADER = ROOT / "xlate" / "third_party" / "ryujinx" / "src" / "Ryujinx.Graphics.Shader"

# NVNtextureTarget and NVNtextureFlags
T1D, T2D, T3D, T2DA, TCUBE = 0, 1, 2, 4, 8
LINEAR, SPARSE, LRT, MINIMAL = 0x10, 0x20, 0x40, 0x200

# Sources of the stated values:
#   DKB: DKB Recomp runtime/nvn_tex_probe.c at 9d8fa02, "original-SDK CPU queries" of the console's NVN driver (size,
#        block height, alignment 512; the mip offsets of RGBA8 40x128; two linear textures).
#   NXT: nxnative dm/gfx/tex/tests/t_tex.c at 865cc35 (groups L, E and S).
#   NXS: nxnative dm/gfx/tests/t_sparse_layout.c at 865cc35 (sizes of the pre-swizzled data games ship).
#   NXC: nxnative dm/gfx/nvn/nvn_tex.c at 865cc35, its comments (Tears of the Kingdom's MINIMAL_LAYOUT textures,
#        Rhythm Heaven Groove's BC7 block heights).
#   RULE: worked out by hand from deko3d's rules (source/dk_image.cpp, as nvn_tex.c cites them), the way DKB works out
#        its mip offsets: no source states a layered texture's size. A 64x64 RGBA8 2D array of three levels takes
#        0x4000 + 0x1000 + 0x400 bytes per layer, aligned to its 8-GOB block (0x1000) as 0x6000; a 32x32 cube map
#        takes six layers of 0x1000.
# A row: (source, (target, format, flags, width, height, depth, levels, stride, packaged, samples), expected), where
# packaged is byte 0 of the packaged layout plus one and expected holds any of size, align, tile (block height, block
# depth, log2 of GOBs) and offsets (of levels 0, 1, ...).
DKB_SIZES = [   # format, width, height, levels, block height, size
    (0x25, 40, 1, 6, 0, 4608), (0x25, 40, 8, 6, 0, 4608), (0x25, 40, 9, 6, 0, 6144), (0x25, 40, 10, 6, 0, 6144),
    (0x25, 40, 11, 6, 1, 6144), (0x25, 40, 16, 6, 1, 6144), (0x25, 40, 17, 6, 1, 9216), (0x25, 40, 21, 6, 1, 10240),
    (0x25, 40, 22, 6, 2, 10240), (0x25, 40, 32, 6, 2, 10240), (0x25, 40, 33, 6, 2, 16384),
    (0x25, 40, 40, 6, 2, 18944), (0x25, 40, 42, 6, 2, 18944), (0x25, 40, 43, 6, 3, 18944),
    (0x25, 40, 64, 7, 3, 19456), (0x25, 40, 65, 7, 3, 31744), (0x25, 40, 85, 7, 3, 37376),
    (0x25, 40, 86, 7, 4, 37376), (0x25, 40, 128, 8, 4, 37888), (0x25, 40, 129, 8, 4, 62464),
    (0x48, 160, 4, 8, 0, 12288), (0x48, 160, 32, 8, 0, 12288), (0x48, 160, 36, 8, 0, 17408),
    (0x48, 160, 40, 8, 0, 17408), (0x48, 160, 44, 8, 1, 17408), (0x48, 160, 64, 8, 1, 17408),
    (0x48, 160, 68, 8, 1, 30208), (0x48, 160, 84, 8, 1, 30208), (0x48, 160, 88, 8, 2, 30208),
    (0x48, 160, 128, 8, 2, 30208), (0x48, 160, 132, 8, 2, 57344), (0x48, 160, 160, 8, 2, 57344),
    (0x48, 160, 168, 8, 2, 57344), (0x48, 160, 172, 8, 3, 57344), (0x48, 160, 256, 9, 3, 57856),
    (0x48, 160, 260, 9, 3, 112128), (0x48, 160, 340, 9, 3, 113152), (0x48, 160, 344, 9, 4, 113152),
    (0x48, 160, 512, 10, 4, 113664), (0x48, 160, 516, 10, 4, 222208),
]
YOSHI = [   # NXS: sparse 2D textures of one level, width, height, format, size
    (384, 384, 0x48, 0x40000), (384, 384, 0x45, 0x20000), (960, 384, 0x48, 0x80000), (480, 480, 0x48, 0x40000),
    (900, 420, 0x48, 0x80000), (1920, 912, 0x48, 0x200000), (1920, 864, 0x48, 0x200000), (1440, 480, 0x48, 0xC0000),
    (840, 448, 0x48, 0x80000), (840, 840, 0x48, 0x100000), (1350, 270, 0x9C, 0x210000), (240, 270, 0x9C, 0x60000),
    (720, 270, 0x9C, 0x120000), (1080, 174, 0x9C, 0x120000), (2010, 390, 0x9C, 0x400000),
]
ASTC_S = (T2D, 0x8B, SPARSE, 1024, 1024, 1, 8, 0, 4, 0)
ROWS = [
    *(("DKB", (T2D, f, 0, w, h, 1, lv, 0, 0, 1), {"size": size, "align": 512, "tile": (bh, 0)})
      for f, w, h, lv, bh, size in DKB_SIZES),
    ("DKB", (T2D, 0x25, 0, 40, 128, 1, 8, 0, 0, 1),
     {"offsets": [0, 24576, 32768, 34816, 35840, 36352, 36864, 37376, 37888]}),
    ("DKB", (T2D, 0x25, LRT, 40, 4, 1, 1, 0, 0, 1), {"size": 1024, "align": 128}),
    ("DKB", (T2D, 0x25, LINEAR, 40, 4, 1, 1, 0, 0, 1), {"size": 640, "align": 32}),
    ("NXT", (T2D, 0x25, 0x44, 16, 16, 1, 1, 128, 0, 0), {"size": 2048, "align": 128}),
    ("NXT", (T2D, 0x25, 0x44, 16, 16, 1, 1, 0, 0, 0), {"size": 2048, "align": 128}),
    ("NXT", (T2D, 0x25, 0x54, 16, 16, 1, 1, 0, 0, 0), {"size": 2048, "align": 128}),
    ("NXT", (T2D, 0x25, LINEAR, 16, 16, 1, 1, 0, 0, 0), {"size": 1024, "align": 32}),
    ("NXT", (T2D, 0x25, 0x44, 16, 16, 1, 1, 256, 0, 0), {"size": 4096}),
    ("NXT", (T3D, 0x25, 0, 32, 16, 20, 1, 0, 0, 0), {"size": 65536, "tile": (0, 4)}),
    ("NXT", (T3D, 0x29, 0, 32, 32, 32, 1, 0, 0, 0), {"size": 262144, "tile": (0, 4)}),
    ("NXT", (T3D, 0x25, 0, 32, 32, 32, 3, 0, 0x13, 0), {"size": 151552, "offsets": [0, 131072, 147456]}),
    ("NXT", (T3D, 0x25, 0, 32, 32, 32, 3, 0, 1, 0), {"size": 151552, "offsets": [0, 131072, 147456]}),
    ("NXT", ASTC_S, {"size": 0xE0000, "align": 0x10000,
                     "offsets": [0, 0x90000, 0xD0000, 0xDB000, 0xDE000, 0xDEC00, 0xDF000, 0xDF200]}),
    ("NXT", (T2D, 0x8B, 0, 1024, 1024, 1, 8, 0, 4, 0), {"size": 0xBC400, "offsets": [0, 0x81000]}),
    *(("NXS", (T2D, f, SPARSE, w, h, 1, 1, 0, 0, 0), {"size": size}) for w, h, f, size in YOSHI),
    ("NXS", (T2D, 0x48, 0, 384, 384, 1, 1, 0, 0, 0), {"size": 0x30000}),
    ("NXS", (T2D, 0x7D, SPARSE, 1024, 1024, 1, 11, 0, 0, 0), {"offsets": [0, 0x90000]}),
    ("NXS", (T3D, 0x42, SPARSE, 512, 256, 368, 1, 0, 0, 0), {"size": 0x1700000, "tile": (1, 4)}),
    ("NXS", (T3D, 0x50, SPARSE, 512, 256, 368, 1, 0, 0, 0), {"size": 0x2E00000, "tile": (1, 4)}),
    ("NXC", (T2D, 0x3B, MINIMAL, 320, 320, 1, 1, 0, 0, 0), {"size": 204800, "tile": (0, 0)}),
    ("NXC", (T2D, 0x3C, MINIMAL, 320, 320, 1, 1, 0, 0, 0), {"size": 204800, "tile": (0, 0)}),
    ("NXC", (T2D, 0x3B, 0, 320, 320, 1, 1, 0, 0, 0), {"size": 245760, "tile": (4, 0)}),
    ("NXC", (T2D, 0x4D, 0, 672, 160, 1, 1, 0, 0, 0), {"tile": (2, 0)}),
    ("NXC", (T2D, 0x4D, 0, 1232, 160, 1, 1, 0, 0, 0), {"tile": (2, 0)}),
    ("RULE", (T2DA, 0x25, 0, 64, 64, 2, 3, 0, 0, 0), {"size": 0xC000, "offsets": [0, 0x4000, 0x5000, 0x5400]}),
    ("RULE", (TCUBE, 0x25, 0, 32, 32, 1, 1, 0, 0, 0), {"size": 0x6000, "tile": (2, 0)}),
]

# Formats past NVN's enumeration: no source sizes them.
UNSIZED = (0x00, 0x9D, 0x9E)


def format_names() -> dict[int, str]:
    """NVNformat's names by value, from nvn.h."""
    text = NVN_H.read_text(encoding="utf-8")
    return {int(v, 16): n for n, v in re.findall(r"NVN_FORMAT_(\w+) = (0x[0-9A-F]+),", text) if n not in ("NONE", "LARGE")}


def size_from_name(name: str) -> tuple[int, int, int]:
    """Bytes per block, block width and height, read from a format's name alone."""
    m = re.fullmatch(r"RGBA_ASTC_(\d+)x(\d+)(_SRGB)?", name)
    if m:
        return 16, int(m[1]), int(m[2])
    if re.search(r"DXT1|RGTC1", name):
        return 8, 4, 4
    if re.search(r"DXT[35]|RGTC2|BPTC", name):
        return 16, 4, 4
    packed = {"STENCIL8": 1, "DEPTH16": 2, "DEPTH24": 4, "DEPTH32F": 4, "DEPTH24_STENCIL8": 4, "DEPTH32F_STENCIL8": 8,
              "R11G11B10F": 4, "RGB9E5F": 4}
    if name in packed:
        return packed[name], 1, 1
    m = re.fullmatch(r"(RGBA|RGB|A1BGR|BGR)(4|5A1|565|5|10A2)\w*", name)   # 16-bit packed colours and 10-10-10-2
    if m:
        return (4 if m[2] == "10A2" else 2), 1, 1
    m = re.fullmatch(r"(RGBA|RGBX|BGRA|BGRX|RGB|RG|R)(8|16|32)\w*", name)
    assert m, name
    return len(m[1]) * int(m[2]) // 8, 1, 1


def translator_decoding() -> tuple[int, int]:
    """The mask and the shift by which the translated shaders take a texture ID and a sampler ID out of a handle."""
    bindless = (SHADER / "Translation" / "Optimizations" / "BindlessElimination.cs").read_text(encoding="utf-8")
    mask = re.search(r"Instruction\.BitwiseAnd, textureHandle, bindlessHandle, OperandHelper\.Const\((0x[0-9a-f]+)\)",
                     bindless)
    shift = re.search(r"Instruction\.ShiftRightU32, samplerHandle, bindlessHandle, OperandHelper\.Const\((\d+)\)",
                      bindless)
    assert mask and shift, "the translator's handle decoding moved: GenerateBindlessAccess"
    handle = (SHADER / "TextureHandle.cs").read_text(encoding="utf-8")
    assert f"return packedId & {mask[1]};" in handle and f"return (packedId >> {shift[1]}) & 0xfff;" in handle
    return int(mask[1], 16), int(shift[1])


class FormatNameTest(unittest.TestCase):
    def test_names_cover_the_enumeration(self) -> None:
        names = format_names()
        self.assertEqual(sorted(names), list(range(0x01, 0x9D)))


class NvnTexTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-nvn-tex-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c")) + sorted(str(p) for p in NVN.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"nvn_tex_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*compile_flags(cc), level, *extra, "-I", str(CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case, *args], capture_output=True, text=True, timeout=120)

    def assert_passes(self, case: str, *args: str) -> list[subprocess.CompletedProcess]:
        runs = []
        for level in LEVELS:
            with self.subTest(level=level, case=case):
                done = self.run_case(level, case, *args)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                runs.append(done)
        return runs

    def assert_stops(self, case: str, code: int, text: str, *args: str) -> None:
        for level in LEVELS:
            with self.subTest(level=level, case=case, args=args):
                done = self.run_case(level, case, *args)
                self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                self.assertIn(text, done.stderr)
                self.assertEqual(done.stderr.count("FATAL"), 1, done.stderr)

    # ---- storage

    def test_storage_equals_the_stated_values(self) -> None:
        args = [":".join(str(v) for v in desc) for _, desc, _ in ROWS]
        for done in self.assert_passes("layout", *args):
            got = {}
            for m in re.finditer(r"^row (\d+) size (\S+) align (\S+) tile (\d+) (\d+) offsets(( \S+)*)$", done.stdout,
                                 re.M):
                got[int(m[1])] = {"size": int(m[2], 16), "align": int(m[3], 16), "tile": (int(m[4]), int(m[5])),
                                  "offsets": [int(o, 16) for o in m[6].split()]}
            self.assertEqual(len(got), len(ROWS), done.stdout)
            for i, (source, desc, want) in enumerate(ROWS):
                with self.subTest(row=i, source=source, desc=desc):
                    for key, value in want.items():
                        have = got[i][key][:len(value)] if key == "offsets" else got[i][key]
                        self.assertEqual(have, value, key)

    def test_every_format_of_the_enumeration_is_sized_by_its_name(self) -> None:
        names = format_names()
        done = self.assert_passes("formats")[0]
        got = {int(m[1], 16): tuple(map(int, m.groups()[1:]))
               for m in re.finditer(r"^format (0x[0-9a-f]+|0) (\d+) (\d+) (\d+)$", done.stdout, re.M)}
        for value, name in sorted(names.items()):
            with self.subTest(format=name):
                self.assertEqual(got[value], size_from_name(name))
        for value in UNSIZED:
            self.assertEqual(got[value][0], 0)

    def test_a_format_no_source_sizes_stops(self) -> None:
        for value in UNSIZED:
            text = f"FATAL: unimplemented nvn: nvnTextureBuilderGetStorageSize: format {value:#x}, which no source sizes"
            self.assert_stops("unknown_format", EXIT_UNIMPLEMENTED, text, f"{value:#x}")

    # ---- textures, views, copies

    def test_textures_and_views(self) -> None:
        self.assert_passes("texture")

    def test_textures_copied_by_value_stay_distinct(self) -> None:
        for done in self.assert_passes("copies"):
            serials = [int(s) for s in re.search(r"^serials (\d+) (\d+) (\d+)$", done.stdout, re.M).groups()]
            self.assertEqual(len(set(serials)), 3)

    # ---- pools and registration

    def test_a_registration_reaches_the_back_end_once(self) -> None:
        self.assert_passes("register")

    def test_bad_registrations_stop(self) -> None:
        cases = {
            "outside": (EXIT_FATAL, "FATAL: nvn: nvnTexturePoolRegisterTexture at ID 8, outside the pool's 8"),
            "negative": (EXIT_FATAL, "FATAL: nvn: nvnTexturePoolRegisterTexture at ID -1, outside the pool's 8"),
            "finalized": (EXIT_FATAL, "FATAL: nvn: nvnTexturePoolRegisterTexture of a texture that is not initialised"),
            "pool": (EXIT_FATAL, "FATAL: nvn: nvnTexturePoolRegisterTexture into a pool that is not initialised"),
            "packaged": (EXIT_UNIMPLEMENTED,
                         "FATAL: unimplemented nvn: nvnTextureInitialize: packaged texture data for storage the CPU "
                         "cannot see"),
        }
        for what, (code, text) in cases.items():
            self.assert_stops("register_bad", code, text, what)

    # ---- handles

    def test_handles_unpack_to_their_ids_as_the_translator_decodes(self) -> None:
        mask, shift = translator_decoding()
        ids = [(0, 0), (1, 0), (0, 1), (0xFFFFF, 0xFFF), (0x12345, 0xABC), (256, 255), (1048575, 0), (0, 4095)]
        for done in self.assert_passes("handles", *(f"{t}:{s}" for t, s in ids)):
            rows = [m.groups() for m in re.finditer(r"^handle (\d+) (\d+) (\S+) (\S+) (\S+) (\S+) (\S+)$",
                                                    done.stdout, re.M)]
            self.assertEqual(len(rows), len(ids))
            for (t, s), row in zip(ids, rows):
                with self.subTest(texture=t, sampler=s):
                    h, sep_t, sep_s, fetch, image = (int(v, 16) for v in row[2:])
                    self.assertEqual((int(row[0]), int(row[1])), (t, s))
                    self.assertEqual(((h & 0xFFFFFFFF) & mask, (h & 0xFFFFFFFF) >> shift), (t, s))
                    self.assertEqual(h >> 32, 0)
                    self.assertEqual((sep_t | sep_s, sep_t, fetch, image), (h, t, t, t))
            self.assertIn(f"unpack {mask:#x} {shift}", done.stdout)

    def test_ids_past_their_bits_stop(self) -> None:
        cases = {
            "sampler": "FATAL: nvn: nvnDeviceGetTextureHandle: ID 4096 does not fit below 4096",
            "texture": "FATAL: nvn: nvnDeviceGetTextureHandle: ID 1048576 does not fit below 1048576",
            "image": "FATAL: nvn: nvnDeviceGetImageHandle: ID -1 does not fit below 1048576",
        }
        for which, text in cases.items():
            self.assert_stops("handle_range", EXIT_FATAL, text, which)

    # ---- samplers

    def test_samplers(self) -> None:
        self.assert_passes("sampler")

    def test_a_wrap_mode_past_nvns_stops(self) -> None:
        text = "FATAL: unimplemented nvn: nvnSamplerBuilderSetWrapMode: value 8, past the last one NVN defines (7)"
        self.assert_stops("sampler_bad", EXIT_UNIMPLEMENTED, text)


if __name__ == "__main__":
    unittest.main()
