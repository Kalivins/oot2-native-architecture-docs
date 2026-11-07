# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The GPU ASTC decode against the CPU reference (runtime/gpu/shaders/astc_decode.comp, tests/runtime/astc_ref.c;
architecture D19), through tests/runtime/astc_decode_test.c.

The program makes its own Vulkan 1.3 instance and device and links no runtime code, so these tests run on any compute
device (the desktop's RTX 3060 Ti, the laptop's GTX 1060). For each of the 14 block sizes NVN has ASTC formats for, in the
linear and the sRGB profile, it decodes seeded blocks (random blocks the reference takes without error or HDR endpoint mode,
and void extents) on the device and compares every texel with the reference, byte for byte.

- the shader as built equals the reference on every block size in both profiles, and stays within one step of astcenc's
  top-eight-bits reading of the same 16-bit value;
- counter-run: a copy of the shader that reads the weight grid transposed for one block size (8x5) is caught, and only
  that size is named; a copy that expands sRGB endpoints the linear way is caught in the sRGB profile and nowhere else.

Without clang or the Vulkan SDK the tests skip, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set; without a usable
device they skip, unless OOT2_REQUIRE_GPU is set.

Run: python -m unittest discover -s tests -p test_runtime_vk_astc.py -v
"""

from __future__ import annotations

import atexit
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXE, ROOT, case_env, clang_or_skip, compile_now, vulkan_or_skip

SHADER = ROOT / "runtime" / "gpu" / "shaders" / "astc_decode.comp"
SOURCES = [ROOT / "tests" / "runtime" / "astc_decode_test.c", ROOT / "tests" / "runtime" / "astc_ref.c"]
EXIT_NO_DEVICE = 3
SIZES = ["4x4", "5x4", "5x5", "6x5", "6x6", "8x5", "8x6", "8x8", "10x5", "10x6", "10x8", "10x10", "12x10", "12x12"]

TRANSPOSE_FROM = "const uint v0 = jt * size.x + js;"
TRANSPOSE_TO = "const uint v0 = (block_dims.x == 8u && block_dims.y == 5u) ? js * size.y + jt : jt * size.x + js;"
LINEAR_FROM = "return srgb != 0u ? ((value << 8u) | 0x80u) : value * 0x101;"
LINEAR_TO = "return value * 0x101;"

_state: dict[str, object] = {}


def tools() -> tuple[Path, str, Path]:
    """The harness program, glslang, and a folder for compiled shaders; built once."""
    if "exe" not in _state:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        tmp = tempfile.TemporaryDirectory(prefix="oot2-astc-", ignore_cleanup_errors=True)
        atexit.register(tmp.cleanup)
        _state["tmp"] = tmp
        exe = Path(tmp.name) / f"astc_decode_test{EXE}"
        cmd = [cc, "-std=gnu11", "-O2", "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", "-I", str(include),
               "-I", str(include / "Volk"), *map(str, SOURCES), "-o", str(exe)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"build of astc_decode_test failed:\n{done.stdout}{done.stderr}")
        _state.update(exe=exe, glslang=glslang, dir=Path(tmp.name))
    return _state["exe"], _state["glslang"], _state["dir"]  # type: ignore[return-value]


def shader_spv(name: str, text: str | None = None) -> Path:
    """The SPIR-V of the shader (or of `text`, a changed copy of it) as the build compiles it."""
    _, glslang, out = tools()
    src = SHADER
    if text is not None:
        src = out / f"{name}.comp"
        src.write_text(text, encoding="utf-8", newline="\n")
    spv = out / f"{name}.spv"
    spv.write_bytes(compile_now(glslang, src, out))
    return spv


def run(spv: Path) -> subprocess.CompletedProcess:
    exe, _, _ = tools()
    return subprocess.run([str(exe), str(spv)], capture_output=True, text=True, timeout=600, env=case_env())


def mutated(name: str, old: str, new: str) -> Path:
    text = SHADER.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"the shader no longer has exactly one {old!r}"
    return shader_spv(name, text.replace(old, new))


class AstcDecodeTest(unittest.TestCase):
    def device(self, spv: Path) -> subprocess.CompletedProcess:
        done = run(spv)
        if done.returncode == EXIT_NO_DEVICE:
            why = f"no usable Vulkan device here ({done.stdout.strip()})"
            if os.environ.get("OOT2_REQUIRE_GPU"):
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but there is {why}")
            self.skipTest(why)
        return done

    @staticmethod
    def differing(done: subprocess.CompletedProcess) -> list[str]:
        return re.findall(r"^DIFFERS (\S+ \S+)$", done.stdout, re.M)

    def test_the_gpu_decode_equals_the_reference_on_every_block_size_in_both_profiles(self) -> None:
        done = self.device(shader_spv("astc_decode"))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-2000:])
        self.assertIn("ok astc_decode", done.stdout)
        lines = re.findall(r"^(\d+x\d+) (linear|sRGB) *: (\d+) blocks \((\d+) void extents\), (\d+) texels, (\d+) differ from the "
                           r"reference, (\d+) further than one step", done.stdout, re.M)
        self.assertEqual(len(lines), 28, done.stdout)
        self.assertEqual({(s, p) for s, p, *_ in lines}, {(s, p) for s in SIZES for p in ("linear", "sRGB")})
        for size, profile, blocks, voids, texels, differ, far in lines:
            self.assertEqual((int(differ), int(far)), (0, 0), f"{size} {profile}")
            self.assertGreater(int(blocks) - int(voids), 500, f"{size} {profile}: too few random blocks")
            self.assertGreater(int(voids), 0)
            self.assertGreater(int(texels), 0)

    def test_a_weight_grid_read_transposed_for_one_block_size_is_caught(self) -> None:
        done = self.device(mutated("astc_transposed", TRANSPOSE_FROM, TRANSPOSE_TO))
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr[-2000:])
        self.assertEqual(sorted(self.differing(done)), ["8x5 linear", "8x5 sRGB"], done.stdout)

    def test_srgb_endpoints_expanded_the_linear_way_are_caught_in_the_srgb_profile_only(self) -> None:
        done = self.device(mutated("astc_linear_srgb", LINEAR_FROM, LINEAR_TO))
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr[-2000:])
        found = self.differing(done)
        self.assertEqual(sorted(found), sorted(f"{s} sRGB" for s in SIZES), done.stdout)


if __name__ == "__main__":
    unittest.main()
