# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the Vulkan back end's module (runtime/gpu, runtime/include/oot2/gpu.h and gpu_vk.h) that need no GPU: the
runtime's own shaders, the back-end choice and Volk; the timeline and its completion thread over a test's counter; the
Vulkan settings and the choice of a device from synthetic candidates; the device-loss report's file. The device's own
tests are tests/test_runtime_vk.py; the replay's are tests/test_runtime_vk_replay.py.

The shaders' part needs glslang only: tools/build_game.py compiles runtime/gpu/shaders with it and writes the C file
that embeds them, and the tests check that the embedded words are what the source compiles to now, including after a
shader is edited (the build compiles every shader at every build of the library, so an embedded shader cannot fall
behind its source), and that a shader which does not compile, or has a name that is not a C identifier, is named.
spirv-val, where it is found (PATH, or the Vulkan SDK's Bin folder), validates each shader once more; the build has
already run glslang's own validator.

The rest builds tests/runtime/gpu_test.c with runtime/core, runtime/nvn and runtime/gpu, the Vulkan headers and Volk
on the include path, and that C file, at -O1 and -O2 with -Werror, then runs each case in its own process (build_program,
which tests/test_runtime_vk.py uses too).

Needs clang (OOT2_CC, else PATH) and a Vulkan SDK with the headers, Volk and glslang (VULKAN_SDK; tools/fetch_vulkan.py
lays one out). Without them the tests skip, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set, as continuous
integration sets them.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_game  # noqa: E402
import layout  # noqa: E402

RUNTIME = ROOT / "runtime"
TEST_SOURCE = ROOT / "tests" / "runtime" / "gpu_test.c"
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000
EXIT_FATAL = 70
EXE = ".exe" if os.name == "nt" else ""
# What a start that asks for the Vulkan back end on a machine that cannot run it stops with (gpu_vk_open).
NO_VULKAN = ("there is no Vulkan loader", "there is no Vulkan device", "no Vulkan device has every item",
             "the Vulkan loader is version")
GLUE_ARRAY = re.compile(r"static const uint32_t spirv_(\w+)\[(\d+)\] = \{(.*?)\};", re.S)


def vulkan_or_skip() -> tuple[Path, str]:
    """The Vulkan SDK's include folder and glslang, or SkipTest (an error when OOT2_REQUIRE_VULKAN is set)."""
    include = build_game.vulkan_include()
    glslang = build_game.find_glslang(include) if include else None
    if include is None or glslang is None:
        why = "no Vulkan SDK with the headers, Volk and glslang (set VULKAN_SDK; tools/fetch_vulkan.py lays one out)"
        if os.environ.get("OOT2_REQUIRE_VULKAN"):
            raise RuntimeError(f"OOT2_REQUIRE_VULKAN is set but there is {why}")
        raise unittest.SkipTest(why)
    return include, glslang


def clang_or_skip() -> str:
    """clang (OOT2_CC, else PATH), or SkipTest (an error when OOT2_REQUIRE_CLANG is set)."""
    cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
    if not cc:
        if os.environ.get("OOT2_REQUIRE_CLANG"):
            raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
        raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
    return cc


def build_program(cc: str, include: Path, glslang: str, source: Path, out: Path, levels=LEVELS) -> dict[str, Path]:
    """source linked with runtime/core, runtime/nvn, runtime/gpu and the shaders' C file, one program per level."""
    layout.write(layout.place({"main": MODULE_SIZE}), out)
    glue = build_game.write_shader_glue(RUNTIME, out, glslang)
    sources = [str(source)]
    for part in ("core", "nvn", "gpu"):
        sources += sorted(str(p) for p in (RUNTIME / part).glob("*.c"))
    sources.append(str(glue))
    extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread", "-ldl"]
    exes = {}
    for level in levels:
        exe = out / f"{source.stem}{level}{EXE}"
        cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror",
               "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(RUNTIME / "nvn"),
               "-I", str(out), "-I", str(include), "-I", str(include / "Volk"), *sources, *extra, "-o", str(exe)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"build of {source.name} at {level} failed:\n{done.stdout}{done.stderr}")
        exes[level] = exe
    return exes


def case_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """This process's environment less census mode, the back-end choice and the Vulkan settings (each opt-in per case),
    with third-party implicit Vulkan layers kept out (overlays and recorders flood the messenger), plus env."""
    full = {k: v for k, v in os.environ.items()
            if not k.startswith(("OOT2_CENSUS", "OOT2_VK_")) and k != "OOT2_GFX"}
    full["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"
    full.update(env or {})
    return full


def fatal_lines(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if line.startswith("FATAL")]


def compile_now(glslang: str, source: Path, out: Path) -> bytes:
    """source compiled by glslang with the build's flags, independently of the build."""
    spv = out / (source.stem + ".spv")
    done = subprocess.run([glslang, *build_game.GLSLANG_FLAGS, "-o", str(spv), str(source)], capture_output=True,
                          text=True)
    if done.returncode:
        raise AssertionError(f"glslang failed on {source}:\n{done.stdout}{done.stderr}")
    return spv.read_bytes()


def embedded(glue: Path) -> dict[str, bytes]:
    """The SPIR-V each array of the shaders' C file holds, by shader name."""
    found = {}
    for m in GLUE_ARRAY.finditer(glue.read_text(encoding="utf-8")):
        words = [int(w, 16) for w in re.findall(r"0x([0-9a-f]{8})u", m.group(3))]
        assert len(words) == int(m.group(2)), m.group(1)
        found[m.group(1)] = b"".join(w.to_bytes(4, "little") for w in words)
    return found


def copy_shaders(dst: Path) -> Path:
    """A runtime folder holding a copy of runtime/gpu/shaders only, which is all the shaders' build reads."""
    shutil.copytree(RUNTIME / build_game.SHADER_DIR, dst / build_game.SHADER_DIR)
    return dst


class ShaderBuildTest(unittest.TestCase):
    """The runtime's shaders through tools/build_game.py, with glslang only."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.include, cls.glslang = vulkan_or_skip()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2-shaders-")
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_every_shader_is_compiled_and_embedded_in_name_order(self) -> None:
        sources = build_game.runtime_shaders(RUNTIME)
        self.assertIn("fill.comp", [s.name for s in sources])
        glue = build_game.write_shader_glue(RUNTIME, self.dir / "build", self.glslang)
        self.assertEqual(glue, build_game.shader_glue_path(self.dir / "build"))
        words = embedded(glue)
        self.assertEqual(list(words), [s.stem for s in sources])
        (self.dir / "now").mkdir()
        for s in sources:
            self.assertEqual(words[s.stem], compile_now(self.glslang, s, self.dir / "now"), s.name)
        text = glue.read_text(encoding="utf-8")
        self.assertIn('{"fill", spirv_fill, sizeof spirv_fill},', text)
        self.assertIn(f"const int gpu_shader_count = {len(sources)};", text)

    def test_an_edited_shader_is_embedded_again(self) -> None:
        runtime = copy_shaders(self.dir / "runtime")
        out = self.dir / "build"
        glue = build_game.write_shader_glue(runtime, out, self.glslang)
        before = embedded(glue)["fill"]
        stamp = glue.stat().st_mtime_ns
        build_game.write_shader_glue(runtime, out, self.glslang)
        self.assertEqual(glue.stat().st_mtime_ns, stamp)   # nothing changed: the C file is not touched
        source = runtime / build_game.SHADER_DIR / "fill.comp"
        text = source.read_text(encoding="utf-8")
        self.assertIn("local_size_x = 64", text)
        source.write_text(text.replace("local_size_x = 64", "local_size_x = 32"), encoding="utf-8")
        build_game.write_shader_glue(runtime, out, self.glslang)
        after = embedded(glue)["fill"]
        self.assertNotEqual(after, before)
        (self.dir / "now").mkdir()
        self.assertEqual(after, compile_now(self.glslang, source, self.dir / "now"))

    def test_a_shader_that_does_not_compile_is_named(self) -> None:
        runtime = copy_shaders(self.dir / "runtime")
        (runtime / build_game.SHADER_DIR / "broken.comp").write_text(
            "#version 460\nlayout(local_size_x = 1) in;\nvoid main() { undeclared = 1; }\n", encoding="utf-8")
        with self.assertRaisesRegex(build_game.BuildError, r"could not compile or validate shader .*broken\.comp"):
            build_game.write_shader_glue(runtime, self.dir / "build", self.glslang)
        self.assertFalse((self.dir / "build" / build_game.SPIRV_DIR / "broken.spv").exists())

    def test_a_shader_name_must_be_a_c_identifier(self) -> None:
        runtime = copy_shaders(self.dir / "runtime")
        shutil.copy(runtime / build_game.SHADER_DIR / "fill.comp", runtime / build_game.SHADER_DIR / "Fill-2.comp")
        with self.assertRaisesRegex(build_game.BuildError, r"Fill-2\.comp: its name must be"):
            build_game.write_shader_glue(runtime, self.dir / "build", self.glslang)

    def test_spirv_val_accepts_every_shader(self) -> None:
        exe = ".exe" if os.name == "nt" else ""
        found = [str(p) for p in (self.include.parent / b / f"spirv-val{exe}" for b in ("Bin", "bin")) if p.is_file()]
        val = found[0] if found else shutil.which("spirv-val")
        if not val:
            self.skipTest("no spirv-val (PATH or the Vulkan SDK's Bin folder); the build ran glslang's own validator")
        for s in build_game.runtime_shaders(RUNTIME):
            compile_now(self.glslang, s, self.dir)
            done = subprocess.run([val, "--target-env", "vulkan1.3", str(self.dir / (s.stem + ".spv"))],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"{s.name}: {done.stdout}{done.stderr}")


class GpuTest(unittest.TestCase):
    """gpu_test.c with the runtime's core, NVN2 front end and gpu module, built against the Vulkan headers and Volk."""

    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = clang_or_skip()
        cls.include, cls.glslang = vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-gpu-", ignore_cleanup_errors=True)
        cls.exes = build_program(cc, cls.include, cls.glslang, TEST_SOURCE, Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, *args: str, env: dict[str, str] | None = None,
                 timeout: int = 60) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case, *args], capture_output=True, text=True, timeout=timeout,
                              env=case_env(env))

    def test_passing_cases(self) -> None:
        for case in ("lookup", "select_null", "volk", "settings", "choose"):
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)

    def test_the_null_back_end_is_chosen_by_default_and_logged(self) -> None:
        done = self.run_case(LEVELS[0], "select_null")
        self.assertIn("[rt] gfx: the null back end (the default)", done.stderr)

    def test_the_embedded_shaders_are_their_sources_compiled_now(self) -> None:
        sources = build_game.runtime_shaders(RUNTIME)
        work = Path(self.tmp.name) / "now"
        work.mkdir(exist_ok=True)
        expected = {s.stem: compile_now(self.glslang, s, work) for s in sources}
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "shaders")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                rows = [line.split() for line in done.stdout.splitlines() if line.startswith("shader ")]
                self.assertEqual([r[1] for r in rows], [s.stem for s in sources])
                for _, name, size, sha in rows:
                    self.assertEqual(int(size), len(expected[name]), name)
                    self.assertEqual(sha, hashlib.sha256(expected[name]).hexdigest(), name)
                self.assertIn(f"ok shaders {len(sources)}", done.stdout)

    def test_the_vulkan_back_end_opens_its_device_or_stops_by_name(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "select_vulkan")
                self.assertIn("[rt] gfx: the vulkan back end (--gfx)", done.stderr)
                if done.returncode == 0:   # a machine with a device that has every item
                    self.assertIn("gfx_select returned for vulkan", done.stdout)
                    self.assertRegex(done.stderr, r"gfx: vulkan: .* device made with the \d+ items' features")
                    self.assertEqual(fatal_lines(done.stderr), [])
                    continue
                self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                fatal = fatal_lines(done.stderr)
                self.assertEqual(len(fatal), 1, done.stderr)
                self.assertTrue(fatal[0].startswith("FATAL: gfx: vulkan: "), fatal[0])
                self.assertTrue(any(why in fatal[0] for why in NO_VULKAN), fatal[0])
                self.assertNotIn("returned", done.stdout)

    def test_a_vulkan_setting_outside_its_values_stops_the_start_by_name(self) -> None:
        for env, line in (({"OOT2_VK_VALIDATION": "loud"},
                           'gfx: vulkan: OOT2_VK_VALIDATION is "loud"; the values are off, on, sync and gpu'),
                          ({"OOT2_VK_FAULT_REPORT": "maybe"},
                           'gfx: vulkan: OOT2_VK_FAULT_REPORT is "maybe"; the values are on and off')):
            with self.subTest(env=env):
                done = self.run_case(LEVELS[0], "select_vulkan", env=env)
                self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                self.assertEqual(fatal_lines(done.stderr), [f"FATAL: {line}"])

    def test_an_unknown_back_end_is_a_fatal_that_lists_the_names(self) -> None:
        done = self.run_case(LEVELS[1], "select_unknown")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn('FATAL: gfx: no back end is named "metal" (asked for by OOT2_GFX); the names are null and vulkan',
                      done.stderr)

    def test_settings_refuse_a_value_outside_their_values_by_name(self) -> None:
        done = self.run_case(LEVELS[0], "settings")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        refused = [line[len("refused: "):] for line in done.stdout.splitlines() if line.startswith("refused: ")]
        self.assertEqual(refused, [
            'OOT2_VK_VALIDATION is "On"; the values are off, on, sync and gpu',
            'OOT2_VK_VALIDATION is "1"; the values are off, on, sync and gpu',
            'OOT2_VK_FAULT_REPORT is "1"; the values are on and off',
            'OOT2_VK_FAULT_REPORT is "yes"; the values are on and off',
            "OOT2_VK_DEVICE is 399 characters long; at most 255",
        ])

    def test_the_device_choice_never_takes_a_device_that_lacks_an_item(self) -> None:
        rtx, gtx = "NVIDIA GeForce RTX 3060 Ti", "NVIDIA GeForce GTX 1060 6GB"
        mesh, atomic = "mesh and task shaders", "32-bit float atomic add on storage buffers"
        expected = {
            "discrete-first": "1",
            "lacking-discrete": "1",
            "cpu-only": "0",
            "none-usable": f"-1 no Vulkan device has every item the renderer requires: device 0 {gtx} (discrete) lacks: "
                           f"{mesh}; device 1 llvmpipe (CPU) lacks: {mesh}, {atomic}",
            "index": "0",
            "index-past": f"-1 OOT2_VK_DEVICE=5, and there are 2 devices: 0 Integrated Thing, 1 {rtx}",
            "name": "1",
            "name-lacking": f"-1 OOT2_VK_DEVICE=gtx names a device the renderer cannot use: device 0 {gtx} (discrete) "
                            f"lacks: {mesh}, {atomic}",
            "name-none": f"-1 OOT2_VK_DEVICE=radeon, and no device's name contains it: 0 Integrated Thing, 1 {rtx}",
            "name-second-usable": "1",
            "index-lacking": f"-1 OOT2_VK_DEVICE=0 names a device the renderer cannot use: device 0 {rtx} (discrete) "
                             f"lacks: {mesh}",
            "unusable": "-1 no Vulkan device has every item the renderer requires: device 0 Queueless (discrete) has "
                        "no use: no queue family with graphics and compute",
            "empty": "-1 no Vulkan device",
        }
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "choose")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                got = dict(line[len("choose "):].split(": ", 1) for line in done.stdout.splitlines()
                           if line.startswith("choose "))
                self.assertEqual(got, expected)

    def test_the_timeline_runs_effects_in_order_before_it_publishes_their_value(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "sync_order", timeout=180)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertRegex(done.stdout, r"ok sync_order values 10000 effects \d+ waits \d+")
                self.assertEqual(fatal_lines(done.stderr), [])

    def test_a_fence_wait_times_out_and_a_later_one_succeeds(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "sync_wait")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok sync_wait timed out after", done.stdout)

    def test_the_completion_thread_is_not_stopped_with_work_in_flight(self) -> None:
        done = self.run_case(LEVELS[0], "sync_stop_in_flight")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertEqual(fatal_lines(done.stderr),
                         ["FATAL: gpu: the completion thread is stopped with values 1 to 1 in flight"])

    def test_the_report_goes_to_a_new_file_in_the_logs_folder_with_the_binary_beside_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-report-") as d:
            done = self.run_case(LEVELS[0], "report_write", d)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            paths = [Path(line[len("report "):]) for line in done.stdout.splitlines() if line.startswith("report ")]
            self.assertEqual(len(paths), 2)
            for p in paths:
                self.assertEqual(p.parent.resolve(), (Path(d) / "logs").resolve())
                self.assertRegex(p.name, r"^device-lost-\d{8}-\d{6}(-\d+)?\.txt$")
            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual(paths[0].read_bytes(), b"first report\n")
            self.assertEqual(paths[1].read_bytes(), b"second report\n")
            self.assertEqual(paths[0].with_suffix(".bin").read_bytes(), bytes((i * 7 + 3) & 0xFF for i in range(300)))
            self.assertFalse(paths[1].with_suffix(".bin").exists())
            self.assertEqual(sorted(p.name for p in (Path(d) / "logs").iterdir()),
                             sorted([paths[0].name, paths[0].with_suffix(".bin").name, paths[1].name]))
            self.assertRegex(done.stdout, r"refused: cannot make the folder .*blocker/logs: ")


if __name__ == "__main__":
    unittest.main()
