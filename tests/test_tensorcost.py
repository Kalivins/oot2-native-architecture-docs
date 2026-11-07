# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/tensorcost: the plan that turns recorded dispatches and translated descriptions into a replay
job, the report arithmetic, and the replay harness (vktensorcost.c): its job-file checks, and on a machine with a
Vulkan device a small program in the translator's compute layout (set 0 global arrays, set 1 push descriptors,
128 bytes of push constants), a texture reached through the heap, and the two counter-run modes.

Synthetic descriptions only: nothing here comes from the game. The harness build skips without clang or the SDK's headers
(VULKAN_SDK) unless OOT2_REQUIRE_CLANG / OOT2_REQUIRE_VULKAN is set; the device cases skip without glslangValidator or a Vulkan
device, whatever is set, since no runner has a GPU.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tensorcost"))

import build_vkdraw  # noqa: E402
import tensor_cost as tc  # noqa: E402


def res(**kw) -> dict:
    d = {"raw_access": {"physical_pointers": False}, "support_buffer": {"set": 1, "binding": 0, "size": 1392, "fields": [
        {"name": "render_scale", "offset": 192, "bytes": 1168}]},
        "cbufs": [{"binding": 1, "slot": 0, "role": "driver0"}, {"binding": 2, "slot": 24, "role": "push"}],
        "sbufs": [], "images": [], "textures": []}
    d.update(kw)
    return d


def sbuf(slot: int, binding: int) -> dict:
    off = tc.DRIVER0_STORAGE_BASE + 16 * slot
    return {"binding": binding, "cb_slot": 0, "cb_byte_offset": off, "address_byte_offset": off, "size_byte_offset": off + 8}


def dispatch(job="a.b", **kw) -> dict:
    d = {"job": job, "grid": [4, 2, 1], "block": [64, 1, 1], "hmma": 0, "buffers": [], "textures": [], "images": []}
    d.update(kw)
    return d


class Plan(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def make(self, descriptions: dict[str, dict], dispatches: list[dict]) -> tc.Planner:
        for job, d in descriptions.items():
            (self.dir / f"{job}.res.json").write_text(json.dumps(d))
        pl = tc.Planner({"phases": [{"name": "p", "dispatches": dispatches}]}, self.dir, self.dir / "work", 64, 8, 10, 2)
        pl.plan()
        return pl

    def text(self, pl: tc.Planner) -> str:
        return "\n".join(pl.lines + pl.phase_lines)

    def test_storage_buffers_zip_by_guest_slot_and_all_take_the_largest_size(self) -> None:
        pl = self.make({"a.b": res(sbufs=[sbuf(5, 9), sbuf(2, 10)])},
                       [dispatch(buffers=[{"type": "storage", "index": 0, "size": 100}, {"type": "storage", "index": 1, "size": 4096}])])
        t = self.text(pl)
        self.assertEqual(t.count("ssbo"), 2)
        self.assertIn("bind 10 ssbo sb1", t)        # slot 2 is the lower slot: it takes the first buffer
        self.assertIn("bind 9 ssbo sb2", t)
        self.assertEqual([ln.split()[2] for ln in t.splitlines() if ln.startswith("buffer sb")], ["4096", "4096"])

    def test_driver0_bank_holds_each_buffer_s_address_and_size(self) -> None:
        pl = self.make({"a.b": res(sbufs=[sbuf(2, 9)])}, [dispatch(buffers=[{"type": "storage", "index": 0, "size": 8192}])])
        bank = (self.dir / "work" / "banks" / "p_0_driver0.bin").read_bytes()
        off = tc.DRIVER0_STORAGE_BASE + 32
        self.assertEqual(struct.unpack_from("<QI", bank, off), (1 << 32, 8192))
        self.assertEqual(len(bank), 65536)
        self.assertIn("buffer cb", self.text(pl))

    def test_push_bank_is_the_recorded_constants_after_the_first_16_bytes(self) -> None:
        raw = self.dir / "c0.bin"
        raw.write_bytes(bytes(range(64)))
        self.make({"a.b": res()}, [dispatch(buffers=[{"type": "uniform", "index": 0, "size": 64, "raw": str(raw)}])])
        bank = (self.dir / "work" / "banks" / "p_0_push.bin").read_bytes()
        self.assertEqual(bank[:48], bytes(range(16, 64)))
        self.assertEqual(bank[48:], bytes(65536 - 48))

    def test_missing_constants_are_named_and_zero(self) -> None:
        pl = self.make({"a.b": res()}, [dispatch(buffers=[{"type": "uniform", "index": 0, "size": 64}])])
        self.assertTrue(any("constants not recorded" in n for n in pl.manifest["phases"][0]["notes"]))

    def test_texture_word_is_heap_index_and_sampler_in_the_push_constants(self) -> None:
        tex = {"via": "push_slot", "push_slot": 1, "push_offset": 20, "c2_slot": 3}
        rec = {"index": 0, "tic": 7, "format": "R8G8_UNORM", "w": 8, "h": 4, "d": 1,
               "sampler": {"mag": "linear", "min": "nearest", "wrap": ["clamp_to_edge"] * 3, "max_lod": 0.25}}
        pl = self.make({"a.b": res(textures=[tex])}, [dispatch(textures=[rec])])
        t = self.text(pl)
        self.assertIn("heapimage 7 tex", t)
        self.assertIn("heapsampler 0 smp0", t)
        self.assertIn("sampler smp0 1 0 2 2 2 0.25", t)
        push = [ln for ln in t.splitlines() if ln.startswith("push ")][0].split()[1]
        self.assertEqual(struct.unpack_from("<I", bytes.fromhex(push), 20)[0], 7)

    def test_textures_zip_by_c2_slot_and_a_second_sampler_gets_its_own_index(self) -> None:
        a = {"via": "push_slot", "push_slot": 0, "push_offset": 16, "c2_slot": 5}
        b = {"via": "push_slot", "push_slot": 1, "push_offset": 20, "c2_slot": 3}
        smp = {"mag": "nearest", "min": "nearest", "wrap": ["repeat"] * 3}
        recs = [{"index": 0, "tic": 2, "format": "R8_UNORM", "w": 4, "h": 4, "d": 1, "sampler": smp},
                {"index": 1, "tic": 3, "format": "R8_UNORM", "w": 4, "h": 4, "d": 1,
                 "sampler": dict(smp, mag="linear")}]
        pl = self.make({"a.b": res(textures=[a, b])}, [dispatch(textures=recs)])
        push = bytes.fromhex([ln for ln in self.text(pl).splitlines() if ln.startswith("push ")][0].split()[1])
        # slot 3 (the lower) takes the recorded texture 0, slot 5 the texture 1
        self.assertEqual(struct.unpack_from("<I", push, 20)[0], 2 | (0 << 20))
        self.assertEqual(struct.unpack_from("<I", push, 16)[0], 3 | (1 << 20))

    def test_same_texture_index_with_another_shape_takes_the_next_free_heap_slot(self) -> None:
        t0 = {"via": "push_slot", "push_slot": 0, "push_offset": 16, "c2_slot": 0}
        smp = {"mag": "nearest", "min": "nearest", "wrap": ["repeat"] * 3}
        d1 = dispatch(textures=[{"index": 0, "tic": 9, "format": "R8_UNORM", "w": 4, "h": 4, "d": 1, "sampler": smp}])
        d2 = dispatch(textures=[{"index": 0, "tic": 9, "format": "R8_UNORM", "w": 8, "h": 8, "d": 1, "sampler": smp}])
        pl = self.make({"a.b": res(textures=[t0])}, [d1, d2])
        t = self.text(pl)
        self.assertIn("heapimage 9 ", t)
        self.assertIn("heapimage 10 ", t)

    def test_raw_access_program_is_named_and_left_out(self) -> None:
        pl = self.make({"a.b": res(raw_access={"physical_pointers": True}), "c.d": res()}, [dispatch("a.b"), dispatch("c.d")])
        m = pl.manifest["phases"][0]
        self.assertEqual((m["replayed"], m["dispatches"]), (1, 2))
        self.assertTrue(any("page table" in n and "a.b" in n for n in m["notes"]))
        self.assertEqual(self.text(pl).count("dispatch d"), 1)

    def test_uint_formats_are_bound_as_float_unless_the_module_declares_them(self) -> None:
        img = {"binding": 25, "c2_slot": 0, "format": "Unknown"}
        rec = {"slot": 0, "format": "R32_UINT", "w": 4, "h": 4, "d": 1}
        pl = self.make({"a.b": res(images=[img])}, [dispatch(images=[rec])])
        self.assertIn(f"image img", self.text(pl))
        self.assertIn(f" {tc.vk_format('R32_FLOAT')} 4 4 1 storage", self.text(pl))
        self.assertEqual(pl.manifest["substituted_formats"], ["R32_UINT as R32_FLOAT"])
        pl = self.make({"a.b": res(images=[dict(img, format="R32Uint")])}, [dispatch(images=[rec])])
        self.assertIn(f" {tc.vk_format('R32_UINT')} 4 4 1 storage", self.text(pl))
        with self.assertRaises(tc.Bad):                  # the module declares one format, the capture another
            self.make({"a.b": res(images=[dict(img, format="R32Uint")])}, [dispatch(images=[dict(rec, format="R16G16_UINT")])])

    def test_image_count_mismatch_is_refused(self) -> None:
        with self.assertRaises(tc.Bad):
            self.make({"a.b": res(images=[{"binding": 25, "c2_slot": 0}])}, [dispatch()])

    def test_pokes_and_fills_reach_the_named_buffer(self) -> None:
        pl = self.make({"a.b": res(sbufs=[sbuf(1, 9), sbuf(2, 10)])},
                       [dispatch(buffers=[{"type": "storage", "index": 0, "size": 8192}, {"type": "storage", "index": 1, "size": 8192}],
                                 pokes=[{"ordinal": 0, "offset": 0, "value": 77}], fills=[{"ordinal": 1, "word": 4294967295}])])
        t = self.text(pl)
        self.assertIn("poke sb1 0 77", t)
        self.assertIn("buffer sb2 8192 word 4294967295", t)
        self.assertIn("buffer sb1 8192 f16", t)

    def test_render_scale_of_the_support_buffer_is_one(self) -> None:
        data = tc.support_bytes(res())
        self.assertEqual(len(data), 1392)
        self.assertEqual(struct.unpack_from("<f", data, 192)[0], 1.0)
        self.assertEqual(struct.unpack_from("<f", data, 192 + 1164)[0], 1.0)
        self.assertEqual(data[:192], bytes(192))

    def test_modules_are_numbered_once_per_program(self) -> None:
        pl = self.make({"a.b": res()}, [dispatch(), dispatch()])
        self.assertEqual(self.text(pl).count("module "), 1)

    def test_job_files_for_the_three_modes(self) -> None:
        pl = self.make({"a.b": res()}, [dispatch()])
        paths = tc.write_jobs(self.dir / "work", pl.lines, pl.phase_lines, 10, 2, (64, 8), True)
        for mode, p in paths.items():
            t = p.read_text()
            self.assertIn(f"mode {mode}", t)
            self.assertIn("validation on", t)
            self.assertIn("phase p", t)

    def test_translator_jobs_use_the_split_and_the_extra_record(self) -> None:
        replay = {"phases": [{"name": "p", "dispatches": [{"job": "a.b"}, {"job": "a.b"}, {"job": "x.y"}]}]}
        jobs = tc.translator_jobs(replay, {"a.b": {"key": "a.b", "stage": "cs"}}, {"x.y": {"key": "x.y", "stage": "cs", "local": [1, 1, 1]}})
        self.assertEqual(sorted(j["key"] for j in jobs), ["a.b", "x.y"])
        with self.assertRaises(tc.Bad):
            tc.translator_jobs(replay, {}, {})


class Report(unittest.TestCase):
    def runs(self, normal: float, empty: float, half: float, hmma: int = 100) -> dict:
        def row(mode, frame, disp):
            return {"round": 0, "mode": mode, "phases": [{"phase": "p", "frames": 10, "frame_p50_ms": frame, "frame_p99_ms": frame * 1.2,
                                                          "dispatches": [{"name": "d0", "p50_ms": disp, "p99_ms": disp * 1.2}]}]}
        plan = {"phases": [{"name": "p", "dispatches": 1, "replayed": 1, "notes": [],
                            "programs": [{"name": "d0", "job": "a.b", "grid": [10, 1, 1], "block": [128, 1, 1], "hmma": hmma}]}]}
        return {"state": {}, "plan": plan, "rounds": [row("normal", normal, normal), row("empty", empty, empty), row("half", half, half / 1.0)]}

    def test_percentiles(self) -> None:
        self.assertAlmostEqual(tc.pct([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertEqual(tc.pct([5.0], 0.99), 5.0)
        self.assertEqual(tc.pct([], 0.5), 0.0)

    def test_hmma_projection_follows_the_instruction_costs(self) -> None:
        h = tc.hmma_ms({"grid": [10, 1, 1], "block": [128, 1, 1], "hmma": 100})
        self.assertEqual(h["hmma_per_frame"], 100 * 10 * 4)
        self.assertAlmostEqual(h["scratch"], 4000 * 0.921e-6)
        self.assertAlmostEqual(h["coop_f16"], 4000 * 0.0533e-6)

    def test_report_ratios_and_budget_share(self) -> None:
        rep = tc.build_report(self.runs(2.0, 0.02, 1.05), 14.0)["phases"]["p"]
        self.assertAlmostEqual(rep["share_of_budget_p99"], 2.4 / 14.0)
        self.assertAlmostEqual(rep["counter_empty"]["ratio_to_normal_p50"], 0.01)
        self.assertAlmostEqual(rep["counter_half"]["ratio_to_normal_p50"], 0.525)
        self.assertAlmostEqual(rep["counter_half"]["per_program"][0]["ratio"], 0.525)
        self.assertLess(rep["projection"]["coop_f16_p50_ms"], rep["frame_p50_ms"])
        self.assertAlmostEqual(rep["projection"]["hmma_share_of_p50"], 4000 * 0.921e-6 / 2.0)
        self.assertTrue(rep["projection"]["valid"])
        # a program that does not run its HMMA on every warp: the projected HMMA time passes the measured frame
        rep = tc.build_report(self.runs(2.0, 0.02, 1.05, hmma=10_000_000), 14.0)["phases"]["p"]
        self.assertFalse(rep["projection"]["valid"])


    def test_a_left_out_dispatch_does_not_shift_the_rows_after_it(self) -> None:
        # the plan has two dispatches, the first left out (raw access): the harness rows hold only the second
        plan = {"phases": [{"name": "p", "dispatches": 2, "replayed": 1, "notes": [], "programs": [
            {"name": "d0_x", "job": "a.b", "grid": [10, 1, 1], "block": [32, 1, 1], "hmma": 50},
            {"name": "d1_y", "job": "c.d", "grid": [20, 1, 1], "block": [128, 1, 1], "hmma": 7}]}]}
        def row(mode, frame):
            return {"round": 0, "mode": mode, "phases": [{"phase": "p", "frames": 10, "frame_p50_ms": frame, "frame_p99_ms": frame,
                                                          "dispatches": [{"name": "d1_y", "p50_ms": frame, "p99_ms": frame}]}]}
        rep = tc.build_report({"state": {}, "plan": plan, "rounds": [row("normal", 1.0), row("half", 0.5)]}, 14.0)["phases"]["p"]
        self.assertEqual([r["name"] for r in rep["programs"]], ["d1_y"])
        self.assertEqual(rep["programs"][0]["hmma"], 7)
        self.assertEqual([x["name"] for x in rep["left_out"]], ["d0_x"])
        self.assertAlmostEqual(rep["left_out_hmma_scratch_ms"], 50 * 10 * 1 * 0.921e-6)
        self.assertAlmostEqual(rep["projection"]["hmma_scratch_ms"], 7 * 20 * 4 * 0.921e-6)
        self.assertEqual(rep["counter_half"]["per_program"], [{"name": "d1_y", "ratio": 0.5}])


# ---- the harness ----------------------------------------------------------------------------------------

def find_clang() -> str | None:
    return shutil.which("clang")


def glslang() -> str | None:
    sdk = os.environ.get("VULKAN_SDK")
    if not sdk:
        return None
    p = Path(sdk) / "Bin" / ("glslangValidator.exe" if os.name == "nt" else "glslangValidator")
    return str(p) if p.is_file() else None


SHADER = """#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(local_size_x = 32) in;
layout(set = 0, binding = 0) uniform texture2D heapImg[];
layout(set = 0, binding = 1) uniform sampler heapSmp[];
layout(set = 1, binding = 1) uniform U { uvec4 v; } u;
layout(set = 1, binding = 9) buffer S { uint o[]; } s;
layout(push_constant) uniform P { uvec4 pad; uint tex; } pc;
void main() {
    uint ti = pc.tex & 0xFFFFFu, si = pc.tex >> 20;
    vec4 c = texture(sampler2D(heapImg[nonuniformEXT(ti)], heapSmp[nonuniformEXT(si)]), vec2(0.5));
    // past the edge: the border colour if the sampler is the clamp-to-border one
    vec4 outside = texture(sampler2D(heapImg[nonuniformEXT(ti)], heapSmp[nonuniformEXT(si)]), vec2(1.5, 0.5));
    uint gi = gl_GlobalInvocationID.x;
    s.o[gi] = u.v.x + gi * 0x10000u + uint(c.r * 255.0 + 0.5) + uint(outside.r * 255.0 + 0.5) * 256u;
}
"""


@unittest.skipUnless(os.name == "nt", "the harness is Windows only")
class Harness(unittest.TestCase):
    exe: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cc = find_clang()
        sdk = os.environ.get("VULKAN_SDK")
        have_sdk = bool(sdk) and (Path(sdk) / "Include" / "Volk" / "volk.c").is_file()
        if not cc or not have_sdk:
            if os.environ.get("OOT2_REQUIRE_CLANG") or os.environ.get("OOT2_REQUIRE_VULKAN"):
                raise RuntimeError("clang or the Vulkan SDK (VULKAN_SDK) is missing but the environment requires them")
            raise unittest.SkipTest("clang or the Vulkan SDK (VULKAN_SDK) is not present")
        out = Path(cls.tmp.name) / "build"
        code = build_vkdraw.main(["--harness", "vktensorcost", "--out", str(out), "--cc", cc])
        if code != 0:
            raise RuntimeError("vktensorcost did not build")
        cls.exe = out / "vktensorcost.exe"

    def run_job(self, text: str) -> subprocess.CompletedProcess:
        job = Path(self.tmp.name) / "job.txt"
        job.write_text(text, encoding="utf-8")
        return subprocess.run([str(self.exe), "run", str(job)], capture_output=True, text=True, timeout=300)

    def test_job_file_errors_exit_1_before_vulkan_is_touched(self) -> None:
        ok = "module m\nphase p\ndispatch d 0 1 1 1\nend\n"          # parses; the missing module file would stop it later, in Vulkan
        for body, message in ((ok + "frobnicate 1\n", "unknown or malformed directive"),
                              ("module m\nphase p\n", "has no `end`"), ("phase p\nend\n", "no module"),
                              ("module m\nphase p\ndispatch d 3 1 1 1\nend\n", "no module 3"),
                              ("module m\nphase p\nbind 1 ubo x\nend\n", "bind outside a dispatch"),
                              ("module m\nphase p\ndispatch d 0 1 1 1\nbind 40 ubo x\nend\n", "binding 40"),
                              ("module m\nphase p\ndispatch d 0 1 1 1\npush abc\nend\n", "3 hex digits"),
                              ("module m\nbuffer b 16 zero\nbuffer c 16 bogus\nphase p\nend\n", "buffer fill bogus"),
                              ("module m\ndispatch d 0 1 1 1\n", "dispatch outside a phase"),
                              ("module m\nphase p\ndispatch d 0 1 1 1\nbind 1 ubo nothere\nend\n", "no resource nothere")):
            with self.subTest(message):
                p = self.run_job(body)
                self.assertEqual(p.returncode, 1, p.stderr)
                self.assertIn(message, p.stderr)

    def device(self) -> bool:
        p = subprocess.run([str(self.exe), "info"], capture_output=True, text=True, timeout=120)
        return p.returncode == 0

    def test_a_program_in_the_compute_layout_reads_its_bank_push_word_and_heap_texture(self) -> None:
        gl = glslang()
        if gl is None or not self.device():
            # not a REQUIRE case: the runners that build the harness have no GPU, so the device cases skip there
            self.skipTest("glslangValidator of the Vulkan SDK or a Vulkan device is not present")
        d = Path(self.tmp.name)
        (d / "t.comp").write_text(SHADER)
        subprocess.run([gl, "-V", "--target-env", "vulkan1.3", str(d / "t.comp"), "-o", str(d / "t.spv")], check=True, capture_output=True)
        (d / "ubo.bin").write_bytes(struct.pack("<4I", 5, 0, 0, 0))
        push = "00" * 16 + struct.pack("<I", 3 | (2 << 20)).hex()
        base = f"""validation on
frames 6
warmup 2
batch 4
out {(d / 'rows.tsv').as_posix()}
heap 64 16
module {(d / 't.spv').as_posix()}
buffer ubo0 256 file {(d / 'ubo.bin').as_posix()}
buffer out0 1024 zero
image im0 37 4 4 1 sampled 1
imagefill im0 2155905152
sampler sp0 1 1 3 3 3 0
heapimage 3 im0
heapsampler 2 sp0
phase p
dispatch d0 0 4 1 1
push {push}
bind 1 ubo ubo0
bind 9 ssbo out0
end
readback out0 {(d / 'out.bin').as_posix()}
"""
        def words(mode: str) -> list[int]:
            p = self.run_job(base + f"mode {mode}\n")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("validation errors 0, warnings 0", p.stderr)
            row = json.loads(p.stdout.splitlines()[0])
            self.assertEqual((row["phase"], row["mode"], row["frames"]), ("p", mode, 6))
            return list(struct.unpack("<128I", (d / "out.bin").read_bytes()[:512]))
        w = words("normal")
        # the bank's 5, plus the texel (0x80 of 255 in its red byte) read through the push word, the sampler and the heap
        self.assertEqual(w, [5 + 128 + (g << 16) for g in range(128)])
        # the counter-runs: an empty shader writes nothing, the halved grid writes half of the words
        self.assertEqual(words("empty"), [0] * 128)
        h = words("half")
        self.assertEqual(h[:64], w[:64])
        self.assertEqual(h[64:], [0] * 64)
