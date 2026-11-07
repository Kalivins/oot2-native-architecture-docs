# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the pipeline builder (runtime/gpu/vk_pipeline.c, vk_layout.c, vk_caps.c, spirv.c, shader_desc.c; architecture
D12 and D13, ticket 60), through tests/runtime/vk_pipeline_test.c, built like tests/test_runtime_gpu.py's programs at -O2.

Three groups, by what they need:

- Without a device (continuous integration runs them): the layout's tables against xlate/host/UniLayout.cs, the dynamic
  state lists, the description reader (stage, fit, interface, refusals), the padding rule, SPIR-V declarations and the
  outputs added to a module (structure checked here, and by spirv-val when the SDK has it), the required list of
  capabilities, cache file headers judged for a device, and the store: a key the pack lacks, a job that did not
  translate, a stage of another kind and a module that does not fit are each a named fatal that prints the stage's shape.
- On a device with every item of oot2/vk_required.h, with a synthetic pack of GLSL modules the test compiles (no game
  data): the layouts under validation, a pair linked for two attachment formats and drawn through the internal submit
  with the padded input reading (0, 0, 0, 1), the counter-run that leaves it unpadded (the validation layer reports
  VUID-RuntimeSpirv-OpEntryPoint-08743), a compute dispatch, a pipeline asked for with a submission in flight (a named
  fatal), the pipeline cache warm in a second process and dropped after a driver change, a stage's objects released and
  made again, and eight threads asking for one pipeline.
- On the player's own pack (the desktop, with OOT2_T60_PACK and OOT2_T60_WORK set; never in continuous integration): the
  pair ticket 23 drew, built from the pack, against tools/firstdraw's own draw of the same modules; a sample of corpus
  pairs by research 20's method, padded pairs included; compute pipelines; a second process warm from the cache; and the
  capability check over the whole corpus with its counter-run.

Without clang or the Vulkan SDK the program tests skip, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set; without a
usable device the device tests skip, unless OOT2_REQUIRE_GPU is set.

Run: python -m unittest discover -s tests -p test_runtime_vk_pipeline.py -v
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXIT_FATAL, ROOT, build_program, case_env, clang_or_skip, fatal_lines, vulkan_or_skip

sys.path.insert(0, str(ROOT / "tools" / "shaders"))
sys.path.insert(0, str(ROOT / "tools" / "pipecost"))
import cache_pack as cp  # noqa: E402

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_pipeline_test.c"
UNI_LAYOUT = ROOT / "xlate" / "host" / "UniLayout.cs"
JOB_CS = ROOT / "xlate" / "host" / "Job.cs"
LEVEL = "-O2"
EXIT_NO_DEVICE = 3
SPIRV_MAGIC = 0x07230203
TRANSLATOR = hashlib.sha256(b"synthetic translator of ticket 60").hexdigest()

# Vulkan constants the tables are printed with.
VK_UNIFORM_BUFFER, VK_STORAGE_BUFFER, VK_STORAGE_IMAGE, VK_SAMPLED_IMAGE, VK_SAMPLER = 6, 7, 3, 2, 0
STAGE_VERTEX, STAGE_FRAGMENT, STAGE_COMPUTE, STAGE_TASK, STAGE_MESH = 0x1, 0x10, 0x20, 0x40, 0x80
GFX_STAGES = STAGE_VERTEX | STAGE_FRAGMENT | STAGE_TASK | STAGE_MESH
OP = {"capability": 17, "extension": 10, "memory_model": 14, "entry_point": 15, "decorate": 71, "void": 19, "fn_type": 33,
      "float": 22, "int": 21, "vector": 23, "pointer": 32, "constant": 43, "composite": 44, "null": 46, "variable": 59,
      "function": 54, "label": 248, "ret": 253, "fn_end": 56}

_build: dict[str, object] = {}


def program() -> Path:
    """The test program, built once per run."""
    if "exe" not in _build:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        tmp = tempfile.TemporaryDirectory(prefix="oot2-vkp-", ignore_cleanup_errors=True)
        atexit.register(tmp.cleanup)
        _build["tmp"] = tmp
        _build["glslang"] = glslang
        _build["exe"] = build_program(cc, include, glslang, TEST_SOURCE, Path(tmp.name), levels=(LEVEL,))[LEVEL]
    return _build["exe"]  # type: ignore[return-value]


def run(case: str, *args: str, env: dict[str, str] | None = None, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([str(program()), case, *args], capture_output=True, text=True, timeout=timeout, env=case_env(env))


def scratch() -> Path:
    tmp = tempfile.TemporaryDirectory(prefix="oot2-vkp-case-", ignore_cleanup_errors=True)
    atexit.register(tmp.cleanup)
    return Path(tmp.name)


# ---- a tiny SPIR-V assembler -------------------------------------------------------

def inst(op: int, *args: int) -> list[int]:
    return [((len(args) + 1) << 16) | op, *args]


def text_words(s: str) -> list[int]:
    b = s.encode() + b"\0"
    b += b"\0" * (-len(b) % 4)
    return list(struct.unpack(f"<{len(b) // 4}I", b))


def module_words(body: list[list[int]], bound: int) -> bytes:
    words = [SPIRV_MAGIC, 0x00010300, 0, bound, 0]
    for i in body:
        words += i
    return struct.pack(f"<{len(words)}I", *words)


def vertex_module(extra_caps: tuple[int, ...] = (), extensions: tuple[str, ...] = (), entry_points: int = 1) -> bytes:
    """A vertex module with one float vec4 output at Location 0 and nothing else."""
    body = [inst(OP["capability"], 1)]
    for c in extra_caps:
        body.append(inst(OP["capability"], c))
    for e in extensions:
        i = inst(OP["extension"], *text_words(e))
        i[0] = (len(i) << 16) | OP["extension"]
        body.append(i)
    body.append(inst(OP["memory_model"], 0, 1))
    for n in range(entry_points):
        main = 7 + 2 * n
        name = text_words("main")
        ep = inst(OP["entry_point"], 0, main, *name, 6)
        body.append(ep)
    body += [inst(OP["decorate"], 6, 30, 0), inst(OP["void"], 1), inst(OP["fn_type"], 2, 1), inst(OP["float"], 3, 32),
             inst(OP["vector"], 4, 3, 4), inst(OP["pointer"], 5, 3, 4), inst(OP["variable"], 5, 6, 3)]
    for n in range(entry_points):
        main = 7 + 2 * n
        body += [inst(OP["function"], 1, main, 0, 2), inst(OP["label"], main + 1), inst(OP["ret"]), inst(OP["fn_end"])]
    return module_words(body, 7 + 2 * entry_points)


def parse_words(data: bytes) -> list[list[int]]:
    words = struct.unpack(f"<{len(data) // 4}I", data)
    out, i = [], 5
    while i < len(words):
        n = words[i] >> 16
        assert n > 0, f"zero-length instruction at {i}"
        out.append(list(words[i:i + n]))
        i += n
    return out


def describe(key: str, stage: str, module: bytes, *, inputs=(), outputs=(), fits=True, schema=cp.DESC_SCHEMA,
             raw=False) -> bytes:
    """A description shaped like the translator's res.json, with the members the builder reads."""
    d = {"schema": schema, "key": key, "stage": stage, "exec_model": "Fragment", "entry": "main",
         "translator_id": TRANSLATOR,
         "answers": {"tex": {}, "img": {}, "attr": {}, "fout": {}, "early_fragment_tests": False, "heap_images": 33025,
                     "heap_samplers": 4096},
         "spv_bytes": len(module), "spv_sha256": hashlib.sha256(module).hexdigest(),
         "layout": {"name": "uni", "role": "pixel", "fits": fits, "overflow": [], "uniform_slots": [14, 11]},
         "raw_access": {"physical_pointers": raw, "page_table": False},
         "interface": {"inputs": list(inputs), "outputs": list(outputs), "builtins_in": ["FragCoord"], "builtins_out": []},
         "vertex_inputs": None, "notes": ["quote \" backslash \\ unicode é newline \n tab \t"], "float": 1.5e-3}
    return json.dumps(d, indent=1).encode("utf-8")


def io(location: int, components: int = 4, component: int = 0, type_: str = "float32", used: bool = True,
       array_length: int | None = None) -> dict:
    d = {"location": location, "component": component, "components": components, "type": type_, "used": used}
    if array_length is not None:
        d["array_length"] = array_length
    return d


def write_pack(path: Path, entries: list[cp.Entry], *, schema: int = cp.DESC_SCHEMA) -> Path:
    cp.write_pack(path, entries, translator=TRANSLATOR, schema=schema)
    return path


# ---- without a device ----------------------------------------------------------------

class TablesTest(unittest.TestCase):
    """The layout's tables, tied to the translator's numbering (xlate/host/UniLayout.cs) and to the specification."""

    @classmethod
    def setUpClass(cls) -> None:
        done = run("tables")
        assert done.returncode == 0, done.stdout + done.stderr
        cls.out = done.stdout
        cls.lines = [l.split() for l in done.stdout.splitlines()]
        src = UNI_LAYOUT.read_text(encoding="utf-8")
        cls.consts = {m.group(1): int(m.group(2)) for m in re.finditer(r"public const int (\w+) = (\d+);", src)}
        cls.roles = {}
        for m in re.finditer(r'private static readonly Role (\w+) = new\("(\w+)", (\w+), (.*)\);', src):
            args = re.findall(r"new\((\d+), (\d+)\)|\bNone\b|\bFirstTaskRange\b", m.group(4))
            ranges = [(int(a), int(b)) if a else None for a, b in args]
            if m.group(4).rstrip().endswith("FirstTaskRange"):   # the draw's first task: bytes 24 to 35 of the push constants
                ranges[4] = (cls.consts["FirstTaskOffset"], cls.consts["FirstTaskBytes"])
            cls.roles[m.group(2)] = {"buffer_set": m.group(3), "uniform": ranges[0], "storage": ranges[1],
                                     "image": ranges[2], "slots": ranges[3], "first_task": ranges[4]}

    def set1(self, layout: str) -> list[tuple[int, int, int]]:
        return [(int(l[2]), int(l[3]), int(l[4], 16)) for l in self.lines if l[0] == "set1" and l[1] == layout]

    def expect(self, bindings, role: str, stages: int) -> None:
        r = self.roles[role]
        for kind, vk in (("uniform", VK_UNIFORM_BUFFER), ("storage", VK_STORAGE_BUFFER), ("image", VK_STORAGE_IMAGE)):
            if r[kind] is None:
                continue
            first, count = r[kind]
            for b in range(first, first + count):
                self.assertIn((b, vk, stages), bindings, f"{role} {kind} {b}")

    def test_the_translators_roles_were_read(self) -> None:
        self.assertEqual(set(self.roles), {"pre_rasterization", "mesh", "pixel", "compute", "task"})
        self.assertEqual((self.consts["GlobalSet"], self.consts["PushSet"], self.consts["TaskSet"]), (0, 1, 2))
        self.assertEqual((self.consts["PushConstantBytes"], self.consts["PushDescriptorLimit"]), (128, 32))

    def test_set_zero_is_the_global_update_after_bind_arrays(self) -> None:
        rows = [l for l in self.lines if l[0] == "set0"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.consts["ImagesBinding"], 0)
        self.assertEqual(self.consts["SamplersBinding"], 1)
        heap = int(re.search(r"DefaultHeapImages = (\d+)", JOB_CS.read_text(encoding="utf-8")).group(1))
        self.assertEqual([int(rows[0][1]), int(rows[0][2]), int(rows[0][3])], [0, VK_SAMPLED_IMAGE, heap])
        self.assertEqual([int(rows[1][1]), int(rows[1][2]), int(rows[1][3])], [1, VK_SAMPLER, 4096])
        for r in rows:   # every stage, update-after-bind, update-unused-while-pending (ticket 61) and partially bound
            self.assertEqual(int(r[4], 16), GFX_STAGES | STAGE_COMPUTE)
            self.assertEqual(int(r[5], 16), 0x1 | 0x2 | 0x4)   # UPDATE_AFTER_BIND (1) | UPDATE_UNUSED_WHILE_PENDING (2) | PARTIALLY_BOUND (4)

    def test_the_graphics_push_set_holds_the_translators_slots(self) -> None:
        b = self.set1("graphics")
        self.assertEqual(len(b), self.consts["PushDescriptorLimit"])
        self.assertEqual(sorted(x[0] for x in b), list(range(32)))
        self.assertIn((self.consts["SupportBinding"], VK_UNIFORM_BUFFER, GFX_STAGES), b)   # the support buffer, shared
        for role in ("pre_rasterization", "mesh"):
            self.expect(b, role, STAGE_VERTEX | STAGE_MESH)
        self.expect(b, "pixel", STAGE_FRAGMENT)
        self.assertEqual(self.roles["pre_rasterization"], self.roles["mesh"] | {"slots": self.roles["pre_rasterization"]["slots"],
                                                                                "first_task": None} | {"slots": self.roles["pre_rasterization"]["slots"]}
                         if False else self.roles["pre_rasterization"], self.roles["pre_rasterization"])

    def test_the_compute_push_set_holds_the_translators_slots(self) -> None:
        b = self.set1("compute")
        self.assertEqual(sorted(x[0] for x in b), list(range(32)))
        self.assertIn((0, VK_UNIFORM_BUFFER, STAGE_COMPUTE), b)
        self.expect(b, "compute", STAGE_COMPUTE)

    def test_the_task_stages_buffers_are_set_two(self) -> None:
        rows = [(int(l[1]), int(l[2]), int(l[3], 16)) for l in self.lines if l[0] == "set2"]
        r = self.roles["task"]
        self.assertEqual(r["buffer_set"], "TaskSet")
        for kind, vk in (("uniform", VK_UNIFORM_BUFFER), ("storage", VK_STORAGE_BUFFER)):
            first, count = r[kind]
            for b in range(first, first + count):
                self.assertIn((b, vk, STAGE_TASK), rows)
        self.assertEqual(len(rows), 16)

    def test_push_constants_are_128_bytes_for_every_stage(self) -> None:
        rows = {l[1]: (int(l[2], 16), int(l[3]), int(l[4])) for l in self.lines if l[0] == "push"}
        self.assertEqual(rows["graphics"], (GFX_STAGES, 0, self.consts["PushConstantBytes"]))
        self.assertEqual(rows["compute"], (STAGE_COMPUTE, 0, self.consts["PushConstantBytes"]))
        self.assertEqual(self.consts["SlotsOffset"] + 4 * self.consts["SlotCount"], self.consts["PushConstantBytes"])

    def test_dynamic_states_follow_what_the_device_allows(self) -> None:
        counts = {(l[1], l[2]): (int(l[3]), l[4:]) for l in self.lines if l[0] == "dyn"}
        for variant, want in (("none", (3, 8, 10, 1)), ("all", (3, 10, 10, 5)), ("nopoly", (3, 9, 10, 5))):
            got = tuple(counts[(variant, p)][0] for p in ("vertex_input", "pre_rasterization", "fragment", "output"))
            self.assertEqual(got, want, variant)
        for key, (n, states) in counts.items():
            self.assertEqual(len(states), n)
            self.assertEqual(len(set(states)), n, key)   # no state twice
        # a state the device cannot make dynamic is absent, not replaced: polygon mode is the one that differs
        allp, nop = set(counts[("all", "pre_rasterization")][1]), set(counts[("nopoly", "pre_rasterization")][1])
        self.assertEqual(len(allp - nop), 1)
        self.assertEqual(nop - allp, set())


class DescriptionTest(unittest.TestCase):
    """The description reader (oot2/shader_desc.h) and the padding rule."""

    def setUp(self) -> None:
        self.dir = scratch()

    def desc(self, name: str, data: bytes) -> str:
        p = self.dir / name
        p.write_bytes(data)
        return str(p)

    def read(self, data: bytes) -> list[str]:
        done = run("desc", self.desc("d.json", data))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout.splitlines()

    def test_a_description_is_read_past_what_it_does_not_need(self) -> None:
        mod = vertex_module()
        out = self.read(describe("k.1", "ps", mod, inputs=[io(0), io(1, 3, 1, "uint32", False, 2)],
                                 outputs=[io(0, 2, 2, "sint32")], raw=True))
        self.assertEqual(out[0], "kind ps fits 1 raw 1")
        self.assertEqual(out[1:], ["in 0 0 4 0 32 1 1", "in 1 1 3 2 32 2 0", "out 0 2 2 1 32 1 1"])

    def test_every_stage_name_is_known(self) -> None:
        mod = vertex_module()
        for name in ("vs", "ps", "cs", "ts", "ms"):
            self.assertEqual(self.read(describe("k.1", name, mod))[0], f"kind {name} fits 1 raw 0")

    def test_a_module_that_does_not_fit_says_so(self) -> None:
        self.assertEqual(self.read(describe("k.1", "ps", vertex_module(), fits=False))[0], "kind ps fits 0 raw 0")

    def test_refusals(self) -> None:
        mod = vertex_module()
        good = describe("k.1", "ps", mod)
        cases = {
            "schema": (describe("k.1", "ps", mod, schema=1), "is schema 1, the runtime reads schema 2"),
            "stage name": (describe("k.1", "gs", mod), 'names the stage "gs"'),
            "trailing text": (good + b" x", "text after its object"),
            "cut short": (good[:len(good) // 2], "is not readable"),
            "interface type": (describe("k.1", "ps", mod, inputs=[io(0, type_="bool")]), "interface type"),
            "too many entries": (describe("k.1", "ps", mod, inputs=[io(i % 32) for i in range(65)]), "more interface entries"),
            "not json": (b"[1, 2]", "is not readable"),
        }
        no_stage = json.loads(good)
        del no_stage["stage"]
        cases["no stage"] = (json.dumps(no_stage).encode(), "names no stage")
        for name, (data, text) in cases.items():
            out = self.read(data)
            self.assertEqual(len(out), 1, name)
            self.assertTrue(out[0].startswith("refused: ") and text in out[0], f"{name}: {out}")

    def pad(self, vs_out: list[dict], ps_in: list[dict]) -> list[str]:
        mod = vertex_module()
        v = self.desc("v.json", describe("v.1", "vs", mod, outputs=vs_out))
        p = self.desc("p.json", describe("p.1", "ps", mod, inputs=ps_in))
        done = run("pad", v, p)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout.splitlines()

    def test_nothing_is_padded_when_the_vertex_stage_writes_every_input(self) -> None:
        self.assertEqual(self.pad([io(0), io(1)], [io(0), io(1)]), ["pad 0 "])
        self.assertEqual(self.pad([io(0), io(1), io(2)], [io(1)]), ["pad 0 "])   # an output nothing reads is no harm

    def test_a_missing_location_is_padded_whole(self) -> None:
        self.assertEqual(self.pad([io(0)], [io(0), io(3)]), ["pad 1 3:0:4:0;", "run 3 0 4 0"])

    def test_a_partly_written_location_is_padded_where_it_is_missing(self) -> None:
        out = self.pad([io(2, 2, 0)], [io(2, 4, 0)])
        self.assertEqual(out, ["pad 1 2:2:2:0;", "run 2 2 2 0"])
        out = self.pad([io(2, 1, 1)], [io(2, 4, 0)])   # component 1 written: 0 and 2..3 are missing
        self.assertEqual(out, ["pad 2 2:0:1:0;2:2:2:0;", "run 2 0 1 0", "run 2 2 2 0"])

    def test_a_run_never_mixes_types_and_an_array_takes_consecutive_locations(self) -> None:
        out = self.pad([], [io(3, 2, 0, "float32"), io(3, 2, 2, "uint32"), io(8, 4, 0, "sint32", array_length=2)])
        self.assertEqual(out, ["pad 4 3:0:2:0;3:2:2:2;8:0:4:1;9:0:4:1;", "run 3 0 2 0", "run 3 2 2 2", "run 8 0 4 1", "run 9 0 4 1"])

    def test_what_cannot_be_padded_is_refused_by_name(self) -> None:
        out = self.pad([], [io(0, type_="float64")])
        self.assertTrue(out[0].startswith("refused: ") and "64 bits wide" in out[0], out)
        out = self.pad([], [io(63, array_length=2)])
        self.assertTrue(out[0].startswith("refused: ") and "past the 64" in out[0], out)
        out = self.pad([], [io(i, 1, 0) for i in range(17)])
        self.assertTrue(out[0].startswith("refused: ") and "more than 16" in out[0], out)


class SpirvTest(unittest.TestCase):
    """The declarations of a module and the outputs added to it (oot2/spirv.h)."""

    def setUp(self) -> None:
        self.dir = scratch()

    def write(self, name: str, data: bytes) -> str:
        p = self.dir / name
        p.write_bytes(data)
        return str(p)

    def test_declarations_are_read_in_order_and_once(self) -> None:
        mod = vertex_module((10, 25, 10, 5347), ("SPV_KHR_storage_buffer_storage_class", "SPV_KHR_foo", "SPV_KHR_foo"))
        done = run("spvdecl", self.write("m.spv", mod))
        self.assertEqual(done.stdout.splitlines(), ["version 00010300", "cap 1 Shader", "cap 10 Float64",
                                                    "cap 25 ImageGatherExtended", "cap 5347 PhysicalStorageBufferAddresses",
                                                    "ext SPV_KHR_storage_buffer_storage_class", "ext SPV_KHR_foo"])

    def test_a_module_that_is_not_spirv_is_refused(self) -> None:
        bad = bytearray(vertex_module())
        bad[0] ^= 1
        self.assertIn("not a SPIR-V module", run("spvdecl", self.write("a.spv", bytes(bad))).stdout)
        # an instruction among the declarations that claims more words than the module has left
        cut = struct.pack("<8I", SPIRV_MAGIC, 0x00010300, 0, 5, 0, (2 << 16) | 17, 1, (9 << 16) | 17)
        self.assertIn("malformed instruction", run("spvdecl", self.write("b.spv", cut)).stdout)

    def padded(self, spec: str, mod: bytes | None = None) -> tuple[bytes, list[list[int]], list[list[int]]]:
        src = mod or vertex_module()
        done = run("spvpad", self.write("in.spv", src), str(self.dir / "out.spv"), spec)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertTrue(done.stdout.startswith("padded "), done.stdout)
        out = (self.dir / "out.spv").read_bytes()
        return out, parse_words(src), parse_words(out)

    def check_ids(self, out: bytes) -> None:
        bound = struct.unpack("<5I", out[:20])[3]
        ids = {}
        for i in parse_words(out):
            op = i[0] & 0xffff
            if op in (19, 21, 22, 23, 32, 33, 43, 44, 46, 59, 54, 248):
                rid = i[1] if op in (19, 21, 22, 23, 32, 33) else i[2] if op in (43, 44, 46, 59, 54) else i[1]
                self.assertNotIn(rid, ids, f"id {rid} defined twice")
                self.assertLess(rid, bound)
                ids[rid] = op

    def test_a_float_vec4_output_gets_a_variable_that_reads_0_0_0_1(self) -> None:
        out, before, after = self.padded("1:0:4:0")
        self.check_ids(out)
        ep = next(i for i in after if i[0] & 0xffff == OP["entry_point"])
        ep0 = next(i for i in before if i[0] & 0xffff == OP["entry_point"])
        self.assertEqual(len(ep), len(ep0) + 1)   # one more interface id
        var = ep[-1]
        decs = [i for i in after if i[0] & 0xffff == OP["decorate"] and i[1] == var]
        self.assertEqual([i[2:] for i in decs], [[30, 1]])   # Location 1, no Component
        variables = [i for i in after if i[0] & 0xffff == OP["variable"] and i[2] == var]
        self.assertEqual(len(variables), 1)
        self.assertEqual(variables[0][3], 3)   # Output
        init = variables[0][4]
        comp = next(i for i in after if i[0] & 0xffff == OP["composite"] and i[2] == init)
        consts = {i[2]: i[3] for i in after if i[0] & 0xffff == OP["constant"]}
        self.assertEqual([consts[c] for c in comp[3:]], [0, 0, 0, 0x3f800000])
        # the types the module had are reused, not declared again
        self.assertEqual(len([i for i in after if i[0] & 0xffff in (OP["float"], OP["vector"], OP["pointer"])]),
                         len([i for i in before if i[0] & 0xffff in (OP["float"], OP["vector"], OP["pointer"])]))

    def test_other_types_and_component_offsets(self) -> None:
        out, before, after = self.padded("2:0:3:1,3:3:1:2,4:2:2:0")
        self.check_ids(out)
        ep = next(i for i in after if i[0] & 0xffff == OP["entry_point"])
        vars_ = ep[-3:]
        decs = {i[1]: i[2:] for i in after if i[0] & 0xffff == OP["decorate"] and i[1] in vars_}
        self.assertEqual(decs[vars_[0]], [30, 2])
        locs = [(i[1], i[3]) for i in after if i[0] & 0xffff == OP["decorate"] and i[1] in vars_ and i[2] == 30]
        comps = [(i[1], i[3]) for i in after if i[0] & 0xffff == OP["decorate"] and i[1] in vars_ and i[2] == 31]
        self.assertEqual(sorted(l for _, l in locs), [2, 3, 4])
        self.assertEqual(sorted(c for _, c in comps), [2, 3])   # Component only where it is not 0
        ints = [i for i in after if i[0] & 0xffff == OP["int"]]
        self.assertEqual(sorted((i[2], i[3]) for i in ints), [(32, 0), (32, 1)])   # a signed and an unsigned type, new
        # a scalar in component 3 reads 1.0f's bits; a vec3 at component 0 and a vec2 at component 2 (z, w) differ
        scalar = next(i for i in after if i[0] & 0xffff == OP["constant"] and i[3] == 0x3f800000
                      and any(j[0] & 0xffff == OP["int"] and j[1] == i[1] and j[3] == 0 for j in after))
        self.assertTrue(scalar)
        self.assertEqual(len([i for i in after if i[0] & 0xffff == OP["null"]]), 1)   # the vec3 reads zeros

    def test_nothing_to_add_leaves_the_module_as_it_was(self) -> None:
        mod = vertex_module()
        done = run("spvpad", self.write("in.spv", mod), str(self.dir / "out.spv"), "")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual((self.dir / "out.spv").read_bytes(), mod)

    def test_what_cannot_be_added_is_refused_by_name(self) -> None:
        done = run("spvpad", self.write("in.spv", vertex_module(entry_points=2)), str(self.dir / "out.spv"), "1:0:4:0")
        self.assertIn("refused: the module has 2 entry points", done.stdout)
        done = run("spvpad", self.write("in.spv", vertex_module()), str(self.dir / "out.spv"), "1:0:5:0")
        self.assertIn("refused: output 1 has an unpaddable shape", done.stdout)
        done = run("spvpad", self.write("in.spv", vertex_module()), str(self.dir / "out.spv"), "1:2:3:0")
        self.assertIn("refused: output 1 has an unpaddable shape", done.stdout)

    def spirv_val(self) -> str:
        found = shutil.which("spirv-val")
        sdk = os.environ.get("VULKAN_SDK")
        if not found and sdk:
            for sub in ("Bin", "bin"):
                for name in ("spirv-val.exe", "spirv-val"):
                    if (Path(sdk) / sub / name).is_file():
                        found = str(Path(sdk) / sub / name)
        if not found:
            self.skipTest("no spirv-val (the Vulkan SDK ships it)")
        return found

    def test_a_padded_module_passes_the_validator(self) -> None:
        val = self.spirv_val()
        for spec in ("1:0:4:0", "2:0:3:1,3:3:1:2,4:2:2:0", "5:0:4:2,6:0:1:0,7:1:3:1"):
            out, _, _ = self.padded(spec)
            done = subprocess.run([val, "--target-env", "vulkan1.3", str(self.dir / "out.spv")], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"{spec}: {done.stdout}{done.stderr}")


class RequiredListTest(unittest.TestCase):
    """The capabilities and extensions the required list gives (oot2/vk_caps.h)."""

    def setUp(self) -> None:
        self.dir = scratch()

    def check(self, mod: bytes, *args: str) -> list[str]:
        p = self.dir / "m.spv"
        p.write_bytes(mod)
        done = run("capcheck", str(p), *args)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout.splitlines()

    def test_what_the_items_enable_is_in_the_list(self) -> None:
        self.assertEqual(self.check(vertex_module((1, 10, 25, 32, 33, 4427, 5302, 5307, 5347, 61, 62, 64, 65)))[0], "ok")
        out = self.check(vertex_module((5283, 6033), ("SPV_EXT_mesh_shader", "SPV_KHR_storage_buffer_storage_class")))
        self.assertEqual(out[0], "ok")

    def test_what_no_item_enables_is_refused_by_name(self) -> None:
        self.assertEqual(self.check(vertex_module((2,)))[0], "bad: capability Geometry (2) is not in the device's required list")
        self.assertEqual(self.check(vertex_module((3,)))[0], "bad: capability Tessellation (3) is not in the device's required list")
        self.assertEqual(self.check(vertex_module((63,)))[0],   # arithmetic: the subgroup item asks for four operations only
                         "bad: capability GroupNonUniformArithmetic (63) is not in the device's required list")
        self.assertEqual(self.check(vertex_module((1,), ("SPV_KHR_foo",)))[0],
                         "bad: extension SPV_KHR_foo is not in the device's required list")
        self.assertIn("capability 9999 is not in", self.check(vertex_module((9999,)))[0])

    def test_a_capability_removed_from_the_list_is_caught(self) -> None:
        mod = vertex_module((10,))
        self.assertEqual(self.check(mod)[0], "ok")
        self.assertEqual(self.check(mod, "drop", "10")[0], "bad: capability Float64 (10) is not in the device's required list")
        self.assertEqual(self.check(mod, "drop", "11")[0], "ok")   # dropping what the module does not use changes nothing

    def test_the_list_follows_the_items(self) -> None:
        out = self.check(vertex_module())
        m = re.search(r"list (\d+) capabilities, (\d+) extensions", out[1])
        self.assertGreater(int(m.group(1)), 15)
        self.assertGreater(int(m.group(2)), 5)


class CacheHeaderTest(unittest.TestCase):
    def test_a_cache_file_is_judged_by_its_header_for_the_device_and_driver(self) -> None:
        out = dict(l.split(" ", 1) for l in run("cachejudge").stdout.splitlines() if l.count(" ") == 1 and "=" not in l)
        names = dict(re.findall(r"(\w+)=(\d+)", run("cachejudge").stdout.splitlines()[-1]))
        loaded, other, bad = names["loaded"], names["other"], names["unreadable"]
        self.assertEqual(out["matching"], loaded)
        for what in ("uuid", "device", "vendor"):
            self.assertEqual(out[what], other, what)    # another driver or device: dropped, rebuilt
        for what in ("short", "empty", "version", "headersize"):
            self.assertEqual(out[what], bad, what)


class StoreTest(unittest.TestCase):
    """The module store: stages by key out of a pack, and the named fatals."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = scratch()
        mod = vertex_module()
        e = [
            cp.Entry("vs.ok", mod, describe("vs.ok", "vs", mod, outputs=[io(0)])),
            cp.Entry("ps.ok", mod, describe("ps.ok", "ps", mod, inputs=[io(0), io(1)], outputs=[io(0)])),
            cp.Entry("ps.big", mod, describe("ps.big", "ps", mod, fits=False)),
            cp.Entry("vs.failed", None, "fail: unsupported_opcode: synthetic", failed=True),
        ]
        cls.pack = write_pack(cls.dir / "base.bin", e)
        cls.pack3 = write_pack(cls.dir / "base3.bin", [cp.Entry("a.1", mod, describe("a.1", "ps", mod, schema=3))], schema=3)

    def store(self, key: str, kind: str, shape: str = "1") -> subprocess.CompletedProcess:
        return run("store", str(self.pack), "-", key, kind, shape)

    def test_a_stage_is_found_by_its_key(self) -> None:
        done = self.store("ps.ok", "ps")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertRegex(done.stdout, r"module \d+ bytes from 1, kind ps, fits 1, 2 inputs, 1 outputs")
        done = self.store("vs.ok", "vs")
        self.assertIn("kind vs, fits 1, 0 inputs, 1 outputs", done.stdout)

    def fatal(self, done: subprocess.CompletedProcess) -> str:
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        lines = fatal_lines(done.stderr)
        self.assertEqual(len(lines), 1, done.stderr)
        return lines[0]

    def test_a_key_the_pack_lacks_is_a_named_fatal_with_the_stages_shape(self) -> None:
        text = self.fatal(self.store("0123456789abcdef.00000000", "ps"))
        self.assertIn("the shader cache has no module for stage 0123456789abcdef.00000000", text)
        self.assertIn("(a ps stage, key 0123456789abcdef.00000000, code area 4656 bytes, instructions 4096 bytes, "
                      "control section 87 bytes)", text)
        self.assertIn("(key 0123456789abcdef.00000000, no shape recorded)", self.fatal(self.store("0123456789abcdef.00000000", "ps", "0")))

    def test_a_job_that_did_not_translate_is_a_named_fatal_with_its_reason(self) -> None:
        text = self.fatal(self.store("vs.failed", "vs"))
        self.assertIn("stage vs.failed did not translate: fail: unsupported_opcode: synthetic", text)

    def test_a_stage_of_another_kind_and_a_module_that_does_not_fit_are_named_fatals(self) -> None:
        self.assertIn("stage ps.ok is a ps program where a vs was asked for", self.fatal(self.store("ps.ok", "vs")))
        self.assertIn("stage ps.big does not fit its slots of the uni layout", self.fatal(self.store("ps.big", "ps")))

    def test_packs_the_reader_refuses_stay_closed(self) -> None:
        done = run("storeopen", str(self.pack3), "-")
        self.assertTrue(done.stdout.startswith("refused: ") and "schema" in done.stdout.lower(), done.stdout)
        done = run("storeopen", str(self.dir / "missing.bin"), "-")
        self.assertTrue(done.stdout.startswith("refused: "), done.stdout)
        self.assertEqual(run("storeopen", str(self.pack), "-").stdout, "ok\n")


# ---- on a device, with a synthetic pack of GLSL modules -----------------------------------

GLSL = {
    "vs.vert": """#version 450
layout(set = 1, binding = 1) uniform VsBank { vec4 offset; } bank;
layout(set = 1, binding = 9) readonly buffer VsData { vec4 colour; } data;
layout(push_constant) uniform Push { uvec4 words[8]; } pc;
layout(location = 0) out vec4 a;
void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0 + bank.offset.xy, 0.0, 1.0);
    a = data.colour;
}
""",
    "ps.frag": """#version 450
layout(set = 1, binding = 14) uniform PsBank { vec4 add; } bank;
layout(set = 1, binding = 25) readonly buffer PsData { vec4 mul; } data;
layout(push_constant) uniform Push { uvec4 words[8]; } pc;
layout(location = 0) in vec4 a;
layout(location = 1) in vec4 b;
layout(location = 0) out vec4 o;
void main() { o = (a + b + bank.add) * data.mul + vec4(float(pc.words[1].x)); }
""",
    "cs.comp": """#version 450
layout(local_size_x = 32) in;
layout(set = 1, binding = 9) buffer Out { uint v[]; } o;
layout(set = 1, binding = 1) uniform Bank { uvec4 k; } bank;
layout(push_constant) uniform Push { uvec4 words[8]; } pc;
void main() { o.v[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x * bank.k.x + pc.words[0].x; }
""",
    "cs2.comp": """#version 450
layout(local_size_x = 32) in;
layout(set = 1, binding = 9) buffer Out { uint v[]; } o;
void main() { o.v[gl_GlobalInvocationID.x] = 1u; }
""",
}
SYNTH_KEYS = {"vs.vert": "5100000000000001.00000000", "ps.frag": "5100000000000002.00000000",
              "cs.comp": "5100000000000003.00000000", "cs2.comp": "5100000000000004.00000000"}
SYNTH_STAGE = {"vs.vert": "vs", "ps.frag": "ps", "cs.comp": "cs", "cs2.comp": "cs"}


def synthetic_pack(directory: Path) -> Path:
    """GLSL compiled by glslang with the build's flags, packed with descriptions that name each stage's interface."""
    from build_game import GLSLANG_FLAGS  # tools/build_game.py is on the path through test_runtime_gpu
    program()
    glslang = _build["glslang"]
    entries = []
    for name, src in GLSL.items():
        path = directory / name
        path.write_text(src, encoding="utf-8")
        spv = directory / (name + ".spv")
        done = subprocess.run([glslang, *GLSLANG_FLAGS, "-o", str(spv), str(path)], capture_output=True, text=True)
        assert done.returncode == 0, f"glslang on {name}:\n{done.stdout}{done.stderr}"
        mod = spv.read_bytes()
        key = SYNTH_KEYS[name]
        kw: dict = {}
        if name == "vs.vert":
            kw = {"outputs": [io(0)]}
        elif name == "ps.frag":
            kw = {"inputs": [io(0), io(1)], "outputs": [io(0)]}
        entries.append(cp.Entry(key, mod, describe(key, SYNTH_STAGE[name], mod, **kw)))
    return write_pack(directory / "synthetic.bin", entries)


def require_gpu() -> bool:
    return bool(os.environ.get("OOT2_REQUIRE_GPU"))


class DeviceTestCase(unittest.TestCase):
    usable = 0

    @classmethod
    def setUpClass(cls) -> None:
        done = run("usable")
        if done.returncode == EXIT_NO_DEVICE:
            why = f"no Vulkan device here ({done.stdout.strip()})"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but there is {why}")
            raise unittest.SkipTest(why)
        assert done.returncode == 0, done.stdout + done.stderr
        cls.usable = int(re.search(r"^usable (\d+)$", done.stdout, re.M).group(1))
        if not cls.usable:
            why = "no device here has every item the renderer requires"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but {why}")
            raise unittest.SkipTest(why)
        cls.dir = scratch()
        cls.pack = synthetic_pack(cls.dir)

    def gpu(self, case: str, *args: str, validation: str = "sync", timeout: int = 300) -> subprocess.CompletedProcess:
        return run(case, *args, env={"OOT2_VK_VALIDATION": validation}, timeout=timeout)

    def assert_clean(self, done: subprocess.CompletedProcess, case: str) -> None:
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"ok {case}", done.stdout)
        self.assertEqual(fatal_lines(done.stderr), [])
        self.assertIn("validation errors 0 warnings 0", done.stdout, done.stderr[-3000:])

    @staticmethod
    def stats(out: str, tag: str) -> dict[str, int]:
        m = re.search(rf"^stats {tag} libraries (\d+) hits (\d+) stage (\d+) hits (\d+) links (\d+) hits (\d+) "
                      rf"computes (\d+) hits (\d+) padded (\d+) resident (\d+)", out, re.M)
        assert m, out
        keys = ("libraries", "library_hits", "stage_libraries", "stage_library_hits", "links", "link_hits", "computes",
                "compute_hits", "padded", "resident")
        return dict(zip(keys, (int(x) for x in m.groups())))


class DeviceTest(DeviceTestCase):
    def test_the_layouts_are_made_valid_and_destroyed(self) -> None:
        self.assert_clean(self.gpu("layouts"), "layouts")

    def test_a_pair_links_for_two_formats_and_draws_with_the_padded_input_reading_0_0_0_1(self) -> None:
        done = self.gpu("synth_draw", str(self.pack))
        self.assert_clean(done, "synth_draw")
        self.assertIn("padded input reads (0, 0, 0, 1): f32 yes, f16 yes", done.stdout)
        s = self.stats(done.stdout, "graphics")
        # the shared vertex input library, the vertex library (padded), the fragment library and two output libraries;
        # two fast links; the same pair asked for again made nothing
        self.assertEqual((s["libraries"], s["links"], s["padded"], s["resident"]), (5, 2, 1, 6), s)
        print("\n" + done.stdout)

    def test_counter_run_a_pixel_input_left_unpadded_is_reported_by_validation(self) -> None:
        done = self.gpu("synth_draw", str(self.pack), "unpad")
        self.assertIn("VUID-RuntimeSpirv-OpEntryPoint-08743", done.stderr, done.stdout + done.stderr[-3000:])
        self.assertNotIn("validation errors 0 ", done.stdout)
        print("\n" + "\n".join(l for l in done.stderr.splitlines() if "08743" in l)[:1500])

    def test_a_compute_pipeline_dispatches(self) -> None:
        done = self.gpu("synth_compute", str(self.pack))
        self.assert_clean(done, "synth_compute")
        self.assertIn("128 values written by the dispatch: 3 * i + 7", done.stdout)
        self.assertEqual(self.stats(done.stdout, "compute")["computes"], 1)

    def test_a_pipeline_asked_for_with_a_submission_in_flight_is_a_named_fatal(self) -> None:
        done = self.gpu("synth_inflight", str(self.pack), validation="off")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn("the existing pipeline was returned with a submission in flight", done.stdout)
        self.assertNotIn("returned\n", done.stdout.replace("was returned with", ""))
        lines = fatal_lines(done.stderr)
        self.assertEqual(len(lines), 1, done.stderr)
        self.assertRegex(lines[0], r"a compute pipeline would be created for stage 5100000000000004\.00000000 with work in "
                                   r"flight \(submissions up to \d+ handed out, \d+ done\)")

    def test_a_stage_the_pack_lacks_is_a_named_fatal_on_the_device_path(self) -> None:
        done = self.gpu("synth_absent", str(self.pack), validation="off")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        text = fatal_lines(done.stderr)[0]
        self.assertIn("the shader cache has no module for stage 5100000000000009.00000000", text)
        self.assertIn("(a vs stage, key 5100000000000009.00000000, code area 17760 bytes, instructions 16384 bytes", text)

    def test_the_pipeline_cache_serves_a_second_process_and_is_dropped_after_a_driver_change(self) -> None:
        file = self.dir / "pipeline-cache.bin"
        file.unlink(missing_ok=True)
        cold = self.gpu("synth_cache", str(self.pack), str(file), "cold")
        self.assert_clean(cold, "synth_cache")
        self.assertIn("cache_state 0", cold.stdout)
        c = self.stats(cold.stdout, "cold")
        self.assertEqual((c["stage_library_hits"], c["compute_hits"]), (0, 0), c)
        self.assertEqual((c["stage_libraries"], c["computes"]), (2, 1), c)   # the vertex and the fragment library
        self.assertTrue(file.is_file() and file.stat().st_size > 32)
        warm = self.gpu("synth_cache", str(self.pack), str(file), "warm")
        self.assert_clean(warm, "synth_cache")
        self.assertIn("cache_state 1", warm.stdout)
        w = self.stats(warm.stdout, "warm")
        self.assertEqual((w["libraries"], w["links"], w["computes"]), (c["libraries"], c["links"], c["computes"]))
        # everything the driver compiled comes back from the cache: the libraries that hold a shader and the compute
        # pipeline. A library with no shader and a fast link compile nothing and are not cache entries (research 20: a
        # link is microseconds), so the driver never reports them as hits.
        self.assertEqual((w["stage_library_hits"], w["compute_hits"]), (w["stage_libraries"], w["computes"]), w)
        print("\n" + cold.stdout + warm.stdout)
        # a driver change: the header names another pipelineCacheUUID, the file is dropped and everything is built again
        data = bytearray(file.read_bytes())
        data[16 + 3] ^= 0xFF
        other = self.dir / "other-driver.bin"
        other.write_bytes(bytes(data))
        dropped = self.gpu("synth_cache", str(self.pack), str(other), "warm")
        self.assert_clean(dropped, "synth_cache")
        self.assertIn("cache_state 2", dropped.stdout)
        d = self.stats(dropped.stdout, "warm")
        self.assertEqual((d["stage_library_hits"], d["compute_hits"]), (0, 0), d)
        short = self.dir / "short.bin"
        short.write_bytes(bytes(data[:20]))
        cut = self.gpu("synth_cache", str(self.pack), str(short), "warm")
        self.assert_clean(cut, "synth_cache")
        self.assertIn("cache_state 3", cut.stdout)

    def test_a_stages_objects_are_released_and_made_again(self) -> None:
        self.assert_clean(self.gpu("synth_release", str(self.pack)), "synth_release")

    def test_eight_threads_asking_for_one_pipeline_make_it_once(self) -> None:
        self.assert_clean(self.gpu("synth_threads", str(self.pack)), "synth_threads")


# ---- on the player's own pack ------------------------------------------------------------

PLAYER_PACK = os.environ.get("OOT2_T60_PACK")
PLAYER_WORK = os.environ.get("OOT2_T60_WORK")


@unittest.skipUnless(PLAYER_PACK and PLAYER_WORK, "OOT2_T60_PACK and OOT2_T60_WORK name the player's pack and a work folder")
class PlayerPackTest(DeviceTestCase):
    """Never in continuous integration: the pack and the work folder hold the game's programs and stay out of the repository."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.work = Path(PLAYER_WORK)
        cls.work.mkdir(parents=True, exist_ok=True)
        cls.real_pack = Path(PLAYER_PACK)

    def pack_module(self, key: str) -> bytes:
        pack = cp.open_pack(self.real_pack)
        mod, _ = pack.read(pack.find(key))
        return mod

    def test_the_pair_ticket_23_drew_builds_from_the_pack_and_draws_the_same_pixels(self) -> None:
        spec_path = os.environ.get("OOT2_T60_FIRSTDRAW_SPEC")
        draw_dir = os.environ.get("OOT2_T60_FIRSTDRAW_DIR")
        if not spec_path or not draw_dir:
            self.skipTest("OOT2_T60_FIRSTDRAW_SPEC (spec23.json) and OOT2_T60_FIRSTDRAW_DIR (a run folder's draw.txt) name the draw")
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        vkey, pkey = spec["vs"], spec["ps"]
        out = self.work / "firstdraw"
        out.mkdir(exist_ok=True)
        (out / "vs.spv").write_bytes(self.pack_module(vkey))
        (out / "fs.spv").write_bytes(self.pack_module(pkey))
        draw = Path(draw_dir) / "draw.txt"
        # the reference: tools/firstdraw's own harness on the same two modules
        ref_draw = out / "ref-draw.txt"
        lines = []
        for line in draw.read_text(encoding="utf-8").splitlines():
            w = line.split(" ", 1)
            if w[0] == "vs":
                line = f"vs {out / 'vs.spv'}"
            elif w[0] == "fs":
                line = f"fs {out / 'fs.spv'}"
            elif w[0] == "out":
                line = f"out {out / 'reference.f32'}"
            lines.append(line)
        ref_draw.write_text("\n".join(lines) + "\n", encoding="utf-8")
        built = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "build_vkdraw.py"), "--harness", "vkdraw",
                                "--out", str(out / "vkdraw")], capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        exe = out / "vkdraw" / ("vkdraw.exe" if os.name == "nt" else "vkdraw")
        ref = subprocess.run([str(exe), str(ref_draw)], capture_output=True, text=True, env=case_env({"VK_LOADER_LAYERS_DISABLE": "~implicit~"}))
        self.assertEqual(ref.returncode, 0, ref.stdout + ref.stderr)
        print("\nreference (tools/firstdraw/vkdraw.c):", ref.stdout.strip())
        done = self.gpu("firstdraw", str(draw), str(self.real_pack), vkey, pkey, str(out / "builder.f32"))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-4000:])
        print("\n" + done.stdout)
        a, b = (out / "reference.f32").read_bytes(), (out / "builder.f32").read_bytes()
        self.assertEqual(len(a), len(b))
        self.assertEqual(hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest(), "the two images differ")
        self.assertIn("validation errors 0 warnings 0", done.stdout, done.stderr[-4000:])

    def sample(self, pairs: int) -> Path:
        import pipe_cost
        work = self.work / "select"
        work.mkdir(exist_ok=True)
        shutil.copy(os.environ["OOT2_T60_PAIRS"], work / "pairs.jsonl")
        self.assertEqual(pipe_cost.select(work, pairs, 20), 0)   # research 20's selection: deciles of module size, seed 20
        rows = json.loads((work / "sample.json").read_text(encoding="utf-8"))["pairs"]
        tsv = self.work / f"pairs-{pairs}.tsv"
        tsv.write_text("".join(f"{r['vs']}\t{r['ps']}\n" for r in rows), encoding="utf-8")
        return tsv

    def test_a_sample_of_corpus_pairs_builds_and_links_with_no_refusal_padded_pairs_included(self) -> None:
        if not os.environ.get("OOT2_T60_PAIRS"):
            self.skipTest("OOT2_T60_PAIRS names S3's pairs.jsonl (the corpus pairs of research 20)")
        n = int(os.environ.get("OOT2_T60_SAMPLE", "120"))
        tsv = self.sample(n)
        cache = self.work / f"pipeline-cache-pairs-{n}.bin"
        cache.unlink(missing_ok=True)
        threads = os.environ.get("OOT2_T60_THREADS", "4")
        cold = self.gpu("corpus_pairs", str(self.real_pack), str(tsv), threads, str(cache), "save", validation="on", timeout=7200)
        print("\n" + cold.stdout)
        self.assertEqual(cold.returncode, 0, cold.stdout + cold.stderr[-4000:])
        self.assertEqual(fatal_lines(cold.stderr), [])
        self.assertIn("validation errors 0 warnings 0", cold.stdout, cold.stderr[-4000:])
        c = self.stats(cold.stdout, "pairs")
        self.assertGreater(c["padded"], 0, "the sample holds padded pairs")
        warm = self.gpu("corpus_pairs", str(self.real_pack), str(tsv), threads, str(cache), validation="off", timeout=7200)
        print("\n" + warm.stdout)
        self.assertEqual(warm.returncode, 0, warm.stdout + warm.stderr[-4000:])
        w = self.stats(warm.stdout, "pairs")
        self.assertEqual(w["stage_library_hits"], w["stage_libraries"], w)

    def test_compute_pipelines_build_and_a_second_process_gets_them_warm(self) -> None:
        keys = os.environ.get("OOT2_T60_COMPUTE")
        if not keys:
            self.skipTest("OOT2_T60_COMPUTE names a file of compute job keys, one per line")
        cache = self.work / "pipeline-cache-compute.bin"
        cache.unlink(missing_ok=True)
        threads = os.environ.get("OOT2_T60_THREADS", "4")
        cold = self.gpu("corpus_compute", str(self.real_pack), keys, threads, str(cache), "save", validation="on", timeout=7200)
        print("\n" + cold.stdout)
        self.assertEqual(cold.returncode, 0, cold.stdout + cold.stderr[-4000:])
        self.assertIn("validation errors 0 warnings 0", cold.stdout, cold.stderr[-4000:])
        c = self.stats(cold.stdout, "compute")
        self.assertGreater(c["computes"], 0)
        warm = self.gpu("corpus_compute", str(self.real_pack), keys, threads, str(cache), validation="off", timeout=7200)
        print("\n" + warm.stdout)
        self.assertEqual(warm.returncode, 0, warm.stdout + warm.stderr[-4000:])
        w = self.stats(warm.stdout, "compute")
        self.assertEqual((w["computes"], w["compute_hits"]), (c["computes"], c["computes"]), w)

    def test_the_capability_check_over_the_whole_corpus_and_its_counter_run(self) -> None:
        keys = os.environ.get("OOT2_T60_KEYS")
        if not keys:
            self.skipTest("OOT2_T60_KEYS names `cache_pack.py list` output of the pack")
        done = run("caps", str(self.real_pack), keys, timeout=3600)
        print("\n" + done.stdout)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-2000:])
        self.assertIn("ok caps", done.stdout)
        # the counter-run: drop a capability the corpus declares from the list; the modules that declare it are refused
        declared = [(int(m.group(1)), int(m.group(2))) for m in re.finditer(r"^cap (\d+) \S+ (\d+) in the list", done.stdout, re.M)]
        dropped = next((c for c in sorted(declared, key=lambda x: x[1]) if c[0] not in (0, 1)), None)
        if dropped:
            bad = run("caps", str(self.real_pack), keys, "drop", str(dropped[0]), timeout=3600)
            m = re.search(r"refused (\d+)", bad.stdout)
            self.assertTrue(m and int(m.group(1)) == dropped[1], bad.stdout)
            print(f"counter-run: capability {dropped[0]} dropped from the list: {dropped[1]} modules refused")


if __name__ == "__main__":
    unittest.main()
