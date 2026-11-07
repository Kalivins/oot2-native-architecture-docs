#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Spike S2: draw one translated vertex and pixel program pair, check every pixel.

    py -3 tools/firstdraw/first_draw.py --spec SPEC.json --work DIR
        [--vkdraw build/vkdraw/vkdraw.exe] [--nvdisasm PATH]
        [--validation on|gpu|sync|off] [--counter swap-uv|swap-interpolation|tex-sources-from-ra]
        [--scenario NAME] [--gpu-only]

For each scenario of the spec, the draw goes twice through the same inputs:

1. on the GPU: the translated SPIR-V of both programs (a corpus pass's output),
   drawn by vkdraw in the descriptor layout `uni` with everything the two
   resource descriptions (schema 2) name and nothing else: the vertex inputs,
   the support buffer, the constant banks and storage buffers at their stage's
   slots of set 1, and the textures in the global arrays of set 0, reached
   through the handles written in their push constant slots;
2. on the CPU: the programs' Ampere code, decoded by NVIDIA's nvdisasm and
   evaluated by sass_eval, with the triangles set up by reference.py.

--gpu-only skips the second half (no nvdisasm needed): the run then checks the
binding and the validation layer only, and passes when the layer reports
nothing and the draw covers pixels; it is not the comparison.

The two images are compared pixel by pixel. A covered pixel passes when every
component is within --atol + --rtol * |expected|; a pixel the pixel program
kills, or one outside the triangles, must keep the clear colour exactly; edge
pixels and pixels whose nearest texel is ambiguous are skipped and counted.
A scenario passes when nothing mismatches and the validation layer reported
nothing (under --validation gpu, no error: GPU-assisted validation always
warns about its own settings). --counter makes the CPU side wrong on purpose (texture coordinates
swapped, perspective and screen-linear interpolation swapped, or every texture
source read from Ra on instead of split between Ra and Rb); the run then
passes only if the comparison catches it.

Everything this writes goes under --work: the programs' code and listings, the
blobs, the images, the report. None of it may enter the repository. The spec
is also kept outside the repository; it holds the program keys and synthetic
inputs, as JSON:

  "jobs": path of the split's jobs.jsonl, "spv_dir": a pass's spv folder,
  "vs", "ps": the two job keys, "size": [W, H], "clear": [r, g, b, a],
  "vertex_stream": {"vertices": [[values], ...]}: per vertex, the values of the
                   vertex program's inputs (its description's vertex_inputs, by
                   location), each with that input's components and type,
  "triangles": [[i, j, k], ...],
  "banks": {"vs"|"ps": {"<slot>": {"<byte offset>": {"f32": [...]} | {"u32": [...]} | {"u64": n}}}},
  "global": [{"address": n, "f32": [...]} | {"address": n, "words": [1.5, "0x64", ...]}],
  "textures": [{"stage": "vs"|"ps", "handle": "0x..", "type": "2d"|"2darray"|"3d"|"cube", "width": W,
                "height": H, "layers"|"depth": N, "filter": "nearest", "address": "clamp", "seed": n,
                "alpha_zero": {"block": B, "every": E}, "set": [...]}],  (make_texture)
  "scenarios": [{"name": "...", "banks": {...}, "global": [...]}, ...]

A texture's handle is the word offset of its handle in c[2] (the description's
tex_word_offset); without "stage" it serves both programs. c[1] is never in the
spec: a program's own constants (split.py's c1) are bound there (stage_banks).

A scenario's banks are applied over the spec's own; its global regions are
added to the spec's and must not overlap them (point a scenario's buffer
addresses at regions of its own).

Exit codes: 0 every scenario passed, 1 a scenario failed, 2 a tool or input is
missing, 3 the reference cannot evaluate the programs (an unsupported
instruction or input).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reference as ref  # noqa: E402
import sass_eval as se  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
BANK_BYTES = 0x10000
SUPPORT_BYTES = 1392
SPH_BYTES = 0x80
SCHEMA = 2          # the resource descriptions this harness binds from (xlate/host/ResultWriter.cs)
PUSH_BYTES = 128    # the layout's push constants (xlate/host/UniLayout.cs)
STAGE_LETTER = {"vs": "v", "ps": "f"}
INPUT_FORMATS = {"float32": "f", "sint32": "i", "uint32": "ui"}
TEXTURE_LAYERS = {"2d": None, "2darray": "layers", "3d": "depth", "cube": None}  # what sets a texture's third size

# SPIR-V capabilities to what the device must enable: a feature name of vkdraw,
# a subgroup operation, or nothing. A capability missing here stops the run.
CAPABILITIES = {
    "Shader": None, "Matrix": None, "Sampled1D": None, "Image1D": None, "SampledBuffer": None,
    "ImageBuffer": None, "ImageQuery": None, "DerivativeControl": None,
    "Float64": "shaderFloat64", "Int64": "shaderInt64", "Int16": "shaderInt16", "Int8": "shaderInt8",
    "Float16": "shaderFloat16", "ClipDistance": "shaderClipDistance", "CullDistance": "shaderCullDistance",
    "ImageGatherExtended": "shaderImageGatherExtended",
    "StorageImageExtendedFormats": "shaderStorageImageExtendedFormats",
    "StorageImageReadWithoutFormat": "shaderStorageImageReadWithoutFormat",
    "StorageImageWriteWithoutFormat": "shaderStorageImageWriteWithoutFormat",
    "StorageImageMultisample": "shaderStorageImageMultisample",
    "SampledCubeArray": "imageCubeArray", "ImageCubeArray": "imageCubeArray",
    "DrawParameters": "shaderDrawParameters", "PhysicalStorageBufferAddresses": "bufferDeviceAddress",
    "DemoteToHelperInvocation": "shaderDemoteToHelperInvocation", "MinLod": "shaderResourceMinLod",
    "GroupNonUniform": "subgroup basic", "GroupNonUniformVote": "subgroup vote",
    "GroupNonUniformArithmetic": "subgroup arithmetic", "GroupNonUniformBallot": "subgroup ballot",
    "GroupNonUniformShuffle": "subgroup shuffle", "GroupNonUniformShuffleRelative": "subgroup relative",
    "GroupNonUniformClustered": "subgroup clustered", "GroupNonUniformQuad": "subgroup quad",
}


class InputError(Exception):
    pass


def needs_stores_and_atomics(res: dict) -> bool:
    """Whether a vertex or fragment module needs its stage's stores-and-atomics feature.

    Vulkan accepts a storage buffer or storage image that is not decorated
    NonWritable in these stages only with vertexPipelineStoresAndAtomics or
    fragmentStoresAndAtomics (VUID-RuntimeSpirv-NonWritable-06340 and -06341).
    The host's post-pass marks every one the module never writes; a description
    written before it (no non_writable field) counts as writable."""
    return (any(not b.get("non_writable", False) for b in [*res["sbufs"], *res["images"]])
            or bool(res.get("raw_access", {}).get("miss_log", False)))


# ---- inputs ------------------------------------------------------------------

def load_program(jobs: dict[str, dict], key: str, work: Path, nvdisasm: str) -> tuple[list[se.Instr], dict, Path]:
    """Extract one program's header and code, disassemble the code, parse both."""
    job = jobs.get(key)
    if job is None:
        raise InputError(f"job {key} is not in the jobs file")
    with open(job["code"], "rb") as f:
        f.seek(job["code_offset"])
        data = f.read(job["code_length"])
    if len(data) != job["code_length"] or len(data) <= SPH_BYTES:
        raise InputError(f"job {key}: short code slice")
    pdir = work / "programs"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{key}.sph").write_bytes(data[:SPH_BYTES])
    (pdir / f"{key}.code").write_bytes(data[SPH_BYTES:])
    out = subprocess.run([nvdisasm, "-b", "SM86", "-hex", str(pdir / f"{key}.code")], capture_output=True, text=True)
    if out.returncode != 0:
        raise InputError(f"nvdisasm failed on {key}: {out.stderr.strip()[:300]}")
    (pdir / f"{key}.sass").write_text(out.stdout)
    return se.parse_listing(out.stdout), se.parse_sph(data[:SPH_BYTES]), pdir


def bank_bytes(spec: dict) -> bytearray:
    """A 64 KiB constant bank from {"<offset>": {"f32"|"u32": [...]} | {"u64": n}}."""
    b = bytearray(BANK_BYTES)
    for off_text, value in spec.items():
        off = int(off_text, 0)
        if "f32" in value:
            raw = struct.pack(f"<{len(value['f32'])}f", *value["f32"])
        elif "u32" in value:
            raw = struct.pack(f"<{len(value['u32'])}I", *(int(v, 0) if isinstance(v, str) else v for v in value["u32"]))
        elif "u64" in value:
            v = value["u64"]
            raw = struct.pack("<Q", int(v, 0) if isinstance(v, str) else v)
        else:
            raise InputError(f"bank value at {off_text} has no f32, u32 or u64")
        if off < 0 or off % 4 or off + len(raw) > BANK_BYTES:
            raise InputError(f"bank value at {off_text} is misaligned or past the bank")
        b[off:off + len(raw)] = raw
    return b


def stage_banks(banks: dict, progs: dict) -> dict[str, dict[int, bytearray]]:
    """Each stage's constant banks by slot: the spec's, and c[1] from the program itself.

    c[1] is the program's own constant data (the jump tables of its indirect
    branches), carried in its code area (split.py's c1 field), so it is bound as
    the program's and never taken from the spec."""
    out = {st: {int(s): bank_bytes(v) for s, v in banks.get(st, {}).items()} for st in ("vs", "ps")}
    for st in ("vs", "ps"):
        c1 = progs[st].get("c1", b"")
        if c1:
            if 1 in out[st]:
                raise InputError(f"the spec sets c[1] of the {st} program, which holds the program's own constants")
            out[st][1] = bytearray(c1) + bytearray(BANK_BYTES - len(c1))
    return out


def merge_banks(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for stage, slots in over.items():
        for slot, values in slots.items():
            out.setdefault(stage, {}).setdefault(slot, {}).update(values)
    return out


class Memory:
    """Global memory: non-overlapping regions of 32-bit words, given as "f32" (floats) or "words" (each a
    float, or a string "0x..." for an integer word)."""

    def __init__(self, regions: list[dict]):
        self.regions = []
        for r in regions:
            addr = int(r["address"], 0) if isinstance(r["address"], str) else r["address"]
            if "f32" in r:
                data = struct.pack(f"<{len(r['f32'])}f", *r["f32"])
            elif "words" in r:
                data = b"".join(struct.pack("<I", int(w, 0) & 0xFFFFFFFF) if isinstance(w, str) else struct.pack("<f", w)
                                for w in r["words"])
            else:
                data = bytes(r.get("bytes", b""))
            for base, other in self.regions:
                if addr < base + len(other) and base < addr + len(data):
                    raise InputError(f"global regions at {base:#x} and {addr:#x} overlap")
            self.regions.append((addr, bytearray(data)))

    def load(self, addr: int, size: int) -> bytes:
        for base, data in self.regions:
            if base <= addr and addr + size <= base + len(data):
                return bytes(data[addr - base:addr - base + size])
        raise se.Unsupported(f"global load of {size} bytes at {addr:#x} outside every region")

    def window(self, addr: int, size: int) -> bytes:
        """size bytes from addr, zero where no region covers them."""
        out = bytearray(size)
        for base, data in self.regions:
            lo, hi = max(addr, base), min(addr + size, base + len(data))
            if lo < hi:
                out[lo - addr:hi - addr] = data[lo - base:hi - base]
        return bytes(out)


def make_texture(t: dict) -> ref.Texture:
    """A texture of random texels: "type" 2d (default), 2darray ("layers"), 3d ("depth") or cube (six faces),
    each layer, slice or face row-major from row 0.

    "alpha_zero": {"block": B, "every": E} zeroes alpha where (x // B + y // B) % E == 0; "set": a list of
    {"channel": c, "value": v, "block": B, "every": E, "invert": false} sets channel c to v in the same
    pattern (block 1 and every 1: every texel), or outside it with "invert"."""
    rng = random.Random(t.get("seed", 1))
    w, h = t["width"], t["height"]
    kind = t.get("type", "2d")
    if kind not in TEXTURE_LAYERS:
        raise InputError(f"texture type {kind!r} is not drawn")
    layers = 6 if kind == "cube" else t[TEXTURE_LAYERS[kind]] if TEXTURE_LAYERS[kind] else 1
    rules = list(t.get("set", []))
    if "alpha_zero" in t:
        rules.append({"channel": 3, "value": 0.0, **t["alpha_zero"]})
    texels = []
    for _ in range(layers):
        for y in range(h):
            for x in range(w):
                px = [rng.randrange(1, 256) / 256 for _ in range(3)] + [rng.randrange(64, 256) / 256]
                for rule in rules:
                    hit = ((x // rule.get("block", 1)) + (y // rule.get("block", 1))) % rule.get("every", 1) == 0
                    if hit != rule.get("invert", False):
                        px[rule["channel"]] = float(rule["value"])
                texels.append(tuple(px))
    return ref.Texture(w, h, texels, t.get("filter", "nearest"), t.get("address", "clamp"), kind, layers)


def support_buffer(width: int, height: int) -> bytes:
    """The translator's support buffer (Ryujinx SupportBuffer): no alpha test, no
    BGRA targets, the viewport's inverse and size, every render scale 1."""
    b = bytearray(SUPPORT_BYTES)
    struct.pack_into("<4f", b, 144, 2.0 / width, 2.0 / height, 1.0, 0.0)
    struct.pack_into("<4f", b, 160, float(width), float(height), 1.0, 0.0)
    for i in range(73):
        struct.pack_into("<4f", b, 192 + 16 * i, 1.0, 1.0, 1.0, 1.0)
    return bytes(b)


# ---- the binding, from the descriptions alone -----------------------------------------

def vertex_stream(spec: dict, vs_res: dict) -> tuple[list[dict[int, int]], bytes, list[str]]:
    """The vertex program's inputs, as its description lists them: per vertex the
    words the reference reads (a[0x80 + 16 L + 4 c]), the interleaved stream and
    vkdraw's attribute lines. Vulkan fills a component the format lacks with 0,
    0, 0, 1, and so does the reference."""
    stream = spec["vertex_stream"]
    if "attributes" in stream:
        raise InputError("vertex_stream.attributes is no longer read: the vertex inputs come from the vertex description")
    inputs = vs_res.get("vertex_inputs")
    if inputs is None:
        raise InputError("the vertex description lists no vertex_inputs")
    lines, offset = [], 0
    for v in inputs:
        kind = INPUT_FORMATS.get(v["type"])
        if kind is None or not 1 <= v["components"] <= 4 or v["component"] != 0:
            raise InputError(f"vertex input {v['location']}: {v['components']} x {v['type']} at component {v['component']} is not drawn")
        lines.append(f"attribute {v['location']} {('r', 'rg', 'rgb', 'rgba')[v['components'] - 1]}32{kind} {offset}")
        offset += 4 * v["components"]
    words_per_vertex = offset // 4
    vertices, blob = [], bytearray()
    for vals in stream["vertices"]:
        if len(vals) != words_per_vertex:
            raise InputError(f"a vertex has {len(vals)} values, the vertex inputs take {words_per_vertex}")
        words, k = {}, 0
        for v in inputs:
            base = 0x80 + 16 * v["location"]
            for c in range(4):
                if c < v["components"]:
                    x = vals[k + c]
                    bits = se.float_to_bits(x) if v["type"] == "float32" else int(x) & 0xFFFFFFFF
                else:
                    bits = se.float_to_bits(1.0 if c == 3 else 0.0) if v["type"] == "float32" else (1 if c == 3 else 0)
                words[base + 4 * c] = bits
                if c < v["components"]:
                    blob += struct.pack("<I", bits)
            k += v["components"]
        vertices.append(words)
    return vertices, bytes(blob), lines


def texture_specs(spec: dict) -> dict[str, dict[int, dict]]:
    """The spec's textures by stage and handle word offset; one without a stage serves both."""
    out: dict[str, dict[int, dict]] = {"vs": {}, "ps": {}}
    for t in spec.get("textures", []):
        for st in ([t["stage"]] if "stage" in t else ["vs", "ps"]):
            out[st][int(t["handle"], 0)] = t
    return out


def bind(res: dict, bank_data: dict, mem: "Memory", textures: dict, width: int, height: int) -> list[tuple]:
    """vkdraw directives for the pair, from the two descriptions alone (layout `uni`):
    ("uniform"|"storage", binding, stages, bytes), ("image"|"sampler", entry, texture),
    ("push", offset, value), ("global", images, samplers), ("feature", name)."""
    out: list[tuple] = []
    for st in ("vs", "ps"):
        r = res[st]
        if r.get("schema") != SCHEMA:
            raise InputError(f"the {st} description is schema {r.get('schema')}, not {SCHEMA}: translate the pair again")
        if r["layout"]["name"] != "uni" or not r["layout"]["fits"]:
            raise InputError(f"the {st} module does not fit the layout uni: {r['layout'].get('overflow')}")
        if r["images"]:
            raise InputError("storage images are not drawn by this harness")
    sb = res["vs"]["support_buffer"]
    out.append(("uniform", sb["binding"], "vf", support_buffer(width, height)))
    entries: dict[tuple[str, int], int] = {}
    heaps = set()
    for st in ("vs", "ps"):
        r, letter = res[st], STAGE_LETTER[st]
        heaps.add((r["answers"]["heap_images"], r["answers"]["heap_samplers"]))
        for cb in r["cbufs"]:
            out.append(("uniform", cb["binding"], letter, bytes(bank_data[st].get(cb["slot"], bytearray(BANK_BYTES)))))
        ra = r.get("raw_access", {})
        if ra.get("page_table") or ra.get("physical_pointers"):
            raise InputError("a module with raw global access is not drawn: it needs the runtime's page table (raw_access in the description)")
        contract = r["storage_buffer_contract"]
        align = contract["alignment"]
        for s in r["sbufs"]:
            if s["role"] != "guest":
                raise InputError(f"a {s['role']} storage buffer (raw global access) is not drawn")
            bank = bank_data[st].get(s["cb_slot"], bytearray(BANK_BYTES))
            addr = struct.unpack_from("<Q", bank, s["address_byte_offset"])[0]
            size = struct.unpack_from("<I", bank, s["size_byte_offset"])[0]
            base = addr & ~(align - 1)
            out.append(("storage", s["binding"], letter, mem.window(base, max(16, size + (addr - base)))))
        for t in r["textures"]:
            if t["via"] != "push_slot":
                raise InputError(f"a {t['via']} texture is not drawn: only textures of push constant slots are")
            if t["kind"] != "combined" or t["type"] not in TEXTURE_LAYERS or t["array_length"] != 1:
                raise InputError(f"{st} texture of slot {t['push_slot']}: a {t['kind']} {t['type']} texture"
                                 f" (array of {t['array_length']}) is not drawn")
            handle = t["tex_word_offset"]
            if handle not in textures[st]:
                raise InputError(f"{st} texture handle {handle:#x} has no spec entry")
            if textures[st][handle].kind != t["type"]:
                raise InputError(f"{st} texture handle {handle:#x} is {t['type']} in the description, {textures[st][handle].kind} in the spec")
            # Each texture gets its own image and sampler entry (from 1, so that a zero handle reads nothing
            # drawn); the slot holds the handle: texture ID in bits 0..19, sampler ID in bits 20..31.
            if (st, handle) not in entries:
                entries[(st, handle)] = len(entries) + 1
                out.append(("image", entries[(st, handle)], textures[st][handle]))
                out.append(("sampler", entries[(st, handle)], textures[st][handle]))
            e = entries[(st, handle)]
            out.append(("push", t["push_offset"], e | (e << 20)))
    if len(heaps) != 1:
        raise InputError(f"the two descriptions answer different heap sizes: {sorted(heaps)}")
    (images, samplers), = heaps
    out.append(("global", images, samplers))
    return out


# ---- one scenario ------------------------------------------------------------------

def run_scenario(spec: dict, scen: dict, progs: dict, res: dict, work: Path, args) -> dict:
    name = scen["name"]
    sdir = work / name
    sdir.mkdir(parents=True, exist_ok=True)
    W, H = spec["size"]
    banks = merge_banks(spec.get("banks", {}), scen.get("banks", {}))
    mem = Memory(spec.get("global", []) + scen.get("global", []))
    bank_data = stage_banks(banks, progs)

    def cbank(stage):
        def read(bank: int, off: int) -> int:
            data = bank_data[stage].get(bank)
            return struct.unpack_from("<I", data, off)[0] if data is not None else 0
        return read

    vertices, vblob, attr_lines = vertex_stream(spec, res["vs"])
    tris = [tuple(t) for t in spec["triangles"]]
    order = [i for t in tris for i in t]
    stride = len(vblob) // max(1, len(spec["vertex_stream"]["vertices"]))
    order_blob = b"".join(vblob[i * stride:(i + 1) * stride] for i in order)
    vfile = sdir / "vertices.dat" if stride else "-"  # no inputs: no vertex buffer
    if stride:
        (sdir / "vertices.dat").write_bytes(order_blob)

    tex_specs = texture_specs(spec)
    textures = {st: {h: make_texture(t) for h, t in by.items()} for st, by in tex_specs.items()}

    lines = [f"vs {sdir / 'vs.spv'}", f"fs {sdir / 'fs.spv'}", f"out {sdir / 'gpu.f32'}",
             f"size {W} {H}", "clear " + " ".join(repr(float(c)) for c in spec["clear"]),
             f"validation {args.validation}", f"vertices {len(order)} {stride} {vfile}"]
    lines += attr_lines
    shutil.copyfile(progs["vs"]["spv"], sdir / "vs.spv")
    shutil.copyfile(progs["ps"]["spv"], sdir / "fs.spv")
    for kind, *rest in bind(res, bank_data, mem, textures, W, H):
        if kind in ("uniform", "storage"):
            binding, stages, blob = rest
            path = sdir / f"binding{binding}.dat"
            path.write_bytes(blob)
            lines.append(f"{kind} {binding} {stages} {path}")
        elif kind == "image":
            entry, tx = rest
            path = sdir / f"image{entry}.dat"
            path.write_bytes(b"".join(struct.pack("<4f", *px) for px in tx.texels))
            lines.append(f"image {entry} {tx.kind} {tx.width} {tx.height} {tx.layers} {path}")
        elif kind == "sampler":
            entry, tx = rest
            lines.append(f"sampler {entry} {tx.filter} {tx.address}")
        else:
            lines.append(" ".join(str(x) for x in (kind, *rest)))

    needs = set()
    for st in ("vs", "ps"):
        for cap in res[st]["capabilities"]:
            if cap not in CAPABILITIES:
                raise InputError(f"capability {cap} has no known device requirement")
            if CAPABILITIES[cap]:
                needs.add(CAPABILITIES[cap])
        if needs_stores_and_atomics(res[st]):
            needs.add("vertexPipelineStoresAndAtomics" if st == "vs" else "fragmentStoresAndAtomics")
    for n in sorted(needs):
        lines.append(n if n.startswith("subgroup") else f"feature {n}")
    (sdir / "draw.txt").write_text("\n".join(lines) + "\n")

    # GPU.
    # Implicit layers (overlays and recorders registered on the machine) stay out
    # of the measurement unless asked for.
    env = dict(os.environ)
    if not args.keep_implicit_layers:
        env["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"
    t0 = time.perf_counter()
    gpu = subprocess.run([str(args.vkdraw), str(sdir / "draw.txt")], capture_output=True, text=True, env=env)
    gpu_s = time.perf_counter() - t0
    (sdir / "vkdraw-stderr.txt").write_text(gpu.stderr)
    if gpu.returncode != 0:
        return {"scenario": name, "pass": False, "error": f"vkdraw exit {gpu.returncode}: {gpu.stderr.strip()[-400:]}"}
    vk = json.loads(gpu.stdout.strip().splitlines()[-1])
    raw = (sdir / "gpu.f32").read_bytes()
    got = [struct.unpack_from("<4f", raw, 16 * k) for k in range(W * H)]

    if args.gpu_only:
        return gpu_report(name, spec, got, vk, gpu_s, sdir, args)

    # CPU. The GPU draws the vertices in triangle order without an index buffer, so a vertex's index is its
    # position in that order: the reference shades the same list.
    drawn = [vertices[i] for i in order]
    tris = [(3 * k, 3 * k + 1, 3 * k + 2) for k in range(len(tris))]
    draw = ref.Draw(W, H, progs["vs"]["prog"], progs["ps"]["prog"], progs["ps"]["sph"], drawn, tris,
                    cbank("vs"), cbank("ps"), mem.load, textures["ps"], vs_textures=textures["vs"],
                    swap_uv=args.counter == "swap-uv", swap_interpolation=args.counter == "swap-interpolation",
                    tex_sources_from_ra=args.counter == "tex-sources-from-ra")
    t0 = time.perf_counter()
    exp = ref.render(draw)
    cpu_s = time.perf_counter() - t0

    return compare(name, spec, exp, got, vk, gpu_s, cpu_s, sdir, args)


def gpu_report(name, spec, got, vk, gpu_s, sdir: Path, args) -> dict:
    """--gpu-only: the binding and the validation layer, without the reference."""
    clear = tuple(float(c) for c in spec["clear"])
    covered = sum(1 for g in got if g != clear)
    nonfinite = sum(1 for g in got if any(c != c or c in (math.inf, -math.inf) for c in g))
    clean = vk["validation_errors"] == 0 and (vk["validation_warnings"] == 0 or args.validation == "gpu")
    write_png(sdir / "gpu.png", *spec["size"], got)
    return {"scenario": name, "pass": clean, "gpu_only": True, "device": vk["device"],
            "driver": vk["driver"], "api": vk["api"], "validation": vk["validation"],
            "validation_errors": vk["validation_errors"], "validation_warnings": vk["validation_warnings"],
            "loader_messages": vk["loader_messages"], "pixels": {"changed": covered, "nonfinite": nonfinite},
            "gpu_seconds": round(gpu_s, 3)}


def compare(name, spec, exp, got, vk, gpu_s, cpu_s, sdir: Path, args) -> dict:
    W, H = spec["size"]
    clear = tuple(float(c) for c in spec["clear"])
    counts = {k: 0 for k in (ref.COLOR, ref.KILLED, ref.OUTSIDE, ref.EDGE, ref.AMBIGUOUS)}
    mismatches, worst_abs, worst_rel, first = 0, 0.0, 0.0, []
    status = []
    for k in range(W * H):
        kind = exp.kind[k]
        counts[kind] += 1
        g = got[k]
        ok = True
        if kind == ref.COLOR:
            for c in range(4):
                e = exp.rgba[k][c]
                d = abs(g[c] - e)
                worst_abs = max(worst_abs, d)
                if e != 0:
                    worst_rel = max(worst_rel, d / abs(e))
                if not d <= args.atol + args.rtol * abs(e):
                    ok = False
        elif kind in (ref.KILLED, ref.OUTSIDE):
            ok = g == clear
        else:
            status.append(2)
            continue
        status.append(0 if ok else 1)
        if not ok:
            mismatches += 1
            if len(first) < 8:
                first.append({"x": k % W, "y": k // W, "kind": kind, "gpu": [round(v, 6) for v in g],
                              "expected": [round(v, 6) for v in exp.rgba[k]] if exp.rgba[k] else None})
    compared = counts[ref.COLOR] + counts[ref.KILLED] + counts[ref.OUTSIDE]
    # GPU-assisted validation always warns about itself (its speed, the features
    # it forces on, the checks it turns off), so under it only errors count.
    clean = vk["validation_errors"] == 0 and (vk["validation_warnings"] == 0 or args.validation == "gpu")
    caught = mismatches > 0
    passed = (caught if args.counter else mismatches == 0) and clean and counts[ref.COLOR] > 0
    write_png(sdir / "gpu.png", W, H, got)
    write_png(sdir / "expected.png", W, H, [e if e else clear for e in exp.rgba])
    write_png(sdir / "status.png", W, H, [((0.1, 0.8, 0.2, 1), (1, 0, 0, 1), (0.4, 0.4, 0.4, 1))[s] for s in status])
    return {
        "scenario": name, "pass": passed, "counter": args.counter, "device": vk["device"], "driver": vk["driver"],
        "api": vk["api"], "validation": vk["validation"], "validation_errors": vk["validation_errors"],
        "validation_warnings": vk["validation_warnings"], "loader_messages": vk["loader_messages"],
        "pixels": counts, "compared": compared,
        "mismatches": mismatches, "max_abs_err": worst_abs, "max_rel_err": worst_rel,
        "tolerance": {"atol": args.atol, "rtol": args.rtol}, "first_mismatches": first,
        "gpu_seconds": round(gpu_s, 3), "cpu_seconds": round(cpu_s, 1),
        "vertex_outputs": [{f"{a:#x}": se.bits_to_float(v) for a, v in sorted(o.items())} for o in exp.vertex_outputs],
    }


def write_png(path: Path, w: int, h: int, pixels) -> None:
    """An 8-bit RGBA preview: each component clamped to [0, 1]."""
    rows = bytearray()
    for y in range(h):
        rows.append(0)
        for x in range(w):
            for c in pixels[y * w + x]:
                c = 0.0 if c != c else min(max(c, 0.0), 1.0)
                rows.append(int(c * 255 + 0.5))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)


# ---- main ------------------------------------------------------------------------

def find_nvdisasm() -> str | None:
    hit = shutil.which("nvdisasm")
    if hit:
        return hit
    cuda = os.environ.get("CUDA_PATH")
    if cuda and (Path(cuda) / "bin" / "nvdisasm.exe").is_file():
        return str(Path(cuda) / "bin" / "nvdisasm.exe")
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--vkdraw", type=Path, default=ROOT / "build" / "vkdraw" / ("vkdraw.exe" if os.name == "nt" else "vkdraw"))
    ap.add_argument("--nvdisasm", default=None)
    ap.add_argument("--validation", choices=("on", "gpu", "sync", "off"), default="on")
    ap.add_argument("--counter", choices=("swap-uv", "swap-interpolation", "tex-sources-from-ra"), default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--keep-implicit-layers", action="store_true",
                    help="leave the machine's implicit Vulkan layers on (they are disabled by default)")
    ap.add_argument("--gpu-only", action="store_true",
                    help="no reference: check the binding and the validation layer only (no nvdisasm needed)")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args(argv)

    nvdisasm = args.nvdisasm or find_nvdisasm()
    if not nvdisasm and not args.gpu_only:
        print("first_draw: nvdisasm (CUDA toolkit 11.8 or later) is required; give --nvdisasm.", file=sys.stderr)
        return 2
    if not args.vkdraw.is_file():
        print(f"first_draw: no harness at {args.vkdraw}; run tools/build_vkdraw.py.", file=sys.stderr)
        return 2
    try:
        spec = json.loads(args.spec.read_text())
        jobs = {}
        wanted = {spec["vs"], spec["ps"]}
        with open(spec["jobs"]) as f:
            for line in f:
                j = json.loads(line)
                if j["key"] in wanted:
                    jobs[j["key"]] = j
        work = args.work.resolve()
        work.mkdir(parents=True, exist_ok=True)
        progs, res = {}, {}
        for st in ("vs", "ps"):
            key = spec[st]
            prog, sph = (None, None) if args.gpu_only else load_program(jobs, key, work, nvdisasm)[:2]
            spv = Path(spec["spv_dir"]) / f"{key}.spv"
            res[st] = json.loads((Path(spec["spv_dir"]) / f"{key}.res.json").read_text())
            if res[st]["stage"] != st:
                raise InputError(f"{key} is a {res[st]['stage']} program, not {st}")
            c1 = bytes.fromhex(jobs[key].get("c1", "")) if key in jobs else b""
            if len(c1) > BANK_BYTES:
                raise InputError(f"{key}: c1 holds {len(c1)} bytes, more than a bank")
            progs[st] = {"prog": prog, "sph": sph, "spv": spv, "c1": c1}
    except (InputError, OSError, KeyError, ValueError) as e:
        print(f"first_draw: {e}", file=sys.stderr)
        return 2

    reports = []
    for scen in spec["scenarios"]:
        if args.scenario and scen["name"] != args.scenario:
            continue
        try:
            rep = run_scenario(spec, scen, progs, res, work, args)
        except InputError as e:
            print(f"first_draw: {scen['name']}: {e}", file=sys.stderr)
            return 2
        except se.Unsupported as e:
            print(f"first_draw: {scen['name']}: the reference cannot evaluate this: {e}", file=sys.stderr)
            return 3
        reports.append(rep)
        line = {k: rep.get(k) for k in ("scenario", "pass", "pixels", "mismatches", "max_abs_err", "validation_errors",
                                        "validation_warnings", "loader_messages", "driver", "error")}
        print("first_draw:", json.dumps(line), flush=True)
    suffix = f"-{args.counter}" if args.counter else ""
    (work / f"report-{args.validation}{suffix}.json").write_text(json.dumps(reports, indent=2))
    return 0 if reports and all(r["pass"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
