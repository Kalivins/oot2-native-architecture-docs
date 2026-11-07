#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Spike S3: what pipeline libraries, shader objects and descriptor layouts cost.

    py -3 tools/pipecost/pipe_cost.py census --s1 DIR --work DIR
    py -3 tools/pipecost/pipe_cost.py select --work DIR [--pairs N] [--seed S]
    py -3 tools/pipecost/pipe_cost.py prepare --s1 DIR --work DIR [--cache-pipelines N] [--compositions N]
    py -3 tools/pipecost/pipe_cost.py builds --work DIR [--threads N] [--phases libs,mono,objects,objects-linked]
    py -3 tools/pipecost/pipe_cost.py cacheload --work DIR [--threads N] [--phases build,build-binaries,load,binaries]
    py -3 tools/pipecost/pipe_cost.py draws|replay --work DIR --variants KIND/BIND,... --tag T [--draws N]
    py -3 tools/pipecost/pipe_cost.py report --work DIR

census reads spike S1's translated modules (run/default/spv/<key>.res.json and
validation.tsv, never a listing of the spv folder) and S1's programs.tsv, and
writes one row per vertex and pixel module with its set 0 resources, plus the
vertex and pixel pairs that share a first label (S2's pairing, B3 and B4).
select draws a sample of pairs spread over module sizes. prepare writes the
job bodies of the harness (tools/pipecost/vkpc_*.c): the sample's pairs with
their pixel set 0 bindings moved past the vertex ones (the harness moves them
in the SPIR-V at load, `rebind`), every valid pair in two or three colour
formats for the launch test, and synthetic modules (glslangValidator) whose
descriptor counts follow the corpus for the per-draw tests. builds, cacheload,
draws and replay run the harness, one process per measurement, after checking
free RAM, CPU load and that no emulator is running; report turns the rows and
summaries into report.json.

Everything this writes goes under --work and holds program keys and modules:
none of it may enter the repository.

Exit codes: 0 done, 1 a measurement failed, 2 an input or a tool is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PUSH_LIMIT = 32  # maxPushDescriptors on the reference GPU (vulkaninfo, driver 617.14)

# ---- module census -----------------------------------------------------------

# Descriptor kinds of a set 0 resource, as the harness names them.
UBO, SSBO, TEX, TEXBUF, SEPTEX, SAMPLER, IMG, IMGBUF = "ubo", "ssbo", "tex", "texbuf", "septex", "sampler", "img", "imgbuf"
KINDS = (UBO, SSBO, TEX, TEXBUF, SEPTEX, SAMPLER, IMG, IMGBUF)


def texture_kind(t: dict) -> str:
    """The descriptor kind of one entry of a description's textures or images list."""
    if t.get("kind") == "sampler":
        return SAMPLER
    if t.get("type") == "buffer":
        return TEXBUF
    if t.get("kind") == "separate_texture":
        return SEPTEX
    return TEX


def describe(res: dict) -> dict:
    """The set 0 and set 1 needs of one module from its resource description.

    Set 0 bindings are listed as (binding, kind, count); the support buffer is
    binding 0 and is counted as a uniform buffer. Set 1 (the global arrays)
    only marks use: those arrays are bound once per command buffer.
    """
    if res.get("schema", 1) != 1:
        # Schema 2 (ticket 23) numbers the layout `uni`: buffers at set 1, textures in push constant slots.
        raise ValueError(f"a schema {res['schema']} description: this census reads schema 1 (set 0 per draw)")
    binds = [(res["support_buffer"]["binding"], UBO, 1)]
    binds += [(c["binding"], UBO, 1) for c in res["cbufs"]]
    binds += [(s["binding"], SSBO, 1) for s in res["sbufs"]]
    set1 = []
    for t in res["textures"]:
        if t["set"] != 0:
            set1.append(t["binding"])
            continue
        binds.append((t["binding"], texture_kind(t), max(1, t.get("array_length", 1))))
    for t in res["images"]:
        if t["set"] != 0:
            set1.append(t["binding"])
            continue
        binds.append((t["binding"], IMGBUF if t.get("type") == "buffer" else IMG, max(1, t.get("array_length", 1))))
    g = res.get("graphics") or {}
    ra = res.get("raw_access") or {}
    return {
        "stage": res["stage"],
        "spv_bytes": res["spv_bytes"],
        "binds": sorted(binds),
        "set1": sorted(set(set1)),
        "raw": bool(ra.get("physical_pointers") or ra.get("page_table")),
        "rt_mask": g.get("frag_out_map", 0),
        "caps": res.get("capabilities", []),
    }


def count(desc: dict, shared_support: bool = False) -> int:
    """Set 0 descriptors of a module; shared_support leaves its support buffer out."""
    n = sum(c for _, _, c in desc["binds"])
    return n - 1 if shared_support else n


def kind_counts(desc: dict) -> dict[str, int]:
    out = {k: 0 for k in KINDS}
    for _, kind, c in desc["binds"]:
        out[kind] += c
    return out


def pair_count(vs: dict, ps: dict) -> int:
    """Set 0 descriptors of a pair once stages are disjoint and the support buffer shared (S2, B4)."""
    return count(vs) + count(ps, shared_support=True)


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        return [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


def census(s1: Path, work: Path) -> int:
    run = s1 / "run" / "default"
    programs = {r["job_key"]: r for r in read_tsv(s1 / "split" / "programs.tsv")}
    results = {r["key"]: r for r in read_tsv(run / "results.tsv")}
    valid = {r["key"]: r["strict_ok"] == "1" for r in read_tsv(run / "validation.tsv")}
    rows = []
    t0 = time.perf_counter()
    for key, prog in programs.items():
        if prog["stage"] not in ("vs", "ps"):
            continue
        r = results.get(key)
        if not r or r["status"] != "ok":
            continue
        res = json.loads((run / "spv" / f"{key}.res.json").read_text(encoding="utf-8"))
        d = describe(res)
        d.update(key=key, label=prog["first_label"], valid=valid.get(key, False))
        rows.append(d)
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "modules.jsonl", "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, separators=(",", ":")) + "\n")
    pairs = make_pairs(rows)
    with open(work / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, separators=(",", ":")) + "\n")
    summary = census_summary(rows, pairs)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    (work / "census.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def make_pairs(rows: list[dict]) -> list[dict]:
    """Vertex and pixel modules sharing a first label, one of each (S2's pairing)."""
    by_label: dict[str, dict[str, list[dict]]] = {}
    for d in rows:
        by_label.setdefault(d["label"], {}).setdefault(d["stage"], []).append(d)
    pairs = []
    for label, st in sorted(by_label.items()):
        if len(st.get("vs", [])) == 1 and len(st.get("ps", [])) == 1:
            vs, ps = st["vs"][0], st["ps"][0]
            pairs.append({"vs": vs["key"], "ps": ps["key"], "descriptors": pair_count(vs, ps),
                          "spv_bytes": vs["spv_bytes"] + ps["spv_bytes"],
                          "valid": vs["valid"] and ps["valid"], "raw": vs["raw"] or ps["raw"]})
    return pairs


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100) of a non-empty list."""
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(-(-q * len(s) // 100)) - 1))
    return s[k]


def census_summary(rows: list[dict], pairs: list[dict]) -> dict:
    out: dict = {"modules": {}, "pairs": {}}
    for st in ("vs", "ps"):
        ds = [d for d in rows if d["stage"] == st]
        kc = [kind_counts(d) for d in ds]
        out["modules"][st] = {
            "count": len(ds), "valid": sum(d["valid"] for d in ds), "raw": sum(d["raw"] for d in ds),
            "set1": sum(bool(d["set1"]) for d in ds),
            "descriptors": {"p50": percentile([count(d) for d in ds], 50),
                            "p99": percentile([count(d) for d in ds], 99), "max": max(count(d) for d in ds)},
            "over_push_limit": sum(count(d) > PUSH_LIMIT for d in ds),
            "max_by_kind": {k: max(c[k] for c in kc) for k in KINDS},
            "with_kind": {k: sum(c[k] > 0 for c in kc) for k in KINDS},
            "buffers_max": max(c[UBO] + c[SSBO] for c in kc),
            "spv_bytes": {"p50": percentile([d["spv_bytes"] for d in ds], 50),
                          "p99": percentile([d["spv_bytes"] for d in ds], 99),
                          "max": max(d["spv_bytes"] for d in ds)},
        }
    n = [p["descriptors"] for p in pairs]
    out["pairs"] = {"count": len(pairs), "valid": sum(p["valid"] for p in pairs),
                    "descriptors": {"p50": percentile(n, 50), "p99": percentile(n, 99), "max": max(n)},
                    "over_push_limit": sum(x > PUSH_LIMIT for x in n)}
    return out


# ---- sample ------------------------------------------------------------------

def select(work: Path, npairs: int, seed: int) -> int:
    """A sample of pairs spread evenly over total module size (deciles), valid modules only."""
    pairs = [json.loads(l) for l in open(work / "pairs.jsonl", encoding="utf-8")]
    usable = [p for p in pairs if p["valid"]]
    usable.sort(key=lambda p: (p["spv_bytes"], p["vs"], p["ps"]))
    rng = random.Random(seed)
    bins = 10
    per = -(-npairs // bins)
    sample = []
    for b in range(bins):
        part = usable[b * len(usable) // bins:(b + 1) * len(usable) // bins]
        sample += rng.sample(part, min(per, len(part)))
    sample = sample[:npairs] if len(sample) > npairs else sample
    # Pairs over the push limit are kept in the sample at their corpus share or more,
    # so every layout meets them.
    over = [p for p in usable if p["descriptors"] > PUSH_LIMIT and p not in sample]
    want_over = -(-len([p for p in usable if p["descriptors"] > PUSH_LIMIT]) * npairs // len(usable))
    have_over = sum(p["descriptors"] > PUSH_LIMIT for p in sample)
    if have_over < want_over:
        sample += rng.sample(over, min(len(over), want_over - have_over))
    (work / "sample.json").write_text(json.dumps({"seed": seed, "pairs": sample}, indent=0), encoding="utf-8")
    sizes = [p["spv_bytes"] for p in sample]
    print(json.dumps({"pairs": len(sample), "usable": len(usable),
                      "vs": len({p["vs"] for p in sample}), "ps": len({p["ps"] for p in sample}),
                      "over_push_limit": sum(p["descriptors"] > PUSH_LIMIT for p in sample),
                      "spv_bytes": {"min": min(sizes), "p50": percentile(sizes, 50), "max": max(sizes)}}))
    return 0


# ---- job files -----------------------------------------------------------------
#
# A job file (tools/pipecost/vkpc_main.c) is a run header (validation, cache,
# threads, out, run parameters) followed by a body (modules, sets, pairs) that
# prepare writes once per measurement.

REBIND = 128            # pixel set 0 bindings move by this; binding 0 (the shared support buffer) stays
PUSH_BYTES = 128        # page-table address and span (D15), then up to 28 texture indices
GLOBAL = [(0, "septex", 33025, "vf"), (1, "sampler", 4096, "vf")]  # the translator's pool arrays (set 1)
FORMATS = ("rgba8", "rgba16f", "r11g11b10f")
SPV_DIR = Path("run") / "default" / "spv"


def targets(rt_mask: int) -> int:
    """Colour attachments a pixel module writes: frag_out_map holds 4 component bits per target."""
    n = 0
    for t in range(8):
        if (rt_mask >> (4 * t)) & 0xF:
            n = t + 1
    return n


def pair_set(vs: dict, ps: dict) -> tuple:
    """A pair's set 0 with stages disjoint: the support buffer shared at binding 0,
    vertex bindings as the translator numbered them, pixel bindings moved by REBIND
    (the harness moves them in the pixel module's SPIR-V at load)."""
    out = {0: ("ubo", 1, "vf")}
    for b, kind, c in vs["binds"]:
        if b:
            out[b] = (kind, c, "v")
    for b, kind, c in ps["binds"]:
        if b:
            if b + REBIND in out:
                raise ValueError(f"binding {b} + {REBIND} collides with a vertex binding")
            out[b + REBIND] = (kind, c, "f")
    return tuple((b,) + out[b] for b in sorted(out))


def body_text(modules: list[tuple[str, str]], sets: list[tuple], pairs: list[tuple], *, rebind: int,
              global_set: int, depth: str, params: dict | None = None) -> str:
    lines = [f"push {PUSH_BYTES}", f"depth {depth}"] + [f"format {f}" for f in FORMATS]
    if rebind:
        lines.append(f"rebind {rebind}")
    lines.append(f"global {global_set}")
    lines += [f"bind {b} {k} {c} {st}" for b, k, c, st in GLOBAL]
    lines += [f"module {st} {path}" for st, path in modules]
    for s in sets:
        lines.append("set")
        lines += [f"bind {b} {k} {c} {st}" for b, k, c, st in s]
    lines += ["pair " + " ".join(str(x) for x in p) for p in pairs]
    lines += [f"param {k} {v}" for k, v in (params or {}).items()]
    return "\n".join(lines) + "\n"


def load_modules(work: Path) -> dict[str, dict]:
    return {d["key"]: d for d in (json.loads(l) for l in open(work / "modules.jsonl", encoding="utf-8"))}


def real_body(s1: Path, mods: dict[str, dict], plist: list[dict], formats_per_pair: list[int] | None = None) -> str:
    """Pipelines of real translated pairs: one per pair and colour format, a pair's
    formats listed together (the cache mode loads a pair's modules once)."""
    spv = s1 / SPV_DIR
    modules: list[tuple[str, str]] = []
    index: dict[str, int] = {}
    sets: list[tuple] = []
    set_index: dict[tuple, int] = {}
    pairs = []
    for n, p in enumerate(plist):
        vs, ps = mods[p["vs"]], mods[p["ps"]]
        for key, st in ((p["vs"], "v"), (p["ps"], "f")):
            if key not in index:
                index[key] = len(modules)
                modules.append((st, str(spv / f"{key}.spv")))
        s = pair_set(vs, ps)
        if s not in set_index:
            set_index[s] = len(sets)
            sets.append(s)
        for f in range(formats_per_pair[n] if formats_per_pair else 1):
            pairs.append((index[p["vs"]], index[p["ps"]], set_index[s], targets(ps["rt_mask"]), f))
    return body_text(modules, sets, pairs, rebind=REBIND, global_set=1, depth="d32s8")


# ---- synthetic modules for the per-draw measurements -------------------------------

# Universal push set (layout `uni`): fixed slots per stage and kind, sized by the
# corpus maxima (census: vertex 8 constant banks besides the support buffer and 4
# storage buffers, pixel 11 and 5), 29 descriptors in all.
UNI_SLOTS = {("v", "ubo"): range(1, 9), ("v", "ssbo"): range(9, 13), ("f", "ubo"): range(13, 24),
             ("f", "ssbo"): range(24, 29)}
TEX_FIRST = 1000
TEX_SLOTS = 28


def composition(vs: dict, ps: dict) -> dict:
    """Descriptor counts of a pair, by stage and kind (support buffer apart); images count as textures."""
    out = {}
    for st, d in (("v", vs), ("f", ps)):
        k = kind_counts(d)
        out[st] = {"ubo": k[UBO] - 1, "ssbo": k[SSBO], "tex": k[TEX] + k[IMG] + k[TEXBUF] + k[SEPTEX]}
    return out


def composition_count(comp: dict) -> int:
    """Set 0 descriptors of a composition, the shared support buffer included."""
    return 1 + sum(comp[st][k] for st in "vf" for k in ("ubo", "ssbo", "tex"))


def synth_bindings(layout: str, comp: dict) -> dict[str, list[tuple[int, str, int]]]:
    """Per stage: (binding, kind, index) of each resource. dkb and d13 number one
    set for both stages; uni puts buffers in their fixed slots and textures at
    TEX_FIRST + push constant slot."""
    out: dict[str, list] = {"v": [], "f": []}
    nxt = 1
    tex_slot = 0
    for st in ("v", "f"):
        for kind in ("ubo", "ssbo", "tex"):
            for i in range(comp[st][kind]):
                if layout == "uni":
                    if kind == "tex":
                        out[st].append((TEX_FIRST + tex_slot, "tex", tex_slot))
                        tex_slot += 1
                    else:
                        out[st].append((UNI_SLOTS[(st, kind)][i], kind, i))
                else:
                    out[st].append((nxt, kind, i))
                    nxt += 1
    if tex_slot > TEX_SLOTS:
        raise ValueError(f"{tex_slot} textures do not fit {TEX_SLOTS} push constant slots")
    return out


def synth_glsl(stage: str, layout: str, binds: dict) -> str:
    """A vertex or fragment shader that reads every resource of its stage once."""
    s = 0 if layout == "dkb" else 1
    src = ["#version 460", f"layout(set = {s}, binding = 0) uniform Sup {{ vec4 s[4]; }} sup;"]
    if layout != "dkb":
        src += ["layout(set = 0, binding = 0) uniform texture2D gImages[33025];",
                "layout(set = 0, binding = 1) uniform sampler gSamplers[4096];",
                "layout(push_constant) uniform PC { uvec4 pt; uint tex[28]; } pc;"]
    if stage == "v":
        src.append("layout(location = 0) out vec4 o;")
        body = ["    vec4 a = sup.s[0];"]
    else:
        src += ["layout(location = 0) in vec4 o;", "layout(location = 0) out vec4 c;"]
        body = ["    vec4 a = o + sup.s[1];"]
    uv = "vec2(0.5)" if stage == "v" else "o.xy"
    for b, kind, i in binds[stage]:
        name = f"{kind}{stage}{i}"
        if kind == "ubo":
            src.append(f"layout(set = {s}, binding = {b}) uniform U_{name} {{ vec4 v[4]; }} {name};")
            body.append(f"    a += {name}.v[{i % 4}];")
        elif kind == "ssbo":
            src.append(f"layout(set = {s}, binding = {b}, std430) readonly buffer B_{name} {{ vec4 v[]; }} {name};")
            body.append(f"    a += {name}.v[{i % 4}];")
        elif layout == "uni":
            body.append(f"    a += textureLod(sampler2D(gImages[pc.tex[{b - TEX_FIRST}]], gSamplers[{i % 16}]), {uv}, 0.0);")
        else:
            src.append(f"layout(set = {s}, binding = {b}) uniform sampler2D {name};")
            body.append(f"    a += textureLod({name}, {uv}, 0.0);")
    if layout == "d13":
        body.append("    a.x += float(pc.pt.x & 1u) * 1e-6;")
        if stage == "f":
            body.append("    a += texture(sampler2D(gImages[0], gSamplers[0]), o.zw);")
    if stage == "v":
        body += ["    uint id = uint(gl_VertexIndex);",
                 "    vec2 p = vec2(float(id & 1u), float(id >> 1u)) * 0.03 + fract(a.xy * 0.37) * 1.8 - 0.9;",
                 "    gl_Position = vec4(p, 0.0, 1.0);", "    o = a;"]
    else:
        body.append("    c = a * 0.001 + vec4(0.1);")
    return "\n".join(src + ["void main() {"] + body + ["}"]) + "\n"


def glslang() -> str:
    sdk = os.environ.get("VULKAN_SDK")
    exe = Path(sdk) / "Bin" / "glslangValidator.exe" if sdk else None
    if exe is not None and exe.is_file():
        return str(exe)
    found = shutil.which("glslangValidator")
    if found:
        return found
    raise FileNotFoundError("no glslangValidator: set VULKAN_SDK or put it on PATH")


def synth_body(work: Path, ncomp: int, seed: int) -> tuple[str, dict]:
    """Synthetic pairs for the draws and replay modes: ncomp compositions drawn
    from the valid corpus pairs, each written for the layouts dkb, d13 and uni."""
    mods = load_modules(work)
    pairs = [json.loads(l) for l in open(work / "pairs.jsonl", encoding="utf-8")]
    rng = random.Random(seed)
    comps = [composition(mods[p["vs"]], mods[p["ps"]]) for p in rng.sample([p for p in pairs if p["valid"]], ncomp)]
    out = work / "synth"
    out.mkdir(parents=True, exist_ok=True)
    gl = glslang()
    modules, sets, plist = [], [], []
    first = {}
    for layout in ("dkb", "d13", "uni"):
        first[layout] = len(plist)
        for c, comp in enumerate(comps):
            binds = synth_bindings(layout, comp)
            idx = []
            for st, ext in (("v", "vert"), ("f", "frag")):
                src = out / f"{layout}{c}.{ext}"
                spv = out / f"{layout}{c}.{ext}.spv"
                src.write_text(synth_glsl(st, layout, binds), encoding="utf-8")
                r = subprocess.run([gl, "-V", "--target-env", "vulkan1.3", "-o", str(spv), str(src)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"glslang failed on {src}: {r.stdout[-800:]}")
                idx.append(len(modules))
                modules.append((st, str(spv)))
            s = [(0, "ubo", 1, "vf")] + [(b, kind, 1, st) for st in ("v", "f") for b, kind, _ in binds[st]]
            sets.append(tuple(s))
            plist.append((idx[0], idx[1], len(sets) - 1, 1, 0))
    sets.append(tuple([(0, "ubo", 1, "vf")] + [(b, kind, 1, st) for (st, kind), r in UNI_SLOTS.items() for b in r]))
    counts = sorted(composition_count(c) for c in comps)
    meta = {"compositions": ncomp, "first": first, "uni_set": len(sets) - 1,
            "over_push_limit": sum(n > PUSH_LIMIT for n in counts), "descriptors": counts}
    return body_text(modules, sets, plist, rebind=0, global_set=0, depth="none", params={"uni_set": len(sets) - 1}), meta


# ---- running the harness ---------------------------------------------------------------

def machine_load() -> dict:
    """Available RAM, total CPU load (two samples a second apart) and any emulator running."""
    ps = ("$o=Get-CimInstance Win32_OperatingSystem;"
          "$a=(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter \"Name='_Total'\").PercentProcessorTime;"
          "Start-Sleep -Milliseconds 1000;"
          "$b=(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter \"Name='_Total'\").PercentProcessorTime;"
          "$h=(Get-Process | Where-Object { $_.Name -match '^(lichii|eden|yuzu|ryujinx|oot)' } | "
          "ForEach-Object { $_.Name }) -join ',';"
          "Write-Output ('{0} {1} {2} {3}' -f [int]($o.FreePhysicalMemory/1KB), $a, $b, $h)")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    parts = r.stdout.split()
    if len(parts) < 3:
        return {"error": r.stderr.strip()[:200]}
    return {"available_mb": int(parts[0]), "cpu_percent": [int(parts[1]), int(parts[2])],
            "emulators": parts[3] if len(parts) > 3 else ""}


def run_harness(work: Path, exe: Path, mode: str, body: str, head: list[str], tag: str,
                phase: str | None = None, env_extra: dict | None = None, min_available_mb: int = 3000) -> dict:
    """Runs one measurement after checking the machine; keeps the summary line,
    the per-item rows and the harness's messages under work/results."""
    load = machine_load()
    if load.get("emulators"):
        raise RuntimeError(f"{load['emulators']} is running: it builds pipelines too, so the times would be wrong")
    if load.get("available_mb", 0) < min_available_mb:
        raise RuntimeError(f"only {load.get('available_mb')} MB available, {min_available_mb} needed")
    jobs = work / "jobs"
    res = work / "results"
    res.mkdir(parents=True, exist_ok=True)
    job = jobs / f"{tag}.txt"
    job.write_text("\n".join(head + [f"out {res / (tag + '.tsv')}"]) + "\n" + (jobs / body).read_text(encoding="utf-8"),
                   encoding="utf-8")
    env = dict(os.environ)
    env["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"
    env.update(env_extra or {})
    cmd = [str(exe), mode, str(job)] + ([phase] if phase else [])
    print(f"pipe_cost: {tag}: available {load.get('available_mb')} MB, cpu {load.get('cpu_percent')} %", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    seconds = time.perf_counter() - t0
    (res / f"{tag}.stderr.txt").write_text(r.stderr, encoding="utf-8")
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"{tag}: exit {r.returncode}: {r.stderr.strip()[-1500:]}")
    summary = json.loads(r.stdout.strip().splitlines()[-1])
    summary.update(tag=tag, process_s=round(seconds, 2), load_before=load)
    with open(res / "summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"pipe_cost: {tag}: done in {seconds:.1f} s", flush=True)
    return summary


def header(threads: int, cache: str, validation: str = "off", params: dict | None = None) -> list[str]:
    return ([f"validation {validation}", f"cache {cache}", f"threads {threads}", "device RTX 3060 Ti"]
            + [f"param {k} {v}" for k, v in (params or {}).items()])


def exe_path() -> Path:
    exe = ROOT / "build" / "vkpipecost" / "vkpipecost.exe"
    if not exe.is_file():
        raise FileNotFoundError(f"no harness at {exe}; run tools/build_vkdraw.py --harness vkpipecost")
    return exe


# ---- commands ----------------------------------------------------------------------

def prepare(s1: Path, work: Path, cache_pipelines: int, comps: int, seed: int) -> int:
    mods = load_modules(work)
    sample = json.loads((work / "sample.json").read_text(encoding="utf-8"))["pairs"]
    jobs = work / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "build.body").write_text(real_body(s1, mods, sample), encoding="utf-8")
    # The launch measurement: every valid pair, two or three colour formats each, up to cache_pipelines.
    valid = [p for p in (json.loads(l) for l in open(work / "pairs.jsonl", encoding="utf-8")) if p["valid"]]
    third = max(0, min(len(valid), cache_pipelines - 2 * len(valid)))
    counts = [3 if i < third else 2 for i in range(len(valid))]
    while sum(counts) > cache_pipelines:
        counts.pop()
    (jobs / "cache.body").write_text(real_body(s1, mods, valid[:len(counts)], counts), encoding="utf-8")
    body, meta = synth_body(work, comps, seed)
    (jobs / "synth.body").write_text(body, encoding="utf-8")
    (work / "synth.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({"build_pairs": len(sample), "cache_pipelines": sum(counts), "cache_pairs": len(counts),
                      "synth": {k: meta[k] for k in ("compositions", "first", "uni_set", "over_push_limit")}}))
    return 0


def builds(work: Path, threads: int, phases: list[str], validation: str, suffix: str, body: str) -> int:
    """Each phase twice, in fresh processes with the driver's internal cache off:
    cold (the application's caches are written) then warm (they are read back).
    The driver's own disk cache is not used for warm: on this PC its path variable
    is ignored and a second run with it on was no faster than the first."""
    exe = exe_path()
    cdir = work / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    for phase in phases:
        params = {"pipeline_cache": cdir / f"build-{phase}-t{threads}{suffix}.cache",
                  "shader_binaries": cdir / f"build-{phase}-t{threads}{suffix}.objbin"}
        for f in params.values():
            f.unlink(missing_ok=True)
        for run in ("cold", "warm"):
            run_harness(work, exe, "build", body, header(threads, "disabled", validation, params),
                        f"build-{phase}-{run}-t{threads}{suffix}", phase)
    return 0


def cacheload(work: Path, threads: int, phases: list[str]) -> int:
    cdir = work / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    params = {"cache_file": cdir / "pipelines.cache", "binaries_file": cdir / "pipelines.bin"}
    for phase in phases:
        run_harness(work, exe_path(), "cache", "cache.body", header(threads, "disabled", params=params),
                    f"cache-{phase}-t{threads}", phase)
    return 0


def draws(work: Path, ndraws: int, frames: int, run_length: int, variants: list[str], tag: str, mode: str,
          validation: str) -> int:
    meta = json.loads((work / "synth.json").read_text(encoding="utf-8"))
    n = meta["compositions"]
    params = {"draws": ndraws, "frames": frames, "run_length": run_length, "warmup": min(30, frames // 4)}
    for i, v in enumerate(variants):
        kind, bind = v.split("/")
        family = "uni" if kind.startswith("uni") else kind
        params[f"variant{i}"] = f"{kind} {bind} {meta['first'][family]} {n}"
    run_harness(work, exe_path(), mode, "synth.body", header(1, "internal", validation, params), tag)
    return 0


def item_stats(values: list[float]) -> dict:
    return {"n": len(values), "p50": round(percentile(values, 50), 4), "p90": round(percentile(values, 90), 4),
            "p99": round(percentile(values, 99), 4), "max": round(max(values), 4), "sum": round(sum(values), 1)}


def report(work: Path) -> int:
    """Per-item statistics of every run (from its rows) next to its summary line;
    the last summary of a tag wins."""
    res = work / "results"
    summaries: dict[str, dict] = {}
    for line in open(res / "summary.jsonl", encoding="utf-8"):
        s = json.loads(line)
        summaries[s["tag"]] = s
    out = {}
    for tag, s in sorted(summaries.items()):
        rows = res / f"{tag}.tsv"
        items: dict[str, list[float]] = {}
        if rows.is_file():
            for r in read_tsv(rows):
                items.setdefault(r["what"], []).append(float(r["ms"]))
        out[tag] = {"summary": s, "items": {k: item_stats(v) for k, v in sorted(items.items())}}
    (work / "report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for tag, r in out.items():
        print(tag, json.dumps({k: [v["n"], v["p50"], v["p99"], v["max"]] for k, v in r["items"].items()}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--work", type=Path, required=True)
    c = sub.add_parser("census")
    c.add_argument("--s1", type=Path, required=True)
    c.add_argument("--work", type=Path, required=True)
    s = sub.add_parser("select")
    s.add_argument("--work", type=Path, required=True)
    s.add_argument("--pairs", type=int, default=600)
    s.add_argument("--seed", type=int, default=20)
    p = sub.add_parser("prepare")
    p.add_argument("--s1", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--cache-pipelines", type=int, default=20000)
    p.add_argument("--compositions", type=int, default=64)
    p.add_argument("--seed", type=int, default=20)
    b = sub.add_parser("builds")
    b.add_argument("--work", type=Path, required=True)
    b.add_argument("--threads", type=int, default=1)
    b.add_argument("--phases", default="libs,mono,objects,objects-linked")
    b.add_argument("--validation", choices=("off", "on"), default="off")
    b.add_argument("--tag", default="")
    b.add_argument("--body", default="build.body")
    k = sub.add_parser("cacheload")
    k.add_argument("--work", type=Path, required=True)
    k.add_argument("--threads", type=int, default=12)
    k.add_argument("--phases", default="build,load,binaries")
    for name in ("draws", "replay"):
        d = sub.add_parser(name)
        d.add_argument("--work", type=Path, required=True)
        d.add_argument("--draws", type=int, default=3000)
        d.add_argument("--frames", type=int, default=300)
        d.add_argument("--run-length", type=int, default=1)
        d.add_argument("--variants", required=True)
        d.add_argument("--validation", choices=("off", "on"), default="off")
        d.add_argument("--tag", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "report":
            return report(args.work)
        if args.cmd == "census":
            return census(args.s1, args.work)
        if args.cmd == "select":
            return select(args.work, args.pairs, args.seed)
        if args.cmd == "prepare":
            return prepare(args.s1, args.work, args.cache_pipelines, args.compositions, args.seed)
        if args.cmd == "builds":
            return builds(args.work, args.threads, args.phases.split(","), args.validation, args.tag, args.body)
        if args.cmd == "cacheload":
            return cacheload(args.work, args.threads, args.phases.split(","))
        if args.cmd in ("draws", "replay"):
            return draws(args.work, args.draws, args.frames, args.run_length, args.variants.split(","), args.tag,
                         args.cmd, args.validation)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"pipe_cost: {e}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
