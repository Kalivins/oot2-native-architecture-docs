#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Spike S8: the GPU time per frame of the compute chains that hold the game's tensor programs.

    py -3 tools/tensorcost/tensor_cost.py jobs   --replay replay.json --split jobs.jsonl --extra extra.jsonl --out jobs85.jsonl
    py -3 tools/tensorcost/tensor_cost.py plan   --replay replay.json --modules DIR --work DIR [--heap-images N --heap-samplers N]
    py -3 tools/tensorcost/tensor_cost.py run    --work DIR [--rounds N --frames N --validation] [--no-quiet-check]
    py -3 tools/tensorcost/tensor_cost.py report --work DIR [--budget-ms 14]

`jobs` turns the replay file's dispatches into translator jobs: the base record of each program from the
split (or from --extra, for a program the split does not hold).
`plan` reads each translated module's description (DIR/<job>.res.json, <job>.spv) and the recorded
dispatches and writes, per mode, a job file for tools/tensorcost/vktensorcost.c plus the constant banks it
names. `run` plays them: normal, then the two counter-runs (an empty shader in place of every program, and the
largest group count of every dispatch halved), with `nvidia-smi` clocks around and during the runs.
`report` computes per-phase and per-program figures and the projection to cooperative matrices.

The replay file (schema 1) is written from the census, outside the repository (it holds counts, formats and
the game's own constants, never code):

    {"schema": 1, "phases": [{"name": "...", "source": "...", "dispatches": [
        {"n": 90, "key": "...", "job": "<job key or 'unmatched'>", "grid": [x,y,z], "block": [x,y,z], "hmma": 0,
         "pokes": [{"ordinal": 0, "offset": 0, "value": 210528}],      (a 32-bit word written into the program's n-th storage buffer)
         "fills": [{"ordinal": 1, "word": 4294967295}],               (that buffer filled with one word instead of halves)
         "buffers": [{"type": "uniform", "index": 0, "size": 304, "raw": "path|absent"},
                     {"type": "storage", "index": 0, "size": 4194304, "written": true}],
         "textures": [{"index": 0, "tic": 31348, "format": "B10G11R11_FLOAT", "w": 2560, "h": 1440, "d": 1,
                       "sampler": {"mag": "linear", "min": "linear", "wrap": ["clamp_to_edge"]*3, "max_lod": 0.25}}],
         "images": [{"slot": 0, "format": "R16G16_UINT", "w": 640, "h": 360, "d": 1}]}]}]}

How a recorded resource finds a program's binding (the census numbers a stage's resources in slot order, the
translator names a slot by its constant-bank offset): the program's storage buffers, textures and images are
sorted by their guest slot and zipped with the recorded ones sorted by index. Every storage buffer is made as
large as the dispatch's largest recorded one, so a mismatch cannot read past its end. A program that reaches
memory through the page table (raw access) is named and left out of its phase: it needs D15's table.

Exit codes: 0 done, 1 the inputs are wrong or the harness reported errors, 2 the machine is not quiet or a tool is missing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DRIVER0_STORAGE_BASE = 0x310        # c[0] + 0x310 + 0x10 s: NVN's compute storage table (InstEmitAmpere.AMapC0)
PUSH_BYTES = 128
QUIET_MIN_FREE_MB = 3000
BUSY_NAMES = ("oot2.exe", "clang.exe", "ninja.exe", "vkpipecost.exe", "vkcomputecost.exe", "vkhmmacost.exe",
              "vktensorcost.exe", "vkdraw.exe", "vkmesh.exe", "vkpresent.exe", "eden.exe", "lichii.exe",
              "oot2-xlate.exe", "dotnet.exe", "msbuild.exe", "nvdisasm.exe")

# census name -> (VkFormat number, Ryujinx TextureFormat name or None where the format is never stored to)
FORMATS = {
    "R8_UNORM": (9, "R8Unorm"), "R8_UINT": (13, "R8Uint"), "R8G8_UNORM": (16, "R8G8Unorm"),
    "R16_UNORM": (70, "R16Unorm"), "R16_FLOAT": (76, "R16Float"), "R16G16_FLOAT": (83, "R16G16Float"),
    "R16G16_UINT": (74, "R16G16Uint"), "R32_UINT": (98, "R32Uint"), "R32_FLOAT": (100, "R32Float"),
    "A8B8G8R8_UNORM": (37, "R8G8B8A8Unorm"), "A8B8G8R8_SRGB": (43, None),
    "A2B10G10R10_UNORM": (64, "R10G10B10A2Unorm"), "B10G11R11_FLOAT": (122, "R11G11B10Float"),
    "E5B9G9R9_FLOAT": (123, None), "R16G16B16A16_FLOAT": (97, "R16G16B16A16Float"),
    "R16G16B16A16_UNORM": (91, "R16G16B16A16Unorm"), "R32G32B32A32_FLOAT": (109, "R32G32B32A32Float"),
}
# The modules declare images with a float sampled type, so a uint format is bound as the float format of its width
# (the same bytes per texel, so the same traffic); the plan names every substitution.
UINT_AS_FLOAT = {"R16G16_UINT": "R16G16_FLOAT", "R32_UINT": "R32_FLOAT", "R8_UINT": "R8_UNORM"}
FILTERS = {"nearest": 0, "linear": 1}
WRAPS = {"repeat": 0, "mirror": 1, "mirrored_repeat": 1, "clamp_to_edge": 2, "clamp_to_border": 3, "mirror_clamp_to_edge": 4}
# the HMMA cost study, row 1: ns per warp-wide HMMA.16816 (f16 accumulator) on the 3060 Ti
HMMA_NS = {"scratch": 0.921, "coop_helper": 0.597, "coop_f16": 0.0533}


class Bad(Exception):
    pass


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def vk_format(name: str) -> int:
    if name not in FORMATS:
        raise Bad(f"unknown format {name}")
    return FORMATS[name][0]


# ---- jobs ----------------------------------------------------------------------------------------

def translator_jobs(replay: dict, split: dict[str, dict], extra: dict[str, dict]) -> list[dict]:
    """One translator job per distinct program of the replay: its base record, default answers.

    Answers are not given: the translator declares every storage image as an image of unknown format with a float
    sampled type, whatever the guest format (an "img" answer for a uint format changed nothing in the modules,
    measured), so the replay binds the float format of the same width (UINT_AS_FLOAT)."""
    out: dict[str, dict] = {}
    for ph in replay["phases"]:
        for d in ph["dispatches"]:
            key = d["job"]
            base = extra.get(key) or split.get(key)
            if base is None:
                raise Bad(f"no base record for job {key}")
            job = dict(base)
            job["key"] = key
            out[key] = job
    return list(out.values())


# ---- plan ----------------------------------------------------------------------------------------

def guest_slot_of_sbuf(sb: dict) -> int:
    """The storage slot a program's buffer was bound at, read from where it looks for its descriptor."""
    off = sb.get("address_byte_offset", sb.get("cb_byte_offset", 0))
    if sb.get("cb_slot") == 0 and off >= DRIVER0_STORAGE_BASE and (off - DRIVER0_STORAGE_BASE) % 16 == 0:
        return (off - DRIVER0_STORAGE_BASE) // 16
    return 1000 + off          # an address read from elsewhere in the bank: ordered after the table, by offset


def support_bytes(res: dict) -> bytes:
    sb = res["support_buffer"]
    data = bytearray(sb["size"])
    for f in sb["fields"]:
        if f["name"] == "render_scale":
            for i in range(f["bytes"] // 4):
                struct.pack_into("<f", data, f["offset"] + 4 * i, 1.0)
    return bytes(data)


class Planner:
    def __init__(self, replay: dict, modules: Path, work: Path, heap_images: int, heap_samplers: int, frames: int, warmup: int):
        self.replay, self.modules, self.work = replay, modules, work
        self.heap_images, self.heap_samplers, self.frames, self.warmup = heap_images, heap_samplers, frames, warmup
        self.lines: list[str] = []          # resources, shared by all phases
        self.phase_lines: list[str] = []
        self.module_index: dict[str, int] = {}
        self.manifest: dict = {"phases": [], "substituted_formats": []}
        self.counter = 0
        self.images: dict[tuple, str] = {}
        self.samplers: dict[tuple, int] = {}
        self.heap_index: dict[tuple, int] = {}
        self.substituted: set[str] = set()
        self.heap_used: set[int] = set()
        (work / "banks").mkdir(parents=True, exist_ok=True)

    def float_format(self, name: str) -> str:
        sub = UINT_AS_FLOAT.get(name)
        if sub is not None:
            self.substituted.add(f"{name} as {sub}")
        return sub or name

    def new_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def module(self, job: str) -> int:
        if job not in self.module_index:
            self.module_index[job] = len(self.module_index)
            self.lines.append(f"module {self.modules / (job + '.spv')}")
        return self.module_index[job]

    def bank_file(self, name: str, data: bytes) -> Path:
        p = self.work / "banks" / name
        p.write_bytes(data)
        return p

    def sampler(self, s: dict) -> int:
        key = (FILTERS[s["mag"]], FILTERS[s["min"]], tuple(WRAPS[w] for w in s["wrap"]), float(s.get("max_lod", 0)))
        if key not in self.samplers:
            idx = len(self.samplers)
            if idx >= self.heap_samplers:
                raise Bad("more samplers than the heap holds")
            self.samplers[key] = idx
            self.lines.append(f"sampler smp{idx} {key[0]} {key[1]} {key[2][0]} {key[2][1]} {key[2][2]} {key[3]}")
            self.lines.append(f"heapsampler {idx} smp{idx}")
        return self.samplers[key]

    def sampled_image(self, t: dict) -> int:
        """The heap index of a recorded texture: its own texture index when free, else the next free one up."""
        t = dict(t, format=self.float_format(t["format"]))
        key = (t["tic"], t["format"], t["w"], t["h"], t["d"])
        if key in self.heap_index:
            return self.heap_index[key]
        if t["tic"] >= self.heap_images:
            raise Bad(f"texture index {t['tic']} past the heap of {self.heap_images}")
        idx = t["tic"]
        while idx in self.heap_used:
            idx += 1
            if idx >= self.heap_images:
                raise Bad("the image heap is full")
        self.heap_used.add(idx)
        self.heap_index[key] = idx
        iid = self.new_id("tex")
        self.lines.append(f"image {iid} {vk_format(t['format'])} {t['w']} {t['h']} {t['d']} sampled {idx}")
        self.lines.append(f"heapimage {idx} {iid}")
        return idx

    def plan_dispatch(self, phase: str, i: int, d: dict, notes: list[str]) -> list[str] | None:
        job = d["job"]
        res = load_json(self.modules / f"{job}.res.json")
        if res.get("raw_access", {}).get("physical_pointers"):
            notes.append(f"{phase}/{i} {job}: raw access, needs the page table (D15), left out")
            return None
        mid = self.module(job)
        name = f"d{i}_{job.split('.')[0][:8]}"
        out = [f"dispatch {name} {mid} {d['grid'][0]} {d['grid'][1]} {d['grid'][2]}"]
        push = bytearray(PUSH_BYTES)
        binds: list[str] = []
        # storage buffers: sorted by guest slot, zipped with the recorded ones sorted by index
        rec_storage = sorted([b for b in d["buffers"] if b["type"] == "storage"], key=lambda b: b["index"])
        sbufs = sorted(res.get("sbufs", []), key=guest_slot_of_sbuf)
        big = max([b["size"] for b in rec_storage] + [4096])
        if len(sbufs) > len(rec_storage):
            notes.append(f"{phase}/{i} {job}: {len(sbufs)} storage buffers in the program, {len(rec_storage)} recorded; the rest take {big} bytes")
        sb_ids: list[str] = []
        driver0 = bytearray(65536)
        for k, sb in enumerate(sbufs):
            bid = self.new_id("sb")
            word = next((f["word"] for f in d.get("fills", []) if f["ordinal"] == k), None)
            self.lines.append(f"buffer {bid} {big} " + (f"word {word}" if word is not None else f"f16 {self.counter}"))
            for pk in d.get("pokes", []):
                if pk["ordinal"] == k:
                    self.lines.append(f"poke {bid} {pk['offset']} {pk['value']}")
            sb_ids.append(bid)
            binds.append(f"bind {sb['binding']} ssbo {bid}")
            addr = (k + 1) << 32          # the shader reads the low 32 bits less the descriptor's: 0 with this address
            struct.pack_into("<QI", driver0, sb["address_byte_offset"], addr, big)
        # constant banks
        uni = [b for b in d["buffers"] if b["type"] == "uniform" and b["index"] == 0]
        c0 = b""
        if uni and uni[0].get("raw") not in (None, "absent"):
            c0 = Path(uni[0]["raw"]).read_bytes()
        else:
            notes.append(f"{phase}/{i} {job}: constants not recorded, zeros")
        for cb in res.get("cbufs", []):
            bid = self.new_id("cb")
            if cb["role"] == "push":
                data = c0[0x10:].ljust(65536, b"\0")[:65536]
            elif cb["role"] == "driver0":
                data = bytes(driver0)
            else:
                data = bytes(65536)
            p = self.bank_file(f"{phase}_{i}_{cb['role']}.bin", data)
            self.lines.append(f"buffer {bid} 65536 file {p}")
            binds.append(f"bind {cb['binding']} ubo {bid}")
        sup = res["support_buffer"]
        bid = self.new_id("su")
        p = self.bank_file(f"{phase}_{i}_support.bin", support_bytes(res))
        self.lines.append(f"buffer {bid} {sup['size']} file {p}")
        binds.append(f"bind {sup['binding']} ubo {bid}")
        # storage images
        rec_img = sorted(d.get("images", []), key=lambda im: im["slot"])
        imgs = sorted(res.get("images", []), key=lambda im: im.get("c2_slot", 0))
        if len(imgs) != len(rec_img):
            raise Bad(f"{phase}/{i} {job}: {len(imgs)} storage images in the program, {len(rec_img)} recorded")
        for im, rec in zip(imgs, rec_img):
            iid = self.new_id("img")
            # a module that names its image's format (a store with a fixed one) gets exactly that format
            declared = {v[1]: k for k, v in FORMATS.items() if v[1]}.get(im.get("format", "Unknown"))
            fmt = declared or self.float_format(rec["format"])
            if declared and declared != rec["format"]:
                raise Bad(f"{phase}/{i} {job}: the module declares {declared}, the capture has {rec['format']}")
            self.lines.append(f"image {iid} {vk_format(fmt)} {rec['w']} {rec['h']} {rec['d']} storage {self.counter}")
            binds.append(f"bind {im['binding']} simg {iid}")
        # textures through the heap and the push constants
        rec_tex = sorted(d.get("textures", []), key=lambda t: t["index"])
        texs = sorted(res.get("textures", []), key=lambda t: t.get("c2_slot", 0))
        if len(texs) != len(rec_tex):
            raise Bad(f"{phase}/{i} {job}: {len(texs)} textures in the program, {len(rec_tex)} recorded")
        for t, rec in zip(texs, rec_tex):
            if t.get("via") != "push_slot":
                raise Bad(f"{phase}/{i} {job}: a texture reached by {t.get('via')}")
            tic = self.sampled_image(rec)
            smp = self.sampler(rec["sampler"])
            struct.pack_into("<I", push, t["push_offset"], tic | (smp << 20))
        out.append(f"push {push.hex()}")
        out.extend(sorted(binds, key=lambda s: int(s.split()[1])))
        return out

    def plan(self) -> tuple[list[str], list[str]]:
        for ph in self.replay["phases"]:
            notes: list[str] = []
            lines = [f"phase {ph['name']}"]
            skipped = 0
            for i, d in enumerate(ph["dispatches"]):
                one = self.plan_dispatch(ph["name"], i, d, notes)
                if one is None:
                    skipped += 1
                else:
                    lines.extend(one)
            lines.append("end")
            self.phase_lines.extend(lines)
            self.manifest["phases"].append({"name": ph["name"], "source": ph.get("source", ""), "dispatches": len(ph["dispatches"]),
                                            "replayed": len(ph["dispatches"]) - skipped, "notes": notes,
                                            "programs": [{"name": f"d{i}_{d['job'].split('.')[0][:8]}", "job": d["job"], "grid": d["grid"],
                                                          "block": d["block"], "hmma": d.get("hmma", 0)}
                                                         for i, d in enumerate(ph["dispatches"])]})
        self.manifest["substituted_formats"] = sorted(self.substituted)
        return self.lines, self.phase_lines


def write_jobs(work: Path, lines: list[str], phase_lines: list[str], frames: int, warmup: int, heap: tuple[int, int],
               validation: bool, batch: int = 16) -> dict[str, Path]:
    paths = {}
    for mode in ("normal", "empty", "half"):
        head = [f"validation {'on' if validation else 'off'}", f"frames {frames}", f"warmup {warmup}", f"batch {batch}", f"mode {mode}",
                f"out {work / ('rows-' + mode + '.tsv')}", f"heap {heap[0]} {heap[1]}"]
        p = work / f"job-{mode}.txt"
        p.write_text("\n".join(head + lines + phase_lines) + "\n", encoding="utf-8")
        paths[mode] = p
    return paths


# ---- run -----------------------------------------------------------------------------------------

def exe_path() -> Path:
    return ROOT / "build" / "vktensorcost" / "vktensorcost.exe"


def busy_processes() -> list[str]:
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    names = {line.split('","')[0].strip('"').lower() for line in out.splitlines() if line}
    return sorted(n for n in names if n in BUSY_NAMES)


def free_memory_mb() -> int:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"], capture_output=True, text=True, timeout=60).stdout
        return int(out.strip()) // 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def gpu_state() -> dict:
    q = "name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used,pstate"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(zip(q.split(","), [x.strip() for x in out.strip().split(",")]))


class ClockSampler:
    def __init__(self, path: Path):
        self.path, self.proc = path, None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,pstate",
                                          "--format=csv,noheader,nounits", "-lms", "50"], stdout=open(self.path, "w"), stderr=subprocess.DEVNULL)
        except OSError:
            self.proc = None
        return self

    def __exit__(self, *a):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()


def clock_summary(path: Path) -> dict:
    sm, power = [], []
    for line in path.read_text().splitlines() if path.exists() else []:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 7 and p[6] == "P0":
            try:
                sm.append(float(p[1])); power.append(float(p[4]))
            except ValueError:
                pass
    if not sm:
        return {}
    return {"samples": len(sm), "sm_mhz_min": min(sm), "sm_mhz_median": statistics.median(sm), "sm_mhz_max": max(sm),
            "power_w_median": statistics.median(power)}


def run_harness(job: Path, timeout_s: float = 1800) -> tuple[int, list[dict], str]:
    p = subprocess.run([str(exe_path()), "run", str(job)], capture_output=True, text=True, timeout=timeout_s)
    rows = [json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]
    return p.returncode, rows, p.stderr


def cmd_run(args) -> int:
    work = Path(args.work)
    state = {"before": gpu_state()}
    if not args.no_quiet_check:
        busy, free = busy_processes(), free_memory_mb()
        state["busy"], state["free_mb"] = busy, free
        if busy or free < QUIET_MIN_FREE_MB:
            print(f"tensor_cost: the machine is not quiet (busy {busy}, free {free} MB)", file=sys.stderr)
            return 2
    plan = load_json(work / "plan.json")
    rounds: list[dict] = []
    clocks = work / "clocks.csv"
    with ClockSampler(clocks):
        for r in range(args.rounds):
            for mode in ("normal", "empty", "half"):
                if mode != "normal" and r > 0:
                    continue
                rc, rows, err = run_harness(work / f"job-{mode}.txt")
                (work / f"stderr-{mode}-{r}.txt").write_text(err)
                if rc != 0:
                    print(f"tensor_cost: harness exit {rc} in {mode} round {r}\n{err[-2000:]}", file=sys.stderr)
                    return 1
                rounds.append({"round": r, "mode": mode, "phases": rows})
    state["after"] = gpu_state()
    state["clocks"] = clock_summary(clocks)
    (work / "runs.json").write_text(json.dumps({"state": state, "rounds": rounds, "plan": plan}, indent=1))
    print(f"tensor_cost: {len(rounds)} runs written to {work / 'runs.json'}")
    return 0


# ---- report --------------------------------------------------------------------------------------

def pct(values: list[float], q: float) -> float:
    v = sorted(values)
    if not v:
        return 0.0
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def hmma_ms(prog: dict) -> dict:
    """Projected GPU time of one dispatch's HMMA in each form, from the HMMA cost study's per-instruction costs
    (each warp runs each static HMMA once: loops would multiply, branches divide)."""
    threads = prog["block"][0] * prog["block"][1] * prog["block"][2]
    warps = prog["grid"][0] * prog["grid"][1] * prog["grid"][2] * max(1, threads // 32)
    count = prog.get("hmma", 0) * warps
    return {form: count * ns * 1e-6 for form, ns in HMMA_NS.items()} | {"hmma_per_frame": count}


def build_report(runs: dict, budget_ms: float) -> dict:
    plan = {p["name"]: p for p in runs["plan"]["phases"]}
    by_mode: dict[str, dict[str, list[dict]]] = {"normal": {}, "empty": {}, "half": {}}
    for r in runs["rounds"]:
        for ph in r["phases"]:
            by_mode[r["mode"]].setdefault(ph["phase"], []).append(ph)
    out = {"state": runs["state"], "budget_ms": budget_ms, "phases": {}}
    for name, rounds in by_mode["normal"].items():
        p50s = [x["frame_p50_ms"] for x in rounds]
        p99s = [x["frame_p99_ms"] for x in rounds]
        progs, left_out = [], []
        proj = {"scratch": 0.0, "coop_helper": 0.0, "coop_f16": 0.0}
        for prog in plan[name]["programs"]:
            # by name: a dispatch the plan left out (raw access) is in the plan but not in the harness's rows
            cols = [d for x in rounds for d in x["dispatches"] if d["name"] == prog["name"]]
            h = hmma_ms(prog)
            if not cols:
                left_out.append({"name": prog["name"], "grid": prog["grid"], "block": prog["block"], "hmma": prog.get("hmma", 0),
                                 "hmma_per_frame": h["hmma_per_frame"], "hmma_scratch_ms": h["scratch"]})
                continue
            row = {"name": prog["name"], "job": prog["job"], "grid": prog["grid"], "block": prog["block"],
                   "p50_ms": statistics.median(c["p50_ms"] for c in cols), "p99_ms": max(c["p99_ms"] for c in cols), "hmma": prog.get("hmma", 0)}
            row["hmma_per_frame"] = h["hmma_per_frame"]
            row["hmma_scratch_ms"] = h["scratch"]
            for form in proj:
                proj[form] += h[form]
            progs.append(row)
        entry = {"frames": rounds[0]["frames"], "rounds": len(rounds), "replayed": plan[name]["replayed"], "dispatches": plan[name]["dispatches"],
                 "notes": plan[name]["notes"], "frame_p50_ms": statistics.median(p50s), "frame_p99_ms": max(p99s),
                 "frame_p50_by_round": p50s, "frame_p99_by_round": p99s, "programs": progs, "left_out": left_out,
                 "left_out_hmma_scratch_ms": sum(x["hmma_scratch_ms"] for x in left_out)}
        scratch_total = proj["scratch"]
        measured = entry["frame_p50_ms"]
        entry["projection"] = {
            "hmma_scratch_ms": scratch_total, "hmma_share_of_p50": scratch_total / measured if measured else 0,
            "coop_helper_p50_ms": max(0.0, measured - scratch_total + proj["coop_helper"]),
            "coop_f16_p50_ms": max(0.0, measured - scratch_total + proj["coop_f16"]),
            "coop_helper_p99_ms": max(0.0, entry["frame_p99_ms"] - scratch_total + proj["coop_helper"]),
            "coop_f16_p99_ms": max(0.0, entry["frame_p99_ms"] - scratch_total + proj["coop_f16"]),
            "valid": scratch_total <= measured,
            "assumption": "each warp runs each static HMMA once; the HMMA cost study's ns per HMMA: scratch 0.921, helper swap 0.597, coop f16 0.0533; "
                          "not valid (valid=false) where the projected HMMA time exceeds the measured frame, i.e. a program that does not run its HMMA on every warp"}
        entry["share_of_budget_p99"] = entry["frame_p99_ms"] / budget_ms
        if name in by_mode["empty"]:
            e = by_mode["empty"][name][0]
            entry["counter_empty"] = {"frame_p50_ms": e["frame_p50_ms"], "frame_p99_ms": e["frame_p99_ms"],
                                      "ratio_to_normal_p50": e["frame_p50_ms"] / measured if measured else 0}
        if name in by_mode["half"]:
            hh = by_mode["half"][name][0]
            ratios = []
            half = {d["name"]: d for d in hh["dispatches"]}
            for prog in progs:
                n = prog["p50_ms"]
                hv = half[prog["name"]]["p50_ms"] if prog["name"] in half else 0
                ratios.append({"name": prog["name"], "ratio": hv / n if n else 0})
            entry["counter_half"] = {"frame_p50_ms": hh["frame_p50_ms"], "ratio_to_normal_p50": hh["frame_p50_ms"] / measured if measured else 0,
                                     "per_program": ratios}
        out["phases"][name] = entry
    return out


def cmd_report(args) -> int:
    work = Path(args.work)
    rep = build_report(load_json(work / "runs.json"), args.budget_ms)
    (work / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "phases"}, indent=1))
    for name, e in rep["phases"].items():
        print(f"{name}: frame p50 {e['frame_p50_ms']:.3f} ms, p99 {e['frame_p99_ms']:.3f} ms ({e['share_of_budget_p99'] * 100:.1f} % of {args.budget_ms} ms), "
              f"replayed {e['replayed']}/{e['dispatches']}; HMMA scratch {e['projection']['hmma_scratch_ms']:.3f} ms "
              f"({e['projection']['hmma_share_of_p50'] * 100:.0f} % of p50{'' if e['projection']['valid'] else ', PROJECTION NOT VALID'}), coop helper p50 {e['projection']['coop_helper_p50_ms']:.3f}, coop f16 p50 {e['projection']['coop_f16_p50_ms']:.3f}")
        if "counter_empty" in e:
            print(f"   empty counter-run: p50 {e['counter_empty']['frame_p50_ms']:.4f} ms ({e['counter_empty']['ratio_to_normal_p50'] * 100:.2f} % of normal)")
        if "counter_half" in e:
            print(f"   half counter-run: frame ratio {e['counter_half']['ratio_to_normal_p50']:.3f}; per program "
                  + ", ".join(f"{r['name']} {r['ratio']:.2f}" for r in e["counter_half"]["per_program"]))
    return 0


# ---- main ----------------------------------------------------------------------------------------

def cmd_jobs(args) -> int:
    replay = load_json(args.replay)
    split = {}
    with open(args.split, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if j["stage"] == "cs":
                split[j["key"]] = j
    extra = {}
    if args.extra:
        with open(args.extra, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    extra[j["key"]] = j
    try:
        jobs = translator_jobs(replay, split, extra)
    except Bad as e:
        print(f"tensor_cost: {e}", file=sys.stderr)
        return 1
    Path(args.out).write_text("".join(json.dumps(j) + "\n" for j in jobs), encoding="utf-8")
    print(f"tensor_cost: {len(jobs)} translator jobs in {args.out}")
    return 0


def cmd_plan(args) -> int:
    work = Path(args.work).resolve()          # the job files name their banks and modules: absolute, so a run needs no particular folder
    work.mkdir(parents=True, exist_ok=True)
    try:
        pl = Planner(load_json(args.replay), Path(args.modules).resolve(), work, args.heap_images, args.heap_samplers, args.frames, args.warmup)
        lines, phase_lines = pl.plan()
    except (Bad, OSError, KeyError) as e:
        print(f"tensor_cost: {e!r}", file=sys.stderr)
        return 1
    write_jobs(work, lines, phase_lines, args.frames, args.warmup, (args.heap_images, args.heap_samplers), args.validation)
    (work / "plan.json").write_text(json.dumps(pl.manifest, indent=1))
    for ph in pl.manifest["phases"]:
        print(f"tensor_cost: {ph['name']}: {ph['replayed']}/{ph['dispatches']} dispatches", *ph["notes"], sep="\n  ")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--replay", required=True); j.add_argument("--split", required=True); j.add_argument("--extra"); j.add_argument("--out", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--replay", required=True); p.add_argument("--modules", required=True); p.add_argument("--work", required=True)
    p.add_argument("--heap-images", type=int, default=33025); p.add_argument("--heap-samplers", type=int, default=4096)
    p.add_argument("--frames", type=int, default=300); p.add_argument("--warmup", type=int, default=30); p.add_argument("--validation", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--work", required=True); r.add_argument("--rounds", type=int, default=3); r.add_argument("--no-quiet-check", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("--work", required=True); rp.add_argument("--budget-ms", type=float, default=14.0)
    args = ap.parse_args(argv)
    return {"jobs": cmd_jobs, "plan": cmd_plan, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
