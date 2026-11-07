#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Spike S6: how long the driver takes to build the game's translated compute programs.

    py -3 tools/computecost/compute_cost.py census --run DIR --split DIR --out FILE
    py -3 tools/computecost/compute_cost.py prepare --census FILE --selection FILE --run DIR --work DIR
    py -3 tools/computecost/compute_cost.py smoke --work DIR [--count 20]
    py -3 tools/computecost/compute_cost.py coldcheck --work DIR [--count 20]
    py -3 tools/computecost/compute_cost.py screen --work DIR [--tier core|tail] [--timeout S] [--budget S]
    py -3 tools/computecost/compute_cost.py batch --work DIR --tag T [--threads N] [--cache none|write|read]
                                                  [--cache-name R1] [--pin32] [--subset N]
    py -3 tools/computecost/compute_cost.py report --work DIR --census FILE [--g12 S] [--budget-min M M]

census reads a translation pass of tools/shaders/corpus.py (results.tsv,
validation.tsv, spv/<key>.res.json, opened by key) and the split's jobs.jsonl
and programs.bin, and writes one row per compute job: size, raw access, the
tensor instructions of its code (HMMA, LDSM, MOVM: the low 12 bits of each
16-byte instruction, the vendored decoder's rule) and its stratum, a kind (T
tensor, R raw access, P plain) by a SPIR-V size bin. prepare turns a chosen
sample into job bodies for the harness (tools/computecost/vkcomputecost.c):
set 0 from each resource description, the device extensions the module
declares, and a warm-up program from outside the sample.

The runs, all with the driver's internal pipeline cache disabled:
  smoke      the sample's first programs once with validation on
  coldcheck  the same programs twice with the cache disabled, twice enabled
  screen     each program alone in its own process, smallest first (R0, and
             R5 for the tail): the uncontended time; a timeout, a memory
             watchdog and a heartbeat, and every program a fresh device
  batch      every program that screen finished under 60 s, longest first,
             in one process on N threads (R1 to R4); `--cache write` saves the
             application's VkPipelineCache and `--cache read` builds from it
report gives per-run statistics by kind and size bin, the stratified projection
to the whole population with a resampled interval, the miss-bound table of
architecture section 6.1 and the tier rule of D21.

Everything this writes goes under --work and names the game's programs: none of
it may enter the repository.

Exit codes: 0 done, 1 a measurement failed or the machine is not quiet, 2 an
input or a tool is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import struct
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "pipecost"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import pipe_cost  # noqa: E402  (describe, machine_load, percentile, read_tsv: spike S3's helpers)
import pool  # noqa: E402  (working_set, available_ram)

TENSOR_OPS = {0x23C: "hmma", 0x83B: "ldsm", 0x23A: "movm"}
BINS = [(0, 10_000, "<10K"), (10_000, 20_000, "10-20K"), (20_000, 50_000, "20-50K"), (50_000, 100_000, "50-100K"),
        (100_000, 200_000, "100-200K"), (200_000, 500_000, "200-500K"), (500_000, 1_000_000, "0.5-1M"),
        (1_000_000, 2_000_000, "1-2M"), (2_000_000, 1 << 62, ">2M")]
BIN_NAMES = [b[2] for b in BINS]
KINDS = ("T", "R", "P")

# SPIR-V extensions a module may declare, and the device extension each needs;
# None where Vulkan 1.3 has it in core. An unknown name stops prepare.
SPV_TO_VK = {
    "SPV_EXT_shader_atomic_float_add": "VK_EXT_shader_atomic_float",
    "SPV_EXT_shader_atomic_float_min_max": "VK_EXT_shader_atomic_float2",
    "SPV_EXT_shader_atomic_float16_add": "VK_EXT_shader_atomic_float2",
    "SPV_KHR_shader_ballot": "VK_EXT_shader_subgroup_ballot",
    "SPV_KHR_subgroup_vote": "VK_EXT_shader_subgroup_vote",
    "SPV_EXT_shader_image_int64": "VK_EXT_shader_image_atomic_int64",
    "SPV_KHR_physical_storage_buffer": None, "SPV_EXT_descriptor_indexing": None,
    "SPV_KHR_storage_buffer_storage_class": None, "SPV_KHR_variable_pointers": None,
    "SPV_KHR_8bit_storage": None, "SPV_KHR_16bit_storage": None, "SPV_KHR_float_controls": None,
    "SPV_KHR_vulkan_memory_model": None, "SPV_EXT_demote_to_helper_invocation": None,
    "SPV_KHR_terminate_invocation": None, "SPV_KHR_integer_dot_product": None,
    "SPV_KHR_non_semantic_info": None, "SPV_KHR_shader_draw_parameters": None,
    "SPV_KHR_no_integer_wrap_decoration": None,
}

R1_LIMIT_MS = 60_000      # programs slower than this alone are reported apart, never batched
TAIL_BYTES = 2_000_000    # programs this large run apart, one at a time, with the long timeout
MEMORY_KILL_MB = 6_000    # a single-program process above this working set is killed
FLOOR_MB = 1_500          # or when the machine's available RAM falls under this
BUSY = ("vkpipecost", "clang", "ld.lld", "oot2-xlate", "oot2.exe", "lichii", "eden", "yuzu", "ryujinx", "ninja",
        "cl.exe", "link.exe", "msbuild", "dotnet")


def size_bin(n: int) -> str:
    for lo, hi, name in BINS:
        if lo <= n < hi:
            return name
    raise ValueError(n)


def tensor_counts(code: bytes) -> dict[str, int]:
    """HMMA, LDSM and MOVM in a code slice: low 12 bits of each 16-byte instruction."""
    out = {name: 0 for name in TENSOR_OPS.values()}
    for at in range(0, len(code) - 15, 16):
        name = TENSOR_OPS.get(struct.unpack_from("<Q", code, at)[0] & 0xFFF)
        if name:
            out[name] += 1
    return out


def kind_of(row: dict) -> str:
    if int(row["hmma"]) + int(row["ldsm"]) + int(row["movm"]) > 0:
        return "T"
    return "R" if str(row["raw"]) == "1" else "P"


def spirv_extensions(data: bytes) -> list[str]:
    """OpExtension names of a module, read up to its OpMemoryModel."""
    if len(data) < 20 or struct.unpack_from("<I", data, 0)[0] != 0x07230203:
        raise ValueError("not SPIR-V")
    out, at = [], 20
    while at + 4 <= len(data):
        head = struct.unpack_from("<I", data, at)[0]
        n, op = head >> 16, head & 0xFFFF
        if n == 0 or at + 4 * n > len(data):
            break
        if op == 10:  # OpExtension
            out.append(data[at + 4:at + 4 * n].split(b"\0")[0].decode("ascii"))
        if op == 14:  # OpMemoryModel ends the preamble
            break
        at += 4 * n
    return out


def device_extensions(spv_exts: list[str]) -> list[str]:
    need = set()
    for e in spv_exts:
        if e not in SPV_TO_VK:
            raise ValueError(f"unknown SPIR-V extension {e}")
        if SPV_TO_VK[e]:
            need.add(SPV_TO_VK[e])
    return sorted(need)


# ---- census and prepare -------------------------------------------------------------

def census(run: Path, split: Path, out: Path) -> int:
    t0 = time.perf_counter()
    progs = {r["job_key"]: r for r in pipe_cost.read_tsv(split / "programs.tsv") if r["stage"] == "cs"}
    results = {r["key"]: r for r in pipe_cost.read_tsv(run / "results.tsv") if r["key"] in progs}
    valid = {r["key"]: r["strict_ok"] for r in pipe_cost.read_tsv(run / "validation.tsv") if r["key"] in progs}
    slices = {}
    with open(split / "jobs.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"stage":"cs"' not in line[:200]:
                continue
            j = json.loads(line)
            slices[j["key"]] = (j["code_offset"], j["code_length"])
    rows = []
    with open(split / "programs.bin", "rb") as binf:
        for key, p in sorted(progs.items()):
            off, ln = slices[key]
            binf.seek(off)
            row = {"key": key, "container": p["container"], "group": p["group"], "code_bytes": ln}
            row.update(tensor_counts(binf.read(ln)))
            r = results.get(key, {})
            row.update(status=r.get("status", "missing"), strict_ok=valid.get(key, ""), spv_bytes=r.get("spv_bytes", 0),
                       translator_us=r.get("micros", ""), raw="", local="", shared="", ndesc="", extensions="",
                       translator_id="")
            if row["status"] == "ok":
                res = json.loads((run / "spv" / f"{key}.res.json").read_text(encoding="utf-8"))
                ra = res.get("raw_access") or {}
                c = res.get("compute") or {}
                row.update(raw=int(bool(ra.get("physical_pointers"))),
                           local="x".join(str(x) for x in c.get("local_size", [])), shared=c.get("shared_bytes", ""),
                           ndesc=res.get("descriptor_count_set0", ""), translator_id=res.get("translator_id", ""),
                           extensions=",".join(spirv_extensions((run / "spv" / f"{key}.spv").read_bytes()[:65536])))
                row["kind"], row["bin"] = kind_of(row), size_bin(int(row["spv_bytes"]))
            else:
                row["kind"], row["bin"] = ("T" if kind_of({**row, "raw": 0}) == "T" else ""), ""
            rows.append(row)
    cols = ["key", "container", "group", "status", "strict_ok", "kind", "bin", "spv_bytes", "code_bytes", "hmma", "ldsm",
            "movm", "raw", "local", "shared", "ndesc", "extensions", "translator_us", "translator_id"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok" and r["strict_ok"] == "1"]
    tensor = [r for r in rows if int(r["hmma"]) + int(r["ldsm"]) + int(r["movm"]) > 0]
    summary = {"compute_jobs": len(rows), "translated_and_valid": len(ok),
               "status": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
               "tensor_programs": len(tensor), "with_ldsm": sum(int(r["ldsm"]) > 0 for r in tensor),
               "with_movm": sum(int(r["movm"]) > 0 for r in tensor),
               "hmma": sum(int(r["hmma"]) for r in rows), "ldsm": sum(int(r["ldsm"]) for r in rows),
               "movm": sum(int(r["movm"]) for r in rows),
               "translator_ids": sorted({r["translator_id"] for r in ok}),
               "cells": {f"{k} {b}": n for (k, b), n in sorted(population_cells(ok).items())},
               "seconds": round(time.perf_counter() - t0, 1)}
    print(json.dumps(summary, indent=1))
    return 0


def population_cells(rows: list[dict]) -> dict[tuple[str, str], int]:
    cells: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        cells[(r["kind"], r["bin"])] += 1
    return dict(cells)


# The descriptor layout `uni` (xlate/host/UniLayout.cs) for a compute pipeline: set 0 is the game's texture pool and its sampler
# pool (the lengths pipecost's GLOBAL names), set 1 the module's own push descriptors, and 128 bytes of push constants.
UNI_GLOBAL = ["global", "bind 0 septex 33025", "bind 1 sampler 4096", "push 128"]


def layout_of(res: dict) -> str:
    """`uni` for a schema 2 description, `s1` for the older one (set 0 per module)."""
    return "uni" if res.get("schema", 1) == 2 else "s1"


def set_lines(res: dict) -> list[str]:
    if layout_of(res) == "uni":
        # The push descriptors of set 1: the support buffer, the constant banks, the storage buffers and the storage images the
        # module binds (a texture takes a push constant slot and the global pool, so it adds nothing here).
        binds = {res["support_buffer"]["binding"]: ("ubo", 1)}
        binds.update({c["binding"]: ("ubo", 1) for c in res["cbufs"]})
        binds.update({s["binding"]: ("ssbo", 1) for s in res["sbufs"]})
        binds.update({i["binding"]: ("img", max(1, i.get("array_length", 1))) for i in res["images"] if i.get("via") == "descriptor"})
        return ["set"] + [f"bind {b} {kind} {count}" for b, (kind, count) in sorted(binds.items())]
    d = pipe_cost.describe(res)
    if d["set1"]:
        raise ValueError("the module uses set 1, which this harness does not build")
    return ["set"] + [f"bind {b} {kind} {count}" for b, kind, count in d["binds"]]


def prepare(census_path: Path, selection: Path, run: Path, work: Path) -> int:
    pop = {r["key"]: r for r in pipe_cost.read_tsv(census_path)}
    chosen = pipe_cost.read_tsv(selection)
    rows = []
    for r in chosen:
        key = r["job_key"]
        c = pop.get(key)
        if not c or c["status"] != "ok" or c["strict_ok"] != "1":
            print(f"compute_cost: {key[:16]} did not translate and validate in this pass; left out", file=sys.stderr)
            continue
        spv = run / "spv" / f"{key}.spv"
        res = json.loads((run / "spv" / f"{key}.res.json").read_text(encoding="utf-8"))
        rows.append({"key": key, "module": str(spv), "bytes": int(c["spv_bytes"]), "kind": c["kind"], "bin": c["bin"],
                     "tier": "tail" if int(c["spv_bytes"]) >= TAIL_BYTES else "core", "group": c["group"],
                     "container": c["container"], "layout": layout_of(res),
                     "extensions": ",".join(device_extensions([e for e in c["extensions"].split(",") if e])),
                     "set": " | ".join(set_lines(res))})
    rows.sort(key=lambda x: (x["bytes"], x["key"]))
    picked = {r["job_key"] for r in chosen}
    spare = sorted((r for r in pop.values() if r["key"] not in picked and r["status"] == "ok"
                    and r["strict_ok"] == "1" and r["kind"] == "P" and int(r["spv_bytes"]) < 10_000),
                   key=lambda r: (int(r["spv_bytes"]), r["key"]))
    if not spare:
        raise ValueError("no plain program under 10 KB outside the sample for the warm-up")
    w = spare[len(spare) // 2]
    wres = json.loads((run / "spv" / f"{w['key']}.res.json").read_text(encoding="utf-8"))
    warm = {"key": w["key"], "module": str(run / "spv" / f"{w['key']}.spv"), "bytes": int(w["spv_bytes"]), "layout": layout_of(wres),
            "extensions": ",".join(device_extensions([e for e in w["extensions"].split(",") if e])),
            "set": " | ".join(set_lines(wres))}
    work.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(work / "programs.tsv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    (work / "warmup.json").write_text(json.dumps(warm, indent=1), encoding="utf-8")
    print(json.dumps({"programs": len(rows), "core": sum(r["tier"] == "core" for r in rows),
                      "tail": sum(r["tier"] == "tail" for r in rows),
                      "by_kind": {k: sum(r["kind"] == k for r in rows) for k in KINDS},
                      "extensions": sorted({e for r in rows for e in r["extensions"].split(",") if e}),
                      "warmup_bytes": warm["bytes"]}))
    return 0


def job_body(programs: list[dict], warm: dict | None) -> list[str]:
    """Modules, deduplicated set descriptions and pipeline records for the harness."""
    lines: list[str] = []
    exts = sorted({e for p in programs + ([warm] if warm else []) for e in p["extensions"].split(",") if e})
    lines += [f"extension {e}" for e in exts]
    layouts = {p.get("layout", "s1") for p in programs + ([warm] if warm else [])}
    if len(layouts) > 1:
        raise ValueError(f"programs of two layouts in one job: {sorted(layouts)}")
    if layouts == {"uni"}:
        lines += UNI_GLOBAL
    sets: dict[str, int] = {}
    records = []
    for i, p in enumerate(([warm] if warm else []) + programs):
        lines.append(f"module {p['module']}")
        if p["set"] not in sets:
            sets[p["set"]] = len(sets)
        records.append(("warmup" if warm and i == 0 else "compute", i, sets[p["set"]]))
    for text in sets:
        lines += text.split(" | ")
    lines += [f"{what} {m} {s}" for what, m, s in records]
    return lines


# ---- running the harness -------------------------------------------------------------

def exe_path() -> Path:
    exe = ROOT / "build" / "vkcomputecost" / "vkcomputecost.exe"
    if not exe.is_file():
        raise FileNotFoundError(f"no harness at {exe}; run tools/build_vkdraw.py --harness vkcomputecost")
    return exe


def busy_processes() -> list[str]:
    """Processes that build, compile or emulate, which would share the CPU and the driver."""
    r = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True)
    names = {line.split('","')[0].strip('"').lower() for line in r.stdout.splitlines() if line}
    return sorted(n for n in names if any(n.startswith(b) for b in BUSY))


def quiet_check(min_available_mb: int = 3000) -> dict:
    load = pipe_cost.machine_load()
    busy = busy_processes()
    load["busy"] = busy
    if load.get("emulators") or busy:
        raise RuntimeError(f"the machine is not quiet: {load.get('emulators') or ''} {' '.join(busy)}".strip())
    if load.get("available_mb", 0) < min_available_mb:
        raise RuntimeError(f"only {load.get('available_mb')} MB available, {min_available_mb} needed")
    return load


def gpu_state() -> dict:
    q = "clocks.gr,clocks.mem,temperature.gpu,pstate,power.draw,driver_version"
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                       text=True)
    parts = [p.strip() for p in r.stdout.strip().split(",")]
    return dict(zip(q.split(","), parts)) if len(parts) == 6 else {"error": r.stderr.strip()[:200]}


def kill_tree(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def run_job(work: Path, tag: str, head: list[str], body: list[str], timeout_s: float, memory_kill_mb: float,
            label: str = "", heartbeat_s: float = 30.0, floor_mb: float = FLOOR_MB) -> dict:
    """One harness process under watch: a timeout, a memory watchdog and a heartbeat.
    Returns the summary line (or what is known of a killed run) and keeps the rows,
    the job and the messages under work/results."""
    res = work / "results"
    jobs = work / "jobs"
    res.mkdir(parents=True, exist_ok=True)
    jobs.mkdir(parents=True, exist_ok=True)
    job = jobs / f"{tag}.txt"
    job.write_text("\n".join(head + [f"out {res / (tag + '.tsv')}"] + body) + "\n", encoding="utf-8")
    env = dict(os.environ)
    env["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"
    out_path, err_path = res / f"{tag}.stdout.txt", res / f"{tag}.stderr.txt"
    gpu0 = gpu_state()
    t0 = time.perf_counter()
    with open(out_path, "wb") as so, open(err_path, "wb") as se:
        proc = subprocess.Popen([str(exe_path()), str(job)], stdout=so, stderr=se, env=env)
        peak_ws, min_avail, status, beat = 0, 1 << 62, "", t0
        while proc.poll() is None:
            time.sleep(0.25)
            ws = pool.working_set(proc) / 1048576
            avail = pool.available_ram() / 1048576
            peak_ws, min_avail = max(peak_ws, ws), min(min_avail, avail) if avail else min_avail
            now = time.perf_counter()
            if now - t0 > timeout_s:
                status = "timeout"
            elif ws > memory_kill_mb or (avail and avail < floor_mb):
                status = "killed_memory"
            if status:
                kill_tree(proc)
                break
            if now - beat >= heartbeat_s:
                beat = now
                print(f"compute_cost: {tag} {label}: {now - t0:.0f} s, working set {ws:.0f} MB, "
                      f"available {avail:.0f} MB", flush=True)
    seconds = time.perf_counter() - t0
    out = {"tag": tag, "process_s": round(seconds, 2), "peak_ws_mb_sampled": round(peak_ws, 1),
           "min_available_mb": round(min_avail, 1) if min_avail < (1 << 62) else None, "gpu_before": gpu0,
           "gpu_after": gpu_state(), "status": status or "exit", "exit": proc.returncode}
    text = out_path.read_text(encoding="utf-8", errors="replace").strip()
    if not status and text:
        out.update(json.loads(text.splitlines()[-1]))
    if not status and proc.returncode not in (0, 1):
        out["status"] = "crash"
        out["error"] = err_path.read_text(encoding="utf-8", errors="replace").strip()[-1500:]
    elif not status and not text:
        out["status"] = "job_error"
        out["error"] = err_path.read_text(encoding="utf-8", errors="replace").strip()[-1500:]
    elif not status:
        out["status"] = "ok" if proc.returncode == 0 else "validation_errors"
    with open(res / "summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    return out


def header(threads: int, cache: str, validation: str = "off", params: dict | None = None) -> list[str]:
    return ([f"validation {validation}", f"cache {cache}", f"threads {threads}", "device RTX 3060 Ti"]
            + [f"param {k} {v}" for k, v in (params or {}).items()])


def parse_extra(extra: str) -> dict:
    return dict(f.split("=", 1) for f in extra.split() if "=" in f)


def rows_of(work: Path, tag: str) -> list[dict]:
    path = work / "results" / f"{tag}.tsv"
    if not path.is_file():
        return []
    out = []
    for r in pipe_cost.read_tsv(path):
        r.update(parse_extra(r.pop("extra", "")))
        out.append(r)
    return out


def load_programs(work: Path) -> tuple[list[dict], dict]:
    progs = pipe_cost.read_tsv(work / "programs.tsv")
    for p in progs:
        p["bytes"] = int(p["bytes"])
    return progs, json.loads((work / "warmup.json").read_text(encoding="utf-8"))


def spread(progs: list[dict], count: int, max_bytes: int) -> list[dict]:
    """`count` programs at evenly spaced ranks of the size order, under max_bytes, kinds mixed."""
    pool_ = [p for p in progs if p["bytes"] < max_bytes]
    if len(pool_) <= count:
        return pool_
    return [pool_[round(i * (len(pool_) - 1) / (count - 1))] for i in range(count)]


# ---- runs ----------------------------------------------------------------------------

def smoke(work: Path, count: int) -> int:
    progs, warm = load_programs(work)
    sample = spread(progs, count, 400_000)
    load = quiet_check()
    s = run_job(work, "smoke", header(12, "disabled", "on"), job_body(sample, warm), 1800, 12_000, "smoke")
    s["load_before"] = load
    print(json.dumps({k: s.get(k) for k in ("status", "pipelines", "failed", "validation_errors",
                                             "validation_warnings", "wall_ms", "process_s")}))
    return 0 if s["status"] == "ok" and not s.get("failed") else 1


def coldcheck(work: Path, count: int) -> int:
    progs, warm = load_programs(work)
    sample = spread(progs, count, 400_000)
    quiet_check()
    for tag, cache in (("cold-0", "disabled"), ("cold-1", "disabled"), ("warmdrv-0", "internal"),
                       ("warmdrv-1", "internal")):
        s = run_job(work, f"coldcheck-{tag}", header(1, cache), job_body(sample, warm), 1800, 12_000, tag)
        rows = [r for r in rows_of(work, f"coldcheck-{tag}") if r["what"] == "pipeline"]
        print(json.dumps({"tag": tag, "status": s["status"], "sum_ms": round(sum(float(r["ms"]) for r in rows), 1),
                          "wall_ms": s.get("wall_ms")}), flush=True)
    return 0


def screen(work: Path, tier: str, timeout_s: float, budget_s: float) -> int:
    """R0 (core) or R5 (tail): each program alone, smallest first; resumable."""
    progs, warm = load_programs(work)
    todo = [p for p in progs if p["tier"] == tier]
    done_path = work / "results" / f"screen-{tier}.tsv"
    done = {r["key"] for r in pipe_cost.read_tsv(done_path)} if done_path.is_file() else set()
    cols = ["key", "kind", "bin", "bytes", "group", "container", "status", "ms", "cpu_ms", "module_ms", "result",
            "warmup_ms", "process_s", "peak_ws_mb", "min_available_mb", "gpu_clock_before", "gpu_temp_after"]
    new_file = not done_path.is_file()
    spent = 0.0
    with open(done_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        if new_file:
            w.writeheader()
        for n, p in enumerate(todo):
            if p["key"] in done:
                continue
            if n % 25 == 0:
                quiet_check()
            tag = f"screen-{tier}-{n:03d}"
            s = run_job(work, tag, header(1, "disabled"), job_body([p], warm), timeout_s, MEMORY_KILL_MB,
                        f"{n + 1}/{len(todo)} {p['bytes']} B {p['kind']}")
            rows = rows_of(work, tag)
            item = next((r for r in rows if r["what"] == "pipeline"), None)
            warmrow = next((r for r in rows if r["what"] == "warmup"), None)
            status = s["status"]
            if item and item.get("result") not in ("0", None) and status == "ok":
                status = "refused"
            if status in ("timeout", "killed_memory"):
                ms = s["process_s"] * 1000.0  # censored: the build ran at least this long
            else:
                ms = float(item["ms"]) if item else ""
            w.writerow({"key": p["key"], "kind": p["kind"], "bin": p["bin"], "bytes": p["bytes"], "group": p["group"],
                        "container": p["container"], "status": status, "ms": ms,
                        "cpu_ms": item.get("cpu_ms", "") if item else "",
                        "module_ms": item.get("module_ms", "") if item else "",
                        "result": item.get("result", "") if item else "",
                        "warmup_ms": warmrow["ms"] if warmrow else "", "process_s": s["process_s"],
                        "peak_ws_mb": s.get("peak_working_set_mb", s.get("peak_ws_mb_sampled")),
                        "min_available_mb": s.get("min_available_mb"),
                        "gpu_clock_before": s["gpu_before"].get("clocks.gr", ""),
                        "gpu_temp_after": s["gpu_after"].get("temperature.gpu", "")})
            f.flush()
            print(f"compute_cost: {tag}: {status} {ms if ms == '' else round(ms)} ms for {p['bytes']} B {p['kind']}",
                  flush=True)
            spent += s["process_s"]
            if spent > budget_s:
                print(f"compute_cost: screen budget of {budget_s:.0f} s spent; stopping after {tag}", flush=True)
                return 1
    return 0


def batch(work: Path, tag: str, threads: int, cache: str, cache_name: str, pin32: bool, subset: int,
          kinds: str = "TRP", floor_mb: float = FLOOR_MB) -> int:
    """R1 to R4: the programs screen finished under 60 s, longest first, in one process."""
    progs, warm = load_programs(work)
    by_key = {p["key"]: p for p in progs}
    screened = pipe_cost.read_tsv(work / "results" / "screen-core.tsv")
    ok = [r for r in screened if r["status"] == "ok" and float(r["ms"]) < R1_LIMIT_MS and r["kind"] in kinds]
    ok.sort(key=lambda r: -float(r["ms"]))
    if subset:
        ok = [ok[round(i * (len(ok) - 1) / (subset - 1))] for i in range(subset)]
    params: dict = {"heartbeat_s": 30}
    if pin32:
        params["pin32"] = 1
    cache_file = work / "cache" / f"{cache_name}.vkcache"
    if cache != "none":
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache == "write":
            cache_file.unlink(missing_ok=True)
        elif not cache_file.is_file():
            raise FileNotFoundError(f"no pipeline cache at {cache_file}; run the batch with --cache write first")
        params["pipeline_cache"] = cache_file
    ceiling_s = min(sum(float(r["ms"]) for r in ok) / 1000.0, 3600.0) + 120.0
    (work / "jobs").mkdir(parents=True, exist_ok=True)
    (work / "jobs" / f"{tag}.keys").write_text("".join(r["key"] + "\n" for r in ok), encoding="utf-8")
    load = quiet_check()
    s = run_job(work, tag, header(threads, "disabled", params=params), job_body([by_key[r["key"]] for r in ok], warm),
                ceiling_s, 1 << 40, f"{len(ok)} programs on {threads} threads", floor_mb=floor_mb)
    s["load_before"] = load
    print(json.dumps({k: s.get(k) for k in ("tag", "status", "pipelines", "failed", "wall_ms", "process_s",
                                             "peak_working_set_mb", "pipeline_cache_out_bytes",
                                             "pipeline_cache_create_ms")}))
    return 0 if s["status"] == "ok" else 1


# ---- statistics and the projection -------------------------------------------------------

def item_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return pipe_cost.item_stats(values)


def stratified_total(pop: dict, sample: dict, model=None) -> tuple[float, list]:
    """Sum over cells of N_h times the mean of the cell's sample. A cell with no
    sample takes `model(cell)` (a total for the cell) and is listed."""
    total, modelled = 0.0, []
    for cell, n in pop.items():
        xs = sample.get(cell)
        if xs:
            total += n * sum(xs) / len(xs)
        elif model is not None:
            total += model(cell)
            modelled.append(cell)
        else:
            raise ValueError(f"cell {cell} has programs but no measurement and no model")
    return total, modelled


def bootstrap(pop: dict, sample: dict, stat, reps: int = 2000, seed: int = 21) -> tuple[float, float]:
    """5th and 95th percentile of stat(resampled sample), each cell resampled within itself.
    A cell measured whole (as many measurements as programs) is known exactly and
    is not resampled."""
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        rs = {c: (xs if len(xs) >= pop.get(c, 0) else [rng.choice(xs) for _ in xs])
              for c, xs in sample.items() if xs}
        vals.append(stat(rs))
    return pipe_cost.percentile(vals, 5), pipe_cost.percentile(vals, 95)


def weighted_percentile(pop: dict, sample: dict, q: float) -> float:
    """Percentile of the population, each measured program weighted N_h / n_h."""
    pts = sorted((x, pop[c] / len(xs)) for c, xs in sample.items() if xs and c in pop for x in xs)
    total = sum(w for _, w in pts)
    acc = 0.0
    for x, w in pts:
        acc += w
        if acc >= q / 100.0 * total - 1e-9:
            return x
    return pts[-1][0]


def projected_over(pop: dict, sample: dict, sizes: dict, threshold: float) -> tuple[float, float]:
    """Programs and bytes of the population whose time exceeds threshold (cell rates)."""
    count = byts = 0.0
    for c, xs in sample.items():
        if not xs or c not in pop:
            continue
        frac = sum(x > threshold for x in xs) / len(xs)
        count += pop[c] * frac
        byts += pop[c] * frac * sizes.get(c, 0.0)
    return count, byts


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def loglog_fit(pairs: list[tuple[float, float]]) -> dict:
    """log(ms) = a + b log(bytes), least squares, with Duan's smearing factor."""
    xs = [math.log(b) for b, _ in pairs]
    ys = [math.log(max(ms, 1e-3)) for _, ms in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in res) / max(1, n - 2))
    return {"a": a, "b": b, "residual_sd": sd, "smear": sum(math.exp(r) for r in res) / n, "n": n,
            "spearman": spearman([p[0] for p in pairs], [p[1] for p in pairs])}


def predict(fit: dict, nbytes: float) -> float:
    return math.exp(fit["a"] + fit["b"] * math.log(nbytes)) * fit["smear"]


def report(work: Path, census_path: Path, g12_s: float, budget: tuple[float, float], groups: dict[str, str]) -> int:
    """groups maps kinds to the batch whose throughput they take, for example
    {"RP": "R1", "T": "R1"} when one batch held everything."""
    pop_rows = [r for r in pipe_cost.read_tsv(census_path) if r["status"] == "ok" and r["strict_ok"] == "1"]
    not_built = [r for r in pipe_cost.read_tsv(census_path) if not (r["status"] == "ok" and r["strict_ok"] == "1")]
    pop = population_cells(pop_rows)
    cell_bytes: dict = defaultdict(list)
    for r in pop_rows:
        cell_bytes[(r["kind"], r["bin"])].append(int(r["spv_bytes"]))
    mean_bytes = {c: sum(v) / len(v) for c, v in cell_bytes.items()}
    screened = []
    for tier in ("core", "tail"):
        p = work / "results" / f"screen-{tier}.tsv"
        if p.is_file():
            screened += pipe_cost.read_tsv(p)
    timed = [r for r in screened if r["status"] in ("ok", "timeout", "killed_memory") and r["ms"] != ""]
    censored = [r for r in timed if r["status"] != "ok"]
    sample: dict = defaultdict(list)
    for r in timed:
        sample[(r["kind"], r["bin"])].append(float(r["ms"]))
    fits = {k: loglog_fit([(int(r["bytes"]), float(r["ms"])) for r in timed if r["kind"] == k and r["status"] == "ok"])
            for k in KINDS if sum(r["kind"] == k and r["status"] == "ok" for r in timed) >= 5}

    def model(cell):
        fit = fits.get(cell[0]) or fits.get("P")
        return sum(predict(fit, b) for b in cell_bytes[cell])

    c1, modelled = stratified_total(pop, sample, model)
    lo, hi = bootstrap(pop, sample, lambda s: stratified_total(pop, {**sample, **s}, model)[0])
    model_sum = sum(predict(fits.get(r["kind"]) or fits["P"], int(r["spv_bytes"])) for r in pop_rows)

    out: dict = {"population": {"programs": len(pop_rows), "not_built": len(not_built),
                                "cells": {f"{k} {b}": n for (k, b), n in sorted(pop.items())}},
                 "screen": {"programs": len(screened), "timed": len(timed), "censored": len(censored),
                            "refused": sum(r["status"] == "refused" for r in screened),
                            "crash": sum(r["status"] == "crash" for r in screened)}}
    ok_ms = [float(r["ms"]) for r in timed]
    out["screen"]["all"] = item_stats(ok_ms)
    out["screen"]["by_kind"] = {k: item_stats([float(r["ms"]) for r in timed if r["kind"] == k]) for k in KINDS}
    out["screen"]["by_bin"] = {b: item_stats([float(r["ms"]) for r in timed if r["bin"] == b]) for b in BIN_NAMES}
    out["screen"]["over_10s"] = sorted(({"bytes": int(r["bytes"]), "kind": r["kind"], "group": r["group"],
                                         "container": r["container"], "ms": round(float(r["ms"])),
                                         "status": r["status"]} for r in timed if float(r["ms"]) > 10_000),
                                       key=lambda x: -x["ms"])
    out["screen"]["thresholds"] = {f">{t}ms": sum(x > t for x in ok_ms) for t in (200, 1000, 10_000, 100_000)}
    out["screen"]["cpu_over_wall"] = round(sum(float(r["cpu_ms"]) for r in timed if r["cpu_ms"])
                                           / max(1.0, sum(float(r["ms"]) for r in timed if r["cpu_ms"])), 3)

    # Runs in one process (smoke, coldcheck, batches).
    runs = {}
    for line in open(work / "results" / "summary.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s["tag"].startswith("screen-"):
            continue
        rows = [r for r in rows_of(work, s["tag"]) if r["what"] == "pipeline"]
        runs[s["tag"]] = {"summary": {k: s.get(k) for k in ("status", "threads", "pipelines", "failed", "wall_ms",
                                                            "process_s", "peak_working_set_mb", "pin32",
                                                            "pipeline_cache_in_bytes", "pipeline_cache_out_bytes",
                                                            "pipeline_cache_create_ms", "available_mb_start",
                                                            "available_mb_end", "gpu_before", "gpu_after", "driver")},
                          "items": item_stats([float(r["ms"]) for r in rows if r.get("result") == "0"]),
                          "cpu_ms_sum": round(sum(float(r["cpu_ms"]) for r in rows if r.get("cpu_ms")), 1)}
    out["runs"] = runs

    r0 = {r["key"]: float(r["ms"]) for r in timed if r["status"] == "ok"}
    longest = max(ok_ms) if ok_ms else 0.0
    # Throughput of each batch: the one-thread time of its programs over its wall.
    speedups = {}
    for tag, run in sorted(runs.items()):
        keys_file = work / "jobs" / f"{tag}.keys"
        if not keys_file.is_file() or run["summary"].get("status") != "ok" or not run["summary"].get("wall_ms"):
            continue
        keys = keys_file.read_text(encoding="utf-8").split()
        speedups[tag] = sum(r0[k] for k in keys) / run["summary"]["wall_ms"]

    def kind_pop(kinds: str) -> dict:
        return {c: n for c, n in pop.items() if c[0] in kinds}

    def wall(s: dict) -> float:
        """Wall of the whole population, each group of kinds at its batch's throughput."""
        return sum(stratified_total(kind_pop(g), s, model)[0] / speedups[t] for g, t in groups.items())

    by_kind = {k: stratified_total(kind_pop(k), sample, model)[0] for k in KINDS}
    by_kind_iv = {k: bootstrap(kind_pop(k), sample, lambda s, k=k: stratified_total(kind_pop(k), {**sample, **s},
                                                                                   model)[0]) for k in KINDS}
    out["projection"] = {
        "C1_ms": round(c1), "C1_interval_ms": [round(lo), round(hi)], "cells_modelled": [f"{k} {b}" for k, b in modelled],
        "C1_by_kind_ms": {k: [round(by_kind[k]), round(by_kind_iv[k][0]), round(by_kind_iv[k][1])] for k in KINDS},
        "lower_bound": bool(censored), "longest_ms": round(longest),
        "speedups": {t: round(v, 2) for t, v in speedups.items()}, "groups": groups,
        "model": {k: {x: round(v, 4) for x, v in f.items()} for k, f in fits.items()},
        "model_total_ms": round(model_sum), "model_vs_cells": round(model_sum / c1, 2) if c1 else None,
        "percentiles_ms": {f"p{q}": round(weighted_percentile(pop, sample, q), 1) for q in (50, 90, 99)},
        "share_500k": round(sum(n * sum(sample[c]) / len(sample[c]) for c, n in pop.items()
                                if sample.get(c) and c[1] in ("0.5-1M", "1-2M", ">2M")) / c1, 3) if c1 else None,
        "share_1m": round(sum(n * sum(sample[c]) / len(sample[c]) for c, n in pop.items()
                              if sample.get(c) and c[1] in ("1-2M", ">2M")) / c1, 3) if c1 else None,
    }
    if groups and all(t in speedups for t in groups.values()):
        w = max(wall(sample), longest)
        wlo, whi = bootstrap(pop, sample, lambda s: wall({**sample, **s}))
        wlo, whi = max(wlo, longest), max(whi, longest)
        out["projection"]["W_ms"] = round(w)
        out["projection"]["W_interval_ms"] = [round(wlo), round(whi)]
        b_lo, b_hi = budget
        hi_total = whi / 60000.0 + g12_s / 60.0
        out["tier"] = {"budget_min": [b_lo, b_hi], "graphics_min": round(g12_s / 60.0, 2),
                       "W_hi_plus_graphics_min": round(hi_total, 2), "rule_A": hi_total <= b_lo, "miss_bound": {}}
        for m in (500, 1000, 2000, 5000, 10_000):
            cnt, byts = projected_over(pop, sample, mean_bytes, m)
            clo, chi = bootstrap(pop, sample, lambda s, m=m: projected_over(pop, s, mean_bytes, m)[0])
            forced = 0.0
            for g, t in groups.items():
                forced += sum(pop[c] / len(xs) * sum(x for x in xs if x > m)
                              for c, xs in sample.items() if xs and c in pop and c[0] in g) / speedups[t]
            w_tier1 = max(forced, max([x for x in ok_ms if x > m], default=0.0)) / 60000.0
            out["tier"]["miss_bound"][f"{m}ms"] = {
                "programs": round(cnt), "programs_interval": [round(clo), round(chi)], "bytes": round(byts),
                "W_tier1_min": round(w_tier1, 2), "fits_low_budget": w_tier1 + g12_s / 60.0 <= b_lo,
                "fits_compute_alone_low_budget": w_tier1 <= b_lo}
    (work / "report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census")
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--split", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--census", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    for name in ("smoke", "coldcheck"):
        s = sub.add_parser(name)
        s.add_argument("--work", type=Path, required=True)
        s.add_argument("--count", type=int, default=20)
    s = sub.add_parser("screen")
    s.add_argument("--work", type=Path, required=True)
    s.add_argument("--tier", choices=("core", "tail"), default="core")
    s.add_argument("--timeout", type=float, default=None, help="seconds per program (core 900, tail 3600)")
    s.add_argument("--budget", type=float, default=7200.0, help="seconds of screening before stopping")
    b = sub.add_parser("batch")
    b.add_argument("--work", type=Path, required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--threads", type=int, default=12)
    b.add_argument("--cache", choices=("none", "write", "read"), default="none")
    b.add_argument("--pin32", action="store_true")
    b.add_argument("--subset", type=int, default=0)
    b.add_argument("--cache-name", default="R1", help="the pipeline cache file under work/cache")
    b.add_argument("--kinds", default="TRP", help="which kinds to batch: T tensor, R raw access, P plain")
    b.add_argument("--floor-mb", type=float, default=FLOOR_MB, help="kill the run when available RAM falls under this")
    r = sub.add_parser("report")
    r.add_argument("--work", type=Path, required=True)
    r.add_argument("--census", type=Path, required=True)
    r.add_argument("--g12", type=float, default=0.0, help="seconds: graphics library builds on 12 threads (S3)")
    r.add_argument("--budget-min", type=float, nargs=2, default=(12.0, 32.0))
    r.add_argument("--groups", default="TRP=R1", help="KINDS=BATCH,...: the batch whose throughput each kind takes")
    args = ap.parse_args(argv)
    if getattr(args, "work", None) and str(args.work.resolve()).startswith(str(ROOT)):
        print(f"compute_cost: --work {args.work} is inside the repository; results name the game's programs.",
              file=sys.stderr)
        return 2
    try:
        if args.cmd == "census":
            return census(args.run, args.split, args.out)
        if args.cmd == "prepare":
            return prepare(args.census, args.selection, args.run, args.work)
        if args.cmd == "smoke":
            return smoke(args.work, args.count)
        if args.cmd == "coldcheck":
            return coldcheck(args.work, args.count)
        if args.cmd == "screen":
            timeout = args.timeout or (900.0 if args.tier == "core" else 3600.0)
            return screen(args.work, args.tier, timeout, args.budget)
        if args.cmd == "batch":
            return batch(args.work, args.tag, args.threads, args.cache, args.cache_name, args.pin32, args.subset,
                         args.kinds, args.floor_mb)
        if args.cmd == "report":
            groups = dict(g.split("=", 1) for g in args.groups.split(",") if g)
            return report(args.work, args.census, args.g12, tuple(args.budget_min), groups)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"compute_cost: {e}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
