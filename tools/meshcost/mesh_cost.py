#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Spike S4: how long the driver takes to build the game's translated task and mesh pipelines cold.

    py -3 tools/meshcost/mesh_cost.py select --split DIR --work DIR
    py -3 tools/meshcost/mesh_cost.py translate --split DIR --xlate EXE --work DIR [--workers N]
    py -3 tools/meshcost/mesh_cost.py prepare --work DIR [--triples N]
    py -3 tools/meshcost/mesh_cost.py smoke --work DIR [--count N]
    py -3 tools/meshcost/mesh_cost.py coldcheck --work DIR [--count N]
    py -3 tools/meshcost/mesh_cost.py screen --work DIR [--count N] [--largest N] [--timeout S]
    py -3 tools/meshcost/mesh_cost.py batch --work DIR --tag T [--threads N] [--pin32 on|off]
                                            [--cache none|write|read] [--cache-name NAME] [--phases pre|pre,frag,link]
    py -3 tools/meshcost/mesh_cost.py report --work DIR [--budget-min M M]

select reads the splitter's output (pairing.tsv and jobs.jsonl) and writes the sets the harness builds: one
pre-rasterization library for every distinct (task, mesh) pair (a mesh stage with no task is a pair with an empty task),
the (task, mesh, pixel) triples, and the task and mesh programs no archive pairs. Pairs and triples carry a scope: `core`
when the variation sits in the update's Foliage or Model archive (research 13 finding 23: 338 pairs, 1,131 triples) and
`package` for ApplicationPackage's own copies, so both readings are reported. translate runs the jobs of those sets
through the translator and spirv-val. prepare writes the job lists. The runs, all with the driver's internal pipeline
cache disabled (S3's method), are:
  smoke      a spread of libraries, fast-linked, once with validation on; then the counter-run that leaves the push
             constants out of the layout, which validation must report
  coldcheck  the same libraries twice with the cache disabled and twice enabled
  screen     each library alone in its own process: the uncontended time and the peak working set of one build
  batch      every library (and with --phases pre,frag,link the fragment libraries and the fast links) in one process on
             N threads, the largest first; --pin32 off drops the required subgroup size; --cache write saves the
             application's VkPipelineCache and --cache read builds from it
report gives per-run statistics, the thread curve, the pin's overhead, the warm path, the memory per build and the
projection against budget B.

Everything this writes goes under --work and names the game's programs: none of it may enter the repository; counts
only go into a note.

Exit codes: 0 done, 1 a measurement failed or the machine is not quiet, 2 an input or a tool is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "computecost"))

import compute_cost as cc  # noqa: E402  (the supervisor: run_job, quiet_check, rows_of; imports pipe_cost and pool)

pipe_cost = cc.pipe_cost
pool = cc.pool
cc.BUSY = tuple(dict.fromkeys(cc.BUSY + ("vkmeshcost", "vkcomputecost")))  # the supervisor's busy list

CORE_ARCHIVES = ("Foliage.Product.110", "Model.Product.110")   # research 13's walk; the rest is ApplicationPackage's copies
MEMORY_KILL_MB = 6_000
FLOOR_MB = 1_500
MESH_SPV_TO_VK = dict(cc.SPV_TO_VK, SPV_EXT_mesh_shader="VK_EXT_mesh_shader")
DEVICE = "RTX 3060 Ti"


# ---- the sets ----------------------------------------------------------------------------------

def read_pairing(path: Path) -> list[dict]:
    """The rows of pairing.tsv that hold a mesh stage: variation, task, mesh, pixel, product100 ('' where absent)."""
    rows = []
    for r in pipe_cost.read_tsv(path):
        if r.get("mesh"):
            rows.append({"variation": r["variation"], "task": r.get("task", ""), "mesh": r["mesh"],
                         "pixel": r.get("pixel", ""), "product100": r.get("product100", "0")})
    return rows


def scope_of(variation: str) -> str:
    """`core` for a variation of the update's Foliage or Model archive, else `package`."""
    name = variation.split("#")[0].rsplit("/", 1)[-1]
    return "core" if any(name.startswith(a) for a in CORE_ARCHIVES) else "package"


def select_sets(rows: list[dict], job_stage: dict[str, str]) -> dict:
    """Pairs, triples and the unpaired programs. `job_stage` maps a job key to its stage (ts, ms, ps, ...)."""
    pairs: dict[tuple[str, str], str] = {}
    triples: dict[tuple[str, str, str], str] = {}
    for r in rows:
        if "?" in (r["task"], r["mesh"]) or not r["mesh"]:
            continue  # a stage the splitter could not read: no pair and no triple
        s = scope_of(r["variation"])
        tables = [(pairs, (r["task"], r["mesh"]))]
        if r["pixel"] not in ("", "?"):  # a pair needs no pixel stage, a triple does
            tables.append((triples, (r["task"], r["mesh"], r["pixel"])))
        for table, key in tables:
            table[key] = "core" if (table.get(key) == "core" or s == "core") else "package"
    used_t = {p[0] for p in pairs if p[0]}
    used_m = {p[1] for p in pairs}
    unpaired = {"ts": sorted(k for k, s in job_stage.items() if s == "ts" and k not in used_t),
                "ms": sorted(k for k, s in job_stage.items() if s == "ms" and k not in used_m)}
    out = {
        "pairs": [{"task": t, "mesh": m, "scope": sc} for (t, m), sc in sorted(pairs.items())],
        "triples": [{"task": t, "mesh": m, "pixel": p, "scope": sc} for (t, m, p), sc in sorted(triples.items())],
        "unpaired": unpaired,
    }
    out["counts"] = {
        "pairs": len(pairs), "pairs_core": sum(v == "core" for v in pairs.values()),
        "pairs_tasked": sum(bool(t) for t, _ in pairs), "triples": len(triples),
        "triples_core": sum(v == "core" for v in triples.values()),
        "pixel_modules": len({p for _, _, p in triples}), "unpaired_task": len(unpaired["ts"]),
        "unpaired_mesh": len(unpaired["ms"]),
    }
    return out


def read_jobs(path: Path) -> dict[str, dict]:
    jobs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        j = json.loads(line)
        jobs[j["key"]] = j
    return jobs


def select(split: Path, work: Path) -> int:
    rows = read_pairing(split / "pairing.tsv")
    jobs = read_jobs(split / "jobs.jsonl")
    sel = select_sets(rows, {k: j["stage"] for k, j in jobs.items()})
    work.mkdir(parents=True, exist_ok=True)
    (work / "selection.json").write_text(json.dumps(sel, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(sel["counts"]))
    return 0


# ---- translation ----------------------------------------------------------------------------------

def needed_keys(sel: dict) -> list[str]:
    keys = {p["mesh"] for p in sel["pairs"]} | {p["task"] for p in sel["pairs"] if p["task"]}
    keys |= set(sel["unpaired"]["ts"]) | set(sel["unpaired"]["ms"]) | {t["pixel"] for t in sel["triples"]}
    return sorted(keys)


def raw_flag(res: dict) -> int:
    ra = res.get("raw_access") or {}
    return int(bool(ra.get("physical_pointers") or ra.get("page_table") or ra.get("table")))


def translate(split: Path, xlate: str, work: Path, workers: int) -> int:
    import validate
    sel = json.loads((work / "selection.json").read_text(encoding="utf-8"))
    jobs = read_jobs(split / "jobs.jsonl")
    keys = needed_keys(sel)
    missing = [k for k in keys if k not in jobs]
    if missing:
        print(f"mesh_cost: {len(missing)} keys of the selection are not in jobs.jsonl", file=sys.stderr)
        return 2
    out = work / "spv"
    out.mkdir(parents=True, exist_ok=True)
    exe = str(Path(xlate).resolve())
    t0 = time.perf_counter()
    results, _ = pool.run_pool([exe], [(jobs[k], out) for k in keys], workers, work / "logs", timeout=300)
    by_key = {r.key: r for r in results}
    tool = validate.find_spirv_val()
    rows = []
    for k in keys:
        r = by_key.get(k)
        row = {"key": k, "stage": jobs[k]["stage"], "status": getattr(r, "status", "none"), "bytes": 0, "raw": 0,
               "exts": "", "valid": 0, "message": ""}
        if r is not None and r.status == "ok":
            spv = out / f"{k}.spv"
            data = spv.read_bytes()
            res = json.loads((out / f"{k}.res.json").read_text(encoding="utf-8"))
            row.update(bytes=len(data), raw=raw_flag(res), exts=",".join(cc.spirv_extensions(data)),
                       valid=int(validate.validate_one(tool, k, spv).ok))
        elif r is not None:
            row["message"] = re.sub(r"[0-9a-f]{3,}|\d+", "N", r.message)[:160]
        rows.append(row)
    with open(work / "translated.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    c = Counter((r["stage"], r["status"], r["valid"]) for r in rows)
    print(json.dumps({"jobs": len(rows), "by_stage_status_valid": {"/".join(map(str, k)): v for k, v in sorted(c.items())},
                      "seconds": round(time.perf_counter() - t0, 1)}))
    return 0 if all(r["status"] == "ok" and r["valid"] for r in rows) else 1


# ---- the job lists -----------------------------------------------------------------------------------

def device_extensions(exts: list[str]) -> list[str]:
    need = set()
    for e in exts:
        if e not in MESH_SPV_TO_VK:
            raise ValueError(f"unknown SPIR-V extension {e}")
        if MESH_SPV_TO_VK[e]:
            need.add(MESH_SPV_TO_VK[e])
    return sorted(need)


def load_translated(work: Path) -> dict[str, dict]:
    out = {}
    for r in pipe_cost.read_tsv(work / "translated.tsv"):
        r["bytes"], r["raw"], r["valid"] = int(r["bytes"]), int(r["raw"]), int(r["valid"])
        out[r["key"]] = r
    return out


def build_lists(sel: dict, tr: dict[str, dict]) -> dict:
    """The pre-rasterization items (pairs, then the unpaired mesh programs), ordered largest first, and the fragment
    items (the pixel modules of the triples), each with what the harness and the report need."""
    def ok(k: str) -> bool:
        return k == "" or (k in tr and tr[k]["status"] == "ok" and tr[k]["valid"])
    pre, excluded = [], 0
    for p in sel["pairs"]:
        if ok(p["task"]) and ok(p["mesh"]):
            pre.append({"task": p["task"], "mesh": p["mesh"], "scope": p["scope"]})
        else:
            excluded += 1
    paired_m = {p["mesh"] for p in sel["pairs"]}
    for m in sel["unpaired"]["ms"]:
        if m not in paired_m and ok(m):
            pre.append({"task": "", "mesh": m, "scope": "unpaired"})
    for it in pre:
        it["task_bytes"] = tr[it["task"]]["bytes"] if it["task"] else 0
        it["mesh_bytes"] = tr[it["mesh"]]["bytes"]
        it["bytes"] = it["task_bytes"] + it["mesh_bytes"]
        it["kind"] = "tm" if it["task"] else "m"
        it["raw"] = int(bool(tr[it["mesh"]]["raw"] or (it["task"] and tr[it["task"]]["raw"])))
    pre.sort(key=lambda it: (-it["bytes"], it["task"], it["mesh"]))
    pixels = sorted({t["pixel"] for t in sel["triples"] if ok(t["pixel"]) and ok(t["mesh"]) and ok(t["task"])})
    frag = [{"pixel": k, "bytes": tr[k]["bytes"], "raw": tr[k]["raw"]} for k in pixels]
    frag.sort(key=lambda f: (-f["bytes"], f["pixel"]))
    return {"pre": pre, "frag": frag, "excluded_pairs": excluded}


def missing_components(mesh_outputs: list[dict], pixel_inputs: list[dict]) -> tuple:
    """The (location, component) pairs a pixel stage reads that the mesh stage before it does not write, sorted: the stage
    interface gap of research 20 finding G (VUID-RuntimeSpirv-OpEntryPoint-08743). `*_outputs` and `*_inputs` are a
    description's interface lists. The runtime pads a stage with exactly these outputs (ticket 60), so the pairs are a pad set."""
    written = {(o["location"], o["component"] + i) for o in mesh_outputs for i in range(o["components"])}
    wanted = {(i["location"], i["component"] + c) for i in pixel_inputs for c in range(i["components"])}
    return tuple(sorted(wanted - written))


def interface_gap(mesh_outputs: list[dict], pixel_inputs: list[dict]) -> int:
    """How many components that is."""
    return len(missing_components(mesh_outputs, pixel_inputs))


def triple_missing(work: Path, triples: list[dict]) -> list[tuple]:
    """The pad set of each triple, from the translator's descriptions (each module read once)."""
    cache: dict[tuple[str, str], list[dict]] = {}

    def side(key: str, which: str) -> list[dict]:
        if (key, which) not in cache:
            res = json.loads((work / "spv" / f"{key}.res.json").read_text(encoding="utf-8"))
            cache[(key, which)] = res["interface"][which]
        return cache[(key, which)]
    return [missing_components(side(t["mesh"], "outputs"), side(t["pixel"], "inputs")) for t in triples]


def triple_gaps(work: Path, triples: list[dict]) -> list[int]:
    return [len(m) for m in triple_missing(work, triples)]


def targets_of(work: Path, key: str) -> int:
    res = json.loads((work / "spv" / f"{key}.res.json").read_text(encoding="utf-8"))
    return max(1, pipe_cost.targets((res.get("graphics") or {}).get("frag_out_map", 0)))


def spread_ids(n_total: int, count: int) -> list[int]:
    """`count` evenly spaced indices of 0..n_total-1 (all of them when count is not smaller)."""
    if count >= n_total:
        return list(range(n_total))
    if count <= 1:
        return [0][:count]
    return sorted({round(i * (n_total - 1) / (count - 1)) for i in range(count)})


def job_lines(work: Path, lists: dict, tr: dict[str, dict], pre_ids: list[int], triples: list[dict],
              warm: dict | None, phases: str) -> list[str]:
    """The body of a job: modules, extensions, warm-up, pre items (pre_ids of lists['pre']), the fragment items and the
    links of `triples` (a triple's pre and frag items must be among the ones listed)."""
    spv = work / "spv"
    modules: list[str] = []
    index: dict[str, int] = {}
    exts: set[str] = set()

    def mod(key: str, stage: str) -> int:
        if key not in index:
            index[key] = len(modules)
            modules.append(f"module {stage} {spv / (key + '.spv')}")
            if tr[key]["exts"]:
                exts.update(tr[key]["exts"].split(","))
        return index[key]

    pre_lines, pre_of = [], {}
    for n, i in enumerate(pre_ids):
        it = lists["pre"][i]
        t = mod(it["task"], "t") if it["task"] else "-"
        pre_lines.append(f"pre {t} {mod(it['mesh'], 'm')}")
        pre_of[(it["task"], it["mesh"])] = n
    frag_lines, frag_of = [], {}
    link_lines = []
    if "frag" in phases:
        needed = sorted({t["pixel"] for t in triples}) if triples else [f["pixel"] for f in lists["frag"]]
        frag_keys = [f["pixel"] for f in lists["frag"] if f["pixel"] in set(needed)]
        for n, k in enumerate(frag_keys):
            frag_lines.append(f"frag {mod(k, 'f')}")
            frag_of[k] = n
        for t in triples:
            link_lines.append(f"link {pre_of[(t['task'], t['mesh'])]} {frag_of[t['pixel']]} {targets_of(work, t['pixel'])}")
    warm_lines = []
    if warm:
        t = mod(warm["task"], "t") if warm["task"] else "-"
        warm_lines.append(f"warmup {t} {mod(warm['mesh'], 'm')}")
    head = [f"extension {e}" for e in device_extensions(sorted(exts))]
    return head + modules + warm_lines + pre_lines + frag_lines + (link_lines if "link" in phases else [])


def header(threads: int, cache: str, validation: str = "off", params: dict | None = None) -> list[str]:
    return ([f"validation {validation}", f"cache {cache}", f"threads {threads}", f"device {DEVICE}"]
            + [f"param {k} {v}" for k, v in (params or {}).items()])


def exe_path() -> Path:
    exe = ROOT / "build" / "vkmeshcost" / "vkmeshcost.exe"
    if not exe.is_file():
        raise FileNotFoundError(f"no harness at {exe}; run tools/build_vkdraw.py --harness vkmeshcost")
    return exe


def run_job(work: Path, tag: str, head: list[str], body: list[str], timeout_s: float, memory_kill_mb: float,
            label: str = "", floor_mb: float = FLOOR_MB) -> dict:
    """The supervisor's run_job (compute_cost.py) pointed at this harness."""
    saved = cc.exe_path
    cc.exe_path = exe_path
    try:
        return cc.run_job(work, tag, head, body, timeout_s, memory_kill_mb, label, floor_mb=floor_mb)
    finally:
        cc.exe_path = saved


def load_lists(work: Path) -> tuple[dict, dict]:
    lists = json.loads((work / "lists.json").read_text(encoding="utf-8"))
    return lists, load_translated(work)


def prepare(work: Path, triples: int) -> int:
    sel = json.loads((work / "selection.json").read_text(encoding="utf-8"))
    tr = load_translated(work)
    lists = build_lists(sel, tr)
    ok = lambda k: k == "" or (k in tr and tr[k]["status"] == "ok" and tr[k]["valid"])  # noqa: E731
    pre_key = {(it["task"], it["mesh"]) for it in lists["pre"]}
    frag_key = {f["pixel"] for f in lists["frag"]}
    linkable = [t for t in sel["triples"] if (t["task"], t["mesh"]) in pre_key and t["pixel"] in frag_key
                and ok(t["task"]) and ok(t["mesh"]) and ok(t["pixel"])]
    pad_sets: dict[tuple[str, str], set] = defaultdict(set)
    for t, missing in zip(linkable, triple_missing(work, linkable)):
        t["gap"] = len(missing)
        pad_sets[(t["task"], t["mesh"])].add(missing)
    lists["pad_libraries"] = sum(len(v) for v in pad_sets.values())
    lists["triples"] = linkable if not triples else [linkable[i] for i in spread_ids(len(linkable), triples)]
    (work / "lists.json").write_text(json.dumps(lists, separators=(",", ":")), encoding="utf-8")
    with open(work / "pre_items.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["index", "scope", "kind", "bytes", "task_bytes", "mesh_bytes", "raw"])
        for i, it in enumerate(lists["pre"]):
            w.writerow([i, it["scope"], it["kind"], it["bytes"], it["task_bytes"], it["mesh_bytes"], it["raw"]])
    by_scope = Counter(it["scope"] for it in lists["pre"])
    print(json.dumps({"pre": len(lists["pre"]), "by_scope": dict(by_scope), "frag": len(lists["frag"]),
                      "triples_linkable": len(linkable), "triples_listed": len(lists["triples"]),
                      "pad_libraries": lists["pad_libraries"], "triples_with_gap": sum(t["gap"] > 0 for t in linkable),
                      "triples_with_gap_core": sum(t["gap"] > 0 and t["scope"] == "core" for t in linkable),
                      "excluded_pairs": lists["excluded_pairs"]}))
    return 0


def warm_item(lists: dict) -> dict:
    """The warm-up library: the smallest of the pre items (it is built again, measured, in the run)."""
    return lists["pre"][-1]


# ---- runs --------------------------------------------------------------------------------------------

def quiet(skip: bool) -> dict:
    return {} if skip else cc.quiet_check()


def smoke(work: Path, count: int, no_quiet: bool, threads: int = 6) -> int:
    lists, tr = load_lists(work)
    ids = spread_ids(len(lists["pre"]), count)
    # one pair of every shape: the smallest and largest of a task and mesh pair, a mesh alone, and a raw-access one
    for kind in ("tm", "m"):
        of = [i for i, it in enumerate(lists["pre"]) if it["kind"] == kind]
        ids += of[:1] + of[-1:]
    ids += [i for i, it in enumerate(lists["pre"]) if it["raw"]][:3]
    ids = sorted(set(ids))
    # The links are those of triples with no interface gap (a gap is a finding of its own, run apart below); their pairs
    # are built too, whatever the spread chose.
    gap_free = [t for t in lists["triples"] if t["gap"] == 0]
    gapped = [t for t in lists["triples"] if t["gap"] > 0]
    triples = [gap_free[i] for i in spread_ids(len(gap_free), 20)]
    pre_index = {(it["task"], it["mesh"]): i for i, it in enumerate(lists["pre"])}
    ids = sorted(set(ids) | {pre_index[(t["task"], t["mesh"])] for t in triples})
    body = job_lines(work, lists, tr, ids, triples, warm_item(lists), "pre,frag,link")
    gap_triples = [gapped[i] for i in spread_ids(len(gapped), 3)]
    gap_ids = sorted({pre_index[(t["task"], t["mesh"])] for t in gap_triples})
    gap_body = job_lines(work, lists, tr, gap_ids, gap_triples, warm_item(lists), "pre,frag,link")
    load = quiet(no_quiet)
    verdict = {}
    params = {"phases": "pre,frag,link", "heartbeat_s": 30}
    for tag, extra, text in (("smoke", {}, body), ("smoke-counter-nopush", {"counter": "nopush"}, body),
                             ("smoke-gap", {}, gap_body)):
        s = run_job(work, tag, header(threads, "disabled", "on", dict(params, **extra)), text, 1800, 12_000, tag)
        s["load_before"] = load
        verdict[tag] = {k: s.get(k) for k in ("status", "pre_items", "frag_items", "link_items", "failed",
                                              "validation_errors", "validation_warnings", "process_s")}
        print(tag, json.dumps(verdict[tag]), flush=True)
    clean, caught = verdict["smoke"], verdict["smoke-counter-nopush"]
    ok = (clean["status"] == "ok" and not clean["failed"] and clean["validation_errors"] == 0
          and clean["validation_warnings"] == 0 and (caught.get("validation_errors") or 0) > 0)
    print("smoke:", "clean, and the counter-run was caught" if ok else "NOT as expected",
          "| gapped triples:", json.dumps({k: verdict["smoke-gap"].get(k) for k in ("validation_errors", "failed")}))
    return 0 if ok else 1


def coldcheck(work: Path, count: int, no_quiet: bool) -> int:
    lists, tr = load_lists(work)
    ids = spread_ids(len(lists["pre"]), count)
    body = job_lines(work, lists, tr, ids, [], warm_item(lists), "pre")
    quiet(no_quiet)
    bad = 0
    for tag, cache in (("cold-0", "disabled"), ("cold-1", "disabled"), ("warmdrv-0", "internal"), ("warmdrv-1", "internal")):
        s = run_job(work, f"coldcheck-{tag}", header(1, cache, params={"phases": "pre"}), body, 1800, 12_000, tag)
        rows = [r for r in cc.rows_of(work, f"coldcheck-{tag}") if r["phase"] == "pre" and r.get("result") == "0"]
        bad += s["status"] != "ok" or bool(s.get("failed"))
        print(json.dumps({"tag": tag, "status": s["status"], "built": len(rows), "sum_ms": round(sum(float(r["ms"]) for r in rows), 1)}),
              flush=True)
    return 1 if bad else 0


def screen(work: Path, count: int, largest: int, timeout_s: float, no_quiet: bool) -> int:
    """Each library alone in its own process (resumable): the uncontended time and the peak working set of one build."""
    lists, tr = load_lists(work)
    n = len(lists["pre"])
    ids = sorted(set(spread_ids(n, count)) | set(range(min(largest, n))))
    path = work / "results" / "screen.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {int(r["index"]) for r in pipe_cost.read_tsv(path)} if path.is_file() else set()
    cols = ["index", "scope", "kind", "bytes", "status", "ms", "cpu_ms", "module_ms", "result", "warmup_ms", "process_s",
            "peak_ws_mb", "min_available_mb"]
    new = not path.is_file()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        if new:
            w.writeheader()
        for step, i in enumerate(ids):
            if i in done:
                continue
            if step % 25 == 0:
                quiet(no_quiet)
            it = lists["pre"][i]
            tag = f"screen-{i:04d}"
            body = job_lines(work, lists, tr, [i], [], warm_item(lists), "pre")
            s = run_job(work, tag, header(1, "disabled", params={"phases": "pre"}), body, timeout_s, MEMORY_KILL_MB,
                        f"{step + 1}/{len(ids)} {it['bytes']} B {it['kind']}")
            rows = cc.rows_of(work, tag)
            item = next((r for r in rows if r["phase"] == "pre"), None)
            warmrow = next((r for r in rows if r["phase"] == "warmup"), None)
            status = s["status"]
            if item and item.get("result") not in ("0", None) and status == "ok":
                status = "refused"
            ms = s["process_s"] * 1000.0 if status in ("timeout", "killed_memory") else (float(item["ms"]) if item else "")
            w.writerow({"index": i, "scope": it["scope"], "kind": it["kind"], "bytes": it["bytes"], "status": status,
                        "ms": ms, "cpu_ms": item.get("cpu_ms", "") if item else "",
                        "module_ms": item.get("module_ms", "") if item else "", "result": item.get("result", "") if item else "",
                        "warmup_ms": warmrow["ms"] if warmrow else "", "process_s": s["process_s"],
                        "peak_ws_mb": s.get("peak_working_set_mb", s.get("peak_ws_mb_sampled")),
                        "min_available_mb": s.get("min_available_mb")})
            f.flush()
            print(f"mesh_cost: {tag}: {status} {ms if ms == '' else round(ms)} ms for {it['bytes']} B {it['kind']}", flush=True)
    return 0


def batch(work: Path, tag: str, threads: int, pin32: bool, cache: str, cache_name: str, phases: str,
          no_quiet: bool, floor_mb: float) -> int:
    lists, tr = load_lists(work)
    ids = list(range(len(lists["pre"])))
    triples = lists["triples"] if "link" in phases else []
    params: dict = {"phases": phases, "pin32": 1 if pin32 else 0, "heartbeat_s": 30}
    cache_file = work / "cache" / f"{cache_name}.vkcache"
    if cache != "none":
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache == "write":
            cache_file.unlink(missing_ok=True)
        elif not cache_file.is_file():
            raise FileNotFoundError(f"no pipeline cache at {cache_file}; run the batch with --cache write first")
        params["pipeline_cache"] = cache_file
    if "link" in phases:
        needed = {(t["task"], t["mesh"]) for t in triples}
        ids = [i for i in ids if (lists["pre"][i]["task"], lists["pre"][i]["mesh"]) in needed]
    body = job_lines(work, lists, tr, ids, triples, warm_item(lists), phases)
    (work / "jobs").mkdir(parents=True, exist_ok=True)
    load = quiet(no_quiet)
    ceiling_s = 3600.0
    attempts = []
    s: dict = {}
    for t in dict.fromkeys([threads, max(1, threads // 2), max(1, threads // 4), 1]):
        if t > threads:
            continue
        s = run_job(work, tag, header(t, "disabled", params=params), body, ceiling_s, 1 << 40,
                    f"{len(ids)} libraries on {t} threads", floor_mb=floor_mb)
        attempts.append({"threads": t, "status": s["status"]})
        if s["status"] != "killed_memory":
            break
    s["load_before"], s["attempts"] = load, attempts
    print(json.dumps({k: s.get(k) for k in ("tag", "status", "threads", "pre_items", "pre_wall_ms", "frag_wall_ms",
                                             "link_wall_ms", "failed", "process_s", "peak_working_set_mb",
                                             "pipeline_cache_out_bytes", "attempts")}))
    return 0 if s["status"] == "ok" and not s.get("failed") else 1


# ---- statistics ----------------------------------------------------------------------------------------

def stats(values: list[float]) -> dict:
    return pipe_cost.item_stats(values) if values else {"n": 0}


def ratio_stats(a: dict[int, float], b: dict[int, float]) -> dict:
    """a over b for the items both runs built: the median, the quartiles and the extremes of the per-item ratio."""
    rs = sorted(a[i] / b[i] for i in a if i in b and b[i] > 0)
    if not rs:
        return {"n": 0}
    q = lambda p: pipe_cost.percentile(rs, p)  # noqa: E731  (nearest rank, as stats)
    return {"n": len(rs), "p25": round(q(25), 3), "p50": round(q(50), 3), "p75": round(q(75), 3),
            "min": round(rs[0], 3), "max": round(rs[-1], 3), "sum_ratio": round(sum(a[i] for i in a if i in b) / sum(b[i] for i in a if i in b), 3)}


def job_pre_ids(work: Path, tag: str, lists: dict) -> list[int] | None:
    """The lists["pre"] index of each `pre` line of a run's job file: the harness numbers its rows by position among them, so a
    run that built only some of the libraries (smoke, coldcheck, the link run) needs this map. None when there is no job file."""
    path = work / "jobs" / f"{tag}.txt"
    if not path.is_file():
        return None
    keys: list[str] = []
    pre: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.split(None, 2)
        if w[:1] == ["module"] and len(w) == 3:
            keys.append(PureWindowsPath(w[2]).stem)  # job files are written on Windows; read them the same on any host
        elif w[:1] == ["pre"] and len(w) == 3:
            pre.append((keys[int(w[1])] if w[1] != "-" else "", keys[int(w[2])]))
    index = {(it["task"], it["mesh"]): i for i, it in enumerate(lists["pre"])}
    return [index[p] for p in pre]


def by_index(rows: list[dict], ids: list[int] | None = None) -> dict[int, float]:
    """Milliseconds by lists["pre"] index (ids maps a row's position to it; identity when the run built every item)."""
    return {(ids[int(r["index"])] if ids else int(r["index"])): float(r["ms"]) for r in rows}


def run_summary(work: Path, tag: str) -> dict | None:
    path = work / "results" / "summary.jsonl"
    if not path.is_file():
        return None
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("tag") == tag:
            last = d
    return last


def phase_ms(rows: list[dict], phase: str) -> list[dict]:
    return [r for r in rows if r.get("phase") == phase and r.get("result") == "0"]


def grouped(rows: list[dict], meta: dict[int, dict], field: str, ids: list[int] | None = None) -> dict:
    """Per-group statistics of the pre-rasterization rows of one run (scope, kind or raw access of the item)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        groups[str(meta[ids[int(r["index"])] if ids else int(r["index"])][field])].append(float(r["ms"]))
    return {k: stats(v) for k, v in sorted(groups.items())}


def report(work: Path, budget: tuple[float, float]) -> int:
    lists, tr = load_lists(work)
    meta = {int(r["index"]): r for r in pipe_cost.read_tsv(work / "pre_items.tsv")}
    out: dict = {"budget_minutes": list(budget)}
    sizes = {"task": [t["bytes"] for t in tr.values() if t["stage"] == "ts" and t["status"] == "ok"],
             "mesh": [t["bytes"] for t in tr.values() if t["stage"] == "ms" and t["status"] == "ok"],
             "pixel": [t["bytes"] for t in tr.values() if t["stage"] == "ps" and t["status"] == "ok"],
             "pre_item": [it["bytes"] for it in lists["pre"]]}
    out["spirv_bytes"] = {k: stats([float(v) for v in vs]) for k, vs in sizes.items()}
    out["items"] = {"pre": len(lists["pre"]), "by_scope": dict(Counter(it["scope"] for it in lists["pre"])),
                    "by_kind": dict(Counter(it["kind"] for it in lists["pre"])), "frag": len(lists["frag"]),
                    "triples": len(lists["triples"]), "excluded_pairs": lists["excluded_pairs"]}
    runs: dict = {}
    for p in sorted((work / "results").glob("*.tsv")):
        tag = p.stem
        if tag.startswith("screen"):
            continue
        rows = cc.rows_of(work, tag)
        pre, frag, link = phase_ms(rows, "pre"), phase_ms(rows, "frag"), phase_ms(rows, "link")
        if not (pre or frag or link):
            continue
        s = run_summary(work, tag) or {}
        ids = job_pre_ids(work, tag, lists)
        e: dict = {"threads": s.get("threads"), "pin32": s.get("pin32"), "failed": s.get("failed"),
                   "peak_working_set_mb": s.get("peak_working_set_mb"), "validation_errors": s.get("validation_errors"),
                   "status": s.get("status")}
        if pre:
            e["pre"] = stats([float(r["ms"]) for r in pre])
            e["pre_wall_s"] = round(s.get("pre_wall_ms", 0) / 1000.0, 1)
            e["pre_by_scope"], e["pre_by_kind"], e["pre_by_raw"] = (grouped(pre, meta, f, ids) for f in ("scope", "kind", "raw"))
            e["pre_cpu_over_wall"] = round(sum(float(r.get("cpu_ms", 0)) for r in pre) / max(1.0, sum(float(r["ms"]) for r in pre)), 3)
        # a killed or timed-out run holds only the first (largest) items, a refused library is a gap: neither is a measurement
        e["complete"] = bool(pre) and s.get("status") == "ok" and not s.get("failed") and len(pre) == s.get("pre_items", len(pre))
        if frag:
            e["frag"] = stats([float(r["ms"]) for r in frag])
            e["frag_wall_s"] = round(s.get("frag_wall_ms", 0) / 1000.0, 1)
        if link:
            e["link_ms"] = stats([float(r["ms"]) for r in link])
        if s.get("pipeline_cache_out_bytes") is not None:
            e["pipeline_cache_out_bytes"] = s["pipeline_cache_out_bytes"]
        if s.get("pipeline_cache_in_bytes"):
            e["pipeline_cache_in_bytes"] = s["pipeline_cache_in_bytes"]
        runs[tag] = e
    out["runs"] = runs
    screen_ms: list[float] = []
    scr = work / "results" / "screen.tsv"
    if scr.is_file():
        all_rows = pipe_cost.read_tsv(scr)
        rows = [r for r in all_rows if r["status"] == "ok"]
        screen_ms = [float(r["ms"]) for r in rows]
        out["screen"] = {"n": len(rows), "ms": stats(screen_ms),
                         "peak_ws_mb": stats([float(r["peak_ws_mb"]) for r in rows if r["peak_ws_mb"]]),
                         "refused": sum(r["status"] == "refused" for r in all_rows),
                         "killed": sum(r["status"] in ("timeout", "killed_memory") for r in all_rows)}
    if runs.get("t1-pin", {}).get("complete") and runs.get("t1-nopin", {}).get("complete"):
        out["pin_overhead_1thread"] = ratio_stats(
            by_index(phase_ms(cc.rows_of(work, "t1-pin"), "pre"), job_pre_ids(work, "t1-pin", lists)),
            by_index(phase_ms(cc.rows_of(work, "t1-nopin"), "pre"), job_pre_ids(work, "t1-nopin", lists)))
    for t in (6, 12):
        a, b = runs.get(f"t{t}-pin"), runs.get(f"t{t}-nopin")
        if a and b and a.get("complete") and b.get("complete") and a.get("pre_wall_s") and b.get("pre_wall_s"):
            out[f"pin_overhead_{t}threads_wall"] = round(a["pre_wall_s"] / b["pre_wall_s"], 3)
    if runs.get("t1-pin", {}).get("complete"):
        one = runs["t1-pin"]["pre"]["sum"] / 1000.0
        curve = {t: {"threads": r["threads"], "wall_min": round(r["pre_wall_s"] / 60.0, 2),
                     "throughput": round(one / r["pre_wall_s"], 2)}
                 for t, r in runs.items() if r.get("pin32") == 1 and r.get("complete") and r.get("pre_wall_s")
                 and t.startswith("t") and t.endswith("-pin")}
        out["thread_curve_pin_on"] = curve
        out["pre_libraries_minutes_vs_budget"] = {k: {"threads": v["threads"], "minutes": v["wall_min"],
                                                      "within_B": v["wall_min"] <= budget[1]}
                                                  for k, v in curve.items()}
    cold, warm = runs.get("t12-cachew") or runs.get("t6-cachew"), runs.get("t12-cacher") or runs.get("t6-cacher")
    if cold and warm and cold.get("complete") and warm.get("complete"):
        out["warm_from_cache"] = {"cold": cold["pre"], "warm": warm["pre"], "cache_bytes": cold.get("pipeline_cache_out_bytes"),
                                  "warm_wall_s": warm.get("pre_wall_s")}
    miss = {}
    if screen_ms:
        miss["alone"] = stats(screen_ms)
    loaded = runs.get("t12-pin") or runs.get("t6-pin")
    if loaded and loaded.get("complete"):
        miss["while_threads_build"] = dict(loaded["pre"], threads=loaded["threads"])
    if warm and warm.get("complete"):
        miss["warm_from_cache"] = warm["pre"]
    if runs.get("link", {}).get("link_ms"):
        miss["fast_link"] = runs["link"]["link_ms"]
    if runs.get("link", {}).get("frag"):
        miss["fragment_library"] = runs["link"]["frag"]
    out["one_miss_ms"] = miss
    validation = {}
    for tag in ("smoke", "smoke-counter-nopush", "smoke-gap"):
        s = run_summary(work, tag)
        if s:
            validation[tag] = {k: s.get(k) for k in ("status", "pre_items", "frag_items", "link_items", "failed",
                                                      "validation_errors", "validation_warnings")}
    if validation:
        out["validation"] = validation
    text = json.dumps(out, indent=1)
    (work / "report.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


# ---- main ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "translate", "prepare", "smoke", "coldcheck", "screen", "batch", "report"):
        p = sub.add_parser(name)
        p.add_argument("--work", type=Path, required=True)
        if name in ("select", "translate"):
            p.add_argument("--split", type=Path, required=True)
        if name == "translate":
            p.add_argument("--xlate", required=True)
            p.add_argument("--workers", type=int, default=3)
        if name == "prepare":
            p.add_argument("--triples", type=int, default=0)
        if name in ("smoke", "coldcheck", "screen"):
            p.add_argument("--count", type=int, default=20 if name != "screen" else 100)
            p.add_argument("--no-quiet-check", action="store_true")
        if name == "smoke":
            p.add_argument("--threads", type=int, default=6)
        if name == "screen":
            p.add_argument("--largest", type=int, default=10)
            p.add_argument("--timeout", type=float, default=900.0)
        if name == "batch":
            p.add_argument("--tag", required=True)
            p.add_argument("--threads", type=int, default=1)
            p.add_argument("--pin32", choices=("on", "off"), default="on")
            p.add_argument("--cache", choices=("none", "write", "read"), default="none")
            p.add_argument("--cache-name", default="m1")
            p.add_argument("--phases", default="pre")
            p.add_argument("--no-quiet-check", action="store_true")
            p.add_argument("--floor-mb", type=float, default=FLOOR_MB)
        if name == "report":
            p.add_argument("--budget-min", type=float, nargs=2, default=(12.0, 32.0))
    args = ap.parse_args(argv)
    work = args.work.resolve()
    if work == ROOT or ROOT in work.parents:
        print(f"mesh_cost: --work {args.work} lies inside the repository; results name the game's programs and must stay outside it",
              file=sys.stderr)
        return 2
    if args.cmd == "select":
        return select(args.split, args.work)
    if args.cmd == "translate":
        return translate(args.split, args.xlate, args.work, args.workers)
    if args.cmd == "prepare":
        return prepare(args.work, args.triples)
    if args.cmd == "smoke":
        return smoke(args.work, args.count, args.no_quiet_check, args.threads)
    if args.cmd == "coldcheck":
        return coldcheck(args.work, args.count, args.no_quiet_check)
    if args.cmd == "screen":
        return screen(args.work, args.count, args.largest, args.timeout, args.no_quiet_check)
    if args.cmd == "batch":
        return batch(args.work, args.tag, args.threads, args.pin32 == "on", args.cache, args.cache_name, args.phases,
                     args.no_quiet_check, args.floor_mb)
    return report(args.work, tuple(args.budget_min))


if __name__ == "__main__":
    sys.exit(main())
