#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Run spike S5's write-watch and host-import probe over its matrix, one process per measurement.

    py -3 tools/watchcost/watch_cost.py run --exe PATH --out DIR [--runs 3] [--modes watch,import,hold,behave]
                                            [--sizes 1024,2048,4096,5632] [--dirty 0,1000,10000,100000]
                                            [--scopes heap,region,alias,pools] [--iters 200] [--import-iters 100]
                                            [--wait 1800]
    py -3 tools/watchcost/watch_cost.py report --out DIR

The probe is tools/watchcost/vkwatchcost.c (py -3 tools/build_vkdraw.py --harness vkwatchcost). Before every
process the machine is checked: available RAM, CPU busy over one second, and the heavy processes that would skew a
timing (compilers, Ninja, the other Vulkan harnesses, the game, an emulator, a Python shader pool). A run waits while
a heavy process is present (up to --wait seconds, then the cell is skipped and the reason recorded), and is skipped
when available RAM is under what the cell commits plus 1.5 GiB. Every process's state, exit code and parsed output go
to DIR/results.jsonl, one line each; `report` turns that file into Markdown tables (DIR/summary.md). DIR should lie
outside the repository. Windows only: the probe needs the system's write watch.

Exit codes: 0 done, 1 the probe is missing or a run failed, 2 usage.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MIB = 1 << 20
MARGIN_MIB = 1536          # available RAM a cell needs beyond what it commits
FRAME_MS = 1000.0 / 60.0   # architecture section 6.2
SUBMITS_PER_FRAME = 6      # the NVN2 call census: the game submits six times a frame in the Forest

HEAVY = ("clang", "clang-cl", "ld.lld", "ninja", "vkpipecost", "vkcomputecost", "vkhmmacost", "vkdraw", "oot2",
         "oot2-xlate", "eden", "eden-cli", "lichii", "yuzu", "ryujinx", "cl", "link", "msbuild")
POOL_SCRIPTS = ("tools/shaders/pool.py", "tools/shaders/corpus.py", "tools/shaders/validate.py",
                "tools\\shaders\\pool.py", "tools\\shaders\\corpus.py", "tools\\shaders\\validate.py")


# ---- pure helpers (unit-tested) ------------------------------------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest sample with at least p percent of the samples at or below it."""
    if not values:
        raise ValueError("no samples")
    s = sorted(values)
    k = max(1, -(-len(s) * p // 100))   # ceil(n * p / 100), at least 1
    return s[int(k) - 1]


def heavy_from_tasklist(text: str) -> list[str]:
    """Names of heavy processes in `tasklist /FO CSV /NH` output (one image name per row, case-insensitive)."""
    found = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        name = row[0].strip().lower()
        stem = name[:-4] if name.endswith(".exe") else name
        if stem in HEAVY and name not in found:
            found.append(name)
    return found


def pool_pythons(cmdlines: list[str]) -> list[str]:
    """Python command lines that run a shader corpus pool."""
    return [c for c in cmdlines if "python" in c.lower() and any(s in c for s in POOL_SCRIPTS)]


FIELD = re.compile(r'(\w+)=("[^"]*"|\S+)')


def fields(line: str) -> dict[str, str]:
    return {k: v.strip('"') for k, v in FIELD.findall(line)}


def parse_probe(text: str) -> dict:
    """The probe's stdout as data: cell, result, samples by kind, memory stages, verdicts and B-lines."""
    out: dict = {"samples": {}, "memory": [], "verdicts": {}, "facts": {}, "lines": []}
    for line in text.splitlines():
        head, _, rest = line.partition(" ")
        if head == "samples":
            f = fields(rest)
            vals = f.get("values", "")
            out["samples"][f["kind"]] = [float(v) for v in vals.split(",") if v]
        elif head in ("cell", "result", "setup", "device", "pools", "reset", "imported", "freed", "invalid"):
            out.setdefault(head, {}).update(fields(rest))
        elif head == "memory":
            out["memory"].append(fields(rest))
        elif head == "verdict":
            m = re.match(r"(B\d+) (PASS|FAIL): (.*)", rest)
            if m:
                out["verdicts"][m.group(1)] = {"pass": m.group(2) == "PASS", "what": m.group(3)}
        elif re.fullmatch(r"B\d+", head):
            out["facts"].setdefault(head, {}).update(fields(rest))
        elif head == "mismatch":
            out["lines"].append(line)
    return out


def import_key(argv: list[str]) -> tuple[int, int, str]:
    """(committed MiB, window MiB, place) of an import job: `import COMMIT ITERS [WINDOW [PLACE]]`."""
    return int(argv[1]), int(argv[3]) if len(argv) > 3 else 64, argv[4] if len(argv) > 4 else "cycle"


def summary(values: list[float]) -> dict[str, float]:
    return {"n": len(values), "p50": percentile(values, 50), "p99": percentile(values, 99), "max": max(values),
            "mean": sum(values) / len(values)}


# ---- the machine ---------------------------------------------------------------------------------------------------

class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def avail_mib() -> float:
    m = _MemStatus()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / MIB


def cpu_busy_percent(seconds: float = 1.0) -> float:
    """System-wide CPU busy share over the interval (GetSystemTimes: kernel time includes idle)."""
    def times() -> tuple[int, int, int]:
        idle, kern, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
        return idle.value, kern.value, user.value
    i0, k0, u0 = times()
    time.sleep(seconds)
    i1, k1, u1 = times()
    total = (k1 - k0) + (u1 - u0)
    return 100.0 * (total - (i1 - i0)) / total if total else 0.0


def machine_state() -> dict:
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    tl = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, env=env).stdout
    ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                         "Get-CimInstance Win32_Process -Filter \"Name like 'python%' or Name like 'py.exe'\" | "
                         "ForEach-Object { $_.CommandLine }"], capture_output=True, text=True, env=env).stdout
    heavy = heavy_from_tasklist(tl) + [f"python pool: {c[:120]}" for c in pool_pythons(ps.splitlines())]
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "avail_mib": round(avail_mib()),
            "cpu_busy_pct": round(cpu_busy_percent(), 1), "heavy": heavy}


# ---- the matrix ----------------------------------------------------------------------------------------------------

def jobs(args: argparse.Namespace) -> list[dict]:
    out = []
    modes = args.modes.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    if "behave" in modes:
        out.append({"label": "behave", "argv": ["behave"], "commit_mib": 256})
    if "watch" in modes:
        for size in sizes:
            for scope in args.scopes.split(","):
                for dirty in (int(d) for d in args.dirty.split(",")):
                    out.append({"label": f"watch {size} {scope} {dirty}", "commit_mib": size,
                                "argv": ["watch", str(size), scope, str(dirty), str(args.iters), str(size + dirty)]})
    if "import" in modes:
        # the window at the heap's start (the whole committed heap after it), at its end, and at its start with the
        # run of committed pages ended right after it; then bounded windows of 2 and 16 MiB at one size
        for size in sizes:
            for place in ("first", "last", "bounded"):
                out.append({"label": f"import {size} 64 {place}", "commit_mib": size,
                            "argv": ["import", str(size), str(args.import_iters), "64", place]})
        for win in (2, 16):
            out.append({"label": f"import 4096 {win} bounded", "commit_mib": 4096,
                        "argv": ["import", "4096", str(args.import_iters), str(win), "bounded"]})
    if "hold" in modes:
        for imp in (1024, 2048, 4096):
            out.append({"label": f"hold {imp} touch", "commit_mib": imp + 512,
                        "argv": ["hold", str(imp + 512), str(imp), "touch"]})
        out.append({"label": "hold 1024 notouch", "commit_mib": 1536, "argv": ["hold", "1536", "1024", "notouch"]})
    return out


def run(args: argparse.Namespace) -> int:
    exe = Path(args.exe)
    if not exe.is_file():
        print(f"watch_cost: no probe at {exe}; run tools/build_vkdraw.py --harness vkwatchcost", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, VK_LOADER_LAYERS_DISABLE="~implicit~")
    todo = jobs(args)
    failed = 0
    with (out / "results.jsonl").open("a", encoding="utf-8") as log:
        for r in range(1, args.runs + 1):
            for job in todo:
                state = machine_state()
                waited = 0
                while state["heavy"] and waited < args.wait:
                    print(f"watch_cost: waiting, heavy processes: {', '.join(state['heavy'])}", flush=True)
                    time.sleep(30)
                    waited += 30
                    state = machine_state()
                rec = {"run": r, "job": job["label"], "argv": job["argv"], "state": state}
                need = job["commit_mib"] + MARGIN_MIB
                if state["heavy"]:
                    rec["skipped"] = f"heavy processes after {waited} s: {', '.join(state['heavy'])}"
                elif state["avail_mib"] < need:
                    rec["skipped"] = f"available RAM {state['avail_mib']} MiB under the {need} MiB the cell needs"
                else:
                    t0 = time.time()
                    done = subprocess.run([str(exe), *job["argv"]], capture_output=True, text=True, env=env,
                                          timeout=1800)
                    rec.update(exit=done.returncode, seconds=round(time.time() - t0, 1),
                               parsed=parse_probe(done.stdout), stderr_tail=done.stderr[-2000:])
                    failed += done.returncode != 0
                print(f"watch_cost: run {r} {job['label']}: "
                      f"{rec.get('skipped') or 'exit %s in %s s' % (rec['exit'], rec['seconds'])} "
                      f"(avail {state['avail_mib']} MiB, CPU {state['cpu_busy_pct']} %)", flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
    return 1 if failed else 0


# ---- the report ----------------------------------------------------------------------------------------------------

def load(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]


def report(out: Path) -> str:
    recs = [r for r in load(out) if "parsed" in r and r.get("exit") in (0, 1)]
    lines = ["# Spike S5 results", ""]
    states = [r["state"] for r in load(out)]
    if states:
        lines.append(f"Processes: {len(states)}; available RAM at start {min(s['avail_mib'] for s in states)} to "
                     f"{max(s['avail_mib'] for s in states)} MiB; CPU busy {min(s['cpu_busy_pct'] for s in states)} to "
                     f"{max(s['cpu_busy_pct'] for s in states)} %; skipped: "
                     f"{sum('skipped' in r for r in load(out))}.")
        lines.append("")
    # write watch: per (size, dirty) a row, per scope the pooled query p50/p99 in ms
    cells: dict[tuple, dict[str, list]] = {}
    for r in recs:
        if r["argv"][0] != "watch":
            continue
        _, size, scope, dirty = r["argv"][:4]
        c = cells.setdefault((int(size), int(dirty), scope), {"query": [], "write": [], "run_p50": [], "ok": []})
        c["query"] += r["parsed"]["samples"].get("query", [])
        c["write"] += r["parsed"]["samples"].get("write", [])
        c["run_p50"].append(percentile(r["parsed"]["samples"]["query"], 50) / 1000)
        c["ok"].append(r["parsed"].get("result", {}).get("count_ok") == "1")
    if cells:
        scopes = sorted({k[2] for k in cells}, key=["heap", "region", "alias", "pools"].index)
        lines += ["## Write watch: one query with reset, milliseconds p50 / p99 (pooled runs)", "",
                  "| Committed MiB | Pages written | " + " | ".join(scopes) + " | runs' p50 spread (heap) |",
                  "|---|---|" + "---|" * len(scopes) + "---|"]
        for size, dirty in sorted({(k[0], k[1]) for k in cells}):
            row = [str(size), str(dirty)]
            for sc in scopes:
                c = cells.get((size, dirty, sc))
                row.append("-" if not c or not c["query"] else
                           f"{percentile(c['query'], 50) / 1000:.2f} / {percentile(c['query'], 99) / 1000:.2f}"
                           + ("" if all(c["ok"]) else " (count wrong)"))
            h = cells.get((size, dirty, "heap"))
            row.append("-" if not h else " to ".join(f"{v:.2f}" for v in (min(h["run_p50"]), max(h["run_p50"]))))
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "## Writes after a reset: microseconds per page written, p50 (heap scope, pooled)", "",
                  "| Committed MiB | 1,000 pages | 10,000 pages | 100,000 pages |", "|---|---|---|---|"]
        for size in sorted({k[0] for k in cells}):
            row = [str(size)]
            for dirty in (1000, 10000, 100000):
                c = cells.get((size, dirty, "heap"))
                row.append("-" if not c or not c["write"] else f"{percentile(c['write'], 50) / dirty:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", f"## Per frame at p50: one query per submit ({SUBMITS_PER_FRAME}) or one per frame, ms of a "
                  f"{FRAME_MS:.2f} ms frame", "", "| Committed MiB | Pages written | Scope | per submit | per frame |",
                  "|---|---|---|---|---|"]
        for (size, dirty, sc), c in sorted(cells.items()):
            if sc in ("heap", "pools") and c["query"]:
                p = percentile(c["query"], 50) / 1000
                lines.append(f"| {size} | {dirty} | {sc} | {p * SUBMITS_PER_FRAME:.1f} | {p:.2f} |")
        lines.append("")
    imp = [r for r in recs if r["argv"][0] == "import"]
    if imp:
        lines += ["## Import of a window: microseconds p50 / p99 (pooled runs)", "",
                  "| Committed MiB | Window MiB | Place | host properties | buffer | import | bind and address | "
                  "destroy and free | total |", "|---|---|---|---|---|---|---|---|---|"]
        for size, win, place in sorted({import_key(r["argv"]) for r in imp}):
            row = [str(size), str(win), place]
            mine = [r for r in imp if import_key(r["argv"]) == (size, win, place)]
            for kind in ("import_host_properties", "import_buffer", "import_allocate", "import_bind_address",
                         "import_destroy_free", "import_total"):
                v = [x for r in mine for x in r["parsed"]["samples"].get(kind, [])]
                row.append(f"{percentile(v, 50):.0f} / {percentile(v, 99):.0f}" if v else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    hold = [r for r in recs if r["argv"][0] == "hold"]
    if hold:
        lines += ["## Holding imports: memory by stage (MiB)", "",
                  "| Run | Import | Stage | available | private | working set | imported resident | other resident |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in hold:
            for m in r["parsed"]["memory"]:
                lines.append(f"| {r['run']} | {r['argv'][2]} {r['argv'][3]} | {m['stage']} | {m['avail_mib']} | "
                             f"{m['private_mib']} | {m['ws_mib']} | {m['imported_resident_mib']} | "
                             f"{m['other_resident_mib']} |")
        lines.append("")
    beh = [r for r in recs if r["argv"][0] == "behave"]
    if beh:
        lines += ["## Behaviours", ""]
        for r in beh:
            v = r["parsed"]["verdicts"]
            facts = "; ".join(f"{k} " + " ".join(f"{a}={b}" for a, b in f.items())
                              for k, f in sorted(r["parsed"]["facts"].items()))
            lines.append(f"- run {r['run']}: " + ", ".join(f"{k} {'PASS' if x['pass'] else 'FAIL'}"
                                                          for k, x in sorted(v.items())) + f". {facts}")
        lines.append("")
    text = "\n".join(lines)
    (out / "summary.md").write_text(text, encoding="utf-8", newline="\n")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--exe", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--runs", type=int, default=3)
    r.add_argument("--modes", default="behave,watch,import,hold")
    r.add_argument("--sizes", default="1024,2048,4096,5632")
    r.add_argument("--dirty", default="0,1000,10000,100000")
    r.add_argument("--scopes", default="heap,region,alias,pools")
    r.add_argument("--iters", type=int, default=200)
    r.add_argument("--import-iters", type=int, default=100)
    r.add_argument("--wait", type=int, default=1800)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        if os.name != "nt":
            print("watch_cost: Windows only (the probe needs the system's write watch)", file=sys.stderr)
            return 2
        return run(args)
    print(report(Path(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
