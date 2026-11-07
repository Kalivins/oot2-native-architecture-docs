#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Run the windowed present spike over its paths and read what each instrument saw.

    py -3 tools/presentcost/present_cost.py run --exe PATH --out DIR [--frames 10000]
                                                [--paths composed-fifo,borderless-mailbox,...] [--wait 600]
    py -3 tools/presentcost/present_cost.py analyse --prefix DIR/NAME
    py -3 tools/presentcost/present_cost.py report --out DIR

The harness is tools/presentcost/vkpresent.c (py -3 tools/build_vkdraw.py --harness vkpresent). A path is
`<window>-<mode>[-option...]`: window `composed` or `borderless`, mode `fifo`, `relaxed`, `mailbox`, `latest` or
`immediate`, options `rel<us>` (relative-time present timing, the previous image held at least that long), `nearest`
(with rel: the nearest refresh cycle), `notiming` (a swapchain without present timing, present wait only), `layers`
(third-party implicit Vulkan layers left on; every other path runs with VK_LOADER_LAYERS_DISABLE=~implicit~),
`skip<K>` (the producer skips every Kth tick: the counter-run), `img<N>` (swapchain images, default 3), `nolock`
(one unbounded present wait per frame outside the swapchain lock, for the wait's own wake-up; outside the
external-synchronisation rule, measurement only) and `occluded` (a small topmost window over the swapchain window's
corner, so the desktop composes the frame).

Before every run the machine is checked as tools/watchcost does (available RAM, CPU busy over one second, the heavy
processes that would skew a timing); a run waits while a heavy process is present (up to --wait seconds, then it is
skipped and the reason recorded). Each run's stdout, frame rows and vertical blanks go to DIR/<path>.*; its machine
state and analysis go to DIR/results.jsonl, one line each; `report` writes DIR/summary.md. DIR should lie outside the
repository. Windows only: the harness opens a window on the primary monitor.

What `analyse` computes, per instrument (first-pixel-out and queue-operations-end from present timing, converted to
the performance counter through the harness's calibration pairs, and the return of vkWaitForPresent2KHR), over the
ticks after the first WARMUP: the intervals between consecutive frames on screen (p50, p99, the share at each whole
number of refresh periods), the latency from the present call and from the tick's due time to the screen, and the
virtual-clock view: the screen sampled once per tick at due + Lmin + h (Lmin the 1st percentile of due-to-screen, h
the smaller of half way between the refresh period and the tick, and three quarters of a tick), where an on-time
frame is always on screen; a tick whose sample shows the previous sample's frame is a repeat ("shown twice"), a tick
that no sample shows is "never shown". Frames the engine reported as never shown (a first-pixel-out time of 0), frames
the host dropped and ticks the producer skipped are counted apart. The vertical-blank thread gives the refresh grid
the first-pixel-out times are checked against, and the present thread's cycles give its CPU time per frame.

A run the harness's watchdog ends (exit 4, nothing moved for 5 s) is analysed on the rows it wrote and counted as a
failure. Exit codes: 0 done, 1 the harness is missing or a run failed, 2 usage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from bisect import bisect_right
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "watchcost"))

import watch_cost  # noqa: E402  (machine checks and the nearest-rank percentile are shared with spike S5)

TICK_HZ = 60
WARMUP = 60                 # ticks left out of every statistic (swapchain and compositor start-up)
STAGE_QOE, STAGE_FPO = 1, 4
ST_SKIPPED, ST_PRODUCED, ST_HOST_DROPPED, ST_PRESENTED = 1, 2, 3, 4
MODES = ("fifo", "relaxed", "mailbox", "latest", "immediate")
DEFAULT_PATHS = ("composed-fifo", "composed-relaxed", "composed-mailbox", "composed-latest", "composed-fifo-rel16667",
                 "borderless-fifo", "borderless-relaxed", "borderless-mailbox", "borderless-latest",
                 "borderless-fifo-rel16667")
percentile = watch_cost.percentile


# ---- paths ---------------------------------------------------------------------------------------------------------

def path_args(name: str) -> tuple[list[str], bool]:
    """The harness arguments (after `run`, before `--out`) of a path name, and whether implicit layers stay on."""
    parts = name.split("-")
    if len(parts) < 2 or parts[0] not in ("composed", "borderless") or parts[1] not in MODES:
        raise ValueError(f"path {name!r}: expected <composed|borderless>-<{'|'.join(MODES)}>[-option...]")
    argv = ["--window", parts[0], "--mode", parts[1]]
    layers = timing_off = nearest = False
    rel = None
    for opt in parts[2:]:
        if m := re.fullmatch(r"rel(\d+)", opt):
            rel = int(m.group(1))
            argv += ["--relative", str(rel * 1000)]
        elif opt == "nearest":
            nearest = True
            argv.append("--nearest")
        elif opt == "notiming":
            timing_off = True
            argv += ["--timing", "off"]
        elif opt == "layers":
            layers = True
        elif m := re.fullmatch(r"skip(\d+)", opt):
            if int(m.group(1)) < 2:
                raise ValueError(f"path {name!r}: skip needs 2 or more")
            argv += ["--skip", m.group(1)]
        elif m := re.fullmatch(r"img(\d+)", opt):
            argv += ["--images", m.group(1)]
        elif opt == "nolock":
            argv += ["--wait-lock", "off"]
        elif opt == "occluded":
            argv.append("--occlude")
        else:
            raise ValueError(f"path {name!r}: unknown option {opt!r}")
    if (rel is not None or nearest) and timing_off:
        raise ValueError(f"path {name!r}: rel and nearest need present timing")
    if nearest and rel is None:
        raise ValueError(f"path {name!r}: nearest needs rel")
    return argv, layers


# ---- reading the harness's output ----------------------------------------------------------------------------------

def parse_stdout(text: str) -> dict:
    """Key=value lines by their first word; repeated words (layer_dll, time_domain, calib, timing_props) as lists."""
    out: dict = {"calib": [], "timing_props": [], "layer_dll": [], "layer_listed": [], "time_domain": [], "vrr": [],
                 "other": []}
    for line in text.splitlines():
        head, _, rest = line.partition(" ")
        f = watch_cost.fields(rest)
        if head in ("calib", "timing_props", "layer_dll", "layer_listed", "time_domain", "vrr"):
            out[head].append(f)
        elif head in ("start", "tsc", "window", "display", "dwm", "env", "layers", "device", "features", "surface",
                      "swapchain", "time_domain_chosen", "clock", "result", "vblank", "stall"):
            out.setdefault(head, {}).update(f)
        elif line.strip():
            out["other"].append(line)
    return out


def read_frames(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            vals = line.rstrip("\n").split("\t")
            if len(vals) == len(header):
                rows.append({k: int(v) for k, v in zip(header, vals)})
    return rows


def read_vblanks(path: Path) -> list[int]:
    with path.open(encoding="utf-8") as f:
        f.readline()
        return [int(line) for line in f if line.strip()]


def converters(calib: list[dict], qpf: int) -> dict[tuple[int, int], tuple[float, float, float]]:
    """(domain index, stage) -> (value0, qpc0, qpc counts per value unit), from the calibration pairs.

    Domain indices follow the order in which (kind, id) first appears, as the harness numbers them. With two pairs
    the slope is measured; with one, the value is taken as nanoseconds."""
    order: list[tuple[str, str]] = []
    pairs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for c in calib:
        if c.get("result") != "0":
            continue
        key = (c["kind"], c["id"])
        if key not in order:
            order.append(key)
        pairs.setdefault((order.index(key), int(c["stage"])), []).append((int(c["value"]), int(c["qpc"])))
    out = {}
    for k, p in pairs.items():
        v0, q0 = p[0]
        if len(p) > 1 and p[-1][0] != v0:
            slope = (p[-1][1] - q0) / (p[-1][0] - v0)
        else:
            slope = qpf / 1e9
        out[k] = (float(v0), float(q0), slope)
    # a stage-local domain calibrated for one stage only: the other stage shares it (swapchain-local)
    for (d, s) in list(out):
        other = STAGE_FPO if s == STAGE_QOE else STAGE_QOE
        out.setdefault((d, other), out[(d, s)])
    return out


def to_qpc(conv: dict, domain: int, stage: int, value: int) -> float | None:
    c = conv.get((domain, stage))
    if c is None or value == 0:
        return None
    v0, q0, slope = c
    return q0 + (value - v0) * slope


# ---- the analysis --------------------------------------------------------------------------------------------------

def summary_ms(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "p50": round(percentile(values, 50), 3), "p99": round(percentile(values, 99), 3),
            "max": round(max(values), 3), "min": round(min(values), 3)}


def interval_shares(intervals_ms: list[float], refresh_ms: float) -> dict[str, float]:
    """Share of intervals at each whole number of refresh periods (rounded), as percentages, largest first."""
    counts: dict[str, int] = {}
    for v in intervals_ms:
        k = max(0, round(v / refresh_ms)) if refresh_ms > 0 else 0
        key = f"{k * refresh_ms:.1f}"
        counts[key] = counts.get(key, 0) + 1
    n = len(intervals_ms) or 1
    return {k: round(100.0 * c / n, 3) for k, c in sorted(counts.items(), key=lambda kv: -kv[1])}


def tick_view(due: dict[int, float], shown: dict[int, float], ticks: list[int], refresh_ms: float,
              qpf: int) -> dict:
    """The screen sampled once per tick (see the module text): repeats, never-shown ticks, late frames, Lmin."""
    tick_ms = 1000.0 / TICK_HZ
    lat = [(shown[k] - due[k]) * 1000.0 / qpf for k in ticks if k in shown]
    if not lat:
        return {"n": 0}
    lmin = percentile(lat, 1)
    h = min((refresh_ms + tick_ms) / 2, 0.75 * tick_ms)
    order = sorted((t, k) for k, t in shown.items())
    times = [t for t, _ in order]
    samples = []
    for k in ticks:
        at = due[k] + (lmin + h) * qpf / 1000.0
        i = bisect_right(times, at) - 1
        samples.append(order[i][1] if i >= 0 else -1)
    repeats = sum(1 for a, b in zip(samples, samples[1:]) if b == a)
    seen = set(samples)
    never = sum(1 for k in ticks[1:] if k not in seen)
    late = sum(1 for k in ticks if k in shown and (shown[k] - due[k]) * 1000.0 / qpf > lmin + h)
    backwards = sum(1 for a, b in zip(samples, samples[1:]) if b < a)
    return {"n": len(ticks), "lmin_ms": round(lmin, 3), "h_ms": round(h, 3), "repeats": repeats,
            "never_shown": never, "late": late, "backwards": backwards}


def nearest_offsets(times: list[float], grid: list[int], qpf: int) -> list[float]:
    """Signed distance in ms from each time to the nearest grid point."""
    out = []
    for t in times:
        i = bisect_right(grid, t)
        cands = [grid[j] for j in (i - 1, i) if 0 <= j < len(grid)]
        if cands:
            g = min(cands, key=lambda x: abs(x - t))
            out.append((t - g) * 1000.0 / qpf)
    return out


def analyse(stdout: str, frames: list[dict], vblanks: list[int]) -> dict:
    meta = parse_stdout(stdout)
    qpf = int(meta["start"]["qpf"])
    tsc = float(meta["tsc"]["hz"])
    conv = converters(meta["calib"], qpf)
    vb_int = [(b - a) * 1000.0 / qpf for a, b in zip(vblanks, vblanks[1:])]
    refresh_ms = percentile(vb_int, 50) if vb_int else 0.0
    props = meta["timing_props"]
    if props:
        dur = int(props[-1].get("refresh_duration_ns", "0"))
        if not refresh_ms and dur:
            refresh_ms = dur / 1e6
    window = [r for r in frames if r["tick"] >= WARMUP]
    ticks = [r["tick"] for r in window]
    due = {r["tick"]: float(r["due"]) for r in window}
    presented = [r for r in window if r["state"] == ST_PRESENTED]
    res: dict = {
        "ticks": len(window),
        "skipped": sum(1 for r in window if r["state"] == ST_SKIPPED),
        "host_dropped": sum(1 for r in window if r["state"] == ST_HOST_DROPPED),
        "presented": len(presented),
        "refresh_ms": round(refresh_ms, 4),
        "vblank_ms": summary_ms(vb_int),
        "timing_props": props,
        "vrr": meta["vrr"],
        "layers_loaded": [d.get("path", "") for d in meta["layer_dll"]],
        "env_layers_disable": meta.get("env", {}).get("layers_disable", ""),
        "swapchain": meta.get("swapchain", {}),
        "window": meta.get("window", {}),
        "result": meta.get("result", {}),
        "stall": meta.get("stall", {}),
    }
    cyc = [r["cycles"] / tsc * 1e6 for r in presented if r["cycles"]]
    res["present_thread_us"] = summary_ms(cyc)
    if cyc:
        res["present_thread_us"]["mean"] = round(sum(cyc) / len(cyc), 3)
    res["present_call_ms"] = summary_ms([(r["present_ret"] - r["present_call"]) * 1000.0 / qpf for r in presented])
    res["lock_wait_ms"] = summary_ms([r["lock_wait"] * 1000.0 / qpf for r in presented])
    res["wake_ms"] = summary_ms([(r["wake"] - r["produced"]) * 1000.0 / qpf for r in presented])

    instruments: dict[str, dict[int, float]] = {"fpo": {}, "qoe": {}, "wait": {}}
    engine_dropped = 0
    for r in presented:
        if r["pt_have"] & 2:
            t = to_qpc(conv, r["pt_domain"], STAGE_FPO, r["pt_fpo"])
            if t is None:
                engine_dropped += 1
            else:
                instruments["fpo"][r["tick"]] = t
        if r["pt_have"] & 1:
            t = to_qpc(conv, r["pt_domain"], STAGE_QOE, r["pt_qoe"])
            if t is not None:
                instruments["qoe"][r["tick"]] = t
        if r["wait_res"] == 0 and r["wait_ret"]:
            instruments["wait"][r["tick"]] = float(r["wait_ret"])
    res["engine_dropped"] = engine_dropped
    res["timing_reported"] = sum(1 for r in presented if r["pt_have"])
    calls = {r["tick"]: float(r["present_call"]) for r in presented}
    for name, shown in instruments.items():
        if not shown:
            continue
        order = sorted(shown.values())
        iv = [(b - a) * 1000.0 / qpf for a, b in zip(order, order[1:])]
        block = {"frames": len(shown), "interval_ms": summary_ms(iv),
                 "interval_shares": interval_shares(iv, refresh_ms) if name != "qoe" else {},
                 "from_present_ms": summary_ms([(t - calls[k]) * 1000.0 / qpf for k, t in shown.items()]),
                 "from_due_ms": summary_ms([(t - due[k]) * 1000.0 / qpf for k, t in shown.items()])}
        if name != "qoe":
            block["ticks"] = tick_view(due, shown, ticks, refresh_ms, qpf)
        if vblanks and name == "fpo":
            block["to_vblank_ms"] = summary_ms(nearest_offsets(list(shown.values()), vblanks, qpf))
        res[name] = block
    if instruments["fpo"] and instruments["wait"]:
        common = [k for k in instruments["wait"] if k in instruments["fpo"]]
        res["wait_after_fpo_ms"] = summary_ms([(instruments["wait"][k] - instruments["fpo"][k]) * 1000.0 / qpf
                                              for k in common])
    return res


# ---- running -------------------------------------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    exe = Path(args.exe)
    if not exe.is_file():
        print(f"present_cost: no harness at {exe}; run tools/build_vkdraw.py --harness vkpresent", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        paths = [(p, *path_args(p)) for p in args.paths.split(",")]
    except ValueError as e:
        print(f"present_cost: {e}", file=sys.stderr)
        return 2
    failed = 0
    with (out / "results.jsonl").open("a", encoding="utf-8") as log:
        for name, argv, layers in paths:
            waited = 0.0
            state = watch_cost.machine_state()
            while state["heavy"] and waited < args.wait:
                time.sleep(15)
                waited += 15
                state = watch_cost.machine_state()
            entry: dict = {"path": name, "frames": args.frames, "machine": state, "waited_s": waited}
            if state["heavy"]:
                entry["skipped"] = "heavy processes: " + ", ".join(state["heavy"])
            elif state["avail_mib"] < 2048:
                entry["skipped"] = f"available RAM {state['avail_mib']} MiB under 2 GiB"
            if "skipped" in entry:
                print(f"present_cost: {name}: skipped, {entry['skipped']}", flush=True)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                continue
            env = dict(os.environ)
            if layers:
                env.pop("VK_LOADER_LAYERS_DISABLE", None)
            else:
                env["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"
            prefix = out / name
            cmd = [str(exe), "run", *argv, "--frames", str(args.frames), "--out", str(prefix)]
            print(f"present_cost: {name}: {' '.join(cmd[1:])} (RAM {state['avail_mib']} MiB, CPU "
                  f"{state['cpu_busy_pct']} %)", flush=True)
            t0 = time.time()
            done = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                  timeout=args.frames / TICK_HZ + 120)
            (out / f"{name}.stdout.txt").write_text(done.stdout + done.stderr, encoding="utf-8")
            entry.update({"exit": done.returncode, "wall_s": round(time.time() - t0, 1),
                          "machine_after": watch_cost.machine_state()})
            if done.returncode in (0, 4):   # 4: the harness's watchdog ended a run that stopped; its rows are kept
                entry["analysis"] = analyse(done.stdout, read_frames(Path(f"{prefix}.frames.tsv")),
                                            read_vblanks(Path(f"{prefix}.vblanks.tsv")))
            if done.returncode != 0:
                failed += 1
                entry["stderr"] = done.stderr[-2000:]
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(f"present_cost: {name}: exit {done.returncode}, {entry['wall_s']} s", flush=True)
    return 1 if failed else 0


# ---- the report ----------------------------------------------------------------------------------------------------

def fmt(block: dict, key: str) -> str:
    return "-" if not block or block.get("n", 0) == 0 else f"{block[key]:.2f}"


def report_lines(entries: list[dict]) -> list[str]:
    lines = ["| Path | Frames on screen | Interval p50 / p99 ms | Share at each interval (ms: %) | Repeats | "
             "Never shown | Engine-dropped / host-dropped | Present to screen p50 / p99 ms | Due to screen p50 / p99 ms "
             "| Present thread us p50 / p99 | CPU busy % | Avail MiB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        a = e.get("analysis")
        if not a:
            lines.append(f"| {e['path']} | {e.get('skipped') or 'exit ' + str(e.get('exit'))} |" + " |" * 10)
            continue
        f = a.get("fpo") or a.get("wait") or {}
        inst = "fpo" if a.get("fpo") else "wait"
        tv = f.get("ticks", {})
        shares = ", ".join(f"{k}: {v}" for k, v in list(f.get("interval_shares", {}).items())[:4])
        pt = a["present_thread_us"]
        lines.append(
            f"| {e['path']} ({inst}) | {f.get('frames', 0)} | {fmt(f.get('interval_ms', {}), 'p50')} / "
            f"{fmt(f.get('interval_ms', {}), 'p99')} | {shares} | {tv.get('repeats', '-')} | "
            f"{tv.get('never_shown', '-')} | {a['engine_dropped']} / {a['host_dropped']} | "
            f"{fmt(f.get('from_present_ms', {}), 'p50')} / {fmt(f.get('from_present_ms', {}), 'p99')} | "
            f"{fmt(f.get('from_due_ms', {}), 'p50')} / {fmt(f.get('from_due_ms', {}), 'p99')} | "
            f"{fmt(pt, 'p50')} / {fmt(pt, 'p99')} | {e['machine']['cpu_busy_pct']} | {e['machine']['avail_mib']} |")
    return lines


def report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    entries = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    lines = ["# Windowed present spike", "", *report_lines(entries), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--exe", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--frames", type=int, default=10000)
    r.add_argument("--paths", default=",".join(DEFAULT_PATHS))
    r.add_argument("--wait", type=float, default=600)
    a = sub.add_parser("analyse")
    a.add_argument("--prefix", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "analyse":
        prefix = args.prefix
        res = analyse(Path(f"{prefix}.stdout.txt").read_text(encoding="utf-8"),
                      read_frames(Path(f"{prefix}.frames.tsv")), read_vblanks(Path(f"{prefix}.vblanks.tsv")))
        print(json.dumps(res, indent=1))
        return 0
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
