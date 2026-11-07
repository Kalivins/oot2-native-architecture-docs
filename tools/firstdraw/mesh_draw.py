#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The mesh bindings work: draw a synthetic task and mesh pair on the GPU, direct and indirect, and check every pixel.

    py -3 tools/firstdraw/mesh_draw.py --work DIR [--xlate oot2-xlate] [--vkmesh build/vkmesh/vkmesh.exe]
        [--validation on|sync|gpu|off] [--scenario NAME] [--counter NAME]

The programs are assembled from instruction words (mesh_programs.py: no game code, no global memory access), translated by the
real translator into TaskEXT, MeshEXT and fragment modules with their resource descriptions (schema 2), and drawn by vkmesh in the
descriptor layout `uni` with everything those descriptions name and nothing else: the constant banks at their stage's places (the
mesh stage's at set 1, the task stage's at set 2), the first task in the push constants as a device address, pipelines built from
libraries with required subgroup size 32 and full subgroups. Every scenario draws twice, once as the direct form and once as the
indirect one (a buffer the harness writes in the conversion pass's output form), from the same module bytes.

The image is compared with a CPU reference that knows only what the programs mean (a triangle for each task workgroup n the draw
launches and each mesh workgroup j under it, at the cell n, j with the colour that encodes n and j), not what the instructions do,
and the workgroups that ran are read back from the colours: the picture says which ones ran.

--counter makes the GPU side wrong on purpose and the reference stay right; the run then passes only if the comparison, or the
validation layer, catches it:
  ignore-first       a direct draw points the first task at the zero word, an indirect one's record holds a first task of 0
  no-barrier         an indirect draw without the barrier after the records' copy (synchronization validation must report it)
  unconverted-record the record holds the game's {count, first} where the extension reads {count, 1, 1}: its Y is the first
                     task and its Z is zero, so nothing is launched
  wrong-count        the record's group count is doubled
  count-ignored      the mesh stage's primitive count is 1 where the scenario says 0

Exit codes: 0 every scenario passed, 1 one failed, 2 a tool or input is missing, 3 the translator refused a program.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import first_draw as fd  # noqa: E402  (its capability table, the constant bank bytes and the PNG writer)
import mesh_programs as mp  # noqa: E402

CLEAR = (0.0, 0.0, 0.0, 0.0)
CAPABILITIES = {**fd.CAPABILITIES, "MeshShadingEXT": None}
COUNTERS = ("ignore-first", "no-barrier", "unconverted-record", "wrong-count", "count-ignored")

# Reference pixel kinds
COLOR, OUTSIDE, EDGE = "color", "outside", "edge"


class InputError(Exception):
    pass


@dataclass(frozen=True)
class Scenario:
    name: str
    form: str                # "direct" or "indirect"
    first: int               # the draw's first task
    count: int               # the draw's group count
    pair: str = "task"       # "task": a task and a mesh stage; "mesh": a mesh stage alone
    groups: int = 3          # mesh workgroups each task launches
    prims: int = 1           # the primitive count a mesh workgroup writes


def scenarios() -> list[Scenario]:
    out = []
    for form in ("direct", "indirect"):
        out += [Scenario(f"{form}-first3", form, 3, 4), Scenario(f"{form}-first0", form, 0, 3),
                Scenario(f"{form}-zero-count", form, 3, 0), Scenario(f"{form}-no-primitives", form, 3, 4, prims=0),
                Scenario(f"{form}-primitives-over-max", form, 3, 4, prims=2),
                Scenario(f"{form}-mesh-only-first3", form, 3, 4, pair="mesh")]
    return out


# ---- what the programs mean: the CPU reference -----------------------------------------

def workgroups(scn: Scenario) -> list[tuple[int, int]]:
    """The (n, j) pairs that draw a triangle: n the task workgroup's number (the draw's first task and on), j the mesh workgroup
    under it (the fixed row when there is no task stage). A primitive count of 0 draws nothing, and 2 is clamped to the 1 the stage
    declares; a group count of 0 launches nothing and is still a draw."""
    if scn.prims == 0 or scn.count == 0:
        return []
    rows = range(scn.groups) if scn.pair == "task" else [mp.ROW]
    return [(n, j) for n in range(scn.first, scn.first + scn.count) for j in rows]


def triangle_window(n: int, j: int, size: int) -> tuple[list[tuple[float, float]], tuple[float, float, float, float]]:
    """A cell's triangle in window coordinates and its colour, in floating point: x = 0.5 + n / 32, y = 0.5 + j / 32, the leg 5 / 256."""
    x, y, d = 0.5 + n / 32, 0.5 + j / 32, 5 / 256
    ndc = [(x, y), (x + d, y), (x, y + d)]
    return [((a + 1) / 2 * size, (b + 1) / 2 * size) for a, b in ndc], (x, y, 0.5, 1.0)


def edge(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def cell_of(red: float, green: float) -> tuple[int, int]:
    """The (n, j) a colour encodes: red is 0.5 + n / 32, green 0.5 + j / 32."""
    return round((red - 0.5) * 32), round((green - 0.5) * 32)


def expected_image(cells: list[tuple[int, int]], size: int = mp.IMAGE, eps: float = 1e-6):
    """Per pixel: its kind (COLOR, OUTSIDE or EDGE: a pixel centre within eps of a triangle's edge is not judged) and the colour a covered
    pixel has, from the triangles sampled at the pixel centres."""
    kind = [OUTSIDE] * (size * size)
    rgba: list[tuple | None] = [None] * (size * size)
    for n, j in cells:
        verts, colour = triangle_window(n, j, size)
        sign = 1.0 if edge(verts[0], verts[1], verts[2]) > 0 else -1.0
        xs, ys = [v[0] for v in verts], [v[1] for v in verts]
        for py in range(max(int(min(ys)) - 1, 0), min(int(max(ys)) + 2, size)):
            for px in range(max(int(min(xs)) - 1, 0), min(int(max(xs)) + 2, size)):
                p = (px + 0.5, py + 0.5)
                w = [sign * edge(verts[i], verts[(i + 1) % 3], p) for i in range(3)]
                k = py * size + px
                if all(v > eps for v in w):
                    kind[k], rgba[k] = COLOR, colour
                elif all(v >= -eps for v in w):
                    kind[k] = EDGE
    return kind, rgba


def compare(got: list[tuple], expected, atol: float = 1e-5, rtol: float = 1e-5) -> dict:
    """The GPU image against the reference: covered pixels within the tolerance, the others exactly the clear colour, and the (n, j) pairs
    the GPU image shows, with their pixels, against those the reference has."""
    kind, rgba = expected
    counts = {COLOR: 0, OUTSIDE: 0, EDGE: 0}
    mismatches, worst, first = 0, 0.0, []
    want: dict[tuple[int, int], int] = {}
    ran: dict[tuple[int, int], int] = {}
    for k, g in enumerate(got):
        counts[kind[k]] += 1
        if tuple(g) != CLEAR:
            cell = cell_of(g[0], g[1])
            ran[cell] = ran.get(cell, 0) + 1
        if kind[k] == COLOR:
            e = rgba[k]
            cell = cell_of(e[0], e[1])
            want[cell] = want.get(cell, 0) + 1
            worst = max(worst, max(abs(g[c] - e[c]) for c in range(4)))
            ok = all(abs(g[c] - e[c]) <= atol + rtol * abs(e[c]) for c in range(4))
        elif kind[k] == OUTSIDE:
            ok = tuple(g) == CLEAR
        else:
            continue
        if not ok:
            mismatches += 1
            if len(first) < 8:
                first.append({"x": k % mp.IMAGE, "y": k // mp.IMAGE, "kind": kind[k], "gpu": [round(v, 6) for v in g]})
    return {"pixels": counts, "mismatches": mismatches, "max_abs_err": worst, "first_mismatches": first,
            "ran": sorted(ran), "expected_ran": sorted(want), "ran_pixels": {f"{n},{j}": v for (n, j), v in sorted(ran.items())},
            "ran_ok": ran == want}


# ---- the binding, from the descriptions alone ---------------------------------------------

def records(scn: Scenario, counter: str | None) -> tuple[bytes, int]:
    """The records file of an indirect draw and the index of the record it draws: three records, the one drawn in the middle, between two
    that no draw uses. A record is 16 bytes in the conversion pass's output form: the three group counts of the extension, then the first
    task; the game's own record is {count, first}."""
    first = 0 if counter == "ignore-first" else scn.first
    count = scn.count * 2 if counter == "wrong-count" else scn.count
    real = struct.pack("<4I", count, first, 0, 0) if counter == "unconverted-record" else struct.pack("<4I", count, 1, 1, first)
    return struct.pack("<4I", 7, 1, 1, 5) + real + struct.pack("<4I", 9, 1, 1, 6), 1


def check_description(name: str, res: dict, stage: str) -> None:
    if res.get("schema") != fd.SCHEMA:
        raise InputError(f"{name}: schema {res.get('schema')}, not {fd.SCHEMA}: translate again")
    lay = res["layout"]
    if lay["name"] != "uni" or not lay["fits"]:
        raise InputError(f"{name} does not fit the layout uni: {lay.get('overflow')}")
    for field in ("sbufs", "textures", "images"):
        if res[field]:
            raise InputError(f"{name} has {field}: the synthetic programs have none")
    if res["raw_access"]["page_table"]:
        raise InputError(f"{name} walks the page table: the synthetic programs make no global access")
    if stage != "ps":
        mesh = res["mesh_stage"]
        if not mesh["requires_subgroup_32"] or mesh["threads"] != 32:
            raise InputError(f"{name}: the pipeline needs a required subgroup size of 32 and 32 threads")
        first = mesh["first_task"]
        if first["push_offset"] != mp.FIRST_TASK_OFFSET or first["form"] != "device_address":
            raise InputError(f"{name}: the first task is not carried the way vkmesh writes it: {first}")


def device_needs(res_list: list[dict]) -> list[str]:
    """Feature and subgroup lines of the draw file, from the modules' capabilities."""
    needs: set[str] = set()
    for res in res_list:
        for cap in res["capabilities"]:
            if cap not in CAPABILITIES:
                raise InputError(f"capability {cap} has no known device requirement")
            if CAPABILITIES[cap]:
                needs.add(CAPABILITIES[cap])
    return [n if n.startswith("subgroup") else f"feature {n}" for n in sorted(needs)]


def bank_files(res: dict, banks: dict, sdir: Path, tag: str) -> list[tuple[int, int, Path]]:
    """The constant banks a module reads, by the set and binding its description gives: (set, binding, file with the bank's 64 KiB)."""
    out = []
    for cb in res["cbufs"]:
        if cb["role"] != "user":
            raise InputError(f"{tag}: a {cb['role']} constant bank is not drawn")
        path = sdir / f"{tag}-bank{cb['slot']}.dat"
        path.write_bytes(bytes(fd.bank_bytes(banks.get(str(cb["slot"]), {}))))
        out.append((cb["set"], cb["binding"], path))
    return out


def draw_file(scn: Scenario, res: dict, modules: dict[str, Path], sdir: Path, counter: str | None, validation: str) -> tuple[Path, Path]:
    """Writes the draw file of a scenario and the files it names; returns the draw file and the image the harness writes."""
    sdir.mkdir(parents=True, exist_ok=True)
    mesh = "ms" if scn.pair == "task" else "ms_only"
    stages = ["ts", mesh, "ps"] if scn.pair == "task" else [mesh, "ps"]
    prims = 1 if counter == "count-ignored" else scn.prims
    banks = {"ts": {"3": {f"{mp.COUNT_BANK_TASK_WORD:#x}": {"u32": [scn.groups]}}},
             mesh: {"3": {f"{mp.COUNT_BANK_MESH_WORD:#x}": {"u32": [prims]}}}, "ps": {}}
    image = sdir / "gpu.f32"
    lines = [f"ms {modules[mesh]}", f"fs {modules['ps']}", f"out {image}", f"size {mp.IMAGE} {mp.IMAGE}",
             "clear " + " ".join(repr(c) for c in CLEAR), f"validation {validation}"]
    if scn.pair == "task":
        lines.insert(0, f"ts {modules['ts']}")
    letters = {"ts": "t", "ms": "m", "ms_only": "m", "ps": "f"}
    for stage in stages:
        for set_no, binding, path in bank_files(res[stage], banks[stage], sdir, stage):
            if set_no == 1:
                lines.append(f"uniform {binding} {letters[stage]} {path}")
            elif set_no == 2:
                lines.append(f"taskuniform {binding} {path}")
            else:
                raise InputError(f"{stage}: a constant bank at set {set_no}")
    lines += device_needs([res[s] for s in stages])
    if scn.form == "direct":
        lines.append(f"draw direct {scn.count} {scn.first}")
        if counter == "ignore-first":
            lines.append("flag ignorefirst")
    else:
        data, index = records(scn, counter)
        rec = sdir / "records.dat"
        rec.write_bytes(data)
        lines.append(f"draw indirect {index} {rec}")
        if counter == "no-barrier":
            lines.append("flag nobarrier")
    path = sdir / "draw.txt"
    path.write_text("\n".join(lines) + "\n")
    return path, image


# ---- running ------------------------------------------------------------------------------

def find_xlate() -> str | None:
    name = "oot2-xlate.exe" if os.name == "nt" else "oot2-xlate"
    hits = sorted((ROOT / "build" / "xlate" / "publish").rglob(name))
    return str(hits[0]) if hits else None


def translate(work: Path, xlate: str) -> tuple[dict[str, Path], dict[str, dict]]:
    """Writes and translates the synthetic programs: the module of each by name and its description."""
    import pool
    jobs = mp.jobs(work / "programs")
    out = work / "modules"
    # The pool starts the child without a shell and its own folder: a relative path or a bare name finds no translator, and the pool answers nothing
    exe = str(Path(shutil.which(xlate) or xlate).resolve())
    results, _ = pool.run_pool([exe], [(j, out) for j in jobs.values()], 1, work / "logs", timeout=180)
    by_key = {r.key: r for r in results}
    modules, res = {}, {}
    for name, job in jobs.items():
        r = by_key.get(job["key"])
        if r is None or r.status != "ok":
            raise RuntimeError(f"{name}: the translator answered {getattr(r, 'status', None)}: {getattr(r, 'message', 'no answer')}")
        modules[name] = out / f"{job['key']}.spv"
        res[name] = json.loads((out / f"{job['key']}.res.json").read_text(encoding="utf-8"))
        check_description(name, res[name], job["stage"])
    return modules, res


def run_scenario(scn: Scenario, modules, res, vkmesh: Path, work: Path, counter: str | None, validation: str, env: dict) -> dict:
    sdir = work / "runs" / (scn.name if not counter else f"{scn.name}.{counter}")
    draw, image = draw_file(scn, res, modules, sdir, counter, validation)
    t0 = time.perf_counter()
    gpu = subprocess.run([str(vkmesh), str(draw)], capture_output=True, text=True, env=env)
    seconds = time.perf_counter() - t0
    (sdir / "vkmesh-stderr.txt").write_text(gpu.stderr)
    if gpu.returncode != 0:
        return {"scenario": scn.name, "counter": counter, "pass": False, "error": f"vkmesh exit {gpu.returncode}: {gpu.stderr.strip()[-600:]}"}
    vk = json.loads(gpu.stdout.strip().splitlines()[-1])
    raw = image.read_bytes()
    got = [struct.unpack_from("<4f", raw, 16 * k) for k in range(mp.IMAGE * mp.IMAGE)]
    expected = expected_image(workgroups(scn))
    cmp = compare(got, expected)
    fd.write_png(sdir / "gpu.png", mp.IMAGE, mp.IMAGE, got)
    fd.write_png(sdir / "expected.png", mp.IMAGE, mp.IMAGE, [c if c else CLEAR for c in expected[1]])
    validated = vk["validation_errors"] == 0 and vk["validation_warnings"] == 0
    right = cmp["mismatches"] == 0 and cmp["ran_ok"] and cmp["pixels"][EDGE] == 0
    caught = not right or not validated
    return {"scenario": scn.name, "counter": counter, "pass": caught if counter else right and validated,
            "caught": caught if counter else None, "device": vk["device"], "driver": vk["driver"], "api": vk["api"],
            "validation": vk["validation"], "validation_errors": vk["validation_errors"],
            "validation_warnings": vk["validation_warnings"], "loader_messages": vk["loader_messages"], **cmp,
            "gpu_seconds": round(seconds, 3)}


def applies(counter: str | None, scn: Scenario) -> bool:
    """Whether a counter-run means something for a scenario: it needs a draw that launches something, and the form it breaks."""
    if counter is None:
        return True
    if counter == "count-ignored":
        return scn.prims == 0 and scn.count > 0
    if counter in ("no-barrier", "unconverted-record", "wrong-count"):
        return scn.form == "indirect" and scn.count > 0 and scn.prims > 0
    return scn.first != 0 and scn.count > 0 and scn.prims > 0       # ignore-first


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--xlate", default=None)
    ap.add_argument("--vkmesh", type=Path, default=ROOT / "build" / "vkmesh" / ("vkmesh.exe" if os.name == "nt" else "vkmesh"))
    ap.add_argument("--validation", choices=("on", "gpu", "sync", "off"), default="sync")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--counter", choices=COUNTERS, default=None)
    ap.add_argument("--keep-implicit-layers", action="store_true",
                    help="leave the machine's implicit Vulkan layers on (they are disabled by default)")
    args = ap.parse_args(argv)

    xlate = args.xlate or find_xlate()
    if not xlate:
        print("mesh_draw: no translator under build/xlate/publish; run tools/build_xlate.py or give --xlate.", file=sys.stderr)
        return 2
    if not args.vkmesh.is_file():
        print(f"mesh_draw: no harness at {args.vkmesh}; run tools/build_vkdraw.py --harness vkmesh.", file=sys.stderr)
        return 2
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    try:
        modules, res = translate(work, xlate)
    except InputError as e:
        print(f"mesh_draw: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"mesh_draw: {e}", file=sys.stderr)
        return 3

    env = dict(os.environ)
    if not args.keep_implicit_layers:
        env["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"
    reports = []
    for scn in scenarios():
        if (args.scenario and scn.name != args.scenario) or not applies(args.counter, scn):
            continue
        try:
            rep = run_scenario(scn, modules, res, args.vkmesh, work, args.counter, args.validation, env)
        except InputError as e:
            print(f"mesh_draw: {scn.name}: {e}", file=sys.stderr)
            return 2
        reports.append(rep)
        print("mesh_draw:", json.dumps({k: rep.get(k) for k in ("scenario", "counter", "pass", "caught", "mismatches", "ran_ok",
                                                               "validation_errors", "validation_warnings", "error")}), flush=True)
    suffix = f"-{args.counter}" if args.counter else ""
    (work / f"report-{args.validation}{suffix}.json").write_text(json.dumps(reports, indent=2))
    return 0 if reports and all(r["pass"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
