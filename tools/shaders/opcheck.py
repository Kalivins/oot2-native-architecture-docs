#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Compare two translations of the corpus: sizes per stage and kind, operations lost, reads moved.

    py -3 tools/shaders/opcheck.py stats --work <dir> --pass NAME [--workers 2] [--limit N] [--tensor-keys FILE]
    py -3 tools/shaders/opcheck.py ab --ref <dir>/<pass> --cand <dir>/<pass> --out summary.json

`stats` reads the modules a corpus.py pass left under <dir>/<pass>/spv and writes <dir>/<pass>/stats.tsv:
one row per module with its size, its kind and the counts of tools/shaders/spvstats.py. The kind is
`raw` for a module whose description says it goes through the page table, `tensor` for a compute
module named in --tensor-keys (one job key per line), else `plain`.

`ab` joins two stats files by job key and reports, for the modules both translate: the SPIR-V size and
instruction count per stage and kind; the modules that lost an atomic, a barrier, a subgroup
operation, a call that writes or a store, with the totals; the modules in which a read moved (a lower
bound, see spvstats; not asked of a module that lost a write, whose epochs no longer line up); and the jobs the candidate fails and the reference does not. The reference is
the translation with the AST optimizer's debug mode on, which moves and drops nothing.

The summary holds counts only. Rows name the game's programs and stay under --work, outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spvstats  # noqa: E402

FIELDS = ("key", "stage", "kind", "bytes", "instructions", "ids", "functions", "constructs", "atomics", "barriers", "memory_barriers", "subgroup",
          "calls_write", "calls_read", "calls_pure", "stores", "loads", "sig")
COUNTS = ("atomics", "barriers", "memory_barriers", "subgroup", "calls_write", "calls_read", "calls_pure")


def _one(args: tuple[str, str, str, bool]) -> list | None:
    folder, key, stage, is_tensor = args
    root = Path(folder) / "spv"
    try:
        data = (root / f"{key}.spv").read_bytes()
        stats = spvstats.analyze(data)
    except (OSError, ValueError):
        return None
    try:
        description = json.loads((root / f"{key}.res.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        description = {}
    if is_tensor and stage == "cs":
        kind = "tensor"
    else:
        kind = "raw" if (description.get("raw_access") or {}).get("page_table") else "plain"
    return [key, stage, kind, stats["bytes"], stats["instructions"], stats["ids"], stats["functions"], stats["constructs"],
            *(stats[c] for c in COUNTS), json.dumps(stats["stores"]), json.dumps(stats["loads"]),
            json.dumps(stats["sig"], separators=(",", ":"))]


def cmd_stats(args: argparse.Namespace) -> int:
    folder = args.work.resolve() / args.pass_name
    results = folder / "results.tsv"
    if not results.exists():
        print(f"opcheck: no results.tsv in {folder}", file=sys.stderr)
        return 2
    with open(results, encoding="utf-8", newline="") as fh:
        ok = sorted(r["key"] for r in csv.DictReader(fh, delimiter="\t") if r["status"] == "ok")
    stage_of = _stages(args.split.resolve())
    tensor = set(Path(args.tensor_keys).read_text(encoding="utf-8").split()) if args.tensor_keys else set()
    if args.limit:
        ok = ok[:args.limit]
    work = [(str(folder), k, stage_of.get(k, "?"), k in tensor) for k in ok]
    out = folder / "stats.tsv"
    with multiprocessing.Pool(args.workers) as pool, open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(FIELDS)
        bad = 0
        for n, row in enumerate(pool.imap_unordered(_one, work, chunksize=8), 1):
            if row is None:
                bad += 1
                continue
            w.writerow(row)
            if n % 5000 == 0:
                print(f"opcheck: {n}/{len(work)}", flush=True)
    print(f"opcheck: {len(work) - bad} modules analysed into {out}, {bad} unreadable", flush=True)
    return 0 if bad == 0 else 1


def _stages(split: Path) -> dict[str, str]:
    stages = {}
    with open(split / "jobs.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                job = json.loads(line)
                stages[job["key"]] = job["stage"]
    return stages


def read_stats(folder: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(folder / "stats.tsv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows[r["key"]] = {
                "stage": r["stage"], "kind": r["kind"], "bytes": int(r["bytes"]), "instructions": int(r["instructions"]),
                "ids": int(r["ids"]), "functions": int(r["functions"]), "constructs": int(r["constructs"]), **{c: int(r[c]) for c in COUNTS},
                "stores": json.loads(r["stores"]), "loads": json.loads(r["loads"]), "sig": json.loads(r["sig"]),
            }
    return rows


def read_status(folder: Path) -> dict[str, tuple[str, str]]:
    with open(folder / "results.tsv", encoding="utf-8", newline="") as fh:
        return {r["key"]: (r["status"], r["message"]) for r in csv.DictReader(fh, delimiter="\t")}


def summarise(ref: dict[str, dict], cand: dict[str, dict], ref_status: dict, cand_status: dict, only100: set[str]) -> dict:
    """The comparison of two passes over the jobs both translated (see the module docstring)."""
    both = sorted(set(ref) & set(cand))
    size: dict[str, list] = defaultdict(lambda: [0, 0, 0, 0, 0, []])
    lost_modules: dict[str, Counter] = defaultdict(Counter)
    lost_total: dict[str, Counter] = defaultdict(Counter)
    moved_modules: dict[str, Counter] = {"reads": Counter(), "convergent": Counter()}
    moved_total: dict[str, Counter] = {"reads": Counter(), "convergent": Counter()}
    unavailable: Counter = Counter()
    restructured: Counter = Counter()
    for key in both:
        a, b = ref[key], cand[key]
        group = (a["stage"], a["kind"])
        for g in (group, (a["stage"], "all"), ("all", a["kind"]), ("all", "all")):
            s = size[f"{g[0]}/{g[1]}"]
            s[0] += 1
            s[1] += a["bytes"]
            s[2] += b["bytes"]
            s[3] += a["instructions"]
            s[4] += b["instructions"]
            s[5].append(b["bytes"] / a["bytes"])
        verdict = spvstats.compare(a, b)
        for name, n in verdict["lost"].items():
            for g in (a["kind"], "all"):
                lost_modules[name][g] += 1
                lost_total[name][g] += n
        if verdict["restructured"]:
            restructured[a["kind"]] += 1
        if verdict["moved"] is None:
            unavailable[a["kind"]] += 1
        for what, n in (("reads", verdict["moved"]), ("convergent", verdict["moved_convergent"])):
            if n:
                for g in (a["kind"], "all"):
                    moved_modules[what][g] += 1
                    moved_total[what][g] += n
    failed: Counter = Counter()
    for key, (status, _msg) in cand_status.items():
        if status != "ok" and ref_status.get(key, ("?",))[0] == "ok":
            failed[key in only100 and "only_product100" or "in_set"] += 1
    per_kind_modules = Counter(ref[k]["kind"] for k in both)
    return {
        "modules_compared": len(both),
        "modules_by_kind": dict(sorted(per_kind_modules.items())),
        "size": {k: {"modules": v[0], "ref_bytes": v[1], "cand_bytes": v[2], "bytes_ratio": round(v[2] / v[1], 4) if v[1] else None,
                     "ref_instructions": v[3], "cand_instructions": v[4],
                     "instructions_ratio": round(v[4] / v[3], 4) if v[3] else None,
                     "median_bytes_ratio": round(statistics.median(v[5]), 4)} for k, v in sorted(size.items())},
        "lost": {n: {"modules": dict(lost_modules[n]), "operations": dict(lost_total[n])} for n in sorted(lost_modules)},
        "moved_reads_lower_bound": {"modules": dict(moved_modules["reads"]), "reads": dict(moved_total["reads"])},
        "moved_convergent_lower_bound": {"modules": dict(moved_modules["convergent"]), "operations": dict(moved_total["convergent"])},
        "restructured_modules": dict(restructured),
        "comparison_unavailable": dict(unavailable),
        "failed_in_candidate_only": dict(failed),
        "translated_only_by_ref": len(set(ref) - set(cand)),
        "translated_only_by_cand": len(set(cand) - set(ref)),
    }


def cmd_ab(args: argparse.Namespace) -> int:
    ref_dir, cand_dir = Path(args.ref).resolve(), Path(args.cand).resolve()
    for d in (ref_dir, cand_dir):
        if not (d / "stats.tsv").exists():
            print(f"opcheck: no stats.tsv in {d}; run `stats` first.", file=sys.stderr)
            return 2
    only100: set[str] = set()
    if args.split:
        with open(args.split.resolve() / "programs.tsv", encoding="utf-8", newline="") as fh:
            rows = csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t")
            only100 = {r["job_key"] for r in rows if r.get("only_product100") == "1"}
    summary = summarise(read_stats(ref_dir), read_stats(cand_dir), read_status(ref_dir), read_status(cand_dir), only100)
    text = json.dumps(summary, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stats")
    st.add_argument("--split", type=Path, required=True)
    st.add_argument("--work", type=Path, required=True)
    st.add_argument("--pass", dest="pass_name", required=True)
    st.add_argument("--workers", type=int, default=2)
    st.add_argument("--limit", type=int, default=None)
    st.add_argument("--tensor-keys", default=None)
    ab = sub.add_parser("ab")
    ab.add_argument("--ref", required=True)
    ab.add_argument("--cand", required=True)
    ab.add_argument("--split", type=Path, default=None, help="to split failures into the set the game loads and the base-only programs")
    ab.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    return cmd_stats(args) if args.cmd == "stats" else cmd_ab(args)


if __name__ == "__main__":
    sys.exit(main())
