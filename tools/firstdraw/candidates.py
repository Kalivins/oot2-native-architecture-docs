#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Pick vertex and pixel pairs for a draw that exercises every kind of binding.

    py -3 tools/firstdraw/candidates.py --split DIR --pass-dir DIR --out FILE
        [--smoke N --work DIR]

Reads a corpus pass (tools/shaders/corpus.py: <pass-dir>/validation.tsv and
<pass-dir>/spv/<key>.res.json) and the split's programs.tsv, pairs the vertex
and pixel programs that share a first label (one of each, spike S2's pairing)
and keeps a pair when both modules validate and:

- both stages have bound textures and storage buffers;
- the pair needs more than 32 set 0 descriptors in S2's count (the support
  buffer, banks, storage buffers, texture elements and storage images of both
  stages, the support buffer once), so that it would not have fit one push
  descriptor set before the layout `uni`;
- nothing the draw harness does not bind: no raw global access, no writable
  storage, no texture array, no storage image.

Candidates are written to --out, smallest total module size first, one JSON
line each; the order is reproducible for one translator and one split. With
--smoke N, the N-th candidate gets a spec for `first_draw.py --gpu-only`
under --work: zero banks except each storage buffer's address and size, a
zero-filled region behind each, an 8 by 8 texture of the description's type
per bound texture, three vertices of zeros. It checks the binding and the
validation layer, not the picture.

--out and --work hold program keys: they must lie outside the repository.
Only counts are printed. Exit codes: 0 done, 2 an input is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PUSH_LIMIT = 32
SMOKE_LAYERS = {"2d": {}, "2darray": {"layers": 2}, "3d": {"depth": 4}, "cube": {}}


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        return [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


def bound(t: dict, schema: int) -> bool:
    """A texture the program names by a constant handle: set 0 in schema 1, a slot (or descriptor) in schema 2."""
    return t["set"] == 0 if schema == 1 else t.get("via") != "pool"


def summary(res: dict) -> dict:
    """What the selection reads from one description (schema 1 or 2)."""
    schema = res.get("schema", 1)
    tex = [t for t in res["textures"] if bound(t, schema)]
    ra = res.get("raw_access") or {}
    return {
        "ubo": len(res["cbufs"]), "ssbo": len(res["sbufs"]), "tex": len(tex),
        "tex_elements": sum(max(1, t["array_length"]) for t in tex),
        "arrays": sum(t["array_length"] != 1 for t in tex), "img": len(res["images"]),
        "raw": bool(ra.get("physical_pointers") or ra.get("page_table")),
        "writable": bool(res.get("writable_storage", True)), "spv_bytes": res["spv_bytes"],
    }


def s2_count(m: dict) -> int:
    """S2's set 0 count of one module: the support buffer, banks, storage buffers, texture elements, images."""
    return 1 + m["ubo"] + m["ssbo"] + m["tex_elements"] + m["img"]


def qualifies(vs: dict, ps: dict) -> bool:
    return (vs["tex"] > 0 and ps["tex"] > 0 and vs["ssbo"] > 0 and ps["ssbo"] > 0
            and s2_count(vs) + s2_count(ps) - 1 > PUSH_LIMIT
            and not any(m["raw"] or m["writable"] or m["arrays"] or m["img"] for m in (vs, ps)))


def candidates(split: Path, pass_dir: Path) -> tuple[list[dict], dict]:
    programs = [r for r in read_tsv(split / "programs.tsv") if r["stage"] in ("vs", "ps")]
    valid = {r["key"]: r["strict_ok"] == "1" for r in read_tsv(pass_dir / "validation.tsv")}
    by: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in programs:
        by[r["first_label"]][r["stage"]].append(r["job_key"])
    pairs = [(st["vs"][0], st["ps"][0]) for st in by.values() if len(st.get("vs", [])) == 1 and len(st.get("ps", [])) == 1]
    out, counts = [], {"pairs": len(pairs), "valid_pairs": 0, "candidates": 0}
    for vk, pk in pairs:
        if not (valid.get(vk) and valid.get(pk)):
            continue
        counts["valid_pairs"] += 1
        vs, ps = (summary(json.loads((pass_dir / "spv" / f"{k}.res.json").read_text(encoding="utf-8"))) for k in (vk, pk))
        if qualifies(vs, ps):
            out.append({"vs": vk, "ps": pk, "s2_count": s2_count(vs) + s2_count(ps) - 1,
                        "spv_bytes": vs["spv_bytes"] + ps["spv_bytes"], "vs_summary": vs, "ps_summary": ps})
    out.sort(key=lambda c: (c["spv_bytes"], c["vs"], c["ps"]))
    counts["candidates"] = len(out)
    if out:
        counts["smallest_spv_bytes"] = out[0]["spv_bytes"]
        counts["s2_count_max"] = max(c["s2_count"] for c in out)
    return out, counts


def smoke_spec(split: Path, pass_dir: Path, c: dict) -> dict:
    """A spec for first_draw.py --gpu-only: every binding filled, nothing meaningful drawn."""
    res = {st: json.loads((pass_dir / "spv" / f"{c[st]}.res.json").read_text(encoding="utf-8")) for st in ("vs", "ps")}
    if any(r.get("schema") != 2 for r in res.values()):
        raise ValueError("the smoke spec needs schema 2 descriptions: run the corpus pass with this translator")
    banks: dict = {"vs": {}, "ps": {}}
    regions, textures = [], []
    addr = 0x10000000
    for st in ("vs", "ps"):
        for s in res[st]["sbufs"]:
            bank = banks[st].setdefault(str(s["cb_slot"]), {})
            bank[hex(s["address_byte_offset"])] = {"u64": addr}
            bank[hex(s["size_byte_offset"])] = {"u32": [4096]}
            regions.append({"address": addr, "f32": [0.0] * 1024})
            addr += 0x10000
        for t in res[st]["textures"]:
            textures.append({"stage": st, "handle": hex(t["tex_word_offset"]), "type": t["type"], "width": 8, "height": 8,
                             "seed": len(textures) + 1, **SMOKE_LAYERS.get(t["type"], {})})
    words = sum(v["components"] for v in res["vs"]["vertex_inputs"] or [])
    return {"jobs": str(split / "jobs.jsonl"), "spv_dir": str(pass_dir / "spv"), "vs": c["vs"], "ps": c["ps"],
            "size": [64, 64], "clear": [0.0, 0.0, 0.0, 0.0], "vertex_stream": {"vertices": [[0.0] * words] * 3},
            "triangles": [[0, 1, 2]], "banks": banks, "global": regions, "textures": textures,
            "scenarios": [{"name": "smoke"}]}


def inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--pass-dir", type=Path, required=True, help="a corpus pass folder: validation.tsv and spv/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", type=int, default=None, help="write a --gpu-only spec for this candidate")
    ap.add_argument("--work", type=Path, default=None)
    args = ap.parse_args(argv)
    for p in (args.out, args.work):
        if p is not None and inside_repo(p):
            print(f"candidates: {p} is inside the repository; program keys stay outside it", file=sys.stderr)
            return 2
    try:
        out, counts = candidates(args.split, args.pass_dir)
    except (OSError, KeyError, ValueError) as e:
        print(f"candidates: {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(json.dumps(counts))
    if args.smoke is not None:
        if args.work is None or not 0 <= args.smoke < len(out):
            print("candidates: --smoke needs --work and a candidate index", file=sys.stderr)
            return 2
        args.work.mkdir(parents=True, exist_ok=True)
        (args.work / "spec.json").write_text(json.dumps(smoke_spec(args.split, args.pass_dir, out[args.smoke]), indent=1),
                                             encoding="utf-8")
        print(f"candidates: spec for candidate {args.smoke} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
