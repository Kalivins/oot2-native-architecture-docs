#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Place the game's modules in the guest address space and write the layout header.

    py -3 tools/layout.py --module main=0x5000000 --module sdk=0x1200000 --out DIR

Guest addresses are host addresses, and the lifter writes absolute addresses
into the generated C, so the lifter, the build and the runtime must agree on
every module base. This is the one place that decides them: it writes
DIR/layout.json for the Python tools and DIR/layout.h for C. DIR is a build
folder outside the repository, since the sizes come from the player's dump.

Modules go in Horizon's load order (main, subsdk0 to subsdk9, sdk; the host
links them, so rtld has no place), each base aligned to 2 MiB from
0x70_0000_0000, and all of them below MODULE_LIMIT, which runtime.h repeats as
RT_MODULE_LIMIT: above it the runtime keeps its own fixed regions, and every
guest address stays below 2^39, the limit of IPC buffer descriptors.

Adapted from DKB Recomp recomp/layout.py at 9d8fa02 (GPL-3.0-or-later). Changed:
sizes are given rather than read from the modules, and the module limit is the
runtime's, not 2^39.

Exit codes: 0 written, 1 the modules do not fit or are malformed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

FIRST_BASE = 0x70_0000_0000
ALIGN = 0x20_0000
PAGE = 0x1000
MODULE_LIMIT = 0x78_0000_0000
ORDER = ["main"] + [f"subsdk{i}" for i in range(10)] + ["sdk"]


@dataclass(frozen=True)
class Module:
    name: str
    base: int
    size: int


def align_up(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def place(sizes: dict[str, int]) -> list[Module]:
    """Modules in load order with their bases. Raises ValueError on a module that is unknown, empty, not a whole
    number of pages, or that would end past MODULE_LIMIT."""
    unknown = sorted(set(sizes) - set(ORDER))
    if unknown:
        raise ValueError(f"unknown module name(s): {', '.join(unknown)}; expected {', '.join(ORDER)}")
    if "main" not in sizes:
        raise ValueError("no main module")
    out, base = [], FIRST_BASE
    for name in ORDER:
        if name not in sizes:
            continue
        size = sizes[name]
        if not isinstance(size, int) or size <= 0 or size % PAGE:
            raise ValueError(f"module {name}: size {size!r} is not a positive multiple of {PAGE:#x}")
        if base + size > MODULE_LIMIT:
            raise ValueError(f"module {name} at {base:#x} of size {size:#x} ends past {MODULE_LIMIT:#x}")
        out.append(Module(name, base, size))
        base = align_up(base + size)
    return out


def header_text(mods: list[Module]) -> str:
    lines = [
        "// Module layout chosen by tools/layout.py for one player's dump. Written by the build; do not edit.",
        "#pragma once",
        "",
        f"#define MOD_LAYOUT_LIMIT {MODULE_LIMIT:#x}ull",
    ]
    for m in mods:
        lines.append(f"#define MOD_{m.name.upper()}_BASE {m.base:#x}ull")
        lines.append(f"#define MOD_{m.name.upper()}_SIZE {m.size:#x}ull")
    lines += [
        "",
        f"#define MOD_COUNT {len(mods)}",
        "#define MOD_LIST(X) " + " ".join(f"X({m.name}, {m.name.upper()})" for m in mods),
        "",
    ]
    return "\n".join(lines)


def write(mods: list[Module], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    data = [{"name": m.name, "base": f"{m.base:#x}", "size": f"{m.size:#x}"} for m in mods]
    (outdir / "layout.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    (outdir / "layout.h").write_text(header_text(mods), encoding="utf-8", newline="\n")


def read(outdir: Path) -> list[Module]:
    data = json.loads((outdir / "layout.json").read_text(encoding="utf-8"))
    return [Module(m["name"], int(m["base"], 16), int(m["size"], 16)) for m in data]


def parse_module(text: str) -> tuple[str, int]:
    name, sep, size = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected NAME=SIZE, got {text!r}")
    try:
        return name, int(size, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"size of {name} is not a number: {size!r}") from None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--module", type=parse_module, action="append", required=True, metavar="NAME=SIZE")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    sizes: dict[str, int] = {}
    for name, size in args.module:
        if name in sizes:
            print(f"layout: module {name} given twice", file=sys.stderr)
            return 1
        sizes[name] = size
    try:
        mods = place(sizes)
    except ValueError as e:
        print(f"layout: {e}", file=sys.stderr)
        return 1
    write(mods, args.out)
    for m in mods:
        print(f"layout: {m.name:8} {m.base:#x} size {m.size:#x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
