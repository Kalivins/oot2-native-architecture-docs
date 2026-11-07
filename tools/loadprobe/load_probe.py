#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Load the player's plain modules with the runtime's loader and report the cost.

    py -3 tools/loadprobe/load_probe.py --module main=PATH --module sdk=PATH --out DIR

Each PATH is a plain module (an NSO0 with uncompressed segments) that the
prepare step wrote from the player's own dump. The image sizes are read from
the module headers and placed with tools/layout.py; DIR (outside the
repository, since the sizes come from the dump) receives layout.h and the
probe program, built with runtime/core. The probe prints relocations by type,
imports by binding, and per region what is mapped, committed and resident,
with the process's private commit and working set after each stage.

Exit codes: 0 loaded and reported, 1 a module is unreadable or compressed or
the build failed, else the probe's own (2 usage, 3 its end marker was linked
below its stand-in function, 70 runtime fatal, 71 unimplemented).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402

CORE = ROOT / "runtime" / "core"
PAGE = 0x1000


def image_size(path: Path) -> int:
    """The image a plain module occupies: data end plus bss, rounded up to a page. Raises ValueError otherwise."""
    with path.open("rb") as f:
        head = f.read(0x100)
    if len(head) < 0x100 or head[:4] != b"NSO0":
        raise ValueError(f"{path} is not an NSO0 module")
    flags = struct.unpack_from("<I", head, 0xC)[0]
    if flags & 0x87:
        raise ValueError(f"{path} is compressed (flags {flags:#x}); decode it to a plain module first")
    _, data_mem, data_size = struct.unpack_from("<III", head, 0x30)
    bss = struct.unpack_from("<I", head, 0x3C)[0]
    return (data_mem + data_size + bss + PAGE - 1) & ~(PAGE - 1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_module(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {text!r}")
    return name, Path(path)


def build(outdir: Path, cc: str) -> Path:
    exe = outdir / f"load_probe{'.exe' if os.name == 'nt' else ''}"
    # probe_end.c right after load_probe.c: its end marker must lie past the stand-in function (the lifted-code end marker).
    sources = [str(Path(__file__).with_name(n)) for n in ("load_probe.c", "probe_end.c")] + \
        sorted(str(p) for p in CORE.glob("*.c"))
    extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
    cmd = [cc, "-std=gnu11", "-O2", "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
           "-I", str(ROOT / "runtime" / "include"), "-I", str(CORE), "-I", str(outdir), *sources, "-o", str(exe)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"build failed:\n{done.stdout}{done.stderr}")
    return exe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--module", type=parse_module, action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
    if not cc:
        print("load_probe: no clang (set OOT2_CC)", file=sys.stderr)
        return 1
    paths = dict(args.module)
    try:
        mods = layout.place({name: image_size(path) for name, path in paths.items()})
        layout.write(mods, args.out)
        exe = build(args.out, cc)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"load_probe: {e}", file=sys.stderr)
        return 1
    run = [str(exe)]
    for m in mods:
        run += [str(paths[m.name]), sha256_file(paths[m.name])]
    return subprocess.run(run).returncode


if __name__ == "__main__":
    sys.exit(main())
