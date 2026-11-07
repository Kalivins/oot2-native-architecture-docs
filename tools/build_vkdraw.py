#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Build the Vulkan harnesses of the spikes with clang.

    py -3 tools/build_vkdraw.py [--harness vkdraw|vkmesh|vkpipecost|vkcomputecost|vkmeshcost|vkhmmacost|vkwatchcost|vkpresent]
                                [--out DIR] [--sdk DIR]
                                [--cc clang]

vkdraw is spike S2's first draw (tools/firstdraw/vkdraw.c); vkmesh is its task and mesh
sibling, for the first translated mesh draw (tools/firstdraw/vkmesh.c, ticket 72); vkpipecost is spike
S3's pipeline and descriptor cost harness (tools/pipecost/*.c, Windows only);
vkcomputecost is spike S6's compute pipeline build harness
(tools/computecost/vkcomputecost.c, Windows only); vkmeshcost is spike S4's task and mesh
pipeline library build harness (tools/meshcost/vkmeshcost.c, Windows only);
vkhmmacost times compute
dispatches for the HMMA cost measurement (tools/hmmacost/vkhmmacost.c, Windows
only); vkwatchcost is spike S5's write-watch and host-import probe
(tools/watchcost/vkwatchcost.c, Windows only), which runs on the runtime's own
address space, so it is built with runtime/core the way the runtime is (gnu11,
x86-64-v3, linked without high-entropy ASLR); vkpresent is the windowed present
spike's harness (tools/presentcost/vkpresent.c, Windows only), built with the
Win32 surface declarations.
All load Vulkan at run time through Volk, which the Vulkan SDK ships as source
(Include/Volk/volk.c, MIT), so no import library is linked. The SDK is taken
from --sdk, else from the VULKAN_SDK variable. The output goes under --out
(default build/<harness>, ignored by git); the script prints the executable's
path, size and SHA-256.

Exit codes: 0 built, 1 the build failed, 2 the tools are missing.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "vkdraw": [ROOT / "tools" / "firstdraw" / "vkdraw.c"],
    "vkmesh": [ROOT / "tools" / "firstdraw" / "vkmesh.c"],
    "vkpipecost": sorted((ROOT / "tools" / "pipecost").glob("vkpc_*.c")),
    "vkcomputecost": [ROOT / "tools" / "computecost" / "vkcomputecost.c"],
    "vkmeshcost": [ROOT / "tools" / "meshcost" / "vkmeshcost.c"],
    "vkhmmacost": [ROOT / "tools" / "hmmacost" / "vkhmmacost.c"],
    "vkwatchcost": [ROOT / "tools" / "watchcost" / "vkwatchcost.c"],
    "vkpresent": [ROOT / "tools" / "presentcost" / "vkpresent.c"],
}
# Harnesses built with the runtime's core: its sources, its include folders and the flags its own tests use.
WITH_RUNTIME = {"vkwatchcost"}
# Harnesses that open a window: Volk and the harness see the Win32 surface declarations.
WITH_WIN32_SURFACE = {"vkpresent"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--harness", choices=sorted(SOURCES), default="vkdraw")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sdk", type=Path, default=Path(os.environ["VULKAN_SDK"]) if os.environ.get("VULKAN_SDK") else None)
    ap.add_argument("--cc", default=os.environ.get("CC", "clang"))
    args = ap.parse_args(argv)

    cc = shutil.which(args.cc)
    if not cc:
        print(f"build_vkdraw: no C compiler named {args.cc!r} on PATH.", file=sys.stderr)
        return 2
    if args.sdk is None:
        print("build_vkdraw: give --sdk or set VULKAN_SDK.", file=sys.stderr)
        return 2
    include = args.sdk / "Include"
    volk = include / "Volk" / "volk.c"
    if not (include / "vulkan" / "vulkan_core.h").is_file() or not volk.is_file():
        print(f"build_vkdraw: {args.sdk} has no Include/vulkan or Include/Volk/volk.c.", file=sys.stderr)
        return 2

    out = (args.out or ROOT / "build" / args.harness).resolve()
    out.mkdir(parents=True, exist_ok=True)
    exe = out / (f"{args.harness}.exe" if os.name == "nt" else args.harness)
    sources = [str(s) for s in SOURCES[args.harness]]
    std, extra = ["-std=c11"], []
    if args.harness in WITH_RUNTIME:
        core = ROOT / "runtime" / "core"
        std = ["-std=gnu11", "-march=x86-64-v3"]
        extra = ["-I", str(ROOT / "runtime" / "include"), "-I", str(core)]
        sources += sorted(str(p) for p in core.glob("*.c"))
        extra.append("-Wl,--disable-high-entropy-va" if os.name == "nt" else "-pthread")
    if args.harness in WITH_WIN32_SURFACE:
        extra += ["-DVK_USE_PLATFORM_WIN32_KHR", "-luser32"]
    cmd = [cc, *std, "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-missing-field-initializers",
           "-I", str(include), "-I", str(include / "Volk"), *extra, *sources, str(volk), "-o", str(exe)]
    if os.name != "nt":
        cmd.append("-ldl")
    print("build_vkdraw:", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode != 0 or not exe.is_file():
        print("build_vkdraw: the compiler failed.", file=sys.stderr)
        return 1
    digest = hashlib.sha256(exe.read_bytes()).hexdigest()
    print(f"build_vkdraw: {exe}")
    print(f"build_vkdraw: size {exe.stat().st_size} bytes, sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
