#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Build the shader translator (xlate/host) as a NativeAOT executable.

    py -3 tools/build_xlate.py [--out build/xlate] [--rid win-x64]

The build goes through `dotnet publish` with every intermediate and output file
under --out (default build/xlate, ignored by git), so the source tree stays
clean. It prints the executable's path, size and SHA-256; the SHA-256 is the
translator identity that shader cache keys include (architecture D21).

Windows: NativeAOT finds the MSVC linker through Visual Studio's vcvarsall,
whose scripts in Visual Studio 18 Build Tools call vswhere.exe without a path.
The Visual Studio Installer folder is therefore put on PATH for the build; an
environment without the variable ProgramFiles(x86) (Git Bash drops names with
parentheses) falls back to the default location.

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
PROJECT = ROOT / "xlate" / "host" / "Oot2Xlate.csproj"
NUGET_CONFIG = ROOT / "xlate" / "NuGet.config"
EXE_NAME = "oot2-xlate"
SDK_MAJOR = 10


def sdk_ok(dotnet: str) -> bool:
    out = subprocess.run([dotnet, "--list-sdks"], capture_output=True, text=True)
    if out.returncode != 0:
        return False
    return any(line.split(".", 1)[0] == str(SDK_MAJOR) for line in out.stdout.splitlines())


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    if os.name == "nt":
        base = env.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
        installer = Path(base) / "Microsoft Visual Studio" / "Installer"
        if installer.is_dir():
            env["PATH"] = str(installer) + os.pathsep + env.get("PATH", "")
    return env


def find_exe(out: Path, rid: str) -> Path | None:
    suffix = ".exe" if rid.startswith("win") else ""
    hits = sorted((out / "publish").rglob(EXE_NAME + suffix))
    return hits[0] if hits else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "xlate")
    ap.add_argument("--rid", default="win-x64" if os.name == "nt" else "linux-x64")
    ap.add_argument("--configuration", default="Release")
    args = ap.parse_args(argv)

    dotnet = shutil.which("dotnet")
    if not dotnet or not sdk_ok(dotnet):
        print(f"build_xlate: the .NET {SDK_MAJOR} SDK is required (dotnet --list-sdks).", file=sys.stderr)
        return 2

    out = args.out.resolve()
    cmd = [dotnet, "publish", str(PROJECT), "-c", args.configuration, "-r", args.rid,
           "--configfile", str(NUGET_CONFIG), "--artifacts-path", str(out), "-nologo"]
    print("build_xlate:", " ".join(cmd), flush=True)
    if subprocess.run(cmd, env=build_env()).returncode != 0:
        print("build_xlate: dotnet publish failed.", file=sys.stderr)
        return 1

    exe = find_exe(out, args.rid)
    if exe is None:
        print(f"build_xlate: no {EXE_NAME} executable under {out / 'publish'}.", file=sys.stderr)
        return 1
    digest = hashlib.sha256(exe.read_bytes()).hexdigest()
    print(f"build_xlate: {exe}")
    print(f"build_xlate: size {exe.stat().st_size} bytes, sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
