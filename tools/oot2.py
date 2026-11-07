#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The oot2 command line. One command so far:

    py -3 tools/oot2.py prepare --dump FILE --dest DIR [--check] [--verify] [--cc CC] [--vulkan-sdk DIR] [--jobs N]
                                [--xlate FILE] [--translate-jobs N] [--plain-modules DIR] [--zbic-archive FILE]

prepare builds the game from the player's own dump into DIR, one step after the
other (tools/prepare/). DIR must be empty or new on the first run, and the disk
that holds it needs about 38 GiB free. A run that stops, whether by Ctrl+C, a
failure, a crash or a power cut, keeps the steps that finished, and the next run
with the same DIR starts at the first step that did not; --dump may then be left
out. Running it again after an update of the tools rebuilds only what the update
changed.

--check runs the check step alone (the dump, the disk, the CPU, the toolchain,
the shader translator and Vulkan) and says which of the other steps would run and why. It records the
dump's hash and what it found in DIR, so a later run does not hash the dump again.

--verify also hashes the dump again and fully checks every kept output.

--jobs sets how many files the compile step compiles at once (default: the CPU
count, capped by the free memory). The program it builds is DIR/oot2.exe.

--xlate gives the shader translator (default: the OOT2_XLATE variable, else
oot2-xlate on PATH), and --translate-jobs how many copies of it the translate
step runs at once (default: the CPU count, capped by the free memory). The
translated shaders go to DIR/shaders/shader-base.bin. The next step, ledger, reads DIR/ledger.jsonl (the game writes
it when a run ends) and translates the shader answers it recorded as variants into DIR/shaders/shader-variants.bin.

--plain-modules gives modules the player already has in plain form (files named main, sdk, ...): they are checked
against the dump's and used instead of decoding, so the ZBIC decoder is not fetched for them. --zbic-archive gives
the decoder's source archive (hexkyz/zstd at the pinned commit, tools/prepare/zbic.py) instead of downloading it.

Exit codes: 0 done, 1 failed (the message says why), 2 bad command line,
3 cancelled.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare import all_steps  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, Reporter, run_steps  # noqa: E402

EXIT = {"ok": 0, "failed": 1, "cancelled": 3}


def cancel(signum, frame):
    raise KeyboardInterrupt


def positive(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {text!r}")
    return n


def prepare(args: argparse.Namespace) -> int:
    rep = Reporter()
    dest = args.dest.resolve()
    try:
        state = statefile.load(dest)
    except statefile.StateError as e:
        rep.error(str(e))
        return EXIT["failed"]
    ctx = Context(dest, args, rep, state=state, verify=args.verify)
    steps = all_steps()
    rep.say(f"oot2 prepare: {'first run' if ctx.fresh else 'resuming'} in {dest}; "
            f"steps: {', '.join(s.id for s in steps)}")
    result = run_steps(steps, ctx, only={"check"} if args.check else None)
    rep.say(f"oot2 prepare: {result}")
    return EXIT[result]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oot2", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="build the game from your own dump",
                       description="Build the game from your own dump, resuming where the last run stopped.")
    p.add_argument("--dump", type=Path, help="your own dump of the game (.nup, update 1.1.1)")
    p.add_argument("--dest", type=Path, required=True, help="output folder, empty or new on the first run")
    p.add_argument("--check", action="store_true", help="run the check step only, and say what the others would do")
    p.add_argument("--verify", action="store_true", help="hash the dump again and fully check every kept output")
    p.add_argument("--cc", help="C compiler (default: the OOT2_CC variable, else clang on PATH)")
    p.add_argument("--vulkan-sdk", type=Path,
                   help="Vulkan SDK whose headers, Volk and glslang the host probe and the build use (default: the "
                        "VULKAN_SDK variable)")
    p.add_argument("--dav1d-dir", type=Path,
                   help="dav1d kit made by tools/fetch_dav1d.py, for the AV1 decoder of the movies (default: the "
                        "DAV1D_DIR variable)")
    p.add_argument("--jobs", type=positive, help="compile jobs (default: the CPU count, capped by the free memory)")
    p.add_argument("--xlate", type=Path, help="shader translator (default: the OOT2_XLATE variable, else oot2-xlate on "
                                             "PATH)")
    p.add_argument("--translate-jobs", type=positive,
                   help="translator processes (default: the CPU count, capped by the free memory)")
    p.add_argument("--plain-modules", type=Path,
                   help="folder of plain modules you already have (main, sdk, ...), used instead of decoding")
    p.add_argument("--zbic-archive", type=Path,
                   help="source archive of the ZBIC decoder, instead of downloading it")
    args = ap.parse_args(argv)
    # Ctrl+Break and a terminate request end the run the way Ctrl+C does: the finished steps stay recorded.
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), cancel)
    return prepare(args)


if __name__ == "__main__":
    sys.exit(main())
