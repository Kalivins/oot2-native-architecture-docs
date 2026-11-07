#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Make the synthetic AV1 clips of tests/fixtures/av1 from generated pictures (never from the game's movies).

    py -3 make_fixtures.py --rav1e PATH/rav1e.exe [--out DIR]

Each clip is a few generated frames of 8-bit 4:2:0 video, written as a Y4M file in a temporary folder and encoded by rav1e
0.8.1 (BSD-2-Clause with the AOM patent licence; its release binary SHA-256 is in README.md) into an IVF file. The pictures
are three distinct ramps, so a swapped plane, a stride off by one row or a shifted picture changes pixels:

    Y(x, y, f) = (x * 255 // (w - 1) + 8 * f) % 256            horizontal ramp, shifted by 8 for each frame
    U(x, y, f) = (y * 255 // (ch - 1) + 8 * f) % 256            vertical ramp (ch = chroma height)
    V(x, y, f) = ((x + y) * 255 // (cw + ch - 2) + 8 * f) % 256  diagonal ramp (cw = chroma width)

The same three formulas are in tests/runtime/av1_test.c, which decodes the clips and compares. The encoder command line is
printed and kept in README.md. A clip is encoded at quantizer 40, speed 10, one key frame (the first) and inter frames after.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

CLIPS = {"clip_200x126": (200, 126, 4), "clip_201x125": (201, 125, 3), "clip_1920x1080": (1920, 1080, 2)}


def planes(w, h, f):
    cw, ch = (w + 1) // 2, (h + 1) // 2
    y = bytes((x * 255 // (w - 1) + 8 * f) % 256 for _ in range(h) for x in range(w))
    u = bytes((r * 255 // (ch - 1) + 8 * f) % 256 for r in range(ch) for _ in range(cw))
    v = bytes(((c + r) * 255 // (cw + ch - 2) + 8 * f) % 256 for r in range(ch) for c in range(cw))
    return y, u, v


def write_y4m(path, w, h, n):
    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{w} H{h} F60:1 Ip A1:1 C420jpeg\n".encode())
        for f in range(n):
            fh.write(b"FRAME\n")
            for p in planes(w, h, f):
                fh.write(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rav1e", required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="oot2-av1-fixtures-") as tmp:
        for name, (w, h, n) in CLIPS.items():
            y4m = Path(tmp) / f"{name}.y4m"
            write_y4m(y4m, w, h, n)
            cmd = [args.rav1e, str(y4m), "-y", "--quantizer", "40", "--speed", "10", "--keyint", "30", "--limit", str(n),
                   "--threads", "1", "--no-scene-detection", "-o", str(args.out / f"{name}.ivf")]
            print(" ".join(["rav1e", f"{name}.y4m"] + cmd[2:-1] + [f"{name}.ivf"]))
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
