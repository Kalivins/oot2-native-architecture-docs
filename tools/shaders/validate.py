# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Validate translated SPIR-V with spirv-val, and summarise timings.

spirv-val comes from the Vulkan SDK (Khronos SPIRV-Tools). Each module is
validated for the Vulkan 1.3 environment with the default (strict) block layout
rules, on a thread pool of child processes. A module that fails only on layout
can be checked again with relaxed rules (--scalar-block-layout), which the
device supports, to separate layout from other failures.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_ARGS = ("--target-env", "vulkan1.3")
RELAXED_ARGS = ("--target-env", "vulkan1.3", "--scalar-block-layout")


@dataclass
class Validation:
    key: str
    ok: bool
    message: str = ""


def find_spirv_val(explicit: str | None = None) -> str | None:
    """spirv-val from --spirv-val, VULKAN_SDK, or PATH."""
    if explicit:
        return explicit if Path(explicit).exists() else None
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for name in ("spirv-val.exe", "spirv-val"):
            cand = Path(sdk) / "Bin" / name
            if cand.exists():
                return str(cand)
            cand = Path(sdk) / "bin" / name
            if cand.exists():
                return str(cand)
    return shutil.which("spirv-val")


def first_error(text: str) -> str:
    """The first line spirv-val prints for a failure, which names the rule."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("error:"):
            return line[:400]
    return (text.strip().splitlines() or ["no message"])[0][:400]


def validate_one(tool: str, key: str, spv: Path, args: Sequence[str] = DEFAULT_ARGS) -> Validation:
    done = subprocess.run([tool, *args, str(spv)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if done.returncode == 0:
        return Validation(key, True)
    return Validation(key, False, first_error(done.stdout + "\n" + done.stderr))


def validate_all(tool: str, items: Iterable[tuple[str, Path]], workers: int,
                 args: Sequence[str] = DEFAULT_ARGS) -> list[Validation]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(validate_one, tool, key, spv, args) for key, spv in items]
        return [f.result() for f in futures]


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-q * len(ordered) // 100))))
    return float(ordered[rank - 1])


def timing_summary(micros: Sequence[int]) -> dict[str, float]:
    """p50, p99, max and sum in milliseconds."""
    ms = [m / 1000.0 for m in micros]
    return {"n": len(ms), "p50_ms": percentile(ms, 50), "p99_ms": percentile(ms, 99),
            "max_ms": max(ms) if ms else 0.0, "sum_s": sum(ms) / 1000.0}
