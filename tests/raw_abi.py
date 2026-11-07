# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The raw-access table's ABI as the two sides write it, read from their sources.

The translator's side is RawTableAbi.cs (the helper's constants and the layout in its comment), the runtime's side
is runtime/include/oot2/raw_table.h. Both are plain constants; this module reads them with regular expressions, so the
tests need neither the translator nor a C compiler. Names map by dropping the RAW_ prefix and joining the words
(RAW_PUSH_SPAN_BASE is PushSpanBase).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CS = ROOT / "xlate/third_party/ryujinx/src/Ryujinx.Graphics.Shader/RawTableAbi.cs"
HEADER = ROOT / "runtime/include/oot2/raw_table.h"
UNI_LAYOUT = ROOT / "xlate/host/UniLayout.cs"


def camel(c_name: str) -> str:
    return "".join(part.capitalize() for part in c_name.removeprefix("RAW_").split("_"))


def header_constants() -> dict[str, int]:
    text = HEADER.read_text(encoding="utf-8")
    return {camel(m.group(1)): int(m.group(2)) for m in re.finditer(r"^#define\s+(RAW_\w+)\s+(\d+)\s*$", text, re.M)}


def cs_constants() -> dict[str, int]:
    text = CS.read_text(encoding="utf-8")
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"^\s*public const int (\w+) = (\d+);", text, re.M)}


def uni_slots_offset() -> int:
    m = re.search(r"public const int SlotsOffset = (\d+);", UNI_LAYOUT.read_text(encoding="utf-8"))
    assert m, "UniLayout.SlotsOffset not found"
    return int(m.group(1))


# The constants the tests use, from the C header (the runtime's side).
A = header_constants()
