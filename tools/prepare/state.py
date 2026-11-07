# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""oot2-prepare.json in the output folder: what the check found and which steps finished.

    {"format": 1,
     "inputs": {"dump": {"path", "size", "mtime_ns", "sha256"}},
     "host": {...the check step's probe of the CPU and Vulkan...},
     "steps": {"<step>": {"key": "<fingerprint>", "finished": "<UTC time>", "seconds": 12.3,
                          ...the step's own fields, such as "bytes" written}}}

The file is replaced atomically after each step, so a run that stops, however it
stops, leaves the steps finished before it recorded and nothing half-written. A
file that cannot be read as this shape is refused with a plain message, never
guessed at or overwritten: it may be the only record of an hour of work.

Adapted from TotK Recomp installer/state.py at 0ae47a5 (MIT, Copyright (c) 2026
Valiantsin Dzerakh, licenses/totk-MIT.txt). Changed: the file name, no release or
language fields, and every field of a step record is checked on load.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

NAME = "oot2-prepare.json"
FORMAT = 1


class StateError(Exception):
    """A state file that cannot be used. The message is meant for the player as it is."""

    def __init__(self, path: Path, why: str):
        self.path, self.why = path, why
        super().__init__(
            f"the state file {path} is damaged: {why}. Nothing was run. Delete it to prepare from the start "
            "(every step runs again), or put back a good copy.")


class NewerStateError(StateError):
    def __init__(self, path: Path, found: int):
        self.path, self.why = path, f"format {found}"
        Exception.__init__(
            self, f"the state file {path} was written by a newer oot2 (format {found}; this one reads format "
            f"{FORMAT}). Nothing was run. Use the newer tools, or delete the file to prepare from the start.")


def empty() -> dict:
    return {"format": FORMAT, "inputs": {}, "steps": {}}


def path_of(dest: Path) -> Path:
    return Path(dest) / NAME


def load(dest: Path) -> dict | None:
    """The state of dest, or None when the folder has none. StateError for a file that is not this shape."""
    path = path_of(dest)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except OSError as e:
        raise StateError(path, f"it cannot be read ({e.strerror or e})") from None
    except UnicodeDecodeError:
        raise StateError(path, "it is not UTF-8 text") from None
    except ValueError as e:
        raise StateError(path, f"it is not valid JSON ({e})") from None
    if not isinstance(data, dict):
        raise StateError(path, "it holds a JSON value that is not an object")
    found = data.get("format")
    if type(found) is not int or found < 1:
        raise StateError(path, "its \"format\" is missing or not a positive whole number")
    if found > FORMAT:
        raise NewerStateError(path, found)
    for field in ("inputs", "steps"):
        if not isinstance(data.get(field), dict):
            raise StateError(path, f"its \"{field}\" is missing or not an object")
    if "host" in data and not isinstance(data["host"], dict):
        raise StateError(path, "its \"host\" is not an object")
    for step, record in data["steps"].items():
        if not isinstance(record, dict):
            raise StateError(path, f"the record of step \"{step}\" is not an object")
        if not isinstance(record.get("key"), str) or not record["key"]:
            raise StateError(path, f"the record of step \"{step}\" has no key")
        if not isinstance(record.get("finished"), str):
            raise StateError(path, f"the record of step \"{step}\" has no finishing time")
        if type(record.get("seconds")) not in (int, float):
            raise StateError(path, f"the record of step \"{step}\" has no duration")
    return data


def save(dest: Path, data: dict) -> None:
    """Write the state through a temporary file and a rename, so the file on disk is always a whole one."""
    path = path_of(dest)
    tmp = path.with_name(NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
