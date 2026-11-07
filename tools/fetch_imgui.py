#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Fetch Dear ImGui at its pinned version into an ignored folder, every file checked against its SHA-256.

    py -3 tools/fetch_imgui.py [--dest DIR]

The runtime's menu (runtime/ui, research 93 and 106) is built on Dear ImGui, which is C++ and too large to commit: the
content guard refuses a file over 512 KiB and flags a source file over 256 KiB, and imgui.cpp is 957 KiB. So none of
it is kept in this repository (THIRD_PARTY_NOTICES.md). This script lays out the pieces the runtime builds, at one
pinned version, in DIR (default fetched/imgui, which .gitignore and the content guard both name). tools/build_game.py
finds that folder (--imgui, else OOT2_IMGUI, else the default), refuses one that is not exactly the pinned layout, and
compiles imgui.cpp, imgui_draw.cpp, imgui_tables.cpp and imgui_widgets.cpp into the runtime library with the flags and
definitions below. The backends (backends/imgui_impl_vulkan and imgui_impl_sdl3) are laid out and pinned too, for the
present path that will draw the menu, and are not compiled yet.

Who runs it: the release build, once, because the runtime is a prebuilt library (architecture D4, D24c) and a player
never fetches Dear ImGui. Continuous integration and a developer fetch it to build the runtime library from source.

Pin: Dear ImGui 1.92.9b, tag v1.92.9b, released 2026-07-31. Why it is old enough to trust: it is the newest release at
least a month old on 2026-11-07, 99 days old, and it is the quick fix of 1.92.9 itself (the 'b' came six days after
1.92.9), so the release it fixes has had its bug reports. 1.93.0 (2026-10-14) was 24 days old and had no point release
yet. The commit is fetched alone and git checks every object against it; each file's SHA-256 below is the hash of the
blob git stores, not of a checked-out file (a Windows checkout with core.autocrlf rewrites the line endings, and the
hash with them), so the script reads blobs from git's object store and never checks a working tree out. CONTENTS_SHA256
is one hash over the whole set, as tools/prepare/zbic.py computes its own: each path, its size and its SHA-256, in path
order.

To move the pin, set VERSION, TAG and COMMIT and run `py tools/fetch_imgui.py --print-pins COMMIT`: it prints the new
SHA256SUMS table and CONTENTS_SHA256 from the commit's blobs. Then read what changed, rebuild, run the tests, and
update the notice rows and licenses/ in the same commit.

Needs git on PATH and network access. Exit codes: 0 done, 1 a download or a check failed, 2 bad command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "fetched" / "imgui"
RECORD = "fetch_imgui.json"

URL = "https://github.com/ocornut/imgui"
VERSION = "1.92.9b"
TAG = "v1.92.9b"
COMMIT = "f1cc2ae15e53a861a874c3034aae6798fde194ab"

# The SHA-256 of each file as git stores it, in sha256sum's own form (the content guard reads "<hash>  <name>" as a
# labelled hash, and `sha256sum -c` takes it as it is), by path in the repository and in DIR.
SHA256SUMS = """\
173506a2d6f7fb67990d257fb2507f188690eca39060c39469ae7bef43aae2a3  LICENSE.txt
5755e1b8d6ab0d7811a9d7cac0f509878b51fc5ee77e09bd0f235bcb414971e7  imconfig.h
5a1e5128c0305f50b6556a77f71fde35836505883b74c1d6ae44723243f1e8de  imgui.cpp
0d8db1045db01d908853adfd26ae07c5bc5ab4789d4515f6ea34234a69ade0ca  imgui.h
83d30419a8e06a5f0a8692ee6de186f7ecfddecdc33769e76244837e2975b400  imgui_draw.cpp
efba9bccc971cc49ec3da3168968ba412c7ea89745c104a7f5c004c8349bc71f  imgui_internal.h
b5deabe5b569ab712c11b6556562a646bf88327320fc60a451071b2a36498d72  imgui_tables.cpp
a8a0b2f65caa5711467d9e3ca4028fded48329ea65e11ec92a968293dc2aa231  imgui_widgets.cpp
889b396795202d1457560a797a7242e96f6f132d4b88ca2d69be58bf05e1771f  imstb_rectpack.h
a985f5fa0ed97353d493b497961e9eef52082edcd045cf6954b69990ec9d0741  imstb_textedit.h
c51a0f7e7ea760f2366bd3752635ec58e21fccfec4a832501639990ba6ce0528  imstb_truetype.h
306060dffa276b574f0541194e8e079c1685c29c1095602a3972de0bb23171e7  backends/imgui_impl_sdl3.cpp
fa184f17d59d31e986df2e827ef964d7b3c28bcba7789643a4aaaed344cb868e  backends/imgui_impl_sdl3.h
659590d8b74bcbd2ba6612d84a6d8871f97314f04da06a8a05508244adabf086  backends/imgui_impl_vulkan.cpp
351aa104d643f8a575b891d99645389d45d321bcbd33d41270d2ebf0276351f3  backends/imgui_impl_vulkan.h
"""
FILES = {name: digest for digest, name in (line.split("  ", 1) for line in SHA256SUMS.splitlines())}
CONTENTS_SHA256 = "b6594bf2553589eb3911a9f4d176a1411d89bb1731f196cada9cf76266e0946a"

# What tools/build_game.py compiles into the runtime library, and how. The definitions apply to every C++ file of the
# runtime, so that the library's own code and the menu that includes imgui.h agree: no windows.h in imgui.cpp (the
# clipboard, the input method and the file names then come from the platform layer or plain stdio, and the program
# keeps importing the system's C library only), and no code that opens a link in the system shell.
CORE_SOURCES = ("imgui.cpp", "imgui_draw.cpp", "imgui_tables.cpp", "imgui_widgets.cpp")
DEFINES = ("IMGUI_DISABLE_WIN32_FUNCTIONS", "IMGUI_DISABLE_DEFAULT_SHELL_FUNCTIONS")


class FetchError(Exception):
    """A download or a check failed. The message says which."""


def pins() -> dict:
    return {"version": VERSION, "tag": TAG, "commit": COMMIT, "contents_sha256": CONTENTS_SHA256}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contents_sha256(files: dict[str, bytes]) -> str:
    """One hash over a set of files: each path, its size and its own SHA-256, in path order (as zbic.py's)."""
    h = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.encode("utf-8")):
        data = files[path]
        h.update(path.encode("utf-8") + b"\0" + len(data).to_bytes(8, "little") + hashlib.sha256(data).digest())
    return h.hexdigest()


def check_files(files: dict[str, bytes]) -> None:
    """files (path: bytes) is exactly the pinned set, byte for byte. FetchError names the first path that is missing,
    added or different, then the contents hash."""
    for path in FILES:
        if path not in files:
            raise FetchError(f"{path} is missing (Dear ImGui {TAG})")
    for path in sorted(files):
        if path not in FILES:
            raise FetchError(f"{path} is not one of the pinned files (Dear ImGui {TAG})")
    for path, want in FILES.items():
        got = sha256(files[path])
        if got != want:
            raise FetchError(f"{path}: SHA-256 {got}, expected {want} (Dear ImGui {TAG})")
    got = contents_sha256(files)
    if got != CONTENTS_SHA256:
        raise FetchError(f"the pinned files hash to {got} as a whole, expected {CONTENTS_SHA256} (Dear ImGui {TAG})")


def git(*args: str, cwd: Path | None = None, binary: bool = False):
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    if done.returncode:
        text = (done.stderr or done.stdout).decode("utf-8", "replace").strip()[-800:]
        raise FetchError(f"git {' '.join(args)} failed:\n{text}")
    return done.stdout if binary else done.stdout.decode("utf-8", "replace").strip()


def fetch_blobs(commit: str, paths: list[str], work: Path) -> dict[str, bytes]:
    """The blobs of paths at commit, fetched alone (git verifies every object against the commit's hash) and read
    from git's object store: no working tree is checked out, so no line ending is rewritten."""
    repo = work / "imgui"
    git("init", "-q", str(repo))
    git("fetch", "-q", "--depth", "1", URL, commit, cwd=repo)
    head = git("rev-parse", "FETCH_HEAD", cwd=repo)
    if head != commit:
        raise FetchError(f"{URL}: fetched commit {head}, expected {commit}")
    return {p: git("cat-file", "blob", f"{commit}:{p}", cwd=repo, binary=True) for p in paths}


Source = Callable[[], dict[str, bytes]]


def pinned_source() -> dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="oot2-fetch-imgui-") as tmp:
        return fetch_blobs(COMMIT, list(FILES), Path(tmp))


def read_folder(dest: Path) -> dict[str, bytes]:
    """The files of dest by path, except the record."""
    return {p.relative_to(dest).as_posix(): p.read_bytes() for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name != RECORD}


def problems(dest: Path) -> list[str]:
    """Why dest is not the pinned layout laid out by this script, one line each; an empty list when it is."""
    if not dest.is_dir():
        return [f"{dest} is not a folder"]
    try:
        record = json.loads((dest / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [f"{dest / RECORD} is missing or unreadable: fetch with tools/fetch_imgui.py"]
    if record != pins():
        return [f"{dest} holds Dear ImGui {record.get('tag')}, this tree pins {TAG}: fetch it again"]
    try:
        check_files(read_folder(dest))
    except (FetchError, OSError) as e:
        return [str(e)]
    return []


def fetch(dest: Path, source: Source | None = None) -> list[str]:
    """Lay out dest from source() (default: the pinned commit from GitHub), checked file by file first; return what
    was written, one line per piece. dest must be absent, empty, or a folder this script laid out before (it holds the
    record), which is then replaced. Nothing is written to dest until every file has matched."""
    if dest.exists() and (not dest.is_dir() or (any(dest.iterdir()) and not (dest / RECORD).is_file())):
        raise FetchError(f"{dest} exists and is not a folder this script laid out; give an empty or new folder")
    files = (source or pinned_source)()
    check_files(files)
    if dest.exists():
        shutil.rmtree(dest)
    for path, data in files.items():
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (dest / RECORD).write_text(json.dumps(pins(), indent=1) + "\n", encoding="utf-8")
    return [f"Dear ImGui {VERSION} ({TAG}, {COMMIT})", f"{len(files)} files, {sum(map(len, files.values())):,} bytes"]


def print_pins(commit: str) -> None:
    """The SHA256SUMS table and the contents hash for commit, in this file's own form, to paste when the pin moves."""
    with tempfile.TemporaryDirectory(prefix="oot2-fetch-imgui-") as tmp:
        files = fetch_blobs(commit, list(FILES), Path(tmp))
    print('SHA256SUMS = """\\')
    for path in FILES:
        print(f"{sha256(files[path])}  {path}")
    print('"""')
    print(f'CONTENTS_SHA256 = "{contents_sha256(files)}"')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help="the folder to lay out (default fetched/imgui; tools/build_game.py finds it there)")
    ap.add_argument("--print-pins", metavar="COMMIT",
                    help="print the table and contents hash of a commit, to move the pin")
    args = ap.parse_args(argv)
    try:
        if args.print_pins:
            print_pins(args.print_pins)
            return 0
        dest = args.dest.resolve()
        if not problems(dest):
            print(f"fetch_imgui: {dest} already holds Dear ImGui {VERSION}, unchanged")
            return 0
        for line in fetch(dest):
            print(f"fetch_imgui: {line}")
    except (FetchError, OSError) as e:
        print(f"fetch_imgui: {e}", file=sys.stderr)
        return 1
    print(f"fetch_imgui: {dest} is ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
