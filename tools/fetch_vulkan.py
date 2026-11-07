#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Fetch the Vulkan headers, Volk and glslang the runtime builds with, laid out as a Vulkan SDK folder.

    py -3 tools/fetch_vulkan.py --dest DIR

The runtime's Vulkan back end (runtime/gpu) compiles against the Vulkan headers and Volk, and the build compiles the
runtime's own shaders (runtime/gpu/shaders) with glslang (tools/build_game.py). A machine with the Vulkan SDK has all
three and sets VULKAN_SDK. This script gives a machine without it, such as a continuous-integration runner, the same
pieces at pinned versions; none of them is kept in this repository (THIRD_PARTY_NOTICES.md).

DIR receives Include/vulkan, Include/vk_video, Include/Volk/volk.c and volk.h, Bin/glslang (.exe on Windows) and
Licenses/, so that VULKAN_SDK=DIR serves tools/build_game.py, the check step of oot2 prepare and tools/build_vkdraw.py.
DIR/fetch_vulkan.json records the pins and the SHA-256 of every file written; a DIR whose record matches the pins and
whose files still match their hashes is kept as it is, so a cached DIR costs no download.

Pins: the Vulkan SDK 1.4.357.0's own Vulkan-Headers and Volk (each repository's tag vulkan-sdk-1.4.357.0, fetched by
commit, which git checks), and glslang 16.6.0's release archive for this system, checked against the SHA-256 GitHub
records for it.

Needs git on PATH and network access. Exit codes: 0 done, 1 a download or a check failed, 2 bad command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

WIN = os.name == "nt"
RECORD = "fetch_vulkan.json"


@dataclass(frozen=True)
class Repo:
    name: str
    url: str
    tag: str
    commit: str


@dataclass(frozen=True)
class Archive:
    url: str
    sha256: str


HEADERS = Repo("Vulkan-Headers", "https://github.com/KhronosGroup/Vulkan-Headers", "vulkan-sdk-1.4.357.0",
               "e3b1eec08173d6b825cd3ac88c885a63b621504a")
VOLK = Repo("volk", "https://github.com/zeux/volk", "vulkan-sdk-1.4.357.0", "776893306c5d3b22b6185b5d4a258b81d94572bf")
GLSLANG_VERSION = "16.6.0"
GLSLANG_RELEASES = "https://github.com/KhronosGroup/glslang/releases/download/" + GLSLANG_VERSION
GLSLANG = Archive(f"{GLSLANG_RELEASES}/glslang-{GLSLANG_VERSION}-windows-x86_64-release.zip",
                  "82bf434e69b9bb4829de7e2b4bc2c5e7a7861e53d66cf75e5cc70f5f694a8d9b") if WIN else \
    Archive(f"{GLSLANG_RELEASES}/glslang-{GLSLANG_VERSION}-linux-x86_64-release.tar.gz",
            "2c34071f56ecf39d16233294d3739cf43f981fd39080f2fd9dccb782101191ce")
GLSLANG_EXE = "glslang.exe" if WIN else "glslang"


class FetchError(Exception):
    """A download or a check failed. The message says which."""


def pins() -> dict:
    return {"headers": [HEADERS.tag, HEADERS.commit], "volk": [VOLK.tag, VOLK.commit],
            "glslang": [GLSLANG.url, GLSLANG.sha256]}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_hashes(dest: Path) -> dict[str, str]:
    return {p.relative_to(dest).as_posix(): file_sha256(p) for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name != RECORD}


def is_current(dest: Path) -> bool:
    """dest holds this script's pins, and every file it wrote is still there, unchanged, with nothing added."""
    try:
        record = json.loads((dest / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return record.get("pins") == pins() and record.get("files") == tree_hashes(dest)


def git(*args: str, cwd: Path | None = None) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if done.returncode:
        raise FetchError(f"git {' '.join(args)} failed:\n{(done.stderr or done.stdout).strip()[-800:]}")
    return done.stdout.strip()


def fetch_repo(repo: Repo, work: Path) -> Path:
    """repo at its pinned commit, fetched alone (git verifies every object against the commit's hash)."""
    path = work / repo.name
    git("init", "-q", str(path))
    git("fetch", "-q", "--depth", "1", repo.url, repo.commit, cwd=path)
    git("checkout", "-q", "FETCH_HEAD", cwd=path)
    head = git("rev-parse", "HEAD", cwd=path)
    if head != repo.commit:
        raise FetchError(f"{repo.url}: fetched commit {head}, expected {repo.commit} ({repo.tag})")
    return path


def download(archive: Archive, path: Path) -> None:
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(archive.url, timeout=120) as r, open(path, "wb") as f:
            for block in iter(lambda: r.read(1 << 20), b""):
                h.update(block)
                f.write(block)
    except OSError as e:
        raise FetchError(f"{archive.url}: download failed: {e}") from None
    if h.hexdigest() != archive.sha256:
        raise FetchError(f"{archive.url}: SHA-256 {h.hexdigest()}, expected {archive.sha256}")


def unpack(path: Path, into: Path) -> None:
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                target = (into / name).resolve()
                if not target.is_relative_to(into.resolve()):
                    raise FetchError(f"{path.name}: {name} would leave the extraction folder")
            z.extractall(into)
    else:
        with tarfile.open(path) as t:
            t.extractall(into, filter="data")


def find_glslang(root: Path) -> Path:
    for name in (GLSLANG_EXE, "glslangValidator" + (".exe" if WIN else "")):
        found = sorted(p for p in root.rglob(name) if p.is_file())
        if found:
            return found[0]
    raise FetchError(f"no {GLSLANG_EXE} in glslang {GLSLANG_VERSION}'s archive")


def copy_files(src: Path, dst: Path) -> None:
    """The files of src (links followed), not its folders, into dst."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if p.is_file():
            shutil.copyfile(p, dst / p.name)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FetchError(f"{src} is missing")
    shutil.copytree(src, dst)


def fetch(dest: Path) -> list[str]:
    """Lay out dest from the pins; return what was written, one line per piece. dest must be absent, empty, or a folder
    this script laid out before (it holds the record), which is then replaced."""
    if dest.exists():
        if not dest.is_dir() or (any(dest.iterdir()) and not (dest / RECORD).is_file()):
            raise FetchError(f"{dest} exists and is not a folder this script laid out; give an empty or new folder")
        shutil.rmtree(dest)
    (dest / "Include").mkdir(parents=True)
    (dest / "Bin").mkdir()
    (dest / "Licenses").mkdir()
    lines = []
    with tempfile.TemporaryDirectory(prefix="oot2-fetch-vulkan-") as tmp:
        work = Path(tmp)
        headers = fetch_repo(HEADERS, work)
        copy_tree(headers / "include" / "vulkan", dest / "Include" / "vulkan")
        copy_tree(headers / "include" / "vk_video", dest / "Include" / "vk_video")
        shutil.copy2(headers / "LICENSE.md", dest / "Licenses" / "Vulkan-Headers-LICENSE.md")
        if (headers / "LICENSES").is_dir():
            copy_tree(headers / "LICENSES", dest / "Licenses" / "Vulkan-Headers-LICENSES")
        lines.append(f"Vulkan-Headers {HEADERS.tag} ({HEADERS.commit})")

        volk = fetch_repo(VOLK, work)
        (dest / "Include" / "Volk").mkdir()
        for name in ("volk.c", "volk.h"):
            shutil.copy2(volk / name, dest / "Include" / "Volk" / name)
        shutil.copy2(volk / "LICENSE.md", dest / "Licenses" / "volk-LICENSE.md")
        lines.append(f"Volk {VOLK.tag} ({VOLK.commit})")

        archive = work / GLSLANG.url.rsplit("/", 1)[1]
        download(GLSLANG, archive)
        unpacked = work / "glslang"
        unpack(archive, unpacked)
        found = find_glslang(unpacked)
        # The whole folder of the executable, and a lib folder beside it, in case it loads libraries from there.
        copy_files(found.parent, dest / "Bin")
        if (found.parent.parent / "lib").is_dir():
            copy_files(found.parent.parent / "lib", dest / "lib")
        exe = dest / "Bin" / found.name
        for p in (dest / "Bin").iterdir():
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        licences = sorted(p for p in unpacked.rglob("LICENSE*") if p.is_file())
        if licences:
            shutil.copyfile(licences[0], dest / "Licenses" / f"glslang-{licences[0].name}")
        lines.append(f"glslang {GLSLANG_VERSION} ({GLSLANG.sha256})")
    done = subprocess.run([str(exe), "--version"], capture_output=True, text=True)
    if done.returncode:
        raise FetchError(f"{exe} does not run here (exit {done.returncode}): {(done.stderr or done.stdout)[-400:]}")
    lines.append(done.stdout.strip().splitlines()[0])
    (dest / RECORD).write_text(json.dumps({"pins": pins(), "files": tree_hashes(dest)}, indent=1) + "\n",
                               encoding="utf-8")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, required=True, help="the folder to lay out (VULKAN_SDK then names it)")
    args = ap.parse_args(argv)
    dest = args.dest.resolve()
    if is_current(dest):
        print(f"fetch_vulkan: {dest} already holds the pinned pieces, unchanged")
        return 0
    try:
        for line in fetch(dest):
            print(f"fetch_vulkan: {line}")
    except FetchError as e:
        print(f"fetch_vulkan: {e}", file=sys.stderr)
        return 1
    print(f"fetch_vulkan: {dest} is ready; set VULKAN_SDK to it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
