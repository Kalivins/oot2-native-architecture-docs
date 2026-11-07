# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The ZBIC segment decoder, fetched and built on the player's machine (architecture D2a).

The game's main module stores its segments as ZBIC frames (NSO flag bit 7): zstd with its own frame magic and coded
entropy tables. No code that decodes them is in this repository. The decoder is the third-party zstd fork
hexkyz/zstd, pinned to one commit: prepare downloads that commit's source archive, checks it, builds the
decompression half of the library with the player's C compiler as a shared library, and calls ZSTD_decompress through
ctypes, only to turn the player's own module into a plain one. A player who already has plain modules gives them
with --plain-modules and nothing is fetched (modules.py).

The check is a SHA-256 over the archive's contents (each regular file's path inside the archive, its size and its
own SHA-256, in path order), not over the archive's bytes: GitHub builds source archives on request and their
compressed bytes have changed before while the files inside did not. A pinned commit fixes the files, so the
contents hash is what this file bakes. Only after it matches is anything written to disk.

    hexkyz/zstd  https://github.com/hexkyz/zstd  BSD-3-Clause (or GPL-2.0, at the user's choice), Meta Platforms
    commit       02a8f10c6f4acfcfba2e31e034fbae9f07938f96 ("ZBIC improvements", 2026-04-02)

Built files: lib/common/{debug,entropy_common,error_private,fse_decompress,xxhash,zstd_common}.c and
lib/decompress/{huf_decompress,zstd_ddict,zstd_decompress,zstd_decompress_block}.c with ZSTD_ZBIC_SUPPORT=1 (the
fork's switch) and the assembly fast path turned off, as oot-pc-recomp's tools/zbic driver does with the same fork.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from .runner import PrepareError

COMMIT = "02a8f10c6f4acfcfba2e31e034fbae9f07938f96"
URL = f"https://codeload.github.com/hexkyz/zstd/tar.gz/{COMMIT}"
CONTENTS_SHA256 = "3db880ae8afd07efb955100d4bad357f97a93389a24096a291651705318de296"
SOURCES = tuple(f"lib/common/{n}.c" for n in ("debug", "entropy_common", "error_private", "fse_decompress", "xxhash",
                                               "zstd_common")) + \
    tuple(f"lib/decompress/{n}.c" for n in ("huf_decompress", "zstd_ddict", "zstd_decompress", "zstd_decompress_block"))
KEEP = ("lib/common/", "lib/decompress/", "lib/zstd.h", "lib/zstd_errors.h", "LICENSE", "COPYING")
DEFINES = ("-DZSTD_ZBIC_SUPPORT=1", "-DZSTD_DISABLE_ASM=1", "-DZSTD_LEGACY_SUPPORT=0") +     (("-DZSTD_DLL_EXPORT=1",) if os.name == "nt" else ())
LIBRARY = "zbic.dll" if os.name == "nt" else "libzbic.so"
DOWNLOAD_TIMEOUT = 60
MAX_ARCHIVE = 64 << 20


def contents_sha256(archive: bytes) -> tuple[str, dict[str, bytes]]:
    """The contents hash of a .tar.gz source archive and its regular files by path (the top folder removed)."""
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for m in tar:
                if not m.isreg():
                    continue
                parts = m.name.split("/", 1)
                if len(parts) != 2 or not parts[1]:
                    continue
                f = tar.extractfile(m)
                files[parts[1]] = f.read() if f is not None else b""
    except (tarfile.TarError, OSError, EOFError) as e:
        raise PrepareError(f"the ZBIC decoder's source archive cannot be read ({e})") from None
    h = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.encode("utf-8")):
        data = files[path]
        h.update(path.encode("utf-8") + b"\0" + len(data).to_bytes(8, "little") + hashlib.sha256(data).digest())
    return h.hexdigest(), files


def download(url: str = URL) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as r:
            data = r.read(MAX_ARCHIVE + 1)
    except (urllib.error.URLError, OSError) as e:
        raise PrepareError(
            f"the ZBIC decoder could not be downloaded from {url} ({getattr(e, 'reason', e)}). The game's main "
            "module needs it. Download that archive yourself and give it with --zbic-archive FILE, or give plain "
            "modules with --plain-modules DIR.") from None
    if len(data) > MAX_ARCHIVE:
        raise PrepareError(f"the archive at {url} is larger than {MAX_ARCHIVE >> 20} MiB; it is not the decoder's")
    return data


def unpack(files: dict[str, bytes], out: Path) -> None:
    """Write the files the build needs under out (only paths this module names, never one from the archive)."""
    for path, data in files.items():
        if not path.startswith(KEEP):
            continue
        parts = path.split("/")
        if any(p in ("", ".", "..") or ":" in p or "\\" in p for p in parts):
            raise PrepareError(f"the ZBIC decoder's archive holds an unsafe path {path!r}")
        target = out.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def build(cc: str, src: Path, out: Path) -> Path:
    """Build the decompression half of the library at src as a shared library in out. Returns its path."""
    lib = out / LIBRARY
    cmd = [cc, "-O2", "-shared", "-std=c11", *DEFINES, "-I", str(src / "lib"), "-I", str(src / "lib" / "common"),
           *(str(src / s) for s in SOURCES), "-o", str(lib)]
    if os.name != "nt":
        cmd[1:1] = ["-fPIC"]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode or not lib.is_file():
        detail = (r.stderr or r.stdout).strip()[-800:]
        raise PrepareError(f"the C compiler {cc} could not build the ZBIC decoder:\n{detail}")
    return lib


class Decoder:
    """ZSTD_decompress of a built library, through ctypes."""

    def __init__(self, lib: Path):
        self.path = lib
        try:
            self.dll = ctypes.CDLL(str(lib))
        except OSError as e:
            raise PrepareError(f"the ZBIC decoder {lib} cannot be loaded ({e})") from None
        self.dll.ZSTD_decompress.restype = ctypes.c_size_t
        self.dll.ZSTD_decompress.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
        self.dll.ZSTD_isError.restype = ctypes.c_uint
        self.dll.ZSTD_isError.argtypes = (ctypes.c_size_t,)
        self.dll.ZSTD_getErrorName.restype = ctypes.c_char_p
        self.dll.ZSTD_getErrorName.argtypes = (ctypes.c_size_t,)

    def __call__(self, data: bytes, size: int) -> bytes:
        """Decode one ZBIC frame to exactly size bytes. Raises ValueError otherwise."""
        out = ctypes.create_string_buffer(max(size, 1))
        src = ctypes.create_string_buffer(data, len(data))
        got = self.dll.ZSTD_decompress(out, size, src, len(data))
        if self.dll.ZSTD_isError(got):
            raise ValueError(self.dll.ZSTD_getErrorName(got).decode("ascii", "replace"))
        if got != size:
            raise ValueError(f"decodes to {got:,} bytes, the header says {size:,}")
        return out.raw[:size]


def prepare_decoder(cc: str, work: Path, rep, archive: Path | None = None, expected: str | None = None,
                    url: str = URL) -> Decoder:
    """Fetch (or read archive), check, unpack and build the decoder under work, reusing an earlier build of the same
    contents with the same compiler. expected overrides the baked contents hash (tests only)."""
    expected = expected or CONTENTS_SHA256
    work.mkdir(parents=True, exist_ok=True)
    stamp_text = f"{expected}\n{cc}\n{' '.join(DEFINES)}\n"
    stamp, lib = work / "built.txt", work / LIBRARY
    if lib.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8") == stamp_text:
        rep.say(f"modules: ZBIC decoder already built in {work}")
        return Decoder(lib)
    if archive is not None:
        rep.say(f"modules: ZBIC decoder source from {archive}")
        data = Path(archive).read_bytes()
    else:
        rep.say(f"modules: downloading the ZBIC decoder (hexkyz/zstd {COMMIT[:7]}) from {url}")
        data = download(url)
    got, files = contents_sha256(data)
    if got != expected:
        raise PrepareError(
            f"the ZBIC decoder's source archive ({len(data):,} bytes) is not the pinned one: its contents hash is "
            f"{got}, the tools expect {expected}. Nothing was built. Give the archive of hexkyz/zstd at {COMMIT} "
            "with --zbic-archive, or plain modules with --plain-modules.")
    src = work / "src"
    unpack(files, src)
    stamp.unlink(missing_ok=True)
    rep.say(f"modules: building the ZBIC decoder with {cc}")
    build(cc, src, work)
    stamp.write_text(stamp_text, encoding="utf-8")
    return Decoder(lib)


if __name__ == "__main__":
    # Print the contents hash of an archive, to pin a new commit: python -m prepare.zbic ARCHIVE
    print(contents_sha256(Path(sys.argv[1]).read_bytes())[0])
