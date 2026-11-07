# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The Vulkan back end's replay on the desktop (ticket 63): the draws of two spikes, recorded through the NVN2 entry
points by tests/runtime/vk_replay_draw.c and replayed on the device, against the CPU's evaluation of the same programs.

Never in continuous integration: the programs, their keys, the banks and the textures are the game's and stay in the work
folder, which this test writes under OOT2_T63_WORK and nowhere else. It needs

  OOT2_T63_PACK      the player's shader pack (shader-base.bin)
  OOT2_T63_PREPARE   the player's prepare folder: its oot2-prepare.json names the game image and romfs/romfs-index.bin
                     indexes it; each pair's control sections and code areas are read from the game's own archives, through
                     the split's scanner, because the split keeps neither the control section (the job key hashes it) nor
                     the whole code area
  OOT2_T63_WORK      the folder this test writes: the programs read from the archives, the draw files, the images
  OOT2_T63_S2_SPEC   spike S2's spec.json (research 19)
  OOT2_T63_T23_SPEC  ticket 23's spec23.json (research 23); its spv_dir holds the schema 2 descriptions of both pairs

and nvdisasm (CUDA 11.8 or later, on PATH or under CUDA_PATH) for the reference. The checks that need a device skip without
a usable one, unless OOT2_REQUIRE_GPU is set; the rest of the module (its helpers on synthetic data) runs everywhere.

The checks, each run under core validation (OOT2_VK_VALIDATION=on) and under synchronization validation (sync), which must
report nothing:

- S2's three scenarios and ticket 23's two: the replayed image matches tools/firstdraw's own CPU evaluation on every
  compared pixel, at its tolerance and with its rule for edge pixels (first_draw.compare). The replay flips Y through a
  negative viewport height (its contract: clip y = +1 lands on memory row 0, and a pixel program that reads its window
  position reads the memory row), so the reference runs in that space: the clip y its vertex program writes is negated
  before the triangles are set up (render_in_replay_space). That is the Vulkan-convention image with its rows reversed
  for a pair that never reads its window y, and the only right one for a pair that does (ticket 23's pixel program takes
  an inverse screen size from a bank), which reversing the rows of S2's image would get wrong.
- Counter-runs, each of which must fail the comparison: the uniform snapshot off (--no-snapshot) while the program writes
  the uniform buffers right after the submission (--rewrite-uniform; host imports, so that the snapshot is what keeps
  the frame whole); S2's two (research 19 answer 4): texture coordinates swapped, perspective and screen-linear
  interpolation swapped, in the reference; the origin flip removed (--no-flip).
- The viewport test: S2's pair drawn with its vertices moved into the upper half of clip space, under NVN's defaults
  (NEAR_IS_MINUS_W, LOWER_LEFT) and a depth range of 0.25 to 0.75 with a depth target. The primitive must land in the
  upper half of the memory rows (row 0 first), where the reference puts it in the replay's space, and its depth must
  equal near + (far - near) * ((z / w + 1) / 2) at every covered pixel. How the row rule was checked: nothing here knows
  the console's own memory layout. The replay's contract (runtime/gpu/vk_replay.c: a viewport of height -h from its
  bottom edge) is what the test pins down, and the rule it follows is the console emulators' y_negate = origin !=
  UpperLeft for the lower-left origin and LAS Recomp's unconditional negative-height viewport (refs/lasrecomp
  runtime/nvn/vulkan/exec.c set_viewport) for both. The replay flips for UPPER_LEFT too, so the upper-left origin is
  not asserted here. With the flip removed the primitive lands in the lower half, and with the device's depth mode
  changed to NEAR_IS_ZERO its depth is no longer the expected one: each is a counter-run the test says so of.

Run (on the desktop, from the repository root): python -m unittest discover -s tests -p test_runtime_vk_replay_draw.py -v
"""

from __future__ import annotations

import atexit
import copy
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_runtime_gpu import EXIT_FATAL, NO_VULKAN, ROOT, build_program, case_env, clang_or_skip, fatal_lines, vulkan_or_skip

sys.path.insert(0, str(ROOT / "tools" / "firstdraw"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))
import first_draw as fd  # noqa: E402
import reference as ref  # noqa: E402

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_replay_draw.c"
LEVEL = "-O1"
PACK = os.environ.get("OOT2_T63_PACK")
PREPARE = os.environ.get("OOT2_T63_PREPARE")
WORK = os.environ.get("OOT2_T63_WORK")
S2_SPEC = os.environ.get("OOT2_T63_S2_SPEC")
T23_SPEC = os.environ.get("OOT2_T63_T23_SPEC")
HAVE_INPUTS = all((PACK, PREPARE, WORK, S2_SPEC, T23_SPEC))
NO_INPUTS = "OOT2_T63_PACK, OOT2_T63_PREPARE, OOT2_T63_WORK, OOT2_T63_S2_SPEC and OOT2_T63_T23_SPEC name the desktop's inputs"
VALIDATIONS = ("on", "sync")
ATTEMPTS = 4           # a counter-run that races the GPU (the write after the submission) is tried this often
EDGE_ROWS = 1          # rows of slack around the analytic row range of the viewport test
DEPTH_ATOL = 2e-5      # 32-bit depth interpolation against the CPU's double computation
NVN_DRAW_TRIANGLES = 4
STORAGE_BASE = {"vs": 0x110, "ps": 0x510}   # row 0 of each stage's storage table in c[0] (oot2/vk_nvn.h)
STORAGE_ROWS = 16
ARCHIVE_PROGRAMS = "archive-programs"   # control sections and code areas read from the game's archives, under WORK/t63
USER_BANK_FIRST = 3    # NVN uniform index r is constant bank c[3 + r]
BANK_BYTES = fd.BANK_BYTES
NEAR_IS_MINUS_W, NEAR_IS_ZERO = "near_is_minus_w", "near_is_zero"


class InputError(Exception):
    pass


# ---- helpers on plain data (no device, no game bytes) --------------------------------------------------------------

def covered_rows(pixels: list, clear: tuple, width: int, height: int) -> list[int]:
    """The memory rows that hold a pixel that is not the clear colour, ascending."""
    return sorted({k // width for k in range(width * height) if tuple(pixels[k]) != clear})


def expand_vertices(spec: dict, vs_res: dict) -> dict:
    """The spec with a vertex stream in the form first_draw reads: a value for every component of every vertex input the
    description lists. A spec that gives an input fewer components than its module declares (S2's second input is two
    components of a vec4) is filled as Vulkan and the reference fill a missing component: 0, 0, 0, 1."""
    out = copy.deepcopy(spec)
    stream = out["vertex_stream"]
    given = stream.pop("attributes", None)
    inputs = vs_res.get("vertex_inputs")
    if inputs is None:
        raise InputError("the vertex description lists no vertex_inputs")
    if given is None:
        return out
    by_location = {a["location"]: a["components"] for a in given}
    vertices = []
    for values in stream["vertices"]:
        flat, at = [], 0
        for v in inputs:
            n = by_location.get(v["location"], v["components"])
            part = list(values[at:at + n])
            if len(part) != n:
                raise InputError(f"a vertex has too few values for input {v['location']}")
            at += n
            flat += part + [(0.0, 0.0, 0.0, 1.0)[c] for c in range(n, v["components"])]
        if at != len(values):
            raise InputError(f"a vertex has {len(values)} values, its attributes take {at}")
        vertices.append(flat)
    stream["vertices"] = vertices
    return out


def storage_rows(stage: str, sbufs: list[dict], bank0: bytes, entries: dict) -> list[tuple[int, int, int]]:
    """(row, address, size) of each storage buffer a module reads, from the spec's words of c[0].

    The replay writes c[0] itself from BindStorageBuffer, row r at STORAGE_BASE + 16 r: the spec's entries for bank 0 must
    be those rows' words and nothing else, or the reference would read bytes the replay never writes. A buffer the spec
    leaves empty (address or size 0) is not bound."""
    base = STORAGE_BASE[stage]
    words = [(base + 16 * r, base + 16 * r + 12) for r in range(STORAGE_ROWS)]
    for off_text, value in entries.items():
        off = int(off_text, 0)
        size = 4 * len(value["f32"]) if "f32" in value else 4 * len(value["u32"]) if "u32" in value else 8
        if not any(lo <= off and off + size <= hi for lo, hi in words):
            raise InputError(f"the spec writes c[0] at {off:#x} in the {stage} bank, outside the storage table the replay owns")
    rows = []
    for s in sbufs:
        off = s["address_byte_offset"]
        if s["cb_slot"] != 0 or s["role"] != "guest" or (off - base) % 16 or not 0 <= (off - base) // 16 < STORAGE_ROWS:
            raise InputError(f"a storage buffer of the {stage} module is not a row of the driver bank's table: {s}")
        addr, size = struct.unpack_from("<Q", bank0, off)[0], struct.unpack_from("<I", bank0, off + 8)[0]
        if addr and size:
            rows.append(((off - base) // 16, addr, size))
    return rows


def depth_window(zn: float, mode: str) -> float:
    """The window depth of an NDC z/w before the depth range: (z/w + 1) / 2 under NEAR_IS_MINUS_W, z/w itself otherwise."""
    return (zn + 1.0) / 2.0 if mode == NEAR_IS_MINUS_W else zn


def window_vertices(clip: list[tuple], width: int, height: int, flip_y: bool, subpixel_bits: int = 8) -> list[tuple]:
    """(x, y, z/w, w) in window space for clip positions, positions snapped as the reference snaps them: Vulkan's
    (clip y = +1 at the last row), or with flip_y the replay's (clip y = +1 at row 0, as a negative-height viewport gives)."""
    q = 1 << subpixel_bits
    sign = -1.0 if flip_y else 1.0
    out = []
    for x, y, z, w in clip:
        out.append((round((x / w + 1) * width / 2 * q) / q, round((sign * y / w + 1) * height / 2 * q) / q, z / w, w))
    return out


def expected_depth(clip: list[tuple], triangles: list[tuple], kinds: list[str], width: int, height: int, near: float,
                   far: float, mode: str, flip_y: bool = True, margin: float = 0.02) -> list:
    """The depth at each pixel the reference marks as covered (in the reference's row order), None elsewhere.

    The hardware interpolates z/w linearly in window space; the depth range then maps window depth w to
    near + (far - near) * w. A pixel within `margin` pixels of an edge is not asked for (the reference skips it too).
    kinds and the result are in the space of clip, whichever flip_y says."""
    screen = window_vertices(clip, width, height, flip_y)
    out: list = [None] * (width * height)
    for k in range(width * height):
        if kinds[k] != ref.COLOR:
            continue
        cx, cy = k % width + 0.5, k // width + 0.5
        for tri in triangles:
            p = [screen[i] for i in tri]
            area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1])
            if area == 0:
                continue
            b, dist = ref._barycentrics(p, area, cx, cy)
            if min(dist) >= margin:
                zn = sum(b[i] * p[i][2] for i in range(3))
                out[k] = near + (far - near) * depth_window(zn, mode)
                break
    return out


def word_float(word: int) -> float:
    """A 32-bit register word as a float (the reference's vertex outputs are words)."""
    return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]


def parse_f32(path: Path, count: int) -> list[tuple]:
    data = path.read_bytes()
    if len(data) != 16 * count:
        raise InputError(f"{path} holds {len(data)} bytes, {16 * count} were expected")
    return list(struct.iter_unpack("<4f", data))


def parse_depth(path: Path, count: int) -> list[float]:
    data = path.read_bytes()
    if len(data) != 4 * count:
        raise InputError(f"{path} holds {len(data)} bytes, {4 * count} were expected")
    return [v for (v,) in struct.iter_unpack("<f", data)]


# ---- a pair, from a spec -------------------------------------------------------------------------------------------

class Pair:
    """A spec's two programs: the CPU's decoded code and the descriptions (for the reference), and the control section
    and code area of each as the game's archive holds them (for the replay)."""

    def __init__(self, spec_path: str, spv_dir: str, work: Path):
        self.spec_path = Path(spec_path)
        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        spec["spv_dir"] = spv_dir   # the schema 2 descriptions, whatever folder the spec was written against
        nvdisasm = fd.find_nvdisasm()
        if not nvdisasm:
            raise unittest.SkipTest("nvdisasm (CUDA toolkit 11.8 or later) is needed for the reference")
        self.work = work
        self.keys = {"vs": spec["vs"], "ps": spec["ps"]}
        wanted = set(self.keys.values())
        self.jobs = {}
        with open(spec["jobs"], encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                if j["key"] in wanted:
                    self.jobs[j["key"]] = j
        self.progs, self.res = {}, {}
        for st, key in self.keys.items():
            prog, sph = fd.load_program(self.jobs, key, work, nvdisasm)[:2]
            self.res[st] = json.loads((Path(spv_dir) / f"{key}.res.json").read_text(encoding="utf-8"))
            if self.res[st]["stage"] != st:
                raise InputError(f"{key} is a {self.res[st]['stage']} program, not {st}")
            c1 = bytes.fromhex(self.jobs[key].get("c1", ""))
            self.progs[st] = {"prog": prog, "sph": sph, "spv": None, "c1": c1}
        self.spec = expand_vertices(spec, self.res["vs"])
        self._files: dict[str, tuple[str, str]] | None = None

    def program_files(self) -> dict[str, tuple[str, str]]:
        """Each stage's control section and code area, written under the work folder's programs folder: file names
        relative to a case folder (programs/ is beside the case folders)."""
        if self._files is None:
            self._files = extract_programs(self.spec, self.jobs, self.work)
        return self._files


def extract_programs(spec: dict, jobs: dict, work: Path) -> dict[str, tuple[str, str]]:
    """Read each stage's control section and whole code area from the game's archive and check them against the split.

    The archive is the one the split's programs.tsv names for the job (its first label); the split's scanner walks it, and
    the program whose key (split.ctl_key: the first 0x48 control bytes and the code area) is the job's is the one used."""
    sys.path.insert(0, str(ROOT / "tools"))
    from prepare import translate   # noqa: PLC0415  (imports the whole prepare package: only the desktop needs it)
    import split                    # noqa: PLC0415

    prepare = Path(PREPARE)
    record = json.loads((prepare / "oot2-prepare.json").read_text(encoding="utf-8"))["steps"]["extract"]
    image = translate.RomfsImage(prepare / translate.ROMFS / translate.INDEX, translate.romfs_image(prepare, record))
    tsv = Path(spec["jobs"]).with_name("programs.tsv")
    labels = {}
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            key = line.split("\t", 1)[0]
            if key in (spec["vs"], spec["ps"]):
                labels[key] = line.rstrip("\n").split("\t")[5]
    out_dir = work / ARCHIVE_PROGRAMS   # beside first_draw's own programs folder, whose .code files are the instructions only
    out_dir.mkdir(parents=True, exist_ok=True)
    problems: list = []
    scanner, decompressor = split.Scanner(problems, "walk"), split.Decompressor(image)
    found: dict[str, tuple[bytes, bytes]] = {}
    wanted = {key.split(".")[0]: key for key in (spec["vs"], spec["ps"])}
    try:
        for archive in sorted({lab.split("#")[0].lstrip("/") for lab in labels.values()}):
            scanner.file = archive
            data = image.read(archive)
            if data[:4] == split.ZSTD_MAGIC:
                data = decompressor.decompress(data)
            view = memoryview(data)
            for prog in scanner.scan(view, "/" + archive):
                key = wanted.get(split.ctl_key(prog.ctl, prog.code))
                if key and key not in found and prog.stage == jobs[key]["stage"]:
                    found[key] = (bytes(prog.ctl), bytes(prog.code))
            view = None
    finally:
        image.close()
    files = {}
    for st, key in (("vs", spec["vs"]), ("ps", spec["ps"])):
        if key not in found:
            raise InputError(f"the program of job {key} is not in its archive {labels.get(key)}")
        ctl, code = found[key]
        job = jobs[key]
        with open(job["code"], "rb") as f:
            f.seek(job["code_offset"])
            stored = f.read(job["code_length"])
        if struct.unpack_from("<I", ctl, 0x44)[0] != len(code) or code[:len(stored)] != stored:
            raise InputError(f"the archive's program for {key} is not the split's (code length or bytes differ)")
        (out_dir / f"{key}.ctl").write_bytes(ctl)
        (out_dir / f"{key}.code").write_bytes(code)
        files[st] = (f"../{ARCHIVE_PROGRAMS}/{key}.ctl", f"../{ARCHIVE_PROGRAMS}/{key}.code")
    return files


# ---- one draw -----------------------------------------------------------------------------------------------------

class Case:
    """A scenario of a pair: the banks, memory, vertices and textures it draws with."""

    def __init__(self, pair: Pair, spec: dict, scen: dict, name: str):
        self.pair, self.spec, self.scen, self.name = pair, spec, scen, name
        self.width, self.height = spec["size"]
        self.clear = tuple(float(v) for v in spec["clear"])
        self.banks = fd.merge_banks(spec.get("banks", {}), scen.get("banks", {}))
        self.mem = fd.Memory(spec.get("global", []) + scen.get("global", []))
        self.bank_data = fd.stage_banks(self.banks, pair.progs)
        self.vertices, self.vblob, self.attr_lines = fd.vertex_stream(spec, pair.res["vs"])
        self.triangles = [tuple(t) for t in spec["triangles"]]
        self.order = [i for t in self.triangles for i in t]
        self.textures = {st: {h: fd.make_texture(t) for h, t in by.items()} for st, by in fd.texture_specs(spec).items()}
        # what the harness cannot draw is refused here, as first_draw refuses it
        fd.bind(pair.res, self.bank_data, self.mem, self.textures, self.width, self.height)

    def with_vertices(self, vertices: list[list[float]], name: str) -> "Case":
        """The same draw with other vertex values (one list per vertex, a value for every component of every input)."""
        other = copy.copy(self)
        other.spec = copy.deepcopy(self.spec)
        other.spec["vertex_stream"]["vertices"] = vertices
        other.name = name
        other.vertices, other.vblob, other.attr_lines = fd.vertex_stream(other.spec, self.pair.res["vs"])
        return other

    def cbank(self, stage: str):
        def read(bank: int, off: int) -> int:
            data = self.bank_data[stage].get(bank)
            return struct.unpack_from("<I", data, off)[0] if data is not None else 0
        return read

    def draw(self, swap_uv: bool = False, swap_interpolation: bool = False) -> ref.Draw:
        """first_draw.run_scenario's CPU half: the vertices in triangle order (the GPU's vertex index is the position in the
        draw), the programs' own code."""
        drawn = [self.vertices[i] for i in self.order]
        tris = [(3 * k, 3 * k + 1, 3 * k + 2) for k in range(len(self.triangles))]
        p = self.pair.progs
        return ref.Draw(self.width, self.height, p["vs"]["prog"], p["ps"]["prog"], p["ps"]["sph"], drawn, tris,
                        self.cbank("vs"), self.cbank("ps"), self.mem.load, self.textures["ps"],
                        vs_textures=self.textures["vs"], swap_uv=swap_uv, swap_interpolation=swap_interpolation)


_pairs: dict[str, tuple[Pair, Pair]] = {}
_reference_cache: dict[tuple, ref.Expected] = {}


CLIP_Y = 0x74   # the vertex program's output attribute for clip y (a[0x70] x, a[0x74] y, a[0x78] z, a[0x7c] w)
SIGN_BIT = 0x80000000


def negate_clip_y(outs: list[dict[int, int]]) -> list[dict[int, int]]:
    """Each vertex's output words with the sign of clip y flipped (an output the program does not write is 0, so -0)."""
    for o in outs:
        o[CLIP_Y] = (o.get(CLIP_Y, 0) ^ SIGN_BIT) & 0xFFFFFFFF
    return outs


def render_in_replay_space(draw: ref.Draw) -> ref.Expected:
    """reference.render with the clip y of every vertex negated before the triangles are set up.

    The replay's viewport has a negative height, so clip y = +1 lands on window row 0 and a pixel program that reads its
    window position reads the row of the memory it writes; the reference's own space is Vulkan's, +1 at the last row.
    Negating y maps one onto the other: the pixel centre (x + 0.5, row + 0.5) is the replay's, the interpolated
    attributes and the edge and ambiguity rules follow it, and nothing else of reference.py is touched."""
    shade = ref.shade_vertices
    with mock.patch.object(ref, "shade_vertices", lambda d: negate_clip_y(shade(d))):
        return ref.render(draw)


def reference_image(case: Case, swap_uv: bool = False, swap_interpolation: bool = False) -> ref.Expected:
    key = (case.pair.spec_path, case.name, swap_uv, swap_interpolation)
    if key not in _reference_cache:
        _reference_cache[key] = render_in_replay_space(case.draw(swap_uv, swap_interpolation))
    return _reference_cache[key]


def write_draw(case: Case, folder: Path, *, indexed: bool, depth: tuple | None = None) -> Path:
    """The draw file of vk_replay_draw.c for a case, and the files it names, in folder."""
    folder.mkdir(parents=True, exist_ok=True)
    pair, res = case.pair, case.pair.res
    files = pair.program_files()
    lines = [f"size {case.width} {case.height}", "clear " + " ".join(repr(c) for c in case.clear)]
    if depth:
        lines.append("depth " + " ".join(repr(float(v)) for v in depth))
    for st in ("vs", "ps"):
        lines.append(f"program {st} {pair.keys[st]} {files[st][0]} {files[st][1]}")
    lines += [f"{line} 0" for line in case.attr_lines]
    stride = len(case.vblob) // len(case.vertices)
    if not stride:
        raise InputError("a vertex program with no vertex inputs is not drawn here")
    lines.append(f"stream 0 {stride} 0")
    if indexed:
        (folder / "vertices.bin").write_bytes(case.vblob)
        (folder / "indices.bin").write_bytes(b"".join(struct.pack("<I", i) for i in case.order))
        lines += ["vertex_buffer 0 vertices.bin", "index_buffer indices.bin u32", f"draw elements {NVN_DRAW_TRIANGLES} {len(case.order)}"]
    else:
        (folder / "vertices.bin").write_bytes(b"".join(case.vblob[i * stride:(i + 1) * stride] for i in case.order))
        lines += ["vertex_buffer 0 vertices.bin", f"draw arrays {NVN_DRAW_TRIANGLES} 0 {len(case.order)}"]
    texture_id = 0
    for st in ("vs", "ps"):
        r = res[st]
        for cb in r["cbufs"]:
            slot = cb["slot"]
            if slot < USER_BANK_FIRST:   # c[0], c[1] and c[2] are the replay's
                continue
            if slot - USER_BANK_FIRST >= 21:
                raise InputError(f"constant bank c[{slot}] has no uniform index")
            name = f"bank_{st}_{slot}.bin"
            (folder / name).write_bytes(bytes(case.bank_data[st].get(slot, bytearray(BANK_BYTES))))
            lines.append(f"uniform {st} {slot - USER_BANK_FIRST} {name}")
        bank0 = bytes(case.bank_data[st].get(0, bytearray(BANK_BYTES)))
        entries = case.banks.get(st, {}).get("0", {})
        for row, addr, size in storage_rows(st, r["sbufs"], bank0, entries):
            name = f"storage_{st}_{row}.bin"
            (folder / name).write_bytes(case.mem.window(addr, size))
            lines.append(f"storage {st} {row} {addr % 16} {name}")
        seen = set()
        for t in r["textures"]:
            if t["c2_slot"] in seen:
                continue
            seen.add(t["c2_slot"])
            tx = case.textures[st][t["tex_word_offset"]]
            texture_id += 1
            name = f"texture_{st}_{t['c2_slot']}.f32"
            (folder / name).write_bytes(b"".join(struct.pack("<4f", *px) for px in tx.texels))
            lines.append(f"texture {st} {t['c2_slot']} {texture_id} {tx.kind} {tx.width} {tx.height} {tx.layers} "
                         f"{tx.filter} {tx.address} {name}")
    path = folder / "draw.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---- the program ----------------------------------------------------------------------------------------------------

_build: dict[str, object] = {}
VALIDATION = re.compile(r"^validation errors (\d+) warnings (\d+)$", re.M)
STATS = re.compile(r"^replay frames (\d+) records (\d+) draws (\d+) dispatches (\d+) clears (\d+) passes (\d+) barriers (\d+)$", re.M)
SNAPS = re.compile(r"^replay snapshots (\d+) snapshot_bytes (\d+) live_uniforms (\d+) storage_writes (\d+) "
                   r"unregistered_textures (\d+) scratch_bytes (\d+)$", re.M)


def program() -> Path:
    if "exe" not in _build:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        tmp = tempfile.TemporaryDirectory(prefix="oot2-vkreplay-", ignore_cleanup_errors=True)
        atexit.register(tmp.cleanup)
        _build["tmp"] = tmp
        _build["exe"] = build_program(cc, include, glslang, TEST_SOURCE, Path(tmp.name), levels=(LEVEL,))[LEVEL]
    return _build["exe"]  # type: ignore[return-value]


def run(*args: str, validation: str = "off", timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run([str(program()), *args], capture_output=True, text=True, timeout=timeout,
                          env=case_env({"OOT2_VK_VALIDATION": validation}))


class Replay:
    """One run of the program: its image, depth and numbers."""

    def __init__(self, done: subprocess.CompletedProcess, out: Path, width: int, height: int, depth_out: Path | None):
        self.done, self.out = done, out
        self.ok = done.returncode == 0 and "ok vk_replay_draw" in done.stdout and not fatal_lines(done.stderr)
        self.image = parse_f32(out, width * height) if self.ok else []
        self.depth = parse_depth(depth_out, width * height) if self.ok and depth_out else []
        m = VALIDATION.search(done.stdout)
        self.errors, self.warnings = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
        s, p = STATS.search(done.stdout), SNAPS.search(done.stdout)
        names = ("frames", "records", "draws", "dispatches", "clears", "passes", "barriers")
        self.stats = dict(zip(names, map(int, s.groups()))) if s else {}
        self.snap = dict(zip(("snapshots", "snapshot_bytes", "live_uniforms", "storage_writes", "unregistered_textures",
                              "scratch_bytes"), map(int, p.groups()))) if p else {}

    def report(self) -> str:
        return f"{self.done.stdout}\n{self.done.stderr[-3000:]}"


def play(case: Case, name: str, *, validation: str = "on", indexed: bool = False, depth: tuple | None = None,
         flags: tuple[str, ...] = ()) -> Replay:
    """Write the case's draw file and replay it on the device."""
    folder = Path(WORK) / "t63" / name
    draw = write_draw(case, folder, indexed=indexed, depth=depth)
    run_name = "".join([f"replay-{validation}", *("-" + f.lstrip("-") for f in flags)])   # one file per run kind
    out = folder / f"{run_name}.f32"
    args = ["draw", str(draw), str(out), "--pack", str(PACK), *flags]
    depth_out = folder / f"{run_name}-depth.f32"
    if depth:
        args += ["--depth-out", str(depth_out)]
    return Replay(run(*args, validation=validation), out, case.width, case.height, depth_out if depth else None)


def compare(case: Case, exp: ref.Expected, got: Replay, validation: str, folder: Path) -> dict:
    """first_draw's comparison (its tolerance, its rule for edge pixels) of the replayed image, in memory row order, with a
    reference rendered in the replay's space (render_in_replay_space)."""
    vk = {"device": "", "driver": "", "api": "", "validation": validation, "validation_errors": got.errors,
          "validation_warnings": got.warnings, "loader_messages": 0}
    args = SimpleNamespace(atol=1e-4, rtol=1e-4, counter=None, validation=validation)
    return fd.compare(case.name, case.spec, exp, got.image, vk, 0.0, 0.0, folder, args)


def summary(rep: dict) -> str:
    keys = ("scenario", "pass", "pixels", "compared", "mismatches", "max_abs_err", "validation_errors", "validation_warnings")
    return json.dumps({k: rep.get(k) for k in keys})


# ---- the tests on plain data ----------------------------------------------------------------------------------------

class HelperTest(unittest.TestCase):
    def test_clip_y_changes_sign_and_nothing_else_does(self) -> None:
        half = struct.unpack("<I", struct.pack("<f", 0.5))[0]
        outs = negate_clip_y([{0x70: half, 0x74: half, 0x78: half}, {0x70: half}])
        self.assertEqual([word_float(o[0x74]) for o in outs], [-0.5, -0.0])
        self.assertEqual([o[0x70] for o in outs], [half, half])
        self.assertEqual(outs[0][0x78], half)

    def test_a_vertex_at_clip_y_plus_1_is_window_row_0_in_the_replays_space_and_the_last_row_in_vulkans(self) -> None:
        clip = [(0.0, 1.0, 0.0, 1.0)]
        self.assertEqual(window_vertices(clip, 8, 8, True)[0][:2], (4.0, 0.0))
        self.assertEqual(window_vertices(clip, 8, 8, False)[0][:2], (4.0, 8.0))

    def test_covered_rows_are_the_rows_with_a_pixel_that_is_not_the_clear_colour(self) -> None:
        clear = (-7.0, -7.0, -7.0, -7.0)
        pixels = [clear] * 12
        pixels[5], pixels[10] = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0)   # 4 wide: rows 1 and 2
        self.assertEqual(covered_rows(pixels, clear, 4, 3), [1, 2])

    def test_a_vertex_input_given_two_components_is_filled_with_0_0_0_1_from_the_third(self) -> None:
        spec = {"vertex_stream": {"attributes": [{"location": 0, "components": 4}, {"location": 1, "components": 2}],
                                  "vertices": [[1, 2, 3, 4, 5, 6]]}}
        res = {"vertex_inputs": [{"location": 0, "components": 4}, {"location": 1, "components": 4}]}
        out = expand_vertices(spec, res)
        self.assertEqual(out["vertex_stream"]["vertices"], [[1, 2, 3, 4, 5, 6, 0.0, 1.0]])
        self.assertNotIn("attributes", out["vertex_stream"])
        self.assertIn("attributes", spec["vertex_stream"])   # the spec as read is left alone

    def test_storage_rows_are_the_rows_of_the_stages_table(self) -> None:
        bank = bytearray(BANK_BYTES)
        struct.pack_into("<QI", bank, 0x170, 0x123456705, 192)
        sbufs = [{"address_byte_offset": 0x170, "cb_slot": 0, "role": "guest"}]
        entries = {"0x170": {"u64": 0x123456705}, "0x178": {"u32": [192]}}
        self.assertEqual(storage_rows("vs", sbufs, bytes(bank), entries), [(6, 0x123456705, 192)])
        struct.pack_into("<QI", bank, 0x510 + 16, 0x2000, 16)
        sbufs = [{"address_byte_offset": 0x520, "cb_slot": 0, "role": "guest"}]
        self.assertEqual(storage_rows("ps", sbufs, bytes(bank), {}), [(1, 0x2000, 16)])

    def test_a_c0_word_outside_the_storage_table_is_refused(self) -> None:
        for entries in ({"0x644": {"u32": [1]}}, {"0x170": {"u64": 1}, "0x700": {"u32": [5]}}):
            with self.assertRaisesRegex(InputError, "outside the storage table"):
                storage_rows("vs", [], bytes(BANK_BYTES), entries)

    def test_a_storage_buffer_that_is_not_a_row_of_the_table_is_refused(self) -> None:
        for s in ({"address_byte_offset": 0x175, "cb_slot": 0, "role": "guest"},
                  {"address_byte_offset": 0x170, "cb_slot": 3, "role": "guest"},
                  {"address_byte_offset": 0x110 + 16 * 16, "cb_slot": 0, "role": "guest"},
                  {"address_byte_offset": 0x170, "cb_slot": 0, "role": "raw_map"}):
            with self.assertRaises(InputError):
                storage_rows("vs", [s], bytes(BANK_BYTES), {})

    def test_depth_of_a_flat_triangle_follows_the_range_and_the_mode(self) -> None:
        # a triangle over the whole 4 by 4 target at z/w = 0.5 (w = 1): window depth 0.75 under NEAR_IS_MINUS_W, 0.5 otherwise
        clip = [(-3.0, -3.0, 0.5, 1.0), (5.0, -3.0, 0.5, 1.0), (-3.0, 5.0, 0.5, 1.0)]
        kinds = [ref.COLOR] * 16
        deep = expected_depth(clip, [(0, 1, 2)], kinds, 4, 4, 0.25, 0.75, NEAR_IS_MINUS_W)
        self.assertTrue(all(abs(d - (0.25 + 0.5 * 0.75)) < 1e-9 for d in deep))
        flat = expected_depth(clip, [(0, 1, 2)], kinds, 4, 4, 0.25, 0.75, NEAR_IS_ZERO)
        self.assertTrue(all(abs(d - (0.25 + 0.5 * 0.5)) < 1e-9 for d in flat))
        self.assertGreater(abs(deep[0] - flat[0]), 0.1)   # a wrong mode is told from the right one
        none = expected_depth(clip, [(0, 1, 2)], [ref.OUTSIDE] * 16, 4, 4, 0.0, 1.0, NEAR_IS_MINUS_W)
        self.assertEqual(none, [None] * 16)

    def test_depth_varies_across_a_triangle_with_its_vertices(self) -> None:
        # z/w of -1, 1 and 1 at window corners (0, 0), (8, 0) and (0, 8) of a 4 by 4 target: z/w = -1 + (x + y) / 4 at a
        # pixel centre, so the window depth (z/w + 1) / 2 is (x + y) / 8
        clip = [(-1.0, -1.0, -1.0, 1.0), (3.0, -1.0, 1.0, 1.0), (-1.0, 3.0, 1.0, 1.0)]
        d = expected_depth(clip, [(0, 1, 2)], [ref.COLOR] * 16, 4, 4, 0.0, 1.0, NEAR_IS_MINUS_W, flip_y=False)
        self.assertAlmostEqual(d[0], 1.0 / 8.0, places=9)    # the pixel centred at (0.5, 0.5)
        self.assertAlmostEqual(d[15], 7.0 / 8.0, places=9)   # (3.5, 3.5)
        self.assertAlmostEqual(d[1], 2.0 / 8.0, places=9)    # (1.5, 0.5)


# ---- the tests against the game's programs on the desktop -----------------------------------------------------------

@unittest.skipUnless(HAVE_INPUTS, NO_INPUTS)
class DesktopTestCase(unittest.TestCase):
    """The pairs, loaded once; the device checked once."""

    s2: Pair
    t23: Pair

    @classmethod
    def setUpClass(cls) -> None:
        if "pairs" not in _pairs:
            work = Path(WORK) / "t63"
            work.mkdir(parents=True, exist_ok=True)
            t23_spec = json.loads(Path(T23_SPEC).read_text(encoding="utf-8"))
            _pairs["pairs"] = (Pair(S2_SPEC, t23_spec["spv_dir"], work), Pair(T23_SPEC, t23_spec["spv_dir"], work))
        cls.s2, cls.t23 = _pairs["pairs"]

    def case(self, pair: Pair, scenario: str, spec: dict | None = None, name: str | None = None) -> Case:
        spec = spec or pair.spec
        scen = next(s for s in spec["scenarios"] if s["name"] == scenario)
        return Case(pair, spec, scen, name or f"{'s2' if pair is self.s2 else 't23'}_{scenario}")


class ReferenceTest(DesktopTestCase):
    """The CPU side alone (no device): research 19's reference of S2's base scenario."""

    def test_the_reference_of_s2s_base_scenario_is_research_19s(self) -> None:
        exp = reference_image(self.case(self.s2, "base"))
        counts = {k: exp.kind.count(k) for k in (ref.COLOR, ref.KILLED, ref.OUTSIDE, ref.EDGE, ref.AMBIGUOUS)}
        print("\nreference of s2 base:", counts)
        self.assertEqual(counts, {ref.COLOR: 3565, ref.KILLED: 0, ref.OUTSIDE: 12802, ref.EDGE: 17, ref.AMBIGUOUS: 0})

    def test_every_scenario_builds_a_draw_file_the_harness_reads(self) -> None:
        for pair, scenarios, indexed in ((self.s2, ("base", "alpha_test", "unpremultiply"), False),
                                         (self.t23, ("base", "lights"), True)):
            for scenario in scenarios:
                case = self.case(pair, scenario)
                folder = Path(WORK) / "t63" / case.name
                text = write_draw(case, folder, indexed=indexed).read_text(encoding="utf-8")
                self.assertEqual(re.findall(r"(?m)^program (vs|ps) ", text), ["vs", "ps"], case.name)
                self.assertRegex(text, r"(?m)^draw %s 4 " % ("elements" if indexed else "arrays"))
                for line in text.splitlines():
                    words = line.split()
                    if words[0] == "uniform":
                        self.assertEqual((folder / words[3]).stat().st_size, BANK_BYTES, line)
                    elif words[0] == "texture":
                        w, h, d = (int(x) for x in words[5:8])
                        layers = 6 if words[4] == "cube" else d if words[4] in ("2darray", "3d") else 1
                        self.assertEqual((folder / words[10]).stat().st_size, 16 * w * h * layers, line)
                    elif words[0] == "storage":
                        self.assertTrue((folder / words[4]).is_file(), line)
                self.assertTrue(all((folder / name).is_file() for name in re.findall(r"(?m)^program \w+ \S+ (\S+) (\S+)$", text)[0]))


class DeviceTestCase(DesktopTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        done = run("probe")
        if done.returncode != 0:
            text = "\n".join(fatal_lines(done.stderr))
            if done.returncode == EXIT_FATAL and any(s in text for s in NO_VULKAN):
                why = f"no usable Vulkan device here ({text})"
                if os.environ.get("OOT2_REQUIRE_GPU"):
                    raise RuntimeError(f"OOT2_REQUIRE_GPU is set but there is {why}")
                raise unittest.SkipTest(why)
            raise RuntimeError(f"the device probe failed:\n{done.stdout}{done.stderr}")
        cls.replays: dict[tuple, Replay] = {}

    def replay(self, case: Case, *, validation: str = "on", indexed: bool = False, flags: tuple[str, ...] = ()) -> Replay:
        key = (case.pair.spec_path, case.name, validation, indexed, flags)
        if key not in self.replays:
            self.replays[key] = play(case, case.name, validation=validation, indexed=indexed, flags=flags)
        got = self.replays[key]
        self.assertTrue(got.ok, got.report())
        return got

    def check_matches(self, case: Case, *, indexed: bool) -> None:
        exp = reference_image(case)
        folder = Path(WORK) / "t63" / case.name
        for validation in VALIDATIONS:
            got = self.replay(case, validation=validation, indexed=indexed)
            self.assertEqual((got.errors, got.warnings), (0, 0), got.report())
            self.assertEqual(got.stats.get("draws"), 1, got.report())
            rep = compare(case, exp, got, validation, folder)
            print(f"\nreplay {validation}: {summary(rep)}\n{got.stats} {got.snap}")
            self.assertEqual(rep["mismatches"], 0, summary(rep))
            self.assertGreater(rep["pixels"][ref.COLOR], 0, summary(rep))
            self.assertTrue(rep["pass"], summary(rep))


class S2Test(DeviceTestCase):
    def test_base_scenario_matches_the_reference(self) -> None:
        self.check_matches(self.case(self.s2, "base"), indexed=False)

    def test_alpha_test_scenario_matches_the_reference(self) -> None:
        self.check_matches(self.case(self.s2, "alpha_test"), indexed=False)

    def test_divide_by_alpha_scenario_matches_the_reference(self) -> None:
        self.check_matches(self.case(self.s2, "unpremultiply"), indexed=False)

    def test_counter_run_texture_coordinates_swapped_in_the_reference_fail_the_comparison(self) -> None:
        case = self.case(self.s2, "base")
        got = self.replay(case)
        rep = compare(case, reference_image(case, swap_uv=True), got, "on", Path(WORK) / "t63" / case.name)
        print(f"\ncounter-run swap-uv: {summary(rep)}")
        self.assertGreater(rep["mismatches"], 0, summary(rep))

    def test_counter_run_interpolation_swapped_in_the_reference_fails_the_comparison(self) -> None:
        case = self.case(self.s2, "base")
        got = self.replay(case)
        rep = compare(case, reference_image(case, swap_interpolation=True), got, "on", Path(WORK) / "t63" / case.name)
        print(f"\ncounter-run swap-interpolation: {summary(rep)}")
        self.assertGreater(rep["mismatches"], 0, summary(rep))

    def test_counter_run_origin_flip_removed_fails_the_comparison(self) -> None:
        case = self.case(self.s2, "base")
        got = self.replay(case, flags=("--no-flip",))
        rep = compare(case, reference_image(case), got, "on", Path(WORK) / "t63" / case.name)
        print(f"\ncounter-run no-flip: {summary(rep)}")
        self.assertGreater(rep["mismatches"], 0, summary(rep))

    def test_a_uniform_rewritten_after_the_submission_does_not_reach_the_frame_and_the_counter_run_does(self) -> None:
        case = self.case(self.s2, "base")
        exp = reference_image(case)
        folder = Path(WORK) / "t63" / case.name
        kept = self.replay(case, flags=("--imports", "--rewrite-uniform"))
        rep = compare(case, exp, kept, "on", folder)
        print(f"\nsnapshot on, uniforms rewritten after the submission: {summary(rep)} {kept.snap}")
        self.assertGreaterEqual(kept.snap.get("snapshots", 0), 2, kept.report())   # the two stages' user banks
        self.assertEqual(rep["mismatches"], 0, summary(rep))
        self.assertTrue(rep["pass"], summary(rep))
        # the counter-run: the GPU reads the buffers live. The write races the GPU's read by design (the program writes
        # as the submission returns, the GPU needs the recorder thread and a queue submission first), so it is tried
        # a few times; what it must show is a frame the write reached.
        caught = None
        for attempt in range(ATTEMPTS):
            live = play(case, case.name, validation="on", flags=("--imports", "--rewrite-uniform", "--no-snapshot"))
            self.assertTrue(live.ok, live.report())
            self.assertEqual(live.snap.get("snapshots"), 0, live.report())
            self.assertGreaterEqual(live.snap.get("live_uniforms", 0), 2, live.report())
            rep = compare(case, exp, live, "on", folder)
            print(f"\nsnapshot off (attempt {attempt + 1}): {summary(rep)}")
            if rep["mismatches"] > 0:
                caught = rep
                break
        self.assertIsNotNone(caught, "the uniforms were rewritten and the image still matched: the counter-run caught nothing")


class T23Test(DeviceTestCase):
    def test_base_scenario_matches_the_reference(self) -> None:
        self.check_matches(self.case(self.t23, "base"), indexed=True)

    def test_lights_scenario_matches_the_reference(self) -> None:
        self.check_matches(self.case(self.t23, "lights"), indexed=True)


class ViewportTest(DeviceTestCase):
    """A primitive in the upper half of clip space, under NEAR_IS_MINUS_W and LOWER_LEFT (the device's defaults)."""

    near, far, cleared = 0.25, 0.75, 1.0

    def upper_case(self) -> tuple[Case, list[tuple]]:
        """S2's base scenario with its vertices moved up until every vertex program output lies in the upper half of clip
        space (y / w between 0.1 and 0.9), inside the view volume; the clip positions the reference's vertex program gives."""
        base = self.case(self.s2, "base")
        for k in range(60):
            dy = 0.05 * k
            case = base.with_vertices([[v[0], v[1] + dy, *v[2:]] for v in base.spec["vertex_stream"]["vertices"]],
                                      "s2_viewport")
            outs = ref.shade_vertices(case.draw())
            clip = [tuple(word_float(o.get(0x70 + 4 * c, 0)) for c in range(4)) for o in outs]
            if all(w > 0 and 0.1 <= y / w <= 0.9 and abs(x / w) <= 0.9 and 0.05 < z / w < 0.95 for x, y, z, w in clip):
                return case, clip
        raise InputError("no shift of S2's quad puts it in the upper half of clip space")

    def run_viewport(self, flags: tuple[str, ...] = ()) -> tuple[Case, list[tuple], Replay]:
        case, clip = self.upper_case()
        got = play(case, case.name, validation="on", depth=(self.near, self.far, self.cleared), flags=flags)
        self.assertTrue(got.ok, got.report())
        self.assertEqual((got.errors, got.warnings), (0, 0), got.report())
        return case, clip, got

    def test_the_primitive_lands_upright_with_its_depth_in_range(self) -> None:
        case, clip, got = self.run_viewport()
        w, h = case.width, case.height
        ys = [y / cw for _, y, _, cw in clip]
        rows = covered_rows(got.image, case.clear, w, h)
        # NVN with LOWER_LEFT: clip y = +1 is memory row 0, so the quad's rows follow from its y / w, read from the top
        lo, hi = math.floor((1 - max(ys)) * h / 2) - EDGE_ROWS, math.ceil((1 - min(ys)) * h / 2) + EDGE_ROWS
        print(f"\nupper half: y/w {min(ys):.3f}..{max(ys):.3f}, memory rows {rows[0]}..{rows[-1]} of {h} (expected within {lo}..{hi})")
        self.assertTrue(rows and rows[-1] < h // 2, f"the primitive reaches row {rows[-1]} of {h}: not upright")
        self.assertTrue(lo <= rows[0] and rows[-1] <= hi, f"rows {rows[0]}..{rows[-1]}, expected within {lo}..{hi}")
        exp = reference_image(case)
        rep = compare(case, exp, got, "on", Path(WORK) / "t63" / case.name)
        print(f"viewport colour: {summary(rep)}")
        self.assertTrue(rep["pass"], summary(rep))
        checked, worst, out_of_range, uncleared = self.depth_report(case, clip, got)
        print(f"viewport depth: {checked} pixels, worst error {worst:.3g} (tolerance {DEPTH_ATOL}), {out_of_range} outside "
              f"[{self.near}, {self.far}], {uncleared} uncovered pixels not at the clear value")
        self.assertGreater(checked, 500)
        self.assertLessEqual(worst, DEPTH_ATOL)
        self.assertEqual((out_of_range, uncleared), (0, 0))

    def depth_report(self, case: Case, clip: list[tuple], got: Replay) -> tuple[int, float, int, int]:
        """The depth the replay wrote at the pixels the reference covers, against near + (far - near) * ((z / w + 1) / 2):
        the pixels checked, the worst error, the covered pixels outside the depth range, and the pixels outside the
        primitive that are not at the clear value. Reference and replay are both in the replay's space (memory row order)."""
        w, h = case.width, case.height
        exp = reference_image(case)
        want = expected_depth(clip, [(0, 1, 2), (3, 4, 5)], exp.kind, w, h, self.near, self.far, NEAR_IS_MINUS_W, flip_y=True)
        checked, worst, out_of_range, uncleared = 0, 0.0, 0, 0
        for k in range(w * h):
            if want[k] is not None:
                checked += 1
                worst = max(worst, abs(got.depth[k] - want[k]))
                out_of_range += not self.near <= got.depth[k] <= self.far
            elif exp.kind[k] == ref.OUTSIDE:
                uncleared += got.depth[k] != self.cleared
        return checked, worst, out_of_range, uncleared

    def test_counter_run_depth_mode_changed_misses_the_depth_the_test_expects(self) -> None:
        # NEAR_IS_ZERO maps z / w to window depth directly, so the same primitive's depth is not (z / w + 1) / 2 any more
        case, clip, got = self.run_viewport(("--depth-mode", "near_is_zero"))
        checked, worst, _, _ = self.depth_report(case, clip, got)
        print(f"\ncounter-run near_is_zero: {checked} pixels, worst error {worst:.3g} (tolerance {DEPTH_ATOL})")
        self.assertGreater(checked, 500)
        self.assertGreater(worst, 100 * DEPTH_ATOL)

    def test_counter_run_origin_flip_removed_lands_in_the_lower_half(self) -> None:
        case, clip, got = self.run_viewport(("--no-flip",))
        rows = covered_rows(got.image, case.clear, case.width, case.height)
        print(f"\nno flip: memory rows {rows[0]}..{rows[-1]} of {case.height}")
        self.assertGreaterEqual(rows[0], case.height // 2 - 1, "the unflipped primitive is not in the lower half")
        rep = compare(case, reference_image(case), got, "on", Path(WORK) / "t63" / case.name)
        self.assertGreater(rep["mismatches"], 0, summary(rep))


if __name__ == "__main__":
    unittest.main()
