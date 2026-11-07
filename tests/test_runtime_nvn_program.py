# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the NVN2 front end's programs and state objects
(architecture D10, D16, D21).

Builds tests/runtime/nvn_program_test.c with runtime/core and runtime/nvn, as
tests/test_runtime_nvn.py builds its harness, at -O1 and -O2. The key test
assembles programs from seeded synthetic bytes (control sections and code
areas, no game data), has the front end set them on programs whose code lies
in a guest memory pool, and compares each stage's key with the one
tools/shaders/split.py computes from the same bytes (stage_of, ctl_key,
build_job and the variant hash), so the back end's cache lookup and the shader
step of prepare share one scheme. Task and mesh stages carry program headers
that read as vertex, tessellation or domain stages, and must be classified by
their control section. The state test round-trips every field of each state
object through its setters and getters, with guard bytes after the SDK's size.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "shaders"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import split  # noqa: E402
import test_runtime_nvn as nvn  # noqa: E402

TEST_SOURCE = ROOT / "tests" / "runtime" / "nvn_program_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
SEED = 35
SHARED_OPS = sorted(split.SHARED_MEMORY_OPCODES)
# Task payloads in bytes: multiples of 16, of 4 only, and of neither, which the stage memory rounds differently.
PAYLOADS = (0, 1, 3, 8, 13, 20, 126, 128, 3200, 3203, 65535)

# Program shapes, by stage: graphics, task and mesh, compute.
SHAPES = (("vs", "ps"), ("ts", "ms", "ps"), ("ms", "ps"), ("cs",), ("vs",), ("ms",))

# The failing cases: (case, exit code, text the FATAL line must contain)
FAILING = (
    ("code_out_of_reach", nvn.EXIT_UNIMPLEMENTED,
     "FATAL: unimplemented nvn: ProgramSetShaders: stage 0's code 0x8"),
    ("code_in_no_pool", nvn.EXIT_UNIMPLEMENTED,
     "+0x200 lies inside no memory pool the GPU can reach"),
    ("geometry_stage", nvn.EXIT_UNIMPLEMENTED,
     "FATAL: unimplemented nvn: ProgramSetShaders: stage 0 is a stage of kind gs, which prepare makes no translator job "
     "for (control +0x56 0, SPH word 0x1001"),
    ("no_job", nvn.EXIT_UNIMPLEMENTED,
     "FATAL: unimplemented nvn: ProgramSetShaders: stage 0 (vs, code 0x"),
    ("two_vertex_stages", nvn.EXIT_UNIMPLEMENTED,
     "FATAL: unimplemented nvn: ProgramSetShaders: stage 1 is a second vs stage"),
    ("ext_field", nvn.EXIT_UNIMPLEMENTED,
     "FATAL: unimplemented nvn: ProgramSetShadersExt: stage 0 has the unknown field 0x1"),
    ("attribute_offset", nvn.EXIT_UNIMPLEMENTED,
     "FATAL: unimplemented nvn: nvnVertexAttribStateSetFormat: format 0x2a at offset 524288 does not fit the packed "
     "state"),
)
FAILING_MORE = {
    "code_out_of_reach": "lies in a pool the CPU cannot see (flags 0x21), so its key cannot be read",
    "no_job": "has no translator job: the instruction length is not a positive multiple of 16 inside the code area",
}

STATES = ("NVNblendState", "NVNcolorState", "NVNchannelMaskState", "NVNmultisampleState", "NVNpolygonState",
          "NVNdepthStencilState", "NVNvertexAttribState", "NVNvertexStreamState")


# --- synthetic stages -----------------------------------------------------------

def sph_word(rng: random.Random, sph_type: int) -> int:
    """A first SPH word of the given type (bits 10 to 13), the other bits random."""
    return (rng.getrandbits(32) & ~(0xF << 10)) | sph_type << 10


def control(rng: random.Random, stage_byte: int, inst: int, code_len: int) -> bytearray:
    """An NVN2 control section of random bytes with the fields split.py and the front end read. +0x30, which the key
    clears, and the bytes between the fields stay random."""
    ctl = bytearray(rng.randbytes(0xE0))
    ctl[0:4], ctl[4:8] = split.CTL_MAGIC, split.CTL_VERSION
    struct.pack_into("<I", ctl, 0x38, inst)
    struct.pack_into("<I", ctl, 0x40, rng.randrange(code_len + 1))
    struct.pack_into("<I", ctl, 0x44, code_len)
    ctl[0x56] = stage_byte
    return ctl


def edge(rng: random.Random, lo: int, hi: int) -> int:
    r = rng.randrange(4)
    return lo if r == 0 else hi if r == 1 else rng.randint(lo, hi)


def graphics(rng: random.Random, kind: str) -> tuple[bytes, bytes]:
    inst = 16 * rng.randint(1, 40)
    code_len = inst + 16 * rng.randint(0, 16)
    ctl = control(rng, rng.choice([b for b in range(256) if b not in (5, 6, 7)]), inst, code_len)
    code = bytearray(rng.randbytes(code_len))
    if kind == "ps":
        word = sph_word(rng, 5)
    elif rng.randrange(2):
        word = sph_word(rng, 1)
    else:
        word = (rng.getrandbits(32) & ~(0xF << 10) & ~0x1F) | 1        # type 0 with version 1: also a vertex shader
    struct.pack_into("<I", code, 0, word)
    return bytes(ctl), bytes(code)


def compute(rng: random.Random) -> tuple[bytes, bytes]:
    inst = 16 * rng.randint(1, 60)
    code_len = inst + 16 * rng.randint(0, 8)
    ctl = control(rng, 5, inst, code_len)
    x = edge(rng, 1, 1024)
    y = edge(rng, 1, 1024 // x)
    z = edge(rng, 1, 1024 // (x * y))
    struct.pack_into("<3I", ctl, 0x58, x, y, z)
    if rng.randrange(4) == 0:
        struct.pack_into("<I", ctl, 0x64, rng.choice((0, 0xFFFFFFF1, 0xFFFFFFFF)))
    return bytes(ctl), rng.randbytes(code_len)


def task_or_mesh(rng: random.Random, kind: str) -> tuple[bytes, bytes]:
    """A task or mesh stage whose program header reads as a vertex, tessellation or domain stage, as the console's do;
    its stage byte says what it is."""
    inst = 0x80 + 16 * rng.randint(1, 60)
    code_len = inst + 16 * rng.randint(0, 40)
    ctl = control(rng, 6 if kind == "ts" else 7, inst, code_len)
    struct.pack_into("<2H", ctl, 0x58, edge(rng, 1, 128), rng.choice(PAYLOADS))
    code = bytearray(rng.randbytes(code_len))
    struct.pack_into("<I", code, 0, sph_word(rng, rng.choice((1, 2, 3))))
    if rng.randrange(2):                                        # a shared-memory instruction of its own
        at = 0x80 + 16 * rng.randrange((inst - 0x80) // 16)
        struct.pack_into("<H", code, at, (rng.getrandbits(16) & 0xF000) | rng.choice(SHARED_OPS))
    if kind == "ms":
        ctl[0x5E] = rng.choice((0, 1, 0x80))
        struct.pack_into("<2H", ctl, 0x60, edge(rng, 0, 256), edge(rng, 0, 256))
        second = rng.randrange(3)
        if second == 0:
            ctl[0x68] &= 0x7F
        else:
            ctl[0x68] |= 0x80
            room = (code_len - 0x80) // 16
            struct.pack_into("<I", ctl, 0x64, 0 if second == 1 or room < 1 else 16 * rng.randint(1, room))
    return bytes(ctl), bytes(code)


def make_stage(rng: random.Random, kind: str) -> tuple[bytes, bytes]:
    if kind in ("vs", "ps"):
        return graphics(rng, kind)
    if kind == "cs":
        return compute(rng)
    return task_or_mesh(rng, kind)


def programs() -> list[list[tuple[bytes, bytes]]]:
    """Every shape eight times, then two programs that repeat stages of the first ones byte for byte."""
    rng = random.Random(SEED)
    out = [[make_stage(rng, kind) for kind in shape] for _ in range(8) for shape in SHAPES]
    out.append([out[0][0], out[6][1]])                          # a vs and a ps of two earlier programs
    out.append([out[1][0], out[2][0]])                          # a ts and an ms of two earlier programs
    return out


def expected(ctl: bytes, code: bytes) -> tuple[str, str]:
    """What the shader step of prepare calls this stage: its stage and its job key."""
    stage = split.stage_of(ctl, code)
    spec = split.build_job(stage, ctl, code)
    return stage, f"{split.ctl_key(ctl, code)}.{spec.variant_hash()}"


class ProgramTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = nvn.find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-nvnprog-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in nvn.CORE.glob("*.c")) + \
            sorted(str(p) for p in nvn.NVN.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in nvn.LEVELS:
            exe = out / f"nvn_program_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*nvn.compile_flags(cc), level, *extra, "-I", str(nvn.CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe
        cls.programs = programs()
        cls.program_file = out / "programs.txt"
        cls.program_file.write_text(
            "\n".join(" ".join(f"{ctl.hex()}:{code.hex()}" for ctl, code in prog) for prog in cls.programs) + "\n",
            encoding="ascii")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case, *args], capture_output=True, text=True, timeout=120,
                              env=env)

    def assert_passes(self, case: str, *args: str) -> list[subprocess.CompletedProcess]:
        runs = []
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=case):
                done = self.run_case(level, case, *args)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                runs.append(done)
        return runs

    def stages_printed(self, done: subprocess.CompletedProcess) -> dict[tuple[int, int], tuple[str, str]]:
        return {(int(m[1]), int(m[2])): (m[3], m[4])
                for m in re.finditer(r"^stage (\d+) (\d+) (\w+) ([0-9a-f]{16}\.[0-9a-f]{8})$", done.stdout, re.M)}

    # ---- programs

    def test_keys_match_split(self) -> None:
        # The fixtures reach both roundings of a task's stage memory: payloads that are not multiples of 16, and of 4.
        payloads = [struct.unpack_from("<H", ctl, 0x5A)[0] for prog in self.programs for ctl, _ in prog if ctl[0x56] == 6]
        self.assertTrue(any(p % 16 for p in payloads) and any(p % 4 for p in payloads), payloads)
        want = {(k, i): expected(ctl, code) for k, prog in enumerate(self.programs) for i, (ctl, code) in enumerate(prog)}
        for done in self.assert_passes("keys", str(self.program_file)):
            self.assertEqual(self.stages_printed(done), want)
            self.assertIn(f"ok keys {len(self.programs)}", done.stdout)
            self.assertIn(f"records {len({key for _, key in want.values()})}\n", done.stdout)

    def test_task_and_mesh_stages_are_classified_by_their_control_section(self) -> None:
        done = self.assert_passes("keys", str(self.program_file))[0]
        printed = self.stages_printed(done)
        misled = 0
        for k, prog in enumerate(self.programs):
            for i, (ctl, code) in enumerate(prog):
                by_control = split.CTL_STAGE.get(ctl[0x56])
                if by_control not in ("ts", "ms"):
                    continue
                by_header = split.SPH_STAGE.get(struct.unpack_from("<I", code, 0)[0] >> 10 & 0xF)
                self.assertEqual(printed[(k, i)][0], by_control)
                misled += by_header != by_control
        self.assertGreater(misled, 20)                       # headers that say vertex, tessellation or domain

    # ---- the stage-key dump (OOT2_STAGE_KEYS)

    def test_stage_key_dump_names_each_key_once_in_the_order_it_was_met(self) -> None:
        first: dict[str, tuple[str, bytes, bytes]] = {}     # key -> (kind, ctl, code), in the order first set
        for prog in self.programs:
            for ctl, code in prog:
                stage, key = expected(ctl, code)
                first.setdefault(key, (stage, ctl, code))
        for level in nvn.LEVELS:
            with self.subTest(level=level), tempfile.TemporaryDirectory(prefix="oot2-keys-") as folder:
                dump = Path(folder) / "keys.tsv"
                done = self.run_case(level, "keys", str(self.program_file),
                                     env={**os.environ, "OOT2_STAGE_KEYS": str(dump)})
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                lines = dump.read_text(encoding="ascii").split("\n")
                self.assertEqual(lines[0], "#oot2-stage-keys 1")
                self.assertEqual(lines[-1], "")
                rows = [line.split("\t") for line in lines[1:-1]]
                self.assertEqual([r[2] for r in rows], list(first))
                for n, (row, (key, (stage, ctl, code))) in enumerate(zip(rows, first.items()), 1):
                    self.assertEqual(len(row), 8, row)
                    self.assertEqual(row[:2], [str(n), "0"])             # no present has happened in this harness
                    self.assertEqual(row[3], stage)
                    self.assertEqual(row[4], str(len(code)))
                    self.assertEqual(row[5], str(struct.unpack_from("<I", ctl, 0x38)[0]))
                    self.assertEqual(row[6], hashlib.sha256(code).hexdigest())
                    self.assertEqual(row[7], ctl[:0xD0].hex())

    def test_stage_key_dump_is_off_without_the_variable_and_never_stops_a_run(self) -> None:
        environ = {k: v for k, v in os.environ.items() if k != "OOT2_STAGE_KEYS"}
        with tempfile.TemporaryDirectory(prefix="oot2-keys-") as folder:
            for level in nvn.LEVELS:
                with self.subTest(level=level):
                    done = self.run_case(level, "keys", str(self.program_file), env={**environ, "OOT2_STAGE_KEYS": ""})
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertNotIn("stage-key dump", done.stdout + done.stderr)
                    nowhere = str(Path(folder) / "missing-folder" / "keys.tsv")
                    done = self.run_case(level, "keys", str(self.program_file), env={**environ, "OOT2_STAGE_KEYS": nowhere})
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn("cannot write the stage-key dump", done.stdout + done.stderr)
                    self.assertFalse(Path(nowhere).exists())

    def test_stages_prepare_has_no_job_for_stop(self) -> None:
        for case, code, text in FAILING:
            for level in nvn.LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    self.assertIn(FAILING_MORE.get(case, ""), done.stderr)
                    self.assertEqual(done.stderr.count("FATAL"), 1, done.stderr)

    # ---- state objects

    def test_state_objects_round_trip_within_their_size(self) -> None:
        for done in self.assert_passes("states"):
            self.assertEqual(re.findall(r"^state (\w+)$", done.stdout, re.M), list(STATES))


if __name__ == "__main__":
    unittest.main()
