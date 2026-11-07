# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Unit tests of the lifter's Python side (tools/lift), without a compiler: the NSO reader, discovery, the D3 rules
of the emitter, chunking, determinism and the generated-code marker. Modules are synthetic, built from instruction
words written here (tests/lift/synth.py); no game bytes.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "tools" / "lift" / "gen.py"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "lift"))
sys.path.insert(0, str(ROOT / "tests" / "lift"))

import check_content  # noqa: E402
import gen  # noqa: E402
import nso  # noqa: E402
import synth  # noqa: E402
from module import Module  # noqa: E402

RET, NOP = 0xD65F03C0, 0xD503201F
ADD_X0_1 = 0x91000400            # add x0, x0, #1
MRS_X0_CNTFRQ = 0xD53BE000       # mrs x0, cntfrq_el0
MRS_X0_MIDR = 0xD5380000         # mrs x0, midr_el1: a register the lifter does not model
FN_DEF = re.compile(r"^(?:__attribute__\(\(noinline\)\) )?void f_([0-9a-f]+)\(Cpu \*c\) \{", re.M)


def b(frm: int, to: int, link: bool = False) -> int:
    return (0x94000000 if link else 0x14000000) | (((to - frm) >> 2) & 0x3FFFFFF)


def words(*ws: int) -> bytes:
    return struct.pack(f"<{len(ws)}I", *ws)


def module_with(functions: list[list[int]]) -> tuple[bytes, dict[str, int]]:
    """Text holding the two reserved words, then each function at a 16-byte boundary; exports f0, f1, ..."""
    text, exports = bytearray(8), {}
    for i, fn in enumerate(functions):
        text += bytes(-len(text) % 16)
        exports[f"f{i}"] = len(text)
        text += words(*fn)
    return bytes(text), exports


class NsoTest(unittest.TestCase):
    def test_lz4_literals_and_matches(self) -> None:
        data = bytes(range(200)) * 3
        self.assertEqual(nso.lz4_block(synth.lz4_literals(data), len(data)), data)
        # 'ab' as literals, then a match at offset 2 of 4 + 6 bytes (no trailing literals: a short last sequence)
        block = bytes([0x26]) + b"ab" + bytes([2, 0])
        self.assertEqual(nso.lz4_block(block, 12), b"ab" * 6)

    def test_lz4_overlapping_match(self) -> None:
        block = bytes([0x1F]) + b"x" + bytes([1, 0, 10]) + bytes([0x30]) + b"end"
        self.assertEqual(nso.lz4_block(block, 1 + 29 + 3), b"x" * 30 + b"end")
        with self.assertRaises(nso.NsoError):
            nso.lz4_block(bytes([0x10]) + b"x" + bytes([5, 0]), 10)   # offset past the output
        with self.assertRaises(nso.NsoError):
            nso.lz4_block(synth.lz4_literals(b"abc"), 4)              # wrong size

    def test_compressed_and_plain_modules_read_alike(self) -> None:
        text, exports = module_with([[ADD_X0_1, RET]])
        plain = synth.Module(text, exports).build()
        raw = bytearray(plain)
        off, _, size = struct.unpack_from("<III", raw, 0x10)
        comp = synth.lz4_literals(bytes(raw[off:off + size]))
        rest = bytes(raw[off + size:])
        packed = bytearray(raw[:off]) + comp + rest
        struct.pack_into("<I", packed, 0xC, 1)                     # text compressed
        struct.pack_into("<I", packed, 0x60, len(comp))
        for i in (1, 2):                                          # later segments move by the size change
            o, = struct.unpack_from("<I", packed, 0x10 + 0x10 * i)
            struct.pack_into("<I", packed, 0x10 + 0x10 * i, o + len(comp) - size)
        with tempfile.TemporaryDirectory() as tmp:
            a, b_ = Path(tmp) / "a", Path(tmp) / "b"
            a.write_bytes(plain)
            b_.write_bytes(bytes(packed))
            na, nb = nso.Nso(str(a)), nso.Nso(str(b_))
            self.assertFalse(na.compressed)
            self.assertTrue(nb.compressed)
            self.assertEqual(na.image, nb.image)
            self.assertEqual([s[0] for s in nb.syms], ["", "f0"])

    def test_segment_hash_is_checked(self) -> None:
        text, exports = module_with([[RET]])
        raw = bytearray(synth.Module(text, exports).build())
        off, _, size = struct.unpack_from("<III", raw, 0x10)
        struct.pack_into("<I", raw, 0xC, 8)                        # check the text hash
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m"
            raw[0xA0:0xC0] = hashlib.sha256(bytes(raw[off:off + size])).digest()
            p.write_bytes(bytes(raw))
            nso.Nso(str(p))                                       # the right hash passes
            raw[0xA0] ^= 1
            p.write_bytes(bytes(raw))
            with self.assertRaisesRegex(nso.NsoError, "SHA-256"):
                nso.Nso(str(p))

    def test_not_an_nso(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m"
            p.write_bytes(b"\0" * 0x100)
            with self.assertRaises(nso.NsoError):
                nso.Nso(str(p))


class EmitterTest(unittest.TestCase):
    def load(self, functions: list[list[int]]) -> tuple[Module, dict[str, int]]:
        text, exports = module_with(functions)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = Path(self.tmp.name) / "main"
        p.write_bytes(synth.Module(text, exports).build())
        base = 0x7000000000
        return Module("main", p, base), {k: base + v for k, v in exports.items()}

    def emitter(self, mod: Module) -> gen.Emitter:
        starts, _ = gen.discover(mod)
        return gen.Emitter(mod, set((mod.lo + starts * 4).tolist()))

    def test_discovery_splits_at_outside_branch_targets(self) -> None:
        # f0 branches into the middle of the words after f1's ret: that target becomes a function of its own.
        shape = [[ADD_X0_1, 0, RET], [NOP, RET, ADD_X0_1, RET]]
        _, off = module_with(shape)
        shape[0][1] = b(off["f0"] + 4, off["f1"] + 8)
        mod, at = self.load(shape)
        starts, stats = gen.discover(mod)
        self.assertIn(at["f1"] + 8, set((mod.lo + starts * 4).tolist()))
        self.assertGreaterEqual(stats["rounds"], 2)

    def test_landing_pad_function_keeps_registers_in_cpu(self) -> None:
        body = [ADD_X0_1] * 6 + [RET]
        mod, at = self.load([body])
        e = self.emitter(mod)
        plain, _ = e.function(at["f0"], at["f0"] + 4 * len(body))
        self.assertTrue(any(line.strip().startswith("uint64_t x0 = c->x[0]") for line in plain))   # counterpart
        e.pads = [at["f0"] + 8]
        e.d3 = []
        lines, _ = e.function(at["f0"], at["f0"] + 4 * len(body))
        self.assertFalse(any("uint64_t x" in line for line in lines))
        self.assertEqual(e.d3[0]["landing_pads"], 1)
        self.assertTrue(e.d3[0]["trust"].startswith("off"))

    def test_cntfrq_reads_the_runtime_rate(self) -> None:
        mod, at = self.load([[MRS_X0_CNTFRQ, RET]])
        lines, _ = self.emitter(mod).function(at["f0"], at["f0"] + 8)
        self.assertTrue(any("rt_tick_hz()" in line for line in lines))

    def test_unsupported_instruction_is_a_trap_in_the_census(self) -> None:
        mod, at = self.load([[MRS_X0_MIDR, RET]])
        e = self.emitter(mod)
        lines, _ = e.function(at["f0"], at["f0"] + 8)
        self.assertTrue(any(line.strip().startswith(f"rt_udf(c, {at['f0']:#x}ull); /* mrs") for line in lines))
        self.assertEqual(e.stats["unsupported"], 1)
        self.assertEqual(list(e.unsupported), ["mrs (3, 0, 0, 0, 0)"])

    def test_tail_branch_to_setjmp_is_refused(self) -> None:
        shape = [[NOP, 0], [RET]]
        _, off = module_with(shape)
        shape[0][1] = b(off["f0"] + 4, off["f1"])
        mod, at = self.load(shape)
        e = self.emitter(mod)
        e.kinds[at["f1"]] = "setjmp"
        with self.assertRaisesRegex(gen.LiftError, "tail branch to setjmp"):
            e.function(at["f0"], at["f0"] + 8)


class GeneratorTest(unittest.TestCase):
    def lift(self, exefs: Path, out: Path, seed: str, *extra: str) -> dict:
        env = dict(os.environ, PYTHONHASHSEED=seed)
        done = subprocess.run([sys.executable, "-I", str(GEN), "--exefs", str(exefs), "--out", str(out),
                               "--modules", "main", "--jobs", "2", "--chunk-fns", "3", *extra],
                              capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads((out / "main" / "manifest.json").read_text(encoding="utf-8"))

    def test_chunks_hold_sorted_addresses_and_unchanged_files_keep_mtime(self) -> None:
        fns = [[ADD_X0_1] * (1 + i % 4) + [RET] for i in range(11)]
        text, exports = module_with(fns)
        with tempfile.TemporaryDirectory() as tmp:
            exefs = Path(tmp) / "exefs"
            exefs.mkdir()
            (exefs / "main").write_bytes(synth.Module(text, exports).build())
            out_a, out_b = Path(tmp) / "a", Path(tmp) / "b"
            first = self.lift(exefs, out_a, "1")
            self.assertEqual(first["function_count"], 11)
            # The end of the module's lifted code (the lifted-code end marker): the manifest names it, and it is the last thing in the
            # function table, a definition of its own that the link places after every chunk.
            self.assertEqual(first["end_marker"], "main_lifted_end")
            table = (out_a / "main" / first["table"]).read_text(encoding="utf-8")
            self.assertEqual(table.count("main_lifted_end"), 1)
            self.assertTrue(table.rstrip().endswith("}"))
            self.assertGreater(table.index("main_lifted_end"), table.index("main_fn_count"))
            last = -1
            for name in first["chunks"]:
                addrs = [int(a, 16) for a in FN_DEF.findall((out_a / "main" / name).read_text(encoding="utf-8"))]
                self.assertEqual(addrs, sorted(addrs), name)
                self.assertGreater(addrs[0], last)
                last = addrs[-1]
            mtimes = {p.name: p.stat().st_mtime_ns for p in (out_a / "main").glob("*.c")}
            again = self.lift(exefs, out_a, "2")
            self.assertEqual(again["stats"]["chunks_written"], 0)
            self.assertEqual(mtimes, {p.name: p.stat().st_mtime_ns for p in (out_a / "main").glob("*.c")})
            # Another hash seed in a fresh folder writes the same bytes (python.md: no set order leaks out).
            self.lift(exefs, out_b, "3")
            for p in sorted((out_a / "main").glob("*.c")):
                self.assertEqual(p.read_bytes(), (out_b / "main" / p.name).read_bytes(), p.name)
            report_a = json.loads((out_a / "lift_report.json").read_text(encoding="utf-8"))
            report_b = json.loads((out_b / "lift_report.json").read_text(encoding="utf-8"))
            for r in (report_a, report_b):
                r.pop("timings")
                for m in r["modules"].values():
                    m.pop("timings")
                    m["stats"].pop("chunks_written")   # 0 in out_a, which already held the same files
            self.assertEqual(report_a, report_b)

    def test_generated_files_carry_the_marker_the_guard_refuses(self) -> None:
        text, exports = module_with([[ADD_X0_1, RET]])
        with tempfile.TemporaryDirectory() as tmp:
            exefs = Path(tmp) / "exefs"
            exefs.mkdir()
            (exefs / "main").write_bytes(synth.Module(text, exports).build())
            self.lift(exefs, Path(tmp) / "out", "0")
            for p in sorted((Path(tmp) / "out" / "main").glob("*.c")):
                data = p.read_bytes()
                self.assertIn(check_content.GENERATED_MARKER, data, p.name)

    def test_bad_input_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exefs = Path(tmp) / "exefs"
            exefs.mkdir()
            (exefs / "main").write_bytes(b"not a module")
            done = subprocess.run([sys.executable, "-I", str(GEN), "--exefs", str(exefs), "--out",
                                   str(Path(tmp) / "out")], capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 2)
            self.assertIn("not an NSO0 file", done.stderr)


if __name__ == "__main__":
    unittest.main()
