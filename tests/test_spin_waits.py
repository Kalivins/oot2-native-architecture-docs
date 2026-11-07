# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the spin wait search (tools/lift/spin_waits.py) and of the spin gates gen.py emits from its table.

Modules are synthetic, built from the instruction words encoded below; no game bytes. Every word has its assembly in
a comment beside it, and the encoders follow the Arm ARM field layout (the loop shapes are the ones the search
documents). The search decodes with capstone, so the tests that run it skip without capstone (pip install
capstone==5.0.7), unless OOT2_REQUIRE_CAPSTONE is set, as continuous integration sets it. The tests of gen.py's side
(a table written by hand) need no capstone.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "tools" / "lift" / "gen.py"
SPIN = ROOT / "tools" / "lift" / "spin_waits.py"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "lift"))
sys.path.insert(0, str(ROOT / "tests" / "lift"))

import build_game  # noqa: E402
import gen  # noqa: E402
import synth  # noqa: E402
from module import Module  # noqa: E402
from prepare import program  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, Reporter  # noqa: E402

try:
    import capstone  # noqa: F401
    import spin_waits
    CAPSTONE_ERROR = None
except ImportError as e:   # capstone is the only import of spin_waits that is not in the repository
    spin_waits = None
    CAPSTONE_ERROR = e


def need_capstone() -> None:
    if CAPSTONE_ERROR is None:
        return
    if os.environ.get("OOT2_REQUIRE_CAPSTONE"):
        raise RuntimeError(f"OOT2_REQUIRE_CAPSTONE is set but capstone cannot be imported: {CAPSTONE_ERROR}")
    raise unittest.SkipTest(f"capstone is not installed (pip install capstone==5.0.7): {CAPSTONE_ERROR}")


NOP = 0xD503201F          # nop
RET = 0xD65F03C0          # ret
CLREX = 0xD5033F5F        # clrex
STLR_WZR_X0 = 0x889FFC1F  # stlr wzr, [x0]


def ldr_w(rt: int, rn: int) -> int:
    return 0xB9400000 | rn << 5 | rt                  # ldr w<rt>, [x<rn>]


def str_w(rt: int, rn: int) -> int:
    return 0xB9000000 | rn << 5 | rt                  # str w<rt>, [x<rn>]


def ldaxr_w(rt: int, rn: int) -> int:
    return 0x885FFC00 | rn << 5 | rt                  # ldaxr w<rt>, [x<rn>]


def stxr_w(rs: int, rt: int, rn: int) -> int:
    return 0x88007C00 | rs << 16 | rn << 5 | rt       # stxr w<rs>, w<rt>, [x<rn>]


def cmp_w0(rn: int) -> int:
    return 0x7100001F | rn << 5                       # cmp w<rn>, #0   (subs wzr, w<rn>, #0)


def add_w1(rd: int) -> int:
    return 0x11000400 | rd << 5 | rd                  # add w<rd>, w<rd>, #1


def add_x1(rd: int) -> int:
    return 0x91000400 | rd << 5 | rd                  # add x<rd>, x<rd>, #1


def sub_x1(rd: int) -> int:
    return 0xD1000400 | rd << 5 | rd                  # sub x<rd>, x<rd>, #1


# Branches take word indices: the branch is word i of the function, its target word j (the offset is j - i words).

def b_(i: int, j: int, link: bool = False) -> int:
    return (0x94000000 if link else 0x14000000) | ((j - i) & 0x3FFFFFF)      # b <j> / bl <j>


def b_eq(i: int, j: int) -> int:
    return 0x54000000 | ((j - i) & 0x7FFFF) << 5 | 0x0                       # b.eq <j>


def cbz_w(rt: int, i: int, j: int) -> int:
    return 0x34000000 | ((j - i) & 0x7FFFF) << 5 | rt                        # cbz w<rt>, <j>


def cbnz_w(rt: int, i: int, j: int) -> int:
    return 0x35000000 | ((j - i) & 0x7FFFF) << 5 | rt                        # cbnz w<rt>, <j>


def cbnz_x(rt: int, i: int, j: int) -> int:
    return 0xB5000000 | ((j - i) & 0x7FFFF) << 5 | rt                        # cbnz x<rt>, <j>


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


# l: ldr w1, [x0]; cbz w1, l
POLL = [ldr_w(1, 0), cbz_w(1, 1, 0), RET]

# l: ldr w1, [x0]; cmp w1, #0; b.eq l
TEST_LOOP = [ldr_w(1, 0), cmp_w0(1), b_eq(2, 0), RET]

# l: ldr w1, [x0]; str w2, [x3]; cbz w1, l      (the store's value and address are never written by the loop)
POLL_STORE = [ldr_w(1, 0), str_w(2, 3), cbz_w(1, 2, 0), RET]

# The acquire of the search's docstring (words 0-5), then a poll of an item, the release and the branch back.
LOCK_SEQUENCE = [
    ldaxr_w(1, 0),    # 0  ldaxr w1, [x0]
    cbz_w(1, 1, 4),   # 1  cbz w1, <4>      (free: go and store)
    CLREX,            # 2  clrex
    b_(3, 0),         # 3  b <0>            (taken: try again)
    stxr_w(2, 3, 0),  # 4  stxr w2, w3, [x0]
    cbnz_w(2, 5, 0),  # 5  cbnz w2, <0>
]
LOCK_LOOP = LOCK_SEQUENCE + [
    ldr_w(4, 5),      # 6  ldr w4, [x5]
    STLR_WZR_X0,      # 7  stlr wzr, [x0]   (release)
    cbz_w(4, 8, 0),   # 8  cbz w4, <0>
    RET,
]

# l: ldr w1, [x0]; bl callee; cbz w1, l; ret; callee: ret
CALL_IN_LOOP = [ldr_w(1, 0), b_(1, 4, link=True), cbz_w(1, 2, 0), RET, RET]

# l: sub x1, x1, #1; cbnz x1, l       (writes x1, which is live at the head)
COUNTED = [sub_x1(1), cbnz_x(1, 1, 0), RET]

# l: ldr w1, [x0]; add w2, w2, #1; str w2, [x3]; cbz w1, l      (stores x2, which every pass changes)
STORE_OF_WRITTEN = [ldr_w(1, 0), add_w1(2), str_w(2, 3), cbz_w(1, 3, 0), RET]

# ldr w1, [x0]; cbz w1, done; add x0, x0, #1; done: ret      (a forward branch only)
FORWARD_ONLY = [ldr_w(1, 0), cbz_w(1, 1, 3), add_x1(0), RET]


class SearchTest(unittest.TestCase):
    """Search.all() on a bare image: two words of padding, the function, then nops. Image offset 8 is its first word."""

    @classmethod
    def setUpClass(cls) -> None:
        need_capstone()

    def found(self, fn: list[int]) -> list[tuple[int, list[int], list[str]]]:
        img = words(NOP, NOP, *fn, *([NOP] * 8))
        return [(g["T"], sorted(g["valid"]), g["kinds"]) for g in spin_waits.Search(img, len(img)).all()]

    def test_plain_poll_is_found(self) -> None:
        self.assertEqual(self.found(POLL), [(8, [12], [])])

    def test_test_loop_is_found(self) -> None:
        (head, edges, kinds), = self.found(TEST_LOOP)
        self.assertEqual((head, edges), (8, [16]))
        self.assertIn(kinds, ([], ["tst"]))

    def test_stores_of_registers_the_loop_never_writes_are_allowed(self) -> None:
        self.assertEqual(self.found(POLL_STORE), [(8, [16], ["stc"])])

    def test_lock_taken_and_released_in_the_loop_is_found(self) -> None:
        # The back-edges inside the acquire sequence (words 3 and 5) belong to the lock; the loop's is word 8.
        self.assertEqual(self.found(LOCK_LOOP), [(8, [8 + 4 * 8], ["acq", "rel"])])

    def test_lone_acquire_is_not_a_spin_wait(self) -> None:
        # Hand-derived from region1: the sequence is one node whose only exit leaves the region, so nothing returns to
        # the head. The acquire counts only inside a loop that releases the lock (previous test).
        self.assertEqual(self.found(LOCK_SEQUENCE + [RET]), [])

    def test_call_in_the_loop_is_refused(self) -> None:
        self.assertEqual(self.found(CALL_IN_LOOP), [])

    def test_counted_loop_is_refused(self) -> None:
        self.assertEqual(self.found(COUNTED), [])

    def test_store_of_a_register_the_loop_writes_is_refused(self) -> None:
        self.assertEqual(self.found(STORE_OF_WRITTEN), [])

    def test_forward_branch_only_is_refused(self) -> None:
        self.assertEqual(self.found(FORWARD_ONLY), [])


class MemoTest(unittest.TestCase):
    """The memo of decoded words (Search.dec) is bounded and changes no answer (ticket 122: unbounded, it held 3.4 KiB
    per word searched and made the search the peak memory of the whole prepare)."""

    SHAPES = [POLL, TEST_LOOP, POLL_STORE, LOCK_LOOP, CALL_IN_LOOP, COUNTED, STORE_OF_WRITTEN, FORWARD_ONLY]
    COPIES = 90

    @classmethod
    def setUpClass(cls) -> None:
        need_capstone()
        # Every shape 90 times over, each followed by padding: thousands of words are decoded, in several kinds of
        # loop, so a word served for the wrong address would change a verdict somewhere.
        ws = [NOP, NOP]
        for _ in range(cls.COPIES):
            for fn in cls.SHAPES:
                ws += fn + [NOP] * 3
        cls.img = words(*ws, *([NOP] * 8))

    def run_search(self, memo: int | None = None, cls=None):
        kw = {} if memo is None else {"memo": memo}
        s = (cls or spin_waits.Search)(self.img, len(self.img), **kw)
        return s, [(g["T"], sorted(g["valid"]), g["size"], g["kinds"]) for g in s.all()]

    def test_the_default_bound_is_the_constant_and_covers_the_reach_of_a_head(self) -> None:
        s = spin_waits.Search(self.img, len(self.img))
        self.assertEqual(s.memo, spin_waits.SPIN_MEMO)
        # A head reads from 4 bytes before it to 24 past its farthest back-edge, (SPIN_WIN + 28) / 4 words: a bound
        # under that decodes the same words again and again within one search, and a few times that keeps a clear
        # from falling inside most heads.
        self.assertGreaterEqual(spin_waits.SPIN_MEMO, 8 * ((spin_waits.SPIN_WIN + 28) // 4 + 1))
        self.assertEqual(spin_waits.Search(self.img, len(self.img), memo=0).memo, 1)

    def test_the_memo_never_holds_more_than_its_bound(self) -> None:
        peaks = []

        class Watched(spin_waits.Search):
            def ins(self, a):
                r = super().ins(a)
                peaks.append(len(self.dec))
                return r

        _, unbounded = self.run_search(10 ** 9, Watched)
        self.assertGreater(max(peaks), 2 * spin_waits.SPIN_MEMO,
                           "the image must decode more words than any bound below")
        for memo in (1, 64, spin_waits.SPIN_MEMO):
            peaks.clear()
            _, found = self.run_search(memo, Watched)
            self.assertLessEqual(max(peaks), memo, memo)
            self.assertEqual(found, unbounded, memo)

    def test_a_bounded_memo_finds_what_an_unbounded_one_does(self) -> None:
        _, unbounded = self.run_search(10 ** 9)
        self.assertEqual(len(unbounded), 4 * self.COPIES)        # POLL, TEST_LOOP, POLL_STORE and LOCK_LOOP each time
        for memo in (1, 5, 100, spin_waits.SPIN_MEMO):
            self.assertEqual(self.run_search(memo)[1], unbounded, memo)

    def test_the_default_search_keeps_a_fraction_of_the_words_it_decoded(self) -> None:
        decoded = []

        class Counting(spin_waits.Search):
            def ins(self, a):
                if a not in self.dec:
                    decoded.append(a)
                return super().ins(a)

        s, _ = self.run_search(None, Counting)
        self.assertLessEqual(len(s.dec), spin_waits.SPIN_MEMO)
        self.assertGreater(len(set(decoded)), 2 * spin_waits.SPIN_MEMO)
        # Dropping the memo costs little: a word is decoded again only after a drop, and a head asks for the same few
        # dozen words many times over.
        self.assertLess(len(decoded), 2 * len(set(decoded)))


def write_main(folder: Path, functions: list[list[int]]) -> dict[str, int]:
    text, exports = module_with(functions)
    (folder / "main").write_bytes(synth.Module(text, exports).build())
    return exports


class EmitterSpinTest(unittest.TestCase):
    """gen.Emitter lifts the back-edge of a listed spin wait through rt_spin_gate."""

    BASE = 0x7000000000

    def load(self, functions: list[list[int]]) -> tuple[Module, set[int], dict[str, int], Path]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        exports = write_main(folder, functions)
        mod = Module("main", folder / "main", self.BASE)
        starts, _ = gen.discover(mod)
        return mod, set((mod.lo + starts * 4).tolist()), {k: self.BASE + v for k, v in exports.items()}, folder / "main"

    def lift(self, mod: Module, starts: set[int], entry: int, n: int, spins: dict[int, int] | None):
        e = gen.Emitter(mod, starts, spins=spins)
        lines, _ = e.function(entry, entry + 4 * n)
        return e, "\n".join(lines)

    def test_gate_stands_right_before_the_goto_to_the_head(self) -> None:
        need_capstone()
        mod, starts, at, path = self.load([POLL])
        waits = spin_waits.find_module(path)
        self.assertEqual(len(waits), 1)
        spins = {self.BASE + b: self.BASE + w["head"] for w in waits for b in w["back_edges"]}
        self.assertEqual(spins, {at["f0"] + 4: at["f0"]})
        e, text = self.lift(mod, starts, at["f0"], len(POLL), spins)
        head = at["f0"]
        self.assertRegex(text, rf"if \(.*\) \{{ rt_spin_gate\(c, {head:#x}ull\); goto L_{head:x}; \}}")
        self.assertEqual(text.count("rt_spin_gate"), 1)
        self.assertEqual(e.stats["spin_gates"], 1)

    def test_without_a_table_there_is_no_gate(self) -> None:
        mod, starts, at, _ = self.load([POLL])
        e, text = self.lift(mod, starts, at["f0"], len(POLL), None)
        self.assertNotIn("rt_spin_gate", text)
        self.assertIn(f"goto L_{at['f0']:x};", text)
        self.assertEqual(e.stats["spin_gates"], 0)

    def test_a_listed_word_that_is_no_branch_is_refused(self) -> None:
        mod, starts, at, _ = self.load([POLL])
        ret = at["f0"] + 8
        with self.assertRaisesRegex(gen.LiftError, r"main\+0x18: not a direct branch"):
            gen.Emitter(mod, starts, spins={ret: at["f0"]})

    def test_a_branch_to_another_target_is_refused(self) -> None:
        mod, starts, at, _ = self.load([POLL])
        with self.assertRaisesRegex(gen.LiftError, r"main\+0x14: not a direct branch"):
            gen.Emitter(mod, starts, spins={at["f0"] + 4: at["f0"] + 8})

    def test_a_head_outside_the_function_is_refused(self) -> None:
        # f1 holds `b f0`: a direct branch to the listed head, but the head is in another function.
        _, off = module_with([POLL, [NOP, 0, RET]])
        f1 = [NOP, b_(off["f1"] // 4 + 1, off["f0"] // 4), RET]
        mod, starts, at, _ = self.load([POLL, f1])
        with self.assertRaisesRegex(gen.LiftError, r"main\+0x24: .*outside"):
            gen.Emitter(mod, starts, spins={at["f1"] + 4: at["f0"]})


class LiftRecordTest(unittest.TestCase):
    """The lift step's record carries the spin-wait search's peak memory beside its seconds, and not the code
    generation's (ticket 122: the search was the peak of the whole prepare and nothing recorded it)."""

    TABLE = {"main": [{"head": 16, "back_edges": [20], "size": 2, "kinds": []}]}
    REPORT = {"total_functions": 3, "spin_gates": 1, "modules": {"main": {"stats": {"c_bytes": 100}}}}

    def context(self, tmp: str) -> tuple[Context, io.StringIO]:
        dest = Path(tmp)
        (dest / "exefs").mkdir()
        out = io.StringIO()
        return Context(dest, argparse.Namespace(), Reporter(out, out), state=statefile.load(dest)), out

    def test_the_record_holds_the_searchs_peaks_and_the_lifts_apart(self) -> None:
        def fake(cmd, cwd, on_line=None):
            out = Path(cmd[cmd.index("--out") + 1])
            if any(str(word).endswith("spin_waits.py") for word in cmd):
                out.write_text(json.dumps(self.TABLE), encoding="utf-8")
                return build_game.Measured(0, 12.34, 111 << 20, 222 << 20, [], 333 << 20)
            out.mkdir(parents=True, exist_ok=True)
            (out / "lift_report.json").write_text(json.dumps(self.REPORT), encoding="utf-8")
            return build_game.Measured(0, 5.0, 7 << 20, 8 << 20, [])

        for win in (True, False):   # the report line says what the platform measures (private bytes are Windows's)
            with self.subTest(windows=win), tempfile.TemporaryDirectory() as tmp:
                ctx, out = self.context(tmp)
                (ctx.dest / "exefs" / "main").write_bytes(b"x")
                with mock.patch.object(build_game, "run_measured", fake), mock.patch.object(build_game, "WIN", win):
                    record = program.Lift().run(ctx)
                self.assertEqual((record["spin_search_seconds"], record["spin_search_peak_process"],
                                  record["spin_search_peak_all"], record["spin_search_peak_working_set"]),
                                 (12.3, 111 << 20, 222 << 20, 333 << 20))
                self.assertEqual((record["peak_process"], record["peak_all"]), (7 << 20, 8 << 20))
                say = "peak 111.0 MiB private, 333.0 MiB resident" if win else "peak 333.0 MiB resident"
                self.assertIn(f"1 spin waits found in 12 s, {say}", out.getvalue())

    def test_a_real_search_is_measured(self) -> None:
        need_capstone()
        with tempfile.TemporaryDirectory() as tmp:
            ctx, _ = self.context(tmp)
            write_main(ctx.dest / "exefs", [POLL])
            record = program.Lift().run(ctx)
            self.assertEqual(record["spin_waits"], 1)
            # An interpreter with capstone loaded is tens of MiB, whatever the platform counts it by.
            self.assertGreater(record["spin_search_peak_process"], 8 << 20)
            self.assertGreater(record["spin_search_peak_working_set"], 8 << 20)
            if build_game.WIN:
                self.assertGreaterEqual(record["spin_search_peak_all"], record["spin_search_peak_process"])
            self.assertGreater(record["spin_search_seconds"], 0)


class SpinFlagTest(unittest.TestCase):
    """The commands: spin_waits.py writes the table, gen.py --spin-waits reads it."""

    def run_py(self, script: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", str(script), *args], capture_output=True, text=True, timeout=300)

    @staticmethod
    def exefs(tmp: str) -> Path:
        """The folder of a one-module game (main: POLL), made on first use."""
        exefs = Path(tmp) / "exefs"
        if not exefs.exists():
            exefs.mkdir()
            write_main(exefs, [POLL])
        return exefs

    def test_spin_waits_command_writes_the_table(self) -> None:
        need_capstone()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "build" / "spin_waits.json"
            done = self.run_py(SPIN, "--exefs", str(self.exefs(tmp)), "--out", str(out), "--modules", "main")
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("main: 1 spin waits", done.stderr)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")),
                             {"main": [{"head": 16, "back_edges": [20], "size": 2, "kinds": []}]})

    def test_spin_waits_command_exits_2_on_bad_input(self) -> None:
        need_capstone()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "spin_waits.json"
            done = self.run_py(SPIN, "--exefs", str(Path(tmp) / "missing"), "--out", str(out))
            self.assertEqual(done.returncode, 2)
            done = self.run_py(SPIN, "--exefs", str(self.exefs(tmp)), "--out", str(out), "--modules", "main,sdk")
            self.assertEqual(done.returncode, 2)
            self.assertIn("sdk", done.stderr)
            self.assertFalse(out.exists())

    def lift(self, tmp: str, out: str, *extra: str) -> subprocess.CompletedProcess:
        return self.run_py(GEN, "--exefs", str(self.exefs(tmp)), "--out", str(Path(tmp) / out), "--modules", "main",
                           "--jobs", "1", *extra)

    def chunk(self, tmp: str, out: str) -> str:
        return (Path(tmp) / out / "main" / "main_0000.c").read_text(encoding="utf-8")

    def test_gen_emits_gates_only_with_the_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "spin_waits.json"
            table.write_text(json.dumps({"main": [{"head": 16, "back_edges": [20], "size": 2, "kinds": []}]}),
                             encoding="utf-8")
            plain = self.lift(tmp, "plain")
            self.assertEqual(plain.returncode, 0, plain.stderr)
            self.assertNotIn("rt_spin_gate", self.chunk(tmp, "plain"))
            report = json.loads((Path(tmp) / "plain" / "lift_report.json").read_text(encoding="utf-8"))
            self.assertNotIn("spin_gates", report)
            self.assertNotIn("spin_gates", report["modules"]["main"]["stats"])

            gated = self.lift(tmp, "gated", "--spin-waits", str(table))
            self.assertEqual(gated.returncode, 0, gated.stderr)
            self.assertRegex(self.chunk(tmp, "gated"), r"rt_spin_gate\(c, 0x[0-9a-f]+ull\); goto L_[0-9a-f]+;")
            report = json.loads((Path(tmp) / "gated" / "lift_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["spin_gates"], 1)
            self.assertEqual(report["modules"]["main"]["spin_gates"], 1)
            self.assertEqual(report["modules"]["main"]["stats"]["spin_gates"], 1)

    def test_gen_exits_2_on_a_table_that_does_not_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cases = {
                "not a branch": ({"main": [{"head": 16, "back_edges": [24]}]}, "main+0x18"),
                "unknown module": ({"sdk": [{"head": 16, "back_edges": [20]}]}, "sdk"),
                "malformed": ({"main": [{"head": "16", "back_edges": [20]}]}, "integer head"),
                "twice": ({"main": [{"head": 16, "back_edges": [20, 20]}]}, "twice"),
            }
            for label, (data, text) in cases.items():
                table = Path(tmp) / "table.json"
                table.write_text(json.dumps(data), encoding="utf-8")
                done = self.lift(tmp, "out", "--spin-waits", str(table))
                self.assertEqual(done.returncode, 2, f"{label}: {done.stderr}")
                self.assertIn(text, done.stderr, label)
            table.write_text("{not json", encoding="utf-8")
            self.assertEqual(self.lift(tmp, "out", "--spin-waits", str(table)).returncode, 2)
            missing = self.lift(tmp, "out", "--spin-waits", str(Path(tmp) / "absent.json"))
            self.assertEqual(missing.returncode, 2)
            self.assertTrue(re.search(r"gen: .*absent\.json", missing.stderr), missing.stderr)


if __name__ == "__main__":
    unittest.main()
