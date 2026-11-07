# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The runtime's page-table builder (runtime/gpu/raw_table.c) against the translator's raw-access helpers.

tests/runtime/raw_table_test.c writes the table of a list of backing runs the way the Vulkan back end does (raw_layout,
raw_write), and the test walks that table with the load and store helpers of the translator's own SPIR-V, through
tests/test_xlate_raw.py's evaluator: every mapped page reads at its device address, a later run's page replaces an
earlier one's, deltas below the guest address and past 4 GiB carry, the window for pools the CPU cannot see resolves,
an unmapped page, an unmapped chunk and an address outside the span read zero with the reason the miss log names, and a
store sets exactly the leaf slot the builder lists for a mirrored page (the slot the write-back shader reads). The
builder's refusals are named. So the two sides of the ABI are checked against each other, not each against its own
model.

Needs clang (OOT2_CC or PATH) and the translator (py -3 tools/build_xlate.py); skips without them, unless
OOT2_REQUIRE_CLANG or OOT2_REQUIRE_XLATE is set.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from raw_abi import A
from test_runtime_gpu import EXE, RUNTIME, clang_or_skip
from test_xlate_host import Harness, compute_job, EXIT
from test_xlate_raw import LDG, M32, STG, TABLE_ADDRESS, TID, Memory, Module, run, scheduled

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "runtime" / "raw_table_test.c"
CHUNK = 1 << A["ChunkLog2"]
PAGE = 1 << A["PageLog2"]
EXTENT, IMPORT, MIRROR = 1, 2, 3   # oot2/backing.h, BACKING_*
WINDOW = 1 << 39                   # pools the CPU cannot see
PROBE = 0x10                       # the offset in each page the tests read

# (lo, hi, device address, guest address, kind); in the order backing_raw_runs lists them: imports, extents and virtual
# runs first, mirrors last
RUNS = [
    (0x10_0000_0000, 0x10_0000_3000, 0x7F_0000_1000, 0x10_0000_0000, IMPORT),    # delta past 4 GiB
    (0x10_001F_E000, 0x10_0060_2000, 0x3_0000_0000, 0x10_001F_E000, IMPORT),     # four chunks, delta below the guest
    (WINDOW + 0x1234_0000, WINDOW + 0x1235_0000, 0x3_4000_0000, WINDOW + 0x1234_0000, EXTENT),
    (0x20_0000_0000, 0x20_0000_2000, 0x5_1000_0000, 0x18_0000_0000, MIRROR),     # a virtual pool's run onto a mirror
    (0x10_0000_1000, 0x10_0000_2000, 0x5_0000_0000, 0x10_0000_1000, MIRROR),     # replaces the first run's middle page
]


def page_map(runs) -> dict[int, tuple[int, int, int]]:
    """Each mapped page's device address, guest address and kind, a later run's replacing an earlier one's."""
    pages = {}
    for lo, hi, address, guest, kind in runs:
        for a in range(lo, hi, PAGE):
            pages[a] = (address + a - lo, guest + a - lo, kind)
    return pages


class RawTableBuilderTest(Harness):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cc = clang_or_skip()
        cls._build = tempfile.TemporaryDirectory(prefix="oot2-rawtable-", ignore_cleanup_errors=True)
        cls.exe = Path(cls._build.name) / f"raw_table_test{EXE}"
        done = subprocess.run([cc, "-std=gnu11", "-O2", "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror",
                               "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), str(SOURCE),
                               str(RUNTIME / "gpu" / "raw_table.c"), "-o", str(cls.exe)], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"build of {SOURCE.name} failed:\n{done.stdout}{done.stderr}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_build"):
            cls._build.cleanup()

    def setUp(self):
        super().setUp()
        code = self.dir / "raw.bin"
        code.write_bytes(b"".join(scheduled(i) for i in TID + [LDG, STG]) + EXIT)
        got = self.run_jobs([compute_job("raw", code)])["raw"]
        self.assertEqual((got.status, got.message), ("ok", ""))
        self.module = Module((self.out / "raw.spv").read_bytes())
        self.assertTrue(json.loads((self.out / "raw.res.json").read_text(encoding="utf-8"))["raw_access"]["store_bits"])
        helpers = self.module.raw_helpers()
        self.load_fn = next(f for f in helpers if len(f["params"]) == 2)
        self.store_fn = next(f for f in helpers if len(f["params"]) == 3)

    def build(self, runs) -> tuple[str, dict, dict, list]:
        """The builder's output for runs: its first line, the push words, the table's nonzero words by device address and
        the mirrored pages (slot, device address, guest address)."""
        listing = self.dir / "runs.txt"
        listing.write_text("".join(f"{lo:x} {hi:x} {a:x} {g:x} {k}\n" for lo, hi, a, g, k in runs), encoding="ascii")
        table = self.dir / "table.bin"
        done = subprocess.run([str(self.exe), str(listing), str(table)], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        lines = done.stdout.splitlines()
        if lines[0].startswith("refused"):
            return lines[0], {}, {}, []
        span_base, span_count, _blocks, words = (int(x) for x in lines[0].split()[1:5])
        data = table.read_bytes()
        self.assertEqual(len(data), 4 * words)
        self.assertLessEqual(len(data), Memory.TABLE_BYTES)
        device = {TABLE_ADDRESS + 4 * i: w for i, w in enumerate(struct.unpack(f"<{words}I", data)) if w}
        push = {A["PushTableLow"]: TABLE_ADDRESS & M32, A["PushTableHigh"]: TABLE_ADDRESS >> 32,
                A["PushSpanBase"]: span_base, A["PushSpanCount"]: span_count}
        mirrors = [tuple(int(x, 16) for x in line.split()[1:]) for line in lines[1:]]
        return lines[0], push, device, mirrors

    def word(self, table: dict, index: int) -> int:
        return table.get(TABLE_ADDRESS + 4 * index, 0)

    def load(self, push, table, data, guest) -> int:
        return run(self.module, self.load_fn, [guest & M32, guest >> 32], Memory(TABLE_ADDRESS, table, data, push))

    def store(self, push, table, data, guest, value):
        run(self.module, self.store_fn, [guest & M32, guest >> 32, value], Memory(TABLE_ADDRESS, table, data, push))

    def misses(self, table) -> list[tuple[int, int, int]]:
        out = []
        for n in range(self.word(table, A["HeaderMissCount"])):
            base = A["MissLogWord"] + n * A["MissEntryWords"]
            out.append((self.word(table, base + A["MissAddressLow"]) | self.word(table, base + A["MissAddressHigh"]) << 32,
                        self.word(table, base + A["MissKind"]), self.word(table, base + A["MissReason"])))
        return out

    def store_bits(self, table) -> set[int]:
        first = TABLE_ADDRESS + 4 * A["StoreBitsWord"]
        return {(a - first) // 4 * 32 + b for a, v in table.items() if first <= a < first + 4 * A["StoreBitsWords"]
                for b in range(32) if v >> b & 1}

    def test_the_header_holds_the_layout_and_the_logs_start_empty(self):
        first, push, table, _ = self.build(RUNS)
        _, span_base, span_count, blocks, *_ = first.split()
        chunks = {a // CHUNK for a in page_map(RUNS)}
        self.assertEqual(int(span_base), min(chunks))
        self.assertEqual(int(span_count), max(chunks) - min(chunks) + 1)
        self.assertEqual(int(blocks), len(chunks))
        self.assertEqual([self.word(table, A[k]) for k in ("HeaderVersion", "HeaderSpanBase", "HeaderSpanCount",
                                                          "HeaderLeafBlocks", "HeaderMissCount")],
                         [A["Version"], min(chunks), int(span_count), len(chunks), 0])
        self.assertEqual(self.store_bits(table), set())
        self.assertEqual(self.misses(table), [])

    def test_every_mapped_page_reads_at_its_device_address(self):
        _, push, table, _ = self.build(RUNS)
        pages = page_map(RUNS)
        data = {address + PROBE: (n * 0x01010101 + 7) & M32 for n, (address, _, _) in enumerate(pages.values())}
        for n, (guest, (address, _, kind)) in enumerate(pages.items()):
            with self.subTest(guest=hex(guest), kind=kind):
                self.assertEqual(self.load(push, table, data, guest + PROBE), data[address + PROBE])
        self.assertEqual(self.misses(table), [])

    def test_a_later_run_replaces_an_earlier_runs_page(self):
        _, push, table, _ = self.build(RUNS)
        guest = 0x10_0000_1000
        data = {0x7F_0000_2000 + PROBE: 1, 0x5_0000_0000 + PROBE: 2}
        self.assertEqual(self.load(push, table, data, guest + PROBE), 2, "the mirror listed after the import wins")
        self.assertEqual(self.load(push, table, data, guest - PAGE + PROBE), 0, "the import's first page is its own")

    def test_the_lowest_and_the_highest_mapped_page_resolve(self):
        _, push, table, _ = self.build(RUNS)
        pages = page_map(RUNS)
        for guest in (min(pages), max(pages)):
            with self.subTest(guest=hex(guest)):
                address = pages[guest][0]
                self.assertEqual(self.load(push, table, {address + PAGE - 4: 0x5EED}, guest + PAGE - 4), 0x5EED)
        self.assertEqual(self.misses(table), [])

    def test_misses_read_zero_with_the_reason_named(self):
        _, push, table, _ = self.build(RUNS)
        pages = page_map(RUNS)
        cases = [
            (0x10_0000_3000, A["MissReasonNoPage"]),                       # past the first run, in its chunk
            (0x10_0100_0000, A["MissReasonNoChunk"]),                      # inside the span, no run touches the chunk
            (min(pages) - CHUNK, A["MissReasonOutsideSpan"]),
            ((max(pages) // CHUNK + 1) * CHUNK, A["MissReasonOutsideSpan"]),
        ]
        data = {address + PROBE: 0xDEAD for address, _, _ in pages.values()}
        for guest, _ in cases:
            self.assertEqual(self.load(push, table, data, guest + PROBE), 0, hex(guest))
        self.assertEqual(self.misses(table), [(g + PROBE, A["MissKindLoad"], reason) for g, reason in cases])

    def test_a_store_sets_the_slot_the_builder_lists_for_its_mirrored_page(self):
        _, push, table, mirrors = self.build(RUNS)
        pages = page_map(RUNS)
        mirrored = {guest: v for guest, v in pages.items() if v[2] == MIRROR}
        self.assertEqual(sorted((a, g) for _, a, g in mirrors), sorted((a, g) for a, g, _ in mirrored.values()),
                         "one entry per mirrored page, with its device address and the guest address it stands for")
        slot_of = {a: slot for slot, a, _ in mirrors}
        for guest, (address, _, _) in mirrored.items():
            with self.subTest(guest=hex(guest)):
                before = self.store_bits(table)
                data = {}
                self.store(push, table, data, guest + PROBE, 0xA5A5)
                self.assertEqual(data, {address + PROBE: 0xA5A5})
                self.assertEqual(self.store_bits(table) - before, {slot_of[address]})
        self.assertEqual(self.misses(table), [])

    def test_a_store_into_an_import_sets_a_slot_no_mirror_lists(self):
        _, push, table, mirrors = self.build(RUNS)
        data = {}
        self.store(push, table, data, 0x10_0040_0000 + PROBE, 3)
        self.assertEqual(data, {0x3_0000_0000 + 0x10_0040_0000 - 0x10_001F_E000 + PROBE: 3})
        (slot,) = self.store_bits(table)
        self.assertNotIn(slot, {s for s, _, _ in mirrors})

    def test_no_runs_make_a_table_where_every_access_misses(self):
        first, push, table, mirrors = self.build([])
        self.assertEqual(first.split()[1:4], ["0", "1", "0"], "span base 0, count 1, no leaf block")
        self.assertEqual(mirrors, [])
        self.assertEqual(self.load(push, table, {}, 0x1000), 0)
        self.assertEqual(self.misses(table), [(0x1000, A["MissKindLoad"], A["MissReasonNoChunk"])])

    def test_the_builder_refuses_by_name(self):
        lo = 0x10_0000_0000
        cases = {
            "more than the 16384 leaf blocks": [(0, (A["LeafBlockCapacity"] + 1) * CHUNK, 0x1_0000_0000, 0, IMPORT)],
            "its delta is not a multiple of 4 KiB": [(lo, lo + PAGE, 0x2_0000_0800, lo, IMPORT)],
            "is not whole 4 KiB pages": [(lo, lo + 0x800, 0x2_0000_0000, lo, IMPORT)],
        }
        for text, runs in cases.items():
            with self.subTest(text):
                first, *_ = self.build(runs)
                self.assertTrue(first.startswith("refused"), first)
                self.assertIn(text, first)


if __name__ == "__main__":
    unittest.main()
