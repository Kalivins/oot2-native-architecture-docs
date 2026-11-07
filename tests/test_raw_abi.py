# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The raw-access table's ABI is one contract written in two places: the translator's RawTableAbi.cs and the runtime's
raw_table.h. These tests keep the two in step, and keep the numbers consistent with each other and with the push
constant layout the uni descriptor layout reserves.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import raw_abi


class RawAbiTest(unittest.TestCase):
    def test_the_two_sides_name_the_same_constants(self):
        c, cs = raw_abi.header_constants(), raw_abi.cs_constants()
        self.assertGreater(len(c), 30, "the header was not read")
        self.assertEqual(sorted(set(c) ^ set(cs)), [], "a constant exists on one side only")

    def test_the_two_sides_agree_on_every_value(self):
        c, cs = raw_abi.header_constants(), raw_abi.cs_constants()
        self.assertEqual({k: (c[k], cs[k]) for k in c if c[k] != cs[k]}, {})

    def test_the_regions_follow_each_other(self):
        a = raw_abi.A
        self.assertEqual(a["MissLogWord"], a["HeaderWords"])
        self.assertEqual(a["StoreBitsWord"], a["MissLogWord"] + a["MissEntries"] * a["MissEntryWords"])
        self.assertEqual(a["StoreBitsWords"] * 32, a["LeafBlockCapacity"] << a["LeafEntriesLog2"], "one bit per leaf slot")
        self.assertEqual(a["TopWord"], a["StoreBitsWord"] + a["StoreBitsWords"])
        self.assertEqual(a["LeafBlockWords"], a["EntryWords"] << a["LeafEntriesLog2"])
        self.assertEqual(1 << (a["ChunkLog2"] - a["PageLog2"]), 1 << a["LeafEntriesLog2"], "a leaf block holds the pages of one chunk")

    def test_the_miss_entry_and_the_header_fit_their_regions(self):
        a = raw_abi.A
        self.assertEqual(sorted((a["MissAddressLow"], a["MissAddressHigh"], a["MissKind"], a["MissReason"])), list(range(a["MissEntryWords"])))
        header = (a["HeaderVersion"], a["HeaderSpanBase"], a["HeaderSpanCount"], a["HeaderLeafBlocks"], a["HeaderMissCount"])
        self.assertEqual(len(set(header)), len(header), "two header fields share a word")
        self.assertLess(a["HeaderMissCount"], a["HeaderWords"])
        self.assertEqual(len({a["MissKindLoad"], a["MissKindStore"], a["MissKindAtomic"]}), 3)
        self.assertEqual(len({a["MissReasonOutsideSpan"], a["MissReasonNoChunk"], a["MissReasonNoPage"]}), 3)
        self.assertNotIn(0, (a["MissKindLoad"], a["MissKindStore"], a["MissKindAtomic"], a["MissReasonOutsideSpan"], a["MissReasonNoChunk"], a["MissReasonNoPage"]),
                         "0 is an empty log word")

    def test_the_push_words_are_the_first_bytes_before_the_texture_slots(self):
        a = raw_abi.A
        self.assertEqual(sorted((a["PushTableLow"], a["PushTableHigh"], a["PushSpanBase"], a["PushSpanCount"])), [0, 1, 2, 3])
        self.assertEqual(a["PushBytes"], 16)
        self.assertEqual(raw_abi.uni_slots_offset(), a["PushBytes"], "the uni layout's texture slots start where the raw words end")


if __name__ == "__main__":
    unittest.main()
