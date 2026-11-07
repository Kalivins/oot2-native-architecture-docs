// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

namespace Ryujinx.Graphics.Shader
{
    /// <summary>
    /// The contract between the translator's raw-access helper and the runtime that builds the page table (architecture D15). The runtime's
    /// copy is runtime/include/oot2/raw_table.h; tests/test_raw_abi.py fails when the two differ, so a number changed here is changed there.
    ///
    /// Push constants, bytes 0 to 15 (the first 16 of the 128 every stage has; the texture slots follow them): four words.
    ///   word 0, 1: the table's device address (low, high), 16-byte aligned
    ///   word 2:    the first chunk of the span, a 2 MiB chunk number (guest address &gt;&gt; 21)
    ///   word 3:    the number of chunks in the span, at least 1
    /// An address is mapped only when its chunk lies in [span base, span base + span count).
    ///
    /// The table, in 32-bit words from its device address:
    ///   [0, 16)                 header: version, span base, span count (copies of the push constants), leaf blocks in use, miss count
    ///   [16, 272)               miss log, 64 entries of 4 words: address low, address high, kind, reason
    ///   [272, 262416)           store bits, one per leaf slot (leaf block n, page p: (n - 1) * 512 + p), word s / 32, bit s % 32
    ///   [262416, + span count)  top level, one word per chunk of the span: 0 when nothing in the chunk is mapped, else the chunk's leaf block n (1 to capacity)
    ///   after the top level     leaf blocks, 1,024 words each: 512 entries of 2 words, one per 4 KiB page: the delta from guest to device address
    ///                           (low, high); the page is mapped when bit 0 of the low word is set, and the other bits of an entry whose bit 0 is
    ///                           clear mean nothing (a delta of 0 is a usable delta; an unmapped page is any entry with bit 0 clear, zero included).
    ///                           A mapped entry is (delta | 1): the delta is a multiple of 4 KiB, so its own bit 0 is clear, and the helper clears
    ///                           bit 0 before it adds the delta
    ///
    /// A miss (outside the span, an unmapped chunk or an unmapped page) reads 0 and drops a store or an atomic. The helper takes the next index
    /// of the miss log with an atomic add on the miss count and writes the entry when the index is below the log's size. A store or an
    /// atomic that lands sets its leaf slot's store bit (it reads the word first and or-s only when the bit is clear).
    /// </summary>
    public static class RawTableAbi
    {
        public const int Version = 1;

        public const int PushTableLow = 0;
        public const int PushTableHigh = 1;
        public const int PushSpanBase = 2;
        public const int PushSpanCount = 3;
        public const int PushBytes = 16;

        public const int ChunkLog2 = 21;
        public const int PageLog2 = 12;
        public const int LeafEntriesLog2 = 9;
        public const int EntryWords = 2;
        public const int LeafBlockWords = 1024;

        public const int HeaderVersion = 0;
        public const int HeaderSpanBase = 1;
        public const int HeaderSpanCount = 2;
        public const int HeaderLeafBlocks = 3;
        public const int HeaderMissCount = 4;
        public const int HeaderWords = 16;

        public const int MissLogWord = 16;
        public const int MissEntries = 64;
        public const int MissEntryWords = 4;

        public const int MissAddressLow = 0;
        public const int MissAddressHigh = 1;
        public const int MissKind = 2;
        public const int MissReason = 3;

        public const int MissKindLoad = 1;
        public const int MissKindStore = 2;
        public const int MissKindAtomic = 3;

        public const int MissReasonOutsideSpan = 1;
        public const int MissReasonNoChunk = 2;
        public const int MissReasonNoPage = 3;

        public const int LeafBlockCapacity = 16384;
        public const int StoreBitsWord = 272;
        public const int StoreBitsWords = 262144;
        public const int TopWord = 262416;
    }
}
