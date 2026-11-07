// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The packet ring between the submitting thread and the recorder (architecture D12, section 6.1; runtime/gpu/gpu_ring.c).
// One producer and one consumer share a ring of 64-bit words. A packet is a header word (an operation number and the
// packet's length in words) and a payload the producer fills in place; the consumer reads the payload where it lies, so
// a packet may hold pointers into itself. A packet never wraps: one that does not fit before the end of the ring is
// preceded by a pad that the consumer skips. Nothing here is Vulkan, so the ring is tested without a device.
//
// The producer publishes after each packet (gpu_ring_publish); the consumer releases a packet once it has made its
// calls (gpu_ring_release), and the producer waits for room when the ring is full (back-pressure, never an overwrite).
// An idle consumer spins for a moment and then sleeps; a publish wakes it.
//
// Design: Savage Wind Recomp's and LAS Recomp's recorder rings, as DKB Recomp's runtime/vk_rec.c carries them (see
// THIRD_PARTY_NOTICES.md); the code is ours.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define GPU_RING_BYTES (16u << 20)   // architecture D12
#define GPU_RING_OP_PAD 0u           // the pad's operation number: the consumer never returns it

typedef struct GpuRing GpuRing;

typedef struct GpuRingStats {
    uint64_t packets, words;         // put, pads not counted
    uint64_t pads;                   // wasted at the end of the ring so that a packet does not wrap
    uint64_t full_waits;             // times the producer found the ring full
    uint64_t full_wait_ns;           // and the time it spent waiting for room
    uint64_t consumer_sleeps;        // times the consumer slept on an empty ring
} GpuRingStats;

// A ring of `bytes` (a power of two, 4 KiB at least; a test uses a small one to make it wrap). NULL on failure.
GpuRing *gpu_ring_new(size_t bytes);
void gpu_ring_free(GpuRing *r);
size_t gpu_ring_bytes(const GpuRing *r);
// Most a packet's payload may hold: a quarter of the ring less its header.
size_t gpu_ring_max_payload(const GpuRing *r);

// ---- the producer (one thread at a time)
// Room for a packet of operation op (not GPU_RING_OP_PAD) with `bytes` of payload (rounded up to 8), waiting for the
// consumer while the ring is full. The caller fills the payload and calls gpu_ring_publish before its next put or at
// the end of its work; a packet bigger than gpu_ring_max_payload is a named fatal.
void *gpu_ring_put(GpuRing *r, uint32_t op, size_t bytes);
// Everything put so far (every packet filled) becomes the consumer's.
void gpu_ring_publish(GpuRing *r);
// Publish, then wait until the consumer has released every packet put so far.
void gpu_ring_drain(GpuRing *r);

// ---- the consumer (one thread)
// The next published packet: its payload, with its operation and payload size set, or NULL when none came within
// wait_ns (0: do not wait, UINT64_MAX: no limit). The packet stays the consumer's until gpu_ring_release.
const void *gpu_ring_get(GpuRing *r, uint32_t *op, size_t *bytes, uint64_t wait_ns);
void gpu_ring_release(GpuRing *r);

void gpu_ring_stats(const GpuRing *r, GpuRingStats *out);
// Tests only: while non-zero, the producer does not wait for room (the counter-run of back-pressure: a ring that
// overwrites packets nobody read).
extern int gpu_ring_overwrite;
