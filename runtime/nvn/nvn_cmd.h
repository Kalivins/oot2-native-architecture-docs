// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Internal header of the command buffer files: nvn_cmd.c (the object, its memory, recordings, the copies behind
// command handles, submit) and nvn_cmd_ops.c (the commands).
#pragma once
#include <stddef.h>
#include <stdio.h>

#include "nvn_api.h"

// The D11 probe (nvn_cmd.c; census mode, setting census_probe): the counts, and the file census-probe.tsv. A run
// that is not in census mode with the key set has no probe: nvn_probe_on is 0, nothing is hashed, no file is written.
#define NVN_PROBE_HIST 40   // size classes of the ranges added: 2^k to 2^(k+1) bytes
typedef struct {
    uint64_t bytes_read, bytes_zero;   // bytes of committed pages the probe read, and of uncommitted pages it did not read
    uint64_t buffers, ranges, bytes_hashed, bytes_checked, unreadable, no_access, over_cap, unchecked_bytes, hits, recordings;
    uint64_t hist[NVN_PROBE_HIST];
} NvnProbeStats;
int nvn_probe_on(void);
void nvn_probe_stats(NvnProbeStats *out);
void nvn_probe_write(FILE *f);

// Room for one record in the command buffer of entry point c (its x0): the header written, the payload zeroed. A
// recording the game began itself (its inline BeginRecording) is taken up first; a pointer the front end keeps that
// the game moved is a named stop. The argument registers are as the game passed them when this returns, even when the
// game's memory callback ran inside.
void *nvn_rec(Cpu *c, uint32_t op, size_t payload);
#define REC(c, OP, T) ((T *)nvn_rec((c), (OP), sizeof(T)))
// A record of an array holds the fields before ARR and its first N elements only.
#define RECN(c, OP, T, ARR, N) ((T *)nvn_rec((c), (OP), offsetof(T, ARR) + sizeof(((T *)0)->ARR[0]) * (size_t)(N)))
// The record just made reads game memory again at submit (a *Deferred bind or push): the copy is resolved then.
void nvn_rec_deferred(const Cpu *c);
