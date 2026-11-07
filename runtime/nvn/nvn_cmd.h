// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Internal header of the command buffer files: nvn_cmd.c (the object, its memory, recordings, the copies behind
// command handles, submit) and nvn_cmd_ops.c (the commands).
#pragma once
#include <stddef.h>

#include "nvn_api.h"

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
