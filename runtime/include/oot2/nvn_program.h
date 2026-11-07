// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// A program stage's identity (runtime/nvn/nvn_program.c, architecture D10, D16, D21): what a back end looks a stage up
// by. ProgramSetShaders reads each stage's control section and code area once, while the control section is still the
// game's to read, and keeps the result here: the job key the shader step of prepare writes (tools/shaders/split.py,
// "<key16>.<variant>"), the stage kind and the code area's size from the control section, and a copy of the control
// bytes the key and the job are made from, so a back end that misses a stage in its cache has what a translator job
// needs. A record is made once per distinct key and never freed: programs copied by value keep a valid pointer.
#pragma once
#include <stdint.h>

#include "nvn_obj.h"

// The stage kinds a translator job exists for, as split.py names them.
enum { NVN_STAGE_VS, NVN_STAGE_PS, NVN_STAGE_CS, NVN_STAGE_TS, NVN_STAGE_MS, NVN_STAGE_KINDS };

#define NVN_STAGE_CTL_MAX 0xBC   // control bytes a compute stage's key and job read; the other kinds read fewer
#define NVN_PROG_MAGIC MAGIC('P', 'R', 'O', 'G')   // prog_t.magic of an initialised program

typedef struct nvn_stage {
    char key[26];            // "<key16>.<variant>": 16 and 8 lowercase hex digits, NUL-terminated
    uint8_t kind;            // NVN_STAGE_*
    uint8_t ctl_size;        // bytes of ctl that hold the control section (0x57, 0x6C or NVN_STAGE_CTL_MAX)
    uint32_t code_size;      // the code area's bytes (control +0x44)
    uint32_t inst_size;      // its instructions' bytes, the SPH included (control +0x38)
    uint8_t ctl[NVN_STAGE_CTL_MAX];
} nvn_stage;

const char *nvn_stage_kind_name(int kind);   // "vs", "ps", "cs", "ts", "ms"; NULL for another value
// Stage i of program p (i below p->nstage), or NULL.
static inline const nvn_stage *nvn_program_stage(const prog_t *p, int i) {
    return i >= 0 && (uint32_t)i < p->nstage ? p->st[i].id : NULL;
}
int nvn_stage_records(void);   // distinct stage records made so far (tests)
