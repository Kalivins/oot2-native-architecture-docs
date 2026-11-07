// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The run-time half of architecture D3: guest setjmp and longjmp over the host's, and the C++ throw entry points as
// guest stops. OoT's main calls setjmp at 3 sites and longjmp at 4, its sdk at 10 and 7; the game throws only on
// error paths, so a throw stops the run with its type rather than unwinding.
//
// RT_SETJMP (cpu.h) runs the host setjmp (hostjmp.c) in the lifted function's own frame and keeps, per guest jmp_buf,
// the guest's callee-saved state in a record of the thread's Cpu. rt_longjmp restores that state and lands in the
// frame with the host longjmp, which restores the host's callee-saved registers: the function that ran RT_SETJMP
// keeps every guest register in the Cpu, so the guest state it reads after landing is the restored one.
//
// A record lives while its function's frame does. Frames are compared by address (the host stack grows down): a
// record made in a frame below the current one belongs to a function that has returned or was jumped over.
#include "core.h"

#define LONGJMP_LOG_LINES 32

static uint64_t g_setjmps, g_longjmps;

void rt_jmp_counts(uint64_t *setjmps, uint64_t *longjmps) {
    *setjmps = __atomic_load_n(&g_setjmps, __ATOMIC_RELAXED);
    *longjmps = __atomic_load_n(&g_longjmps, __ATOMIC_RELAXED);
}

static void save_state(const Cpu *c, RtJmp *j) {
    for (int r = 0; r < 12; r++) j->gpr[r] = c->x[19 + r];
    j->gpr[12] = c->sp;
    for (int r = 0; r < 8; r++) j->fpr[r] = (uint64_t)c->v[8 + r];
}

static void restore_state(Cpu *c, const RtJmp *j) {
    for (int r = 0; r < 12; r++) c->x[19 + r] = j->gpr[r];
    c->sp = j->gpr[12];
    for (int r = 0; r < 8; r++) c->v[8 + r] = j->fpr[r];   // writing a D register clears the upper half, as ldp d8 does
}

RtJmp *rt_jmp_enter(Cpu *c, uint64_t env, uintptr_t frame) {
    RtJmp *same = NULL, *free_slot = NULL;
    for (int i = 0; i < RT_JMP_SLOTS; i++) {
        RtJmp *j = &c->jmp[i];
        if (j->env && j->frame < frame) j->env = 0;   // a function below this one: gone
        if (!j->env) {
            if (!free_slot) free_slot = j;
        } else if (j->env == env) {
            same = j;   // setjmp on a buffer refills it, whichever frame filled it before
        }
    }
    RtJmp *j = same ? same : free_slot;
    if (!j) {
        char at[96];
        rt_fatal("setjmp at %s: more than %d guest jmp_bufs live on this thread", rt_guest_name(c->x[30], at, sizeof at),
                 RT_JMP_SLOTS);
    }
    j->env = env;
    j->frame = frame;
    j->value = 0;
    save_state(c, j);
    __atomic_fetch_add(&g_setjmps, 1, __ATOMIC_RELAXED);
    return j;
}

void rt_jmp_leave(Cpu *c, uintptr_t frame) {
    for (int i = 0; i < RT_JMP_SLOTS; i++)
        if (c->jmp[i].env && c->jmp[i].frame <= frame) c->jmp[i].env = 0;
}

void rt_longjmp(Cpu *c, uint64_t env, uint32_t value) {
    char lr[96];
    uintptr_t here = (uintptr_t)__builtin_frame_address(0);
    RtJmp *j = NULL;
    for (int i = 0; i < RT_JMP_SLOTS && !j; i++)
        if (c->jmp[i].env == env && env) j = &c->jmp[i];
    if (!j || j->frame <= here)
        rt_fatal("longjmp to guest jmp_buf %#llx from %s: no live setjmp filled it", (unsigned long long)env,
                 rt_guest_name(c->x[30], lr, sizeof lr));
    uint64_t n = __atomic_fetch_add(&g_longjmps, 1, __ATOMIC_RELAXED) + 1;
    if (n <= LONGJMP_LOG_LINES)
        rt_log("longjmp %llu: jmp_buf %#llx, value %u, from %s", (unsigned long long)n, (unsigned long long)env, value,
               rt_guest_name(c->x[30], lr, sizeof lr));
    restore_state(c, j);
    j->value = value ? value : 1;
    for (int i = 0; i < RT_JMP_SLOTS; i++)
        if (c->jmp[i].env && c->jmp[i].frame < j->frame) c->jmp[i].env = 0;   // jumped over
    rt_host_longjmp(j->host, 1);
}

// ---- guest symbols bound to the runtime (the loader replaces both imports and definitions)

static void host_longjmp(Cpu *c) { rt_longjmp(c, c->x[0], (uint32_t)c->x[1]); }
HOST_OVERRIDE("longjmp", host_longjmp);

// The lifter compiles every direct call to setjmp to RT_SETJMP; only a call through a pointer gets here.
static void host_setjmp(Cpu *c) { rt_unimplemented(c, "setjmp", "setjmp called through a pointer"); }
HOST_OVERRIDE("setjmp", host_setjmp);

// __cxa_throw(object, std::type_info *, destructor): the type's name is the second word of the type_info.
static void host_cxa_throw(Cpu *c) {
    uint64_t tinfo = c->x[1];
    const char *name = tinfo ? (const char *)(uintptr_t)LD64(tinfo + 8) : NULL;
    rt_guest_stop(c, "C++ throw of %s (object %#llx)", name ? name : "(no type)", (unsigned long long)c->x[0]);
}
HOST_OVERRIDE("__cxa_throw", host_cxa_throw);

static void host_cxa_rethrow(Cpu *c) { rt_guest_stop(c, "C++ rethrow"); }
HOST_OVERRIDE("__cxa_rethrow", host_cxa_rethrow);

static void host_unwind_resume(Cpu *c) { rt_guest_stop(c, "_Unwind_Resume outside an unwind"); }
HOST_OVERRIDE("_Unwind_Resume", host_unwind_resume);

// The sdk's unwinder entries the lifter replaces with these (tools/lift/gen.py): a throw that bypassed __cxa_throw.
static void host_unwind_raise(Cpu *c) { rt_guest_stop(c, "_Unwind_RaiseException (a C++ throw)"); }
HOST_OVERRIDE("_Unwind_RaiseException", host_unwind_raise);

static void host_unwind_resume_or_rethrow(Cpu *c) { rt_guest_stop(c, "_Unwind_Resume_or_Rethrow (a C++ rethrow)"); }
HOST_OVERRIDE("_Unwind_Resume_or_Rethrow", host_unwind_resume_or_rethrow);
