// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Shared between the kernel's files (kernel.c: threads and virtual cores; ksync.c: synchronization objects; kinfo.c:
// information and process SVCs). Everything here runs under the kernel lock K.g unless a comment says otherwise.
#pragma once
#include "core.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "plat.h"

// Horizon kernel results: module 1, description << 9 (Switchbrew, "Error codes").
#define KR_OK 0u
#define KR_INVALID_ADDRESS 0xCC01u          // 102
#define KR_OUT_OF_RESOURCE 0xCE01u          // 103
#define KR_OUT_OF_HANDLES 0xD201u           // 105
#define KR_INVALID_CURRENT_MEMORY 0xD401u   // 106
#define KR_INVALID_PRIORITY 0xE001u         // 112
#define KR_INVALID_CORE_ID 0xE201u          // 113
#define KR_INVALID_HANDLE 0xE401u           // 114
#define KR_INVALID_POINTER 0xE601u          // 115
#define KR_INVALID_COMBINATION 0xE801u      // 116
#define KR_TIMED_OUT 0xEA01u                // 117
#define KR_CANCELLED 0xEC01u                // 118
#define KR_OUT_OF_RANGE 0xEE01u             // 119
#define KR_INVALID_ENUM 0xF001u             // 120
#define KR_BUSY 0xF401u                     // 122
#define KR_INVALID_STATE 0xFA01u            // 125

#define H_CURRENT_THREAD 0xFFFF8000u
#define H_CURRENT_PROCESS 0xFFFF8001u
#define WAITERS_BIT 0x40000000u             // in a guest mutex word: other threads wait for it (ArbitrateLock)
#define MAX_CORES 64

enum { O_FREE, O_THREAD, O_EVENT_R, O_EVENT_W, O_HOST };  // handle kinds; O_HOST: rt_kernel_handle_new
enum { W_NONE, W_LOCK, W_CONDVAR, W_ADDRESS, W_SYNC };    // what a blocked thread waits for
enum { T_CREATED, T_STARTED, T_EXITED };

typedef struct KEvent { int signaled, refs; } KEvent;
typedef struct KHost { void *obj; void (*close)(void *obj); } KHost;

typedef struct RtThread RtThread;
typedef struct KFibre KFibre;
struct RtThread {
    Cpu cpu;                     // first: Cpu wants 16-byte alignment, which calloc gives. The thread's own context
    Cpu *cur;                    // the Cpu running guest code on this thread now: &cpu, or its fibre's (changed under K.g)
    uint32_t handle;
    uint64_t id;
    uint64_t entry, arg, stack_top;
    int prio, ideal;             // base priority (inheritance is computed, prio_of), ideal core
    uint64_t mask;               // cores the thread may run on
    int state, closed, is_main;
    int cancel;                  // CancelSynchronization came while not waiting: the next wait returns Cancelled
    int pause_req, paused;       // SetThreadActivity
    PlatCond cv;                 // every wait of this thread sleeps on it, under K.g
    PlatThread *host;
    uint64_t tl;                 // guest address of its thread-local block: TLR, thread object, TLS
    uint64_t exit_buf[32];       // rt_host_setjmp at the thread's start; ExitThread lands there
    // a wait (W_*)
    int wkind, woke;
    uint64_t waddr, wkey, wseq;
    uint32_t wtag, wres;
    RtThread *wnext;             // K.waiters
    // the handles of the WaitSynchronization it is in, or was last in when it timed out (rt_kernel_waiters), under K.g
    uint32_t ws_h[64];
    int ws_n, ws_in, ws_run;     // handles kept; in a wait now; the run of timed-out waits on this set goes on
    uint64_t ws_since, ws_end;   // start of the wait, or of the run of timed-out waits on the same set; end of the last wait
    // the virtual core
    int cur_core;                // the core held, -1 for none (written atomically: waiters spin on it)
    int ready, kicked;           // in K.ready_q; asked by a better thread to give its core up
    uint64_t cseq;
    RtThread *cnext;             // K.ready_q
    RtThread *target;            // as a ready thread: the owner it asked to give a core up
    // asynchronous preemption
    uint64_t ap_rip, ap_last;
    int ap_pending;
    // spin waits (written by the thread itself, without the lock)
    uint64_t sw_head, sw_t0, sw_last, spin_passes;
    // time: host CPU and virtual-core holding, total and at the last frame mark
    uint64_t core_since, core_ns, cpu_final, mark_cpu, mark_core, frame_cpu, frame_core;
    RtThread *all;               // K.all
    // fibres (fibre.c), written by the thread itself
    PlatFibre *own;              // the host thread's own context as a host fibre, from its first switch
    KFibre *fibre;               // the guest fibre it runs, NULL on its own context
    KFibre *leaving;             // the context it left at its last switch, parked by whatever it switched to
    int exit_pending;            // ExitThread on a fibre: ends once back on the thread's own context
};

typedef struct Kernel {
    PlatMutex g;
    PlatCond done;               // rt_kernel_run waits on it for the last thread
    RtKernelConfig cfg;
    int ready;
    struct { int kind; uint32_t id; void *obj; } *h;   // handle table, index 0 unused
    uint32_t hcap, hnext, hid;
    RtThread *all, *waiters, *ready_q, *owner[MAX_CORES];
    uint64_t seq, next_thread_id, live;
    uint64_t exception_context;  // GetInfo 20, allocated at init
    int async_on;
    uintptr_t code_lo[16], code_end[16];   // each module's lifted code [lo, end), for asynchronous preemption
    int code_n;
    RtKernelStats st;
} Kernel;

extern Kernel K;

// Whether an interrupted host pc lies in lifted code (any module's [lowest function, end marker)), so it may be sent
// to the preemption stub. The ranges are loaded by kern_load_code_ranges, which rt_kernel_run calls (the lifted-code end marker).
void kern_load_code_ranges(void);
int kern_pc_lifted(uintptr_t pc);

void klock(void);
void kunlock(void);
RtThread *kcurrent(void);                      // the calling guest thread, NULL on a host thread
RtThread *kthread(uint32_t svc);               // the calling guest thread; fatal on a host thread
uint32_t h_new(int kind, void *obj);           // 0 when the table is full
void *h_get(uint32_t h, int kind);             // NULL for a bad handle or another kind
int h_kind(uint32_t h);                        // O_FREE for a bad handle
void h_free(uint32_t h);

uint32_t kblock(RtThread *t, int kind, int64_t ns);   // wait until kwake or the timeout; 0 ns is TimedOut at once
void kwake(RtThread *t, uint32_t res);
void kwake_sync(void);                         // every WaitSynchronization re-checks its objects
void kkick(void);                              // priorities changed: ask owners to give cores up where due
int kthread_free_if_done(RtThread *t);         // frees an exited thread whose handle is closed; 1 when freed

// guest memory words under the lock
static inline uint32_t kld32(uint64_t a) { return __atomic_load_n((uint32_t *)(uintptr_t)a, __ATOMIC_SEQ_CST); }
static inline void kst32(uint64_t a, uint32_t v) { __atomic_store_n((uint32_t *)(uintptr_t)a, v, __ATOMIC_SEQ_CST); }
static inline int kcas32(uint64_t a, uint32_t *expected, uint32_t v) {
    return __atomic_compare_exchange_n((uint32_t *)(uintptr_t)a, expected, v, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

void ksync_register(void);
void kinfo_register(void);

// fibre.c: ExitThread called on a fibre goes back to the thread's own context first and does not return; a thread that
// ends gives its host fibre up on its own context.
__attribute__((noreturn)) void kfibre_exit_thread(RtThread *t);
void kfibre_thread_end(RtThread *t);
uint64_t kfibre_addr(const KFibre *f);   // the guest FiberType, 0 for none
