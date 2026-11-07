// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The guest threads and the kernel objects under the SDK (architecture D7). One host thread per guest thread, with a
// 256 MiB reserved host stack and its own Cpu. The program's virtual cores come from its main.npdm; a guest thread runs
// guest code only while it holds one, so at most that many run at once, as on the console, where the game's per-core
// data and its spin locks rely on it. The SDK's nn::os runs as lifted code; only the kernel under it is the host.
//
// Scheduling, as Horizon's: the best priority (the lowest number) first, then first come. A thread that becomes ready
// takes a free core it may run on, or asks the worst owner worse than itself to give one up; the owner does at its
// next safe point (rt_dispatch entry, SVC exit, runtime.h), or, when it runs lifted code without reaching one, after a
// short wait the host thread is interrupted and sent through a stub that gives the core up (asynchronous preemption).
// An owner whose SDK spin lock section is marked in its thread-local region (preemption disabled) switches at the
// SynchronizePreemptionState SVC instead. A thread that waits, sleeps or blocks in a host call gives its core up.
// A spin wait the lifter gated (rt_spin_gate) gives its core to a waiting thread once it has lasted spin_handoff_ns.
//
// nn::os fibres run on host fibres with a Cpu each (runtime/core/fibre.c): the four public fibre entry points are host
// functions. A thread that switches fibres keeps its virtual core, its thread-local region, its handle, its FPCR and
// FPSR and its CPU time; the fibre takes its registers and its guest and host stacks, and may later run on another
// thread.
//
// Start-up order: rt_mem_init, rt_kernel_init (registers the SVCs and the safe-point hook), rt_load_modules, host
// registrations, rt_dispatch_init, then rt_kernel_run.
#pragma once
#include <stdint.h>

#include "runtime.h"

// What the program's main.npdm says about its threads, and two policy settings.
typedef struct RtKernelConfig {
    uint64_t core_mask;                      // virtual cores the program may use (ThreadInfo: min..max core)
    int highest_priority, lowest_priority;   // ThreadInfo: the best (lowest number) and the worst priority allowed
    int main_priority, main_core;            // META: the main thread's priority and ideal core
    uint64_t main_stack_size;                // META
    uint64_t system_resource_size;           // META: for RtMemConfig.system_resource_size
    uint32_t handle_table_size;              // HandleTableSize capability; 0 means 1024
    uint64_t svc_mask[4];                    // EnableSystemCalls capabilities; all zero leaves every SVC allowed
    uint64_t program_id;                     // ACI0
    int is_application;                      // MiscParams: program type 1
    uint64_t spin_handoff_ns;                // 0 means 50 us
    int no_async_preempt;                    // 1: owners give their cores up at safe points only
} RtKernelConfig;

// main.npdm (META, ACI0 and its kernel capabilities) into a configuration: 1, or 0 with the reason in *why.
int rt_npdm_parse(const void *npdm, uint64_t size, RtKernelConfig *out, const char **why);

// Register the thread, synchronization and information SVCs and the safe-point hook. After rt_mem_init, before
// rt_dispatch_init; fatal on a malformed configuration.
void rt_kernel_init(const RtKernelConfig *cfg);
// Run the program: its main thread starts at entry with x0 = arg and x1 = its own handle, on a stack of
// main_stack_size at the top of the stack region. Returns when every guest thread has ended (ExitProcess ends the
// host process instead). Called once, on a host thread, after rt_dispatch_init.
void rt_kernel_run(uint64_t entry, uint64_t arg);

// A host function about to block (a file read, a wait on the host) gives its thread's virtual core up, and takes one
// again after: other guest threads run meanwhile, as they would while the console's thread waits on a service.
int rt_kernel_block_begin(void);       // returns what rt_kernel_block_end takes
void rt_kernel_block_end(int token);

// Kernel events for host services: the writable end's handle is returned, the readable end's in *readable. Any thread
// may signal, host threads included; the guest waits on the readable end (WaitSynchronization).
uint32_t rt_kernel_event_create(uint32_t *readable);
void rt_kernel_event_signal(uint32_t writable);

// Handles to host objects (a service session, transfer memory). The close function names the object's kind: get
// returns the object only for a handle made with the same function, and CloseHandle calls it once the handle is gone,
// outside the kernel lock, so it may take the service's own locks. new returns 0 when the handle table is full.
uint32_t rt_kernel_handle_new(void *obj, void (*close)(void *obj));
void *rt_kernel_handle_get(uint32_t h, void (*close)(void *obj));

// Per-thread time (D26, from M1): rt_kernel_frame marks a frame boundary; rt_kernel_thread_times gives, per thread not
// yet freed, the host CPU time and the time it held a virtual core over the last frame and in total. Returns the
// number of threads, of which at most max are written.
typedef struct RtThreadTime {
    uint32_t handle;
    int priority, core;          // core: the virtual core held now, -1 for none
    uint64_t entry;
    uint64_t cpu_ns, core_ns;    // over the last frame
    uint64_t cpu_total_ns, core_total_ns;
    int exited;
} RtThreadTime;
void rt_kernel_frame(void);
int rt_kernel_thread_times(RtThreadTime *out, int max);
// The handle of the guest thread that calls it, 0 on a host thread (the M1 capture names the submitting thread with it).
uint32_t rt_kernel_current(void);

typedef struct RtKernelStats {
    uint64_t threads_created, threads_live;
    uint64_t safepoint_yields;   // cores given up at a safe point for a better thread
    uint64_t async_sent;         // owners interrupted in lifted code and sent to the preemption stub
    uint64_t async_taken;        // of those, owners that gave their core up there
    uint64_t async_elsewhere;    // interruptions that found the owner outside lifted code (it went on)
    uint64_t async_failed;       // interruptions the host refused
    uint64_t deferred;           // preemptions put off to SynchronizePreemptionState
    uint64_t spin_passes, spin_handoffs;
    uint64_t fibres_created, fibres_live, fibre_switches;   // nn::os fibres (fibre.c)
    int async_on;                // asynchronous preemption is in use
} RtKernelStats;
RtKernelStats rt_kernel_stats(void);
// For a run that hangs: one log line per core owner and per thread, then the counters. Reads without the kernel lock,
// which the hang may hold, so a line can mix two moments.
void rt_kernel_dump(void);
