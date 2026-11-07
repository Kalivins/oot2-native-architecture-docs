// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Shared between the files of runtime/core, never included by generated code or other runtime modules.
#pragma once
#include "oot2/runtime.h"

// hostfn.c: every synthetic address in the host-function window dispatches here; c->pc names the slot.
void hostfn_entry(Cpu *c);
GuestFn hostfn_lookup(uint64_t addr);   // hostfn_entry for a registered slot, else NULL
void hostfn_seal(void);
// svc.c
void svc_seal(void);
// memory.c, for the loader: commit and seal module pages (counted in the module region), then hand over the module so
// that QueryMemory, SetMemoryPermission and the first-touch handler know it (bss is committed on touch).
typedef struct RtLoaded RtLoaded;
void mem_module_commit(uint64_t addr, uint64_t size);
void mem_module_seal(uint64_t addr, uint64_t size);
void mem_module_add(const RtLoaded *m);
// memory.c, for the kernel: committed, zeroed pages of the kernel region (thread-local regions, the user exception
// context), which QueryMemory reports as ThreadLocal; and the main thread's stack at the top of the stack region,
// committed on touch, which returns the stack's top. Fatal when guest memory is not set up or the space is used up.
uint64_t mem_kernel_alloc(uint64_t size);
void mem_kernel_free(uint64_t addr, uint64_t size);
uint64_t mem_main_stack(uint64_t size);
// dispatch.c, for the kernel's asynchronous preemption: the host range of each module's lifted code, [lowest function
// start, end marker) (after rt_dispatch_init; none before). Fills up to max pairs, returns how many modules have one.
int dispatch_code_ranges(uintptr_t *lo, uintptr_t *end, int max);
// dispatch.c, for the page ranges (memory.c): whether a host pc lies in a lifted function, and which guest function
// starts there, by the function tables. The index is built on first use (or by dispatch_index_host, outside a fault);
// the extent of a function is up to the next function start in host address order, or its module's end marker.
void dispatch_index_host(void);
int dispatch_host_pc_guest(uintptr_t host_pc, uint64_t *guest_fn);

// clock.c: the guest system counter counts COUNTER_HZ per second from its origin, the host time (plat_now_ns) of
// process start; one count is COUNTER_NS nanoseconds exactly, so rt_ticks() is
// (plat_now_ns() - counter_origin_ns()) / COUNTER_NS.
#define COUNTER_HZ 31250000u   // Switch 2
#define COUNTER_NS 32u
uint64_t counter_origin_ns(void);
