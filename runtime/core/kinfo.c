// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The kernel's information and process SVCs (GetInfo, Break, OutputDebugString, ExitProcess, GetSystemTick,
// GetProcessId), the reasons the SVCs left to other layers stay named fatals, and main.npdm's reader.
//
// GetInfo's ids and answers are adapted from DKB Recomp runtime/svc.c at 9d8fa02 (MIT file, adapted there from
// MacNativeGames games/totk/runtime/services.c at b75be8c, MIT): the core and priority masks, no debugger, random
// entropy, the user exception context page, and the Switch 2 ids 29 to 33 the SDK aborts without. Changed here: memory
// ids come from rt_mem_info, the masks and program id from main.npdm, an unknown id is a named fatal of kind GetInfo,
// and Break without the notification bit is a guest stop.
#include <string.h>

#include "kern.h"

#define PROCESS_ID 0x81ull   // the one process; Horizon numbers applications from about 0x80

// 0x29 GetInfo(w1 id, w2 handle, x3 sub id -> x1)
static void svc_get_info(Cpu *c, uint64_t pc) {
    uint32_t id = (uint32_t)c->x[1];
    uint64_t v = 0;
    if (!rt_mem_info(id, &v)) {
        switch (id) {
        case 0: v = K.cfg.core_mask; break;                                // CoreMask
        case 1:                                                            // PriorityMask
            v = (~0ull >> (63 - K.cfg.lowest_priority)) & (~0ull << K.cfg.highest_priority);
            break;
        case 8: v = 0; break;                                              // DebuggerAttached
        case 10: v = 0; break;                                             // IdleTickCount
        case 11:                                                           // RandomEntropy
            if (!plat_random(&v, sizeof v)) rt_fatal("GetInfo 11: no random bytes from the system");
            break;
        case 18: v = K.cfg.program_id; break;                              // ProgramId
        case 20: v = K.exception_context; break;                           // UserExceptionContextAddress
        case 23: v = (uint64_t)K.cfg.is_application; break;                // IsApplication
        // Switch 2 (Switchbrew "SVC", GetInfo [S2]): no remote region is given to the process, and pages are 4 KiB.
        case 29: case 30: case 31: case 32: v = 0; break;
        case 33: v = 0x1000; break;
        default:
            c->pc = pc;
            rt_unimplemented(c, "GetInfo", "id %u (handle %#x, sub id %#llx)", id, (uint32_t)c->x[2],
                             (unsigned long long)c->x[3]);
        }
    }
    c->x[0] = KR_OK;
    c->x[1] = v;
}

// 0x26 Break(x0 reason, x1 info, x2 info size): bit 31 of the reason marks a notification, which returns.
static void svc_break(Cpu *c, uint64_t pc) {
    uint32_t result = 0;
    if (c->x[2] >= 4 && rt_mem_commit_io(c->x[1], 4, 0)) result = LD32(c->x[1]);
    if (c->x[0] & 0x80000000ull) {
        rt_log("guest Break notification %#llx (result %#x)", (unsigned long long)c->x[0], result);
        c->x[0] = KR_OK;
        return;
    }
    c->pc = pc;
    rt_guest_stop(c, "Break reason %#llx (result %#x): the SDK aborted", (unsigned long long)c->x[0], result);
}

// 0x27 OutputDebugString(x0 text, x1 size): one log line, at most 4 KiB.
static void svc_output_debug_string(Cpu *c, uint64_t pc) {
    (void)pc;
    uint64_t p = c->x[0], n = c->x[1] < 4096 ? c->x[1] : 4096;
    if (n && !rt_mem_commit_io(p, n, 0)) {
        c->x[0] = KR_INVALID_CURRENT_MEMORY;
        return;
    }
    char buf[4097];
    memcpy(buf, (const void *)(uintptr_t)p, n);
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) n--;
    buf[n] = 0;
    rt_log("guest: %s", buf);
    c->x[0] = KR_OK;
}

static void svc_exit_process(Cpu *c, uint64_t pc) {   // 0x07 ExitProcess: the host process ends, code 0
    (void)c;
    char at[96];
    rt_log("guest ExitProcess at %s", rt_guest_name(pc, at, sizeof at));
    rt_run_stop_hook(0, "");
    plat_exit(0);
}

static void svc_get_system_tick(Cpu *c, uint64_t pc) {   // 0x1E -> x0 (the SDK reads CNTPCT_EL0 instead)
    (void)pc;
    c->x[0] = rt_ticks();
}

static void svc_get_process_id(Cpu *c, uint64_t pc) {   // 0x24 (w1 process or thread handle -> x1)
    (void)pc;
    uint32_t h = (uint32_t)c->x[1];
    klock();
    int ok = h == H_CURRENT_PROCESS || h_kind(h) == O_THREAD;
    kunlock();
    c->x[0] = ok ? KR_OK : KR_INVALID_HANDLE;
    if (ok) c->x[1] = PROCESS_ID;
}

void kinfo_register(void) {
    rt_svc_register(0x07, svc_exit_process);
    rt_svc_register(0x1E, svc_get_system_tick);
    rt_svc_register(0x24, svc_get_process_id);
    rt_svc_register(0x26, svc_break);
    rt_svc_register(0x27, svc_output_debug_string);
    rt_svc_register(0x29, svc_get_info);
    // The SVCs an application may declare that stay named fatals here, and why.
    static const char service[] = "the service layer serves it, and it is not written yet";
    rt_svc_defer(0x13, service);   // MapSharedMemory: a service's shared memory
    rt_svc_defer(0x14, service);   // UnmapSharedMemory
    rt_svc_defer(0x15, service);   // CreateTransferMemory: memory lent to a service
    rt_svc_defer(0x1F, service);   // ConnectToNamedPort
    rt_svc_defer(0x21, service);   // SendSyncRequest
    rt_svc_defer(0x22, service);   // SendSyncRequestWithUserBuffer
    rt_svc_defer(0x23, service);   // SendAsyncRequestWithUserBuffer
    rt_svc_defer(0x20, "light IPC, for which the SDK has no wrapper");
    rt_svc_defer(0x28, "it returns from a user exception handler, and the runtime delivers no user exception");
    rt_svc_defer(0x80, "only the NVN driver calls it, and the NVN2 front end replaces the driver");
}

// ---- main.npdm: META, then ACI0 and its kernel capabilities (Switchbrew, "NPDM")
static uint32_t u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t u64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }

int rt_npdm_parse(const void *npdm, uint64_t size, RtKernelConfig *out, const char **why) {
    const uint8_t *d = npdm;
    memset(out, 0, sizeof *out);
    *why = NULL;
    if (size < 0x80 || memcmp(d, "META", 4)) {
        *why = "no META header";
        return 0;
    }
    out->main_priority = d[0xE];
    out->main_core = d[0xF];
    out->system_resource_size = u32(d + 0x14);
    out->main_stack_size = u32(d + 0x1C);
    uint32_t aoff = u32(d + 0x70), asize = u32(d + 0x74);
    if (aoff > size || asize > size - aoff || asize < 0x40 || memcmp(d + aoff, "ACI0", 4)) {
        *why = "no ACI0 section inside the file";
        return 0;
    }
    const uint8_t *a = d + aoff;
    out->program_id = u64(a + 0x10);
    uint32_t koff = u32(a + 0x30), ksize = u32(a + 0x34);
    if (koff > asize || ksize > asize - koff || ksize % 4) {
        *why = "kernel capabilities outside ACI0";
        return 0;
    }
    int thread_info = 0;
    for (uint32_t i = 0; i < ksize; i += 4) {
        uint32_t cap = u32(a + koff + i);
        if (cap == 0xFFFFFFFFu) continue;
        switch (__builtin_ctz(~cap)) {   // a capability's kind is its number of trailing one bits
        case 3: {                        // ThreadInfo
            int lo = (int)(cap >> 4 & 63), hi = (int)(cap >> 10 & 63), cmin = (int)(cap >> 16 & 0xFF), cmax = (int)(cap >> 24);
            if (hi > lo || cmin > cmax || cmax >= 64) {
                *why = "a malformed ThreadInfo capability";
                return 0;
            }
            out->lowest_priority = lo;
            out->highest_priority = hi;
            out->core_mask = (~0ull >> (63 - cmax)) & (~0ull << cmin);
            thread_info = 1;
            break;
        }
        case 4:                          // EnableSystemCalls: 24 SVCs per word, the index in bits 29-31
            for (int b = 0; b < 24; b++)
                if (cap >> (5 + b) & 1) {
                    unsigned svc = (cap >> 29) * 24 + (unsigned)b;
                    if (svc < 256) out->svc_mask[svc >> 6] |= 1ull << (svc & 63);
                }
            break;
        case 13: out->is_application = (cap >> 14 & 7) == 1; break;   // MiscParams: the program type
        case 15: out->handle_table_size = cap >> 16 & 0x3FF; break;    // HandleTableSize
        default: break;
        }
    }
    if (!thread_info) {
        *why = "no ThreadInfo capability";
        return 0;
    }
    return 1;
}
