// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The SVC table. Each SVC the runtime implements is registered by the module that owns it (memory, threads, IPC);
// every other one is the named fatal "unimplemented svc 0x.. Name", followed by the reason its owner recorded with
// rt_svc_defer, so a run that stops on one is a census line (architecture D9). An SVC the program's mask (main.npdm)
// does not allow is a guest stop, as the console's kernel refuses it. The names are the kernel's public ones
// (Switchbrew, "SVC").
#include "core.h"

static const char *const svc_names[256] = {
    [0x01] = "SetHeapSize", [0x02] = "SetMemoryPermission", [0x03] = "SetMemoryAttribute",
    [0x04] = "MapMemory", [0x05] = "UnmapMemory", [0x06] = "QueryMemory", [0x07] = "ExitProcess",
    [0x08] = "CreateThread", [0x09] = "StartThread", [0x0A] = "ExitThread", [0x0B] = "SleepThread",
    [0x0C] = "GetThreadPriority", [0x0D] = "SetThreadPriority", [0x0E] = "GetThreadCoreMask",
    [0x0F] = "SetThreadCoreMask", [0x10] = "GetCurrentProcessorNumber", [0x11] = "SignalEvent",
    [0x12] = "ClearEvent", [0x13] = "MapSharedMemory", [0x14] = "UnmapSharedMemory",
    [0x15] = "CreateTransferMemory", [0x16] = "CloseHandle", [0x17] = "ResetSignal",
    [0x18] = "WaitSynchronization", [0x19] = "CancelSynchronization", [0x1A] = "ArbitrateLock",
    [0x1B] = "ArbitrateUnlock", [0x1C] = "WaitProcessWideKeyAtomic", [0x1D] = "SignalProcessWideKey",
    [0x1E] = "GetSystemTick", [0x1F] = "ConnectToNamedPort", [0x20] = "SendSyncRequestLight",
    [0x21] = "SendSyncRequest", [0x22] = "SendSyncRequestWithUserBuffer",
    [0x23] = "SendAsyncRequestWithUserBuffer", [0x24] = "GetProcessId", [0x25] = "GetThreadId",
    [0x26] = "Break", [0x27] = "OutputDebugString", [0x28] = "ReturnFromException", [0x29] = "GetInfo",
    [0x2A] = "FlushEntireDataCache", [0x2B] = "FlushDataCache", [0x2C] = "MapPhysicalMemory",
    [0x2D] = "UnmapPhysicalMemory", [0x2E] = "GetDebugFutureThreadInfo", [0x2F] = "GetLastThreadInfo",
    [0x30] = "GetResourceLimitLimitValue", [0x31] = "GetResourceLimitCurrentValue",
    [0x32] = "SetThreadActivity", [0x33] = "GetThreadContext3", [0x34] = "WaitForAddress",
    [0x35] = "SignalToAddress", [0x36] = "SynchronizePreemptionState", [0x37] = "GetResourceLimitPeakValue",
    [0x39] = "CreateIoPool", [0x3A] = "CreateIoRegion", [0x3C] = "KernelDebug",
    [0x3D] = "ChangeKernelTraceState", [0x40] = "CreateSession", [0x41] = "AcceptSession",
    [0x42] = "ReplyAndReceiveLight", [0x43] = "ReplyAndReceive", [0x44] = "ReplyAndReceiveWithUserBuffer",
    [0x45] = "CreateEvent", [0x46] = "MapIoRegion", [0x47] = "UnmapIoRegion",
    [0x48] = "MapPhysicalMemoryUnsafe", [0x49] = "UnmapPhysicalMemoryUnsafe", [0x4A] = "SetUnsafeLimit",
    [0x4B] = "CreateCodeMemory", [0x4C] = "ControlCodeMemory", [0x4D] = "SleepSystem",
    [0x4E] = "ReadWriteRegister", [0x4F] = "SetProcessActivity", [0x50] = "CreateSharedMemory",
    [0x51] = "MapTransferMemory", [0x52] = "UnmapTransferMemory", [0x53] = "CreateInterruptEvent",
    [0x54] = "QueryPhysicalAddress", [0x55] = "QueryMemoryMapping", [0x56] = "CreateDeviceAddressSpace",
    [0x57] = "AttachDeviceAddressSpace", [0x58] = "DetachDeviceAddressSpace",
    [0x59] = "MapDeviceAddressSpaceByForce", [0x5A] = "MapDeviceAddressSpaceAligned",
    [0x5C] = "UnmapDeviceAddressSpace", [0x5D] = "InvalidateProcessDataCache",
    [0x5E] = "StoreProcessDataCache", [0x5F] = "FlushProcessDataCache", [0x60] = "DebugActiveProcess",
    [0x61] = "BreakDebugProcess", [0x62] = "TerminateDebugProcess", [0x63] = "GetDebugEvent",
    [0x64] = "ContinueDebugEvent", [0x65] = "GetProcessList", [0x66] = "GetThreadList",
    [0x67] = "GetDebugThreadContext", [0x68] = "SetDebugThreadContext", [0x69] = "QueryDebugProcessMemory",
    [0x6A] = "ReadDebugProcessMemory", [0x6B] = "WriteDebugProcessMemory", [0x6C] = "SetHardwareBreakPoint",
    [0x6D] = "GetDebugThreadParam", [0x6F] = "GetSystemInfo", [0x70] = "CreatePort",
    [0x71] = "ManageNamedPort", [0x72] = "ConnectToPort", [0x73] = "SetProcessMemoryPermission",
    [0x74] = "MapProcessMemory", [0x75] = "UnmapProcessMemory", [0x76] = "QueryProcessMemory",
    [0x77] = "MapProcessCodeMemory", [0x78] = "UnmapProcessCodeMemory", [0x79] = "CreateProcess",
    [0x7A] = "StartProcess", [0x7B] = "TerminateProcess", [0x7C] = "GetProcessInfo",
    [0x7D] = "CreateResourceLimit", [0x7E] = "SetResourceLimitLimitValue", [0x7F] = "CallSecureMonitor",
};

static RtSvcFn handlers[256];
static const char *reasons[256];   // why an SVC without a handler stays a named fatal (rt_svc_defer)
static uint64_t mask[4] = {~0ull, ~0ull, ~0ull, ~0ull};   // the program's SVC mask; every SVC until one is set
static uint64_t calls[256];
static uint64_t first_pc[256];   // the guest pc of each SVC's first call, for the census
static int sealed;

void svc_seal(void) { sealed = 1; }

void rt_svc_register(uint32_t imm, RtSvcFn fn) {
    if (sealed) rt_fatal("rt_svc_register(%#x) after rt_dispatch_init", imm);
    if (imm > 0xFF || !fn) rt_fatal("rt_svc_register: bad entry %#x", imm);
    if (handlers[imm]) rt_fatal("rt_svc_register: svc %#x registered twice", imm);
    handlers[imm] = fn;
}

void rt_svc_defer(uint32_t imm, const char *reason) {
    if (sealed) rt_fatal("rt_svc_defer(%#x) after rt_dispatch_init", imm);
    if (imm > 0xFF || !reason || !*reason) rt_fatal("rt_svc_defer: bad entry %#x", imm);
    reasons[imm] = reason;
}

void rt_svc_set_mask(const uint64_t m[4]) {
    if (sealed) rt_fatal("rt_svc_set_mask after rt_dispatch_init");
    for (int i = 0; i < 4; i++) mask[i] = m[i];
}

int rt_svc_allowed(uint32_t imm) { return imm <= 0xFF && (mask[imm >> 6] >> (imm & 63) & 1); }

const char *rt_svc_name(uint32_t imm) { return imm <= 0xFF ? svc_names[imm] : NULL; }

int rt_svc_status(uint32_t imm, const char **reason) {
    *reason = imm <= 0xFF ? reasons[imm] : NULL;
    if (imm <= 0xFF && handlers[imm]) return RT_SVC_IMPLEMENTED;
    return *reason ? RT_SVC_NAMED : RT_SVC_MISSING;
}

void rt_svc_census(void) {
    int n[3] = {0, 0, 0};
    for (uint32_t imm = 0; imm <= 0xFF; imm++) {
        if (!rt_svc_allowed(imm) || (!svc_names[imm] && !handlers[imm] && !reasons[imm])) continue;
        const char *why, *name = svc_names[imm] ? svc_names[imm] : "(no known name)";
        int s = rt_svc_status(imm, &why);
        n[s]++;
        if (s == RT_SVC_IMPLEMENTED) rt_log("svc %#04x %s: implemented", imm, name);
        else if (s == RT_SVC_NAMED) rt_log("svc %#04x %s: named fatal, %s", imm, name, why);
        else rt_log("svc %#04x %s: named fatal, no reason recorded", imm, name);
    }
    rt_log("svc census: %d implemented, %d named fatals with a reason, %d without one", n[RT_SVC_IMPLEMENTED],
           n[RT_SVC_NAMED], n[RT_SVC_MISSING]);
}

uint64_t rt_svc_calls(uint32_t imm) { return imm <= 0xFF ? __atomic_load_n(&calls[imm], __ATOMIC_RELAXED) : 0; }

uint64_t rt_svc_first(uint32_t imm) { return imm <= 0xFF ? __atomic_load_n(&first_pc[imm], __ATOMIC_RELAXED) : 0; }

__attribute__((noinline, noreturn, cold)) static void svc_missing(Cpu *c, uint32_t imm, uint64_t pc) {
    const char *name = rt_svc_name(imm), *why = imm <= 0xFF ? reasons[imm] : NULL;
    c->pc = pc;
    if (!rt_svc_allowed(imm))
        rt_guest_stop(c, "svc %#04x %s: the program's SVC mask does not allow it, so the console's kernel refuses it", imm,
                      name ? name : "(no known name)");
    rt_unimplemented(c, "svc", "%#04x %s%s%s", imm, name ? name : "(no known name)", why ? ": " : "", why ? why : "");
}

void rt_svc(Cpu *c, uint32_t imm, uint64_t pc) {
    RtSvcFn fn = imm <= 0xFF ? handlers[imm] : NULL;
    if (imm <= 0xFF && !__atomic_fetch_add(&calls[imm], 1, __ATOMIC_RELAXED))
        __atomic_store_n(&first_pc[imm], pc, __ATOMIC_RELAXED);
    if (__builtin_expect(!fn || !rt_svc_allowed(imm), 0)) svc_missing(c, imm, pc);
    fn(c, pc);
    if (__builtin_expect(__atomic_load_n(&c->pending, __ATOMIC_RELAXED) != 0, 0)) rt_safepoint(c);
}
