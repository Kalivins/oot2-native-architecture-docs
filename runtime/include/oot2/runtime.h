// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The runtime side of the boundary: what the build glue, the loader and the runtime's own modules use to set up the
// tables the generated code runs through. Generated code includes cpu.h only.
//
// Start-up order: register host functions and SVC handlers, then rt_dispatch_init, which seals every registry; guest
// threads start after it, so lookups take no lock.
//
// Fixed guest address map (the modules are placed by tools/layout.py, written to layout.h):
//   modules           MOD_<NAME>_BASE, from 0x70_0000_0000, all below RT_MODULE_LIMIT
//   host functions    RT_HOSTFN_BASE, 16 bytes per function, never mapped: synthetic addresses only
// Every guest-visible address stays below 2^39, the limit of the addresses IPC buffer descriptors carry.
#pragma once
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "cpu.h"

#define RT_ADDRESS_LIMIT (1ull << 39)
#define RT_MODULE_LIMIT 0x7800000000ull   // tools/layout.py MODULE_LIMIT
#define RT_HOSTFN_BASE 0x7ff0000000ull
#define RT_HOSTFN_SIZE 0x100000ull        // 65,536 functions
#define RT_RET_HOST (RT_HOSTFN_BASE + RT_HOSTFN_SIZE - 16)   // x30 while host code calls guest code (rt_call)

// ---- exit codes, so a run that stops can be sorted without reading its log
enum {
    RT_EXIT_FATAL = 70,           // the runtime or the build broke a rule (bad table, dispatch to nothing, ...)
    RT_EXIT_UNIMPLEMENTED = 71,   // the game asked for something the runtime does not have yet (census)
    RT_EXIT_GUEST_STOP = 72,      // the guest stopped itself: brk, an undefined instruction, a C++ throw
    RT_EXIT_CENSUS = 73,          // a run in census mode ended at its frame limit (OOT2_CENSUS_FRAMES): never a default result
};

// ---- logging and fatal errors (fatal.c). One line per call, to stderr.
void rt_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void rt_vlog(const char *prefix, const char *fmt, va_list ap);
void rt_fatal(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
// A named fatal: "FATAL: unimplemented <kind>: <message> (pc ..., lr ...)", exit RT_EXIT_UNIMPLEMENTED. Every
// unimplemented SVC, IPC command, import or host function ends here, so a run that stops is one census line.
void rt_unimplemented(const Cpu *c, const char *kind, const char *fmt, ...)
    __attribute__((noreturn, format(printf, 3, 4)));
// The same, followed by the guest backtrace from x29 (rt_guest_backtrace): for a gap met inside SDK code, whose
// callers name the path that reached it.
void rt_unimplemented_trace(const Cpu *c, const char *kind, const char *fmt, ...)
    __attribute__((noreturn, format(printf, 3, 4)));
// The guest stopped itself: "FATAL: guest stop: <message> (pc ..., lr ...)" and a guest backtrace from x29.
void rt_guest_stop(const Cpu *c, const char *fmt, ...) __attribute__((noreturn, format(printf, 2, 3)));
// The guest's frame records from x29, one log line each (at most 16): the SDK functions whose path led to c.
void rt_guest_backtrace(const Cpu *c);
// Called once when the run stops (any fatal, a guest stop, or the process ending through plat_exit), after the FATAL
// line is written: the exit code and that line without its newline ("" when the run ends without one). A fault inside
// the hook ends the process at once with the same code. One hook; set it before rt_dispatch_init.
void rt_set_stop_hook(void (*hook)(int code, const char *line));
// A second slot, for a writer that must save what it holds when the run stops (the ledger): called first, once, by
// the same stop path, whatever the stop hook is. Set it once, before the run starts.
void rt_set_stop_flush(void (*flush)(void));
// Writers that record the stop itself (the M1 capture's closing record): called once each, in the order added, with the
// exit code and the stop line, after the flush and before the stop hook. At most four; add them before the run starts.
void rt_add_stop_writer(void (*writer)(int code, const char *line));
void rt_run_stop_hook(int code, const char *line);

// ---- modules and indirect dispatch (dispatch.c, architecture D8)
//
// The lifter writes, for each module <mod>, three tables in its function-table file:
//   const uint64_t <mod>_fn_addrs[];   guest function starts, strictly ascending
//   const GuestFn <mod>_fn_ptrs[];     the lifted function for each start
//   const uint64_t <mod>_fn_count;
//   void <mod>_lifted_end(Cpu *);      the end marker: linked after the module's last lifted code (ticket 92)
// and the build glue describes each module with an RtModule (base and size from layout.h).
//
// The end marker is what says where a module's lifted code stops. The tables give function starts only, so without it
// the extent of the last function would run into whatever the linker placed next (runtime code, another module's
// helpers), and host code would be taken for guest code by every consumer that asks "is this host address lifted?":
// the lookup rt_host_pc_module below, the "who read" index of the page ranges (memory.h) and the kernel's
// asynchronous preemption (kernel.h). code_end is the address of the marker, the first byte that is not the module's.
typedef struct RtModule {
    const char *name;
    uint64_t base, size;
    const uint64_t *addrs;
    const GuestFn *fns;
    uint64_t count;
    const void *code_end;   // address of <mod>_lifted_end; every function of fns lies below it; fatal when NULL
} RtModule;

// Build the flat table: one 32-bit slot per guest instruction word from the lowest function start to past the
// highest, holding the host function's offset from an anchor; then make it read-only and seal the registries.
// Fatal on a malformed input: unsorted or duplicate starts, a start outside its module, overlapping modules, a module
// past RT_MODULE_LIMIT, a host function inside a module, lifted code spread over 4 GiB of host addresses, a module
// with no end marker, a function at or past its module's end marker.
void rt_dispatch_init(const RtModule *mods, int count);
// The load address of the module registered under name, or 0 when there is none (the pacing trace's probes, nvn_trace.c).
uint64_t rt_module_base(const char *name);
// Which module's lifted code a host address lies in: from its lowest function to the byte before its end marker.
// The marker's own address and everything past it is host code (NULL), and so is the gap between two modules.
const char *rt_host_pc_module(uintptr_t host_pc);
int rt_sealed(void);
GuestFn rt_dispatch_lookup(uint64_t pc);   // NULL when nothing is there; runs no guest code
// "0x7000001234 (main+0x1234)", "0x7ff0000010 (host nn::os::Foo)", or the bare address. Returns buf.
const char *rt_guest_name(uint64_t pc, char *buf, int size);
// A host function at a fixed guest address outside every module (before rt_dispatch_init). Most host functions go
// through the host-function table below instead.
void rt_register_host(uint64_t addr, GuestFn fn);

typedef struct RtDispatchStats {
    uint64_t lo, hi;          // guest range the table covers
    uint64_t slots;           // (hi - lo) / 4
    uint64_t table_bytes;     // reserved for the table
    uint64_t functions;       // filled slots
    uint64_t host_fns;        // host functions at fixed addresses (the host-function table counts as one each)
} RtDispatchStats;
RtDispatchStats rt_dispatch_stats(void);

// Safe points: rt_dispatch on entry and rt_svc on exit call rt_safepoint when c->pending is set, once: the guest code
// then runs on to its next safe point whatever c->pending says by then. The thread scheduler installs the hook (before
// rt_dispatch_init); without one, the request is cleared.
void rt_set_safepoint_hook(void (*hook)(Cpu *c));
void rt_safepoint(Cpu *c);

// ---- host functions (hostfn.c)
//
// A host function is a GuestFn at a synthetic address RT_HOSTFN_BASE + 16 * i. It runs on the calling guest thread
// and returns to the guest call site with a C return; on entry c->pc holds its own address. Like an AAPCS64 callee it
// may change x0..x18, x30 and v0..v7, v16..v31, and must leave x19..x29, sp, the low halves of v8..v15, tpidr and
// tpidrro as it found them.
//
// HOST_OVERRIDE(symbol, fn), at file scope in any object linked into the program, binds the guest symbol to fn: the
// loader asks rt_hostfn_bind for every import and every definition it may replace. The entries of every object file
// meet in one linker section, which hostfn.c walks.
typedef struct RtHostEntry { const char *name; GuestFn fn; } RtHostEntry;
#define RT_CAT2(a, b) a##b
#define RT_CAT(a, b) RT_CAT2(a, b)
#ifdef _WIN32
// PE/COFF: the linker orders ".ovr$a" < ".ovr$m" < ".ovr$z" by the text after '$'.
#define RT_HOST_SECTION section(".ovr$m")
#else
// ELF: the linker brackets a section named like a C identifier with __start_ and __stop_ symbols.
#define RT_HOST_SECTION section("oot2_ovr")
#endif
#define HOST_OVERRIDE(sym, fn) \
    __attribute__((used, RT_HOST_SECTION)) static const RtHostEntry RT_CAT(rt_ovr_, __COUNTER__) = {sym, fn}

// The synthetic address of the override for symbol, registering it on first use; 0 when there is none.
uint64_t rt_hostfn_bind(const char *symbol);
// A host function outside the override table (runtime internals, tests). The same (name, fn) gives the same address.
uint64_t rt_hostfn_register(const char *name, GuestFn fn);
// An address whose call is the named fatal "unimplemented import <symbol>": what the loader binds an import to when
// neither a module nor an override defines it.
uint64_t rt_hostfn_missing(const char *symbol);
const char *rt_hostfn_name(uint64_t addr);   // NULL when addr is not a registered host function
uint64_t rt_hostfn_calls(uint64_t addr);     // calls so far
int rt_hostfn_override_count(void);          // HOST_OVERRIDE entries linked into this program
// The census of host functions: slot i in registration order, its calls and the x30 of its first call (0 when never
// called), and whether it is a missing import. NULL past the last slot.
int rt_hostfn_count(void);
const char *rt_hostfn_at(int i, uint64_t *calls, uint64_t *first_lr, int *missing);
// The calls of every slot at once (relaxed reads, not one instant): at most max values into calls, returns the slot count.
int rt_hostfn_snapshot(uint64_t *calls, int max);

// Census mode (ticket 38; research 31 Q5): the setting census at 1 (oot2/settings.h; OOT2_CENSUS=1 in the environment),
// read once, never the default. A run in census mode may go on past a gap that stops a default run (an NVN2 entry point
// without a body returns 0), so every line it writes is marked "[census mode]" (log) or "#census-mode" (files), and its
// normal end has its own exit code, RT_EXIT_CENSUS. The setting census_frames=N (OOT2_CENSUS_FRAMES; census mode only)
// ends the run at its N-th present, after the census and the ledger are written.
int rt_census_on(void);
uint64_t rt_census_frames(void);   // N, or 0 for no limit
// Where the census is rewritten while the run goes on (a run killed by a time limit writes nothing at its end):
// census.c sets it; nvn_census.c calls rt_census_flush every 600 presents.
void rt_census_set_flush(void (*flush)(void));
void rt_census_flush(void);
// Run the stop hooks (census.tsv, the ledger) and end the process with code; line is the stop row's text.
__attribute__((noreturn)) void rt_census_end(int code, const char *line);
// The index-th HOST_OVERRIDE that no import or definition of the loaded modules named (rt_hostfn_bind never registered
// it): an override written for a name this game's SDK does not define. NULL past the last.
const char *rt_hostfn_unbound(int index);

// Host code calling guest code: run the guest function at addr on this thread's Cpu and return when it returns.
// Arguments and results in c->x and c->v as AAPCS64 places them. Guest code may lose its virtual core at any
// instruction (architecture D7: safe points, and asynchronous preemption inside lifted code), so a host function holds
// no lock while guest code runs on its thread.
void rt_call(Cpu *c, uint64_t addr);

// AAPCS64 argument access for host functions.
static inline uint64_t arg_x(const Cpu *c, int i) { return i < 8 ? c->x[i] : LD64(c->sp + 8 * (uint64_t)(i - 8)); }
static inline void *arg_p(const Cpu *c, int i) { return (void *)(uintptr_t)arg_x(c, i); }
static inline float arg_s(const Cpu *c, int i) { uint32_t b = (uint32_t)c->v[i]; float f; memcpy(&f, &b, 4); return f; }
static inline double arg_d(const Cpu *c, int i) { uint64_t b = (uint64_t)c->v[i]; double d; memcpy(&d, &b, 8); return d; }
static inline void ret_x(Cpu *c, uint64_t v) { c->x[0] = v; }
static inline void ret_s(Cpu *c, float f) { uint32_t b; memcpy(&b, &f, 4); c->v[0] = b; }
static inline void ret_d(Cpu *c, double d) { uint64_t b; memcpy(&b, &d, 8); c->v[0] = b; }

// ---- SVCs (svc.c). Every SVC without a handler is a named fatal that gives its number, its name and the reason its
// owner recorded; an SVC outside the program's mask is a guest stop.
typedef void (*RtSvcFn)(Cpu *c, uint64_t pc);
void rt_svc_register(uint32_t imm, RtSvcFn fn);   // before rt_dispatch_init
void rt_svc_defer(uint32_t imm, const char *reason);   // why imm stays a named fatal (before rt_dispatch_init)
void rt_svc_set_mask(const uint64_t mask[4]);     // the SVCs the program's main.npdm allows; all until it is set
int rt_svc_allowed(uint32_t imm);
const char *rt_svc_name(uint32_t imm);            // "SetHeapSize", or NULL for a number without a known name
enum { RT_SVC_MISSING, RT_SVC_IMPLEMENTED, RT_SVC_NAMED };
int rt_svc_status(uint32_t imm, const char **reason);   // RT_SVC_NAMED: a named fatal whose reason is in *reason
void rt_svc_census(void);                         // one log line per SVC the mask allows, then the counts
uint64_t rt_svc_calls(uint32_t imm);
uint64_t rt_svc_first(uint32_t imm);              // the guest pc of its first call, 0 when never called

// ---- the D3 log (jmp.c): guest setjmp records made and longjmps taken since start-up. The first 32 longjmps are also
// logged one line each; a C++ throw stops the run with its type (a guest stop).
void rt_jmp_counts(uint64_t *setjmps, uint64_t *longjmps);
