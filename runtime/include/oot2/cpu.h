// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The boundary between generated code and the runtime: the guest CPU state, guest memory access, and the runtime
// entry points the lifted C calls. Generated code includes this header (and the lifter's own helper header, which
// includes it); the runtime implements every function declared here. Changing an offset or a signature here is a
// change of contract: the lifter, the runtime and their tests change in the same commit.
//
// The field order and offsets of Cpu, the exclusive monitor and the FPCR mirror follow DKB Recomp's runtime/cpu.h
// (MIT, adapted there from LAS Recomp runtime/cpu.h), so that the lifter adapted from DKB Recomp keeps its register
// offsets. See THIRD_PARTY_NOTICES.md.
#pragma once
#include <stddef.h>
#include <stdint.h>

#if !defined(__clang__) || !defined(__x86_64__)
#error "oot2-native's generated code and runtime are built with clang for x86-64"
#endif
#if !defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16) || !defined(__AVX2__)
#error "build with -march=x86-64-v3 (architecture D4): 128-bit exclusives need cmpxchg16b"
#endif

// ---- setjmp support (architecture D3), one record per live guest jmp_buf of a thread
//
// A guest setjmp is not a call into the runtime: the lifter compiles each setjmp call site to RT_SETJMP, which runs
// the host-level setjmp in the lifted function's own frame. A guest longjmp (rt_longjmp) restores the callee-saved
// state the setjmp saw and lands back in that frame. Records are keyed by the guest jmp_buf address and live in the
// thread's Cpu, so the guest's buffer is never written.
#define RT_JMP_SLOTS 16

typedef struct RtJmp {
    uint64_t env;            // guest jmp_buf address; 0 marks a free record
    uintptr_t frame;         // host frame address of the lifted function that ran RT_SETJMP
    uint64_t host[32];       // rt_host_setjmp's buffer: the host's callee-saved registers (runtime/core/hostjmp.c)
    uint64_t gpr[13];        // guest x19..x30, then sp, as they were at the setjmp
    uint64_t fpr[8];         // guest d8..d15 (AAPCS64 keeps only their low halves)
    uint32_t value;          // what RT_SETJMP yields after a longjmp lands
    uint32_t _pad;
} RtJmp;

// ---- guest CPU state: one per guest thread, owned by the runtime
typedef struct __attribute__((aligned(16))) Cpu {
    uint64_t x[32];          //    0: x0..x30 (x[31] unused; sp is separate)
    uint64_t sp;             //  256
    uint64_t pc;             //  264: the target of rt_dispatch; a host function's own address on entry to it
    uint64_t nzcv;           //  272: flags in bits 31..28, where generated code keeps them outside a function
    uint64_t tpidr;          //  280: TPIDR_EL0
    uint64_t tpidrro;        //  288: TPIDRRO_EL0, the thread-local region
    uint64_t excl_addr;      //  296: exclusive monitor, ~0 when clear
    uint64_t excl_val;       //  304
    void *thread;            //  312: the runtime's thread object
    uint64_t fpcr;           //  320: guest FPCR, mirrored into the host MXCSR (rt_fpcr_sync)
    uint64_t fpsr;           //  328
    uint64_t excl_val_hi;    //  336: upper half of a 128-bit exclusive pair
    uint32_t pending;        //  344: non-zero asks the thread to stop at its next safe point (rt_safepoint)
    uint32_t _pad0;          //  348
    uint8_t _pad1[512 - 352];
    __uint128_t v[32];       //  512: v0..v31
    RtJmp jmp[RT_JMP_SLOTS]; // 1024: setjmp records, runtime-private
} Cpu;

_Static_assert(offsetof(Cpu, sp) == 256, "sp");
_Static_assert(offsetof(Cpu, pc) == 264, "pc");
_Static_assert(offsetof(Cpu, nzcv) == 272, "nzcv");
_Static_assert(offsetof(Cpu, tpidr) == 280, "tpidr");
_Static_assert(offsetof(Cpu, tpidrro) == 288, "tpidrro");
_Static_assert(offsetof(Cpu, excl_addr) == 296, "excl_addr");
_Static_assert(offsetof(Cpu, thread) == 312, "thread");
_Static_assert(offsetof(Cpu, fpcr) == 320, "fpcr");
_Static_assert(offsetof(Cpu, fpsr) == 328, "fpsr");
_Static_assert(offsetof(Cpu, excl_val_hi) == 336, "excl_val_hi");
_Static_assert(offsetof(Cpu, pending) == 344, "pending");
_Static_assert(offsetof(Cpu, v) == 512, "v");
_Static_assert(offsetof(Cpu, jmp) == 1024, "jmp");

// A lifted guest function, and a host function standing at a guest address: both take the thread's Cpu. A lifted
// function returns (a C return) when the guest function returns; an indirect branch is a tail call to rt_dispatch.
typedef void (*GuestFn)(Cpu *);

// ---- runtime entry points called by generated code
//
// rt_dispatch: run the guest function at c->pc. Generated code reaches it with a tail call for an indirect branch
// (`br`) and a normal call for an indirect call (`blr`, after writing x30). Never returns to the caller for a branch
// to an address that holds nothing: that is a fatal error naming the address and the caller's x30.
void rt_dispatch(Cpu *c);
// rt_svc: the guest's `svc #imm` at guest address pc. Arguments and results in c->x as the kernel defines them.
void rt_svc(Cpu *c, uint32_t imm, uint64_t pc);
// rt_brk, rt_udf: the guest stopped itself (`brk #imm`, or an instruction that is permanently undefined).
void rt_brk(Cpu *c, uint32_t imm, uint64_t pc) __attribute__((noreturn));
void rt_udf(Cpu *c, uint64_t pc) __attribute__((noreturn));
// rt_spin_gate: one pass of a guest spin-wait loop the lifter found (head = the loop's first instruction; the search is
// tools/lift/spin_waits.py). It pauses the host core once, and a wait that has lasted a while gives its virtual core to
// a thread waiting for one (runtime/core/kernel.c). Being a call, it also keeps the compiler from hoisting the loop's
// loads out of the loop.
void rt_spin_gate(Cpu *c, uint64_t head);
// The guest system counter (CNTVCT_EL0, CNTFRQ_EL0): 31.25 MHz on Switch 2, counted from process start.
uint64_t rt_ticks(void);
uint32_t rt_tick_hz(void);

// ---- guest memory: guest addresses are host addresses (architecture D6)
typedef uint16_t __attribute__((aligned(1), may_alias)) rt_u16u;
typedef uint32_t __attribute__((aligned(1), may_alias)) rt_u32u;
typedef uint64_t __attribute__((aligned(1), may_alias)) rt_u64u;
typedef uint8_t __attribute__((may_alias)) rt_u8a;

#define LD8(a)  (*(const rt_u8a *)(uintptr_t)(a))
#define LD16(a) (*(const rt_u16u *)(uintptr_t)(a))
#define LD32(a) (*(const rt_u32u *)(uintptr_t)(a))
#define LD64(a) (*(const rt_u64u *)(uintptr_t)(a))
#define ST8(a, v)  (*(rt_u8a *)(uintptr_t)(a) = (uint8_t)(v))
#define ST16(a, v) (*(rt_u16u *)(uintptr_t)(a) = (uint16_t)(v))
#define ST32(a, v) (*(rt_u32u *)(uintptr_t)(a) = (uint32_t)(v))
#define ST64(a, v) (*(rt_u64u *)(uintptr_t)(a) = (uint64_t)(v))

// ---- exclusive monitor, emulated with compare-and-swap. A store-exclusive returns 0 on success, as STXR does.
#define RT_EXCL(N, T)                                                                      \
    static inline uint64_t ldx##N(Cpu *c, uint64_t a) {                                    \
        T v = __atomic_load_n((T *)(uintptr_t)a, __ATOMIC_ACQUIRE);                        \
        c->excl_addr = a;                                                                  \
        c->excl_val = v;                                                                   \
        return v;                                                                          \
    }                                                                                      \
    static inline uint32_t stx##N(Cpu *c, uint64_t a, uint64_t v) {                        \
        if (c->excl_addr != a) return 1;                                                   \
        c->excl_addr = ~0ull;                                                              \
        T expected = (T)c->excl_val;                                                       \
        return !__atomic_compare_exchange_n((T *)(uintptr_t)a, &expected, (T)v, 0,         \
                                            __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);           \
    }
RT_EXCL(8, uint8_t)
RT_EXCL(16, uint16_t)
RT_EXCL(32, uint32_t)
RT_EXCL(64, uint64_t)
#undef RT_EXCL

// An exclusive pair of two X registers: one 16-byte-aligned 128-bit access (cmpxchg16b, part of x86-64-v3).
static inline __uint128_t ldxp128(Cpu *c, uint64_t a) {
    __uint128_t v = __atomic_load_n((__uint128_t *)(uintptr_t)a, __ATOMIC_ACQUIRE);
    c->excl_addr = a;
    c->excl_val = (uint64_t)v;
    c->excl_val_hi = (uint64_t)(v >> 64);
    return v;
}
static inline uint32_t stxp128(Cpu *c, uint64_t a, uint64_t lo, uint64_t hi) {
    if (c->excl_addr != a) return 1;
    c->excl_addr = ~0ull;
    __uint128_t expected = ((__uint128_t)c->excl_val_hi << 64) | c->excl_val;
    __uint128_t v = ((__uint128_t)hi << 64) | lo;
    return !__atomic_compare_exchange_n((__uint128_t *)(uintptr_t)a, &expected, v, 0, __ATOMIC_SEQ_CST,
                                        __ATOMIC_SEQ_CST);
}

// ---- floating point: the MXCSR of a host thread running guest code mirrors c->fpcr. FTZ and DAZ follow FPCR.FZ
// (bit 24); rounding stays round-to-nearest, and the lifter's helpers switch it around the few operations that need
// another FPCR.RMode. Set wherever FPCR or the host FP context changes: MSR FPCR in generated code, thread entry and
// rt_call in the runtime.
static inline void rt_fpcr_sync(const Cpu *c) {
    unsigned cur = __builtin_ia32_stmxcsr();
    unsigned want = (cur & ~0xE040u) | (c->fpcr & (1u << 24) ? 0x8040u : 0);   // FTZ 0x8000, DAZ 0x40, RC 0x6000
    if (cur != want) __builtin_ia32_ldmxcsr(want);
}

// ---- setjmp call sites (D3)
//
// At a guest call to setjmp the lifter writes x30 as for any call, then emits `c->x[0] = RT_SETJMP(c, c->x[0]);`.
// The value is 0 on the first return and the longjmp's value (1 for 0) when a longjmp lands. The function that does
// this keeps every guest register in the Cpu (no callee-saved trust), makes no tail call, and emits RT_JMP_LEAVE(c)
// before each of its returns, so a longjmp to a buffer whose function has returned is caught, not undefined.
RtJmp *rt_jmp_enter(Cpu *c, uint64_t env, uintptr_t frame);
void rt_jmp_leave(Cpu *c, uintptr_t frame);
// The host setjmp under RT_SETJMP: saves every callee-saved register of the host ABI, returns 0, and returns again
// with the longjmp's value (never 0). Not __builtin_setjmp, which keeps only the frame and stack pointers.
int rt_host_setjmp(uint64_t *buf) __attribute__((returns_twice));
void rt_host_longjmp(uint64_t *buf, int value) __attribute__((noreturn));
// The guest longjmp(env, value): restores the guest state RT_SETJMP saved and lands there. A buffer no live setjmp
// filled is a fatal error. Also bound to the guest symbol "longjmp" through the host-function table.
void rt_longjmp(Cpu *c, uint64_t env, uint32_t value) __attribute__((noreturn));
// A module's own definition of a guest symbol the runtime replaces (setjmp, the C++ throw entry points): the lifter
// emits the function's body as `c->pc = <its address>; rt_replaced(c, "<symbol>");` instead of lifting it, so a direct
// call inside the defining module reaches the same HOST_OVERRIDE an import reaches through the loader's binding. A
// symbol without an override is the named fatal "unimplemented replaced: <symbol>".
void rt_replaced(Cpu *c, const char *symbol);

#define RT_SETJMP(c, env)                                                                  \
    __extension__({                                                                        \
        RtJmp *rt_jmp_ = rt_jmp_enter((c), (env), (uintptr_t)__builtin_frame_address(0)); \
        rt_host_setjmp(rt_jmp_->host) ? rt_jmp_->value : 0u;                               \
    })
#define RT_JMP_LEAVE(c) rt_jmp_leave((c), (uintptr_t)__builtin_frame_address(0))
