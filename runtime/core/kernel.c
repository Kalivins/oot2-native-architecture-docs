// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Guest threads on virtual cores (architecture D7): threads, the core scheduler, safe points, asynchronous preemption,
// the spin-wait gate, per-thread time, and the thread SVCs. Synchronization objects are in ksync.c, the information and
// process SVCs in kinfo.c. See oot2/kernel.h for the model.
//
// Adapted from DKB Recomp runtime/kernel.c at 9d8fa02 (GPL-3.0-or-later; adapted there from MacNativeGames
// games/totk/runtime/threads.c at b75be8c and 33e8fa8, d2r-mobile dm/host/thread.c at 53e7c5b and LAS Recomp
// runtime/kernel.c at da35b83, all MIT): one host thread and Cpu per guest thread with a reserved host stack, the
// virtual cores and their kick of the worst owner, priority inheritance through the mutex word, the preemption-disable
// words of the thread-local region and SynchronizePreemptionState, asynchronous preemption through a stub that keeps
// every volatile register (kern_ap_stub, here rt_kern_preempt_stub), the spin-wait hand-off (itself from MacNativeGames
// threads.c runtime_spin_wait at a038e6b and nxnative dm/host/hle_thread.c dm_spin_wait at 6d34984, MIT), the
// thread-local block and its current-thread-object pointer at TLR+0x1F8. From oot-pc-recomp runtime/kernel.c at
// 8e09aaa (GPL-3.0-or-later): the thread's own handle at TLR+0x110, which SDK 22's mutexes compare with their owner tag.
// Changed here: a thread that becomes ready is queued for a core by whoever makes it ready (StartThread, a wake, the
// end of a sleep), under the kernel lock, so priority order does not depend on when its host thread runs; safe points
// are the boundary's c->pending (runtime.h); the core count, priorities, main thread and handle table come from
// main.npdm; handles carry a linear id as Horizon's do, so a stale one reads as invalid; Horizon's result codes and
// argument checks; ExitThread lands through the runtime's own setjmp (hostjmp.c); a thread's host thread starts at
// StartThread; per-thread CPU time and core time are kept per frame; the trace and diagnosis code is left out. The
// fibres are in fibre.c: a thread's kicks go to the Cpu it runs (t->cur), its own or its fibre's.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kern.h"

#define HOST_STACK (256ull << 20)      // reserved, committed as it grows: every guest call is a host call
#define TLR_SIZE 0x200
#define THREAD_OBJ_SIZE 0x400          // the "current thread object", TLR+0x1F8 points at it
#define TLS_SIZE 0x2000                // TPIDR_EL0
#define TL_SIZE (TLR_SIZE + THREAD_OBJ_SIZE + TLS_SIZE)
#define TLR_DISABLE_COUNT 0x100        // u16: the SDK's spin locks count here while they try or hold a lock
#define TLR_INTERRUPT_FLAG 0x102       // u16: set by the kernel when it put a preemption off; the SDK then calls 0x36
#define TLR_OWN_HANDLE 0x110
#define TLR_THREAD_OBJ 0x1F8
#define AP_FIRST_NS 250000ull          // a ready thread spins this long before it interrupts the owner it asked
#define AP_GAP_NS 1000000ull           // and interrupts the same owner at most this often
#define SPIN_GAP_NS 20000ull           // two passes of one spin wait come at most this far apart
#define SLEEP_SPIN_NS (2 * PLAT_SLEEP_LATE_NS)   // shorter sleeps spin on the clock: the host timer is too late
#define READY_MAX 128                  // ready threads kkick considers, best first

Kernel K = {.g = PLAT_MUTEX_INIT, .done = PLAT_COND_INIT};
static _Thread_local RtThread *tcur;

void klock(void) { plat_mutex_lock(&K.g); }
void kunlock(void) { plat_mutex_unlock(&K.g); }
RtThread *kcurrent(void) { return tcur; }

RtThread *kthread(uint32_t svc) {
    if (!tcur) rt_fatal("kernel: svc %#x on a host thread, which is no guest thread", svc);
    return tcur;
}

// ---- handles: index in bits 0-14, a linear id (1 to 0x7FFF) in bits 15-29, as Horizon's
uint32_t h_new(int kind, void *obj) {
    for (uint32_t n = 0; n < K.hcap; n++) {
        uint32_t i = (K.hnext + n) % K.hcap;
        if (!i || K.h[i].kind) continue;
        uint32_t id = K.hid;
        K.hid = K.hid == 0x7FFF ? 1 : K.hid + 1;
        K.h[i].kind = kind;
        K.h[i].id = id;
        K.h[i].obj = obj;
        K.hnext = i + 1;
        return id << 15 | i;
    }
    return 0;
}

static int h_index(uint32_t h) {
    uint32_t i = h & 0x7FFF;
    return i && i < K.hcap && K.h[i].kind && K.h[i].id == (h >> 15) ? (int)i : 0;
}

int h_kind(uint32_t h) {
    if (h == H_CURRENT_THREAD) return tcur ? O_THREAD : O_FREE;
    int i = h_index(h);
    return i ? K.h[i].kind : O_FREE;
}

void *h_get(uint32_t h, int kind) {
    if (h == H_CURRENT_THREAD) return kind == O_THREAD ? tcur : NULL;
    int i = h_index(h);
    return i && K.h[i].kind == kind ? K.h[i].obj : NULL;
}

void h_free(uint32_t h) {
    int i = h_index(h);
    if (i) K.h[i].kind = O_FREE, K.h[i].obj = NULL;
}

// ---- the thread-local region
static uint16_t tl_disable_count(const RtThread *t) { return LD16(t->tl + TLR_DISABLE_COUNT); }
static void tl_set_interrupt(const RtThread *t, uint16_t v) { ST16(t->tl + TLR_INTERRUPT_FLAG, v); }

// ---- virtual cores
static int core_ok(const RtThread *t, int c) {
    return c >= 0 && c < MAX_CORES && (K.cfg.core_mask >> c & 1) && (t->mask >> c & 1);
}

// Horizon's priority inheritance, one level: a thread counts at the priority of the best thread blocked on a guest
// mutex it owns (the owner is read from the mutex word, so it follows hand-offs).
static int prio_of(const RtThread *o) {
    int p = o->prio;
    for (const RtThread *w = K.waiters; w; w = w->wnext)
        if (w->wkind == W_LOCK && w->prio < p && (kld32(w->waddr) & ~WAITERS_BIT) == o->handle) p = w->prio;
    return p;
}

static void unready(RtThread *t) {
    for (RtThread **p = &K.ready_q; *p; p = &(*p)->cnext)
        if (*p == t) {
            *p = t->cnext;
            break;
        }
    t->cnext = NULL;
    t->ready = 0;
}

static void grant(RtThread *t, int c) {
    unready(t);
    t->kicked = 0;
    t->target = NULL;
    K.owner[c] = t;
    t->core_since = plat_now_ns();
    __atomic_store_n(&t->cur_core, c, __ATOMIC_RELEASE);
    plat_cond_signal(&t->cv);
}

static RtThread *best_ready(int c) {
    RtThread *b = NULL;
    int bp = 0;
    for (RtThread *w = K.ready_q; w; w = w->cnext) {
        if (!core_ok(w, c)) continue;
        int p = prio_of(w);
        if (!b || p < bp || (p == bp && w->cseq < b->cseq)) b = w, bp = p;
    }
    return b;
}

// Asks owners to give their cores up, one owner per ready thread: ready threads best first, each claims the worst owner
// worse than itself on a core it may run on that no better thread claimed (DKB Recomp's core_kick).
void kkick(void) {
    RtThread *w[READY_MAX];
    int wp[READY_MAX], op[MAX_CORES], n = 0;
    for (int i = 0; i < MAX_CORES; i++) op[i] = K.owner[i] ? prio_of(K.owner[i]) : 0;
    for (RtThread *p = K.ready_q; p; p = p->cnext) {
        p->target = NULL;
        if (n == READY_MAX) continue;
        int i = n++, pp = prio_of(p);
        for (; i > 0 && (wp[i - 1] > pp || (wp[i - 1] == pp && w[i - 1]->cseq > p->cseq)); i--) w[i] = w[i - 1], wp[i] = wp[i - 1];
        w[i] = p;
        wp[i] = pp;
    }
    uint64_t claimed = 0;
    for (int k = 0; k < n; k++) {
        int v = -1;
        for (int i = 0; i < MAX_CORES; i++) {
            RtThread *o = K.owner[i];
            if (!o || (claimed >> i & 1) || !core_ok(w[k], i) || op[i] <= wp[k]) continue;
            if (v < 0 || op[i] > op[v] || (op[i] == op[v] && o->kicked && !K.owner[v]->kicked)) v = i;
        }
        if (v < 0) continue;
        claimed |= 1ull << v;
        RtThread *o = K.owner[v];
        w[k]->target = o;
        plat_cond_signal(&w[k]->cv);   // it may now interrupt o
        if (!o->kicked) {
            o->kicked = 1;
            __atomic_store_n(&o->cur->pending, 1, __ATOMIC_RELEASE);
        }
    }
}

// t holds no core and is not queued: queue it, and give it a free core it may run on (its ideal core first). No other
// ready thread may run on that core, or it would have taken it already.
static void make_ready(RtThread *t) {
    t->ready = 1;
    t->cseq = ++K.seq;
    t->target = NULL;
    t->cnext = K.ready_q;
    K.ready_q = t;
    int c = core_ok(t, t->ideal) && !K.owner[t->ideal] ? t->ideal : -1;
    for (int i = 0; c < 0 && i < MAX_CORES; i++)
        if (core_ok(t, i) && !K.owner[i]) c = i;
    if (c >= 0) grant(t, c);
    else kkick();
}

static void release_core(RtThread *t) {
    int c = t->cur_core;
    if (c < 0) return;
    t->core_ns += plat_now_ns() - t->core_since;
    __atomic_store_n(&t->cur_core, -1, __ATOMIC_RELEASE);
    t->kicked = 0;
    K.owner[c] = NULL;
    RtThread *b = best_ready(c);
    if (b) grant(b, c);
    kkick();
}

static int ap_try(RtThread *w);

// Until t holds a core. A ready thread that asked an owner to give its core up spins a while (most waits end in
// microseconds), then interrupts the owner every AP_GAP_NS while it runs lifted code without a safe point.
static void wait_core(RtThread *t) {
    int spun = 0;
    while (__atomic_load_n(&t->cur_core, __ATOMIC_ACQUIRE) < 0) {
        if (!K.async_on || !t->target) {
            plat_cond_wait(&t->cv, &K.g);
            continue;
        }
        if (!spun) {
            spun = 1;
            uint64_t end = plat_now_ns() + AP_FIRST_NS;
            kunlock();
            while (__atomic_load_n(&t->cur_core, __ATOMIC_ACQUIRE) < 0 && plat_now_ns() < end) plat_pause();
            klock();
            continue;
        }
        ap_try(t);
        plat_cond_wait_ns(&t->cv, &K.g, AP_GAP_NS);
    }
}

// Whether t, which holds a core, must give it up now: the core is no longer one it may run on, or a better ready thread
// may run there.
static int must_leave(const RtThread *t) {
    int c = t->cur_core;
    if (c < 0) return 0;
    if (!core_ok(t, c)) return 1;
    int tp = prio_of(t);
    for (const RtThread *w = K.ready_q; w; w = w->cnext)
        if (core_ok(w, c) && prio_of(w) < tp) return 1;
    return 0;
}

// At a safe point: give the core up if it is due, unless the SDK's preemption disable holds (then SVC 0x36 does).
static int reschedule(RtThread *t) {
    if (!must_leave(t)) {
        t->kicked = 0;
        return 0;
    }
    if (core_ok(t, t->cur_core) && tl_disable_count(t)) {
        tl_set_interrupt(t, 1);
        t->kicked = 0;
        K.st.deferred++;
        return 0;
    }
    release_core(t);
    make_ready(t);
    wait_core(t);
    return 1;
}

static void pause_here(RtThread *t) {   // SetThreadActivity(Paused) reached at a safe point
    release_core(t);
    t->paused = 1;
    while (t->pause_req) plat_cond_wait(&t->cv, &K.g);
    t->paused = 0;
    make_ready(t);
    wait_core(t);
}

static void safepoint(Cpu *c) {
    __atomic_store_n(&c->pending, 0, __ATOMIC_RELAXED);
    RtThread *t = c->thread;
    if (!t || t != tcur) return;   // a Cpu no guest thread runs (host code calling guest code before the kernel)
    klock();
    if (t->pause_req) pause_here(t);
    if (reschedule(t)) K.st.safepoint_yields++;
    kunlock();
}

// ---- asynchronous preemption. rt_kern_preempt_stub is entered in place of the interrupted instruction of lifted
// code: it keeps every register the code may have live (volatile GPRs, flags, YMM0-15, MXCSR; the C calls keep the
// rest), gives the core up in kern_preempt_yield, and jumps back to kern_preempt_rip(), whose value it first stores in
// the return slot it reserved at entry, so a stack walk sees the interrupted function as its caller. No ret, so call
// and return pairs stay balanced. The interrupting thread never writes the owner's stack. (DKB Recomp kern_ap_stub.)
void rt_kern_preempt_stub(void);
uint64_t kern_preempt_rip(void);
void kern_preempt_yield(void);
#ifdef _WIN32
__asm__(".text\n"
        ".globl rt_kern_preempt_stub\n"
        ".def rt_kern_preempt_stub; .scl 2; .type 32; .endef\n"
        ".p2align 4\n"
        "rt_kern_preempt_stub:\n"
        ".seh_proc rt_kern_preempt_stub\n"
        "    leaq -8(%rsp), %rsp\n"   // not sub: the flags are still the interrupted code's until pushfq
        ".seh_stackalloc 8\n"
        "    pushfq\n"
        ".seh_stackalloc 8\n"
        "    pushq %rax\n"
        ".seh_pushreg %rax\n"
        "    pushq %rcx\n"
        ".seh_pushreg %rcx\n"
        "    pushq %rdx\n"
        ".seh_pushreg %rdx\n"
        "    pushq %r8\n"
        ".seh_pushreg %r8\n"
        "    pushq %r9\n"
        ".seh_pushreg %r9\n"
        "    pushq %r10\n"
        ".seh_pushreg %r10\n"
        "    pushq %r11\n"
        ".seh_pushreg %r11\n"
        "    pushq %rbp\n"
        ".seh_pushreg %rbp\n"
        "    movq %rsp, %rbp\n"
        ".seh_setframe %rbp, 0\n"
        ".seh_endprologue\n"
        "    andq $-32, %rsp\n"
        "    subq $576, %rsp\n"   // 0: shadow space, 32: MXCSR saved, 36: MXCSR default, 64: YMM0-15
        "    vmovdqa %ymm0, 64(%rsp)\n"
        "    vmovdqa %ymm1, 96(%rsp)\n"
        "    vmovdqa %ymm2, 128(%rsp)\n"
        "    vmovdqa %ymm3, 160(%rsp)\n"
        "    vmovdqa %ymm4, 192(%rsp)\n"
        "    vmovdqa %ymm5, 224(%rsp)\n"
        "    vmovdqa %ymm6, 256(%rsp)\n"
        "    vmovdqa %ymm7, 288(%rsp)\n"
        "    vmovdqa %ymm8, 320(%rsp)\n"
        "    vmovdqa %ymm9, 352(%rsp)\n"
        "    vmovdqa %ymm10, 384(%rsp)\n"
        "    vmovdqa %ymm11, 416(%rsp)\n"
        "    vmovdqa %ymm12, 448(%rsp)\n"
        "    vmovdqa %ymm13, 480(%rsp)\n"
        "    vmovdqa %ymm14, 512(%rsp)\n"
        "    vmovdqa %ymm15, 544(%rsp)\n"
        "    stmxcsr 32(%rsp)\n"
        "    movl $0x1f80, 36(%rsp)\n"
        "    ldmxcsr 36(%rsp)\n"
        "    cld\n"
        "    vzeroupper\n"
        "    call kern_preempt_rip\n"
        "    movq %rax, 72(%rbp)\n"   // the return slot: rbp + 8 pushes + pushfq
        "    call kern_preempt_yield\n"
        "    ldmxcsr 32(%rsp)\n"
        "    vmovdqa 64(%rsp), %ymm0\n"
        "    vmovdqa 96(%rsp), %ymm1\n"
        "    vmovdqa 128(%rsp), %ymm2\n"
        "    vmovdqa 160(%rsp), %ymm3\n"
        "    vmovdqa 192(%rsp), %ymm4\n"
        "    vmovdqa 224(%rsp), %ymm5\n"
        "    vmovdqa 256(%rsp), %ymm6\n"
        "    vmovdqa 288(%rsp), %ymm7\n"
        "    vmovdqa 320(%rsp), %ymm8\n"
        "    vmovdqa 352(%rsp), %ymm9\n"
        "    vmovdqa 384(%rsp), %ymm10\n"
        "    vmovdqa 416(%rsp), %ymm11\n"
        "    vmovdqa 448(%rsp), %ymm12\n"
        "    vmovdqa 480(%rsp), %ymm13\n"
        "    vmovdqa 512(%rsp), %ymm14\n"
        "    vmovdqa 544(%rsp), %ymm15\n"
        "    movq %rbp, %rsp\n"
        "    popq %rbp\n"
        "    popq %r11\n"
        "    popq %r10\n"
        "    popq %r9\n"
        "    popq %r8\n"
        "    popq %rdx\n"
        "    popq %rcx\n"
        "    popq %rax\n"
        "    popfq\n"
        "    leaq 8(%rsp), %rsp\n"
        "    jmpq *-8(%rsp)\n"
        ".seh_endproc\n");
#else
// System V: rsi and rdi are volatile too, and the interrupted code may keep data in the 128 bytes below its stack
// pointer (the red zone), so the return slot and the saves go below that.
__asm__(".text\n"
        ".globl rt_kern_preempt_stub\n"
        ".type rt_kern_preempt_stub, @function\n"
        ".p2align 4\n"
        "rt_kern_preempt_stub:\n"
        "    .cfi_startproc\n"
        "    leaq -136(%rsp), %rsp\n"   // not sub: the flags are still the interrupted code's until pushfq
        "    pushfq\n"
        "    pushq %rax\n"
        "    pushq %rcx\n"
        "    pushq %rdx\n"
        "    pushq %rsi\n"
        "    pushq %rdi\n"
        "    pushq %r8\n"
        "    pushq %r9\n"
        "    pushq %r10\n"
        "    pushq %r11\n"
        "    pushq %rbp\n"
        "    movq %rsp, %rbp\n"
        "    andq $-32, %rsp\n"
        "    subq $576, %rsp\n"   // 32: MXCSR saved, 36: MXCSR default, 64: YMM0-15
        "    vmovdqa %ymm0, 64(%rsp)\n"
        "    vmovdqa %ymm1, 96(%rsp)\n"
        "    vmovdqa %ymm2, 128(%rsp)\n"
        "    vmovdqa %ymm3, 160(%rsp)\n"
        "    vmovdqa %ymm4, 192(%rsp)\n"
        "    vmovdqa %ymm5, 224(%rsp)\n"
        "    vmovdqa %ymm6, 256(%rsp)\n"
        "    vmovdqa %ymm7, 288(%rsp)\n"
        "    vmovdqa %ymm8, 320(%rsp)\n"
        "    vmovdqa %ymm9, 352(%rsp)\n"
        "    vmovdqa %ymm10, 384(%rsp)\n"
        "    vmovdqa %ymm11, 416(%rsp)\n"
        "    vmovdqa %ymm12, 448(%rsp)\n"
        "    vmovdqa %ymm13, 480(%rsp)\n"
        "    vmovdqa %ymm14, 512(%rsp)\n"
        "    vmovdqa %ymm15, 544(%rsp)\n"
        "    stmxcsr 32(%rsp)\n"
        "    movl $0x1f80, 36(%rsp)\n"
        "    ldmxcsr 36(%rsp)\n"
        "    cld\n"
        "    vzeroupper\n"
        "    call kern_preempt_rip\n"
        "    movq %rax, 88(%rbp)\n"   // the return slot: rbp + 10 pushes + pushfq
        "    call kern_preempt_yield\n"
        "    ldmxcsr 32(%rsp)\n"
        "    vmovdqa 64(%rsp), %ymm0\n"
        "    vmovdqa 96(%rsp), %ymm1\n"
        "    vmovdqa 128(%rsp), %ymm2\n"
        "    vmovdqa 160(%rsp), %ymm3\n"
        "    vmovdqa 192(%rsp), %ymm4\n"
        "    vmovdqa 224(%rsp), %ymm5\n"
        "    vmovdqa 256(%rsp), %ymm6\n"
        "    vmovdqa 288(%rsp), %ymm7\n"
        "    vmovdqa 320(%rsp), %ymm8\n"
        "    vmovdqa 352(%rsp), %ymm9\n"
        "    vmovdqa 384(%rsp), %ymm10\n"
        "    vmovdqa 416(%rsp), %ymm11\n"
        "    vmovdqa 448(%rsp), %ymm12\n"
        "    vmovdqa 480(%rsp), %ymm13\n"
        "    vmovdqa 512(%rsp), %ymm14\n"
        "    vmovdqa 544(%rsp), %ymm15\n"
        "    movq %rbp, %rsp\n"
        "    popq %rbp\n"
        "    popq %r11\n"
        "    popq %r10\n"
        "    popq %r9\n"
        "    popq %r8\n"
        "    popq %rdi\n"
        "    popq %rsi\n"
        "    popq %rdx\n"
        "    popq %rcx\n"
        "    popq %rax\n"
        "    popfq\n"
        "    leaq 136(%rsp), %rsp\n"
        "    jmpq *-136(%rsp)\n"
        "    .cfi_endproc\n"
        ".size rt_kern_preempt_stub, .-rt_kern_preempt_stub\n");
#endif

uint64_t kern_preempt_rip(void) { return tcur->ap_rip; }

void kern_preempt_yield(void) {
    RtThread *t = tcur;
    klock();
    t->ap_pending = 0;
    __atomic_store_n(&t->cur->pending, 0, __ATOMIC_RELAXED);
    if (reschedule(t)) K.st.async_taken++;
    kunlock();
}

// The ready thread w interrupts the owner it asked, if that owner runs lifted code: its host pc is moved to the stub.
// The owner is stopped only between plat_intr_stop and plat_intr_resume, during which nothing here takes a lock or
// allocates. Returns 1 when the owner was sent to the stub.
static int ap_try(RtThread *w) {
    RtThread *o = w->target;
    if (!o || o->cur_core < 0 || !o->kicked || !o->host || o->ap_pending) return 0;
    if (tl_disable_count(o)) {   // the SDK holds a spin lock: it switches at SVC 0x36
        tl_set_interrupt(o, 1);
        return 0;
    }
    uint64_t now = plat_now_ns(), rip, rsp, to = 0;
    if (o->ap_last && now - o->ap_last < AP_GAP_NS) return 0;
    o->ap_last = now;
    if (!plat_intr_stop(o->host, &rip, &rsp)) {
        K.st.async_failed++;
        return 0;
    }
    if (kern_pc_lifted(rip) && !tl_disable_count(o)) {
        o->ap_rip = rip;
        o->ap_pending = 1;
        to = (uint64_t)(uintptr_t)rt_kern_preempt_stub;
    }
    if (!plat_intr_resume(o->host, to)) {
        o->ap_pending = 0;
        K.st.async_failed++;
        return 0;
    }
    if (!to) K.st.async_elsewhere++;
    else K.st.async_sent++;
    return to != 0;
}

// The lifted code's host ranges, one per module: from its lowest function to its end marker (ticket 92). A range
// that ran to the highest function start would take whatever the linker placed after it for lifted code.
void kern_load_code_ranges(void) {
    uintptr_t lo[16], end[16];
    int n = dispatch_code_ranges(lo, end, 16);
    K.code_n = n < 16 ? n : 16;
    for (int i = 0; i < K.code_n; i++) {
        K.code_lo[i] = lo[i];
        K.code_end[i] = end[i];
    }
}

int kern_pc_lifted(uintptr_t pc) {
    for (int i = 0; i < K.code_n; i++)
        if (pc >= K.code_lo[i] && pc < K.code_end[i]) return 1;
    return 0;
}

// Asynchronous preemption needs the lifted code's host range, with no runtime code inside it (the runtime may hold
// locks; lifted code holds none). Set up at rt_kernel_run, after rt_dispatch_init.
static void async_setup(void) {
    kern_load_code_ranges();
    const uintptr_t runtime[] = {(uintptr_t)rt_kern_preempt_stub, (uintptr_t)kern_preempt_yield, (uintptr_t)rt_svc,
                                 (uintptr_t)rt_dispatch, (uintptr_t)rt_call, (uintptr_t)rt_spin_gate,
                                 (uintptr_t)safepoint, (uintptr_t)klock};
    const char *why = K.cfg.no_async_preempt ? "the configuration turns it off"
                    : !K.code_n ? "no range of lifted code"
                    : !plat_intr_init() ? "this host cannot interrupt a thread"
                    : NULL;
    for (unsigned i = 0; !why && i < sizeof runtime / sizeof *runtime; i++)
        if (kern_pc_lifted(runtime[i])) why = "runtime code lies inside the lifted code's host range";
    if (why) {
        rt_log("kernel: asynchronous preemption off: %s", why);
        return;
    }
    K.async_on = K.st.async_on = 1;
}

// ---- waits
uint32_t kblock(RtThread *t, int kind, int64_t ns) {
    if (ns == 0) return KR_TIMED_OUT;
    t->wkind = kind;
    t->woke = 0;
    t->wseq = ++K.seq;
    t->wnext = K.waiters;
    K.waiters = t;
    release_core(t);
    uint64_t now = plat_now_ns(), deadline = ns > 0 && (uint64_t)ns < UINT64_MAX - now ? now + (uint64_t)ns : 0;
    while (!t->woke) {
        if (!deadline) {
            plat_cond_wait(&t->cv, &K.g);
            continue;
        }
        now = plat_now_ns();
        if (now >= deadline) {
            for (RtThread **p = &K.waiters; *p; p = &(*p)->wnext)
                if (*p == t) {
                    *p = t->wnext;
                    break;
                }
            t->wnext = NULL;
            t->wkind = W_NONE;
            make_ready(t);
            wait_core(t);
            return KR_TIMED_OUT;
        }
        plat_cond_wait_ns(&t->cv, &K.g, deadline - now);
    }
    wait_core(t);   // kwake made it ready
    return t->wres;
}

void kwake(RtThread *t, uint32_t res) {
    for (RtThread **p = &K.waiters; *p; p = &(*p)->wnext)
        if (*p == t) {
            *p = t->wnext;
            break;
        }
    t->wnext = NULL;
    t->wkind = W_NONE;
    t->woke = 1;
    t->wres = res;
    make_ready(t);
    plat_cond_signal(&t->cv);
}

void kwake_sync(void) {
    for (RtThread *w = K.waiters, *n; w; w = n) {
        n = w->wnext;
        if (w->wkind == W_SYNC) kwake(w, UINT32_MAX);   // the wait checks its objects again
    }
}

// ---- spin waits (tools/lift/spin_waits.py finds them; the lifter calls this on every pass)
// A guest loop that polls memory until another thread changes it fills only its own core on the console, where a
// better thread pinned there preempts it. Here it holds a virtual core and a host core. Every pass pauses the host core
// once. A wait is a run of passes through the same loop at most SPIN_GAP_NS apart; once one has lasted
// spin_handoff_ns and a thread waits for the core, each pass gives the core to that thread, whatever its priority,
// yields the host processor, and queues for a core again. Shorter waits keep the core: a lock held for a moment would
// otherwise make the next pass wait for another thread's next safe point.
void rt_spin_gate(Cpu *c, uint64_t head) {
    plat_pause();
    RtThread *t = c->thread;
    if (!t || t != tcur) return;
    uint64_t now = plat_now_ns();
    if (head != t->sw_head || now - t->sw_last > SPIN_GAP_NS) {   // a new wait
        t->sw_head = head;
        t->sw_t0 = now;
    }
    t->sw_last = now;
    t->spin_passes++;
    if (now - t->sw_t0 < K.cfg.spin_handoff_ns || !__atomic_load_n(&K.ready_q, __ATOMIC_RELAXED)) return;
    klock();
    int give = 0;
    for (RtThread *w = K.ready_q; w && !give; w = w->cnext) give = core_ok(w, t->cur_core);
    if (give) {
        release_core(t);
        kunlock();
        plat_yield();
        klock();
        make_ready(t);
        wait_core(t);
        K.st.spin_handoffs++;
    }
    kunlock();
    t->sw_last = plat_now_ns();
}

// ---- threads
static RtThread *thread_new(uint64_t entry, uint64_t arg, uint64_t stack_top, int prio, int ideal, uint64_t mask,
                            uint32_t *res) {
    RtThread *t = calloc(1, sizeof *t);
    if (!t) rt_fatal("kernel: thread allocation");
    PlatCond cv = PLAT_COND_INIT;
    t->cv = cv;
    t->tl = mem_kernel_alloc(TL_SIZE);
    ST64(t->tl + TLR_THREAD_OBJ, t->tl + TLR_SIZE);
    Cpu *c = &t->cpu;
    t->cur = c;
    c->tpidrro = t->tl;
    c->tpidr = t->tl + TLR_SIZE + THREAD_OBJ_SIZE;
    c->excl_addr = ~0ull;
    c->thread = t;
    t->entry = entry;
    t->arg = arg;
    t->stack_top = stack_top;
    t->prio = prio;
    t->ideal = ideal;
    t->mask = mask;
    t->state = T_CREATED;
    t->cur_core = -1;
    klock();
    t->handle = h_new(O_THREAD, t);
    if (!t->handle) {
        kunlock();
        mem_kernel_free(t->tl, TL_SIZE);
        free(t);
        *res = KR_OUT_OF_HANDLES;
        return NULL;
    }
    t->id = K.next_thread_id++;
    ST32(t->tl + TLR_OWN_HANDLE, t->handle);
    t->all = K.all;
    K.all = t;
    K.st.threads_created++;
    kunlock();
    *res = KR_OK;
    return t;
}

int kthread_free_if_done(RtThread *t) {
    if (t->state != T_EXITED || !t->closed) return 0;
    for (RtThread **p = &K.all; *p; p = &(*p)->all)
        if (*p == t) {
            *p = t->all;
            break;
        }
    mem_kernel_free(t->tl, TL_SIZE);
    plat_thread_free(t->host);   // the host thread may still be returning; it touches neither t nor its PlatThread
    free(t);
    return 1;
}

// The end of every guest thread, on its own host thread: give the core up, signal the thread object, free it if its
// handle is closed already. t is not touched after the unlock.
static void thread_finish(RtThread *t) {
    kfibre_thread_end(t);
    uint64_t cpu = plat_thread_cpu_ns(NULL);
    klock();
    t->cpu_final = cpu;
    release_core(t);
    t->state = T_EXITED;
    kwake_sync();
    K.st.threads_live = --K.live;
    if (!K.live) plat_cond_broadcast(&K.done);
    kthread_free_if_done(t);
    kunlock();
    tcur = NULL;
}

static void thread_main(void *p) {
    RtThread *t = p;
    tcur = t;
    klock();
    wait_core(t);
    kunlock();
    Cpu *c = &t->cpu;
    c->sp = t->stack_top & ~15ull;
    c->x[0] = t->arg;
    if (t->is_main) c->x[1] = t->handle;
    if (!rt_host_setjmp(t->exit_buf)) {
        rt_call(c, t->entry);
        char at[96];
        rt_guest_stop(c, "thread %#x returned from its entry %s, and Horizon threads end with ExitThread", t->handle,
                      rt_guest_name(t->entry, at, sizeof at));
    }
    thread_finish(tcur);
}

static uint32_t thread_start(RtThread *t) {   // holding the lock
    if (t->state != T_CREATED) return KR_INVALID_STATE;
    t->state = T_STARTED;
    K.st.threads_live = ++K.live;
    t->host = plat_thread_start(thread_main, t, HOST_STACK);
    if (!t->host) rt_fatal("kernel: no host thread for guest thread %#x", t->handle);
    make_ready(t);
    return KR_OK;
}

// ---- thread SVCs
// 0x08 CreateThread(x1 entry, x2 argument, x3 stack top, w4 priority, w5 ideal core -> w1 handle). The new thread may
// run on its ideal core only, until SetThreadCoreMask widens that, as on Horizon.
static void svc_create_thread(Cpu *c, uint64_t pc) {
    (void)pc;
    int prio = (int)(uint32_t)c->x[4], core = (int)(uint32_t)c->x[5];
    if (core == -2) core = K.cfg.main_core;   // IdealCoreUseProcessValue
    uint32_t r = core < 0 || core >= MAX_CORES || !(K.cfg.core_mask >> core & 1) ? KR_INVALID_CORE_ID
               : prio < K.cfg.highest_priority || prio > K.cfg.lowest_priority ? KR_INVALID_PRIORITY
               : KR_OK;
    RtThread *t = r ? NULL : thread_new(c->x[1], c->x[2], c->x[3], prio, core, 1ull << core, &r);
    if (t) {
        char at[96];
        rt_log("thread %#x: entry %s, priority %d, core %d, stack top %#llx", t->handle,
               rt_guest_name(t->entry, at, sizeof at), prio, core, (unsigned long long)t->stack_top);
    }
    c->x[0] = r;
    c->x[1] = t ? t->handle : 0;
}

static void svc_start_thread(Cpu *c, uint64_t pc) {   // 0x09 StartThread(w0 handle)
    (void)pc;
    klock();
    RtThread *t = h_get((uint32_t)c->x[0], O_THREAD);
    uint32_t r = t ? thread_start(t) : KR_INVALID_HANDLE;
    kunlock();
    c->x[0] = r;
}

// 0x0A ExitThread: back to the thread's start, then its end. On a fibre, back to the thread's own context first: the
// thread's start lies on that context's host stack.
static void svc_exit_thread(Cpu *c, uint64_t pc) {
    (void)c, (void)pc;
    RtThread *t = kthread(0x0A);
    if (t->fibre) kfibre_exit_thread(t);
    rt_host_longjmp(t->exit_buf, 1);
}

// 0x0B SleepThread(x0 ns). 0, -1 and -2 yield: to a ready thread of equal or better priority (0 and -1), to any ready
// thread (-2: it runs until its next safe point, where this thread, if better, takes the core back).
static void svc_sleep_thread(Cpu *c, uint64_t pc) {
    (void)pc;
    int64_t ns = (int64_t)c->x[0];
    RtThread *t = kthread(0x0B);
    klock();
    if (ns > 0) {
        release_core(t);
        kunlock();
        if ((uint64_t)ns <= SLEEP_SPIN_NS) {
            uint64_t end = plat_now_ns() + (uint64_t)ns;
            while (plat_now_ns() < end) plat_pause();
        } else {
            plat_sleep_ns((uint64_t)ns);
        }
        klock();
        make_ready(t);
        wait_core(t);
    } else if (ns >= -2) {
        int give = 0, tp = prio_of(t);
        for (RtThread *w = K.ready_q; w && !give; w = w->cnext) give = core_ok(w, t->cur_core) && (ns == -2 || prio_of(w) <= tp);
        if (give) {
            release_core(t);
            make_ready(t);
            wait_core(t);
        }
        kunlock();
        if (!give) plat_yield();
        return;
    }
    kunlock();
}

static void svc_get_thread_priority(Cpu *c, uint64_t pc) {   // 0x0C (w1 handle -> w1 priority)
    (void)pc;
    klock();
    RtThread *t = h_get((uint32_t)c->x[1], O_THREAD);
    c->x[0] = t ? KR_OK : KR_INVALID_HANDLE;
    if (t) c->x[1] = (uint32_t)t->prio;
    kunlock();
}

static void svc_set_thread_priority(Cpu *c, uint64_t pc) {   // 0x0D (w0 handle, w1 priority)
    (void)pc;
    int prio = (int)(uint32_t)c->x[1];
    uint32_t r = prio < K.cfg.highest_priority || prio > K.cfg.lowest_priority ? KR_INVALID_PRIORITY : KR_OK;
    klock();
    RtThread *t = r ? NULL : h_get((uint32_t)c->x[0], O_THREAD);
    if (!r && !t) r = KR_INVALID_HANDLE;
    if (t) {
        t->prio = prio;
        kkick();
        if (t == tcur && must_leave(t)) __atomic_store_n(&t->cur->pending, 1, __ATOMIC_RELAXED);   // at this SVC's exit
    }
    kunlock();
    c->x[0] = r;
}

static void svc_get_thread_core_mask(Cpu *c, uint64_t pc) {   // 0x0E (w2 handle -> w1 ideal core, x2 mask)
    (void)pc;
    klock();
    RtThread *t = h_get((uint32_t)c->x[2], O_THREAD);
    c->x[0] = t ? KR_OK : KR_INVALID_HANDLE;
    if (t) {
        c->x[1] = (uint32_t)t->ideal;
        c->x[2] = t->mask;
    }
    kunlock();
}

// 0x0F SetThreadCoreMask(w0 handle, w1 ideal core, x2 mask): -2 the process's ideal core alone, -1 no ideal core,
// -3 keep the ideal core (Atmosphere's checks).
static void svc_set_thread_core_mask(Cpu *c, uint64_t pc) {
    (void)pc;
    int ideal = (int)(uint32_t)c->x[1];
    uint64_t mask = c->x[2];
    uint32_t r = KR_OK;
    if (ideal == -2) {
        ideal = K.cfg.main_core;
        mask = 1ull << ideal;
    } else if (mask & ~K.cfg.core_mask) {
        r = KR_INVALID_CORE_ID;
    } else if (!mask) {
        r = KR_INVALID_COMBINATION;
    } else if (ideal >= 0) {
        if (ideal >= MAX_CORES || !(K.cfg.core_mask >> ideal & 1)) r = KR_INVALID_CORE_ID;
        else if (!(mask >> ideal & 1)) r = KR_INVALID_COMBINATION;
    } else if (ideal != -1 && ideal != -3) {
        r = KR_INVALID_CORE_ID;
    }
    klock();
    RtThread *t = r ? NULL : h_get((uint32_t)c->x[0], O_THREAD);
    if (!r && !t) r = KR_INVALID_HANDLE;
    if (!r && ideal == -3 && t->ideal >= 0 && !(mask >> t->ideal & 1)) r = KR_INVALID_COMBINATION;
    if (!r) {
        if (ideal != -3) t->ideal = ideal;
        t->mask = mask;
        if (t->ready) {   // queued: its cores changed, so look for a free one again
            unready(t);
            make_ready(t);
        } else if (t->cur_core >= 0 && !core_ok(t, t->cur_core)) {
            t->kicked = 1;
            __atomic_store_n(&t->cur->pending, 1, __ATOMIC_RELEASE);
        }
        kkick();
    }
    kunlock();
    c->x[0] = r;
}

static void svc_get_current_processor_number(Cpu *c, uint64_t pc) {   // 0x10 -> w0 the virtual core held
    (void)pc;
    c->x[0] = (uint32_t)kthread(0x10)->cur_core;
}

static void svc_get_thread_id(Cpu *c, uint64_t pc) {   // 0x25 (w1 handle -> x1)
    (void)pc;
    klock();
    RtThread *t = h_get((uint32_t)c->x[1], O_THREAD);
    c->x[0] = t ? KR_OK : KR_INVALID_HANDLE;
    if (t) c->x[1] = t->id;
    kunlock();
}

// 0x32 SetThreadActivity(w0 handle, w1 0 runnable or 1 paused). A paused thread stops at its next safe point and gives
// its core up until it runs again; the calling thread cannot pause itself.
static void svc_set_thread_activity(Cpu *c, uint64_t pc) {
    (void)pc;
    uint32_t act = (uint32_t)c->x[1], r = KR_OK;
    klock();
    RtThread *t = h_get((uint32_t)c->x[0], O_THREAD);
    if (act > 1) r = KR_INVALID_ENUM;
    else if (!t) r = KR_INVALID_HANDLE;
    else if (t == tcur) r = KR_BUSY;
    else if (t->state != T_STARTED || t->pause_req == (int)act) r = KR_INVALID_STATE;
    else if (act) {
        t->pause_req = 1;
        __atomic_store_n(&t->cur->pending, 1, __ATOMIC_RELEASE);
    } else {
        t->pause_req = 0;
        plat_cond_signal(&t->cv);
    }
    kunlock();
    c->x[0] = r;
}

// 0x36 SynchronizePreemptionState: the SDK's spin locks call it after clearing TLR+0x100 when TLR+0x102 is set; a
// preemption put off there happens here.
static void svc_synchronize_preemption_state(Cpu *c, uint64_t pc) {
    (void)pc;
    RtThread *t = kthread(0x36);
    tl_set_interrupt(t, 0);
    klock();
    if (reschedule(t)) K.st.safepoint_yields++;
    kunlock();
    c->x[0] = KR_OK;
}

// ---- start-up and the program
void rt_kernel_init(const RtKernelConfig *cfg) {
    if (K.ready) rt_fatal("rt_kernel_init called twice");
    if (rt_sealed()) rt_fatal("rt_kernel_init after rt_dispatch_init");
    if (!rt_mem_ready()) rt_fatal("rt_kernel_init before rt_mem_init");
    const char *bad = !cfg->core_mask ? "no virtual core"
                    : cfg->highest_priority < 0 || cfg->lowest_priority > 63 ||
                              cfg->highest_priority > cfg->lowest_priority ? "a bad priority range"
                    : cfg->main_core < 0 || cfg->main_core >= MAX_CORES || !(cfg->core_mask >> cfg->main_core & 1)
                          ? "a main thread core outside the core mask"
                    : cfg->main_priority < cfg->highest_priority || cfg->main_priority > cfg->lowest_priority
                          ? "a main thread priority outside the range"
                    : !cfg->main_stack_size ? "no main thread stack"
                    : cfg->handle_table_size > 0x7FFE ? "a handle table over 32766 entries"
                    : NULL;
    if (bad)
        rt_fatal("rt_kernel_init: %s (cores %#llx, priorities %d to %d, main thread priority %d on core %d)", bad,
                 (unsigned long long)cfg->core_mask, cfg->highest_priority, cfg->lowest_priority, cfg->main_priority,
                 cfg->main_core);
    K.cfg = *cfg;
    if (!K.cfg.handle_table_size) K.cfg.handle_table_size = 1024;
    if (!K.cfg.spin_handoff_ns) K.cfg.spin_handoff_ns = 50000;
    K.hcap = K.cfg.handle_table_size + 1;
    K.h = calloc(K.hcap, sizeof *K.h);
    if (!K.h) rt_fatal("kernel: handle table allocation");
    K.hid = 1;
    K.hnext = 1;
    K.next_thread_id = 0x51;
    if (cfg->svc_mask[0] | cfg->svc_mask[1] | cfg->svc_mask[2] | cfg->svc_mask[3]) rt_svc_set_mask(cfg->svc_mask);
    plat_thread_cpu_ns(NULL);   // settles the host's cycle rate now, not under the lock at the first frame
    K.exception_context = mem_kernel_alloc(0x1000);
    rt_svc_register(0x08, svc_create_thread);
    rt_svc_register(0x09, svc_start_thread);
    rt_svc_register(0x0A, svc_exit_thread);
    rt_svc_register(0x0B, svc_sleep_thread);
    rt_svc_register(0x0C, svc_get_thread_priority);
    rt_svc_register(0x0D, svc_set_thread_priority);
    rt_svc_register(0x0E, svc_get_thread_core_mask);
    rt_svc_register(0x0F, svc_set_thread_core_mask);
    rt_svc_register(0x10, svc_get_current_processor_number);
    rt_svc_register(0x25, svc_get_thread_id);
    rt_svc_register(0x32, svc_set_thread_activity);
    rt_svc_register(0x36, svc_synchronize_preemption_state);
    rt_svc_defer(0x33, "the registers of a paused thread are not all in its Cpu, since lifted code keeps callee-saved "
                       "ones in host locals (architecture D3)");
    ksync_register();
    kinfo_register();
    rt_set_safepoint_hook(safepoint);
    K.ready = 1;
    rt_log("kernel: virtual cores %#llx, priorities %d to %d, main thread priority %d on core %d with a %#llx-byte "
           "stack, %u handles", (unsigned long long)K.cfg.core_mask, K.cfg.highest_priority, K.cfg.lowest_priority,
           K.cfg.main_priority, K.cfg.main_core, (unsigned long long)K.cfg.main_stack_size, K.cfg.handle_table_size);
}

void rt_kernel_run(uint64_t entry, uint64_t arg) {
    if (!K.ready) rt_fatal("rt_kernel_run before rt_kernel_init");
    if (!rt_sealed()) rt_fatal("rt_kernel_run before rt_dispatch_init");
    if (tcur) rt_fatal("rt_kernel_run on a guest thread");
    async_setup();
    uint64_t top = mem_main_stack(K.cfg.main_stack_size);
    uint32_t r;
    RtThread *t = thread_new(entry, arg, top, K.cfg.main_priority, K.cfg.main_core, 1ull << K.cfg.main_core, &r);
    if (!t) rt_fatal("kernel: no handle for the main thread (%#x)", r);
    klock();
    t->is_main = 1;
    thread_start(t);
    while (K.live) plat_cond_wait(&K.done, &K.g);
    kunlock();
}

int rt_kernel_block_begin(void) {
    RtThread *t = tcur;
    if (!t || t->cur_core < 0) return 0;
    klock();
    release_core(t);
    kunlock();
    return 1;
}

void rt_kernel_block_end(int token) {
    if (!token) return;
    klock();
    make_ready(tcur);
    wait_core(tcur);
    kunlock();
}

// ---- time and statistics
void rt_kernel_frame(void) {
    klock();
    uint64_t now = plat_now_ns();
    for (RtThread *t = K.all; t; t = t->all) {
        uint64_t cpu = t->state == T_EXITED ? t->cpu_final : t->state == T_STARTED ? plat_thread_cpu_ns(t->host) : 0;
        uint64_t core = t->core_ns + (t->cur_core >= 0 ? now - t->core_since : 0);
        t->frame_cpu = cpu > t->mark_cpu ? cpu - t->mark_cpu : 0;
        t->frame_core = core - t->mark_core;
        t->mark_cpu = cpu > t->mark_cpu ? cpu : t->mark_cpu;
        t->mark_core = core;
    }
    kunlock();
}

int rt_kernel_thread_times(RtThreadTime *out, int max) {
    int n = 0;
    klock();
    for (RtThread *t = K.all; t; t = t->all, n++)
        if (n < max)
            out[n] = (RtThreadTime){t->handle, t->prio, t->cur_core, t->entry, t->frame_cpu, t->frame_core,
                                    t->mark_cpu, t->mark_core, t->state == T_EXITED};
    kunlock();
    return n;
}

uint32_t rt_kernel_current(void) { return tcur ? tcur->handle : 0; }

RtKernelStats rt_kernel_stats(void) {
    klock();
    RtKernelStats s = K.st;
    s.spin_passes = 0;
    for (RtThread *t = K.all; t; t = t->all) s.spin_passes += __atomic_load_n(&t->spin_passes, __ATOMIC_RELAXED);
    kunlock();
    return s;
}

void rt_kernel_dump(void) {
    for (int i = 0; i < MAX_CORES; i++) {
        RtThread *o = K.owner[i];
        if (o) rt_log("kernel dump: core %d held by %#x", i, o->handle);
    }
    for (RtThread *t = K.all; t; t = t->all)
        rt_log("kernel dump: thread %#x state %d priority %d ideal %d mask %#llx core %d ready %d kicked %d pending %u "
               "wait %d woke %d pause %d/%d target %#x ap_pending %d spin passes %llu fibre %#llx",
               t->handle, t->state, t->prio, t->ideal, (unsigned long long)t->mask, t->cur_core, t->ready, t->kicked,
               t->cur->pending, t->wkind, t->woke, t->pause_req, t->paused, t->target ? t->target->handle : 0,
               t->ap_pending, (unsigned long long)t->spin_passes, (unsigned long long)kfibre_addr(t->fibre));
    rt_log("kernel dump: live %llu, async on %d sent %llu taken %llu elsewhere %llu failed %llu, deferred %llu, "
           "safe-point yields %llu, spin hand-offs %llu", (unsigned long long)K.live, K.async_on,
           (unsigned long long)K.st.async_sent, (unsigned long long)K.st.async_taken,
           (unsigned long long)K.st.async_elsewhere, (unsigned long long)K.st.async_failed,
           (unsigned long long)K.st.deferred, (unsigned long long)K.st.safepoint_yields,
           (unsigned long long)K.st.spin_handoffs);
}
