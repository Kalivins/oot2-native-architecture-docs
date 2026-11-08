// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The kernel's synchronization objects and their SVCs: guest mutexes (ArbitrateLock, ArbitrateUnlock), process-wide
// key waits (the SDK's condition variables), the address arbiter, events, handles and WaitSynchronization. Semantics
// follow the public Horizon kernel (KConditionVariable, KAddressArbiter, KSynchronization); waiters wake in kernel
// order, the best priority first, then first come.
//
// Adapted from DKB Recomp runtime/kernel.c at 9d8fa02 (GPL-3.0-or-later; adapted there from MacNativeGames
// games/totk/runtime/threads.c at b75be8c and LAS Recomp runtime/kernel.c at da35b83, MIT): the mutex word (owner
// handle | waiters bit) and its hand-off, the condition variable's key set before the lock is released, the arbiter's
// three signal kinds, events with two ends. From oot-pc-recomp runtime/kernel.c at 8e09aaa (GPL-3.0-or-later):
// WaitForAddress type 3, a 64-bit WaitIfEqual the Switch 2 SDK's atomic wait uses. Changed here: Horizon's argument
// checks and results (alignment, guest memory, handle kinds, OutOfRange), arbitration and signal types the kernel does
// not know are named fatals, and a woken thread is queued for a core by its waker (kernel.c).
#include <stdlib.h>
#include <string.h>

#include "kern.h"

static int guest_word(uint64_t a, uint64_t size, int write) { return rt_mem_commit_io(a, size, write); }

// The waiter of this kind to wake first: lock waits match by lock address, condition waits by key, arbiter waits by
// address.
static RtThread *pick(int kind, uint64_t a) {
    RtThread *b = NULL;
    for (RtThread *t = K.waiters; t; t = t->wnext) {
        if (t->wkind != kind || (kind == W_CONDVAR ? t->wkey : t->waddr) != a) continue;
        if (!b || t->prio < b->prio || (t->prio == b->prio && t->wseq < b->wseq)) b = t;
    }
    return b;
}

// Release the mutex at a: hand it to the best waiter, or clear it.
static void lock_handoff(uint64_t a) {
    RtThread *w = pick(W_LOCK, a);
    if (!w) {
        kst32(a, 0);
        return;
    }
    int more = 0;
    for (RtThread *t = K.waiters; t && !more; t = t->wnext) more = t != w && t->wkind == W_LOCK && t->waddr == a;
    kst32(a, w->wtag | (more ? WAITERS_BIT : 0));
    kwake(w, KR_OK);
}

// A signalled condition waiter takes its mutex: at once if it is free, else it waits for it behind the others.
static void lock_acquire_for(RtThread *w, uint64_t a, uint32_t tag) {
    uint32_t v = kld32(a);
    for (;;) {
        if (!v) {
            if (kcas32(a, &v, tag)) {
                kwake(w, KR_OK);
                return;
            }
            continue;
        }
        if (kcas32(a, &v, v | WAITERS_BIT)) break;
    }
    w->wkind = W_LOCK;
    w->waddr = a;
    w->wtag = tag;
    w->wseq = ++K.seq;
    kkick();   // the owner inherits w's priority
}

// 0x1A ArbitrateLock(w0 owner handle, x1 mutex address, w2 tag): waits while the word still reads owner | waiters.
// The word is read before the handle, as Horizon does: a waiter can set the waiters bit and reach this SVC after the
// owner has released the mutex, ended and had its handle closed, and that is a success, not an invalid handle.
static void svc_arbitrate_lock(Cpu *c, uint64_t pc) {
    (void)pc;
    RtThread *t = kthread(0x1A);
    uint64_t a = c->x[1];
    uint32_t owner = (uint32_t)c->x[0], r = (a & 3) ? KR_INVALID_ADDRESS : !guest_word(a, 4, 1) ? KR_INVALID_CURRENT_MEMORY : KR_OK;
    klock();
    if (!r && kld32(a) == (owner | WAITERS_BIT)) {
        if (!h_get(owner, O_THREAD)) {
            r = KR_INVALID_HANDLE;
        } else {
            t->waddr = a;
            t->wtag = (uint32_t)c->x[2];
            r = kblock(t, W_LOCK, -1);
        }
    }
    kunlock();
    c->x[0] = r;
}

static void svc_arbitrate_unlock(Cpu *c, uint64_t pc) {   // 0x1B ArbitrateUnlock(x0 mutex address)
    (void)pc;
    uint64_t a = c->x[0];
    uint32_t r = (a & 3) ? KR_INVALID_ADDRESS : !guest_word(a, 4, 1) ? KR_INVALID_CURRENT_MEMORY : KR_OK;
    klock();
    if (!r) lock_handoff(a);
    kunlock();
    c->x[0] = r;
}

// 0x1C WaitProcessWideKeyAtomic(x0 mutex address, x1 key address, w2 tag, x3 timeout). The key is set before the mutex
// is released: the SDK's signal reads the key in user mode and skips SVC 0x1D when it is 0, and user mode takes the
// mutex without the kernel lock, so releasing first lets another thread take the mutex, change the condition, read
// key 0 and wake nobody (MacNativeGames threads.c). On a timeout the waiter returns without the mutex; user mode takes
// it.
static void svc_wait_process_wide_key(Cpu *c, uint64_t pc) {
    (void)pc;
    RtThread *t = kthread(0x1C);
    uint64_t a = c->x[0], key = c->x[1];
    uint32_t r = ((a | key) & 3) ? KR_INVALID_ADDRESS
               : !guest_word(a, 4, 1) || !guest_word(key, 4, 1) ? KR_INVALID_CURRENT_MEMORY : KR_OK;
    klock();
    if (!r) {
        kst32(key, 1);
        t->wkey = key;
        t->waddr = a;
        t->wtag = (uint32_t)c->x[2];
        lock_handoff(a);
        r = kblock(t, W_CONDVAR, (int64_t)c->x[3]);
    }
    kunlock();
    c->x[0] = r;
}

// 0x1D SignalProcessWideKey(x0 key address, w1 count; 0 or less wakes them all)
static void svc_signal_process_wide_key(Cpu *c, uint64_t pc) {
    (void)pc;
    uint64_t key = c->x[0];
    int count = (int)(uint32_t)c->x[1], n = 0;
    if ((key & 3) || !guest_word(key, 4, 1)) {
        c->x[0] = (key & 3) ? KR_INVALID_ADDRESS : KR_INVALID_CURRENT_MEMORY;
        return;
    }
    klock();
    for (RtThread *w; (count <= 0 || n < count) && (w = pick(W_CONDVAR, key)); n++) lock_acquire_for(w, w->waddr, w->wtag);
    int left = 0;
    for (RtThread *w = K.waiters; w && !left; w = w->wnext) left = w->wkind == W_CONDVAR && w->wkey == key;
    if (!left) kst32(key, 0);
    kunlock();
    c->x[0] = KR_OK;
}

// 0x34 WaitForAddress(x0 address, w1 type, x2 value, x3 timeout): 0 wait if less than, 1 decrement and wait if less
// than, 2 wait if equal, 3 wait if equal on 64 bits.
static void svc_wait_for_address(Cpu *c, uint64_t pc) {
    RtThread *t = kthread(0x34);
    uint64_t a = c->x[0];
    int type = (int)(uint32_t)c->x[1];
    if (type < 0 || type > 3) {
        c->pc = pc;
        rt_unimplemented(c, "WaitForAddress", "arbitration type %d", type);
    }
    uint64_t size = type == 3 ? 8 : 4;
    uint32_t r = (a & (size - 1)) ? KR_INVALID_ADDRESS : !guest_word(a, size, type == 1) ? KR_INVALID_CURRENT_MEMORY : KR_OK;
    klock();
    if (!r) {
        int ok;
        if (type == 3) {
            ok = __atomic_load_n((uint64_t *)(uintptr_t)a, __ATOMIC_SEQ_CST) == c->x[2];
        } else {
            int32_t val = (int32_t)c->x[2];
            uint32_t v = kld32(a);
            ok = type == 2 ? (int32_t)v == val : (int32_t)v < val;
            while (ok && type == 1 && !kcas32(a, &v, v - 1)) ok = (int32_t)v < val;
        }
        if (!ok) r = KR_INVALID_STATE;
        else {
            t->waddr = a;
            r = kblock(t, W_ADDRESS, (int64_t)c->x[3]);
        }
    }
    kunlock();
    c->x[0] = r;
}

// 0x35 SignalToAddress(x0 address, w1 type, w2 value, w3 count): 0 wake, 1 increment if equal and wake, 2 if equal
// modify by the number of waiters and wake.
static void svc_signal_to_address(Cpu *c, uint64_t pc) {
    uint64_t a = c->x[0];
    int type = (int)(uint32_t)c->x[1], count = (int)(uint32_t)c->x[3];
    int32_t val = (int32_t)c->x[2];
    if (type < 0 || type > 2) {
        c->pc = pc;
        rt_unimplemented(c, "SignalToAddress", "signal type %d", type);
    }
    uint32_t r = (a & 3) ? KR_INVALID_ADDRESS : !guest_word(a, 4, type != 0) ? KR_INVALID_CURRENT_MEMORY : KR_OK;
    klock();
    if (!r && type) {
        int waiting = 0;
        for (RtThread *w = K.waiters; w; w = w->wnext) waiting += w->wkind == W_ADDRESS && w->waddr == a;
        int32_t nv = val + 1;
        if (type == 2) {
            if (count <= 0) nv = waiting > 0 ? val - 1 : val + 1;
            else nv = waiting == 0 ? val + 1 : waiting <= count ? val - 1 : val;
        }
        uint32_t e = (uint32_t)val;
        if (!kcas32(a, &e, (uint32_t)nv)) r = KR_INVALID_STATE;
    }
    for (int n = 0; !r && (count <= 0 || n < count); n++) {
        RtThread *w = pick(W_ADDRESS, a);
        if (!w) break;
        kwake(w, KR_OK);
    }
    kunlock();
    c->x[0] = r;
}

// ---- events and handles
static KEvent *event_of(uint32_t h, int writable_only) {
    KEvent *e = h_get(h, O_EVENT_W);
    return e || writable_only ? e : h_get(h, O_EVENT_R);
}

static void svc_signal_event(Cpu *c, uint64_t pc) {   // 0x11 SignalEvent(w0 writable end)
    (void)pc;
    klock();
    KEvent *e = event_of((uint32_t)c->x[0], 1);
    if (e) {
        e->signaled = 1;
        kwake_sync();
    }
    kunlock();
    c->x[0] = e ? KR_OK : KR_INVALID_HANDLE;
}

static void svc_clear_event(Cpu *c, uint64_t pc) {   // 0x12 ClearEvent(w0 either end)
    (void)pc;
    klock();
    KEvent *e = event_of((uint32_t)c->x[0], 0);
    if (e) e->signaled = 0;
    kunlock();
    c->x[0] = e ? KR_OK : KR_INVALID_HANDLE;
}

static void svc_reset_signal(Cpu *c, uint64_t pc) {   // 0x17 ResetSignal(w0 readable end): InvalidState if not signalled
    (void)pc;
    klock();
    KEvent *e = h_get((uint32_t)c->x[0], O_EVENT_R);
    uint32_t r = !e ? KR_INVALID_HANDLE : !e->signaled ? KR_INVALID_STATE : KR_OK;
    if (!r) e->signaled = 0;
    kunlock();
    c->x[0] = r;
}

// 0x16 CloseHandle(w0): an event goes with its last end; a thread object once it has also ended; a host object's close
// function runs after the lock is released.
static void svc_close_handle(Cpu *c, uint64_t pc) {
    (void)pc;
    uint32_t h = (uint32_t)c->x[0], r = KR_OK;
    KHost *host = NULL;
    klock();
    int kind = h == H_CURRENT_THREAD || h == H_CURRENT_PROCESS ? O_FREE : h_kind(h);
    if (kind == O_THREAD) {
        RtThread *t = h_get(h, O_THREAD);
        h_free(h);
        t->closed = 1;
        kthread_free_if_done(t);
    } else if (kind == O_EVENT_R || kind == O_EVENT_W) {
        KEvent *e = h_get(h, kind);
        h_free(h);
        if (--e->refs == 0) free(e);
    } else if (kind == O_HOST) {
        host = h_get(h, O_HOST);
        h_free(h);
    } else {
        r = KR_INVALID_HANDLE;
    }
    kunlock();
    if (host) {
        host->close(host->obj);
        free(host);
    }
    c->x[0] = r;
}

static int signaled(uint32_t h) {   // holding the lock; h is a thread or a readable event
    RtThread *t = h_get(h, O_THREAD);
    if (t) return t->state == T_EXITED;
    KEvent *e = h_get(h, O_EVENT_R);
    return e && e->signaled;
}

// 0x18 WaitSynchronization(x1 handles, w2 count, x3 timeout -> w1 index of the first signalled object). Threads are
// signalled once they end, readable events while set. A count of 0 sleeps until the timeout.
static void svc_wait_synchronization(Cpu *c, uint64_t pc) {
    (void)pc;
    RtThread *t = kthread(0x18);
    uint64_t hs = c->x[1];
    int32_t n = (int32_t)c->x[2];
    int64_t ns = (int64_t)c->x[3];
    uint32_t h[64], r = n < 0 || n > 64 ? KR_OUT_OF_RANGE
                      : n && !guest_word(hs, 4ull * (uint64_t)n, 0) ? KR_INVALID_POINTER : KR_OK;
    for (int i = 0; !r && i < n; i++) h[i] = LD32(hs + 4ull * (uint64_t)i);
    int idx = -1;
    uint64_t start = plat_now_ns();
    klock();
    for (int i = 0; !r && i < n; i++)
        if (h_kind(h[i]) != O_THREAD && h_kind(h[i]) != O_EVENT_R) r = KR_INVALID_HANDLE;
    if (!r && n) {   // the waiter view: a wait that follows a timed-out wait on the same handles continues its run
        int same = t->ws_run && t->ws_n == n && !memcmp(t->ws_h, h, 4u * (unsigned)n) && start - t->ws_end < RT_WAIT_RUN_GAP_NS;
        if (!same) {
            t->ws_since = start;
            t->ws_n = n;
            memcpy(t->ws_h, h, 4u * (unsigned)n);
        }
        t->ws_run = 1;
        t->ws_in = 1;
    }
    while (!r) {
        if (t->cancel) {
            t->cancel = 0;
            r = KR_CANCELLED;
            break;
        }
        for (int i = 0; i < n && idx < 0; i++)
            if (signaled(h[i])) idx = i;
        if (idx >= 0) break;
        int64_t left = ns;
        if (ns > 0) {
            uint64_t spent = plat_now_ns() - start;
            if (spent >= (uint64_t)ns) {
                r = KR_TIMED_OUT;
                break;
            }
            left = ns - (int64_t)spent;
        }
        uint32_t w = kblock(t, W_SYNC, left);
        if (w == KR_TIMED_OUT) r = w;
        else if (w == KR_CANCELLED) r = w, t->cancel = 0;
    }
    if (n) {
        t->ws_in = 0;
        t->ws_end = plat_now_ns();
        if (r != KR_TIMED_OUT) t->ws_run = 0;   // only timeouts keep a run going
    }
    kunlock();
    c->x[0] = r;
    c->x[1] = idx >= 0 ? (uint64_t)idx : 0;
}

// The guest threads that wait on a readable handle (or a thread handle): blocked in a WaitSynchronization that names it, or
// whose last such wait timed out less than RT_WAIT_RUN_GAP_NS ago, a thread that polls with a timeout. since is the start of
// the wait, or of the run of timed-out waits on the same handles.
int rt_kernel_waiters(uint32_t handle, RtWaiter *out, int max) {
    int n = 0;
    uint64_t now = plat_now_ns();
    klock();
    for (RtThread *t = K.all; t; t = t->all) {
        if (t->state == T_EXITED || !t->ws_n) continue;
        int has = 0;
        for (int i = 0; i < t->ws_n && !has; i++) has = t->ws_h[i] == handle;
        if (!has || !(t->ws_in || (t->ws_run && now - t->ws_end < RT_WAIT_RUN_GAP_NS))) continue;
        if (n < max) out[n] = (RtWaiter){t->handle, t->entry, t->ws_since, t->ws_in};
        n++;
    }
    kunlock();
    return n < max ? n : max;
}

// 0x19 CancelSynchronization(w0 thread): its WaitSynchronization returns Cancelled, now or at its next one.
static void svc_cancel_synchronization(Cpu *c, uint64_t pc) {
    (void)pc;
    klock();
    RtThread *t = h_get((uint32_t)c->x[0], O_THREAD);
    if (t && t->wkind == W_SYNC) kwake(t, KR_CANCELLED);
    else if (t) t->cancel = 1;
    kunlock();
    c->x[0] = t ? KR_OK : KR_INVALID_HANDLE;
}

// ---- events for host services
uint32_t rt_kernel_event_create(uint32_t *readable) {
    KEvent *e = calloc(1, sizeof *e);
    if (!e) rt_fatal("kernel: event allocation");
    e->refs = 2;
    klock();
    uint32_t w = h_new(O_EVENT_W, e), rd = w ? h_new(O_EVENT_R, e) : 0;
    kunlock();
    if (!rd) rt_fatal("kernel: the handle table (%u entries) is full", K.cfg.handle_table_size);
    *readable = rd;
    return w;
}

void rt_kernel_event_signal(uint32_t writable) {
    klock();
    KEvent *e = h_get(writable, O_EVENT_W);
    if (!e) {
        kunlock();
        rt_fatal("rt_kernel_event_signal(%#x): not the writable end of an event", writable);
    }
    e->signaled = 1;
    kwake_sync();
    kunlock();
}

// The host's own end of an event it made, either one: the handle goes, and the event goes with its last end (a guest still
// holding the other end keeps the event until its own CloseHandle).
void rt_kernel_event_close(uint32_t handle) {
    klock();
    int kind = h_kind(handle);
    KEvent *e = kind == O_EVENT_W || kind == O_EVENT_R ? h_get(handle, kind) : NULL;
    if (e) {
        h_free(handle);
        if (--e->refs == 0) free(e);
    }
    kunlock();
    if (!e) rt_fatal("rt_kernel_event_close(%#x): not an end of an event", handle);
}

// ---- host objects
uint32_t rt_kernel_handle_new(void *obj, void (*close)(void *obj)) {
    if (!close) rt_fatal("rt_kernel_handle_new: no close function");
    KHost *o = malloc(sizeof *o);
    if (!o) rt_fatal("kernel: host object allocation");
    *o = (KHost){obj, close};
    klock();
    uint32_t h = h_new(O_HOST, o);
    kunlock();
    if (!h) free(o);
    return h;
}

void *rt_kernel_handle_get(uint32_t h, void (*close)(void *obj)) {
    klock();
    KHost *o = h_get(h, O_HOST);
    void *obj = o && o->close == close ? o->obj : NULL;
    kunlock();
    return obj;
}

void ksync_register(void) {
    rt_svc_register(0x11, svc_signal_event);
    rt_svc_register(0x12, svc_clear_event);
    rt_svc_register(0x16, svc_close_handle);
    rt_svc_register(0x17, svc_reset_signal);
    rt_svc_register(0x18, svc_wait_synchronization);
    rt_svc_register(0x19, svc_cancel_synchronization);
    rt_svc_register(0x1A, svc_arbitrate_lock);
    rt_svc_register(0x1B, svc_arbitrate_unlock);
    rt_svc_register(0x1C, svc_wait_process_wide_key);
    rt_svc_register(0x1D, svc_signal_process_wide_key);
    rt_svc_register(0x34, svc_wait_for_address);
    rt_svc_register(0x35, svc_signal_to_address);
}
