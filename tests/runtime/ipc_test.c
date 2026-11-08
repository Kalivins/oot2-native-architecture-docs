// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the services' side of the IPC layer (runtime/hos/ipc.c, ipc.h): buffer descriptors of every kind
// in a request (A, B, W, X and C, with addresses above 4 GiB), copied and moved handles, the process id, answers with
// handles and pointer descriptors, sessions with a state and a close callback, the names sm: serves, CMIF domains
// (convert, requests to objects, object ids in answers, close, destruction with the domain), and the named fatals of a
// command nobody serves. The service is a test one made of host functions; the client is tests/runtime/cmif_client.h;
// the guest programs are C functions in the lifter's shape. No game code is involved.
//
//   ipc_test CASE DIR   runs one case with DIR as the game folder; prints "ok CASE" and exits 0, or exits 1 with a
//                       FAIL line. The cases that end in a named fatal exit with the runtime's code
//                       (tests/test_runtime_ipc.py).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/boot.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "plat.h"
#include "av1_fake.h"   // runtime/hos links the AV1 service; the fake stands for the decoder
#include "../../runtime/hos/hos.h"
#include "../../runtime/hos/ipc.h"

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#define MB MOD_MAIN_BASE
#define G(i) (MB + 0x100ull * (i))
#define PC 0x7000000040ull
#define CL_PC PC
#include "cmif_client.h"

#define R_INVALID_HANDLE 0xE401u

static uint64_t HEAP;
static const char *g_case;
static volatile int g_opened, g_closed;   // child sessions made and closed

// ---- the test service: commands 0 to 7 on tsta:u to tstd:u, and a child session
typedef struct Child { uint32_t n; } Child;
static void child_closed(void *st) {
    free(st);
    __atomic_add_fetch(&g_closed, 1, __ATOMIC_ACQ_REL);
}
static void child_handler(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    Child *ch = s->state;
    switch (rq->cmd) {
    case 0: {   // bump: the child's own count
        uint32_t n = ++ch->n;
        ipc_out(rs, &n, 4);
        return;
    }
    default: ipc_unknown_command(c, s, rq);
    }
}
static void service_handler(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    switch (rq->cmd) {
    case 0: {   // sum(u32 a, u32 b) -> u32 sum, u32 has_pid, u64 pid
        uint32_t a, b;
        memcpy(&a, ipc_arg(c, s, rq, 0, 4), 4);
        memcpy(&b, ipc_arg(c, s, rq, 4, 4), 4);
        uint32_t sum = a + b, has = rq->has_pid;
        uint64_t pid = rq->pid;
        ipc_out(rs, &sum, 4);
        ipc_out(rs, &has, 4);
        ipc_out(rs, &pid, 8);
        return;
    }
    case 1: {   // buffers: A reversed into B, X upper-cased into the receive buffer C, W incremented in place
        uint8_t t[32];
        const IpcBuffer *a = &rq->a[0], *b = &rq->b[0], *w = &rq->w[0], *x = &rq->x[0];
        if (rq->na != 1 || rq->nb != 1 || rq->nw != 1 || rq->nx != 1 || rq->nc < 1) {
            rs->result = 0x7000;
            return;
        }
        if (a->size > sizeof t || !ipc_read(a, 0, t, a->size)) { rs->result = 0x7001; return; }
        for (uint64_t i = 0; i < a->size / 2; i++) {
            uint8_t u = t[i];
            t[i] = t[a->size - 1 - i];
            t[a->size - 1 - i] = u;
        }
        if (b->size < a->size || !ipc_write(b, 0, t, a->size)) { rs->result = 0x7002; return; }
        if (x->size > sizeof t || !ipc_read(x, 0, t, x->size)) { rs->result = 0x7003; return; }
        for (uint64_t i = 0; i < x->size; i++)
            if (t[i] >= 'a' && t[i] <= 'z') t[i] = (uint8_t)(t[i] - 32);
        if (!ipc_answer_pointer(rs, rq, 0, t, x->size)) { rs->result = 0x7004; return; }
        if (w->size > sizeof t || !ipc_read(w, 0, t, w->size)) { rs->result = 0x7005; return; }
        for (uint64_t i = 0; i < w->size; i++) t[i]++;
        if (!ipc_write(w, 0, t, w->size)) { rs->result = 0x7006; return; }
        if (ipc_read(w, 1, t, w->size)) { rs->result = 0x7007; return; }   // one byte past the buffer: refused
        uint64_t d[8] = {a->addr, a->size, b->addr, b->size, w->addr, x->addr, x->size,
                         (uint64_t)a->attr | (uint64_t)x->attr << 8 | (uint64_t)b->attr << 16 | (uint64_t)w->attr << 24};
        ipc_out(rs, d, sizeof d);
        return;
    }
    case 2: {   // handles: a copied readable end and a moved writable end of a new event; what the request carried
        uint32_t readable, writable = rt_kernel_event_create(&readable);
        rs->copy[rs->ncopy++] = readable;
        rs->move[rs->nmove++] = writable;
        uint32_t d[4] = {rq->ncopy, rq->ncopy ? rq->copy[0] : 0, rq->nmove, rq->nmove ? rq->move[0] : 0};
        ipc_out(rs, d, sizeof d);
        return;
    }
    case 3: {   // open(u32 count): count child sessions
        uint32_t n;
        memcpy(&n, ipc_arg(c, s, rq, 0, 4), 4);
        for (uint32_t i = 0; i < n; i++) {
            Child *ch = calloc(1, sizeof *ch);
            __atomic_add_fetch(&g_opened, 1, __ATOMIC_ACQ_REL);
            ipc_answer_session(rs, ipc_session_new("tchild", child_handler, ch, child_closed));
        }
        return;
    }
    case 4: {   // transfer memory of the copied handle: found, addr, size, perm
        IpcTransferMemory t = {0};
        uint32_t ok = rq->ncopy && ipc_transfer_memory(rq->copy[0], &t);
        ipc_out(rs, &ok, 4);
        ipc_out(rs, &t.addr, 8);
        ipc_out(rs, &t.size, 8);
        ipc_out(rs, &t.perm, 4);
        return;
    }
    case 5:   // 64 bytes of arguments, whatever the request carries
        ipc_arg(c, s, rq, 0, 64);
        return;
    default: ipc_unknown_command(c, s, rq);
    }
}
static const char *const NAMES[] = {"tsta:u", "tstb:u", "tstc:u", "tstd:u"};

// ---- guest helpers
static inline uint64_t sv(Cpu *c, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    c->x[0] = a0, c->x[1] = a1, c->x[2] = a2, c->x[3] = a3;
    rt_svc(c, imm, PC);
    return c->x[0];
}
static uint32_t open_sm(Cpu *c) {
    memcpy((void *)(uintptr_t)(HEAP + 0x100), "sm:", 4);
    CHECK(sv(c, 0x1F, 0, HEAP + 0x100, 0, 0) == 0);
    uint32_t h = (uint32_t)c->x[1];
    ClReq rq = {.tipc = 1, .cmd = 0, .pid = 1};
    ClRep rp;
    CHECK(cl_send(c, h, &rq, &rp, 0) == 0 && rp.result == 0);   // RegisterClient
    return h;
}
// GetServiceHandle over TIPC: the moved session handle, or 0 with the result in *result
static uint32_t get_service(Cpu *c, uint32_t sm, const char *name, uint32_t *result) {
    uint64_t n = 0;
    memcpy(&n, name, strlen(name));
    ClReq rq = {.tipc = 1, .cmd = 1, .data = &n, .size = 8};
    ClRep rp;
    CHECK(cl_send(c, sm, &rq, &rp, 0) == 0);
    if (result) *result = rp.result;
    if (rp.result) return 0;
    CHECK(rp.nmove == 1 && rp.ncopy == 0 && rp.move[0] != 0);
    return rp.move[0];
}
// A CMIF request of command cmd with arguments to a session (object 0: no domain), the answer in rp
static uint32_t call(Cpu *c, uint32_t h, uint32_t domain_obj, uint32_t cmd, const void *data, uint32_t size, ClRep *rp,
                     uint32_t expect) {
    ClReq rq = {.cmd = cmd, .data = data, .size = size, .domain_obj = domain_obj};
    CHECK(cl_send(c, h, &rq, rp, expect) == 0);
    return rp->result;
}
static uint32_t open_children(Cpu *c, uint32_t h, uint32_t domain_obj, uint32_t n, ClRep *rp) {
    return call(c, h, domain_obj, 3, &n, 4, rp, 0);
}
static uint32_t bump(Cpu *c, uint32_t h, uint32_t domain_obj) {
    ClRep rp;
    CHECK(call(c, h, domain_obj, 0, NULL, 0, &rp, 4) == 0);
    return cl_u32(rp.data);
}
static uint32_t control(Cpu *c, uint32_t h, uint32_t cmd, ClRep *rp) {
    ClReq rq = {.control = 1, .cmd = cmd};
    CHECK(cl_send(c, h, &rq, rp, 4) == 0);
    return rp->result;
}
static void close_handle(Cpu *c, uint32_t h) { CHECK(sv(c, 0x16, h, 0, 0, 0) == 0); }

// ---- guest programs (each ends its thread)
static void g_basic(Cpu *c) {
    uint32_t sm = open_sm(c), r;
    // four names served, each its own session
    uint32_t svc[4];
    for (int i = 0; i < 4; i++) {
        svc[i] = get_service(c, sm, NAMES[i], &r);
        CHECK(svc[i] && r == 0);
        for (int j = 0; j < i; j++) CHECK(svc[i] != svc[j]);
    }
    uint32_t h = svc[0];
    // sum, in CMIF with the process id sent in the special header, and in TIPC
    uint32_t args[2] = {40, 2};
    ClRep rp;
    CHECK(call(c, h, 0, 0, args, 8, &rp, 16) == 0);
    CHECK(cl_u32(rp.data) == 42 && cl_u32(rp.data + 4) == 0 && rp.ncopy == 0 && rp.nmove == 0);
    ClReq rq = {.cmd = 0, .data = args, .size = 8, .pid = 1};
    CHECK(cl_send(c, h, &rq, &rp, 16) == 0 && rp.result == 0);
    CHECK(cl_u32(rp.data) == 42 && cl_u32(rp.data + 4) == 1 && cl_u32(rp.data + 8) == 0x1234);
    rq = (ClReq){.tipc = 1, .cmd = 0, .data = args, .size = 8};
    CHECK(cl_send(c, svc[3], &rq, &rp, 0) == 0 && rp.result == 0);   // any session answers TIPC
    CHECK(cl_u32(rp.data) == 42);
    // buffers of every kind. The addresses are above 4 GiB when the heap is: the descriptors carry the high bits
    uint64_t base = HEAP + 0x2000;
    static const char in_a[] = "0123456789abcdef";
    static const char in_x[] = "mixed Case 9";
    memcpy((void *)(uintptr_t)base, in_a, 16);
    memcpy((void *)(uintptr_t)(base + 0x100), in_x, sizeof in_x - 1);
    memset((void *)(uintptr_t)(base + 0x200), 0x55, 32);   // B
    memset((void *)(uintptr_t)(base + 0x300), 0x55, 32);   // C
    memset((void *)(uintptr_t)(base + 0x400), 9, 8);       // W
    rq = (ClReq){.cmd = 1, .na = 1, .nb = 1, .nw = 1, .nx = 1, .nc = 1};
    rq.a[0] = (ClBuf){base, 16, 1};
    rq.b[0] = (ClBuf){base + 0x200, 32, 3};
    rq.w[0] = (ClBuf){base + 0x400, 8, 0};
    rq.x[0] = (ClBuf){base + 0x100, sizeof in_x - 1, 0};
    rq.c[0] = (ClBuf){base + 0x300, 64, 0};
    for (int offset_zero = 0; offset_zero < 2; offset_zero++) {
        memset((void *)(uintptr_t)(base + 0x200), 0x55, 32);
        memset((void *)(uintptr_t)(base + 0x300), 0x55, 32);
        memset((void *)(uintptr_t)(base + 0x400), 9, 8);
        rq.recv_offset_zero = offset_zero;   // the receive list at the header's offset, and where the raw data ends
        CHECK(cl_send(c, h, &rq, &rp, 64) == 0 && rp.result == 0);
        const uint64_t *d = (const uint64_t *)rp.data;
        CHECK(d[0] == base && d[1] == 16 && d[2] == base + 0x200 && d[3] == 32 && d[4] == base + 0x400 &&
              d[5] == base + 0x100 && d[6] == sizeof in_x - 1);
        CHECK(d[7] == (1ull | 0ull << 8 | 3ull << 16 | 0ull << 24));   // the modes of A and B, the index of X
        CHECK(!memcmp((const void *)(uintptr_t)(base + 0x200), "fedcba9876543210", 16));
        CHECK(((const uint8_t *)(uintptr_t)(base + 0x200))[16] == 0x55);
        CHECK(!memcmp((const void *)(uintptr_t)(base + 0x300), "MIXED CASE 9", 12));
        CHECK(((const uint8_t *)(uintptr_t)(base + 0x300))[12] == 0x55);
        CHECK(((const uint8_t *)(uintptr_t)(base + 0x400))[0] == 10 && ((const uint8_t *)(uintptr_t)(base + 0x400))[7] == 10);
        CHECK(rp.nx == 1 && rp.x[0].addr == base + 0x300 && rp.x[0].size == 12 && rp.x[0].attr == 0);   // names what was written
    }
    // a receive buffer too small for the answer is the service's to refuse; a write outside guest memory is refused
    rq.c[0].size = 4;
    CHECK(cl_send(c, h, &rq, &rp, 64) == 0 && rp.result == 0x7004 && rp.nx == 0);
    rq.c[0].size = 64;
    rq.b[0].addr = 0x7e00000000ull;   // mapped nowhere
    CHECK(cl_send(c, h, &rq, &rp, 64) == 0 && rp.result == 0x7002);
    rq.b[0].addr = base + 0x200;
    // two receive buffers: the list is read in its own mode, which is not the single-buffer mode
    rq.nc = 2;
    rq.c[1] = (ClBuf){base + 0x800, 16, 0};
    CHECK(cl_send(c, h, &rq, &rp, 64) == 0 && rp.result == 0 && rp.nx == 1);
    // handles: one copied in and one moved in reach the service; it answers one copied and one moved
    uint32_t readable, writable = rt_kernel_event_create(&readable);
    rq = (ClReq){.cmd = 2, .ncopy = 1, .nmove = 1};
    rq.copy[0] = readable;
    rq.move[0] = writable;
    CHECK(cl_send(c, h, &rq, &rp, 16) == 0 && rp.result == 0);
    CHECK(rp.ncopy == 1 && rp.nmove == 1 && rp.copy[0] && rp.move[0] && rp.copy[0] != rp.move[0]);
    const uint32_t *d4 = (const uint32_t *)rp.data;
    CHECK(d4[0] == 1 && d4[1] == readable && d4[2] == 1 && d4[3] == writable);
    // the answered ends are one event: signal the moved writable end, the copied readable end is signalled
    uint32_t rd = rp.copy[0], wr = rp.move[0];
    memcpy((void *)(uintptr_t)(HEAP + 0x180), &rd, 4);
    CHECK(sv(c, 0x18, 0, HEAP + 0x180, 1, 0) == 0xEA01u);   // nothing yet: timed out
    CHECK(sv(c, 0x11, wr, 0, 0, 0) == 0);
    CHECK(sv(c, 0x18, 0, HEAP + 0x180, 1, 1000000000ull) == 0);
    // transfer memory: a handle the client made, found by the service; a handle that is no transfer memory is not
    CHECK(sv(c, 0x15, 0, HEAP + 0x10000, 0x2000, 3) == 0);
    uint32_t t = (uint32_t)c->x[1];
    rq = (ClReq){.cmd = 4, .ncopy = 1};
    rq.copy[0] = t;
    CHECK(cl_send(c, h, &rq, &rp, 24) == 0 && rp.result == 0);
    CHECK(cl_u32(rp.data) == 1 && !memcmp(rp.data + 4, &(uint64_t){HEAP + 0x10000}, 8) &&
          !memcmp(rp.data + 12, &(uint64_t){0x2000}, 8) && cl_u32(rp.data + 20) == 3);
    rq.copy[0] = h;   // a session
    CHECK(cl_send(c, h, &rq, &rp, 24) == 0 && cl_u32(rp.data) == 0);
    close_handle(c, t);
    // the same request through SendSyncRequestWithUserBuffer, on a page of its own
    uint64_t page = (HEAP + 0x40000 + 0xfff) & ~0xfffull;
    rq = (ClReq){.cmd = 0, .data = args, .size = 8, .user_buffer = page};
    CHECK(cl_send(c, h, &rq, &rp, 16) == 0 && rp.result == 0 && cl_u32(rp.data) == 42);
    // children: a session with state and a close callback, as a moved handle; its handle is a session of its own
    CHECK(open_children(c, h, 0, 1, &rp) == 0 && rp.nmove == 1);
    uint32_t child = rp.move[0];
    CHECK(bump(c, child, 0) == 1 && bump(c, child, 0) == 2);
    CHECK(g_opened == 1 && g_closed == 0);
    // a clone is the same object: closing one handle leaves it alive, closing both ends it
    CHECK(control(c, child, 2, &rp) == 0 && rp.nmove == 1);
    uint32_t clone = rp.move[0];
    CHECK(clone != child && bump(c, clone, 0) == 3);
    close_handle(c, child);
    CHECK(g_closed == 0);
    CHECK(bump(c, clone, 0) == 4);
    close_handle(c, clone);
    CHECK(g_closed == 1);
    // QueryPointerBufferSize
    CHECK(control(c, h, 3, &rp) == 0 && (cl_u32(rp.data) & 0xffff) == 0x8000);
    // three children in one answer
    CHECK(open_children(c, h, 0, 3, &rp) == 0 && rp.nmove == 3);
    for (int i = 0; i < 3; i++) CHECK(bump(c, rp.move[i], 0) == 1);
    for (int i = 0; i < 3; i++) close_handle(c, rp.move[i]);
    CHECK(g_opened == 4 && g_closed == 4);
    for (int i = 0; i < 4; i++) close_handle(c, svc[i]);
    close_handle(c, sm);
    sv(c, 0x0A, 0, 0, 0, 0);
}

static void g_domain(Cpu *c) {
    uint32_t sm = open_sm(c);
    uint32_t h = get_service(c, sm, "tsta:u", NULL);
    ClRep rp;
    // convert: the session is object 1, a second conversion of the same session is a fault (case domain_twice)
    CHECK(control(c, h, 0, &rp) == 0 && cl_u32(rp.data) == 1);
    // requests now carry the domain header; the answer to this one is a plain CMIF answer with no objects
    uint32_t args[2] = {40, 2};
    CHECK(call(c, h, 1, 0, args, 8, &rp, 16) == 0 && rp.nobj == 0 && cl_u32(rp.data) == 42);
    // the control commands stay plain on a domain
    CHECK(control(c, h, 3, &rp) == 0 && (cl_u32(rp.data) & 0xffff) == 0x8000);
    // a command that makes a session answers with an object id, not a handle
    CHECK(open_children(c, h, 1, 1, &rp) == 0 && rp.nmove == 0 && rp.nobj == 1 && rp.obj[0] == 2);
    CHECK(bump(c, h, 2) == 1 && bump(c, h, 2) == 2);
    CHECK(open_children(c, h, 1, 1, &rp) == 0 && rp.nobj == 1 && rp.obj[0] == 3);
    CHECK(bump(c, h, 3) == 1 && bump(c, h, 2) == 3);   // each object has its own state
    // buffers and handles through a domain object work as on a session
    uint64_t base = HEAP + 0x2000;
    memcpy((void *)(uintptr_t)base, "0123456789abcdef", 16);
    memcpy((void *)(uintptr_t)(base + 0x100), "abc", 3);
    memset((void *)(uintptr_t)(base + 0x400), 1, 4);
    ClReq rq = {.cmd = 1, .domain_obj = 1, .na = 1, .nb = 1, .nw = 1, .nx = 1, .nc = 1};
    rq.a[0] = (ClBuf){base, 16, 1};
    rq.b[0] = (ClBuf){base + 0x200, 16, 2};
    rq.w[0] = (ClBuf){base + 0x400, 4, 0};
    rq.x[0] = (ClBuf){base + 0x100, 3, 0};
    rq.c[0] = (ClBuf){base + 0x300, 16, 0};
    CHECK(cl_send(c, h, &rq, &rp, 64) == 0 && rp.result == 0 && rp.nx == 1);
    CHECK(!memcmp((const void *)(uintptr_t)(base + 0x300), "ABC", 3));
    rq = (ClReq){.cmd = 2, .domain_obj = 1};
    CHECK(cl_send(c, h, &rq, &rp, 16) == 0 && rp.result == 0 && rp.ncopy == 1 && rp.nmove == 1);   // handles stay handles
    // close object 2: its close callback runs; its id is free and the next session takes it
    CHECK(g_closed == 0);
    rq = (ClReq){.cmd = 0, .domain_obj = 2, .domain_kind = 2};
    CHECK(cl_send(c, h, &rq, &rp, 0) == 0 && rp.result == 0);
    CHECK(g_closed == 1);
    CHECK(open_children(c, h, 1, 2, &rp) == 0 && rp.nobj == 2 && rp.obj[0] == 2 && rp.obj[1] == 4);
    CHECK(bump(c, h, 2) == 1 && bump(c, h, 4) == 1);
    // a clone of the domain's session is the same domain
    CHECK(control(c, h, 2, &rp) == 0 && rp.nmove == 1);
    uint32_t clone = rp.move[0];
    CHECK(bump(c, clone, 3) == 2);
    close_handle(c, h);
    CHECK(g_closed == 1);   // the clone still holds the domain
    CHECK(bump(c, clone, 4) == 2);
    close_handle(c, clone);
    CHECK(g_opened == 4 && g_closed == 4);   // the domain went with its last handle, and its objects with it
    close_handle(c, sm);
    // a domain that is not full: filling it takes 62 objects, the 63rd is a fault (case domain_full)
    uint32_t h2 = get_service(c, open_sm(c), "tstb:u", NULL);
    CHECK(control(c, h2, 0, &rp) == 0);
    for (int i = 0; i < 31; i++) CHECK(open_children(c, h2, 1, 2, &rp) == 0 && rp.nobj == 2);
    CHECK(g_opened == 4 + 62);
    close_handle(c, h2);
    CHECK(g_closed == 4 + 62);
    sv(c, 0x0A, 0, 0, 0, 0);
}

static void g_fatal(Cpu *c) {
    uint32_t sm = open_sm(c), h;
    ClRep rp;
    ClReq rq = {0};
    if (!strcmp(g_case, "fifth")) {
        get_service(c, sm, "tste:u", NULL);
    } else if (!strcmp(g_case, "fifth_census")) {   // census mode: the fifth name is refused and the run goes on
        uint32_t r = 0;
        CHECK(get_service(c, sm, "tste:u", &r) == 0 && r == 0x415);
        CHECK(get_service(c, sm, "tstd:u", &r) != 0 && r == 0);
        sv(c, 0x0A, 0, 0, 0, 0);
    } else if (!strcmp(g_case, "unknown")) {
        h = get_service(c, sm, "tsta:u", NULL);
        call(c, h, 0, 99, NULL, 0, &rp, 0);
    } else if (!strcmp(g_case, "unknown_tipc")) {
        h = get_service(c, sm, "tstb:u", NULL);
        rq = (ClReq){.tipc = 1, .cmd = 99};
        cl_send(c, h, &rq, &rp, 0);
    } else if (!strcmp(g_case, "unknown_domain")) {
        h = get_service(c, sm, "tsta:u", NULL);
        control(c, h, 0, &rp);
        open_children(c, h, 1, 1, &rp);
        call(c, h, 2, 99, NULL, 0, &rp, 0);
    } else if (!strcmp(g_case, "arg_beyond")) {
        h = get_service(c, sm, "tsta:u", NULL);
        rq = (ClReq){.cmd = 5, .tipc = 1};   // 0 bytes of the 64 the command reads
        cl_send(c, h, &rq, &rp, 0);
    } else if (!strcmp(g_case, "domain_twice")) {
        h = get_service(c, sm, "tsta:u", NULL);
        control(c, h, 0, &rp);
        control(c, h, 0, &rp);
    } else if (!strcmp(g_case, "domain_gone")) {
        h = get_service(c, sm, "tsta:u", NULL);
        control(c, h, 0, &rp);
        call(c, h, 9, 0, NULL, 0, &rp, 0);
    } else if (!strcmp(g_case, "domain_closed")) {
        h = get_service(c, sm, "tsta:u", NULL);
        control(c, h, 0, &rp);
        open_children(c, h, 1, 1, &rp);
        rq = (ClReq){.cmd = 0, .domain_obj = 2, .domain_kind = 2};
        cl_send(c, h, &rq, &rp, 0);
        call(c, h, 2, 0, NULL, 0, &rp, 0);   // the object is gone
    } else if (!strcmp(g_case, "domain_nin")) {
        h = get_service(c, sm, "tsta:u", NULL);
        control(c, h, 0, &rp);
        rq = (ClReq){.cmd = 0, .domain_obj = 1, .domain_nin = 1};
        cl_send(c, h, &rq, &rp, 0);
    } else if (!strcmp(g_case, "domain_kind")) {
        h = get_service(c, sm, "tsta:u", NULL);
        control(c, h, 0, &rp);
        rq = (ClReq){.cmd = 0, .domain_obj = 1, .domain_kind = 7};
        cl_send(c, h, &rq, &rp, 0);
    } else if (!strcmp(g_case, "domain_full")) {
        h = get_service(c, sm, "tstb:u", NULL);
        control(c, h, 0, &rp);
        for (int i = 0; i < 32; i++) open_children(c, h, 1, 2, &rp);   // 62 fit, 64 do not
    } else if (!strcmp(g_case, "inline_receive")) {
        h = get_service(c, sm, "tsta:u", NULL);
        uint32_t *m = (uint32_t *)(uintptr_t)c->tpidrro;
        memset(m, 0, 0x100);
        m[0] = 4;
        m[1] = 8 | 1u << 10;   // a receive list in mode 1
        m[4] = 0x49434653u;
        sv(c, 0x21, h, 0, 0, 0);
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}

static void guest_end(Cpu *c) { (void)c; }

// ---- host side
static const uint32_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                    0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                    0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

static void boot(const char *folder) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512ull << 20, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256ull << 20};
    rt_mem_init(&m);
    RtKernelConfig k = {.core_mask = 0x3, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512, .is_application = 1};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++) k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    rt_kernel_init(&k);
    hos_init(folder, NULL, 0);
    for (unsigned i = 0; i < sizeof NAMES / sizeof *NAMES; i++) ipc_service_add(NAMES[i], service_handler);
    static const uint64_t addrs[] = {G(1), G(2), G(3), G(4)};
    static const GuestFn fns[] = {g_basic, g_domain, g_fatal, guest_end};
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, addrs, fns, 4, TEST_END};
    rt_dispatch_init(mods, 1);
    Cpu h;
    memset(&h, 0, sizeof h);
    h.x[1] = 16ull << 20;
    rt_svc(&h, 0x01, 0);   // SetHeapSize, from the host before the run
    CHECK(h.x[0] == 0);
    HEAP = h.x[1];
}

static char *read_census(const char *folder) {
    char path[1024];
    snprintf(path, sizeof path, "%s/census.tsv", folder);
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    static char buf[1 << 20];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: ipc_test CASE DIR\n");
        return 2;
    }
    g_case = argv[1];
    const char *dir = argv[2];
    if (!strcmp(g_case, "basic") || !strcmp(g_case, "domain")) {
        boot(dir);
        fprintf(stderr, "heap at %#llx\n", (unsigned long long)HEAP);
        rt_kernel_run(!strcmp(g_case, "basic") ? G(1) : G(2), 0);
        hos_census_write("the case ended");
        const char *t = read_census(dir);
        CHECK(strstr(t, "\nservice\ttsta:u\timplemented\t"));
        CHECK(strstr(t, "\ncommand\ttsta:u cmif 0\timplemented\t"));
        if (!strcmp(g_case, "basic")) CHECK(strstr(t, "\ncommand\ttstd:u tipc 0\timplemented\t"));
        CHECK(strstr(t, "\ncommand\ttchild cmif 0\timplemented\t"));
        CHECK(strstr(t, "\nstop\texit 0\timplemented\t1\t\tthe case ended\n"));
    } else {
        boot(dir);
        rt_kernel_run(G(3), 0);
    }
    printf("ok %s\n", g_case);
    return 0;
}
