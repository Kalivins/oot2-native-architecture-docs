// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// In-process Horizon IPC (architecture D9): the service SVCs, sessions, HIPC framing with CMIF and TIPC payloads, and
// sm:, the service manager. The SDK's own client code runs unchanged: a request is read from the calling thread's
// message buffer (the first 0x100 bytes of its thread-local region, or the user buffer of 0x22) and answered in place.
//
// D9 overrides every nn:: entry point that reaches a service, so the only port the game should open is sm:, and the
// service sm: hands out for any name is none yet (the audio services wait for M2): GetServiceHandle of a name is the
// named fatal "unimplemented service", with the guest backtrace that names the SDK function whose path reached it. An
// unknown port, command or message type stops the same way.
//
// Adapted from oot-pc-recomp runtime/ipc.c at 8e09aaa (GPL-3.0-or-later repository; the file's header: adapted from
// MacNativeGames games/totk/runtime/ipc.c at b75be8c, MIT, Copyright (c) 2026 MacNativeGames contributors; message
// framing per libnx sf/hipc.h, cmif.h, tipc.h, ISC, read for the layout only): the HIPC header and descriptor
// decoding, the CMIF and TIPC payloads, the control commands, sm:'s two commands, the transfer memory record. Changed
// here: sessions are kernel host-object handles (rt_kernel_handle_new), closed by CloseHandle; no service is served
// beyond sm:, and CMIF domains are a named fatal until one is; every request is counted for the census.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hos.h"
#include "plat.h"

#define MSG_BYTES 0x100
#define MAX_BUF 8
#define CMIF_IN_MAGIC 0x49434653u    // "SFCI"
#define CMIF_OUT_MAGIC 0x4f434653u   // "SFCO"
#define POINTER_BUFFER_SIZE 0x8000

// Kernel results (module 1): Switchbrew, "Error codes".
#define R_INVALID_SIZE 0xCA01u
#define R_INVALID_ADDRESS 0xCC01u
#define R_OUT_OF_HANDLES 0xD201u
#define R_INVALID_HANDLE 0xE401u
#define R_INVALID_COMBINATION 0xE801u

enum { T_CLOSE = 2, T_REQUEST = 4, T_CONTROL = 5, T_REQUEST_CTX = 6, T_CONTROL_CTX = 7, T_TIPC_CLOSE = 15,
       T_TIPC_BASE = 16 };

typedef struct Request {
    unsigned type, cmd;
    unsigned nx, na, nb, nw, nc, ncopy, nmove, has_pid;
    uint32_t copy[8], move[8];
    const uint8_t *data;   // the command's arguments (after the CMIF header)
    uint32_t data_size;
} Request;

typedef struct Response {
    uint32_t result;
    uint32_t move[8];
    unsigned nmove;
    uint8_t data[0x80];
    uint32_t data_size;
} Response;

typedef struct Session Session;
typedef void (*Handler)(Session *s, Cpu *c, const Request *rq, Response *rs);
struct Session {
    const char *name;
    Handler handler;
    int refs;   // one per handle, one per request in flight
};

typedef struct TransferMemory { uint64_t addr, size; uint32_t perm; } TransferMemory;

// ---- sessions
static void session_release(Session *s) {
    if (__atomic_sub_fetch(&s->refs, 1, __ATOMIC_ACQ_REL) == 0) free(s);
}

static void session_closed(void *obj) { session_release(obj); }   // CloseHandle, outside the kernel lock

static uint32_t session_handle(Session *s) {   // takes one reference for the handle
    __atomic_add_fetch(&s->refs, 1, __ATOMIC_ACQ_REL);
    uint32_t h = rt_kernel_handle_new(s, session_closed);
    if (!h) session_release(s);
    return h;
}

static uint32_t session_new(const char *name, Handler handler) {
    Session *s = calloc(1, sizeof *s);
    if (!s) rt_fatal("ipc: session allocation");
    *s = (Session){name, handler, 0};
    return session_handle(s);
}

static Session *session_acquire(uint32_t h) {   // NULL for a handle that is no session
    Session *s = rt_kernel_handle_get(h, session_closed);
    if (s) __atomic_add_fetch(&s->refs, 1, __ATOMIC_ACQ_REL);
    return s;
}

static void transfer_memory_closed(void *obj) { free(obj); }

// ---- arguments and answers
static const void *arg(Cpu *c, const Session *s, const Request *rq, uint32_t off, uint32_t n) {
    if (off + n > rq->data_size)
        hos_gap(c, "ipc", "%s command %u: argument bytes %u+%u beyond the %u the request carries", s->name, rq->cmd,
                off, n, rq->data_size);
    return rq->data + off;
}

static void out(Response *rs, const void *p, uint32_t n) {
    if (rs->data_size + n > sizeof rs->data) rt_fatal("ipc: response larger than %u bytes", (unsigned)sizeof rs->data);
    memcpy(rs->data + rs->data_size, p, n);
    rs->data_size += n;
}

static const char *framing(const Request *rq) {
    return rq->type >= T_TIPC_BASE ? "tipc" : rq->type == T_CONTROL || rq->type == T_CONTROL_CTX ? "control" : "cmif";
}

__attribute__((noreturn)) static void unknown_command(Cpu *c, const Session *s, const Request *rq) {
    hos_census_command(s->name, framing(rq), rq->cmd, "named-fatal", c);
    hos_gap(c, "service command", "%s %s command %u (%u bytes of arguments, buffers x%u a%u b%u w%u c%u, handles copy %u "
            "move %u)", s->name, framing(rq), rq->cmd, rq->data_size, rq->nx, rq->na, rq->nb, rq->nw, rq->nc, rq->ncopy,
            rq->nmove);
}

// ---- HIPC framing
static void parse(Cpu *c, const uint32_t *m, Request *rq, const uint32_t **words, unsigned *nwords) {
    memset(rq, 0, sizeof *rq);
    rq->type = m[0] & 0xffff;
    rq->nx = (m[0] >> 16) & 0xf;
    rq->na = (m[0] >> 20) & 0xf;
    rq->nb = (m[0] >> 24) & 0xf;
    rq->nw = (m[0] >> 28) & 0xf;
    *nwords = m[1] & 0x3ff;
    unsigned recv_mode = (m[1] >> 10) & 0xf, special = m[1] >> 31;
    if (rq->nx > MAX_BUF || rq->na > MAX_BUF || rq->nb > MAX_BUF || rq->nw > MAX_BUF)
        hos_gap(c, "ipc", "a request with %u, %u, %u and %u buffer descriptors", rq->nx, rq->na, rq->nb, rq->nw);
    const uint32_t *p = m + 2;
    if (special) {
        uint32_t h = *p++;
        rq->has_pid = h & 1;
        rq->ncopy = (h >> 1) & 0xf;
        rq->nmove = (h >> 5) & 0xf;
        if (rq->ncopy > 8 || rq->nmove > 8)
            hos_gap(c, "ipc", "a request with %u copied and %u moved handles", rq->ncopy, rq->nmove);
        if (rq->has_pid) p += 2;
        for (unsigned i = 0; i < rq->ncopy; i++) rq->copy[i] = *p++;
        for (unsigned i = 0; i < rq->nmove; i++) rq->move[i] = *p++;
    }
    p += 2 * rq->nx + 3 * (rq->na + rq->nb + rq->nw);
    *words = p;
    rq->nc = recv_mode == 2 ? 1 : recv_mode > 2 ? recv_mode - 2 : 0;
    if ((const uint8_t *)(p + *nwords + 2 * rq->nc) - (const uint8_t *)m > MSG_BYTES)
        hos_gap(c, "ipc", "a request longer than the %#x-byte message buffer", MSG_BYTES);
}

static void respond(Cpu *c, uint32_t *m, int tipc, const Response *rs) {
    uint32_t o[MSG_BYTES / 4];
    memset(o, 0, sizeof o);
    uint32_t *p = o + 2;
    if (rs->nmove) {
        *p++ = rs->nmove << 5;
        for (unsigned i = 0; i < rs->nmove; i++) *p++ = rs->move[i];
    }
    uint8_t *words = (uint8_t *)p;
    // TIPC: the result, then the data, right after the handles. CMIF: the SFCO header 16-byte aligned in the message,
    // then the data; the word count keeps libnx's alignment slack.
    uint8_t *payload = tipc ? words : (uint8_t *)o + (((words - (uint8_t *)o) + 15) & ~15);
    unsigned header = tipc ? 4 : 16;
    unsigned used = (unsigned)(payload - words) + header + rs->data_size;
    unsigned n = tipc ? (used + 3) / 4 : (used + 16 + 3) / 4;
    if ((size_t)(words - (uint8_t *)o) + 4 * n > MSG_BYTES)
        hos_gap(c, "ipc", "a response longer than the %#x-byte message buffer", MSG_BYTES);
    if (tipc) {
        memcpy(payload, &rs->result, 4);
    } else {
        uint32_t h[4] = {CMIF_OUT_MAGIC, 0, rs->result, 0};
        memcpy(payload, h, 16);
    }
    memcpy(payload + header, rs->data, rs->data_size);
    o[1] = (n & 0x3ff) | (rs->nmove ? 1u << 31 : 0);
    memcpy(m, o, MSG_BYTES);
}

// CMIF control requests on any session.
static void control(Session *s, Cpu *c, const Request *rq, Response *rs) {
    switch (rq->cmd) {
    case 0:   // ConvertCurrentObjectToDomain
        hos_census_command(s->name, "control", rq->cmd, "named-fatal", c);
        hos_gap(c, "service command", "%s control 0 ConvertCurrentObjectToDomain: CMIF domains come with the first "
                "served service that uses them (audio, M2)", s->name);
    case 2: case 4: {   // CloneCurrentObject, CloneCurrentObjectEx: a second handle to the same session
        uint32_t h = session_handle(s);
        if (!h) rs->result = R_OUT_OF_HANDLES;
        else rs->move[rs->nmove++] = h;
        return;
    }
    case 3: {   // QueryPointerBufferSize
        uint16_t size = POINTER_BUFFER_SIZE;
        out(rs, &size, 2);
        return;
    }
    default: unknown_command(c, s, rq);
    }
}

// ---- sm:, as the SDK's client speaks it (TIPC since 12.0.0; CMIF answered the same way)
static void sm_handler(Session *s, Cpu *c, const Request *rq, Response *rs) {
    switch (rq->cmd) {
    case 0: return;   // RegisterClient (the client's process id, sent in the special header)
    case 1: {         // GetServiceHandle(u64 name) -> moved session handle
        char name[9] = {0};
        memcpy(name, arg(c, s, rq, 0, 8), 8);
        if (rt_census_on()) {   // census mode: list the gap once per name and refuse the service, as sm: refuses an unknown one
            static char seen[32][sizeof name];
            static int nseen;
            static PlatMutex seen_lock = PLAT_MUTEX_INIT;   // guest threads reach sm: at once
            int fresh = 1;
            plat_mutex_lock(&seen_lock);
            for (int i = 0; i < nseen && fresh; i++) fresh = strcmp(seen[i], name) != 0;
            if (fresh && nseen < 32) memcpy(seen[nseen++], name, sizeof name);
            else fresh = 0;
            plat_mutex_unlock(&seen_lock);
            if (fresh) {
                char at[96];
                rt_log("unserved service: sm: GetServiceHandle(\"%s\") refused with 0x415, at %s", name,
                       rt_guest_name(c->x[30], at, sizeof at));
            }
            hos_census_service(name, "census-gap", c);
            rs->result = 0x415;   // sm: ServiceNotRegistered
            return;
        }
        hos_census_service(name, "named-fatal", c);
        hos_gap(c, "service", "sm: GetServiceHandle(\"%s\"): no service of that name is served; D9 overrides the nn:: "
                "entry points that reach it, and this path through the SDK is not overridden", name);
    }
    default: unknown_command(c, s, rq);
    }
}

static void dispatch(Session *s, Cpu *c, uint32_t *m) {
    Request rq;
    const uint32_t *w;
    unsigned nwords;
    parse(c, m, &rq, &w, &nwords);
    if (rq.type == T_CLOSE || rq.type == T_TIPC_CLOSE) return;   // the client closes its handle next
    const uint8_t *bytes = (const uint8_t *)w, *end = bytes + 4 * nwords;
    int tipc = rq.type >= T_TIPC_BASE;
    if (tipc) {
        rq.cmd = rq.type - T_TIPC_BASE;
        rq.data = bytes;
        rq.data_size = (uint32_t)(end - bytes);
    } else if (rq.type == T_REQUEST || rq.type == T_REQUEST_CTX || rq.type == T_CONTROL || rq.type == T_CONTROL_CTX) {
        const uint8_t *cmif = (const uint8_t *)m + (((bytes - (const uint8_t *)m) + 15) & ~15);
        uint32_t h[4];
        if (cmif + 16 > end) hos_gap(c, "ipc", "%s: a CMIF request with no room for its header", s->name);
        memcpy(h, cmif, 16);
        if (h[0] != CMIF_IN_MAGIC) hos_gap(c, "ipc", "%s: CMIF request magic %#x", s->name, h[0]);
        rq.cmd = h[2];
        rq.data = cmif + 16;
        rq.data_size = (uint32_t)(end - cmif - 16);
    } else {
        hos_gap(c, "ipc", "%s: HIPC message type %u", s->name, rq.type);
    }
    Response rs;
    memset(&rs, 0, sizeof rs);
    if (rq.type == T_CONTROL || rq.type == T_CONTROL_CTX) control(s, c, &rq, &rs);
    else s->handler(s, c, &rq, &rs);
    hos_census_command(s->name, framing(&rq), rq.cmd, "implemented", c);
    respond(c, m, tipc, &rs);
}

// ---- the service SVCs
// 0x1F ConnectToNamedPort(x1 name -> w1 handle): sm: only.
static void svc_connect_to_named_port(Cpu *c, uint64_t pc) {
    char name[16];
    hos_str(c->x[1], name, 13);
    if (strcmp(name, "sm:")) {
        c->pc = pc;
        hos_gap(c, "port", "ConnectToNamedPort(\"%s\"): only sm: is served", name);
    }
    uint32_t h = session_new("sm:", sm_handler);
    c->x[0] = h ? 0 : R_OUT_OF_HANDLES;
    c->x[1] = h;
}

// 0x21 SendSyncRequest(w0 session) and 0x22 SendSyncRequestWithUserBuffer(x0 buffer, x1 size, w2 session).
static void send_sync(Cpu *c, uint64_t pc, int user_buffer) {
    uint32_t h = (uint32_t)(user_buffer ? c->x[2] : c->x[0]);
    Session *s = session_acquire(h);
    if (!s) {
        c->x[0] = R_INVALID_HANDLE;
        return;
    }
    uint64_t buf = user_buffer ? c->x[0] : c->tpidrro;
    if (user_buffer && (c->x[1] < MSG_BYTES || (buf & 0xfff) || (c->x[1] & 0xfff))) {
        session_release(s);
        c->x[0] = R_INVALID_SIZE;
        return;
    }
    if (!rt_mem_commit_io(buf, MSG_BYTES, 1)) {
        session_release(s);
        c->x[0] = R_INVALID_ADDRESS;
        return;
    }
    uint64_t lr = c->x[30];
    c->pc = pc;   // a gap inside the request names the SVC's call site
    dispatch(s, c, (uint32_t *)(uintptr_t)buf);
    c->x[30] = lr;
    session_release(s);
    c->x[0] = 0;
}

static void svc_send_sync_request(Cpu *c, uint64_t pc) { send_sync(c, pc, 0); }
static void svc_send_sync_request_with_user_buffer(Cpu *c, uint64_t pc) { send_sync(c, pc, 1); }

// 0x15 CreateTransferMemory(x1 address, x2 size, w3 permission -> w1 handle): a record a served service could map.
static void svc_create_transfer_memory(Cpu *c, uint64_t pc) {
    (void)pc;
    uint64_t a = c->x[1], n = c->x[2];
    uint32_t perm = (uint32_t)c->x[3];
    uint32_t r = (a & 0xfff) || a + n < a || a + n > RT_ADDRESS_LIMIT ? R_INVALID_ADDRESS
               : !n || (n & 0xfff) ? R_INVALID_SIZE
               : perm > 3 || perm == 2 ? R_INVALID_COMBINATION
               : 0;
    uint32_t h = 0;
    if (!r) {
        TransferMemory *t = malloc(sizeof *t);
        if (!t) rt_fatal("ipc: transfer memory allocation");
        *t = (TransferMemory){a, n, perm};
        h = rt_kernel_handle_new(t, transfer_memory_closed);
        if (!h) {
            free(t);
            r = R_OUT_OF_HANDLES;
        }
    }
    c->x[0] = r;
    c->x[1] = h;
}

void ipc_register(void) {
    rt_svc_register(0x15, svc_create_transfer_memory);
    rt_svc_register(0x1F, svc_connect_to_named_port);
    rt_svc_register(0x21, svc_send_sync_request);
    rt_svc_register(0x22, svc_send_sync_request_with_user_buffer);
    rt_svc_defer(0x13, "only a served service hands out shared memory, and the services that do (hid, applet) are "
                       "answered by host functions instead (architecture D9)");
    rt_svc_defer(0x14, "only a served service hands out shared memory, and the services that do (hid, applet) are "
                       "answered by host functions instead (architecture D9)");
    rt_svc_defer(0x23, "an asynchronous request goes to a served service, and only sm: is served");
}
