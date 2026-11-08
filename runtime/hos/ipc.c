// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// In-process Horizon IPC (architecture D9): the service SVCs, sessions, HIPC framing with CMIF and TIPC payloads, CMIF
// domains, and sm:, the service manager. The SDK's own client code runs unchanged: a request is read from the calling
// thread's message buffer (the first 0x100 bytes of its thread-local region, or the user buffer of 0x22) and answered
// in place.
//
// D9 overrides every nn:: entry point that reaches a service, so the only port the game opens is sm:, and the services
// sm: hands out are the ones added with ipc_service_add (ipc.h). GetServiceHandle of any other name is the named fatal
// "unimplemented service", with the guest backtrace that names the SDK function whose path reached it. An unknown
// port, command or message type stops the same way.
//
// Adapted from oot-pc-recomp runtime/ipc.c at 8e09aaa (GPL-3.0-or-later repository; the file's header: adapted from
// MacNativeGames games/totk/runtime/ipc.c at b75be8c, MIT, Copyright (c) 2026 MacNativeGames contributors; message
// framing per libnx sf/hipc.h, cmif.h, tipc.h, ISC, read for the layout only): the HIPC header and descriptor
// decoding, the CMIF and TIPC payloads, the control commands, the domain framing, sm:'s two commands, the transfer
// memory record. Changed here: sessions are kernel host-object handles (rt_kernel_handle_new), closed by CloseHandle,
// and hold a state and a close callback; a service's new session is an object the framing turns into a moved handle or
// a domain object id; handlers get the calling Cpu; every request is counted for the census.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ipc.h"
#include "plat.h"

#define CMIF_IN_MAGIC 0x49434653u    // "SFCI"
#define CMIF_OUT_MAGIC 0x4f434653u   // "SFCO"
#define MAX_SERVICES 16

// Kernel results (module 1): Switchbrew, "Error codes".
#define R_INVALID_SIZE 0xCA01u
#define R_INVALID_ADDRESS 0xCC01u
#define R_OUT_OF_HANDLES 0xD201u
#define R_INVALID_HANDLE 0xE401u
#define R_INVALID_COMBINATION 0xE801u

enum { T_CLOSE = 2, T_REQUEST = 4, T_CONTROL = 5, T_REQUEST_CTX = 6, T_CONTROL_CTX = 7, T_TIPC_CLOSE = 15,
       T_TIPC_BASE = 16 };
enum { DOMAIN_SEND = 1, DOMAIN_CLOSE = 2 };   // the domain header's request kinds

typedef struct Service { const char *name; IpcHandler handler; } Service;
static Service g_services[MAX_SERVICES];
static int g_nservices;
static PlatMutex g_table_lock = PLAT_MUTEX_INIT;   // the service table and the slots of every domain; held for table edits only

// ---- sessions
void ipc_session_release(IpcSession *s) {
    if (__atomic_sub_fetch(&s->refs, 1, __ATOMIC_ACQ_REL) != 0) return;
    if (s->objs) {
        for (int i = 2; i < IPC_DOMAIN_OBJECTS; i++)
            if (s->objs[i]) ipc_session_release(s->objs[i]);
        free(s->objs);
    }
    if (s->close) s->close(s->state);
    free(s);
}

static void session_closed(void *obj) { ipc_session_release(obj); }   // CloseHandle, outside the kernel lock

static uint32_t session_handle(IpcSession *s) {   // takes one reference for the handle
    __atomic_add_fetch(&s->refs, 1, __ATOMIC_ACQ_REL);
    uint32_t h = rt_kernel_handle_new(s, session_closed);
    if (!h) ipc_session_release(s);
    return h;
}

IpcSession *ipc_session_new(const char *name, IpcHandler handler, void *state, void (*close)(void *state)) {
    IpcSession *s = calloc(1, sizeof *s);
    if (!s) rt_fatal("ipc: session allocation");
    s->name = name;
    s->handler = handler;
    s->state = state;
    s->close = close;
    s->refs = 1;
    return s;
}

void ipc_answer_session(IpcResponse *rs, IpcSession *s) {
    if (rs->nobjs == sizeof rs->objs / sizeof *rs->objs) rt_fatal("ipc: %s: an answer with more than 8 sessions", s->name);
    rs->objs[rs->nobjs++] = s;
}

static IpcSession *session_acquire(uint32_t h) {   // NULL for a handle that is no session
    IpcSession *s = rt_kernel_handle_get(h, session_closed);
    if (s) __atomic_add_fetch(&s->refs, 1, __ATOMIC_ACQ_REL);
    return s;
}

static void transfer_memory_closed(void *obj) { free(obj); }

int ipc_transfer_memory(uint32_t handle, IpcTransferMemory *out) {
    const IpcTransferMemory *t = rt_kernel_handle_get(handle, transfer_memory_closed);
    if (!t) return 0;
    *out = *t;
    return 1;
}

void ipc_service_add(const char *name, IpcHandler handler) {
    if (strlen(name) > 8) rt_fatal("ipc: the service name \"%s\" is longer than the 8 bytes sm: carries", name);
    plat_mutex_lock(&g_table_lock);
    int i = 0;
    while (i < g_nservices && strcmp(g_services[i].name, name)) i++;
    if (i == MAX_SERVICES) {
        plat_mutex_unlock(&g_table_lock);
        rt_fatal("ipc: more than %d services", MAX_SERVICES);
    }
    g_services[i] = (Service){name, handler};
    if (i == g_nservices) g_nservices++;
    plat_mutex_unlock(&g_table_lock);
}

// ---- domains: a session converted by control command 0 holds up to IPC_DOMAIN_OBJECTS objects by id
static uint32_t domain_add(IpcSession *d, IpcSession *child) {   // takes the caller's reference on child
    plat_mutex_lock(&g_table_lock);
    for (uint32_t id = 2; id < IPC_DOMAIN_OBJECTS; id++)
        if (!d->objs[id]) {
            d->objs[id] = child;
            plat_mutex_unlock(&g_table_lock);
            return id;
        }
    plat_mutex_unlock(&g_table_lock);
    rt_fatal("ipc: %s: the domain holds %d objects already", d->name, IPC_DOMAIN_OBJECTS - 2);
}

static void domain_close(IpcSession *d, uint32_t id) {
    IpcSession *child = NULL;
    plat_mutex_lock(&g_table_lock);
    if (id >= 2 && id < IPC_DOMAIN_OBJECTS) child = d->objs[id], d->objs[id] = NULL;
    plat_mutex_unlock(&g_table_lock);
    if (child) ipc_session_release(child);
}

// The object a domain request is for, with a reference (id 1 is the domain's own session); NULL for an unknown id.
static IpcSession *domain_target(IpcSession *d, uint32_t id) {
    IpcSession *t = NULL;
    plat_mutex_lock(&g_table_lock);
    if (id == 1) t = d;
    else if (id > 1 && id < IPC_DOMAIN_OBJECTS) t = d->objs[id];
    if (t) __atomic_add_fetch(&t->refs, 1, __ATOMIC_ACQ_REL);
    plat_mutex_unlock(&g_table_lock);
    return t;
}

// ---- arguments and answers
const void *ipc_arg(Cpu *c, const IpcSession *s, const IpcRequest *rq, uint32_t off, uint32_t n) {
    if (off + n > rq->data_size)
        hos_gap(c, "ipc", "%s command %u: argument bytes %u+%u beyond the %u the request carries", s->name, rq->cmd,
                off, n, rq->data_size);
    return rq->data + off;
}

void ipc_out(IpcResponse *rs, const void *p, uint32_t n) {
    if (rs->data_size + n > sizeof rs->data) rt_fatal("ipc: response larger than %u bytes", (unsigned)sizeof rs->data);
    memcpy(rs->data + rs->data_size, p, n);
    rs->data_size += n;
}

static const char *framing(const IpcRequest *rq) {
    return rq->type >= T_TIPC_BASE ? "tipc" : rq->type == T_CONTROL || rq->type == T_CONTROL_CTX ? "control" : "cmif";
}

void ipc_unknown_command(Cpu *c, const IpcSession *s, const IpcRequest *rq) {
    hos_census_command(s->name, framing(rq), rq->cmd, "named-fatal", c);
    hos_gap(c, "service command", "%s %s command %u (%u bytes of arguments, buffers x%u a%u b%u w%u c%u, handles copy %u "
            "move %u)", s->name, framing(rq), rq->cmd, rq->data_size, rq->nx, rq->na, rq->nb, rq->nw, rq->nc, rq->ncopy,
            rq->nmove);
}

int ipc_read(const IpcBuffer *b, uint64_t off, void *dst, uint64_t n) {
    if (off > b->size || n > b->size - off) return 0;
    return hos_in(dst, b->addr + off, n);
}

int ipc_write(const IpcBuffer *b, uint64_t off, const void *src, uint64_t n) {
    if (off > b->size || n > b->size - off) return 0;
    return hos_out(b->addr + off, src, n);
}

int ipc_answer_pointer(IpcResponse *rs, const IpcRequest *rq, unsigned index, const void *data, uint64_t size) {
    if (index >= rq->nc || rs->nxout == IPC_MAX_BUF || size > 0xffff || !ipc_write(&rq->c[index], 0, data, size)) return 0;
    rs->xout[rs->nxout++] = (IpcBuffer){rq->c[index].addr, size, index};
    return 1;
}

// ---- HIPC framing (libnx sf/hipc.h): descriptors are read for their layout only
// X: {index:6, address[36..41]:6, address[32..35]:4, size:16}, address[0..31]
static uint64_t static_addr(const uint32_t *d) {
    return d[1] | ((uint64_t)((d[0] >> 12) & 0xf) << 32) | ((uint64_t)((d[0] >> 6) & 0x3f) << 36);
}
// A, B, W: size[0..31], address[0..31], {mode:2, address[36..57]:22, size[32..35]:4, address[32..35]:4}
static uint64_t buffer_addr(const uint32_t *d) {
    return d[1] | ((uint64_t)((d[2] >> 28) & 0xf) << 32) | ((uint64_t)((d[2] >> 2) & 0x3fffff) << 36);
}
static uint64_t buffer_size(const uint32_t *d) { return d[0] | ((uint64_t)((d[2] >> 24) & 0xf) << 32); }

static void parse(Cpu *c, const uint32_t *m, IpcRequest *rq, const uint32_t **words, unsigned *nwords) {
    memset(rq, 0, sizeof *rq);
    rq->type = m[0] & 0xffff;
    rq->nx = (m[0] >> 16) & 0xf;
    rq->na = (m[0] >> 20) & 0xf;
    rq->nb = (m[0] >> 24) & 0xf;
    rq->nw = (m[0] >> 28) & 0xf;
    *nwords = m[1] & 0x3ff;
    unsigned recv_mode = (m[1] >> 10) & 0xf, recv_offset = (m[1] >> 20) & 0x7ff, special = m[1] >> 31;
    if (rq->nx > IPC_MAX_BUF || rq->na > IPC_MAX_BUF || rq->nb > IPC_MAX_BUF || rq->nw > IPC_MAX_BUF)
        hos_gap(c, "ipc", "a request with %u, %u, %u and %u buffer descriptors", rq->nx, rq->na, rq->nb, rq->nw);
    const uint32_t *p = m + 2;
    if (special) {
        uint32_t h = *p++;
        rq->has_pid = h & 1;
        rq->ncopy = (h >> 1) & 0xf;
        rq->nmove = (h >> 5) & 0xf;
        if (rq->ncopy > 8 || rq->nmove > 8)
            hos_gap(c, "ipc", "a request with %u copied and %u moved handles", rq->ncopy, rq->nmove);
        if (rq->has_pid) {
            memcpy(&rq->pid, p, 8);
            p += 2;
        }
        for (unsigned i = 0; i < rq->ncopy; i++) rq->copy[i] = *p++;
        for (unsigned i = 0; i < rq->nmove; i++) rq->move[i] = *p++;
    }
    if ((const uint8_t *)(p + 2 * rq->nx + 3 * (rq->na + rq->nb + rq->nw)) - (const uint8_t *)m > IPC_MSG_BYTES)
        hos_gap(c, "ipc", "a request longer than the %#x-byte message buffer", IPC_MSG_BYTES);
    for (unsigned i = 0; i < rq->nx; i++, p += 2) rq->x[i] = (IpcBuffer){static_addr(p), p[0] >> 16, p[0] & 0x3f};
    IpcBuffer *lists[3] = {rq->a, rq->b, rq->w};
    const unsigned counts[3] = {rq->na, rq->nb, rq->nw};
    for (unsigned k = 0; k < 3; k++)
        for (unsigned i = 0; i < counts[k]; i++, p += 3)
            lists[k][i] = (IpcBuffer){buffer_addr(p), buffer_size(p), p[2] & 3};
    *words = p;
    if (recv_mode == 1) hos_gap(c, "ipc", "a receive list in the inline mode (1), which no service of this runtime uses");
    rq->nc = recv_mode == 2 ? 1 : recv_mode > 2 ? recv_mode - 2 : 0;
    // the receive list follows the raw data, at the offset the header gives when it gives one
    const uint32_t *recv = recv_offset ? m + recv_offset : p + *nwords;
    if ((const uint8_t *)(recv + 2 * rq->nc) - (const uint8_t *)m > IPC_MSG_BYTES ||
        (const uint8_t *)(p + *nwords) - (const uint8_t *)m > IPC_MSG_BYTES)
        hos_gap(c, "ipc", "a request longer than the %#x-byte message buffer", IPC_MSG_BYTES);
    for (unsigned i = 0; i < rq->nc; i++, recv += 2)
        rq->c[i] = (IpcBuffer){recv[0] | ((uint64_t)(recv[1] & 0xffff) << 32), recv[1] >> 16, 0};
}

// domain: the answer carries the 16-byte domain header (the object count) before the CMIF header, and the object ids
// of the sessions made for the client after the data.
static void respond(Cpu *c, uint32_t *m, int tipc, const IpcResponse *rs, int domain, const uint32_t *ids,
                    unsigned nids) {
    uint32_t o[IPC_MSG_BYTES / 4];
    memset(o, 0, sizeof o);
    unsigned special = rs->ncopy || rs->nmove;
    uint32_t *p = o + 2;
    if (special) {
        *p++ = (rs->ncopy << 1) | (rs->nmove << 5);
        for (unsigned i = 0; i < rs->ncopy; i++) *p++ = rs->copy[i];
        for (unsigned i = 0; i < rs->nmove; i++) *p++ = rs->move[i];
    }
    for (unsigned i = 0; i < rs->nxout; i++, p += 2) {
        uint64_t a = rs->xout[i].addr;
        p[0] = (rs->xout[i].attr & 0x3f) | (uint32_t)(((a >> 36) & 0x3f) << 6) | (uint32_t)(((a >> 32) & 0xf) << 12) |
               (uint32_t)(rs->xout[i].size << 16);
        p[1] = (uint32_t)a;
    }
    uint8_t *words = (uint8_t *)p;
    // TIPC: the result, then the data, right after the handles. CMIF: the SFCO header 16-byte aligned in the message,
    // then the data; the word count keeps libnx's alignment slack.
    uint8_t *payload = tipc ? words : (uint8_t *)o + (((words - (uint8_t *)o) + 15) & ~15);
    unsigned header = tipc ? 4 : domain ? 32 : 16;
    unsigned used = (unsigned)(payload - words) + header + rs->data_size + (domain ? 4 * nids : 0);
    unsigned n = tipc ? (used + 3) / 4 : (used + 16 + 3) / 4;
    if ((size_t)(words - (uint8_t *)o) + 4 * n > IPC_MSG_BYTES)
        hos_gap(c, "ipc", "a response longer than the %#x-byte message buffer", IPC_MSG_BYTES);
    if (tipc) {
        memcpy(payload, &rs->result, 4);
    } else {
        uint32_t h[4] = {CMIF_OUT_MAGIC, 0, rs->result, 0};
        if (domain) {
            uint32_t d[4] = {nids, 0, 0, 0};
            memcpy(payload, d, 16);
        }
        memcpy(payload + (domain ? 16 : 0), h, 16);
    }
    memcpy(payload + header, rs->data, rs->data_size);
    if (domain && nids) memcpy(payload + header + rs->data_size, ids, 4 * nids);
    o[0] = (rs->nxout & 0xf) << 16;
    o[1] = (n & 0x3ff) | (special ? 1u << 31 : 0);
    memcpy(m, o, IPC_MSG_BYTES);
}

// CMIF control requests on any session.
static void control(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    switch (rq->cmd) {
    case 0: {   // ConvertCurrentObjectToDomain: the session becomes object 1 of a domain, and its requests carry a domain header
        plat_mutex_lock(&g_table_lock);
        int fresh = !s->objs;
        if (fresh) s->objs = calloc(IPC_DOMAIN_OBJECTS, sizeof *s->objs);
        plat_mutex_unlock(&g_table_lock);
        if (!s->objs) rt_fatal("ipc: domain allocation");
        if (!fresh) {
            hos_census_command(s->name, "control", rq->cmd, "named-fatal", c);
            hos_gap(c, "service command", "%s control 0 ConvertCurrentObjectToDomain: the session is a domain already",
                    s->name);
        }
        uint32_t id = 1;
        ipc_out(rs, &id, 4);
        rt_log("ipc: %s: converted to a CMIF domain", s->name);   // whether a game needs domains is read from this line
        return;
    }
    case 2: case 4: {   // CloneCurrentObject, CloneCurrentObjectEx: a second handle to the same session
        uint32_t h = session_handle(s);
        if (!h) rs->result = R_OUT_OF_HANDLES;
        else rs->move[rs->nmove++] = h;
        return;
    }
    case 3: {   // QueryPointerBufferSize
        uint16_t size = IPC_POINTER_BUFFER_SIZE;
        ipc_out(rs, &size, 2);
        return;
    }
    default: ipc_unknown_command(c, s, rq);
    }
}

// ---- sm:, as the SDK's client speaks it (TIPC since 12.0.0; CMIF answered the same way)
static void sm_handler(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    switch (rq->cmd) {
    case 0: return;   // RegisterClient (the client's process id, sent in the special header)
    case 1: {         // GetServiceHandle(u64 name) -> moved session handle
        char name[9] = {0};
        memcpy(name, ipc_arg(c, s, rq, 0, 8), 8);
        Service svc = {0};
        plat_mutex_lock(&g_table_lock);
        for (int i = 0; i < g_nservices; i++)
            if (!strcmp(g_services[i].name, name)) svc = g_services[i];
        plat_mutex_unlock(&g_table_lock);
        if (svc.name) {
            hos_census_service(name, "implemented", c);
            ipc_answer_session(rs, ipc_session_new(svc.name, svc.handler, NULL, NULL));
            return;
        }
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
    default: ipc_unknown_command(c, s, rq);
    }
}

static void dispatch(IpcSession *s, Cpu *c, uint32_t *m) {
    IpcRequest rq;
    const uint32_t *w;
    unsigned nwords;
    parse(c, m, &rq, &w, &nwords);
    if (rq.type == T_CLOSE || rq.type == T_TIPC_CLOSE) return;   // the client closes its handle next
    const uint8_t *bytes = (const uint8_t *)w, *end = bytes + 4 * nwords;
    int tipc = rq.type >= T_TIPC_BASE, domain_msg = 0;
    int control_msg = rq.type == T_CONTROL || rq.type == T_CONTROL_CTX;
    IpcSession *target = s;   // the object the request is for: s, or one of its domain's objects
    if (tipc) {
        rq.cmd = rq.type - T_TIPC_BASE;
        rq.data = bytes;
        rq.data_size = (uint32_t)(end - bytes);
    } else if (rq.type == T_REQUEST || rq.type == T_REQUEST_CTX || control_msg) {
        const uint8_t *cmif = (const uint8_t *)m + (((bytes - (const uint8_t *)m) + 15) & ~15);
        if (s->objs && !control_msg) {
            // A domain request: DomainInHeader {u8 kind, u8 objects in, u16 data size, u32 object id, u32 pad, u32
            // token}, then the CMIF header and the arguments (the data size counts both), then the object ids sent.
            if (cmif + 16 > end) hos_gap(c, "ipc", "%s: a domain request with no room for its header", s->name);
            uint8_t kind = cmif[0], nin = cmif[1];
            uint16_t dsize;
            uint32_t oid;
            memcpy(&dsize, cmif + 2, 2);
            memcpy(&oid, cmif + 4, 4);
            if (kind == DOMAIN_CLOSE) {
                domain_close(s, oid);
                IpcResponse ok;
                memset(&ok, 0, sizeof ok);
                respond(c, m, 0, &ok, 0, NULL, 0);
                return;
            }
            if (kind != DOMAIN_SEND || nin || dsize < 16 || cmif + 16 + dsize > end)
                hos_gap(c, "ipc", "%s: domain request kind %u with %u objects in and %u bytes of data", s->name, kind, nin,
                        dsize);
            target = domain_target(s, oid);
            if (!target) hos_gap(c, "ipc", "%s: domain object %u does not exist", s->name, oid);
            domain_msg = 1;
            cmif += 16;
            rq.data_size = dsize - 16;
        } else {
            rq.data_size = cmif + 16 <= end ? (uint32_t)(end - cmif - 16) : 0;
        }
        if (cmif + 16 > end) hos_gap(c, "ipc", "%s: a CMIF request with no room for its header", s->name);
        uint32_t h[4];
        memcpy(h, cmif, 16);
        if (h[0] != CMIF_IN_MAGIC) hos_gap(c, "ipc", "%s: CMIF request magic %#x", s->name, h[0]);
        rq.cmd = h[2];
        rq.data = cmif + 16;
    } else {
        hos_gap(c, "ipc", "%s: HIPC message type %u", s->name, rq.type);
    }
    IpcResponse rs;
    memset(&rs, 0, sizeof rs);
    if (control_msg) control(s, c, &rq, &rs);
    else target->handler(target, c, &rq, &rs);
    hos_census_command(target->name, framing(&rq), rq.cmd, "implemented", c);
    // The sessions the service made: moved handles, or object ids when the request came through a domain.
    uint32_t ids[8];
    unsigned nids = 0;
    for (unsigned i = 0; i < rs.nobjs; i++) {
        if (domain_msg) {
            ids[nids++] = domain_add(s, rs.objs[i]);
            continue;
        }
        uint32_t h = session_handle(rs.objs[i]);
        ipc_session_release(rs.objs[i]);
        if (!h) rs.result = R_OUT_OF_HANDLES;
        else if (rs.nmove < 8) rs.move[rs.nmove++] = h;
        else rt_fatal("ipc: an answer with more than 8 moved handles");
    }
    respond(c, m, tipc, &rs, domain_msg, ids, nids);
    if (domain_msg) ipc_session_release(target);   // the reference domain_target took (object 1 is s itself)
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
    IpcSession *s = ipc_session_new("sm:", sm_handler, NULL, NULL);
    uint32_t h = session_handle(s);
    ipc_session_release(s);
    c->x[0] = h ? 0 : R_OUT_OF_HANDLES;
    c->x[1] = h;
}

// 0x21 SendSyncRequest(w0 session) and 0x22 SendSyncRequestWithUserBuffer(x0 buffer, x1 size, w2 session).
static void send_sync(Cpu *c, uint64_t pc, int user_buffer) {
    uint32_t h = (uint32_t)(user_buffer ? c->x[2] : c->x[0]);
    IpcSession *s = session_acquire(h);
    if (!s) {
        c->x[0] = R_INVALID_HANDLE;
        return;
    }
    uint64_t buf = user_buffer ? c->x[0] : c->tpidrro;
    if (user_buffer && (c->x[1] < IPC_MSG_BYTES || (buf & 0xfff) || (c->x[1] & 0xfff))) {
        ipc_session_release(s);
        c->x[0] = R_INVALID_SIZE;
        return;
    }
    if (!rt_mem_commit_io(buf, IPC_MSG_BYTES, 1)) {
        ipc_session_release(s);
        c->x[0] = R_INVALID_ADDRESS;
        return;
    }
    uint64_t lr = c->x[30];
    c->pc = pc;   // a gap inside the request names the SVC's call site
    dispatch(s, c, (uint32_t *)(uintptr_t)buf);
    c->x[30] = lr;
    ipc_session_release(s);
    c->x[0] = 0;
}

static void svc_send_sync_request(Cpu *c, uint64_t pc) { send_sync(c, pc, 0); }
static void svc_send_sync_request_with_user_buffer(Cpu *c, uint64_t pc) { send_sync(c, pc, 1); }

// 0x15 CreateTransferMemory(x1 address, x2 size, w3 permission -> w1 handle): a record a served service can look up
// (ipc_transfer_memory).
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
        IpcTransferMemory *t = malloc(sizeof *t);
        if (!t) rt_fatal("ipc: transfer memory allocation");
        *t = (IpcTransferMemory){a, n, perm};
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
