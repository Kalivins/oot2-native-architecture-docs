// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// A small IPC client for the tests of the services: it builds a HIPC request (CMIF or TIPC, optionally through a CMIF
// domain) with buffers and handles in a guest thread's message buffer, sends it with SendSyncRequest (or with a user
// buffer), and reads the answer. The layouts are written out here as numbers, from the public description of the
// message format, and not taken from the runtime's own structures, so that a fault in the runtime's reader cannot make
// the test agree with itself. The including file defines CL_PC, the pc a synthetic SVC is issued from.
#pragma once
#include <stdint.h>
#include <string.h>

#include "oot2/kernel.h"

typedef struct ClBuf { uint64_t addr, size; uint32_t attr; } ClBuf;

typedef struct ClReq {
    int tipc;                       // TIPC framing (the command is the message type minus 16)
    int control;                    // a CMIF control request (type 5)
    uint32_t cmd;
    uint32_t domain_obj;            // not 0: a CMIF domain request to that object id
    unsigned domain_kind;           // 0 means send (1), 2 is close
    unsigned domain_nin;            // object ids claimed as sent with the request (a fault when not 0)
    const void *data;
    uint32_t size;
    int pid;                        // the process id in the special header
    uint32_t copy[8], move[8];
    unsigned ncopy, nmove;
    ClBuf x[8], a[8], b[8], w[8], c[8];
    unsigned nx, na, nb, nw, nc;
    int recv_offset_zero;           // leave the receive list offset of the header at 0
    uint64_t user_buffer;           // not 0: SendSyncRequestWithUserBuffer on a 0x1000-byte page there
} ClReq;

typedef struct ClRep {
    uint32_t result;
    unsigned ncopy, nmove, nx;
    uint32_t copy[8], move[8];
    ClBuf x[8];                     // answer pointer descriptors: addr, size, index in attr
    const uint8_t *data;            // the first byte of the answer's data
    uint32_t ndata;                 // bytes from there to the end of the message's words (the slack included)
    unsigned nobj;                  // a domain answer: the number of object ids
    const uint32_t *obj;            // and where they are, after the data of the size the caller expects
    unsigned words;                 // the message's word count as the answer states it
} ClRep;

static inline uint32_t cl_u32(const void *p) { uint32_t v; memcpy(&v, p, 4); return v; }

// Send rq to session; fills rp. Returns the SVC's own result (0, or a kernel error for a bad handle or buffer).
static inline uint32_t cl_send(Cpu *c, uint32_t session, const ClReq *rq, ClRep *rp, uint32_t expect_data) {
    uint32_t *m = rq->user_buffer ? (uint32_t *)(uintptr_t)rq->user_buffer : (uint32_t *)(uintptr_t)c->tpidrro;
    memset(m, 0, rq->user_buffer ? 0x1000 : 0x100);
    uint32_t *p = m + 2;
    unsigned special = rq->pid || rq->ncopy || rq->nmove;
    if (special) {
        *p++ = (rq->pid ? 1u : 0) | rq->ncopy << 1 | rq->nmove << 5;
        if (rq->pid) *p++ = 0x1234, *p++ = 0;
        for (unsigned i = 0; i < rq->ncopy; i++) *p++ = rq->copy[i];
        for (unsigned i = 0; i < rq->nmove; i++) *p++ = rq->move[i];
    }
    for (unsigned i = 0; i < rq->nx; i++, p += 2) {
        uint64_t a = rq->x[i].addr;
        p[0] = (uint32_t)i | (uint32_t)(((a >> 36) & 0x3f) << 6) | (uint32_t)(((a >> 32) & 0xf) << 12) |
               (uint32_t)(rq->x[i].size << 16);
        p[1] = (uint32_t)a;
    }
    const ClBuf *lists[3] = {rq->a, rq->b, rq->w};
    const unsigned counts[3] = {rq->na, rq->nb, rq->nw};
    for (unsigned k = 0; k < 3; k++)
        for (unsigned i = 0; i < counts[k]; i++, p += 3) {
            uint64_t a = lists[k][i].addr, n = lists[k][i].size;
            p[0] = (uint32_t)n;
            p[1] = (uint32_t)a;
            p[2] = (lists[k][i].attr & 3) | (uint32_t)(((a >> 36) & 0x3fffff) << 2) | (uint32_t)(((n >> 32) & 0xf) << 24) |
                   (uint32_t)(((a >> 32) & 0xf) << 28);
        }
    uint32_t *raw = p;
    unsigned nwords;
    uint32_t type;
    if (rq->tipc) {
        type = 16 + rq->cmd;
        if (rq->size) memcpy(raw, rq->data, rq->size);
        nwords = (rq->size + 3) / 4;
    } else {
        type = rq->control ? 5 : 4;
        uint8_t *payload = (uint8_t *)m + ((((uint8_t *)raw - (uint8_t *)m) + 15) & ~15);
        uint8_t *at = payload;
        uint32_t cmif[4] = {0x49434653u, 0, rq->cmd, 0};
        if (rq->domain_obj) {   // DomainInHeader: kind, objects in, data size (the CMIF header and the arguments), id
            uint8_t kind = rq->domain_kind ? (uint8_t)rq->domain_kind : 1, nin = (uint8_t)rq->domain_nin;
            uint16_t dsize = (uint16_t)(16 + rq->size);
            at[0] = kind, at[1] = nin;
            memcpy(at + 2, &dsize, 2);
            memcpy(at + 4, &rq->domain_obj, 4);
            at += 16;
        }
        memcpy(at, cmif, 16);
        if (rq->size) memcpy(at + 16, rq->data, rq->size);
        at += 16 + rq->size;
        nwords = (unsigned)((at - (uint8_t *)raw + 3) / 4);
    }
    uint32_t *recv = raw + nwords;
    for (unsigned i = 0; i < rq->nc; i++, recv += 2) {
        recv[0] = (uint32_t)rq->c[i].addr;
        recv[1] = (uint32_t)(rq->c[i].addr >> 32) | (uint32_t)(rq->c[i].size << 16);
    }
    unsigned mode = rq->nc == 0 ? 0 : rq->nc == 1 ? 2 : rq->nc + 2;
    unsigned offset = rq->nc && !rq->recv_offset_zero ? (unsigned)(raw + nwords - m) : 0;
    m[0] = type | rq->nx << 16 | rq->na << 20 | rq->nb << 24 | rq->nw << 28;
    m[1] = nwords | mode << 10 | offset << 20 | (special ? 1u << 31 : 0);
    uint32_t r;
    c->x[0] = rq->user_buffer ? rq->user_buffer : session;
    if (rq->user_buffer) c->x[1] = 0x1000, c->x[2] = session;
    rt_svc(c, rq->user_buffer ? 0x22 : 0x21, CL_PC);
    r = (uint32_t)c->x[0];
    memset(rp, 0, sizeof *rp);
    if (r) return r;
    // the answer
    const uint32_t *q = m + 2;
    rp->nx = (m[0] >> 16) & 0xf;
    rp->words = m[1] & 0x3ff;
    if (m[1] >> 31) {
        uint32_t h = *q++;
        rp->ncopy = (h >> 1) & 0xf;
        rp->nmove = (h >> 5) & 0xf;
        for (unsigned i = 0; i < rp->ncopy; i++) rp->copy[i] = *q++;
        for (unsigned i = 0; i < rp->nmove; i++) rp->move[i] = *q++;
    }
    for (unsigned i = 0; i < rp->nx; i++, q += 2) {
        uint64_t a = q[1] | ((uint64_t)((q[0] >> 12) & 0xf) << 32) | ((uint64_t)((q[0] >> 6) & 0x3f) << 36);
        rp->x[i] = (ClBuf){a, q[0] >> 16, q[0] & 0x3f};
    }
    const uint8_t *words = (const uint8_t *)q;
    const uint8_t *end = words + 4 * rp->words;
    if (rq->tipc) {
        rp->result = cl_u32(words);
        rp->data = words + 4;
    } else {
        const uint8_t *pay = (const uint8_t *)m + ((((const uint8_t *)q - (const uint8_t *)m) + 15) & ~15);
        if (rq->domain_obj && rq->domain_kind != 2) {
            rp->nobj = cl_u32(pay);
            pay += 16;
        }
        if (cl_u32(pay) != 0x4f434653u) return 0xffff0001u;   // not an SFCO header
        rp->result = cl_u32(pay + 8);
        rp->data = pay + 16;
        if (rp->nobj) rp->obj = (const uint32_t *)(rp->data + expect_data);
    }
    rp->ndata = (uint32_t)(end - rp->data);
    return 0;
}
