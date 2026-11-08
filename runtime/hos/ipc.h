// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The services' side of the in-process IPC (ipc.c): a request as the service sees it, the answer it builds, sessions
// with state and a close callback, CMIF domains, and the table of names that sm: hands out. A service is a handler
// function; the framing (HIPC headers, descriptors, CMIF and TIPC payloads, domain headers) stays in ipc.c.
//
// A handler runs on the calling guest thread, with that thread's Cpu: a named fatal (hos_gap) names the SDK function
// that sent the request, and a service that needs to know who asked reads rt_kernel_current().
#pragma once
#include "hos.h"

#define IPC_MAX_BUF 8            // descriptors of one kind in a request or an answer
#define IPC_MSG_BYTES 0x100      // the message buffer: the first 0x100 bytes of the thread's local region
#define IPC_POINTER_BUFFER_SIZE 0x8000
#define IPC_DOMAIN_OBJECTS 64    // object ids of one domain (0 is unused, 1 is the converted session itself)

// A buffer the client described: A, B and W (map-alias) buffers carry their mode in attr, an X (pointer) buffer its
// index in the client's list, a C (receive) buffer nothing. Addresses are the client's, in guest memory.
typedef struct IpcBuffer {
    uint64_t addr, size;
    uint32_t attr;
} IpcBuffer;

typedef struct IpcRequest {
    unsigned type;   // HIPC message type (4 request, 5 control, 6 and 7 with context, 16 and above TIPC)
    unsigned cmd;
    unsigned nx, na, nb, nw, nc, ncopy, nmove, has_pid;
    uint64_t pid;
    uint32_t copy[8], move[8];   // handles the client sent: copied (still the client's) and moved (now ours)
    IpcBuffer x[IPC_MAX_BUF], a[IPC_MAX_BUF], b[IPC_MAX_BUF], w[IPC_MAX_BUF], c[IPC_MAX_BUF];
    const uint8_t *data;   // the command's arguments, after the CMIF header
    uint32_t data_size;
} IpcRequest;

typedef struct IpcSession IpcSession;
typedef struct IpcResponse {
    uint32_t result;
    uint32_t copy[8], move[8];   // handles the answer carries; copy ones stay ours as well, moved ones are the client's
    unsigned ncopy, nmove;
    IpcBuffer xout[IPC_MAX_BUF];   // answers written into the client's receive buffers: addr, size, index
    unsigned nxout;
    IpcSession *objs[8];   // sessions made for the client, see ipc_answer_session
    unsigned nobjs;
    uint8_t data[0x80];
    uint32_t data_size;
} IpcResponse;

typedef void (*IpcHandler)(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs);

struct IpcSession {
    const char *name;
    IpcHandler handler;
    void *state;                   // the service's own, handed over by ipc_session_new
    void (*close)(void *state);    // run once, when the last reference to the session is gone
    int refs;                      // the creator's, one per handle, one per request in flight, one per domain slot
    IpcSession **objs;             // a converted session's domain: IPC_DOMAIN_OBJECTS slots, else NULL
};

// ---- sessions
// A new session with one reference, the caller's. Hand it to the client with ipc_answer_session; release it
// yourself only when you keep it for the host's own use.
IpcSession *ipc_session_new(const char *name, IpcHandler handler, void *state, void (*close)(void *state));
void ipc_session_release(IpcSession *s);
// Answer a command with the session s: the caller's reference passes to the answer. The client gets a moved handle,
// or, when the request came through a CMIF domain, an object id in that domain.
void ipc_answer_session(IpcResponse *rs, IpcSession *s);

// ---- the names sm: serves. Any other name stays the named fatal (and 0x415 in census mode).
void ipc_service_add(const char *name, IpcHandler handler);

// ---- reading a request and building the answer
// The arguments as the CMIF or TIPC payload carries them; a read past what the request holds is a named fatal.
const void *ipc_arg(Cpu *c, const IpcSession *s, const IpcRequest *rq, uint32_t off, uint32_t n);
void ipc_out(IpcResponse *rs, const void *p, uint32_t n);
// A command the service does not know: the census row and the named fatal, with the framing, the argument size and
// the buffer and handle counts.
void ipc_unknown_command(Cpu *c, const IpcSession *s, const IpcRequest *rq) __attribute__((noreturn));
// Bytes of a buffer the client described: 1, or 0 when the range lies outside it or is not guest memory the guest may
// read (write).
int ipc_read(const IpcBuffer *b, uint64_t off, void *dst, uint64_t n);
int ipc_write(const IpcBuffer *b, uint64_t off, const void *src, uint64_t n);
// Answer into the client's receive (C) buffer number index: the bytes are written there and the answer carries the
// pointer descriptor that names them. 1, or 0 when the client gave no such buffer or it is too small.
int ipc_answer_pointer(IpcResponse *rs, const IpcRequest *rq, unsigned index, const void *data, uint64_t size);

// ---- transfer memory the client made (SVC 0x15) and sent as a handle
typedef struct IpcTransferMemory {
    uint64_t addr, size;
    uint32_t perm;
} IpcTransferMemory;
int ipc_transfer_memory(uint32_t handle, IpcTransferMemory *out);   // 1, or 0 when the handle is no transfer memory
