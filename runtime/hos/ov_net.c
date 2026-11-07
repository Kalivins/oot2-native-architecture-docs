// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Host overrides of the network calls the game reaches through public nn:: entry points: nn::socket, nn::nifm,
// nn::ssl and nn::nsd, on a host with no network. They answer as a console with no network answers (the rule at the
// top of hos.h), so the game's online library, a gRPC client, starts, finds nobody to talk to and goes on offline:
//   - nn::socket::Initialize succeeds and keeps no state that another call depends on (the services study); every
//     other socket call fails with -1 and ENETDOWN, a name lookup with EAI_FAIL or NULL;
//   - event file descriptors (EventFd, EventFdRead, EventFdWrite) are counters in host memory, and Close, Fcntl, Poll
//     and Select work on them, because gRPC and the SDK's movie library wake a waiting thread through one; Poll and
//     Select on any other descriptor report nothing ready and wait out their timeout, as a socket with nothing to say
//     does;
//   - nn::nifm reports no network and a request that never gets one, nn::ssl creates a context and no connection,
//     nn::nsd names the retail environment and resolves no name.
//
// Adapted from two projects, each at the commit read for the services study. oot-pc-recomp at 8e09aaa
// (https://github.com/philneitz/oot-pc-recomp, a GPL-3.0-or-later repository): runtime/ov_eventfd.c (no file header,
// so GPL-3.0-or-later: the eventfd table as host counters, and Poll over it) and runtime/ov_system.c (MIT file header,
// Copyright (c) 2026 MacNativeGames contributors: the nn::socket, nn::nifm, nn::nsd and nn::ssl overrides that
// succeed or fail as an offline console does). The answers and their reasons also follow NXNative
// (https://github.com/seedds/nxnative, MIT, Copyright (c) 2026 seedds), dm/host/hle_net.c, hle_offline.c and
// hle_misc.c on the branch pr1299 at ee9b776: Horizon's eventfd flags, the counter rules, readiness in slices of at
// most 100 ms, what Poll and Select do with a descriptor the host never made, the held network request, the ssl and
// nsd answers. NXNative's code is for a native ABI, so no code was taken from it, only its answers.
//
// Changed here:
//   - EventFd's flags are not Linux's (decision below), and the counters are served at EventFdRead and EventFdWrite,
//     the SDK functions the game imports, as well as at the public Read and Write; the donor served only Read and
//     Write;
//   - Select works on the descriptors (the donor failed it), and Poll and Select on a descriptor the host never made
//     report nothing ready where the donor reported POLLNVAL;
//   - errno is the guest's own, written and read through the SDK's __errno_location, with the numbers this SDK uses
//     (below);
//   - a wait gives the virtual core up (rt_kernel_block_begin) and runs in slices of at most 100 ms; a wait with no
//     eventfd in it, which is how one of the game's libc sleep functions waits, uses the high-resolution timer instead
//     of the condition variable;
//   - every override is a HOS_OVERRIDE with its status and source; guest pointers go through hos_in and hos_out; a
//     path with no known answer is a named gap (hos_gap).
//
// ---- What the SDK and the game do, measured on the modules and not taken from a donor
//
// EventFd flags (the services study). nn::socket::EventFd hands its flags word to the bsd client interface unchanged,
// with no mask and no translation, so the numbering is that of the caller's headers and the service. The game's libc
// eventfd is a shim that passes it on unchanged as well. The two calls the game's gRPC makes pass 6 (the wake-up
// descriptor: non-blocking plus close-on-exec) and 0 (an availability probe, closed at once), and the SDK's movie
// library passes 0; Linux's 0x800 and 0x80000 never occur. The numbering is therefore NXNative's: 0x1 semaphore, 0x2
// close-on-exec (nothing execs here, so no effect), 0x4 non-blocking. oot-pc-recomp reads 0x800 as non-blocking, so
// gRPC's non-blocking wake-up descriptor would block on an empty counter. The SDK's EventFdRead and EventFdWrite are
// an 8-byte Read and Write through the SDK's internal pair, answering 0 or -1, and the game's libc eventfd_read and
// eventfd_write branch straight into them; the public Read and Write that oot-pc-recomp overrides are not on that
// path, so its counters would not have served this game's reads and writes.
//
// errno. In this SDK the socket library's last error is the C library's errno: GetLastErrno reads it and
// SetLastErrno writes it, and the SDK's own functions store the service's code there. The numbers are Linux's: the
// game's libc compares what GetLastErrno returns with EAGAIN 11 and EINPROGRESS 115, and the SDK's movie library
// compares what GetLastError returns with EAGAIN 11. So ENETDOWN is 100, not the FreeBSD 50 that NXNative uses, and
// an override sets the guest's errno through the SDK's own __errno_location instead of keeping a second error code.
// The same libc passes Linux's O_NONBLOCK (0x800) to fcntl, with F_GETFD, F_SETFD, F_GETFL and F_SETFL as 1 to 4.
//
// Sleeping. One of the game's libc sleep functions waits with Poll(NULL, 0, ms), so a Poll with nothing to wait on
// has to be an accurate sleep.
#include <string.h>

#include "hos.h"
#include "plat.h"

// Errno numbers (see above)
enum {
    NET_E_BADF = 9,
    NET_E_AGAIN = 11,
    NET_E_FAULT = 14,
    NET_E_INVAL = 22,
    NET_E_MFILE = 24,
    NET_E_NETDOWN = 100,
};
#define NET_EAI_FAIL 4u            // nn::socket::AiErrno: EAi_Fail

enum {
    NET_EFD_BASE = 512,            // the host's eventfds are the descriptors from here: below an fd_set's 1024 bits,
    NET_EFD_MAX = 128,             // and the host makes no other descriptor
    NET_EFD_SEMAPHORE = 0x1,       // EventFd flags
    NET_EFD_NONBLOCK = 0x4,
    NET_POLLIN = 0x1,              // poll events, in the values Linux and the BSDs share
    NET_POLLOUT = 0x4,
    NET_POLLRDNORM = 0x40,
    NET_POLLWRNORM = 0x4,
    NET_F_GETFD = 1,
    NET_F_SETFD = 2,
    NET_F_GETFL = 3,
    NET_F_SETFL = 4,
    NET_O_NONBLOCK = 0x800,
    NET_POLL_MAX = 1024,           // descriptors in one Poll
    NET_FDSET_BYTES = 128,         // an fd_set: 1024 bits, bit n of byte n / 8
    NET_FDSET_BITS = 1024,
};
#define NET_EFD_LIMIT (UINT64_MAX - 1)   // the largest value a counter holds
#define NET_SLICE_NS 100000000ull        // no wait is longer than this at a time

typedef struct {
    int32_t fd;
    int16_t events, revents;
} PollFd;                          // the C library's pollfd and the SDK's PollFd
_Static_assert(sizeof(PollFd) == 8, "a pollfd is 8 bytes");

// An output of n bytes: null is not written; an address that is not guest memory the guest may write stops the guest.
static void out_put(Cpu *c, const char *fn, uint64_t addr, const void *src, uint64_t n) {
    if (addr && !hos_out(addr, src, n))
        rt_guest_stop(c, "%s: %#llx is not guest memory the guest may write", fn, (unsigned long long)addr);
}

// ---- errno: the guest's, through the SDK's __errno_location
static uint64_t errno_cell(Cpu *c) {
    static uint64_t location;
    uint64_t fn = __atomic_load_n(&location, __ATOMIC_RELAXED);
    if (!fn) {
        fn = hos_sdk(c, "__errno_location");
        __atomic_store_n(&location, fn, __ATOMIC_RELAXED);
    }
    return hos_call(c, fn, 0);
}

static void set_errno(Cpu *c, int e) {
    uint64_t cell = errno_cell(c);
    int32_t v = e;
    if (cell && !hos_out(cell, &v, sizeof v))
        rt_guest_stop(c, "nn::socket: errno at %#llx is not guest memory the guest may write",
                      (unsigned long long)cell);
}

static int get_errno(Cpu *c) {
    uint64_t cell = errno_cell(c);
    int32_t v = 0;
    if (cell && !hos_in(&v, cell, sizeof v))
        rt_guest_stop(c, "nn::socket: errno at %#llx is not guest memory the guest may read", (unsigned long long)cell);
    return v;
}

// A failed call: -1 in the whole of x0 (an int or an ssize_t) and errno e
static void fail(Cpu *c, int e) {
    set_errno(c, e);
    ret_x(c, ~0ull);
}

// ---- the host's event descriptors
typedef struct {
    int used, nonblock, semaphore;
    uint32_t gen;                  // changes at every creation: a waiter learns that its descriptor was closed
    uint64_t count;
} Efd;

static PlatMutex net_lock = PLAT_MUTEX_INIT;   // guards the table, socket_users and the waits on net_cv
static PlatCond net_cv = PLAT_COND_INIT;       // broadcast at every change of a counter and at every close
static Efd efds[NET_EFD_MAX];
static int socket_users;                       // Initialize calls minus Finalize calls
static int nifm_hold;                          // queries IsNetworkRequestOnHold still answers "on hold" to (atomic)

static Efd *efd_find(int32_t fd) {   // lock held
    if (fd < NET_EFD_BASE || fd >= NET_EFD_BASE + NET_EFD_MAX) return NULL;
    Efd *e = &efds[fd - NET_EFD_BASE];
    return e->used ? e : NULL;
}

static int is_efd(int32_t fd) {
    plat_mutex_lock(&net_lock);
    int yes = efd_find(fd) != NULL;
    plat_mutex_unlock(&net_lock);
    return yes;
}

// The counter of e, taken as eventfd(2) does: all of it, or 1 for a semaphore. Lock held, count not 0.
static uint64_t efd_take(Efd *e) {
    uint64_t v = e->semaphore ? 1 : e->count;
    e->count -= v;
    plat_cond_broadcast(&net_cv);
    return v;
}

// Add v to e; 0 when the sum would pass the limit. Lock held.
static int efd_add(Efd *e, uint64_t v) {
    if (e->count > NET_EFD_LIMIT - v) return 0;
    e->count += v;
    plat_cond_broadcast(&net_cv);
    return 1;
}

// An 8-byte read: 0 with the value in *out, else the errno. A blocking descriptor with a zero counter waits for a
// write, with the virtual core given up, and ends with EBADF if the descriptor is closed meanwhile.
static int efd_read(int32_t fd, uint64_t *out) {
    plat_mutex_lock(&net_lock);
    Efd *e = efd_find(fd);
    if (!e) {
        plat_mutex_unlock(&net_lock);
        return NET_E_BADF;
    }
    if (e->count) {
        *out = efd_take(e);
        plat_mutex_unlock(&net_lock);
        return 0;
    }
    if (e->nonblock) {
        plat_mutex_unlock(&net_lock);
        return NET_E_AGAIN;
    }
    uint32_t gen = e->gen;
    plat_mutex_unlock(&net_lock);
    int token = rt_kernel_block_begin();
    int err = 0;
    plat_mutex_lock(&net_lock);
    for (;;) {
        e = efd_find(fd);
        if (!e || e->gen != gen) {
            err = NET_E_BADF;
            break;
        }
        if (e->count) {
            *out = efd_take(e);
            break;
        }
        plat_cond_wait_ns(&net_cv, &net_lock, NET_SLICE_NS);
    }
    plat_mutex_unlock(&net_lock);
    rt_kernel_block_end(token);
    return err;
}

// An 8-byte write of v: 0 or the errno. A blocking descriptor whose counter would pass the limit waits for a read.
static int efd_write(int32_t fd, uint64_t v) {
    if (v == UINT64_MAX) return NET_E_INVAL;
    plat_mutex_lock(&net_lock);
    Efd *e = efd_find(fd);
    if (!e) {
        plat_mutex_unlock(&net_lock);
        return NET_E_BADF;
    }
    if (efd_add(e, v)) {
        plat_mutex_unlock(&net_lock);
        return 0;
    }
    if (e->nonblock) {
        plat_mutex_unlock(&net_lock);
        return NET_E_AGAIN;
    }
    uint32_t gen = e->gen;
    plat_mutex_unlock(&net_lock);
    int token = rt_kernel_block_begin();
    int err = 0;
    plat_mutex_lock(&net_lock);
    for (;;) {
        e = efd_find(fd);
        if (!e || e->gen != gen) {
            err = NET_E_BADF;
            break;
        }
        if (efd_add(e, v)) break;
        plat_cond_wait_ns(&net_cv, &net_lock, NET_SLICE_NS);
    }
    plat_mutex_unlock(&net_lock);
    rt_kernel_block_end(token);
    return err;
}

// ---- waiting for readiness (Poll, Select)
typedef struct {
    uint64_t deadline;
    int forever, nowait;
} Timeout;

static Timeout timeout_of(int forever, uint64_t ns) {
    Timeout t;
    t.forever = forever;
    t.nowait = !forever && ns == 0;
    t.deadline = forever ? 0 : plat_now_ns() + ns;
    return t;
}

// How long to wait next, one slice at most; 0 when the time is up.
static uint64_t slice_of(const Timeout *t) {
    if (t->forever) return NET_SLICE_NS;
    uint64_t now = plat_now_ns();
    if (now >= t->deadline) return 0;
    uint64_t left = t->deadline - now;
    return left < NET_SLICE_NS ? left : NET_SLICE_NS;
}

// Looks at the descriptors under net_lock: how many are ready, and *mine says whether any is a host eventfd.
typedef int (*EvalFn)(void *ctx, int *mine);

// Until eval finds something ready or the timeout passes. The first look costs nothing; a wait gives the virtual core
// up and goes in slices. With an eventfd among the descriptors a slice is a wait on the condition variable, which
// every counter change ends; without one it is a plain sleep, accurate to the high-resolution timer, and the
// descriptors are looked at again after it (a descriptor number can become an eventfd meanwhile).
static int wait_ready(EvalFn eval, void *ctx, const Timeout *t) {
    int mine = 0;
    plat_mutex_lock(&net_lock);
    int ready = eval(ctx, &mine);
    plat_mutex_unlock(&net_lock);
    if (ready || t->nowait) return ready;
    int token = rt_kernel_block_begin();
    for (;;) {
        plat_mutex_lock(&net_lock);
        ready = eval(ctx, &mine);
        uint64_t slice = ready ? 0 : slice_of(t);
        if (slice && mine) plat_cond_wait_ns(&net_cv, &net_lock, slice);
        plat_mutex_unlock(&net_lock);
        if (!slice) break;
        if (!mine) plat_sleep_ns(slice);
    }
    rt_kernel_block_end(token);
    return ready;
}

typedef struct {
    PollFd *p;
    uint32_t n;
    int foreign;                   // some descriptor is not a host eventfd
} PollCtx;

static int poll_eval(void *ctx, int *mine) {
    PollCtx *x = ctx;
    int ready = 0;
    *mine = 0;
    for (uint32_t i = 0; i < x->n; i++) {
        int rev = 0;
        if (x->p[i].fd >= 0) {     // a negative descriptor is ignored, as POSIX says
            Efd *e = efd_find(x->p[i].fd);
            if (e) {
                *mine = 1;
                if (e->count) rev |= x->p[i].events & (NET_POLLIN | NET_POLLRDNORM);
                if (e->count < NET_EFD_LIMIT) rev |= x->p[i].events & (NET_POLLOUT | NET_POLLWRNORM);
            } else {
                x->foreign = 1;
            }
        }
        x->p[i].revents = (int16_t)rev;
        ready += rev != 0;
    }
    return ready;
}

// int Poll(pollfd *fds, nfds_t n, int timeout_ms) and Poll(PollFd *, ...): the number of descriptors with an event.
// Negative timeout: no limit.
static void ov_poll(Cpu *c) {
    uint64_t addr = arg_x(c, 0), n = arg_x(c, 1);
    int32_t ms = (int32_t)arg_x(c, 2);
    PollFd fds[NET_POLL_MAX];
    if (n > (uint64_t)NET_POLL_MAX) {
        fail(c, NET_E_INVAL);
        return;
    }
    if (n && !hos_in(fds, addr, n * sizeof fds[0])) {
        fail(c, NET_E_FAULT);
        return;
    }
    PollCtx x = {fds, (uint32_t)n, 0};
    Timeout t = timeout_of(ms < 0, ms > 0 ? (uint64_t)ms * 1000000ull : 0);
    int ready = wait_ready(poll_eval, &x, &t);
    for (uint64_t i = 0; i < n; i++)
        if (!hos_out(addr + i * sizeof fds[0] + offsetof(PollFd, revents), &fds[i].revents, sizeof fds[i].revents)) {
            fail(c, NET_E_FAULT);
            return;
        }
    static int logged;
    if (x.foreign && !__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
        rt_log("socket: Poll of %llu descriptors with one the host never made: it reports nothing, the timeout (%d ms) "
               "is waited out", (unsigned long long)n, ms);
    ret_x(c, (uint64_t)(int64_t)ready);
}

typedef struct {
    uint8_t b[NET_FDSET_BYTES];
} FdBits;

typedef struct {
    int32_t nfds;
    int have[3];                   // the read, write and exception sets the caller passed
    FdBits want[3], got[3];
    int foreign;
} SelCtx;

static int select_eval(void *ctx, int *mine) {
    SelCtx *x = ctx;
    int ready = 0;
    *mine = 0;
    memset(x->got, 0, sizeof x->got);
    for (int k = 0; k < 2; k++) {  // read and write; nothing is ever exceptional
        if (!x->have[k]) continue;
        for (int32_t fd = 0; fd < x->nfds; fd++) {
            if (!((x->want[k].b[fd >> 3] >> (fd & 7)) & 1)) continue;
            Efd *e = efd_find(fd);
            if (!e) {
                x->foreign = 1;
                continue;
            }
            *mine = 1;
            if (k == 0 ? e->count != 0 : e->count < NET_EFD_LIMIT) {
                x->got[k].b[fd >> 3] |= (uint8_t)(1u << (fd & 7));
                ready++;
            }
        }
    }
    return ready;
}

// int Select(int nfds, fd_set *r, fd_set *w, fd_set *e, timeval *tv): the descriptors left in the sets are the ready
// ones. A null timeout: no limit. timeval is {int64 seconds, int64 microseconds}.
static void ov_select(Cpu *c) {
    int32_t nfds = (int32_t)arg_x(c, 0);
    uint64_t sets[3] = {arg_x(c, 1), arg_x(c, 2), arg_x(c, 3)}, tv = arg_x(c, 4);
    if (nfds < 0 || nfds > NET_FDSET_BITS) {
        fail(c, NET_E_INVAL);
        return;
    }
    SelCtx x;
    memset(&x, 0, sizeof x);
    x.nfds = nfds;
    for (int k = 0; k < 3; k++) {
        x.have[k] = sets[k] != 0;
        if (x.have[k] && !hos_in(&x.want[k], sets[k], sizeof x.want[k])) {
            fail(c, NET_E_FAULT);
            return;
        }
    }
    int64_t t[2] = {0, 0};
    if (tv && !hos_in(t, tv, sizeof t)) {
        fail(c, NET_E_FAULT);
        return;
    }
    if (tv && (t[0] < 0 || t[1] < 0 || t[1] >= 1000000)) {
        fail(c, NET_E_INVAL);
        return;
    }
    int forever = !tv || t[0] > 100000000;   // no timeout, or more than three years
    Timeout to = timeout_of(forever, forever ? 0 : (uint64_t)t[0] * 1000000000ull + (uint64_t)t[1] * 1000ull);
    int ready = wait_ready(select_eval, &x, &to);
    for (int k = 0; k < 3; k++)
        if (x.have[k] && !hos_out(sets[k], &x.got[k], sizeof x.got[k])) {
            fail(c, NET_E_FAULT);
            return;
        }
    static int logged;
    if (x.foreign && !__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
        rt_log("socket: Select with a descriptor the host never made: it reports nothing, the timeout is waited out");
    ret_x(c, (uint64_t)(int64_t)ready);
}

// ---- the calls on the descriptors
// int EventFd(unsigned long initval, EventFdFlags flags): the descriptor, or -1
static void ov_eventfd(Cpu *c) {
    uint64_t init = arg_x(c, 0);
    uint32_t flags = (uint32_t)arg_x(c, 1);
    if (init > NET_EFD_LIMIT) {
        fail(c, NET_E_INVAL);
        return;
    }
    int fd = -1;
    plat_mutex_lock(&net_lock);
    for (int i = 0; i < NET_EFD_MAX && fd < 0; i++) {
        Efd *e = &efds[i];
        if (e->used) continue;
        e->used = 1;
        e->nonblock = (flags & NET_EFD_NONBLOCK) != 0;
        e->semaphore = (flags & NET_EFD_SEMAPHORE) != 0;
        e->count = init;
        e->gen++;
        fd = NET_EFD_BASE + i;
        if (init) plat_cond_broadcast(&net_cv);   // a Poll or Select that waits on this number may see it ready now
    }
    plat_mutex_unlock(&net_lock);
    if (fd < 0) {
        fail(c, NET_E_MFILE);
        return;
    }
    static int logged;
    if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
        rt_log("socket: event descriptors are served by the host (first: descriptor %d, flags %#x)", fd, flags);
    ret_x(c, (uint64_t)fd);
}
HOS_OVERRIDE("_ZN2nn6socket7EventFdEmNS0_12EventFdFlagsE", ov_eventfd, HOS_IMPLEMENTED,
             "a counter in a host table, descriptors from 512; flags 1 semaphore, 4 non-blocking (the game passes 6 "
             "and 0); nxnative hle_net.c:91, oot ov_eventfd.c:36");

// int EventFdRead(int fd, unsigned long *out): 0, or -1 with errno
static void ov_eventfd_read(Cpu *c) {
    int32_t fd = (int32_t)arg_x(c, 0);
    uint64_t out = arg_x(c, 1), v = 0;
    if (!out) {                    // the SDK answers a null pointer with -1 and leaves errno alone
        ret_x(c, ~0ull);
        return;
    }
    int err = efd_read(fd, &v);
    if (err) {
        fail(c, err);
        return;
    }
    out_put(c, "nn::socket::EventFdRead", out, &v, sizeof v);
    ret_x(c, 0);
}
HOS_OVERRIDE("_ZN2nn6socket11EventFdReadEiPm", ov_eventfd_read, HOS_IMPLEMENTED,
             "takes the counter (1 for a semaphore); EAGAIN when a non-blocking one is empty; nxnative hle_net.c:100");

// int EventFdWrite(int fd, unsigned long value): 0, or -1 with errno
static void ov_eventfd_write(Cpu *c) {
    int err = efd_write((int32_t)arg_x(c, 0), arg_x(c, 1));
    if (err) {
        fail(c, err);
        return;
    }
    ret_x(c, 0);
}
HOS_OVERRIDE("_ZN2nn6socket12EventFdWriteEim", ov_eventfd_write, HOS_IMPLEMENTED,
             "adds to the counter; EAGAIN when a non-blocking one would overflow; nxnative hle_net.c:108");

// ssize_t Read(int fd, void *buf, size_t n) and Write(int fd, const void *buf, size_t n): the 8-byte read and write of
// an eventfd; any other descriptor fails like every socket call
static void ov_read(Cpu *c) {
    int32_t fd = (int32_t)arg_x(c, 0);
    uint64_t buf = arg_x(c, 1), n = arg_x(c, 2), v = 0;
    if (!is_efd(fd)) {
        fail(c, NET_E_NETDOWN);
        return;
    }
    if (n < 8) {
        fail(c, NET_E_INVAL);
        return;
    }
    if (!rt_mem_commit_io(buf, 8, 1)) {
        fail(c, NET_E_FAULT);
        return;
    }
    int err = efd_read(fd, &v);
    if (err) {
        fail(c, err);
        return;
    }
    out_put(c, "nn::socket::Read", buf, &v, sizeof v);
    ret_x(c, 8);
}
HOS_OVERRIDE("_ZN2nn6socket4ReadEiPvm", ov_read, HOS_IMPLEMENTED,
             "the 8-byte read of an eventfd (as EventFdRead); any other descriptor fails; oot ov_eventfd.c:52");

static void ov_write(Cpu *c) {
    int32_t fd = (int32_t)arg_x(c, 0);
    uint64_t buf = arg_x(c, 1), n = arg_x(c, 2), v = 0;
    if (!is_efd(fd)) {
        fail(c, NET_E_NETDOWN);
        return;
    }
    if (n < 8) {
        fail(c, NET_E_INVAL);
        return;
    }
    if (!hos_in(&v, buf, sizeof v)) {
        fail(c, NET_E_FAULT);
        return;
    }
    int err = efd_write(fd, v);
    if (err) {
        fail(c, err);
        return;
    }
    ret_x(c, 8);
}
HOS_OVERRIDE("_ZN2nn6socket5WriteEiPKvm", ov_write, HOS_IMPLEMENTED,
             "the 8-byte write of an eventfd (as EventFdWrite); any other descriptor fails; oot ov_eventfd.c:78");

// int Close(int fd): the eventfd goes and its waiters wake with EBADF; any other descriptor fails
static void ov_close(Cpu *c) {
    int32_t fd = (int32_t)arg_x(c, 0);
    plat_mutex_lock(&net_lock);
    Efd *e = efd_find(fd);
    if (e) {
        e->used = 0;
        plat_cond_broadcast(&net_cv);
    }
    plat_mutex_unlock(&net_lock);
    if (e)
        ret_x(c, 0);
    else
        fail(c, NET_E_NETDOWN);
}
HOS_OVERRIDE("_ZN2nn6socket5CloseEi", ov_close, HOS_IMPLEMENTED,
             "closes a host eventfd and wakes its waiters; any other descriptor fails (ENETDOWN); "
             "nxnative hle_net.c:77,167");

// int Fcntl(int fd, int cmd, ...): the first variadic argument is in w2
static void ov_fcntl(Cpu *c) {
    int32_t fd = (int32_t)arg_x(c, 0), cmd = (int32_t)arg_x(c, 1);
    uint32_t arg = (uint32_t)arg_x(c, 2);
    int mine = 0, result = 0, err = 0;
    plat_mutex_lock(&net_lock);
    Efd *e = efd_find(fd);
    if (e) {
        mine = 1;
        switch (cmd) {
        case NET_F_GETFD:
        case NET_F_SETFD:
            break;                 // close-on-exec: nothing execs here
        case NET_F_GETFL:
            result = e->nonblock ? NET_O_NONBLOCK : 0;
            break;
        case NET_F_SETFL:
            e->nonblock = (arg & NET_O_NONBLOCK) != 0;
            break;
        default:
            err = NET_E_INVAL;
            break;
        }
    }
    plat_mutex_unlock(&net_lock);
    if (!mine)
        fail(c, NET_E_NETDOWN);
    else if (err)
        fail(c, err);
    else
        ret_x(c, (uint64_t)result);
}
HOS_OVERRIDE("_ZN2nn6socket5FcntlEiiz", ov_fcntl, HOS_IMPLEMENTED,
             "F_GETFL and F_SETFL on a host eventfd, O_NONBLOCK as the game's libc passes it (0x800); others fail; "
             "nxnative hle_net.c:81,172");
HOS_OVERRIDE("_ZN2nn6socket5FcntlEiNS0_12FcntlCommandEz", ov_fcntl, HOS_IMPLEMENTED,
             "as the int form of Fcntl: F_GETFL and F_SETFL on a host eventfd; others fail; nxnative hle_net.c:81,172");

HOS_OVERRIDE("_ZN2nn6socket4PollEP6pollfdmi", ov_poll, HOS_IMPLEMENTED,
             "readiness of host eventfds; any other descriptor reports nothing and the timeout is waited out in "
             "slices of at most 100 ms (one of the game's libc sleep functions waits this way); "
             "nxnative hle_net.c:125, oot ov_eventfd.c:106");
HOS_OVERRIDE("_ZN2nn6socket4PollEPNS0_6PollFdEmi", ov_poll, HOS_IMPLEMENTED,
             "as the pollfd form of Poll; nxnative hle_net.c:180");
HOS_OVERRIDE("_ZN2nn6socket6SelectEiPNS0_5FdSetES2_S2_PNS0_7TimeValE", ov_select, HOS_IMPLEMENTED,
             "readiness of host eventfds in the sets (the SDK's movie library waits on one); other descriptors report "
             "nothing; nxnative hle_net.c:143,184");
HOS_OVERRIDE("_ZN2nn6socket6SelectEiP6fd_setS2_S2_P7timeval", ov_select, HOS_IMPLEMENTED,
             "as the FdSet form of Select; nxnative hle_offline.c:86");

// Ioctl on an eventfd: the request codes of an event descriptor are not recorded; any other descriptor fails
static void ov_ioctl(Cpu *c) {
    int32_t fd = (int32_t)arg_x(c, 0);
    if (is_efd(fd)) hos_gap(c, "socket", "Ioctl on the event descriptor %d: the request codes are not recorded", fd);
    fail(c, NET_E_NETDOWN);
}

// ---- the library's state
// Result Initialize(...) and Result Finalize(...): success, with no session to open or close
static void ov_socket_initialize(Cpu *c) {
    plat_mutex_lock(&net_lock);
    socket_users++;
    plat_mutex_unlock(&net_lock);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn6socket10InitializeERKNS0_6ConfigE", ov_socket_initialize, HOS_OFFLINE,
             "success, no session and no state a later call depends on (trap T2); nxnative hle_net.c:9, "
             "oot ov_system.c:946");
HOS_OVERRIDE("_ZN2nn6socket10InitializeEPvmmi", ov_socket_initialize, HOS_OFFLINE,
             "success, no session; nxnative hle_offline.c:76, oot ov_system.c:945");
HOS_OVERRIDE("_ZN2nn6socket8internal10InitializeEi", ov_socket_initialize, HOS_OFFLINE,
             "success, no session; nxnative hle_net.c:11");

static void ov_socket_finalize(Cpu *c) {
    plat_mutex_lock(&net_lock);
    if (socket_users > 0) socket_users--;
    plat_mutex_unlock(&net_lock);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn6socket8FinalizeEv", ov_socket_finalize, HOS_OFFLINE,
             "success, pairs with Initialize; nxnative hle_offline.c:77, oot ov_system.c:948");
HOS_OVERRIDE("_ZN2nn6socket8FinalizeEi", ov_socket_finalize, HOS_OFFLINE,
             "success, pairs with Initialize; oot ov_system.c:947");

// bool IsInitialized(): after an Initialize that no Finalize undid
static void ov_socket_is_initialized(Cpu *c) {
    plat_mutex_lock(&net_lock);
    int on = socket_users > 0;
    plat_mutex_unlock(&net_lock);
    ret_x(c, (uint64_t)on);
}
HOS_OVERRIDE("_ZN2nn6socket13IsInitializedEv", ov_socket_is_initialized, HOS_IMPLEMENTED,
             "true between Initialize and Finalize; oot ov_system.c:949 answers always true");

// int GetLastErrno(), Errno GetLastError(), void SetLastErrno(int), void SetLastError(Errno): the guest's errno
static void ov_get_errno(Cpu *c) {
    int e = get_errno(c);
    ret_x(c, (uint32_t)e);
}
static void ov_set_errno(Cpu *c) { set_errno(c, (int)arg_x(c, 0)); }
#define WHY_ERRNO \
    "errno is the socket library's last error in this SDK; nxnative hle_net.c:195,212-213 keeps a second code"
HOS_OVERRIDE("_ZN2nn6socket12GetLastErrnoEv", ov_get_errno, HOS_IMPLEMENTED, WHY_ERRNO);
HOS_OVERRIDE("_ZN2nn6socket12GetLastErrorEv", ov_get_errno, HOS_IMPLEMENTED, WHY_ERRNO);
HOS_OVERRIDE("_ZN2nn6socket12SetLastErrnoEi", ov_set_errno, HOS_IMPLEMENTED, WHY_ERRNO);
HOS_OVERRIDE("_ZN2nn6socket12SetLastErrorENS0_5ErrnoE", ov_set_errno, HOS_IMPLEMENTED, WHY_ERRNO);

// ---- every other socket call: there is no network, so no socket
static void ov_net_fail(Cpu *c) { fail(c, NET_E_NETDOWN); }
#define WHY_FAIL \
    "-1 with ENETDOWN: no network, so no socket exists; nxnative hle_net.c:162-195, oot ov_system.c:957-1000"
#define SOCKET_FAIL(sym) HOS_OVERRIDE(sym, ov_net_fail, HOS_OFFLINE, WHY_FAIL)
SOCKET_FAIL("_ZN2nn6socket6AcceptEiPNS0_8SockAddrEPj");
SOCKET_FAIL("_ZN2nn6socket6AcceptEiP8sockaddrPj");
SOCKET_FAIL("_ZN2nn6socket4BindEiPKNS0_8SockAddrEj");
SOCKET_FAIL("_ZN2nn6socket4BindEiPK8sockaddrj");
SOCKET_FAIL("_ZN2nn6socket7ConnectEiPKNS0_8SockAddrEj");
SOCKET_FAIL("_ZN2nn6socket7ConnectEiPK8sockaddrj");
SOCKET_FAIL("_ZN2nn6socket11GetPeerNameEiPNS0_8SockAddrEPj");
SOCKET_FAIL("_ZN2nn6socket11GetPeerNameEiP8sockaddrPj");
SOCKET_FAIL("_ZN2nn6socket11GetSockNameEiPNS0_8SockAddrEPj");
SOCKET_FAIL("_ZN2nn6socket11GetSockNameEiP8sockaddrPj");
SOCKET_FAIL("_ZN2nn6socket10GetSockOptEiNS0_5LevelENS0_6OptionEPvPj");
SOCKET_FAIL("_ZN2nn6socket10GetSockOptEiiiPvPj");
SOCKET_FAIL("_ZN2nn6socket10SetSockOptEiNS0_5LevelENS0_6OptionEPKvj");
SOCKET_FAIL("_ZN2nn6socket10SetSockOptEiiiPKvj");
SOCKET_FAIL("_ZN2nn6socket6ListenEii");
SOCKET_FAIL("_ZN2nn6socket10SockAtMarkEi");
SOCKET_FAIL("_ZN2nn6socket8ShutdownEiNS0_14ShutdownMethodE");
SOCKET_FAIL("_ZN2nn6socket8ShutdownEii");
SOCKET_FAIL("_ZN2nn6socket18ShutdownAllSocketsEPKNS0_8SockAddrEjNS0_22ShutdownAllSocketsMaskE");
SOCKET_FAIL("_ZN2nn6socket6SocketENS0_6FamilyENS0_4TypeENS0_8ProtocolE");
SOCKET_FAIL("_ZN2nn6socket6SocketEiii");
SOCKET_FAIL("_ZN2nn6socket12SocketExemptENS0_6FamilyENS0_4TypeENS0_8ProtocolE");
SOCKET_FAIL("_ZN2nn6socket12SocketExemptEiii");
SOCKET_FAIL("_ZN2nn6socket4RecvEiPvmNS0_7MsgFlagE");
SOCKET_FAIL("_ZN2nn6socket4RecvEiPvmi");
SOCKET_FAIL("_ZN2nn6socket8RecvFromEiPvmNS0_7MsgFlagEPNS0_8SockAddrEPj");
SOCKET_FAIL("_ZN2nn6socket8RecvFromEiPvmiP8sockaddrPj");
SOCKET_FAIL("_ZN2nn6socket4SendEiPKvmNS0_7MsgFlagE");
SOCKET_FAIL("_ZN2nn6socket4SendEiPKvmi");
SOCKET_FAIL("_ZN2nn6socket6SendToEiPKvmNS0_7MsgFlagEPKNS0_8SockAddrEj");
SOCKET_FAIL("_ZN2nn6socket6SendToEiPKvmiPK8sockaddrj");
SOCKET_FAIL("_ZN2nn6socket7RecvMsgEiPNS0_6MsgHdrENS0_7MsgFlagE");
SOCKET_FAIL("_ZN2nn6socket7RecvMsgEiP6msghdri");
SOCKET_FAIL("_ZN2nn6socket7SendMsgEiPKNS0_6MsgHdrENS0_7MsgFlagE");
SOCKET_FAIL("_ZN2nn6socket7SendMsgEiPK6msghdri");
SOCKET_FAIL("_ZN2nn6socket8RecvMMsgEiPNS0_7MMsgHdrEmNS0_7MsgFlagEPNS_8TimeSpanE");
SOCKET_FAIL("_ZN2nn6socket8RecvMMsgEiP7mmsghdrmiPNS_8TimeSpanE");
SOCKET_FAIL("_ZN2nn6socket8SendMMsgEiPNS0_7MMsgHdrEmNS0_7MsgFlagE");
SOCKET_FAIL("_ZN2nn6socket8SendMMsgEiP7mmsghdrmi");
SOCKET_FAIL("_ZN2nn6socket4OpenEPKcNS0_8OpenFlagE");
SOCKET_FAIL("_ZN2nn6socket4OpenEPKci");
SOCKET_FAIL("_ZN2nn6socket6SysctlEPimPvPmS2_m");
SOCKET_FAIL("_ZN2nn6socket15DuplicateSocketEim");
SOCKET_FAIL("_ZN2nn6socket17ResolverGetOptionEPNS0_14ResolverOptionENS0_17ResolverOptionKeyE");
SOCKET_FAIL("_ZN2nn6socket17ResolverSetOptionERKNS0_14ResolverOptionE");
HOS_OVERRIDE("_ZN2nn6socket5IoctlEiNS0_12IoctlCommandEPvm", ov_ioctl, HOS_OFFLINE,
             "-1 with ENETDOWN; a named gap on a host eventfd; nxnative hle_net.c:162-195, oot ov_eventfd.c:149");
HOS_OVERRIDE("_ZN2nn6socket5IoctlEijPvm", ov_ioctl, HOS_OFFLINE,
             "-1 with ENETDOWN; a named gap on a host eventfd; oot ov_eventfd.c:148");
HOS_OVERRIDE("_ZN2nn6socket5IoctlEiNS_4util7BitPackIjNS_4anif17IoctlRequestFieldEEEPvm", ov_ioctl, HOS_OFFLINE,
             "-1 with ENETDOWN; a named gap on a host eventfd; oot ov_eventfd.c:150");

// ---- name lookups
// int GetAddrInfo(node, service, hints, AddrInfo **res [, options, count]): EAI_FAIL and no list
static void ov_getaddrinfo(Cpu *c) {
    uint64_t none = 0;
    out_put(c, "nn::socket::GetAddrInfo", arg_x(c, 3), &none, sizeof none);
    ret_x(c, NET_EAI_FAIL);
}
#define WHY_GAI \
    "EAI_FAIL and no result list: nothing resolves without a network; nxnative hle_net.c:190, hle_offline.c:90"
HOS_OVERRIDE("_ZN2nn6socket11GetAddrInfoEPKcS2_PKNS0_8AddrInfoEPPS3_", ov_getaddrinfo, HOS_OFFLINE, WHY_GAI);
HOS_OVERRIDE("_ZN2nn6socket11GetAddrInfoEPKcS2_PK8addrinfoPPS3_", ov_getaddrinfo, HOS_OFFLINE, WHY_GAI);
HOS_OVERRIDE("_ZN2nn6socket11GetAddrInfoEPKcS2_PKNS0_8AddrInfoEPPS3_PKNS0_14ResolverOptionEm", ov_getaddrinfo,
             HOS_OFFLINE, WHY_GAI);
HOS_OVERRIDE("_ZN2nn6socket11GetAddrInfoEPKcS2_PK8addrinfoPPS3_PKNS0_14ResolverOptionEm", ov_getaddrinfo,
             HOS_OFFLINE, WHY_GAI);

// void FreeAddrInfo(list): GetAddrInfo never makes one
static void ov_nothing(Cpu *c) { (void)c; }
HOS_OVERRIDE("_ZN2nn6socket12FreeAddrInfoEPNS0_8AddrInfoE", ov_nothing, HOS_STUB,
             "nothing to free: GetAddrInfo never returns a list; nxnative hle_net.c:193");
HOS_OVERRIDE("_ZN2nn6socket12FreeAddrInfoEP8addrinfo", ov_nothing, HOS_STUB,
             "nothing to free: GetAddrInfo never returns a list; nxnative hle_offline.c:91");

// HostEnt *GetHostByName(...), GetHostByAddr(...), GetHostEntByName(...), GetHostEntByAddr(...): NULL
static void ov_hostent_null(Cpu *c) {
    set_errno(c, NET_E_NETDOWN);
    ret_x(c, 0);
}
#define WHY_HOSTENT \
    "NULL with errno ENETDOWN: nothing resolves without a network; nxnative hle_net.c:178,194, oot ov_system.c:991"
HOS_OVERRIDE("_ZN2nn6socket13GetHostByNameEPKc", ov_hostent_null, HOS_OFFLINE, WHY_HOSTENT);
HOS_OVERRIDE("_ZN2nn6socket13GetHostByNameEPKcPKNS0_14ResolverOptionEm", ov_hostent_null, HOS_OFFLINE, WHY_HOSTENT);
HOS_OVERRIDE("_ZN2nn6socket13GetHostByAddrEPKvji", ov_hostent_null, HOS_OFFLINE, WHY_HOSTENT);
HOS_OVERRIDE("_ZN2nn6socket13GetHostByAddrEPKvjiPKNS0_14ResolverOptionEm", ov_hostent_null, HOS_OFFLINE, WHY_HOSTENT);
HOS_OVERRIDE("_ZN2nn6socket16GetHostEntByNameEPKc", ov_hostent_null, HOS_OFFLINE, WHY_HOSTENT);
HOS_OVERRIDE("_ZN2nn6socket16GetHostEntByNameEPKcPKNS0_14ResolverOptionEm", ov_hostent_null, HOS_OFFLINE, WHY_HOSTENT);
HOS_OVERRIDE("_ZN2nn6socket16GetHostEntByAddrEPKvjNS0_6FamilyE", ov_hostent_null, HOS_OFFLINE, WHY_HOSTENT);
HOS_OVERRIDE("_ZN2nn6socket16GetHostEntByAddrEPKvjNS0_6FamilyEPKNS0_14ResolverOptionEm", ov_hostent_null, HOS_OFFLINE,
             WHY_HOSTENT);

// ==== nn::nifm: no network, and a request for one that does not get it
static void ov_zero(Cpu *c) { ret_x(c, 0); }
static void ov_one(Cpu *c) { ret_x(c, 1); }
static void ov_unavailable(Cpu *c) { ret_x(c, HOS_R_UNAVAILABLE); }

HOS_OVERRIDE("_ZN2nn4nifm10InitializeEv", ov_zero, HOS_OFFLINE,
             "success, no nifm session to open; nxnative hle_net.c:12, oot ov_system.c:1007");
HOS_OVERRIDE("_ZN2nn4nifm18IsNetworkAvailableEv", ov_zero, HOS_OFFLINE,
             "false: the host has no network; nxnative hle_net.c:30, oot ov_system.c:1005");
HOS_OVERRIDE("_ZN2nn4nifm28IsAnyInternetRequestAcceptedENS0_8ClientIdE", ov_zero, HOS_OFFLINE,
             "false: no internet request is accepted; oot ov_system.c:1003");
HOS_OVERRIDE("_ZN2nn4nifm21IsIpAddressConflictedEv", ov_zero, HOS_OFFLINE,
             "false: there is no address to conflict; oot ov_system.c:1004");
HOS_OVERRIDE("_ZN2nn4nifm31HandleNetworkRequestErrorResultEv", ov_zero, HOS_OFFLINE,
             "false: not resolved, no system network screen; nxnative hle_net.c:32");
HOS_OVERRIDE("_ZN2nn4nifm22GetCurrentProxySettingEPNS0_12ProxySettingE", ov_unavailable, HOS_OFFLINE,
             "fails: no connection, so no proxy setting; the output is left alone; nxnative hle_net.c:34");
HOS_OVERRIDE("_ZN2nn4nifm19SetLocalNetworkModeEb", ov_nothing, HOS_STUB,
             "accepted, no effect: the host has no local network; nxnative hle_offline.c:137, oot ov_system.c:1006");
HOS_OVERRIDE("_ZN2nn4nifm11GetClientIdEv", ov_zero, HOS_STUB,
             "client id 0: the host has no nifm client; oot ov_system.c:1010");

// void SubmitNetworkRequestAndWait(): returns at once with the request not accepted; IsNetworkAvailable stays false
static void ov_nifm_submit_wait(Cpu *c) {
    (void)c;
    __atomic_store_n(&nifm_hold, 0, __ATOMIC_SEQ_CST);
}
HOS_OVERRIDE("_ZN2nn4nifm27SubmitNetworkRequestAndWaitEv", ov_nifm_submit_wait, HOS_OFFLINE,
             "returns at once with the request refused, IsNetworkAvailable stays false; nxnative hle_offline.c:138 (no "
             "reference run exercised the waiting form: trap T4)");

// SubmitNetworkRequest, IsNetworkRequestOnHold, CancelNetworkRequest: on a console the nifm service evaluates a
// request as it serves the queries, so the first query after a submission finds it on hold and the next finds it
// settled (refused). A caller that polls this form needs the first answer (the services study); this game
// imports the waiting form.
static void ov_nifm_submit(Cpu *c) {
    (void)c;
    __atomic_store_n(&nifm_hold, 1, __ATOMIC_SEQ_CST);
}
HOS_OVERRIDE("_ZN2nn4nifm20SubmitNetworkRequestEv", ov_nifm_submit, HOS_OFFLINE,
             "the request is on hold for the first query, then refused; nxnative hle_net.c:23");

static void ov_nifm_on_hold(Cpu *c) {
    int h = __atomic_load_n(&nifm_hold, __ATOMIC_SEQ_CST);
    while (h > 0 && !__atomic_compare_exchange_n(&nifm_hold, &h, h - 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
    }
    ret_x(c, h > 0);
}
HOS_OVERRIDE("_ZN2nn4nifm22IsNetworkRequestOnHoldEv", ov_nifm_on_hold, HOS_OFFLINE,
             "true for the first query after a submission, then false; nxnative hle_net.c:25");

HOS_OVERRIDE("_ZN2nn4nifm20CancelNetworkRequestEv", ov_nifm_submit_wait, HOS_OFFLINE,
             "drops the pending request; nxnative hle_net.c:24, oot ov_system.c:1009");

// ==== nn::nsd
// void GetEnvironmentIdentifier(EnvironmentIdentifier *out): char[8], the server environment; "lp1" is production. The
// detail::ipc form asks the nsd:a service, which the host does not run.
static void ov_nsd_environment(Cpu *c) {
    static const char id[8] = "lp1";
    out_put(c, "nn::nsd::GetEnvironmentIdentifier", arg_x(c, 0), id, sizeof id);
    ret_x(c, 0);
}
HOS_OVERRIDE("_ZN2nn3nsd24GetEnvironmentIdentifierEPNS0_21EnvironmentIdentifierE", ov_nsd_environment, HOS_OFFLINE,
             "\"lp1\", the retail environment; nxnative hle_offline.c:151, oot ov_system.c:360-366");
HOS_OVERRIDE("_ZN2nn3nsd6detail3ipc24GetEnvironmentIdentifierEPNS0_21EnvironmentIdentifierE", ov_nsd_environment,
             HOS_OFFLINE, "\"lp1\" without asking nsd:a; oot ov_system.c:954");

HOS_OVERRIDE("_ZN2nn3nsd37IsChangeEnvironmentIdentifierDisabledEv", ov_one, HOS_OFFLINE,
             "true: a retail console cannot change its environment; the donor's ipc form says true "
             "(oot ov_system.c:368-373), its public form false (953)");

// Result IsChangeEnvironmentIdentifierDisabled(bool *out)
static void ov_nsd_change_disabled(Cpu *c) {
    uint8_t yes = 1;
    out_put(c, "nn::nsd::detail::ipc::IsChangeEnvironmentIdentifierDisabled", arg_x(c, 0), &yes, sizeof yes);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn3nsd6detail3ipc37IsChangeEnvironmentIdentifierDisabledEPb", ov_nsd_change_disabled, HOS_OFFLINE,
             "true without asking nsd:a; oot ov_system.c:955");

HOS_OVERRIDE("_ZN2nn3nsd9ResolveExEPNS0_4FqdnERKS1_", ov_unavailable, HOS_OFFLINE,
             "fails as a console without a network does; the output is left alone; nxnative hle_misc.c:177");
HOS_OVERRIDE("_ZN2nn3nsd7ResolveEPNS0_4FqdnERKS1_", ov_unavailable, HOS_OFFLINE,
             "fails as a console without a network does; the output is left alone; nxnative hle_misc.c:176");

// ==== nn::ssl: a context is a session of the local ssl service, so it exists with no network; a connection does not
HOS_OVERRIDE("_ZN2nn3ssl10InitializeEv", ov_zero, HOS_OFFLINE,
             "success, no session: the online library's global start needs it; nxnative hle_net.c:35,219, "
             "oot ov_system.c:936");
HOS_OVERRIDE("_ZN2nn3ssl8FinalizeEv", ov_zero, HOS_OFFLINE,
             "success, pairs with Initialize; nxnative hle_net.c:36,220, oot ov_system.c:937");
HOS_OVERRIDE("_ZN2nn3ssl7Context6CreateENS1_10SslVersionE", ov_zero, HOS_OFFLINE,
             "success with no session behind it (a failure ends the game's online start-up); the object stays as "
             "constructed; nxnative hle_net.c:221");
HOS_OVERRIDE("_ZN2nn3ssl7Context7DestroyEv", ov_zero, HOS_OFFLINE,
             "success, pairs with Create; nxnative hle_net.c:222");

// Result Context::ImportClientPki(unsigned long *id, ...), ImportServerPki(...), ImportCrl(...): nothing is imported
static void ov_ssl_import(Cpu *c) {
    uint64_t none = 0;
    out_put(c, "nn::ssl::Context::Import", arg_x(c, 1), &none, sizeof none);
    ret_x(c, HOS_R_UNAVAILABLE);
}
#define WHY_PKI "fails, id 0: no certificate store without the ssl service; nxnative hle_offline.c:130-132"
HOS_OVERRIDE("_ZN2nn3ssl7Context15ImportClientPkiEPmPKcS4_jj", ov_ssl_import, HOS_OFFLINE, WHY_PKI);
HOS_OVERRIDE("_ZN2nn3ssl7Context15ImportServerPkiEPmPKcjNS0_17CertificateFormatE", ov_ssl_import, HOS_OFFLINE, WHY_PKI);
HOS_OVERRIDE("_ZN2nn3ssl7Context9ImportCrlEPmPKcj", ov_ssl_import, HOS_OFFLINE, WHY_PKI);

// Result BuiltInManager::GetBuiltInCertificates(BuiltInCertificateInfoBase **out, ...) and
// GetBuiltInCertificateBufSize(unsigned *size, ...): static members, the first argument is the output
static void ov_ssl_builtin_list(Cpu *c) {
    uint64_t none = 0;
    out_put(c, "nn::ssl::BuiltInManager::GetBuiltInCertificates", arg_x(c, 0), &none, sizeof none);
    ret_x(c, HOS_R_UNAVAILABLE);
}
HOS_OVERRIDE(
    "_ZN2nn3ssl14BuiltInManager22GetBuiltInCertificatesEPPNS0_26BuiltInCertificateInfoBaseEPhjPNS0_15CaCertificateIdEj",
    ov_ssl_builtin_list, HOS_OFFLINE,
    "fails, no list: the built-in certificates are a service's; nxnative hle_offline.c:133");

static void ov_ssl_builtin_size(Cpu *c) {
    uint32_t none = 0;
    out_put(c, "nn::ssl::BuiltInManager::GetBuiltInCertificateBufSize", arg_x(c, 0), &none, sizeof none);
    ret_x(c, HOS_R_UNAVAILABLE);
}
HOS_OVERRIDE("_ZN2nn3ssl14BuiltInManager28GetBuiltInCertificateBufSizeEPjPNS0_15CaCertificateIdEj", ov_ssl_builtin_size,
             HOS_OFFLINE, "fails, size 0: the built-in certificates are a service's; nxnative hle_net.c:225");

// Connection: the calls the game's nnssl wrappers reach. A connection needs a network, so none is made.
#define WHY_SSL_FAIL "fails: a connection needs a network; nxnative hle_net.c:227-239"
#define SSL_FAIL(sym) HOS_OVERRIDE(sym, ov_unavailable, HOS_OFFLINE, WHY_SSL_FAIL)
SSL_FAIL("_ZN2nn3ssl10Connection6CreateEPNS0_7ContextE");
SSL_FAIL("_ZN2nn3ssl10Connection7DestroyEv");
SSL_FAIL("_ZN2nn3ssl10Connection11DoHandshakeEPjS2_");
SSL_FAIL("_ZN2nn3ssl10Connection16GetNextAlpnProtoEPNS1_14AlpnProtoStateEPhPjj");
SSL_FAIL("_ZNK2nn3ssl10Connection19GetServerCertDetailEPNS1_16ServerCertDetailEPKcj");
SSL_FAIL("_ZN2nn3ssl10Connection18GetVerifyCertErrorEPNS_6ResultE");
SSL_FAIL("_ZN2nn3ssl10Connection4PeekEPcPij");
SSL_FAIL("_ZN2nn3ssl10Connection7PendingEPi");
SSL_FAIL("_ZN2nn3ssl10Connection4ReadEPcPij");
SSL_FAIL("_ZN2nn3ssl10Connection11SetHostNameEPKcj");
SSL_FAIL("_ZN2nn3ssl10Connection9SetIoModeENS1_6IoModeE");
SSL_FAIL("_ZN2nn3ssl10Connection16SetNextAlpnProtoEPKhj");
SSL_FAIL("_ZN2nn3ssl10Connection9SetOptionENS1_10OptionTypeEb");
SSL_FAIL("_ZN2nn3ssl10Connection19SetServerCertBufferEPcj");
SSL_FAIL("_ZN2nn3ssl10Connection19SetSessionCacheModeENS1_16SessionCacheModeE");
SSL_FAIL("_ZN2nn3ssl10Connection19SetSocketDescriptorEi");
SSL_FAIL("_ZN2nn3ssl10Connection15SetVerifyOptionENS1_12VerifyOptionE");
SSL_FAIL("_ZN2nn3ssl10Connection5WriteEPKcPij");
