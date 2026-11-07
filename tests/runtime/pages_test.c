// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the page ranges of runtime/core/memory.c (lever L1: counting and releasing, architecture D19, D27),
// driven by tests/test_runtime_pages.py: `pages_test <case>`. Each case runs in its own process, with OOT2_PAGE_RELEASE
// set (or not) by the driver. A passing case prints "ok <case>"; a failing case must end in one named fatal. The pool is
// synthetic: the heap and alias regions of a runtime that maps nothing the game owns, filled with numbers made here.
//
// The functions below stand for lifted guest code (guest_read, guest_write, guest_end, registered in a function table)
// and for runtime or back-end code (host_read32, host_write32, in pages_host.c, in no table). A test touches a page that
// is armed or released only through these two kinds of function: the table gives a function's start, so the rest of
// this file lies in the extent of the last guest function.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/loader.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define MiB (1ull << 20)
#define MB MOD_MAIN_BASE
#define P 0x1000ull   // one page

static Cpu c;
static int g_id = -1;   // the range the stop hook reports

// ---- runtime or back-end code: in no function table, in pages_host.c, linked before this file
uint32_t host_read32(uint64_t a);
void host_write32(uint64_t a, uint32_t v);
// Runtime code linked right after the module's end marker (endmark.c): not guest code, though it follows the last
// function of the table (guest_end) closer than 256 KiB.
uint32_t host_read32_after_end(uint64_t a);

// ---- lifted guest code, as the function table sees it
__attribute__((noinline)) static void guest_read(Cpu *cpu) { cpu->x[1] = LD32(cpu->x[0]) + 0x11; }
__attribute__((noinline)) static void guest_write(Cpu *cpu) { ST32(cpu->x[0], cpu->x[1]); cpu->x[2] = 0x22; }
__attribute__((noinline)) static void guest_end(Cpu *cpu) { cpu->x[2] = 0xE0; }

enum { FN_READ = MB + 0x200, FN_WRITE = MB + 0x300, FN_END = MB + 0x400 };

static uint32_t gread(uint64_t a) {
    c.x[0] = a;
    rt_call(&c, FN_READ);
    return (uint32_t)c.x[1] - 0x11;
}

static void gwrite(uint64_t a, uint32_t v) {
    c.x[0] = a;
    c.x[1] = v;
    rt_call(&c, FN_WRITE);
}

// ---- the owner
typedef struct Owner {
    int busy, busy_calls;
    uint64_t busy_addr, busy_size;
    uint64_t written[16];
    int nwritten;
} Owner;

static int o_busy(void *u, uint64_t a, uint64_t n) {
    Owner *o = u;
    o->busy_calls++;
    o->busy_addr = a;
    o->busy_size = n;
    return o->busy;
}

static void o_written(void *u, uint64_t pg) {
    Owner *o = u;
    if (o->nwritten < 16) o->written[o->nwritten] = pg;
    o->nwritten++;
}

static Owner own;
static RtPageOwner owner(void) { return (RtPageOwner){"pool", &own, o_busy, o_written, NULL}; }

// ---- the synthetic pool
static RtMemConfig test_config(void) {
    return (RtMemConfig){.pool_size = 256 * MiB, .heap_region_size = 64 * MiB, .alias_region_size = 1024 * MiB,
                         .stack_region_size = 64 * MiB, .system_resource_size = MiB};
}

static void stop_hook(int code, const char *line) {
    (void)code, (void)line;
    RtPageStats s;
    if (g_id >= 0 && rt_pages_stats(g_id, &s))
        fprintf(stderr, "census: pages range %d mode %d: released reads guest %llu host %llu, released now %llu\n", g_id,
                s.mode, (unsigned long long)s.guest_reads, (unsigned long long)s.host_reads,
                (unsigned long long)s.released);
}

static uint64_t svc(uint32_t imm, uint64_t x0, uint64_t x1, uint64_t x2) {
    c.x[0] = x0, c.x[1] = x1, c.x[2] = x2;
    rt_svc(&c, imm, 0);
    return c.x[0];
}

static uint64_t info(uint32_t id) {
    uint64_t v = 0;
    CHECK(rt_mem_info(id, &v));
    return v;
}

static RtMemRegion region(int r) {
    RtMemStats s;
    rt_mem_measure(&s);
    return s.r[r];
}

static RtPageStats stats(int id) {
    RtPageStats s;
    CHECK(rt_pages_stats(id, &s));
    return s;
}

// The heap, 4 MiB of it, the function table and the stop hook.
static uint64_t setup(void) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    static const uint64_t addrs[] = {FN_READ, FN_WRITE, FN_END};
    static const GuestFn fns[] = {guest_read, guest_write, guest_end};
    static const RtModule mods[] = {{"main", MB, MOD_MAIN_SIZE, addrs, fns, 3, TEST_END}};
    rt_set_stop_hook(stop_hook);
    rt_dispatch_init(mods, 1);
    CHECK(svc(0x01, 0, 4 * MiB, 0) == 0);
    memset(&own, 0, sizeof own);
    return info(4);
}

// Write a number in every page of the 16 pages from base: one first touch commits the chunk.
static void fill(uint64_t base, uint32_t tag) {
    for (uint64_t i = 0; i < 16; i++) ST32(base + i * P, tag + (uint32_t)i);
}

// ---- passing cases
// Before the function tables are sealed a range is refused loudly: the host index must not be built in a fault.
static void case_unsealed(void) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    CHECK(svc(0x01, 0, 4 * MiB, 0) == 0);
    RtPageOwner ow = owner();
    rt_pages_watch(info(4) + 0x100000, 0x10000, RT_PAGES_COUNTING, &ow);
}

static void case_off(void) {
    uint64_t heap = setup();
    RtPageOwner ow = owner();
    CHECK(rt_pages_level() == RT_PAGES_OFF);
    CHECK(rt_pages_watch(heap, 0x10000, RT_PAGES_COUNTING, &ow) == -1);
    CHECK(rt_pages_watch(heap, 0x10000, RT_PAGES_RELEASING, &ow) == -1);
    // first-touch commit as it was: the whole 64 KiB chunk, once
    ST8(heap + 0x1000, 1);
    RtMemRegion h = region(RT_MEM_HEAP);
    CHECK(h.committed == 0x10000 && h.touch_commits == 1);
    CHECK(LD8(heap + 0x5000) == 0 && region(RT_MEM_HEAP).touch_commits == 1);
    ST8(heap + 0x20000, 1);
    CHECK(region(RT_MEM_HEAP).committed == 0x20000 && region(RT_MEM_HEAP).touch_commits == 2);
}

static void case_count(void) {
    uint64_t heap = setup(), base = heap + 0x100000;
    RtPageOwner ow = owner();
    CHECK(rt_pages_level() == RT_PAGES_COUNTING);
    CHECK(rt_pages_watch(base, 0x10000, RT_PAGES_RELEASING, &ow) == -1);   // the switch allows counting only
    int id = rt_pages_watch(base, 0x10000, RT_PAGES_COUNTING, &ow);
    CHECK(id >= 0);
    CHECK(rt_pages_watch(base + 0x8000, 0x10000, RT_PAGES_COUNTING, &ow) == -1);   // overlaps
    CHECK(rt_pages_watch(base + 0x20000 + 0x10, 0x1000, RT_PAGES_COUNTING, &ow) == -1);   // not aligned
    CHECK(rt_pages_watch(MB, 0x1000, RT_PAGES_COUNTING, &ow) == -1);   // not heap or alias
    CHECK(rt_pages_arm(id, base, 0x10000) == 0);   // nothing committed yet: nothing to arm
    fill(base, 0xA000);
    CHECK(region(RT_MEM_HEAP).committed == 0x10000 && region(RT_MEM_HEAP).touch_commits == 1);
    CHECK(rt_pages_arm(id, base, 0x10000) == 16);
    RtPageStats s = stats(id);
    CHECK(s.mode == RT_PAGES_COUNTING && s.armed == 16 && s.arms == 16);

    CHECK(gread(base + 3 * P) == 0xA003);          // a read by lifted code: counted, restored, and the value is right
    CHECK(host_read32(base + 5 * P) == 0xA005);    // a read by runtime code
    gwrite(base + 7 * P + 8, 0xBEEF);              // a write by lifted code
    s = stats(id);
    CHECK(s.guest_reads == 1 && s.host_reads == 1 && s.guest_writes == 1 && s.host_writes == 0 && s.armed == 13);
    CHECK(gread(base + 3 * P) == 0xA003 && host_read32(base + 5 * P) == 0xA005 && LD32(base + 7 * P + 8) == 0xBEEF);
    s = stats(id);
    CHECK(s.guest_reads == 1 && s.host_reads == 1 && s.armed == 13);   // restored pages do not fault again
    CHECK(region(RT_MEM_HEAP).committed == 0x10000 && region(RT_MEM_HEAP).touch_commits == 1);   // never a first touch

    CHECK(rt_pages_arm(id, base, 0x10000) == 3);   // arming again takes the three restored pages back
    host_write32(base + 9 * P, 7);
    CHECK(rt_mem_commit_io(base + 11 * P + 0x10, 0x20, 0));   // host I/O is a host access
    s = stats(id);
    CHECK(s.host_writes == 1 && s.host_reads == 2 && s.armed == 14 && s.arms == 19);
    rt_pages_unwatch(id);   // the rest get their access back
    CHECK(LD32(base + 13 * P) == 0xA00D && LD32(base + 15 * P) == 0xA00F && LD32(base + 9 * P) == 7);
    CHECK(!rt_pages_stats(id, &s));
    CHECK(region(RT_MEM_HEAP).touch_commits == 1);
}

// An armed page beside a first touch: the chunk's commit leaves it armed.
static void case_count_neighbour(void) {
    setup();
    uint64_t alias = info(2);
    RtPageOwner ow = owner();
    CHECK(svc(0x2C, alias, 4 * P, 0) == 0);
    int id = rt_pages_watch(alias, 0x10000, RT_PAGES_COUNTING, &ow);
    CHECK(id >= 0);
    for (uint64_t i = 0; i < 4; i++) ST32(alias + i * P, 0x7000 + (uint32_t)i);
    CHECK(rt_pages_arm(id, alias, 4 * P) == 4);
    CHECK(svc(0x2C, alias + 4 * P, 12 * P, 0) == 0);
    ST32(alias + 8 * P, 1);   // a first touch whose chunk holds four armed pages
    RtMemRegion a = region(RT_MEM_ALIAS);
    CHECK(a.touch_commits == 2 && a.committed == 0x10000);
    CHECK(stats(id).armed == 4 && stats(id).guest_reads == 0);
    CHECK(gread(alias + P) == 0x7001 && stats(id).guest_reads == 1 && stats(id).armed == 3);
    CHECK(host_read32(alias + 3 * P) == 0x7003 && stats(id).host_reads == 1);
}

// MapMemory moves committed pages by copy, under the exclusive lock: armed pages must not fault there.
static void case_move(void) {
    uint64_t heap = setup(), base = heap + 0x100000, dst = info(14) + 0x100000;
    RtPageOwner ow = owner();
    int id = rt_pages_watch(base, 0x10000, RT_PAGES_COUNTING, &ow);
    CHECK(id >= 0);
    fill(base, 0x9000);
    CHECK(rt_pages_arm(id, base, 0x10000) == 16);
    CHECK(svc(0x04, dst, base, 0x10000) == 0);
    RtPageStats s = stats(id);
    CHECK(s.armed == 0 && s.host_reads == 0);   // the copy lifted the protection, and was not counted as a read
    for (uint64_t i = 0; i < 16; i++) CHECK(LD32(dst + i * P) == 0x9000 + (uint32_t)i);
    CHECK(svc(0x05, dst, base, 0x10000) == 0);
    for (uint64_t i = 0; i < 16; i++) CHECK(LD32(base + i * P) == 0x9000 + (uint32_t)i);
    CHECK(rt_pages_arm(id, base, 0x10000) == 16);   // the range is still usable
    CHECK(gread(base + 2 * P) == 0x9002 && stats(id).guest_reads == 1);
}

static void case_release(void) {
    uint64_t heap = setup(), base = heap + 0x100000;
    RtPageOwner ow = owner();
    CHECK(rt_pages_level() == RT_PAGES_RELEASING);
    int id = rt_pages_watch(base, 0x10000, RT_PAGES_RELEASING, &ow);
    CHECK(id >= 0 && rt_pages_arm(id, base, 0x10000) == 0);   // a releasing range does not count
    fill(base, 0xC000);
    CHECK(region(RT_MEM_HEAP).committed == 0x10000);
    CHECK(rt_pages_release(id, base + 2 * P, 2 * P) == 2);
    CHECK(own.busy_calls == 1 && own.busy_addr == base + 2 * P && own.busy_size == 2 * P);
    CHECK(region(RT_MEM_HEAP).committed == 0x10000 - 2 * P);   // committed bytes follow the release
    RtPageStats s = stats(id);
    CHECK(s.released == 2 && s.releases == 2);
    // the pages beside the released ones keep their bytes and cost no fault
    CHECK(LD32(base + 1 * P) == 0xC001 && LD32(base + 4 * P) == 0xC004 && LD32(base + 0) == 0xC000);
    CHECK(region(RT_MEM_HEAP).touch_commits == 1 && own.nwritten == 0);

    gwrite(base + 2 * P + 0x10, 0x77);   // a write: recommitted, the owner called with the page, one page only
    CHECK(own.nwritten == 1 && own.written[0] == base + 2 * P);
    CHECK(region(RT_MEM_HEAP).committed == 0x10000 - P && region(RT_MEM_HEAP).touch_commits == 1);
    CHECK(LD32(base + 2 * P + 0x10) == 0x77 && LD32(base + 2 * P) == 0);   // zero-filled, then written
    s = stats(id);
    CHECK(s.recommits == 1 && s.guest_writes == 1 && s.released == 1);
    host_write32(base + 3 * P + 4, 9);   // the same from runtime code
    CHECK(own.nwritten == 2 && own.written[1] == base + 3 * P && LD32(base + 3 * P + 4) == 9);
    s = stats(id);
    CHECK(s.recommits == 2 && s.host_writes == 1 && s.released == 0);
    CHECK(region(RT_MEM_HEAP).committed == 0x10000);
    CHECK(LD32(base + 1 * P) == 0xC001 && LD32(base + 4 * P) == 0xC004);   // the neighbours are as they were

    // releasing a released or an uncommitted page releases nothing; a second write does not call the owner again
    CHECK(rt_pages_release(id, base, 0x10000) == 16);
    CHECK(rt_pages_release(id, base, 0x10000) == 0);
    CHECK(region(RT_MEM_HEAP).committed == 0);
    gwrite(base + 6 * P, 1);
    gwrite(base + 6 * P + 4, 2);
    CHECK(own.nwritten == 3 && own.written[2] == base + 6 * P && region(RT_MEM_HEAP).committed == P);

    // a heap that shrinks takes its pages' state with them: the page reads zero, as a first touch, afterwards
    int id2 = rt_pages_watch(heap + 0x300000, 0x10000, RT_PAGES_RELEASING, &ow);
    CHECK(id2 >= 0);
    fill(heap + 0x300000, 0xD000);
    CHECK(rt_pages_release(id2, heap + 0x300000, 0x4000) == 4 && stats(id2).released == 4);
    CHECK(svc(0x01, 0, 2 * MiB, 0) == 0 && stats(id2).released == 0);
    CHECK(svc(0x01, 0, 4 * MiB, 0) == 0);
    CHECK(LD32(heap + 0x300000) == 0 && LD32(heap + 0x305000) == 0);
    rt_pages_unwatch(id2);
}

static void case_busy(void) {
    uint64_t heap = setup(), base = heap + 0x100000;
    RtPageOwner ow = owner();
    int id = rt_pages_watch(base, 0x10000, RT_PAGES_RELEASING, &ow);
    CHECK(id >= 0);
    fill(base, 0xB000);
    own.busy = 1;   // the owner holds the range
    CHECK(rt_pages_release(id, base, 0x10000) == 0);
    CHECK(own.busy_calls == 1 && stats(id).busy_refusals == 1 && stats(id).released == 0);
    CHECK(region(RT_MEM_HEAP).committed == 0x10000);
    CHECK(gread(base + 4 * P) == 0xB004 && host_read32(base + 9 * P) == 0xB009);   // still readable
    own.busy = 0;
    CHECK(rt_pages_release(id, base, 0x10000) == 16);
    CHECK(own.busy_calls == 2 && stats(id).busy_refusals == 1 && region(RT_MEM_HEAP).committed == 0);
}

static void case_neighbour(void) {
    setup();
    uint64_t alias = info(2);
    RtPageOwner ow = owner();
    // The chunk's first four pages are mapped and touched; two of them are released; then the rest of the chunk is
    // mapped and a first touch there commits the chunk around it.
    CHECK(svc(0x2C, alias, 4 * P, 0) == 0);
    int id = rt_pages_watch(alias, 0x10000, RT_PAGES_RELEASING, &ow);
    CHECK(id >= 0);
    ST32(alias, 0x1111);
    ST32(alias + P, 0x2222);
    ST32(alias + 2 * P, 0x3333);
    CHECK(region(RT_MEM_ALIAS).committed == 4 * P && region(RT_MEM_ALIAS).touch_commits == 1);
    CHECK(rt_pages_release(id, alias + P, 2 * P) == 2);
    CHECK(region(RT_MEM_ALIAS).committed == 2 * P);
    CHECK(svc(0x2C, alias + 4 * P, 12 * P, 0) == 0);
    ST32(alias + 8 * P, 0x4444);   // a first touch whose chunk holds two released pages
    RtMemRegion a = region(RT_MEM_ALIAS);
    CHECK(a.touch_commits == 2 && a.committed == 0x10000 - 2 * P);   // the released pages were not taken for it
    CHECK(stats(id).released == 2 && own.nwritten == 0);
    CHECK(LD32(alias) == 0x1111 && LD32(alias + 8 * P) == 0x4444 && LD32(alias + 3 * P) == 0);
    gwrite(alias + P, 0x55);   // still released: this write recommits it, and only it
    CHECK(own.nwritten == 1 && own.written[0] == alias + P);
    a = region(RT_MEM_ALIAS);
    CHECK(a.touch_commits == 2 && a.committed == 0x10000 - P && LD32(alias + P) == 0x55);
    CHECK(LD32(alias) == 0x1111 && stats(id).released == 1);
    // and the other way: a fault on a released page commits no neighbour. Page 2 is still released.
    host_write32(alias + 2 * P + 4, 6);
    CHECK(region(RT_MEM_ALIAS).committed == 0x10000 && region(RT_MEM_ALIAS).touch_commits == 2);
    CHECK(LD32(alias + 2 * P) == 0);
}

// After a decommit and a recommit the write watch must still tell the written pages (Windows; elsewhere the platform
// has no watch and says so).
static void case_writewatch(void) {
    uint64_t heap = setup(), base = heap + 0x100000;
    RtPageOwner ow = owner();
    int id = rt_pages_watch(base, 0x10000, RT_PAGES_RELEASING, &ow);
    CHECK(id >= 0);
    fill(base, 0xE000);
    uint64_t pages[64];
    uint64_t n = rt_mem_written(heap, 4 * MiB, pages, 64, 1);   // reset
    if (n == UINT64_MAX) {
        printf("skip writewatch: write watch not available on this platform\n");
        exit(0);
    }
    CHECK(rt_mem_written(heap, 4 * MiB, pages, 64, 0) == 0);
    CHECK(rt_pages_release(id, base + 4 * P, 2 * P) == 2);
    CHECK(rt_mem_written(heap, 4 * MiB, pages, 64, 0) == 0);   // a release is not a write
    gwrite(base + 4 * P + 8, 1);                               // recommit by a write: the page is written
    n = rt_mem_written(heap, 4 * MiB, pages, 64, 0);
    CHECK(n == 1 && pages[0] == base + 4 * P);
    CHECK(rt_mem_written(heap, 4 * MiB, pages, 64, 1) == 1);   // reset
    CHECK(rt_mem_written(heap, 4 * MiB, pages, 64, 0) == 0);
    // a page written, not reset, then released and written again after its recommit: it is in the answer
    ST32(base + 8 * P, 5);
    CHECK(rt_pages_release(id, base + 8 * P, P) == 1);
    host_write32(base + 8 * P, 6);
    n = rt_mem_written(heap, 4 * MiB, pages, 64, 0);
    CHECK(n == 1 && pages[0] == base + 8 * P);
    // A release clears the page's mark (the owner has uploaded the page by then; reset the watch before releasing what
    // must be kept): a page written, not reset, then released is no longer in the answer, and a write after the recommit
    // puts it back. The page written and not released before stays throughout.
    ST32(base + 10 * P, 5);
    CHECK(rt_mem_written(heap, 4 * MiB, pages, 64, 0) == 2);
    CHECK(rt_pages_release(id, base + 10 * P, P) == 1);
    n = rt_mem_written(heap, 4 * MiB, pages, 64, 0);
    CHECK(n == 1 && pages[0] == base + 8 * P);
    gwrite(base + 10 * P, 1);
    n = rt_mem_written(heap, 4 * MiB, pages, 64, 0);
    CHECK(n == 2);
    ST32(base + 9 * P, 5);   // a neighbour that was never released
    CHECK(rt_mem_written(heap, 4 * MiB, pages, 64, 0) == 3);
}

static void case_io(void) {
    uint64_t heap = setup(), base = heap + 0x100000;
    RtPageOwner ow = owner();
    int id = rt_pages_watch(base, 0x10000, RT_PAGES_RELEASING, &ow);
    CHECK(id >= 0);
    fill(base, 0xF000);
    CHECK(rt_pages_release(id, base + P, 2 * P) == 2);
    CHECK(!rt_mem_commit_io(base + P + 0x100, 8, 0));   // a read of a released page: refused, and counted
    CHECK(stats(id).host_reads == 1 && own.nwritten == 0 && region(RT_MEM_HEAP).committed == 0x10000 - 2 * P);
    CHECK(rt_mem_commit_io(base + 2 * P + 0xFF8, 16, 1));   // a write across pages 2 and 3: page 2 comes back
    CHECK(own.nwritten == 1 && own.written[0] == base + 2 * P && region(RT_MEM_HEAP).committed == 0x10000 - P);
    CHECK(rt_mem_commit_io(base, 0x1000, 0) && stats(id).released == 1);
}

// ---- failing cases: a read of a released page is the named fault
static void case_read_released(int guest, int after_end) {
    uint64_t heap = setup(), base = heap + 0x100000;
    RtPageOwner ow = owner();
    g_id = rt_pages_watch(base, 0x10000, RT_PAGES_RELEASING, &ow);
    CHECK(g_id >= 0);
    fill(base, 0xC000);
    CHECK(rt_pages_release(g_id, base + 2 * P, P) == 1);
    CHECK(gread(base + 1 * P) == 0xC001);   // a page beside it reads fine
    if (guest) printf("read %#x\n", gread(base + 2 * P + 8));
    else if (after_end) printf("read %#x\n", host_read32_after_end(base + 2 * P + 8));
    else printf("read %#x\n", host_read32(base + 2 * P + 8));
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: pages_test <case>\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "off")) case_off();
    else if (!strcmp(cs, "count")) case_count();
    else if (!strcmp(cs, "count_neighbour")) case_count_neighbour();
    else if (!strcmp(cs, "move")) case_move();
    else if (!strcmp(cs, "release")) case_release();
    else if (!strcmp(cs, "busy")) case_busy();
    else if (!strcmp(cs, "neighbour")) case_neighbour();
    else if (!strcmp(cs, "writewatch")) case_writewatch();
    else if (!strcmp(cs, "io")) case_io();
    else if (!strcmp(cs, "unsealed")) case_unsealed();
    else if (!strcmp(cs, "read_released_guest")) case_read_released(1, 0);
    else if (!strcmp(cs, "read_released_host")) case_read_released(0, 0);
    else if (!strcmp(cs, "read_released_after_end")) case_read_released(0, 1);
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    printf("ok %s\n", cs);
    return 0;
}
