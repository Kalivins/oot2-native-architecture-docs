// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Load the player's plain modules into the runtime's address space and report what it cost: relocations by type,
// imports by binding, then per region what is mapped, committed and resident, and the process's private commit and
// working set after start-up, after loading, and after a dispatch table over the modules' whole span.
//
// Built and run by tools/loadprobe/load_probe.py against a layout.h for the player's modules:
//   load_probe <path> <sha256> [<path> <sha256> ...]     in layout order (layout.h MOD_LIST)
// No lifted code exists here, so the dispatch table gets one stand-in function at the start and end of each module.
// The runtime needs each module's end marker past its functions (ticket 92): probe_end.c, linked right after this
// file as a module's <mod>_table.c is linked after its lifted code, holds it, and the probe checks the order first.
#include <stdio.h>
#include <stdlib.h>

#include "layout.h"
#include "oot2/loader.h"
#include "plat.h"

#define X(name, NAME) {#name, NULL, MOD_##NAME##_BASE, MOD_##NAME##_SIZE, NULL},
static RtModuleFile files[] = {MOD_LIST(X)};
#undef X

static void stand_in(Cpu *c) { (void)c; }
void probe_lifted_end(Cpu *c);   // probe_end.c

static double mib(uint64_t v) { return (double)v / (1 << 20); }

static void report(const char *when) {
    if (!rt_mem_ready()) {
        PlatProcMem p;
        plat_process_memory(&p);
        printf("memory %s: private %.1f MiB, working set %.1f MiB\n", when, mib(p.private_bytes), mib(p.working_set));
        return;
    }
    RtMemStats s;
    rt_mem_measure(&s);
    printf("memory %s: private %.1f MiB, working set %.1f MiB (peak %.1f MiB)\n", when, mib(s.process_private),
           mib(s.process_working_set), mib(s.process_peak_working_set));
    for (int r = 0; r < RT_MEM_REGIONS; r++)
        printf("  region %-7s %#llx+%#llx: mapped %.1f MiB, committed %.1f MiB, resident %.1f MiB, %llu touch commits\n",
               s.r[r].name, (unsigned long long)s.r[r].base, (unsigned long long)s.r[r].size, mib(s.r[r].mapped),
               mib(s.r[r].committed), mib(s.r[r].resident), (unsigned long long)s.r[r].touch_commits);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if ((uintptr_t)stand_in >= (uintptr_t)probe_lifted_end) {   // rt_dispatch_init would refuse it, after the load
        fprintf(stderr, "load_probe: the end marker probe_lifted_end lies below the stand-in function: probe_end.c "
                        "must be linked after load_probe.c\n");
        return 3;
    }
    if (argc != 1 + 2 * MOD_COUNT) {
        fprintf(stderr, "usage: load_probe <path> <sha256> per module of the layout (%d modules)\n", MOD_COUNT);
        return 2;
    }
    for (int i = 0; i < MOD_COUNT; i++) files[i].path = argv[1 + 2 * i], files[i].sha256 = argv[2 + 2 * i];
    report("at start");
    RtMemConfig cfg = rt_mem_default_config();
    rt_mem_init(&cfg);
    report("after rt_mem_init");
    uint64_t t0 = plat_now_ns();
    rt_load_modules(files, MOD_COUNT);
    uint64_t t1 = plat_now_ns();
    printf("load: %.0f ms (files mapped, whole-file and segment SHA-256 checked, relocated, bound, sealed)\n",
           (double)(t1 - t0) / 1e6);
    rt_load_report();
    for (int i = 0; i < rt_loaded_count(); i++) {
        const RtLoadStats *st = rt_load_stats(i);
        printf("module %s: relocations %llu (RELATIVE %llu, RELR %llu, ABS64 %llu, GLOB_DAT %llu, JUMP_SLOT %llu); "
               "imports %u (to modules", rt_loaded(i)->name,
               (unsigned long long)(st->relative + st->relr + st->abs64 + st->glob_dat + st->jump_slot),
               (unsigned long long)st->relative, (unsigned long long)st->relr, (unsigned long long)st->abs64,
               (unsigned long long)st->glob_dat, (unsigned long long)st->jump_slot, st->imports);
        for (int j = 0; j < rt_loaded_count(); j++) printf(" %s %u", rt_loaded(j)->name, st->imports_to[j]);
        printf(", host %u, weak %u, missing %u); definitions replaced %u\n", st->imports_to_host, st->imports_weak,
               st->imports_missing, st->replaced);
    }
    report("after loading");

    static uint64_t addrs[MOD_COUNT][2];
    static GuestFn fns[MOD_COUNT][2];
    RtModule mods[MOD_COUNT];
    for (int i = 0; i < MOD_COUNT; i++) {
        const RtLoaded *m = rt_loaded(i);
        addrs[i][0] = m->base;
        addrs[i][1] = m->base + ((m->text_size - 4) & ~3ull);
        fns[i][0] = fns[i][1] = stand_in;
        mods[i] = (RtModule){m->name, m->base, m->size, addrs[i], fns[i], 2, (const void *)probe_lifted_end};
    }
    rt_dispatch_init(mods, MOD_COUNT);
    RtDispatchStats d = rt_dispatch_stats();
    printf("dispatch table: guest %#llx..%#llx, %llu slots, %.1f MiB\n", (unsigned long long)d.lo,
           (unsigned long long)d.hi, (unsigned long long)d.slots, mib(d.table_bytes));
    report("after the dispatch table");
    return 0;
}
