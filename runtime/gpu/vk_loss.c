// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The device-loss report of the Vulkan back end (architecture D17b; oot2/gpu_vk.h, oot2/gpu.h). Every failed Vulkan
// call of the back end comes to gpu_vk_failed. VK_ERROR_DEVICE_LOST writes a report and stops the run by name; any
// other result stops by name at once.
//
// The report is a text file in the game folder's logs folder (gpu_write_report): where the loss was met, the device and
// its driver, the settings, the timeline's values (handed out, issued, done, and the device's own counter when it can
// still be read), the memory heaps with their budget, what VK_EXT_device_fault describes (the description, every
// faulting address with its precision, every vendor record) and the last diagnostic checkpoints the queue reached, each
// with the value and kind of the submission it marks. The vendor's binary fault data, when the device offers it, is
// saved byte for byte beside the report (research 16 D2: DKB Recomp logged only its size). The first thread that meets
// the loss writes the report; another waits on the lock until the run ends, so no other fatal cuts the report short.
// When the file cannot be written, the report goes to the log instead and the stop says so.
//
// The fault query (VUID-vkGetDeviceFaultInfoEXT-device-07336) and the checkpoint query (-queue-03892) are valid only on a
// lost device, so the path is tested by replacing the Volk entry points of the failing call and of the two queries
// (tests/runtime/vk_test.c); a real GPU hang is never caused on purpose.
//
// The fault query is adapted from DKB Recomp runtime/vk_device.c at 9d8fa02 (vk_device_fault_report, GPL-3.0-or-later:
// the counts first, then the address and vendor records, the address types named, one report per process with other
// threads held on a lock). Changed here: the vendor's binary is fetched and saved, the checkpoints and the timeline are
// reported, everything goes to a file of its own, and the run stops here.
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "oot2/vk_rec.h"
#include "plat.h"

#define PATH_BYTES 4096

static PlatMutex g_folder_lock = PLAT_MUTEX_INIT;
static char g_folder[PATH_BYTES];

void gpu_set_report_folder(const char *dir) {
    plat_mutex_lock(&g_folder_lock);
    snprintf(g_folder, sizeof g_folder, "%s", dir ? dir : "");
    plat_mutex_unlock(&g_folder_lock);
}

static int make_folder(const char *path) {
#ifdef _WIN32
    return _mkdir(path) == 0 || errno == EEXIST;
#else
    return mkdir(path, 0777) == 0 || errno == EEXIST;
#endif
}

static void local_time(struct tm *out) {
    time_t now = time(NULL);
#ifdef _WIN32
    localtime_s(out, &now);
#else
    localtime_r(&now, out);
#endif
}

// fopen(path, "wbx"): never over a file that is there.
static FILE *create_new(const char *path) { return fopen(path, "wbx"); }

int gpu_write_report(const char *dir, const char *text, const void *bin, size_t size, char *path, size_t path_size) {
    char folder[PATH_BYTES + 16], stem[64], txt[PATH_BYTES + 96], bpath[PATH_BYTES + 96];
    int n = dir && *dir ? snprintf(folder, sizeof folder, "%s/%s", dir, GPU_REPORT_DIR)
                        : snprintf(folder, sizeof folder, "%s", GPU_REPORT_DIR);
    if (n < 0 || (size_t)n >= sizeof folder) {
        snprintf(path, path_size, "the folder %s/%s is too long", dir, GPU_REPORT_DIR);
        return 0;
    }
    if (!make_folder(folder)) {
        snprintf(path, path_size, "cannot make the folder %s: %s", folder, strerror(errno));
        return 0;
    }
    struct tm tm;
    local_time(&tm);
    strftime(stem, sizeof stem, "device-lost-%Y%m%d-%H%M%S", &tm);
    for (int k = 1; k <= 99; k++) {
        if (k == 1) snprintf(txt, sizeof txt, "%s/%s.txt", folder, stem);
        else snprintf(txt, sizeof txt, "%s/%s-%d.txt", folder, stem, k);
        FILE *f = create_new(txt);
        if (!f && errno == EEXIST) continue;
        if (!f) {
            snprintf(path, path_size, "cannot create %s: %s", txt, strerror(errno));
            return 0;
        }
        int ok = fputs(text, f) >= 0;
        ok = (fclose(f) == 0) && ok;
        if (ok && size) {
            memcpy(bpath, txt, strlen(txt) - 4);
            strcpy(bpath + strlen(txt) - 4, ".bin");
            FILE *b = create_new(bpath);
            ok = b && fwrite(bin, 1, size, b) == size;
            if (b) ok = (fclose(b) == 0) && ok;
            if (!ok) {
                snprintf(path, path_size, "cannot write %s: %s", bpath, strerror(errno));
                return 0;
            }
        }
        if (!ok) {
            snprintf(path, path_size, "cannot write %s: %s", txt, strerror(errno));
            return 0;
        }
        snprintf(path, path_size, "%s", txt);
        return 1;
    }
    snprintf(path, path_size, "99 reports named %s are in %s already", stem, folder);
    return 0;
}

// ---------- the report's text
typedef struct {
    char *p;
    size_t n, cap;
} Text;

static void put(Text *t, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void put(Text *t, const char *fmt, ...) {
    for (;;) {
        va_list ap;
        va_start(ap, fmt);
        size_t room = t->cap - t->n;
        int k = vsnprintf(t->p ? t->p + t->n : NULL, room, fmt, ap);
        va_end(ap);
        if (k < 0) return;
        if ((size_t)k < room) {
            t->n += (size_t)k;
            return;
        }
        size_t cap = t->cap ? 2 * t->cap : 8192;
        while (cap - t->n <= (size_t)k) cap *= 2;
        char *p = realloc(t->p, cap);
        if (!p) return;   // the report keeps what fits
        t->p = p, t->cap = cap;
    }
}

static const char *address_type(VkDeviceFaultAddressTypeEXT t) {
    switch (t) {
    case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT: return "none";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT: return "read invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT: return "write invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT: return "execute invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "instruction pointer (unknown)";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "instruction pointer (invalid)";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT: return "instruction pointer (fault)";
    default: return "unknown type";
    }
}

static const char *kind_name(uint64_t kind) {
    return kind == GPU_VK_REPLAY ? "replayed" : kind == GPU_VK_INTERNAL ? "internal" : "unknown kind";
}

static void report_device(Text *t) {
    const VkPhysicalDeviceProperties *p = &gpu_vk.facts.props;
    if (!gpu_vk.phys) {
        put(t, "device: none chosen yet\n");
        return;
    }
    put(t, "device: %s (index %d), vendor %#x, device %#x, Vulkan %u.%u.%u\n", p->deviceName, gpu_vk.index, p->vendorID,
        p->deviceID, VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion),
        VK_API_VERSION_PATCH(p->apiVersion));
    if (p->vendorID == 0x10de)   // NVIDIA's encoding: 10 bits major, 8 bits minor
        put(t, "driver: %u.%02u (%#x)\n", (p->driverVersion >> 22) & 0x3ff, (p->driverVersion >> 14) & 0xff,
            p->driverVersion);
    else put(t, "driver: %#x\n", p->driverVersion);
    static const char *const validation[] = {"off", "on", "sync", "gpu"};
    put(t, "settings: OOT2_VK_DEVICE \"%s\", OOT2_VK_VALIDATION %s, OOT2_VK_FAULT_REPORT %s\n", gpu_vk.settings.device,
        validation[gpu_vk.settings.validation & 3], gpu_vk.settings.fault_report ? "on" : "off");
}

static void report_timeline(Text *t) {
    uint64_t handed = gpu_sync_handed(), issued = gpu_sync_issued(), done = gpu_sync_done();
    put(t, "timeline: values handed out to %llu, issued to %llu, done to %llu", (unsigned long long)handed,
        (unsigned long long)issued, (unsigned long long)done);
    if (gpu_vk.device && gpu_vk.timeline) {
        uint64_t counter = 0;
        VkResult r = vkGetSemaphoreCounterValue(gpu_vk.device, gpu_vk.timeline, &counter);
        if (r == VK_SUCCESS) put(t, "; the device's counter %llu", (unsigned long long)counter);
        else put(t, "; the device's counter cannot be read (%s)", gpu_vk_result_name(r));
    }
    put(t, "\n");
    if (issued > done)
        put(t, "in flight: values %llu to %llu\n", (unsigned long long)done + 1, (unsigned long long)issued);
    if (handed > issued) put(t, "being submitted: value %llu\n", (unsigned long long)handed);
}

static void report_heaps(Text *t) {
    if (!gpu_vk.phys) return;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
                                            .pNext = &budget};
    vkGetPhysicalDeviceMemoryProperties2(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryProperties.memoryHeapCount; i++) {
        const VkMemoryHeap *h = &mp.memoryProperties.memoryHeaps[i];
        put(t, "heap %u%s: %llu MiB, usage %llu MiB of a budget of %llu MiB\n", i,
            (h->flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " (device local)" : "", (unsigned long long)(h->size >> 20),
            (unsigned long long)(budget.heapUsage[i] >> 20), (unsigned long long)(budget.heapBudget[i] >> 20));
    }
}

// The fault's records into t, and its vendor binary into *bin (malloc'd; NULL and 0 when there is none).
static void report_fault(Text *t, void **bin, size_t *bin_size, char *summary, size_t summary_size) {
    *bin = NULL, *bin_size = 0;
    snprintf(summary, summary_size, "no fault description");
    if (!gpu_vk.fault) {
        put(t, "fault: VK_EXT_device_fault is not enabled (OOT2_VK_FAULT_REPORT=off)\n");
        return;
    }
    if (!gpu_vk.device || !vkGetDeviceFaultInfoEXT) {
        put(t, "fault: no device, or no vkGetDeviceFaultInfoEXT\n");
        return;
    }
    VkDeviceFaultCountsEXT counts = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    VkResult r = vkGetDeviceFaultInfoEXT(gpu_vk.device, &counts, NULL);
    if (r != VK_SUCCESS) {
        put(t, "fault: vkGetDeviceFaultInfoEXT failed: %s\n", gpu_vk_result_name(r));
        return;
    }
    if (gpu_vk.fault != 2) counts.vendorBinarySize = 0;   // the device offers no binary, or it is not enabled
    VkDeviceFaultAddressInfoEXT *addr = calloc(counts.addressInfoCount + 1, sizeof *addr);
    VkDeviceFaultVendorInfoEXT *vendor = calloc(counts.vendorInfoCount + 1, sizeof *vendor);
    void *data = counts.vendorBinarySize ? malloc(counts.vendorBinarySize) : NULL;
    if (!addr || !vendor || (counts.vendorBinarySize && !data)) {
        put(t, "fault: out of memory for %u addresses, %u vendor records and %llu bytes\n", counts.addressInfoCount,
            counts.vendorInfoCount, (unsigned long long)counts.vendorBinarySize);
        free(addr), free(vendor), free(data);
        return;
    }
    VkDeviceFaultInfoEXT info = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT, .pAddressInfos = addr,
                                 .pVendorInfos = vendor, .pVendorBinaryData = data};
    r = vkGetDeviceFaultInfoEXT(gpu_vk.device, &counts, &info);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
        put(t, "fault: vkGetDeviceFaultInfoEXT failed: %s\n", gpu_vk_result_name(r));
        free(addr), free(vendor), free(data);
        return;
    }
    info.description[sizeof info.description - 1] = 0;
    put(t, "fault: \"%s\"%s\n", info.description, r == VK_INCOMPLETE ? " (the driver answered VK_INCOMPLETE)" : "");
    snprintf(summary, summary_size, "\"%s\"", info.description);
    for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
        uint64_t prec = addr[i].addressPrecision ? addr[i].addressPrecision : 1, low = addr[i].reportedAddress & ~(prec - 1);
        put(t, "fault address %u: %s at %#llx, precision %#llx (in [%#llx, %#llx))\n", i,
            address_type(addr[i].addressType), (unsigned long long)addr[i].reportedAddress,
            (unsigned long long)addr[i].addressPrecision, (unsigned long long)low, (unsigned long long)(low + prec));
    }
    for (uint32_t i = 0; i < counts.vendorInfoCount; i++) {
        vendor[i].description[sizeof vendor[i].description - 1] = 0;
        put(t, "fault vendor record %u: \"%s\", code %#llx, data %#llx\n", i, vendor[i].description,
            (unsigned long long)vendor[i].vendorFaultCode, (unsigned long long)vendor[i].vendorFaultData);
    }
    if (!counts.vendorBinarySize) {
        put(t, "fault vendor binary: none (%s)\n", gpu_vk.fault == 2 ? "the driver gave none" : "the device offers none");
    } else {
        put(t, "fault vendor binary: %llu bytes, saved beside this report with the same name and .bin\n",
            (unsigned long long)counts.vendorBinarySize);
        VkDeviceFaultVendorBinaryHeaderVersionOneEXT h;
        if (counts.vendorBinarySize >= sizeof h) {
            memcpy(&h, data, sizeof h);
            if (h.headerVersion == VK_DEVICE_FAULT_VENDOR_BINARY_HEADER_VERSION_ONE_EXT) {
                put(t, "fault vendor binary header: version one, %u bytes, vendor %#x, device %#x, driver %#x, API "
                    "%u.%u.%u\n", h.headerSize, h.vendorID, h.deviceID, h.driverVersion,
                    VK_API_VERSION_MAJOR(h.apiVersion), VK_API_VERSION_MINOR(h.apiVersion),
                    VK_API_VERSION_PATCH(h.apiVersion));
            }
        }
        *bin = data, *bin_size = counts.vendorBinarySize;
        data = NULL;
    }
    free(addr), free(vendor), free(data);
}

static void report_checkpoints(Text *t) {
    if (!gpu_vk.checkpoints) {
        put(t, "checkpoints: not enabled (OOT2_VK_FAULT_REPORT=off)\n");
        return;
    }
    if (!gpu_vk.queue || !vkGetQueueCheckpointData2NV) {
        put(t, "checkpoints: no queue, or no vkGetQueueCheckpointData2NV\n");
        return;
    }
    uint32_t n = 0;
    vkGetQueueCheckpointData2NV(gpu_vk.queue, &n, NULL);
    VkCheckpointData2NV *c = calloc(n + 1, sizeof *c);
    if (!c) {
        put(t, "checkpoints: out of memory for %u\n", n);
        return;
    }
    for (uint32_t i = 0; i < n; i++) c[i].sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_2_NV;
    vkGetQueueCheckpointData2NV(gpu_vk.queue, &n, c);
    put(t, "checkpoints: the last %u the queue reached, one per pipeline stage\n", n);
    // A per-draw marker (the recorder's breadcrumb, oot2/vk_rec.h) names a draw by its submission and number, and the
    // recorder's table gives the stage identities it carried. Markers order by (value, draw number), which is marker >> 4.
    uint64_t ahead = 0, behind = 0;
    int draws = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t m = (uint64_t)(uintptr_t)c[i].pCheckpointMarker;
        if (((m >> 1) & 7) == GPU_VK_DRAW) {
            char text[512];
            vkrec_marker_text(m, text, sizeof text);
            put(t, "checkpoint %u: stage %#llx, %s\n", i, (unsigned long long)c[i].stage, text);
            if (!draws || (m >> 4) > (ahead >> 4)) ahead = m;
            if (!draws || (m >> 4) < (behind >> 4)) behind = m;
            draws++;
            continue;
        }
        put(t, "checkpoint %u: stage %#llx, value %llu, %s submission, %s of its commands\n", i,
            (unsigned long long)c[i].stage, (unsigned long long)(m >> 4), kind_name((m >> 1) & 7),
            (m & 1) ? "end" : "start");
    }
    if (draws) {
        char text[512];
        vkrec_marker_text(ahead, text, sizeof text);
        put(t, "last draw the queue reached: %s\n", text);
        if (behind != ahead) {
            vkrec_marker_text(behind, text, sizeof text);
            put(t, "oldest draw a pipeline stage was still in: %s\n", text);
        }
    }
    free(c);
}

static PlatMutex g_loss_lock = PLAT_MUTEX_INIT;

__attribute__((noreturn)) static void device_lost(const char *what) {
    plat_mutex_lock(&g_loss_lock);   // never released: this thread stops the run, any other waits here until it ends
    Text t = {0};
    char when[64], summary[300], path[PATH_BYTES + 128], folder[PATH_BYTES];
    struct tm tm;
    local_time(&tm);
    strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", &tm);
    put(&t, "oot2 device-lost report\nwhen: %s (local time)\nlost at: %s returned VK_ERROR_DEVICE_LOST\n", when, what);
    report_device(&t);
    report_timeline(&t);
    report_heaps(&t);
    void *bin;
    size_t bin_size;
    report_fault(&t, &bin, &bin_size, summary, sizeof summary);
    report_checkpoints(&t);
    plat_mutex_lock(&g_folder_lock);
    snprintf(folder, sizeof folder, "%s", g_folder);
    plat_mutex_unlock(&g_folder_lock);
    if (!t.p) rt_fatal("gfx: vulkan: the device was lost (VK_ERROR_DEVICE_LOST from %s); no memory for a report", what);
    if (gpu_write_report(folder, t.p, bin, bin_size, path, sizeof path)) {
        rt_log("gfx: vulkan: the device was lost: %s", summary);
        rt_fatal("gfx: vulkan: the device was lost (VK_ERROR_DEVICE_LOST from %s); the report is %s", what, path);
    }
    rt_log("gfx: vulkan: the device-loss report could not be written (%s); here it is:", path);
    plat_write_stderr(t.p, t.n);
    rt_fatal("gfx: vulkan: the device was lost (VK_ERROR_DEVICE_LOST from %s); the report could not be written (%s), it "
             "is in the log above", what, path);
}

void gpu_vk_failed(VkResult r, const char *what) {
    if (r == VK_ERROR_DEVICE_LOST) device_lost(what);
    rt_fatal("gfx: vulkan: %s failed: %s", what, gpu_vk_result_name(r));
}
