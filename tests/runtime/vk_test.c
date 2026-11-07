// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Tests of the Vulkan back end on a real device (runtime/gpu, oot2/gpu_vk.h), driven by tests/test_runtime_vk.py:
// `vk_test <case> [folder]`, one case per process. They need a Vulkan device and skip without one, unless
// OOT2_REQUIRE_GPU is set; continuous integration has none.
//
//   devices          every device, judged by the items of oot2/vk_required.h: what each lacks (no device is opened)
//   open             the device opens with every item, and closes
//   submits          1,000 empty replayed submissions from two threads, mixed with 100 of the back end's own that fill a
//                    word with their value on the GPU: values in order, each effect after the GPU's write and before
//                    its value is done
//   wait             a submission held by a semaphore the test signals: a fence wait times out, a later one succeeds,
//                    finish leaves nothing in flight
//   loss_submit      vkQueueSubmit2 answers VK_ERROR_DEVICE_LOST: the report is written to the folder, the run stops
//   loss_completion  the completion thread's vkWaitSemaphores answers it: the same, from that thread
//   refuse           OOT2_VK_DEVICE names no device: the start stops by name
//
// The loss cases replace Volk's entry points for the failing call and for the two queries that are valid only on a lost
// device (the fault and the checkpoints) with answers made here, so the whole path runs without hanging the GPU.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define EXIT_NO_DEVICE 3   // no loader or no device: the driver skips

static void close_and_report(void) {
    gpu_vk_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation errors %llu warnings %llu\n", (unsigned long long)errors, (unsigned long long)warnings);
}

// ---------- devices: the judgement of every device, without opening one
static void case_devices(void) {
    if (volkInitialize() != VK_SUCCESS) {
        printf("no Vulkan loader\n");
        exit(EXIT_NO_DEVICE);
    }
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    if (vkCreateInstance(&ci, NULL, &inst) != VK_SUCCESS) {
        printf("no Vulkan 1.3 instance\n");
        exit(EXIT_NO_DEVICE);
    }
    volkLoadInstanceOnly(inst);
    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL) == VK_SUCCESS);
    if (!n) {
        printf("no Vulkan device\n");
        exit(EXIT_NO_DEVICE);
    }
    VkPhysicalDevice devs[16];
    if (n > 16) n = 16;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, devs) >= 0);
    int usable = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t ne = 0;
        CHECK(vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, NULL) == VK_SUCCESS);
        VkExtensionProperties *ext = calloc(ne ? ne : 1, sizeof *ext);
        CHECK(ext && vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, ext) == VK_SUCCESS);
        Oot2VkFacts f;
        oot2_vk_query(vkGetPhysicalDeviceProperties2, vkGetPhysicalDeviceFeatures2, devs[i], ext, ne, &f);
        free(ext);
        unsigned char has[OOT2_VK_ITEM_COUNT];
        int count = oot2_vk_judge(&f, has);
        printf("device %u\t%s\tVulkan %u.%u.%u\t%d of %d", i, f.props.deviceName, VK_API_VERSION_MAJOR(f.props.apiVersion),
               VK_API_VERSION_MINOR(f.props.apiVersion), VK_API_VERSION_PATCH(f.props.apiVersion), count,
               OOT2_VK_ITEM_COUNT);
        for (int k = 0, first = 1; k < OOT2_VK_ITEM_COUNT; k++)
            if (!has[k]) printf("%s%s", first ? "\tlacks " : ", ", oot2_vk_item_id(k)), first = 0;
        printf("\n");
        printf("facts %u\tsubgroup stages %#x operations %#x\trequired-size stages %#x\tvendor binary %u\n", i,
               f.p11.subgroupSupportedStages, f.p11.subgroupSupportedOperations, f.p13.requiredSubgroupSizeStages,
               f.fault.deviceFaultVendorBinary);
        usable += count == OOT2_VK_ITEM_COUNT;
    }
    vkDestroyInstance(inst, NULL);
    printf("usable %d\n", usable);
    printf("ok devices\n");
}

static void case_open(void) {
    gpu_vk_open();
    CHECK(gpu_vk.device && gpu_vk.queue && gpu_vk.timeline && gpu_vk.pool);
    unsigned char has[OOT2_VK_ITEM_COUNT];
    int count = oot2_vk_judge(&gpu_vk.facts, has);
    printf("opened %s, %d of %d items, queue family %u, fault %d, checkpoints %d\n", gpu_vk.facts.props.deviceName,
           count, OOT2_VK_ITEM_COUNT, gpu_vk.family, gpu_vk.fault, gpu_vk.checkpoints);
    CHECK(count == OOT2_VK_ITEM_COUNT);
    gpu_vk_open();   // a second call returns at once
    close_and_report();
    printf("ok open\n");
}

// ---------- a host-visible buffer the GPU writes words into
static VkBuffer g_buf;
static VkDeviceMemory g_mem;
static volatile uint32_t *g_words;
#define WORDS 4096

static void make_buffer(void) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = WORDS * 4,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    CHECK(vkCreateBuffer(gpu_vk.device, &bi, NULL, &g_buf) == VK_SUCCESS);
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, g_buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; i++)
        if ((mr.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) type = i;
    CHECK(type != UINT32_MAX);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                               .memoryTypeIndex = type};
    CHECK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &g_mem) == VK_SUCCESS);
    CHECK(vkBindBufferMemory(gpu_vk.device, g_buf, g_mem, 0) == VK_SUCCESS);
    void *p;
    CHECK(vkMapMemory(gpu_vk.device, g_mem, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS);
    g_words = p;
    memset(p, 0, WORDS * 4);
}

static void free_buffer(void) {
    vkUnmapMemory(gpu_vk.device, g_mem);
    vkDestroyBuffer(gpu_vk.device, g_buf, NULL);
    vkFreeMemory(gpu_vk.device, g_mem, NULL);
}

// Records "write value into its word, then make it visible to the host" into the open submission's command buffer.
static void record_fill(uint64_t value) {
    VkCommandBuffer cb = gpu_vk_cmd();
    vkCmdFillBuffer(cb, g_buf, (value % WORDS) * 4, 4, (uint32_t)value);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
                           .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// ---------- submits
#define SUBMITTERS 2
#define REPLAYS_PER_THREAD 500
#define INTERNAL_EVERY 10
static _Alignas(8) uint8_t g_empty[16];   // {OP_BEGIN, OP_END}
static uint64_t g_internal_done, g_last_effect;
static int g_left = SUBMITTERS;
static PlatMutex g_join_lock = PLAT_MUTEX_INIT;
static PlatCond g_join_cond = PLAT_COND_INIT;

// The completion thread: the GPU wrote this value's word, every earlier value is done, its own is not.
static void fill_effect(void *arg, uint64_t value) {
    (void)arg;
    CHECK(gpu_sync_done() == value - 1);
    CHECK(g_words[value % WORDS] == (uint32_t)value);
    CHECK(value > g_last_effect);
    g_last_effect = value;
    g_internal_done++;
}

static void submitter(void *arg) {
    uint64_t last = 0, handle = (uint64_t)(uintptr_t)g_empty;
    for (int i = 0; i < REPLAYS_PER_THREAD; i++) {
        uint64_t v = gfx_vulkan_backend.submit(&handle, 1);
        CHECK(v > last);
        last = v;
        if (i % INTERNAL_EVERY == (int)(uintptr_t)arg) {
            uint64_t w = gpu_vk_begin(GPU_VK_INTERNAL);
            record_fill(w);
            gpu_vk_effect(fill_effect, NULL);
            CHECK(gpu_vk_end(NULL, 0) == w && w > last);
            last = w;
            if (i % 50 == 0) {   // a guest that waits: the effect ran before the value was done
                CHECK(gfx_vulkan_backend.fence_wait(w, UINT64_MAX) == 1);
                CHECK(g_last_effect >= w && gfx_vulkan_backend.fence_done() >= w);
            }
        }
    }
    plat_mutex_lock(&g_join_lock);
    g_left--;
    plat_cond_broadcast(&g_join_cond);
    plat_mutex_unlock(&g_join_lock);
}

static void case_submits(void) {
    op_hdr rec[2] = {{OP_BEGIN, 8}, {OP_END, 8}};
    memcpy(g_empty, rec, sizeof rec);
    gpu_vk_open();
    make_buffer();
    uint64_t t0 = plat_now_ns();
    for (int i = 0; i < SUBMITTERS; i++) CHECK(plat_thread_start(submitter, (void *)(uintptr_t)i, 1 << 20));
    plat_mutex_lock(&g_join_lock);
    while (g_left) plat_cond_wait(&g_join_cond, &g_join_lock);
    plat_mutex_unlock(&g_join_lock);
    gfx_vulkan_backend.finish();
    double ms = (double)(plat_now_ns() - t0) / 1e6;
    uint64_t handed = gpu_sync_handed(), counter = 0;
    CHECK(vkGetSemaphoreCounterValue(gpu_vk.device, gpu_vk.timeline, &counter) == VK_SUCCESS);
    const uint64_t internal = SUBMITTERS * REPLAYS_PER_THREAD / INTERNAL_EVERY;
    printf("values %llu (replayed %d, internal %llu, effects run %llu), done %llu, device counter %llu, %.1f ms\n",
           (unsigned long long)handed, SUBMITTERS * REPLAYS_PER_THREAD, (unsigned long long)internal,
           (unsigned long long)g_internal_done, (unsigned long long)gfx_vulkan_backend.fence_done(),
           (unsigned long long)counter, ms);
    CHECK(handed == SUBMITTERS * REPLAYS_PER_THREAD + internal);
    CHECK(g_internal_done == internal);
    CHECK(gfx_vulkan_backend.fence_done() == handed && counter == handed);
    free_buffer();
    close_and_report();
    printf("ok submits\n");
}

// ---------- wait: a submission held by a semaphore the host signals
static int g_gated_ran;
static void gated_effect(void *arg, uint64_t value) {
    (void)arg, (void)value;
    __atomic_store_n(&g_gated_ran, 1, __ATOMIC_RELEASE);
}

static VkSemaphore make_gate(void) {
    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    VkSemaphore gate;
    CHECK(vkCreateSemaphore(gpu_vk.device, &si, NULL, &gate) == VK_SUCCESS);
    return gate;
}

static void case_wait(void) {
    gpu_vk_open();
    VkSemaphore gate = make_gate();
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = gate, .value = 1,
                                  .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    gpu_vk_cmd();
    gpu_vk_effect(gated_effect, NULL);
    CHECK(gpu_vk_end(&wait, 1) == v);
    uint64_t t0 = plat_now_ns();
    CHECK(gfx_vulkan_backend.fence_wait(v, 50000000ull) == 0);
    double waited = (double)(plat_now_ns() - t0) / 1e6;
    CHECK(waited >= 50.0);
    CHECK(gfx_vulkan_backend.fence_done() < v && !__atomic_load_n(&g_gated_ran, __ATOMIC_ACQUIRE));
    CHECK(gfx_vulkan_backend.fence_wait(v + 1, 0) == 0);   // never handed out
    VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = gate, .value = 1};
    CHECK(vkSignalSemaphore(gpu_vk.device, &si) == VK_SUCCESS);
    t0 = plat_now_ns();
    CHECK(gfx_vulkan_backend.fence_wait(v, 5000000000ull) == 1);
    double then = (double)(plat_now_ns() - t0) / 1e6;
    CHECK(__atomic_load_n(&g_gated_ran, __ATOMIC_ACQUIRE));
    uint64_t handle = (uint64_t)(uintptr_t)g_empty;
    op_hdr rec[2] = {{OP_BEGIN, 8}, {OP_END, 8}};
    memcpy(g_empty, rec, sizeof rec);
    for (int i = 0; i < 20; i++) gfx_vulkan_backend.submit(&handle, 1);
    gfx_vulkan_backend.finish();
    uint64_t counter = 0, handed = gpu_sync_handed();
    CHECK(vkGetSemaphoreCounterValue(gpu_vk.device, gpu_vk.timeline, &counter) == VK_SUCCESS);
    CHECK(gfx_vulkan_backend.fence_done() == handed && counter == handed && gpu_sync_issued() == handed);
    printf("timed out after %.1f ms; done %.2f ms after the gate opened; finish left values %llu, counter %llu\n",
           waited, then, (unsigned long long)handed, (unsigned long long)counter);
    vkDestroySemaphore(gpu_vk.device, gate, NULL);
    close_and_report();
    printf("ok wait\n");
}

// ---------- the device lost: Volk's entry points replaced
static int g_fault_calls, g_checkpoint_calls;

static VkResult VKAPI_CALL lost_submit(VkQueue q, uint32_t n, const VkSubmitInfo2 *s, VkFence f) {
    (void)q, (void)n, (void)s, (void)f;
    return VK_ERROR_DEVICE_LOST;
}

static VkResult VKAPI_CALL lost_wait(VkDevice d, const VkSemaphoreWaitInfo *w, uint64_t t) {
    (void)d, (void)w, (void)t;
    return VK_ERROR_DEVICE_LOST;
}

#define FAKE_BINARY 200
static VkResult VKAPI_CALL fake_fault(VkDevice d, VkDeviceFaultCountsEXT *counts, VkDeviceFaultInfoEXT *info) {
    (void)d;
    g_fault_calls++;
    if (!info) {
        counts->addressInfoCount = 2, counts->vendorInfoCount = 1, counts->vendorBinarySize = FAKE_BINARY;
        return VK_SUCCESS;
    }
    snprintf(info->description, sizeof info->description, "injected by the test: no real fault");
    if (counts->addressInfoCount >= 2) {
        info->pAddressInfos[0] = (VkDeviceFaultAddressInfoEXT){VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT,
                                                                0x7f0012345678ull, 0x1000};
        info->pAddressInfos[1] = (VkDeviceFaultAddressInfoEXT){
            VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT, 0x1234, 0};
    }
    if (counts->vendorInfoCount >= 1) {
        snprintf(info->pVendorInfos[0].description, sizeof info->pVendorInfos[0].description, "injected vendor record");
        info->pVendorInfos[0].vendorFaultCode = 0xdead, info->pVendorInfos[0].vendorFaultData = 0xbeef;
    }
    if (info->pVendorBinaryData && counts->vendorBinarySize) {
        uint8_t *b = info->pVendorBinaryData;
        for (size_t i = 0; i < counts->vendorBinarySize; i++) b[i] = (uint8_t)(i ^ 0x5a);
        VkDeviceFaultVendorBinaryHeaderVersionOneEXT h = {
            .headerSize = sizeof h, .headerVersion = VK_DEVICE_FAULT_VENDOR_BINARY_HEADER_VERSION_ONE_EXT,
            .vendorID = 0x10de, .deviceID = 0x2489, .driverVersion = 0x9a438000, .apiVersion = VK_API_VERSION_1_3};
        if (counts->vendorBinarySize >= sizeof h) memcpy(b, &h, sizeof h);
    }
    return VK_SUCCESS;
}

static void VKAPI_CALL fake_checkpoints(VkQueue q, uint32_t *n, VkCheckpointData2NV *data) {
    (void)q;
    g_checkpoint_calls++;
    if (!data) {
        *n = 2;
        return;
    }
    if (*n >= 2) {
        data[0].stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        data[0].pCheckpointMarker = (void *)(uintptr_t)GPU_VK_MARKER(41, GPU_VK_REPLAY, 0);
        data[1].stage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        data[1].pCheckpointMarker = (void *)(uintptr_t)GPU_VK_MARKER(40, GPU_VK_INTERNAL, 1);
    }
}

static void fake_queries(void) {
    vkGetDeviceFaultInfoEXT = fake_fault;
    vkGetQueueCheckpointData2NV = fake_checkpoints;
}

static void case_loss_submit(const char *dir) {
    gpu_set_report_folder(dir);
    gpu_vk_open();
    uint64_t handle = (uint64_t)(uintptr_t)g_empty;
    op_hdr rec[2] = {{OP_BEGIN, 8}, {OP_END, 8}};
    memcpy(g_empty, rec, sizeof rec);
    for (int i = 0; i < 3; i++) gfx_vulkan_backend.submit(&handle, 1);
    gfx_vulkan_backend.finish();
    fake_queries();
    vkQueueSubmit2 = lost_submit;
    gfx_vulkan_backend.submit(&handle, 1);
    printf("submit returned after the loss\n");
}

static void case_loss_completion(const char *dir) {
    gpu_set_report_folder(dir);
    gpu_vk_open();
    fake_queries();
    vkWaitSemaphores = lost_wait;
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    gpu_vk_cmd();
    gpu_vk_end(NULL, 0);
    gfx_vulkan_backend.fence_wait(v, 10000000000ull);
    printf("fence_wait returned after the loss\n");
}

static void case_refuse(void) {
    gpu_vk_open();
    printf("gpu_vk_open returned\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: vk_test <case> [folder]\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "devices")) case_devices();
    else if (!strcmp(cs, "open")) case_open();
    else if (!strcmp(cs, "submits")) case_submits();
    else if (!strcmp(cs, "wait")) case_wait();
    else if (!strcmp(cs, "loss_submit") && argc >= 3) case_loss_submit(argv[2]);
    else if (!strcmp(cs, "loss_completion") && argc >= 3) case_loss_completion(argv[2]);
    else if (!strcmp(cs, "refuse")) case_refuse();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
