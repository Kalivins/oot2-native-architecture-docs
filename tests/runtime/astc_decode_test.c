// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The GPU ASTC decode against the CPU reference (runtime/gpu/shaders/astc_decode.comp, tests/runtime/astc_ref.c; architecture
// D19). Driven by tests/test_runtime_vk_astc.py: `astc_decode_test <shader.spv>`. It makes its own instance and device (Vulkan
// 1.3, buffer device addresses and 64-bit integers, nothing else), so it runs on any compute device, the laptop's included, and
// it links no runtime code.
//
// For each of the 14 block sizes NVN has ASTC formats for, in the linear and the sRGB profile, it builds blocks from a seeded
// generator (random 128-bit blocks that the reference decodes without error and without an HDR endpoint mode, and void-extent
// blocks of random colours), decodes them on the device in one dispatch, and compares every texel with the reference. The
// reference is run twice: rounding the 16-bit result to the nearest of 255 steps, which the shader does and the comparison
// demands byte for byte, and keeping its top eight bits, as astcenc does, which the shader may differ from by one and no more.
// Error blocks and blocks with an HDR endpoint mode are not compared (the shader does not make the reference's magenta of them);
// their count is printed.
//
// Prints one line per block size and profile, then `ok astc_decode` and exit 0 when every texel agreed; a size whose texels
// differ is named, and the exit status is 1. Exit 3: no Vulkan loader or device.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif
#include "volk.c"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "astc_blocks.h"

typedef unsigned long long ull;
#define EXIT_NO_DEVICE 3
#define CHECK(c) do { if (!(c)) { printf("check failed at line %d: %s\n", __LINE__, #c); exit(2); } } while (0)

static const uint32_t k_sizes[14][2] = {{4, 4}, {5, 4}, {5, 5}, {6, 5}, {6, 6}, {8, 5}, {8, 6},
                                        {8, 8}, {10, 5}, {10, 6}, {10, 8}, {10, 10}, {12, 10}, {12, 12}};
#define RANDOM_BLOCKS 600
#define VOID_BLOCKS 64
#define BLOCKS (RANDOM_BLOCKS + VOID_BLOCKS)
#define BLOCKS_X 32u
#define BLOCKS_Y ((BLOCKS + BLOCKS_X - 1) / BLOCKS_X)

// ---------- a device of our own

typedef struct Dev {
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t family;
    VkPhysicalDeviceMemoryProperties mem;
} Dev;

static Dev d;

static void device_or_exit(void) {
    if (volkInitialize() != VK_SUCCESS) {
        printf("no Vulkan loader\n");
        exit(EXIT_NO_DEVICE);
    }
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    if (vkCreateInstance(&ci, NULL, &d.inst) != VK_SUCCESS) {
        printf("no Vulkan 1.3 instance\n");
        exit(EXIT_NO_DEVICE);
    }
    volkLoadInstance(d.inst);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(d.inst, &n, NULL);
    if (!n) {
        printf("no Vulkan device\n");
        exit(EXIT_NO_DEVICE);
    }
    VkPhysicalDevice devs[16];
    if (n > 16) n = 16;
    vkEnumeratePhysicalDevices(d.inst, &n, devs);
    for (uint32_t i = 0; i < n && !d.phys; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.apiVersion < VK_API_VERSION_1_3) continue;
        VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 f = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
        vkGetPhysicalDeviceFeatures2(devs[i], &f);
        if (!f12.bufferDeviceAddress || !f.features.shaderInt64) continue;
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, NULL);
        VkQueueFamilyProperties qp[16];
        if (qn > 16) qn = 16;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qp);
        for (uint32_t q = 0; q < qn; q++)
            if (qp[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                d.phys = devs[i], d.family = q;
                printf("device: %s\n", p.deviceName);
                break;
            }
    }
    if (!d.phys) {
        printf("no Vulkan 1.3 compute device with buffer device addresses and 64-bit integers\n");
        exit(EXIT_NO_DEVICE);
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = d.family,
                                  .queueCount = 1, .pQueuePriorities = &prio};
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .bufferDeviceAddress = VK_TRUE};
    VkPhysicalDeviceFeatures2 f = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12,
                                   .features = {.shaderInt64 = VK_TRUE}};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qi};
    CHECK(vkCreateDevice(d.phys, &di, NULL, &d.dev) == VK_SUCCESS);
    volkLoadDevice(d.dev);
    vkGetDeviceQueue(d.dev, d.family, 0, &d.queue);
    vkGetPhysicalDeviceMemoryProperties(d.phys, &d.mem);
}

typedef struct Buf {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint8_t *map;
    uint64_t addr, size;
} Buf;

static Buf buf_new(uint64_t size) {
    Buf b = {.size = size};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    CHECK(vkCreateBuffer(d.dev, &bi, NULL, &b.buf) == VK_SUCCESS);
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(d.dev, b.buf, &mr);
    uint32_t type = UINT32_MAX;
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < d.mem.memoryTypeCount; i++)
        if ((mr.memoryTypeBits >> i & 1) && (d.mem.memoryTypes[i].propertyFlags & want) == want) {
            type = i;
            break;
        }
    CHECK(type != UINT32_MAX);
    VkMemoryAllocateFlagsInfo fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                    .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo alloc_info = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &fi, .allocationSize = mr.size,
                               .memoryTypeIndex = type};
    CHECK(vkAllocateMemory(d.dev, &alloc_info, NULL, &b.mem) == VK_SUCCESS);
    CHECK(vkBindBufferMemory(d.dev, b.buf, b.mem, 0) == VK_SUCCESS);
    void *map;
    CHECK(vkMapMemory(d.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &map) == VK_SUCCESS);
    b.map = map;
    VkBufferDeviceAddressInfo da = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b.buf};
    b.addr = vkGetBufferDeviceAddress(d.dev, &da);
    return b;
}

static void buf_free(Buf *b) {
    vkDestroyBuffer(d.dev, b->buf, NULL);
    vkFreeMemory(d.dev, b->mem, NULL);
}

// ---------- the pass

typedef struct Push {
    uint64_t src, dst;
    uint32_t bw, bh, blocks_x, blocks_y, srgb;
} Push;
_Static_assert(sizeof(Push) == 40, "the push constants of astc_decode.comp are 40 bytes");

static VkPipeline g_pipe;
static VkPipelineLayout g_layout;
static VkCommandPool g_pool;
static VkCommandBuffer g_cb;
static VkFence g_fence;

static void pipeline_make(const char *spv_path) {
    FILE *f = fopen(spv_path, "rb");
    CHECK(f);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    CHECK(n > 0 && n % 4 == 0);
    uint32_t *code = malloc((size_t)n);
    CHECK(code && fread(code, 1, (size_t)n, f) == (size_t)n);
    fclose(f);
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)n, .pCode = code};
    VkShaderModule m;
    CHECK(vkCreateShaderModule(d.dev, &mi, NULL, &m) == VK_SUCCESS);
    free(code);
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
                                     .pPushConstantRanges = &pr};
    CHECK(vkCreatePipelineLayout(d.dev, &li, NULL, &g_layout) == VK_SUCCESS);
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = g_layout};
    CHECK(vkCreateComputePipelines(d.dev, VK_NULL_HANDLE, 1, &ci, NULL, &g_pipe) == VK_SUCCESS);
    vkDestroyShaderModule(d.dev, m, NULL);
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = d.family};
    CHECK(vkCreateCommandPool(d.dev, &pi, NULL, &g_pool) == VK_SUCCESS);
    VkCommandBufferAllocateInfo alloc_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    CHECK(vkAllocateCommandBuffers(d.dev, &alloc_info, &g_cb) == VK_SUCCESS);
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK(vkCreateFence(d.dev, &fi, NULL, &g_fence) == VK_SUCCESS);
}

static void run_pass(const Push *p) {
    CHECK(vkResetCommandBuffer(g_cb, 0) == VK_SUCCESS);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK(vkBeginCommandBuffer(g_cb, &bi) == VK_SUCCESS);
    vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipe);
    vkCmdPushConstants(g_cb, g_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof *p, p);
    vkCmdDispatch(g_cb, (p->blocks_x + 7) / 8, (p->blocks_y + 7) / 8, 1);
    CHECK(vkEndCommandBuffer(g_cb) == VK_SUCCESS);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g_cb};
    CHECK(vkQueueSubmit(d.queue, 1, &si, g_fence) == VK_SUCCESS);
    CHECK(vkWaitForFences(d.dev, 1, &g_fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS);
    CHECK(vkResetFences(d.dev, 1, &g_fence) == VK_SUCCESS);
}

// ---------- the blocks

typedef struct Cov {
    unsigned part[4], dual, voids, err, hdr;
} Cov;

// Blocks[0, RANDOM_BLOCKS) are random blocks the reference takes (no error, no HDR endpoint), in the form an encoder writes
// them (astc_blocks.h), the rest void extents. The shader (Eden's) reads the unused bits of a last partial trit or quint
// block from the stream where the reference takes them as zero, so a block drawn at random with them set says nothing about
// the shader; the padding an encoder writes is zero.
static void make_blocks(uint32_t bw, uint32_t bh, uint8_t (*blocks)[16], Cov *cov) {
    for (unsigned have = 0; have < RANDOM_BLOCKS; have++) {
        CHECK(astc_draw_block(bw, bh, blocks[have], &cov->err, &cov->hdr));
        uint64_t lo = 0;
        for (int i = 7; i >= 0; i--) lo = lo << 8 | blocks[have][i];
        if ((lo & 0x1FF) != 0x1FC) {   // a void extent that came up by chance is not counted
            cov->part[(lo >> 11) & 3]++;
            cov->dual += (unsigned)((lo >> 10) & 1);
        }
    }
    for (unsigned i = RANDOM_BLOCKS; i < BLOCKS; i++) astc_void_block(blocks[i]), cov->voids++;
}

// ---------- the comparison

static int g_failed_sizes;

static void compare(uint32_t bw, uint32_t bh, int srgb, const Cov *cov, uint8_t (*blocks)[16], const uint8_t *gpu) {
    const uint32_t row_texels = BLOCKS_X * bw;
    unsigned differ = 0, texels = 0, off_by_one = 0;
    uint8_t want[12 * 12 * 4], top[12 * 12 * 4];
    for (unsigned i = 0; i < BLOCKS; i++) {
        int s1 = astc_ref_block(blocks[i], bw, bh, srgb != 0, 1, want);
        int s2 = astc_ref_block(blocks[i], bw, bh, srgb != 0, 0, top);
        CHECK(s1 == ASTC_REF_OK && s2 == ASTC_REF_OK);   // the generator took only what the reference decodes without error: in either profile
        unsigned bx = i % BLOCKS_X, by = i / BLOCKS_X;
        for (uint32_t y = 0; y < bh; y++)
            for (uint32_t x = 0; x < bw; x++) {
                const uint8_t *g = gpu + ((size_t)(by * bh + y) * row_texels + bx * bw + x) * 4;
                const uint8_t *w = want + (y * bw + x) * 4, *t = top + (y * bw + x) * 4;
                texels++;
                if (memcmp(g, w, 4)) {
                    if (differ++ < 3)
                        printf("  block %u texel (%u,%u): gpu %02x%02x%02x%02x reference %02x%02x%02x%02x\n", i, x, y, g[0], g[1], g[2],
                               g[3], w[0], w[1], w[2], w[3]);
                }
                for (int c = 0; c < 4; c++) {
                    int dd = (int)g[c] - (int)t[c];
                    if (dd < -1 || dd > 1) off_by_one++;   // the nearest step is at most one from the top eight bits of the 16-bit value, either way
                }
            }
    }
    printf("%ux%u %s: %u blocks (%u void extents), %u texels, %u differ from the reference, %u further than one step from the top eight bits; "
           "drawn and left out: %u error blocks, %u with an HDR endpoint mode; partitions 1..4: %u %u %u %u, dual plane %u\n",
           bw, bh, srgb ? "sRGB  " : "linear", BLOCKS, cov->voids, texels, differ, off_by_one, cov->err, cov->hdr, cov->part[0], cov->part[1],
           cov->part[2], cov->part[3], cov->dual);
    if (differ || off_by_one) {
        g_failed_sizes++;
        printf("DIFFERS %ux%u %s\n", bw, bh, srgb ? "sRGB" : "linear");
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: astc_decode_test <astc_decode.spv>\n");
        return 2;
    }
    device_or_exit();
    pipeline_make(argv[1]);
    for (int s = 0; s < 14; s++) {
        const uint32_t bw = k_sizes[s][0], bh = k_sizes[s][1];
        for (int srgb = 0; srgb < 2; srgb++) {
            astc_seed(0x9E3779B97F4A7C15ull ^ ((uint64_t)bw << 32 | (uint64_t)bh << 8));   // the same blocks for both profiles
            uint8_t(*blocks)[16] = calloc(BLOCKS_X * BLOCKS_Y, 16);
            Cov cov = {.dual = 0};
            CHECK(blocks);
            make_blocks(bw, bh, blocks, &cov);
            Buf in = buf_new((uint64_t)BLOCKS_X * BLOCKS_Y * 16);
            const uint64_t out_bytes = (uint64_t)BLOCKS_X * bw * BLOCKS_Y * bh * 4;
            Buf out = buf_new(out_bytes);
            memcpy(in.map, blocks, (size_t)in.size);
            memset(out.map, 0xEE, (size_t)out_bytes);
            Push p = {.src = in.addr, .dst = out.addr, .bw = bw, .bh = bh, .blocks_x = BLOCKS_X, .blocks_y = BLOCKS_Y, .srgb = (uint32_t)srgb};
            run_pass(&p);
            compare(bw, bh, srgb, &cov, blocks, out.map);
            buf_free(&in), buf_free(&out), free(blocks);
        }
    }
    if (g_failed_sizes) {
        printf("%d of 28 block sizes and profiles differ\n", g_failed_sizes);
        return 1;
    }
    printf("ok astc_decode\n");
    return 0;
}
