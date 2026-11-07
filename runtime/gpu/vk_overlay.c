// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The menu's renderer (oot2/vk_present.h, vk_present_int.h): Dear ImGui's draw lists, flattened by the menu (oot2/ui.h),
// drawn with one pipeline, one sampler and dynamic rendering over the scaled picture. Dear ImGui's own Vulkan back end is
// not used because it submits and waits on the queue itself (texture uploads end in vkQueueWaitIdle), and this back end has
// one queue that only the holder of a submission may touch (oot2/gpu_vk.h).
//
// Everything that can be made before the first frame is: the pipeline for the swapchain's format, the sampler, the
// descriptor pool, the three frames' vertex and index buffers (host-visible, mapped for good, grown only when a frame
// needs more) and the font atlas, which the menu hands over through the texture ops when vkw_open asks it to prepare.
// A texture's pixels are copied at the call that gives them and carried to the GPU by the next submission that draws (a
// staging buffer per upload, freed by an effect once that submission is done), so the first frame that shows the menu
// creates nothing but, rarely, a glyph's upload.
#include <stdlib.h>
#include <string.h>

#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "oot2/vk_present.h"
#include "vk_present_int.h"
#include "plat.h"

#define MAX_TEXTURES 8
#define MAX_UPLOADS 16
#define VERTEX_BYTES (1u << 20)   // first sizes of a frame's buffers; they double when a frame needs more
#define INDEX_BYTES (1u << 19)

typedef struct {
    int live, ready;   // ready: the image holds pixels (its layout is SHADER_READ_ONLY_OPTIMAL)
    int w, h;
    VkImage image;
    VkDeviceMemory mem;
    VkImageView view;
    VkDescriptorSet set;
} Tex;

typedef struct {
    int tex, x, y, w, h;
    uint8_t *rgba;   // w * h * 4, tightly packed
} Upload;

typedef struct {
    VkBuffer vb, ib;
    VkDeviceMemory vm, im;
    void *vmap, *imap;
    VkDeviceSize vcap, icap;
} FrameBufs;

static const VkwUi *g_ui;
static VkFormat g_format;
static VkDescriptorSetLayout g_set_layout;
static VkPipelineLayout g_layout;
static VkPipeline g_pipeline;
static VkSampler g_sampler;
static VkDescriptorPool g_pool;
static FrameBufs g_fb[VKO_FRAMES];
static Tex g_tex[MAX_TEXTURES];
static Upload g_up[MAX_UPLOADS];
static int g_nup;
static uint64_t g_last_value;   // the newest submission that drew: a texture goes once it is done
static PlatMutex g_mu = PLAT_MUTEX_INIT;   // the uploads and textures: the ops run on the thread that builds the frame

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

static void host_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer *buf, VkDeviceMemory *mem, void **map) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    GPU_VK_OK(vkCreateBuffer(gpu_vk.device, &bi, NULL, buf), "vkCreateBuffer (the overlay)");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, *buf, &mr);
    uint32_t type = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) rt_fatal("gfx: vulkan: no host-visible coherent memory for the overlay");
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                  .memoryTypeIndex = type};
    GPU_VK_OK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, mem), "vkAllocateMemory (the overlay)");
    GPU_VK_OK(vkBindBufferMemory(gpu_vk.device, *buf, *mem, 0), "vkBindBufferMemory (the overlay)");
    GPU_VK_OK(vkMapMemory(gpu_vk.device, *mem, 0, VK_WHOLE_SIZE, 0, map), "vkMapMemory (the overlay)");
}

static void drop_buffer(VkBuffer *buf, VkDeviceMemory *mem, void **map) {
    if (*buf) {
        vkUnmapMemory(gpu_vk.device, *mem);
        vkDestroyBuffer(gpu_vk.device, *buf, NULL);
        vkFreeMemory(gpu_vk.device, *mem, NULL);
    }
    *buf = VK_NULL_HANDLE, *mem = VK_NULL_HANDLE, *map = NULL;
}

static VkShaderModule shader_module(const char *name) {
    const GpuShader *s = gpu_shader(name);
    if (!s) rt_fatal("gfx: vulkan: the overlay's shader %s is not in the program", name);
    VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = s->size, .pCode = s->code};
    VkShaderModule m;
    GPU_VK_OK(vkCreateShaderModule(gpu_vk.device, &ci, NULL, &m), "vkCreateShaderModule (the overlay)");
    return m;
}

static void make_pipeline(VkFormat format) {
    VkShaderModule vs = shader_module("overlay_vert"), fs = shader_module("overlay_frag");
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs,
         .pName = "main"}};
    VkVertexInputBindingDescription bind = {.binding = 0, .stride = sizeof(UiVertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attr[3] = {
        {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(UiVertex, x)},
        {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(UiVertex, u)},
        {.location = 2, .binding = 0, .format = VK_FORMAT_R8G8B8A8_UNORM, .offset = offsetof(UiVertex, color)}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                               .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bind,
                                               .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attr};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                            .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
                                                 .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState ba = {
        .blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_A_BIT};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = 1, .pAttachments = &ba};
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                        .colorAttachmentCount = 1, .pColorAttachmentFormats = &format};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
                                       .stageCount = 2, .pStages = stages, .pVertexInputState = &vi,
                                       .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
                                       .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &ds,
                                       .layout = g_layout};
    GPU_VK_OK(vkCreateGraphicsPipelines(gpu_vk.device, VK_NULL_HANDLE, 1, &pi, NULL, &g_pipeline),
              "vkCreateGraphicsPipelines (the overlay)");
    vkDestroyShaderModule(gpu_vk.device, vs, NULL);
    vkDestroyShaderModule(gpu_vk.device, fs, NULL);
    g_format = format;
}

// ---------- the texture ops (run inside ui_menu_frame, on the thread that builds the frame)
static void free_texture(Tex *t) {
    if (!t->live) return;
    vkFreeDescriptorSets(gpu_vk.device, g_pool, 1, &t->set);
    vkDestroyImageView(gpu_vk.device, t->view, NULL);
    vkDestroyImage(gpu_vk.device, t->image, NULL);
    vkFreeMemory(gpu_vk.device, t->mem, NULL);
    *t = (Tex){0};
}

// w x h texels of 1 or 4 bytes at src (rows pitch bytes apart) as tightly packed RGBA; alpha-only texels become white with
// that alpha.
static uint8_t *expand(const uint8_t *src, int w, int h, int alpha_only, int pitch) {
    uint8_t *out = malloc((size_t)w * h * 4);
    if (!out) rt_fatal("gfx: vulkan: out of memory for an overlay texture upload");
    for (int y = 0; y < h; y++) {
        const uint8_t *row = src + (size_t)y * pitch;
        uint8_t *dst = out + (size_t)y * w * 4;
        if (alpha_only) {
            for (int x = 0; x < w; x++) dst[4 * x] = dst[4 * x + 1] = dst[4 * x + 2] = 255, dst[4 * x + 3] = row[x];
        } else {
            memcpy(dst, row, (size_t)w * 4);
        }
    }
    return out;
}

static void queue_upload(int tex, int x, int y, int w, int h, uint8_t *rgba) {
    if (g_nup == MAX_UPLOADS) {   // more waiting than a frame ever makes: fold the oldest into nothing is wrong, so say it
        rt_fatal("gfx: vulkan: the overlay has %d texture uploads waiting", MAX_UPLOADS);
    }
    g_up[g_nup++] = (Upload){tex, x, y, w, h, rgba};
}

static uint64_t tex_create(int width, int height, int alpha_only, const uint8_t *pixels) {
    plat_mutex_lock(&g_mu);
    int i = 0;
    while (i < MAX_TEXTURES && g_tex[i].live) i++;
    if (i == MAX_TEXTURES) {
        plat_mutex_unlock(&g_mu);
        return 0;
    }
    Tex *t = &g_tex[i];
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {(uint32_t)width, (uint32_t)height, 1},
                            .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    GPU_VK_OK(vkCreateImage(gpu_vk.device, &ci, NULL, &t->image), "vkCreateImage (an overlay texture)");
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(gpu_vk.device, t->image, &mr);
    uint32_t type = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) type = memory_type(mr.memoryTypeBits, 0);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                  .memoryTypeIndex = type};
    GPU_VK_OK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &t->mem), "vkAllocateMemory (an overlay texture)");
    GPU_VK_OK(vkBindImageMemory(gpu_vk.device, t->image, t->mem, 0), "vkBindImageMemory (an overlay texture)");
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->image,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    GPU_VK_OK(vkCreateImageView(gpu_vk.device, &vi, NULL, &t->view), "vkCreateImageView (an overlay texture)");
    VkDescriptorSetAllocateInfo set_alloc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = g_pool,
                                      .descriptorSetCount = 1, .pSetLayouts = &g_set_layout};
    GPU_VK_OK(vkAllocateDescriptorSets(gpu_vk.device, &set_alloc, &t->set), "vkAllocateDescriptorSets (an overlay texture)");
    VkDescriptorImageInfo di = {.sampler = g_sampler, .imageView = t->view,
                                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = t->set, .dstBinding = 0,
                              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              .pImageInfo = &di};
    vkUpdateDescriptorSets(gpu_vk.device, 1, &w, 0, NULL);
    t->live = 1, t->ready = 0, t->w = width, t->h = height;
    queue_upload(i, 0, 0, width, height, expand(pixels, width, height, alpha_only, alpha_only ? width : width * 4));
    plat_mutex_unlock(&g_mu);
    return (uint64_t)i + 1;
}

static void tex_update(uint64_t id, int x, int y, int w, int h, int alpha_only, const uint8_t *pixels, int pitch) {
    plat_mutex_lock(&g_mu);
    if (id >= 1 && id <= MAX_TEXTURES && g_tex[id - 1].live && w > 0 && h > 0)
        queue_upload((int)id - 1, x, y, w, h, expand(pixels, w, h, alpha_only, pitch));
    plat_mutex_unlock(&g_mu);
}

static void tex_destroy(uint64_t id) {
    plat_mutex_lock(&g_mu);
    if (id >= 1 && id <= MAX_TEXTURES && g_tex[id - 1].live) {
        gpu_sync_wait(g_last_value, UINT64_MAX);   // what could still read it is done
        for (int i = 0; i < g_nup; i++)
            if (g_up[i].tex == (int)id - 1) free(g_up[i].rgba), g_up[i].rgba = NULL, g_up[i].tex = -1;
        free_texture(&g_tex[id - 1]);
    }
    plat_mutex_unlock(&g_mu);
}

static const UiTextureOps k_ops = {tex_create, tex_update, tex_destroy};

// ---------- init and close
int vko_init(const VkwUi *ui, VkFormat color_format) {
    g_ui = ui;
    if (!ui || !ui->frame) return 0;
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR,
                              .minFilter = VK_FILTER_LINEAR, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 1.0f,
                              .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK};
    GPU_VK_OK(vkCreateSampler(gpu_vk.device, &si, NULL, &g_sampler), "vkCreateSampler (the overlay)");
    VkDescriptorSetLayoutBinding b = {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .bindingCount = 1, .pBindings = &b};
    GPU_VK_OK(vkCreateDescriptorSetLayout(gpu_vk.device, &li, NULL, &g_set_layout), "vkCreateDescriptorSetLayout (the overlay)");
    VkPushConstantRange pc = {.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = 4 * sizeof(float)};
    VkPipelineLayoutCreateInfo pl = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &g_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};
    GPU_VK_OK(vkCreatePipelineLayout(gpu_vk.device, &pl, NULL, &g_layout), "vkCreatePipelineLayout (the overlay)");
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = MAX_TEXTURES};
    VkDescriptorPoolCreateInfo dp = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                     .maxSets = MAX_TEXTURES, .poolSizeCount = 1, .pPoolSizes = &ps};
    GPU_VK_OK(vkCreateDescriptorPool(gpu_vk.device, &dp, NULL, &g_pool), "vkCreateDescriptorPool (the overlay)");
    make_pipeline(color_format);
    for (int i = 0; i < VKO_FRAMES; i++) {
        host_buffer(VERTEX_BYTES, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &g_fb[i].vb, &g_fb[i].vm, &g_fb[i].vmap);
        host_buffer(INDEX_BYTES, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &g_fb[i].ib, &g_fb[i].im, &g_fb[i].imap);
        g_fb[i].vcap = VERTEX_BYTES, g_fb[i].icap = INDEX_BYTES;
    }
    if (ui->set_texture_ops) ui->set_texture_ops(&k_ops);
    return 1;
}

void vko_format(VkFormat color_format) {
    if (!g_pipeline || color_format == g_format) return;
    vkDestroyPipeline(gpu_vk.device, g_pipeline, NULL);
    g_pipeline = VK_NULL_HANDLE;
    make_pipeline(color_format);
}

void vko_close(void) {
    if (g_ui && g_ui->set_texture_ops) g_ui->set_texture_ops(NULL);
    for (int i = 0; i < g_nup; i++) free(g_up[i].rgba);
    g_nup = 0;
    for (int i = 0; i < MAX_TEXTURES; i++) free_texture(&g_tex[i]);
    for (int i = 0; i < VKO_FRAMES; i++) {
        drop_buffer(&g_fb[i].vb, &g_fb[i].vm, &g_fb[i].vmap);
        drop_buffer(&g_fb[i].ib, &g_fb[i].im, &g_fb[i].imap);
    }
    if (g_pipeline) vkDestroyPipeline(gpu_vk.device, g_pipeline, NULL);
    if (g_layout) vkDestroyPipelineLayout(gpu_vk.device, g_layout, NULL);
    if (g_pool) vkDestroyDescriptorPool(gpu_vk.device, g_pool, NULL);
    if (g_set_layout) vkDestroyDescriptorSetLayout(gpu_vk.device, g_set_layout, NULL);
    if (g_sampler) vkDestroySampler(gpu_vk.device, g_sampler, NULL);
    g_pipeline = VK_NULL_HANDLE, g_layout = VK_NULL_HANDLE, g_pool = VK_NULL_HANDLE, g_set_layout = VK_NULL_HANDLE;
    g_sampler = VK_NULL_HANDLE;
    g_ui = NULL;
}

// ---------- a frame
static void grow(VkBuffer *buf, VkDeviceMemory *mem, void **map, VkDeviceSize *cap, VkDeviceSize need,
                 VkBufferUsageFlags usage) {
    if (need <= *cap) return;
    VkDeviceSize size = *cap;
    while (size < need) size *= 2;
    drop_buffer(buf, mem, map);   // the caller waited for the frame that used it
    host_buffer(size, usage, buf, mem, map);
    *cap = size;
    rt_log("gfx: vulkan: the overlay's buffer grew to %llu bytes", (unsigned long long)size);
}

void vko_upload(int frame, const UiDrawData *dd) {
    FrameBufs *f = &g_fb[frame];
    grow(&f->vb, &f->vm, &f->vmap, &f->vcap, (VkDeviceSize)dd->vertex_count * sizeof(UiVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    grow(&f->ib, &f->im, &f->imap, &f->icap, (VkDeviceSize)dd->index_count * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    memcpy(f->vmap, dd->vertices, (size_t)dd->vertex_count * sizeof(UiVertex));
    memcpy(f->imap, dd->indices, (size_t)dd->index_count * sizeof(uint32_t));
}

int vko_uploads_waiting(void) {
    plat_mutex_lock(&g_mu);
    int n = g_nup;
    plat_mutex_unlock(&g_mu);
    return n;
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; } Staging;
static void staging_done(void *arg, uint64_t value) {
    (void)value;
    Staging *s = arg;
    drop_buffer(&s->buf, &s->mem, &s->map);
    free(s);
}

static void image_barrier(VkCommandBuffer cmd, VkImage image, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                          VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = src_stage,
                                .srcAccessMask = src_access, .dstStageMask = dst_stage, .dstAccessMask = dst_access,
                                .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cmd, &di);
}

// The uploads waiting: a staging buffer each, copied into the image behind a layout change to and from the shader's.
static void record_uploads(VkCommandBuffer cmd) {
    plat_mutex_lock(&g_mu);
    for (int i = 0; i < g_nup; i++) {
        Upload *u = &g_up[i];
        if (!u->rgba || u->tex < 0) continue;
        Tex *t = &g_tex[u->tex];
        Staging *s = calloc(1, sizeof *s);
        if (!s) rt_fatal("gfx: vulkan: out of memory for an overlay upload");
        size_t bytes = (size_t)u->w * u->h * 4;
        host_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &s->buf, &s->mem, &s->map);
        memcpy(s->map, u->rgba, bytes);
        image_barrier(cmd, t->image, t->ready ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
                      0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      t->ready ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy r = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                               .imageOffset = {u->x, u->y, 0}, .imageExtent = {(uint32_t)u->w, (uint32_t)u->h, 1}};
        vkCmdCopyBufferToImage(cmd, s->buf, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        image_barrier(cmd, t->image, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        t->ready = 1;
        gpu_vk_effect(staging_done, s);
        free(u->rgba);
        u->rgba = NULL;
    }
    g_nup = 0;
    plat_mutex_unlock(&g_mu);
}

void vko_record(VkCommandBuffer cmd, uint64_t value, int frame, VkImageView view, VkExtent2D extent,
                const UiDrawData *dd) {
    record_uploads(cmd);
    g_last_value = value;
    VkRenderingAttachmentInfo att = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view,
                                     .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, extent}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &att};
    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeline);
    VkViewport vp = {0, 0, (float)extent.width, (float)extent.height, 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    float push[4] = {2.0f / (float)extent.width, 2.0f / (float)extent.height, -1.0f, -1.0f};
    vkCmdPushConstants(cmd, g_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof push, push);
    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &g_fb[frame].vb, &zero);
    vkCmdBindIndexBuffer(cmd, g_fb[frame].ib, 0, VK_INDEX_TYPE_UINT32);
    uint64_t bound = 0;
    for (uint32_t i = 0; i < dd->cmd_count; i++) {
        const UiCmd *c = &dd->cmds[i];
        float x0 = c->clip[0] < 0 ? 0 : c->clip[0], y0 = c->clip[1] < 0 ? 0 : c->clip[1];
        float x1 = c->clip[2] > (float)extent.width ? (float)extent.width : c->clip[2];
        float y1 = c->clip[3] > (float)extent.height ? (float)extent.height : c->clip[3];
        if (x1 <= x0 || y1 <= y0 || c->texture < 1 || c->texture > MAX_TEXTURES || !g_tex[c->texture - 1].live) continue;
        VkRect2D sc = {{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
        vkCmdSetScissor(cmd, 0, 1, &sc);
        if (c->texture != bound) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_layout, 0, 1, &g_tex[c->texture - 1].set, 0, NULL);
            bound = c->texture;
        }
        vkCmdDrawIndexed(cmd, c->index_count, 1, c->first_index, 0, 0);
    }
    vkCmdEndRendering(cmd);
}
