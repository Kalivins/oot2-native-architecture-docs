// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Build costs (spike S3, D12): graphics pipeline libraries with fast and
// optimised linking, monolithic pipelines, shader objects; and the launch path
// of prepared pipelines through a VkPipelineCache or pipeline binaries.
//
// Every pipeline here follows D12: vertex input, topology, viewport, culling,
// depth, stencil and blend state are dynamic wherever the device allows, so a
// pipeline is its two stages, its layout and its attachment formats.

#include "vkpc.h"

#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// ---- dynamic state, by library part ----------------------------------------------

static const VkDynamicState DYN_VERTEX_INPUT[] = {
    VK_DYNAMIC_STATE_VERTEX_INPUT_EXT, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
};
static const VkDynamicState DYN_PRE_RASTER[] = {
    VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT, VK_DYNAMIC_STATE_LINE_WIDTH,
    VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
    VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE, VK_DYNAMIC_STATE_POLYGON_MODE_EXT,
    VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT,
};
static const VkDynamicState DYN_FRAGMENT[] = {
    VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
    VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP,
    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    VK_DYNAMIC_STATE_DEPTH_BOUNDS,
};
static const VkDynamicState DYN_OUTPUT[] = {
    VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT, VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
    VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT, VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,
};

void dynamic_states(VkGraphicsPipelineLibraryFlagsEXT part, VkDynamicState *out, uint32_t *n) {
    *n = 0;
#define ADD(list)                                                                   \
    for (size_t i_ = 0; i_ < sizeof list / sizeof list[0]; i_++) out[(*n)++] = list[i_]
    if (part & VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT) ADD(DYN_VERTEX_INPUT);
    if (part & VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT) ADD(DYN_PRE_RASTER);
    if (part & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) ADD(DYN_FRAGMENT);
    if (part & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT) ADD(DYN_OUTPUT);
#undef ADD
}

#define ALL_PARTS                                                                                                 \
    (VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT |                                               \
     VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT |                                             \
     VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT | VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT)

// Every piece of state a draw needs set once per command buffer, for pipelines
// built with the lists above, or (shader_objects) for shader objects, which have
// no static state at all.
void set_all_dynamic_state(VkCommandBuffer cb, uint32_t width, uint32_t height, uint32_t targets, int shader_objects) {
    VkViewport vp = {0, 0, (float)width, (float)height, 0, 1};
    VkRect2D sc = {{0, 0}, {width, height}};
    vkCmdSetViewportWithCount(cb, 1, &vp);
    vkCmdSetScissorWithCount(cb, 1, &sc);
    vkCmdSetLineWidth(cb, 1.0f);
    vkCmdSetDepthBias(cb, 0, 0, 0);
    vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
    vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetDepthBiasEnable(cb, VK_FALSE);
    vkCmdSetRasterizerDiscardEnable(cb, VK_FALSE);
    vkCmdSetPolygonModeEXT(cb, VK_POLYGON_MODE_FILL);
    vkCmdSetDepthClampEnableEXT(cb, VK_FALSE);
    vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetPrimitiveRestartEnable(cb, VK_FALSE);
    vkCmdSetDepthTestEnable(cb, VK_FALSE);
    vkCmdSetDepthWriteEnable(cb, VK_FALSE);
    vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_ALWAYS);
    vkCmdSetDepthBoundsTestEnable(cb, VK_FALSE);
    vkCmdSetStencilTestEnable(cb, VK_FALSE);
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP,
                      VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
    vkCmdSetDepthBounds(cb, 0.0f, 1.0f);
    float bc[4] = {0, 0, 0, 0};
    vkCmdSetBlendConstants(cb, bc);
    vkCmdSetLogicOpEnableEXT(cb, VK_FALSE);
    VkBool32 en[8] = {0};
    VkColorBlendEquationEXT eq[8];
    VkColorComponentFlags wm[8];
    for (int i = 0; i < 8; i++) {
        eq[i] = (VkColorBlendEquationEXT){VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
                                          VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD};
        wm[i] = 0xF;
    }
    uint32_t t = targets ? targets : 1;
    vkCmdSetColorBlendEnableEXT(cb, 0, t, en);
    vkCmdSetColorBlendEquationEXT(cb, 0, t, eq);
    vkCmdSetColorWriteMaskEXT(cb, 0, t, wm);
    if (shader_objects) {
        vkCmdSetRasterizationSamplesEXT(cb, VK_SAMPLE_COUNT_1_BIT);
        VkSampleMask mask = 0xFFFFFFFF;
        vkCmdSetSampleMaskEXT(cb, VK_SAMPLE_COUNT_1_BIT, &mask);
        vkCmdSetAlphaToCoverageEnableEXT(cb, VK_FALSE);
        vkCmdSetAlphaToOneEnableEXT(cb, VK_FALSE);
        vkCmdSetDepthBiasEnable(cb, VK_FALSE);
    }
}

// ---- layouts ------------------------------------------------------------------------

typedef struct {
    VkDescriptorSetLayout set;      // the pair's per-draw set
    VkDescriptorSetLayout global;
    VkPipelineLayout layout;
    uint32_t pushed;                // the per-draw set is a push-descriptor set
} PairLayout;

static uint32_t set_descriptors(const SetDesc *d) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < d->n; i++) n += d->b[i].count;
    return n;
}

// Layouts of a job: the pair's set at its number, the global set at the other.
static PairLayout *layouts_create(Job *job, VkDescriptorSetLayout *global_out) {
    VkDescriptorSetLayout global = VK_NULL_HANDLE;
    if (job->has_global) global = set_layout_create(&job->global, 0, 1);
    VkDescriptorSetLayout empty;
    SetDesc none = {0};
    empty = set_layout_create(&none, 0, 0);
    PairLayout *pl = calloc(job->nsets, sizeof *pl);
    for (uint32_t i = 0; i < job->nsets; i++) {
        pl[i].pushed = set_descriptors(&job->sets[i]) <= 32;
        pl[i].set = set_layout_create(&job->sets[i], pl[i].pushed ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0, 0);
        pl[i].global = global;
        VkDescriptorSetLayout sets[2];
        uint32_t nsets = 1;
        uint32_t per_draw = job->has_global ? 1 - job->global_set : 0;
        sets[per_draw] = pl[i].set;
        if (job->has_global) {
            sets[job->global_set] = global;
            nsets = 2;
        }
        VkPushConstantRange pcr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, job->push_bytes};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = nsets,
                                         .pSetLayouts = sets, .pushConstantRangeCount = job->push_bytes ? 1 : 0,
                                         .pPushConstantRanges = &pcr};
        VK_CHECK(vkCreatePipelineLayout(dev, &ci, NULL, &pl[i].layout));
    }
    vkDestroyDescriptorSetLayout(dev, empty, NULL);
    if (global_out) *global_out = global;
    return pl;
}

// ---- pipeline pieces -----------------------------------------------------------------

typedef struct {
    VkPipelineVertexInputStateCreateInfo vis;
    VkPipelineInputAssemblyStateCreateInfo ias;
    VkPipelineViewportStateCreateInfo vps;
    VkPipelineRasterizationStateCreateInfo rs;
    VkPipelineMultisampleStateCreateInfo ms;
    VkPipelineDepthStencilStateCreateInfo ds;
    VkPipelineColorBlendAttachmentState cba[8];
    VkPipelineColorBlendStateCreateInfo cbs;
    VkFormat colour[8];
    VkPipelineRenderingCreateInfo rendering;
    VkDynamicState dyn[64];
    VkPipelineDynamicStateCreateInfo dynamic;
    VkPipelineShaderStageCreateInfo stages[2];
    VkGraphicsPipelineLibraryCreateInfoEXT lib;
    VkPipelineLibraryCreateInfoKHR libs;
    VkPipelineCreateFlags2CreateInfoKHR flags2;
    VkGraphicsPipelineCreateInfo gp;
} PipeInfo;

// Fills a create info for the parts `part` of a pipeline (ALL_PARTS for a
// monolithic one). vs and fs may be VK_NULL_HANDLE where the part has no stage.
static void pipe_info(PipeInfo *p, const Job *job, VkGraphicsPipelineLibraryFlagsEXT part, VkShaderModule vs,
                      VkShaderModule fs, VkPipelineLayout layout, uint32_t targets, VkFormat format, VkPipelineCreateFlags flags) {
    memset(p, 0, sizeof *p);
    p->vis.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    p->ias = (VkPipelineInputAssemblyStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    p->vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    p->rs = (VkPipelineRasterizationStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                     .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1.0f};
    p->ms = (VkPipelineMultisampleStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                   .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    p->ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    for (uint32_t i = 0; i < 8; i++) {
        p->cba[i].colorWriteMask = 0xF;
        p->colour[i] = format;
    }
    p->cbs = (VkPipelineColorBlendStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                   .attachmentCount = targets, .pAttachments = p->cba};
    p->rendering = (VkPipelineRenderingCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                                   .colorAttachmentCount = targets, .pColorAttachmentFormats = p->colour,
                                                   .depthAttachmentFormat = job->depth_format,
                                                   .stencilAttachmentFormat = job->depth_format == VK_FORMAT_D32_SFLOAT
                                                                                  ? VK_FORMAT_UNDEFINED
                                                                                  : job->depth_format};
    uint32_t nd;
    dynamic_states(part, p->dyn, &nd);
    p->dynamic = (VkPipelineDynamicStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                                    .dynamicStateCount = nd, .pDynamicStates = p->dyn};
    uint32_t ns = 0;
    if (vs && (part & VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT))
        p->stages[ns++] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                            .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"};
    if (fs && (part & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT))
        p->stages[ns++] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                            .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"};
    void *chain = &p->rendering;
    if (part != ALL_PARTS) {
        p->lib = (VkGraphicsPipelineLibraryCreateInfoEXT){.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
                                                          .pNext = chain, .flags = part};
        chain = &p->lib;
    }
    int pre = part & VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT;
    int frag = part & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT;
    int out = part & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT;
    int vin = part & VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT;
    p->gp = (VkGraphicsPipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = chain, .flags = flags,
        .stageCount = ns, .pStages = ns ? p->stages : NULL, .pVertexInputState = vin ? &p->vis : NULL,
        .pInputAssemblyState = vin ? &p->ias : NULL, .pViewportState = pre ? &p->vps : NULL,
        .pRasterizationState = pre ? &p->rs : NULL, .pMultisampleState = (frag || out) ? &p->ms : NULL,
        .pDepthStencilState = frag ? &p->ds : NULL, .pColorBlendState = out ? &p->cbs : NULL,
        .pDynamicState = &p->dynamic, .layout = (pre || frag) ? layout : VK_NULL_HANDLE};
}

VkPipeline pipeline_for(const Job *job, int method, VkShaderModule vs, VkShaderModule fs, VkPipelineLayout layout,
                        uint32_t targets, VkFormat format, VkPipeline *libs, uint32_t *nlibs) {
    PipeInfo pi;
    VkPipeline p;
    if (method == 0) {
        pipe_info(&pi, job, ALL_PARTS, vs, fs, layout, targets, format, 0);
        VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi.gp, NULL, &p));
        return p;
    }
    static const VkGraphicsPipelineLibraryFlagsEXT parts[4] = {
        VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
    VkPipeline part[4];
    for (int i = 0; i < 4; i++) {
        pipe_info(&pi, job, parts[i], vs, fs, layout, targets, format,
                  VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT);
        VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi.gp, NULL, &part[i]));
        libs[(*nlibs)++] = part[i];
    }
    VkPipelineLibraryCreateInfoKHR li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 4,
                                         .pLibraries = part};
    VkGraphicsPipelineCreateInfo gp = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li,
                                       .flags = method == 2 ? VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT : 0,
                                       .layout = layout};
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, NULL, &p));
    return p;
}

static void shader_info(VkShaderCreateInfoEXT *si, const Module *m, VkShaderCreateFlagsEXT flags, VkShaderStageFlags next,
                        const VkDescriptorSetLayout *sets, uint32_t nsets, const VkPushConstantRange *pcr);

void shader_objects_for(const Job *job, const Module *vs, const Module *fs, const VkDescriptorSetLayout *sets,
                        uint32_t nsets, VkShaderEXT out[2]) {
    VkPushConstantRange pcr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, job->push_bytes};
    VkShaderCreateInfoEXT si[2];
    shader_info(&si[0], vs, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sets, nsets, &pcr);
    shader_info(&si[1], fs, 0, 0, sets, nsets, &pcr);
    VK_CHECK(vkCreateShadersEXT(dev, 1, &si[0], NULL, &out[0]));
    VK_CHECK(vkCreateShadersEXT(dev, 1, &si[1], NULL, &out[1]));
}

// ---- build mode ------------------------------------------------------------------------

typedef struct {
    Job *job;
    PairLayout *pl;
    VkShaderModule *mods;
    VkPipeline vi;              // vertex input library, shared
    VkPipeline *out;            // output library per (format, targets): index format * 9 + targets
    VkPipeline *pre, *frag, *fast, *opt, *mono;
    VkShaderEXT *so_vs, *so_fs, *so_link;
    const char *phase;
    VkPipelineCache cache;      // `param pipeline_cache`: the application cache, read if it exists, saved after
    uint8_t **bin_in;           // `param shader_binaries`: per item and stage, a shader object binary read back
    size_t *bin_in_size;
    FILE *bin_out;              // or the file the binaries are written to
    CRITICAL_SECTION bin_lock;
} BuildCtx;

static VkPipelineCreateFlags lib_flags(void) {
    return VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;
}

static double create_pipeline(VkPipelineCache cache, PipeInfo *pi, VkPipeline *out) {
    double t0 = now_ms();
    VK_CHECK(vkCreateGraphicsPipelines(dev, cache, 1, &pi->gp, NULL, out));
    return now_ms() - t0;
}

static void w_module(void *c, uint32_t i, uint32_t w) {
    BuildCtx *b = c;
    double t0 = now_ms();
    b->mods[i] = module_create(&b->job->modules[i]);
    out_row(b->phase, "module", i, now_ms() - t0, (int)w, b->job->modules[i].stage == VK_SHADER_STAGE_VERTEX_BIT ? "v" : "f");
}

static void w_pre(void *c, uint32_t i, uint32_t w) {
    BuildCtx *b = c;
    Pair *p = &b->job->pairs[i];
    PipeInfo pi;
    pipe_info(&pi, b->job, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT, b->mods[p->vs], VK_NULL_HANDLE,
              b->pl[p->set].layout, 0, VK_FORMAT_UNDEFINED, lib_flags());
    out_row(b->phase, "pre_raster_lib", i, create_pipeline(b->cache, &pi, &b->pre[i]), (int)w, NULL);
}

static void w_frag(void *c, uint32_t i, uint32_t w) {
    BuildCtx *b = c;
    Pair *p = &b->job->pairs[i];
    PipeInfo pi;
    pipe_info(&pi, b->job, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, VK_NULL_HANDLE, b->mods[p->fs],
              b->pl[p->set].layout, 0, VK_FORMAT_UNDEFINED, lib_flags());
    out_row(b->phase, "fragment_lib", i, create_pipeline(b->cache, &pi, &b->frag[i]), (int)w, NULL);
}

static void link_one(BuildCtx *b, uint32_t i, uint32_t w, int optimised) {
    Pair *p = &b->job->pairs[i];
    VkPipeline libs[4] = {b->vi, b->pre[i], b->frag[i], b->out[p->format * 9 + p->targets]};
    VkPipelineLibraryCreateInfoKHR li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 4,
                                         .pLibraries = libs};
    VkGraphicsPipelineCreateInfo gp = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li,
                                       .flags = optimised ? VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT : 0,
                                       .layout = b->pl[p->set].layout};
    double t0 = now_ms();
    VK_CHECK(vkCreateGraphicsPipelines(dev, b->cache, 1, &gp, NULL, optimised ? &b->opt[i] : &b->fast[i]));
    out_row(b->phase, optimised ? "optimised_link" : "fast_link", i, now_ms() - t0, (int)w, NULL);
}

static void w_fast(void *c, uint32_t i, uint32_t w) { link_one(c, i, w, 0); }
static void w_opt(void *c, uint32_t i, uint32_t w) { link_one(c, i, w, 1); }

static void w_mono(void *c, uint32_t i, uint32_t w) {
    BuildCtx *b = c;
    Pair *p = &b->job->pairs[i];
    PipeInfo pi;
    pipe_info(&pi, b->job, ALL_PARTS, b->mods[p->vs], b->mods[p->fs], b->pl[p->set].layout, p->targets,
              b->job->formats[p->format], 0);
    out_row(b->phase, "monolithic", i, create_pipeline(b->cache, &pi, &b->mono[i]), (int)w, NULL);
}

static void shader_info(VkShaderCreateInfoEXT *si, const Module *m, VkShaderCreateFlagsEXT flags, VkShaderStageFlags next,
                        const VkDescriptorSetLayout *sets, uint32_t nsets, const VkPushConstantRange *pcr) {
    *si = (VkShaderCreateInfoEXT){.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .flags = flags, .stage = m->stage,
                                  .nextStage = next, .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT, .codeSize = m->size,
                                  .pCode = m->code, .pName = "main", .setLayoutCount = nsets, .pSetLayouts = sets,
                                  .pushConstantRangeCount = pcr->size ? 1 : 0, .pPushConstantRanges = pcr};
}

static uint32_t pair_sets(BuildCtx *b, const Pair *p, VkDescriptorSetLayout *sets) {
    const Job *job = b->job;
    if (!job->has_global) {
        sets[0] = b->pl[p->set].set;
        return 1;
    }
    sets[1 - job->global_set] = b->pl[p->set].set;
    sets[job->global_set] = b->pl[p->set].global;
    return 2;
}

// Shader object binaries, as records: a 16-byte head (item, which: 0 vertex, 1
// fragment, 2 and 3 the linked pair, size, zero), then the data padded to 16
// bytes, so that every binary stays 16-byte aligned as Vulkan requires.
static void binary_save(BuildCtx *b, uint32_t i, uint32_t which, VkShaderEXT s) {
    if (!b->bin_out) return;
    size_t size = 0;
    VK_CHECK(vkGetShaderBinaryDataEXT(dev, s, &size, NULL));
    size_t padded = (size + 15) & ~(size_t)15;
    uint8_t *data = _aligned_malloc(padded + 16, 16);
    memset(data, 0, padded + 16);
    VK_CHECK(vkGetShaderBinaryDataEXT(dev, s, &size, data + 16));
    uint32_t head[4] = {i, which, (uint32_t)size, 0};
    memcpy(data, head, 16);
    EnterCriticalSection(&b->bin_lock);
    if (fwrite(data, 1, padded + 16, b->bin_out) != padded + 16) die(1, "short write of the shader binaries");
    LeaveCriticalSection(&b->bin_lock);
    _aligned_free(data);
}

static void binary_code(BuildCtx *b, VkShaderCreateInfoEXT *si, uint32_t i, uint32_t which) {
    if (!b->bin_in) return;
    uint8_t *code = b->bin_in[4 * i + which];
    if (!code) die(1, "item %u has no shader binary %u", i, which);
    si->codeType = VK_SHADER_CODE_TYPE_BINARY_EXT;
    si->pCode = code;
    si->codeSize = b->bin_in_size[4 * i + which];
}

static void w_objects(void *c, uint32_t i, uint32_t w) {
    BuildCtx *b = c;
    Pair *p = &b->job->pairs[i];
    VkDescriptorSetLayout sets[2];
    uint32_t ns = pair_sets(b, p, sets);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, b->job->push_bytes};
    VkShaderCreateInfoEXT si;
    shader_info(&si, &b->job->modules[p->vs], 0, VK_SHADER_STAGE_FRAGMENT_BIT, sets, ns, &pcr);
    binary_code(b, &si, i, 0);
    double t0 = now_ms();
    VK_CHECK(vkCreateShadersEXT(dev, 1, &si, NULL, &b->so_vs[i]));
    out_row(b->phase, "shader_object_vs", i, now_ms() - t0, (int)w, NULL);
    shader_info(&si, &b->job->modules[p->fs], 0, 0, sets, ns, &pcr);
    binary_code(b, &si, i, 1);
    t0 = now_ms();
    VK_CHECK(vkCreateShadersEXT(dev, 1, &si, NULL, &b->so_fs[i]));
    out_row(b->phase, "shader_object_fs", i, now_ms() - t0, (int)w, NULL);
    binary_save(b, i, 0, b->so_vs[i]);
    binary_save(b, i, 1, b->so_fs[i]);
}

static void w_objects_linked(void *c, uint32_t i, uint32_t w) {
    BuildCtx *b = c;
    Pair *p = &b->job->pairs[i];
    VkDescriptorSetLayout sets[2];
    uint32_t ns = pair_sets(b, p, sets);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, b->job->push_bytes};
    VkShaderCreateInfoEXT si[2];
    shader_info(&si[0], &b->job->modules[p->vs], VK_SHADER_CREATE_LINK_STAGE_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT, sets, ns, &pcr);
    shader_info(&si[1], &b->job->modules[p->fs], VK_SHADER_CREATE_LINK_STAGE_BIT_EXT, 0, sets, ns, &pcr);
    binary_code(b, &si[0], i, 2);
    binary_code(b, &si[1], i, 3);
    double t0 = now_ms();
    VK_CHECK(vkCreateShadersEXT(dev, 2, si, NULL, &b->so_link[2 * i]));
    out_row(b->phase, "shader_object_linked", i, now_ms() - t0, (int)w, NULL);
    binary_save(b, i, 2, b->so_link[2 * i]);
    binary_save(b, i, 3, b->so_link[2 * i + 1]);
}

static void *read_whole(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    _fseeki64(f, 0, SEEK_END);
    long long n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    void *p = _aligned_malloc(n > 0 ? (size_t)n : 16, 16);
    if (!p || (n > 0 && fread(p, 1, (size_t)n, f) != (size_t)n)) die(1, "cannot read %s", path);
    fclose(f);
    *size = (size_t)n;
    return p;
}

// The application's caches of a build run: a VkPipelineCache and shader object
// binaries, read when their files exist (a warm run) and written otherwise.
static void caches_open(BuildCtx *b, Job *job) {
    InitializeCriticalSection(&b->bin_lock);
    const char *pc_path = job_param(job, "pipeline_cache", NULL);
    if (pc_path) {
        size_t size = 0;
        void *data = read_whole(pc_path, &size);
        double t0 = now_ms();
        VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                                        .initialDataSize = data ? size : 0, .pInitialData = data};
        VK_CHECK(vkCreatePipelineCache(dev, &ci, NULL, &b->cache));
        summary_num("pipeline_cache_in_bytes", data ? (double)size : 0);
        summary_num("pipeline_cache_create_ms", now_ms() - t0);
        _aligned_free(data);
    }
    const char *bin_path = strncmp(b->phase, "objects", 7) ? NULL : job_param(job, "shader_binaries", NULL);
    if (bin_path) {
        size_t size = 0;
        uint8_t *data = read_whole(bin_path, &size);
        if (data) {
            b->bin_in = calloc(4 * job->npairs, sizeof *b->bin_in);
            b->bin_in_size = calloc(4 * job->npairs, sizeof *b->bin_in_size);
            for (size_t at = 0; at + 16 <= size;) {
                uint32_t head[4];
                memcpy(head, data + at, 16);
                size_t padded = ((size_t)head[2] + 15) & ~(size_t)15;
                if (head[0] >= job->npairs || head[1] > 3 || at + 16 + padded > size) die(1, "shader binaries damaged");
                b->bin_in[4 * head[0] + head[1]] = data + at + 16;
                b->bin_in_size[4 * head[0] + head[1]] = head[2];
                at += 16 + padded;
            }
            summary_num("shader_binaries_in_bytes", (double)size);
        } else {
            b->bin_out = fopen(bin_path, "wb");
            if (!b->bin_out) die(1, "cannot write %s", bin_path);
        }
    }
}

static void caches_close(BuildCtx *b, Job *job) {
    const char *pc_path = job_param(job, "pipeline_cache", NULL);
    if (b->cache) {
        size_t size = 0;
        VK_CHECK(vkGetPipelineCacheData(dev, b->cache, &size, NULL));
        void *data = malloc(size);
        VK_CHECK(vkGetPipelineCacheData(dev, b->cache, &size, data));
        FILE *f = fopen(pc_path, "wb");
        if (!f || fwrite(data, 1, size, f) != size) die(1, "cannot write %s", pc_path);
        fclose(f);
        free(data);
        summary_num("pipeline_cache_out_bytes", (double)size);
        vkDestroyPipelineCache(dev, b->cache, NULL);
    }
    if (b->bin_out) {
        summary_num("shader_binaries_out_bytes", (double)_ftelli64(b->bin_out));
        fclose(b->bin_out);
    }
}

static void step(const char *name, uint32_t n, BuildCtx *b, WorkFn fn) {
    double ws0, ws1, priv1;
    memory_mb(&ws0, NULL, NULL);
    double wall = run_parallel(n, opt.threads, fn, b);
    memory_mb(&ws1, &priv1, NULL);
    char key[96];
    snprintf(key, sizeof key, "%s_wall_ms", name);
    summary_num(key, wall);
    snprintf(key, sizeof key, "%s_count", name);
    summary_num(key, n);
    snprintf(key, sizeof key, "%s_working_set_delta_mb", name);
    summary_num(key, ws1 - ws0);
    fprintf(stderr, "vkpipecost: %s: %u in %.0f ms, working set %.0f MB, available %.0f MB\n", name, n, wall, ws1,
            available_mb());
}

int mode_build(Job *job, const char *phase) {
    summary_begin("build");
    summary_str("phase", phase);
    BuildCtx b = {.job = job, .phase = phase};
    b.pl = layouts_create(job, NULL);
    uint32_t np = job->npairs;
    b.mods = calloc(job->nmodules, sizeof *b.mods);
    caches_open(&b, job);
    if (!strcmp(phase, "libs") || !strcmp(phase, "mono")) step("module", job->nmodules, &b, w_module);
    if (!strcmp(phase, "libs")) {
        PipeInfo pi;
        double t0 = now_ms();
        pipe_info(&pi, job, VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, VK_NULL_HANDLE, VK_NULL_HANDLE,
                  VK_NULL_HANDLE, 0, VK_FORMAT_UNDEFINED, lib_flags());
        VK_CHECK(vkCreateGraphicsPipelines(dev, b.cache, 1, &pi.gp, NULL, &b.vi));
        out_row(phase, "vertex_input_lib", 0, now_ms() - t0, 0, NULL);
        b.out = calloc(job->nformats * 9, sizeof *b.out);
        for (uint32_t i = 0; i < np; i++) {
            uint32_t k = job->pairs[i].format * 9 + job->pairs[i].targets;
            if (b.out[k]) continue;
            pipe_info(&pi, job, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT, VK_NULL_HANDLE,
                      VK_NULL_HANDLE, VK_NULL_HANDLE, job->pairs[i].targets, job->formats[job->pairs[i].format], lib_flags());
            out_row(phase, "output_lib", k, create_pipeline(b.cache, &pi, &b.out[k]), 0, NULL);
        }
        b.pre = calloc(np, sizeof *b.pre);
        b.frag = calloc(np, sizeof *b.frag);
        b.fast = calloc(np, sizeof *b.fast);
        b.opt = calloc(np, sizeof *b.opt);
        step("pre_raster_lib", np, &b, w_pre);
        step("fragment_lib", np, &b, w_frag);
        step("fast_link", np, &b, w_fast);
        step("optimised_link", np, &b, w_opt);
    } else if (!strcmp(phase, "mono")) {
        b.mono = calloc(np, sizeof *b.mono);
        step("monolithic", np, &b, w_mono);
    } else if (!strcmp(phase, "objects") || !strcmp(phase, "objects-linked")) {
        // Each in its own process: a linked build after the unlinked ones of the same
        // modules would not be cold.
        b.so_vs = calloc(np, sizeof *b.so_vs);
        b.so_fs = calloc(np, sizeof *b.so_fs);
        b.so_link = calloc(2 * np, sizeof *b.so_link);
        if (!strcmp(phase, "objects")) step("shader_object", np, &b, w_objects);
        else step("shader_object_linked", np, &b, w_objects_linked);
    } else die(1, "build phases are libs, mono, objects and objects-linked");
    caches_close(&b, job);

    double ws, priv, peak;
    memory_mb(&ws, &priv, &peak);
    summary_num("pairs", np);
    // Teardown.
    for (uint32_t i = 0; i < np; i++) {
        if (b.pre) vkDestroyPipeline(dev, b.pre[i], NULL);
        if (b.frag) vkDestroyPipeline(dev, b.frag[i], NULL);
        if (b.fast) vkDestroyPipeline(dev, b.fast[i], NULL);
        if (b.opt) vkDestroyPipeline(dev, b.opt[i], NULL);
        if (b.mono) vkDestroyPipeline(dev, b.mono[i], NULL);
        if (b.so_vs) {
            if (b.so_vs[i]) vkDestroyShaderEXT(dev, b.so_vs[i], NULL);
            if (b.so_fs[i]) vkDestroyShaderEXT(dev, b.so_fs[i], NULL);
            if (b.so_link[2 * i]) vkDestroyShaderEXT(dev, b.so_link[2 * i], NULL);
            if (b.so_link[2 * i + 1]) vkDestroyShaderEXT(dev, b.so_link[2 * i + 1], NULL);
        }
    }
    if (b.vi) vkDestroyPipeline(dev, b.vi, NULL);
    if (b.out)
        for (uint32_t k = 0; k < job->nformats * 9; k++)
            if (b.out[k]) vkDestroyPipeline(dev, b.out[k], NULL);
    for (uint32_t i = 0; i < job->nmodules; i++)
        if (b.mods[i]) vkDestroyShaderModule(dev, b.mods[i], NULL);
    for (uint32_t i = 0; i < job->nsets; i++) {
        vkDestroyPipelineLayout(dev, b.pl[i].layout, NULL);
        vkDestroyDescriptorSetLayout(dev, b.pl[i].set, NULL);
    }
    if (job->has_global) vkDestroyDescriptorSetLayout(dev, b.pl[0].global, NULL);
    summary_end();
    return val_errors ? 1 : 0;
}

// ---- cache mode: prepared pipelines through a VkPipelineCache or pipeline binaries ----
//
// The job's pairs come in groups that share their two modules (one pair per
// colour format); a worker takes a group, loads and creates its two modules,
// builds the group's pipelines and frees the modules, so that 20,000 pipelines
// never hold their SPIR-V in memory at once. Captured binaries are appended to
// the binaries file as records: pipeline index, size, then the serialised
// binaries (count, then per binary: key size, key, data size, data).

typedef struct {
    Job *job;
    PairLayout *pl;
    uint32_t *group;            // first pair of each group, plus npairs at the end
    uint32_t ngroups;
    VkPipelineCache cache;
    VkPipeline *pipes;
    const uint8_t **in;         // per pipeline: its serialised binaries (binaries load)
    FILE *bin_out;
    CRITICAL_SECTION bin_lock;
    double bin_bytes;
    long hits;
    long refused;               // binaries the driver would not create a pipeline from
    const char *phase;
    int capture;
    int keep;
} CacheCtx;

static void capture_binaries(CacheCtx *c, uint32_t i, uint32_t w, VkPipeline pipe) {
    VkPipelineBinaryCreateInfoKHR bci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR, .pipeline = pipe};
    VkPipelineBinaryHandlesInfoKHR hi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
    double t0 = now_ms();
    VK_CHECK(vkCreatePipelineBinariesKHR(dev, &bci, NULL, &hi));
    VkPipelineBinaryKHR bins[8];
    if (hi.pipelineBinaryCount > 8) die(3, "more than 8 binaries in one pipeline");
    hi.pPipelineBinaries = bins;
    VK_CHECK(vkCreatePipelineBinariesKHR(dev, &bci, NULL, &hi));
    size_t cap = 64, used = 12;
    uint8_t *blob = malloc(cap);
    uint32_t nb = hi.pipelineBinaryCount;
    memcpy(blob + 8, &nb, 4);
    for (uint32_t k = 0; k < nb; k++) {
        VkPipelineBinaryDataInfoKHR di = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR, .pipelineBinary = bins[k]};
        VkPipelineBinaryKeyKHR key = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
        size_t size = 0;
        VK_CHECK(vkGetPipelineBinaryDataKHR(dev, &di, &key, &size, NULL));
        size_t need = used + 8 + key.keySize + size;
        if (need > cap) {
            cap = need * 2;
            blob = realloc(blob, cap);
        }
        uint32_t ks = key.keySize, ds = (uint32_t)size;
        memcpy(blob + used, &ks, 4);
        memcpy(blob + used + 4, key.key, ks);
        memcpy(blob + used + 4 + ks, &ds, 4);
        VK_CHECK(vkGetPipelineBinaryDataKHR(dev, &di, &key, &size, blob + used + 8 + ks));
        used = need;
        vkDestroyPipelineBinaryKHR(dev, bins[k], NULL);
    }
    uint32_t head[2] = {i, (uint32_t)(used - 8)};
    memcpy(blob, head, 8);
    EnterCriticalSection(&c->bin_lock);
    if (fwrite(blob, 1, used, c->bin_out) != used) die(1, "short write of the binaries file");
    c->bin_bytes += (double)(used - 8);
    LeaveCriticalSection(&c->bin_lock);
    free(blob);
    out_row(c->phase, "capture_binaries", i, now_ms() - t0, (int)w, NULL);
}

static void w_cache_group(void *c_, uint32_t g, uint32_t w) {
    CacheCtx *c = c_;
    Job *job = c->job;
    uint32_t first = c->group[g], end = c->group[g + 1];
    Module *vm = &job->modules[job->pairs[first].vs], *fm = &job->modules[job->pairs[first].fs];
    double t0 = now_ms();
    module_load(job, vm);
    module_load(job, fm);
    double t1 = now_ms();
    VkShaderModule vs = module_create(vm), fs = module_create(fm);
    out_row(c->phase, "modules_read", g, t1 - t0, (int)w, NULL);
    out_row(c->phase, "modules_create", g, now_ms() - t1, (int)w, NULL);
    for (uint32_t i = first; i < end; i++) {
        Pair *p = &job->pairs[i];
        PipeInfo pi;
        pipe_info(&pi, job, ALL_PARTS, vs, fs, c->pl[p->set].layout, p->targets, job->formats[p->format], 0);
        VkPipelineCreationFeedback fb = {0}, sfb[2] = {{0}};
        VkPipelineCreationFeedbackCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
                                                    .pNext = pi.gp.pNext, .pPipelineCreationFeedback = &fb,
                                                    .pipelineStageCreationFeedbackCount = 2,
                                                    .pPipelineStageCreationFeedbacks = sfb};
        pi.gp.pNext = &fci;
        if (c->capture) {
            pi.flags2 = (VkPipelineCreateFlags2CreateInfoKHR){.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
                                                              .pNext = pi.gp.pNext,
                                                              .flags = VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR};
            pi.gp.pNext = &pi.flags2;
        }
        VkPipeline pipe;
        double ms = create_pipeline(c->cache, &pi, &pipe);
        if (fb.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT)
            __atomic_add_fetch(&c->hits, 1, __ATOMIC_RELAXED);
        out_row(c->phase, c->keep ? "load_from_cache" : "build", i, ms, (int)w, NULL);
        if (c->capture) capture_binaries(c, i, w, pipe);
        if (c->keep) c->pipes[i] = pipe;
        else vkDestroyPipeline(dev, pipe, NULL);
    }
    vkDestroyShaderModule(dev, vs, NULL);
    vkDestroyShaderModule(dev, fs, NULL);
    module_free(vm);
    module_free(fm);
}

static void w_binaries_load(void *c_, uint32_t i, uint32_t w) {
    CacheCtx *c = c_;
    Pair *p = &c->job->pairs[i];
    const uint8_t *q = c->in[i];
    if (!q) die(1, "pipeline %u has no binaries in the file", i);
    uint32_t nb;
    memcpy(&nb, q, 4);
    q += 4;
    VkPipelineBinaryKeyKHR keys[8];
    VkPipelineBinaryDataKHR data[8];
    if (nb == 0 || nb > 8) die(1, "pipeline %u: %u binaries", i, nb);
    for (uint32_t k = 0; k < nb; k++) {
        uint32_t ks, ds;
        memcpy(&ks, q, 4);
        keys[k] = (VkPipelineBinaryKeyKHR){.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR, .keySize = ks};
        memcpy(keys[k].key, q + 4, ks);
        memcpy(&ds, q + 4 + ks, 4);
        data[k] = (VkPipelineBinaryDataKHR){ds, (void *)(q + 8 + ks)};
        q += 8 + ks + ds;
    }
    double t0 = now_ms();
    VkPipelineBinaryKeysAndDataKHR kd = {nb, keys, data};
    VkPipelineBinaryCreateInfoKHR bci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR, .pKeysAndDataInfo = &kd};
    VkPipelineBinaryKHR bins[8];
    VkPipelineBinaryHandlesInfoKHR hi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
                                         .pipelineBinaryCount = nb, .pPipelineBinaries = bins};
    VK_CHECK(vkCreatePipelineBinariesKHR(dev, &bci, NULL, &hi));
    PipeInfo pi;
    // With binaries the stages' modules are ignored: no SPIR-V is read or created.
    pipe_info(&pi, c->job, ALL_PARTS, VK_NULL_HANDLE, VK_NULL_HANDLE, c->pl[p->set].layout, p->targets,
              c->job->formats[p->format], 0);
    pi.stages[0] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                     .stage = VK_SHADER_STAGE_VERTEX_BIT, .pName = "main"};
    pi.stages[1] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                     .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .pName = "main"};
    pi.gp.stageCount = 2;
    pi.gp.pStages = pi.stages;
    VkPipelineBinaryInfoKHR bi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR, .pNext = pi.gp.pNext,
                                  .binaryCount = nb, .pPipelineBinaries = bins};
    pi.gp.pNext = &bi;
    // The driver may refuse its own binaries with VK_PIPELINE_COMPILE_REQUIRED; the
    // pipeline then has to be compiled from SPIR-V, so it is counted, not created.
    VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi.gp, NULL, &c->pipes[i]);
    if (r == VK_PIPELINE_COMPILE_REQUIRED) {
        c->pipes[i] = VK_NULL_HANDLE;
        __atomic_add_fetch(&c->refused, 1, __ATOMIC_RELAXED);
    } else if (r != VK_SUCCESS) die(3, "pipeline %u from binaries: %d", i, (int)r);
    for (uint32_t k = 0; k < nb; k++) vkDestroyPipelineBinaryKHR(dev, bins[k], NULL);
    out_row(c->phase, r == VK_SUCCESS ? "load_from_binaries" : "binaries_refused", i, now_ms() - t0, (int)w, NULL);
}

static void *read_all(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) die(1, "cannot open %s", path);
    _fseeki64(f, 0, SEEK_END);
    long long n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    void *p = malloc(n > 0 ? (size_t)n : 1);
    if (!p) die(3, "out of memory reading %s", path);
    if (n > 0 && fread(p, 1, (size_t)n, f) != (size_t)n) die(1, "cannot read %s", path);
    fclose(f);
    *size = (size_t)n;
    return p;
}

static void cache_groups(CacheCtx *c) {
    Job *job = c->job;
    c->group = calloc(job->npairs + 1, sizeof *c->group);
    uint32_t *owner = malloc(job->nmodules * sizeof *owner);
    memset(owner, 0xFF, job->nmodules * sizeof *owner);
    for (uint32_t i = 0; i < job->npairs; i++) {
        Pair *p = &job->pairs[i];
        if (i == 0 || p->vs != job->pairs[i - 1].vs || p->fs != job->pairs[i - 1].fs) c->group[c->ngroups++] = i;
        uint32_t g = c->ngroups - 1;
        // A module in two groups would be loaded and freed by two workers at once.
        if ((owner[p->vs] != UINT32_MAX && owner[p->vs] != g) || (owner[p->fs] != UINT32_MAX && owner[p->fs] != g))
            die(1, "pair %u: its modules belong to another group; list a module's pairs together", i);
        owner[p->vs] = owner[p->fs] = g;
    }
    c->group[c->ngroups] = job->npairs;
    free(owner);
}

int mode_cache(Job *job, const char *phase) {
    summary_begin("cache");
    summary_str("phase", phase);
    const char *cache_file = job_param(job, "cache_file", NULL);
    const char *bin_file = job_param(job, "binaries_file", NULL);
    if (!cache_file || !bin_file) die(1, "cache mode needs param cache_file and param binaries_file");
    CacheCtx c = {.job = job, .phase = phase};
    InitializeCriticalSection(&c.bin_lock);
    c.pl = layouts_create(job, NULL);
    cache_groups(&c);
    uint32_t np = job->npairs;
    c.pipes = calloc(np, sizeof *c.pipes);
    summary_num("available_mb_before", available_mb());
    if (!strcmp(phase, "build") || !strcmp(phase, "build-binaries")) {
        // Two cold builds: Vulkan forbids capturing binaries into a pipeline built with a cache.
        c.capture = !strcmp(phase, "build-binaries");
        if (c.capture && !have_binaries) die(2, "no VK_KHR_pipeline_binary");
        if (!c.capture) {
            VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
            VK_CHECK(vkCreatePipelineCache(dev, &ci, NULL, &c.cache));
        }
        if (c.capture) {
            c.bin_out = fopen(bin_file, "wb");
            if (!c.bin_out) die(1, "cannot write %s", bin_file);
        }
        double wall = run_parallel(c.ngroups, opt.threads, w_cache_group, &c);
        summary_num("build_wall_ms", wall);
        if (c.bin_out) {
            fclose(c.bin_out);
            summary_num("binaries_bytes", c.bin_bytes);
        }
        double ws;
        memory_mb(&ws, NULL, NULL);
        summary_num("working_set_built_mb", ws);
        if (c.capture) goto done;
        size_t size = 0;
        double t0 = now_ms();
        VK_CHECK(vkGetPipelineCacheData(dev, c.cache, &size, NULL));
        void *data = malloc(size);
        if (!data) die(3, "out of memory for the cache data");
        VK_CHECK(vkGetPipelineCacheData(dev, c.cache, &size, data));
        FILE *f = fopen(cache_file, "wb");
        if (!f || fwrite(data, 1, size, f) != size) die(1, "cannot write %s", cache_file);
        fclose(f);
        free(data);
        summary_num("cache_save_ms", now_ms() - t0);
        summary_num("cache_bytes", (double)size);
        vkDestroyPipelineCache(dev, c.cache, NULL);
    done:;
    } else if (!strcmp(phase, "load")) {
        size_t size;
        double t0 = now_ms();
        void *data = read_all(cache_file, &size);
        double t1 = now_ms();
        VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = size,
                                        .pInitialData = data};
        VK_CHECK(vkCreatePipelineCache(dev, &ci, NULL, &c.cache));
        free(data);
        summary_num("cache_read_ms", t1 - t0);
        summary_num("cache_create_ms", now_ms() - t1);
        c.keep = 1;
        double wall = run_parallel(c.ngroups, opt.threads, w_cache_group, &c);
        summary_num("load_wall_ms", wall);
        summary_num("cache_hits", (double)c.hits);
        vkDestroyPipelineCache(dev, c.cache, NULL);
    } else if (!strcmp(phase, "binaries")) {
        if (!have_binaries) die(2, "no VK_KHR_pipeline_binary");
        size_t size;
        double t0 = now_ms();
        uint8_t *data = read_all(bin_file, &size);
        summary_num("binaries_read_ms", now_ms() - t0);
        c.in = calloc(np, sizeof *c.in);
        for (size_t at = 0; at + 8 <= size;) {
            uint32_t head[2];
            memcpy(head, data + at, 8);
            if (head[0] >= np || at + 8 + head[1] > size) die(1, "binaries file is damaged at byte %zu", at);
            c.in[head[0]] = data + at + 8;
            at += 8 + head[1];
        }
        c.keep = 1;
        double wall = run_parallel(np, opt.threads, w_binaries_load, &c);
        summary_num("load_wall_ms", wall);
        summary_num("binaries_refused", (double)c.refused);
        free(data);
    } else die(1, "cache phases are build, build-binaries, load and binaries");

    double ws, priv, peak;
    memory_mb(&ws, &priv, &peak);
    summary_num("pipelines", np);
    summary_num("groups", c.ngroups);
    summary_num("available_mb_loaded", available_mb());
    summary_num("working_set_loaded_mb", ws);
    summary_num("private_loaded_mb", priv);
    for (uint32_t i = 0; i < np; i++)
        if (c.pipes[i]) vkDestroyPipeline(dev, c.pipes[i], NULL);
    for (uint32_t i = 0; i < job->nsets; i++) {
        vkDestroyPipelineLayout(dev, c.pl[i].layout, NULL);
        vkDestroyDescriptorSetLayout(dev, c.pl[i].set, NULL);
    }
    if (job->has_global) vkDestroyDescriptorSetLayout(dev, c.pl[0].global, NULL);
    summary_end();
    return val_errors ? 1 : 0;
}
