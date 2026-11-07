// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The NVN2 front end's commands (architecture D11): every nvnCommandBuffer entry point the game names that records a
// command writes one record (nvn_obj.h, command records) into its command buffer's recording (nvn_cmd.c). What the game
// may change after the call returns is taken when it records: state objects and textures by value, a program by serial
// and stage records, uniform data, push constants, viewports and handle arrays by value. A *Deferred bind or push keeps
// the game's array and reads it at submit, as the console's driver does. Values past what a record holds, or past what
// NVN defines, stop by name; nothing is clamped or dropped where the console's driver does not clamp or drop it.
//
// Signatures of the NVN2 calls nvn.h has no prototype for are read from the console's driver (the game's sdk, the
// registers each entry point reads) and from nxnative dm/gfx/nvn/hle_nvn.c at 865cc35 (MIT, licenses/nxnative-MIT.txt);
// where the two disagree, the driver wins (BindShadingRateImage takes the texture first, then the view).
//
// Adapted from oot-pc-recomp runtime/nvn_api_cmd.c and runtime/nvn_api_nvn2.c at 8e09aaa (GPL-3.0-or-later), DKB
// Recomp's at 9d8fa02 with additions, which port MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c at
// b75be8cf (MIT, Copyright (c) 2026 jhaizhou; licenses/dkb-xlate-MIT.txt): the record per entry point, push constants
// as the driver clamps them, deferred binds, ReportValue as a 4-byte buffer clear. Changed here: textures by value
// instead of pinned copies; a program by serial and stages instead of its pointer; 32 vertex bindings; every
// viewport, scissor and depth range of a call kept; zero-instance draws, mesh draws and the NVN2 state calls recorded.
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "nvn_cmd.h"
#include "oot2/nvn_program.h"
#include "oot2/nvn_tex.h"

#define NVN_MAX_VIEWPORTS 16
#define PUSH_TABLE_BYTES 0x100     // a root table of push constants

// A texture as a record keeps it: the object's bytes and the view, as they are when the call is made.
static void tex_rec(Cpu *c, nvn_tex_rec *r, const NVNtexture *t, const NVNtextureView *v, const char *fn) {
    if (!t) return;
    const tex_t *x = (const tex_t *)t;
    if (x->magic != NVN_TEX_MAGIC || !x->serial)
        rt_unimplemented(c, "nvn", "%s: the texture %p is not initialised", fn, (const void *)t);
    r->tex = *x, r->present = 1;
    if (v) r->view = *(const view_t *)v, r->has_view = 1;
}

static void range_check(Cpu *c, int first, int n, int max, const char *fn) {
    if (first < 0 || n < 0 || first > max || n > max - first)
        rt_unimplemented(c, "nvn", "%s: first %d, %d entries, past the %d the console has", fn, first, n, max);
}

// ---------- commands with CPU-visible results
// void nvnCommandBufferSignalEvent(NVNcommandBuffer *cmd, const NVNevent *event, NVNeventSignalMode mode,
// NVNeventSignalLocation location, int flags, uint32_t value)
static volatile uint32_t *event_word(Cpu *c, const ev_t *e, const char *fn) {
    if (!e || !e->val) rt_unimplemented(c, "nvn", "%s: the event %p has no storage", fn, (const void *)e);
    return e->val;
}
NVN_API(nvnCommandBufferSignalEvent) {
    volatile uint32_t *val = event_word(c, arg_p(c, 1), "CommandBufferSignalEvent");
    *REC(c, OP_SIGNAL_EVENT, p_signal_event) = (p_signal_event){val, (uint32_t)arg_x(c, 2), (uint32_t)arg_x(c, 5),
                                                                (uint32_t)arg_x(c, 3), (uint32_t)arg_x(c, 4)};
}
// void nvnCommandBufferWaitEvent(NVNcommandBuffer *cmd, const NVNevent *event, NVNeventWaitMode mode, uint32_t value):
// with the game's return address, so a back end can say which call site a wait came from
NVN_API(nvnCommandBufferWaitEvent) {
    volatile uint32_t *val = event_word(c, arg_p(c, 1), "CommandBufferWaitEvent");
    uint32_t mode = (uint32_t)arg_x(c, 2);
    if (mode != NVN_EVENT_WAIT_MODE_EQUAL && mode != NVN_EVENT_WAIT_MODE_GEQUAL_WRAP)
        rt_unimplemented(c, "nvn", "CommandBufferWaitEvent: mode %u is not one of NVN's", mode);
    *REC(c, OP_WAIT_EVENT, p_wait_event) = (p_wait_event){val, mode, (uint32_t)arg_x(c, 3), c->x[30]};
}
// void nvnCommandBufferResetCounter(NVNcommandBuffer *cmd, NVNcounterType type)
NVN_API(nvnCommandBufferResetCounter) { REC(c, OP_RESET_COUNTER, p_reset_counter)->type = (uint32_t)arg_x(c, 1); }
// void nvnCommandBufferReportCounter(NVNcommandBuffer *cmd, NVNcounterType type, NVNbufferAddress address)
NVN_API(nvnCommandBufferReportCounter) {
    *REC(c, OP_REPORT_COUNTER, p_report_counter) = (p_report_counter){(uint32_t)arg_x(c, 1), arg_x(c, 2)};
}
// void nvnCommandBufferReportValue(NVNcommandBuffer *cmd, uint32_t value, NVNbufferAddress address): a 4-byte buffer
// clear, so it stays in GPU order with the clears of the same word
NVN_API(nvnCommandBufferReportValue) {
    *REC(c, OP_CLEAR_BUF, p_clear_buf) = (p_clear_buf){arg_x(c, 2), 4, (uint32_t)arg_x(c, 1), 0};
}
// void nvnCommandBufferClearBuffer(NVNcommandBuffer *cmd, NVNbufferAddress address, size_t size, uint32_t value)
NVN_API(nvnCommandBufferClearBuffer) {
    *REC(c, OP_CLEAR_BUF, p_clear_buf) = (p_clear_buf){arg_x(c, 1), arg_x(c, 2), (uint32_t)arg_x(c, 3), 0};
}
// void nvnCommandBufferUpdateUniformBuffer(NVNcommandBuffer *cmd, NVNbufferAddress buffer, size_t buffer_size,
// ptrdiff_t offset, size_t size, const void *data): the data is copied now, the game may change it on return
NVN_API(nvnCommandBufferUpdateUniformBuffer) {
    uint64_t dst = arg_x(c, 1) + arg_x(c, 3), n = arg_x(c, 4);
    if (n > (16u << 20)) rt_unimplemented(c, "nvn", "CommandBufferUpdateUniformBuffer of %#llx bytes", (unsigned long long)n);
    p_update_ub *r = nvn_rec(c, OP_UPDATE_UB, offsetof(p_update_ub, data) + n);
    r->dst = dst, r->size = n;
    if (n) memcpy(r->data, arg_p(c, 5), n);
}

// ---------- render targets and clears
// void nvnCommandBufferSetRenderTargets(NVNcommandBuffer *cmd, int n, const NVNtexture *const *colors,
// const NVNtextureView *const *color_views, const NVNtexture *depth, const NVNtextureView *depth_view)
NVN_API(nvnCommandBufferSetRenderTargets) {
    const char *fn = "CommandBufferSetRenderTargets";
    int n = (int)arg_x(c, 1);
    range_check(c, 0, n, 8, fn);
    p_set_rt *r = RECN(c, OP_SET_RT, p_set_rt, color, n);
    const NVNtexture *const *ct = arg_p(c, 2);
    const NVNtextureView *const *cv = arg_p(c, 3);
    r->n = n;
    for (int i = 0; i < n; i++) tex_rec(c, &r->color[i], ct ? ct[i] : NULL, cv ? cv[i] : NULL, fn);
    tex_rec(c, &r->depth, arg_p(c, 4), arg_p(c, 5), fn);
}
// void nvnCommandBufferClearColor(NVNcommandBuffer *cmd, int index, const float *rgba, int mask)
NVN_API(nvnCommandBufferClearColor) {
    p_clear_color *r = REC(c, OP_CLEAR_COLOR, p_clear_color);
    r->index = (int)arg_x(c, 1), r->mask = (int)arg_x(c, 3);
    memcpy(r->rgba, arg_p(c, 2), sizeof r->rgba);
}
// void nvnCommandBufferClearDepthStencil(NVNcommandBuffer *cmd, float depth, NVNboolean depth_mask, int stencil,
// int stencil_mask)
NVN_API(nvnCommandBufferClearDepthStencil) {
    *REC(c, OP_CLEAR_DS, p_clear_ds) = (p_clear_ds){arg_s(c, 0), nvn_arg_bool(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3)};
}
// void nvnCommandBufferClearTexture(NVNcommandBuffer *cmd, const NVNtexture *texture, const NVNtextureView *view,
// const NVNcopyRegion *region, const float *rgba, int mask)
NVN_API(nvnCommandBufferClearTexture) {
    p_clear_tex *r = REC(c, OP_CLEAR_TEX, p_clear_tex);
    tex_rec(c, &r->tex, arg_p(c, 1), arg_p(c, 2), "CommandBufferClearTexture");
    r->r = *(const NVNcopyRegion *)arg_p(c, 3);
    memcpy(r->c.f, arg_p(c, 4), sizeof r->c.f);
    r->mask = (int)arg_x(c, 5);
}

// ---------- viewports, scissors, depth
static void viewports(Cpu *c, int first, int n, const float *xywh, const char *fn) {
    range_check(c, first, n, NVN_MAX_VIEWPORTS, fn);
    p_viewports *r = RECN(c, OP_VIEWPORTS, p_viewports, v, n);
    r->first = first, r->n = n;
    memcpy(r->v, xywh, sizeof r->v[0] * (size_t)n);
}
// void nvnCommandBufferSetViewport(NVNcommandBuffer *cmd, int x, int y, int w, int h): viewport 0
NVN_API(nvnCommandBufferSetViewport) {
    float v[4] = {(float)(int)arg_x(c, 1), (float)(int)arg_x(c, 2), (float)(int)arg_x(c, 3), (float)(int)arg_x(c, 4)};
    viewports(c, 0, 1, v, "CommandBufferSetViewport");
}
// void nvnCommandBufferSetViewports(NVNcommandBuffer *cmd, int first, int n, const float *xywh)
NVN_API(nvnCommandBufferSetViewports) {
    viewports(c, (int)arg_x(c, 1), (int)arg_x(c, 2), arg_p(c, 3), "CommandBufferSetViewports");
}
static void scissors(Cpu *c, int first, int n, const int32_t *xywh, const char *fn) {
    range_check(c, first, n, NVN_MAX_VIEWPORTS, fn);
    p_scissor *r = RECN(c, OP_SCISSOR, p_scissor, r, n);
    r->first = first, r->n = n;
    memcpy(r->r, xywh, sizeof r->r[0] * (size_t)n);
}
// void nvnCommandBufferSetScissor(NVNcommandBuffer *cmd, int x, int y, int w, int h): scissor 0
NVN_API(nvnCommandBufferSetScissor) {
    int32_t r[4] = {(int32_t)arg_x(c, 1), (int32_t)arg_x(c, 2), (int32_t)arg_x(c, 3), (int32_t)arg_x(c, 4)};
    scissors(c, 0, 1, r, "CommandBufferSetScissor");
}
// void nvnCommandBufferSetScissors(NVNcommandBuffer *cmd, int first, int n, const int *xywh)
NVN_API(nvnCommandBufferSetScissors) { scissors(c, (int)arg_x(c, 1), (int)arg_x(c, 2), arg_p(c, 3), "CommandBufferSetScissors"); }
static void depth_ranges(Cpu *c, int first, int n, const float *near_far, const char *fn) {
    range_check(c, first, n, NVN_MAX_VIEWPORTS, fn);
    p_depth_range *r = RECN(c, OP_DEPTH_RANGE, p_depth_range, r, n);
    r->first = first, r->n = n;
    memcpy(r->r, near_far, sizeof r->r[0] * (size_t)n);
}
// void nvnCommandBufferSetDepthRange(NVNcommandBuffer *cmd, float near, float far): viewport 0
NVN_API(nvnCommandBufferSetDepthRange) {
    float r[2] = {arg_s(c, 0), arg_s(c, 1)};
    depth_ranges(c, 0, 1, r, "CommandBufferSetDepthRange");
}
// void nvnCommandBufferSetDepthRanges(NVNcommandBuffer *cmd, int first, int n, const float *ranges)
NVN_API(nvnCommandBufferSetDepthRanges) {
    depth_ranges(c, (int)arg_x(c, 1), (int)arg_x(c, 2), arg_p(c, 3), "CommandBufferSetDepthRanges");
}
// void nvnCommandBufferSetDepthBounds(NVNcommandBuffer *cmd, NVNboolean enable, float min, float max)
NVN_API(nvnCommandBufferSetDepthBounds) {
    *REC(c, OP_DEPTH_BOUNDS, p_depth_bounds) = (p_depth_bounds){nvn_arg_bool(c, 1) != 0, arg_s(c, 0), arg_s(c, 1)};
}

// ---------- one value of render state
static void flag(Cpu *c, uint32_t op, int32_t value) { REC(c, op, p_flag)->value = value; }
// void nvnCommandBufferSet{DepthClamp,RenderEnable,RasterizerDiscard}(NVNcommandBuffer *cmd, NVNboolean enable)
NVN_API(nvnCommandBufferSetDepthClamp) { flag(c, OP_DEPTH_CLAMP, nvn_arg_bool(c, 1) != 0); }
NVN_API(nvnCommandBufferSetRenderEnable) { flag(c, OP_RENDER_ENABLE, nvn_arg_bool(c, 1) != 0); }
NVN_API(nvnCommandBufferSetRasterizerDiscard) { flag(c, OP_RASTER_DISCARD, nvn_arg_bool(c, 1) != 0); }
// void nvnCommandBufferSetGeometryProcessingMode(NVNcommandBuffer *cmd, int mode) (NVN2)
NVN_API(nvnCommandBufferSetGeometryProcessingMode) { flag(c, OP_GEOMETRY_MODE, (int32_t)arg_x(c, 1)); }
// void nvnCommandBufferSetSampleMask(NVNcommandBuffer *cmd, int mask): the driver keeps its low 16 bits
NVN_API(nvnCommandBufferSetSampleMask) { flag(c, OP_SAMPLE_MASK, (int32_t)(arg_x(c, 1) & 0xffff)); }
// void nvnCommandBufferSetShadingRatePaletteMode(NVNcommandBuffer *cmd, int mode) (NVN2)
NVN_API(nvnCommandBufferSetShadingRatePaletteMode) { flag(c, OP_SHADING_RATE_MODE, (int32_t)arg_x(c, 1)); }
// void nvnCommandBufferSetPrimitiveRestart(NVNcommandBuffer *cmd, NVNboolean enable, int index)
NVN_API(nvnCommandBufferSetPrimitiveRestart) {
    *REC(c, OP_PRIM_RESTART, p_prim_restart) = (p_prim_restart){nvn_arg_bool(c, 1) != 0, (uint32_t)arg_x(c, 2)};
}
// void nvnCommandBufferSetRenderEnableConditional(NVNcommandBuffer *cmd, NVNconditionalRenderMode mode,
// NVNbufferAddress counters): render when the two counters at counters and counters + 16 are equal (0) or not (1)
NVN_API(nvnCommandBufferSetRenderEnableConditional) {
    int mode = (int)arg_x(c, 1);
    if (mode != 0 && mode != 1)
        rt_unimplemented(c, "nvn", "CommandBufferSetRenderEnableConditional: mode %d is not one of NVN's", mode);
    *REC(c, OP_RENDER_COND, p_render_cond) = (p_render_cond){mode, 0, arg_x(c, 2)};
}
// void nvnCommandBufferSetConservativeRasterEnable(NVNcommandBuffer *cmd, NVNboolean enable); NVN2's
// SetConservativeRasterMode(NVNcommandBuffer *cmd, int mode, int field), the driver keeping 4 bits of field
NVN_API(nvnCommandBufferSetConservativeRasterEnable) {
    *REC(c, OP_CONSERVATIVE, p_conservative) = (p_conservative){0, nvn_arg_bool(c, 1) != 0, 0};
}
NVN_API(nvnCommandBufferSetConservativeRasterMode) {
    *REC(c, OP_CONSERVATIVE, p_conservative) = (p_conservative){1, (int32_t)arg_x(c, 1), (int32_t)(arg_x(c, 2) & 15)};
}

// ---------- programs and state objects
// void nvnCommandBufferBindProgram(NVNcommandBuffer *cmd, const NVNprogram *program, int stages): the program as it
// is now, since the game may set its stages again or finalize it before the submit
NVN_API(nvnCommandBufferBindProgram) {
    const prog_t *p = arg_p(c, 1);
    if (p && (p->magic != NVN_PROG_MAGIC || p->nstage > 6))
        rt_unimplemented(c, "nvn", "CommandBufferBindProgram: the program %p is not initialised", (const void *)p);
    p_bind_program *r = REC(c, OP_BIND_PROGRAM, p_bind_program);
    r->stages = (int32_t)arg_x(c, 2);
    if (!p) return;
    r->serial = p->serial, r->nstage = (int32_t)p->nstage;
    for (uint32_t i = 0; i < p->nstage; i++) r->code[i] = p->st[i].code, r->id[i] = p->st[i].id;
}
// void nvnCommandBufferBind{Blend,Color,ChannelMask,Multisample,Polygon,DepthStencil}State(NVNcommandBuffer *cmd,
// const NVN*State *state): the state's packed bytes (p_bind_state.kind order)
static void bind_state(Cpu *c, int kind, const void *state, size_t n) {
    p_bind_state *r = REC(c, OP_BIND_STATE, p_bind_state);
    r->kind = kind;
    memcpy(r->bytes, state, n);
}
_Static_assert(sizeof(blend_st) <= 0x18 && sizeof(ms_st) <= 0x18 && sizeof(ds_st) <= 0x18, "p_bind_state.bytes");
NVN_API(nvnCommandBufferBindBlendState) { bind_state(c, 0, arg_p(c, 1), sizeof(blend_st)); }
NVN_API(nvnCommandBufferBindColorState) { bind_state(c, 1, arg_p(c, 1), sizeof(color_st)); }
NVN_API(nvnCommandBufferBindChannelMaskState) { bind_state(c, 2, arg_p(c, 1), sizeof(chmask_st)); }
NVN_API(nvnCommandBufferBindMultisampleState) { bind_state(c, 3, arg_p(c, 1), sizeof(ms_st)); }
NVN_API(nvnCommandBufferBindPolygonState) { bind_state(c, 4, arg_p(c, 1), sizeof(poly_st)); }
NVN_API(nvnCommandBufferBindDepthStencilState) { bind_state(c, 5, arg_p(c, 1), sizeof(ds_st)); }
// void nvnCommandBufferSetPointSize / SetLineWidth(NVNcommandBuffer *cmd, float size): bind-state kinds 6 and 8
NVN_API(nvnCommandBufferSetPointSize) {
    float s = arg_s(c, 0);
    bind_state(c, 6, &s, sizeof s);
}
NVN_API(nvnCommandBufferSetLineWidth) {
    float s = arg_s(c, 0);
    bind_state(c, 8, &s, sizeof s);
}
// void nvnCommandBufferBindVertexAttribState(NVNcommandBuffer *cmd, int n, const NVNvertexAttribState *states)
NVN_API(nvnCommandBufferBindVertexAttribState) {
    int n = (int)arg_x(c, 1);
    range_check(c, 0, n, NVN_VERTEX_BINDINGS, "CommandBufferBindVertexAttribState");
    p_bind_vattr *r = RECN(c, OP_BIND_VATTR, p_bind_vattr, st, n);
    r->n = n;
    memcpy(r->st, arg_p(c, 2), sizeof r->st[0] * (size_t)n);
}
// void nvnCommandBufferBindVertexStreamState(NVNcommandBuffer *cmd, int n, const NVNvertexStreamState *states)
NVN_API(nvnCommandBufferBindVertexStreamState) {
    int n = (int)arg_x(c, 1);
    range_check(c, 0, n, NVN_VERTEX_BINDINGS, "CommandBufferBindVertexStreamState");
    p_bind_vstream *r = RECN(c, OP_BIND_VSTREAM, p_bind_vstream, st, n);
    r->n = n;
    memcpy(r->st, arg_p(c, 2), sizeof r->st[0] * (size_t)n);
}
// void nvnCommandBufferBindVertexBuffer(NVNcommandBuffer *cmd, int index, NVNbufferAddress address, size_t size)
NVN_API(nvnCommandBufferBindVertexBuffer) {
    int index = (int)arg_x(c, 1);
    range_check(c, index, 1, NVN_VERTEX_BINDINGS, "CommandBufferBindVertexBuffer");
    p_bind_vbuf *r = RECN(c, OP_BIND_VBUF, p_bind_vbuf, r, 1);
    r->first = index, r->n = 1;
    r->r[0] = (NVNbufferRange){arg_x(c, 2), arg_x(c, 3)};
}

// ---------- buffers and handles
// void nvnCommandBufferBind{Uniform,Storage}Buffer(NVNcommandBuffer *cmd, NVNshaderStage stage, int index,
// NVNbufferAddress address, size_t size); BindStorageBuffers(cmd, stage, first, n, const NVNbufferRange *ranges): one
// record per binding
static void bind_buffer(Cpu *c, uint32_t op) {
    *REC(c, op, p_bind_buf) = (p_bind_buf){(int)arg_x(c, 1), (int)arg_x(c, 2), arg_x(c, 3), arg_x(c, 4)};
}
NVN_API(nvnCommandBufferBindUniformBuffer) { bind_buffer(c, OP_BIND_UBO); }
NVN_API(nvnCommandBufferBindStorageBuffer) { bind_buffer(c, OP_BIND_SSBO); }
NVN_API(nvnCommandBufferBindStorageBuffers) {
    int stage = (int)arg_x(c, 1), first = (int)arg_x(c, 2), n = (int)arg_x(c, 3);
    if (n < 0) rt_unimplemented(c, "nvn", "CommandBufferBindStorageBuffers: %d buffers", n);
    for (int i = 0; i < n; i++) {
        const NVNbufferRange *rg = (const NVNbufferRange *)arg_p(c, 4) + i;
        *REC(c, OP_BIND_SSBO, p_bind_buf) = (p_bind_buf){stage, first + i, rg->address, rg->size};
    }
}
// Texture, image and sampler handles (nvn_tex.h unpacks them). A deferred bind keeps the game's array: the console's
// driver reads it at submit, and the game fills it after recording.
static void bind_handles(Cpu *c, uint32_t op, int stage, int first, int n, const uint64_t *h, int deferred,
                         const char *fn) {
    if (n < 0 || n > 32 || (n && !h)) rt_unimplemented(c, "nvn", "%s: %d handles at %p", fn, n, (const void *)h);
    p_bind_handles *r = RECN(c, op, p_bind_handles, h, n);
    r->stage = stage, r->first = first, r->n = n;
    if (n) memcpy(r->h, h, sizeof r->h[0] * (size_t)n);
    if (deferred) r->src = (uintptr_t)h, nvn_rec_deferred(c);
}
// void nvnCommandBufferBind{Texture,Image}(NVNcommandBuffer *cmd, NVNshaderStage stage, int index, handle)
NVN_API(nvnCommandBufferBindTexture) {
    uint64_t h = arg_x(c, 3);
    bind_handles(c, OP_BIND_TEX, (int)arg_x(c, 1), (int)arg_x(c, 2), 1, &h, 0, "CommandBufferBindTexture");
}
NVN_API(nvnCommandBufferBindImage) {
    uint64_t h = arg_x(c, 3);
    bind_handles(c, OP_BIND_IMG, (int)arg_x(c, 1), (int)arg_x(c, 2), 1, &h, 0, "CommandBufferBindImage");
}
// void nvnCommandBufferBind{Textures,Images,TexturesDeferred}(NVNcommandBuffer *cmd, NVNshaderStage stage, int first,
// int n, const handle *handles)
NVN_API(nvnCommandBufferBindTextures) {
    bind_handles(c, OP_BIND_TEX, (int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), arg_p(c, 4), 0,
                 "CommandBufferBindTextures");
}
NVN_API(nvnCommandBufferBindImages) {
    bind_handles(c, OP_BIND_IMG, (int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), arg_p(c, 4), 0,
                 "CommandBufferBindImages");
}
NVN_API(nvnCommandBufferBindTexturesDeferred) {
    bind_handles(c, OP_BIND_TEX, (int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), arg_p(c, 4), 1,
                 "CommandBufferBindTexturesDeferred");
}
// void nvnCommandBufferBindAccelerationStructuresDeferred(NVNcommandBuffer *cmd, NVNshaderStage stage, int first,
// int n, const handle *handles) (NVN2; the driver reads those five arguments): kept with the game's array, which no
// back end reads (the replay has no ray tracing)
NVN_API(nvnCommandBufferBindAccelerationStructuresDeferred) {
    *REC(c, OP_BIND_ACCEL, p_bind_accel) = (p_bind_accel){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), 0,
                                                          arg_x(c, 4)};
}
// void nvnCommandBufferSetTexturePool / SetSamplerPool(NVNcommandBuffer *cmd, const NVNtexturePool *pool): where the
// pool is now, as nvn_dpool_snapshot takes it
static void set_pool(Cpu *c, uint32_t op, uint32_t limit, const char *fn) {
    dpool_ref pool;
    if (!nvn_dpool_snapshot(arg_p(c, 1), limit, &pool))
        rt_unimplemented(c, "nvn", "%s: %p is not an initialised descriptor pool", fn, arg_p(c, 1));
    REC(c, op, p_set_pool)->pool = pool;
}
NVN_API(nvnCommandBufferSetTexturePool) { set_pool(c, OP_SET_TEX_POOL, 1u << 20, "CommandBufferSetTexturePool"); }
NVN_API(nvnCommandBufferSetSamplerPool) { set_pool(c, OP_SET_SAMP_POOL, 1u << 12, "CommandBufferSetSamplerPool"); }
// void nvnCommandBufferSetShaderScratchMemory(NVNcommandBuffer *cmd, const NVNmemoryPool *pool, ptrdiff_t offset,
// size_t size): the range's GPU address (translated shaders keep their local memory on the host)
NVN_API(nvnCommandBufferSetShaderScratchMemory) {
    const pool_t *p = arg_p(c, 1);
    if (p && p->magic != NVN_POOL_MAGIC)
        rt_unimplemented(c, "nvn", "CommandBufferSetShaderScratchMemory: the memory pool %p is not initialised",
                         (const void *)p);
    *REC(c, OP_SCRATCH, p_scratch) = (p_scratch){p ? (uintptr_t)p->mem + arg_x(c, 2) : 0, arg_x(c, 3)};
}
// void nvnCommandBufferPush{Graphics,Compute}Constants(NVNcommandBuffer *cmd, int table, int offset, int bytes,
// const void *data) (NVN2). The driver keeps 3 bits of table, takes at most 0x100 bytes and drops a push of fewer than 4;
// an offset past the table or not a whole word stops by name.
static void push_constants(Cpu *c, int compute, const char *fn) {
    int table = (int)(arg_x(c, 1) & 7), offset = (int)arg_x(c, 2);
    uint32_t bytes = (uint32_t)arg_x(c, 3);
    if (bytes > PUSH_TABLE_BYTES) bytes = PUSH_TABLE_BYTES;
    if (bytes < 4) return;
    if (offset < 0 || (offset & 3) || (uint32_t)offset + bytes > PUSH_TABLE_BYTES)
        rt_unimplemented(c, "nvn", "%s: %u bytes at offset %#x of a %u-byte table", fn, bytes, offset, PUSH_TABLE_BYTES);
    uint32_t words = bytes / 4;
    p_push_const *r = RECN(c, OP_PUSH_CONST, p_push_const, w, words);
    r->compute = compute, r->table = table, r->offset = offset, r->bytes = (int32_t)(4 * words);
    memcpy(r->w, arg_p(c, 4), 4 * (size_t)words);
}
NVN_API(nvnCommandBufferPushGraphicsConstants) { push_constants(c, 0, "CommandBufferPushGraphicsConstants"); }
NVN_API(nvnCommandBufferPushComputeConstants) { push_constants(c, 1, "CommandBufferPushComputeConstants"); }

// ---------- stencil, polygon offset, blend colour
// void nvnCommandBufferSetStencil{ValueMask,Mask,Ref}(NVNcommandBuffer *cmd, NVNface face, int value)
NVN_API(nvnCommandBufferSetStencilValueMask) {
    *REC(c, OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), (int)arg_x(c, 2), 0, 0, 0};
}
NVN_API(nvnCommandBufferSetStencilMask) {
    *REC(c, OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), 0, (int)arg_x(c, 2), 0, 1};
}
NVN_API(nvnCommandBufferSetStencilRef) {
    *REC(c, OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), 0, 0, (int)arg_x(c, 2), 2};
}
// void nvnCommandBufferSetPolygonOffsetClamp(NVNcommandBuffer *cmd, float factor, float units, float clamp)
NVN_API(nvnCommandBufferSetPolygonOffsetClamp) {
    *REC(c, OP_POLY_OFFSET, p_poly_offset) = (p_poly_offset){arg_s(c, 0), arg_s(c, 1), arg_s(c, 2)};
}
// void nvnCommandBufferSetBlendColor(NVNcommandBuffer *cmd, const float *rgba)
NVN_API(nvnCommandBufferSetBlendColor) { memcpy(REC(c, OP_BLEND_COLOR, p_blend_color)->rgba, arg_p(c, 1), 16); }

// ---------- draws and dispatches
// void nvnCommandBufferDrawArrays(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, int first, int count)
NVN_API(nvnCommandBufferDrawArrays) {
    *REC(c, OP_DRAW, p_draw) = (p_draw){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), 0, 1};
}
// void nvnCommandBufferDrawArraysInstanced(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, int first, int count,
// int base_instance, int instances)
NVN_API(nvnCommandBufferDrawArraysInstanced) {
    *REC(c, OP_DRAW, p_draw) =
        (p_draw){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), (int)arg_x(c, 4), (int)arg_x(c, 5)};
}
// void nvnCommandBufferDrawElements(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, NVNindexType type, int count,
// NVNbufferAddress indices)
NVN_API(nvnCommandBufferDrawElements) {
    *REC(c, OP_DRAW_IDX, p_draw_idx) = (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), 0, 0, 1,
                                                    arg_x(c, 4)};
}
// void nvnCommandBufferDrawElementsBaseVertex(..., NVNbufferAddress indices, int base_vertex)
NVN_API(nvnCommandBufferDrawElementsBaseVertex) {
    *REC(c, OP_DRAW_IDX, p_draw_idx) = (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3),
                                                    (int)arg_x(c, 5), 0, 1, arg_x(c, 4)};
}
// void nvnCommandBufferDrawElementsInstanced(..., NVNbufferAddress indices, int base_vertex, int base_instance,
// int instances)
NVN_API(nvnCommandBufferDrawElementsInstanced) {
    *REC(c, OP_DRAW_IDX, p_draw_idx) = (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3),
                                                    (int)arg_x(c, 5), (int)arg_x(c, 6), (int)arg_x(c, 7), arg_x(c, 4)};
}
// void nvnCommandBufferDrawElementsIndirect(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, NVNindexType type,
// NVNbufferAddress indices, NVNbufferAddress indirect)
NVN_API(nvnCommandBufferDrawElementsIndirect) {
    *REC(c, OP_DRAW_IDX_IND, p_draw_idx_ind) = (p_draw_idx_ind){(int)arg_x(c, 1), (int)arg_x(c, 2), arg_x(c, 3),
                                                                arg_x(c, 4)};
}
// void nvnCommandBufferDrawMeshTasks(NVNcommandBuffer *cmd, int first, int count) (NVN2: the driver writes both words
// with one method; nxnative names them first and count)
NVN_API(nvnCommandBufferDrawMeshTasks) {
    *REC(c, OP_DRAW_MESH, p_draw_mesh) = (p_draw_mesh){(int)arg_x(c, 1), (int)arg_x(c, 2)};
}
// void nvnCommandBufferDrawMeshTasksIndirect(NVNcommandBuffer *cmd, NVNbufferAddress indirect) (NVN2: the driver reads
// that one argument)
NVN_API(nvnCommandBufferDrawMeshTasksIndirect) { REC(c, OP_DRAW_MESH_IND, p_draw_mesh_ind)->ind = arg_x(c, 1); }
// void nvnCommandBufferDispatchCompute(NVNcommandBuffer *cmd, int x, int y, int z): caller = the game's return address
NVN_API(nvnCommandBufferDispatchCompute) {
    *REC(c, OP_DISPATCH, p_dispatch) = (p_dispatch){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), c->x[30]};
}
// void nvnCommandBufferDispatchComputeIndirect(NVNcommandBuffer *cmd, NVNbufferAddress indirect)
NVN_API(nvnCommandBufferDispatchComputeIndirect) {
    *REC(c, OP_DISPATCH_IND, p_dispatch_ind) = (p_dispatch_ind){arg_x(c, 1), c->x[30]};
}
// void nvnCommandBufferBarrier(NVNcommandBuffer *cmd, int bits)
NVN_API(nvnCommandBufferBarrier) { REC(c, OP_BARRIER, p_barrier)->bits = (int)arg_x(c, 1); }

// ---------- copies
// void nvnCommandBufferCopyBufferToTexture(NVNcommandBuffer *cmd, NVNbufferAddress src, const NVNtexture *dst,
// const NVNtextureView *view, const NVNcopyRegion *region, int flags)
NVN_API(nvnCommandBufferCopyBufferToTexture) {
    p_copy_b2t *r = REC(c, OP_COPY_B2T, p_copy_b2t);
    r->addr = arg_x(c, 1);
    tex_rec(c, &r->tex, arg_p(c, 2), arg_p(c, 3), "CommandBufferCopyBufferToTexture");
    r->r = *(const NVNcopyRegion *)arg_p(c, 4);
    r->flags = (int)arg_x(c, 5);
}
// void nvnCommandBufferCopyTextureToBuffer(NVNcommandBuffer *cmd, const NVNtexture *src, const NVNtextureView *view,
// const NVNcopyRegion *region, NVNbufferAddress dst, int flags)
NVN_API(nvnCommandBufferCopyTextureToBuffer) {
    p_copy_b2t *r = REC(c, OP_COPY_T2B, p_copy_b2t);
    tex_rec(c, &r->tex, arg_p(c, 1), arg_p(c, 2), "CommandBufferCopyTextureToBuffer");
    r->r = *(const NVNcopyRegion *)arg_p(c, 3);
    r->addr = arg_x(c, 4);
    r->flags = (int)arg_x(c, 5);
}
// void nvnCommandBufferCopyTextureToTexture(NVNcommandBuffer *cmd, const NVNtexture *src, const NVNtextureView *
// src_view, const NVNcopyRegion *src_region, const NVNtexture *dst, const NVNtextureView *dst_view,
// const NVNcopyRegion *dst_region, int flags)
NVN_API(nvnCommandBufferCopyTextureToTexture) {
    const char *fn = "CommandBufferCopyTextureToTexture";
    p_copy_t2t *r = REC(c, OP_COPY_T2T, p_copy_t2t);
    tex_rec(c, &r->src, arg_p(c, 1), arg_p(c, 2), fn);
    tex_rec(c, &r->dst, arg_p(c, 4), arg_p(c, 5), fn);
    r->sr = *(const NVNcopyRegion *)arg_p(c, 3);
    r->dr = *(const NVNcopyRegion *)arg_p(c, 6);
    r->flags = (int)arg_x(c, 7);
}
// void nvnCommandBufferSetCopy{Row,Image}Stride(NVNcommandBuffer *cmd, ptrdiff_t stride)
NVN_API(nvnCommandBufferSetCopyRowStride) { REC(c, OP_COPY_STRIDE, p_copy_stride)->stride = (int64_t)arg_x(c, 1); }
NVN_API(nvnCommandBufferSetCopyImageStride) {
    REC(c, OP_COPY_IMAGE_STRIDE, p_copy_stride)->stride = (int64_t)arg_x(c, 1);
}

// ---------- variable-rate shading (NVN2)
// void nvnCommandBufferBindShadingRateImage(NVNcommandBuffer *cmd, const NVNtexture *texture, const NVNtextureView
// *view): the driver turns the first argument into a texture descriptor with the second; a null texture unbinds
NVN_API(nvnCommandBufferBindShadingRateImage) {
    p_shading_rate_image *r = REC(c, OP_SHADING_RATE_IMAGE, p_shading_rate_image);
    tex_rec(c, &r->tex, arg_p(c, 1), arg_p(c, 2), "CommandBufferBindShadingRateImage");
}
// void nvnCommandBufferShadingRateImagePalette(NVNcommandBuffer *cmd, int index, int n, const int *rates): the driver
// packs n rates of 4 bits into 16, the default ones when rates is null
NVN_API(nvnCommandBufferShadingRateImagePalette) {
    int index = (int)arg_x(c, 1), n = (int)arg_x(c, 2);
    const uint32_t *rates = arg_p(c, 3);
    if (n > 16) rt_unimplemented(c, "nvn", "CommandBufferShadingRateImagePalette: %d rates, past the 16 a palette holds", n);
    if (n < 0) n = 0;
    p_shading_rate_palette *r = REC(c, OP_SHADING_RATE_PALETTE, p_shading_rate_palette);
    r->index = index, r->n = n, r->has_rates = rates != NULL;
    if (rates) memcpy(r->rates, rates, sizeof r->rates[0] * (size_t)n);
}

// ---------- hints: how the GPU schedules or caches, never what it computes
static void hint(Cpu *c, int what, int32_t a, int32_t b, int32_t d) {
    *REC(c, OP_HINT, p_hint) = (p_hint){what, {a, b, d}};
}
// void nvnCommandBufferSetComputeOverlapMode(NVNcommandBuffer *cmd, int mode), SetComputeWorkloadPriority(cmd, int)
NVN_API(nvnCommandBufferSetComputeOverlapMode) { hint(c, NVN_HINT_COMPUTE_OVERLAP, (int32_t)arg_x(c, 1), 0, 0); }
NVN_API(nvnCommandBufferSetComputeWorkloadPriority) { hint(c, NVN_HINT_COMPUTE_PRIORITY, (int32_t)arg_x(c, 1), 0, 0); }
// void nvnCommandBufferSetColorReductionEnable(NVNcommandBuffer *cmd, NVNboolean enable); NVN2's
// SetColorReductionThresholds(cmd, NVNformatClass class, int low, int high), two integers in registers
NVN_API(nvnCommandBufferSetColorReductionEnable) { hint(c, NVN_HINT_COLOR_REDUCTION, nvn_arg_bool(c, 1) != 0, 0, 0); }
NVN_API(nvnCommandBufferSetColorReductionThresholds) {
    hint(c, NVN_HINT_COLOR_REDUCTION_THRESHOLDS, (int32_t)arg_x(c, 1), (int32_t)arg_x(c, 2), (int32_t)arg_x(c, 3));
}
// void nvnCommandBufferSetStencilCullCriteria(NVNcommandBuffer *cmd, NVNstencilFunc func, int ref, int mask)
NVN_API(nvnCommandBufferSetStencilCullCriteria) {
    hint(c, NVN_HINT_STENCIL_CULL, (int32_t)arg_x(c, 1), (int32_t)arg_x(c, 2), (int32_t)arg_x(c, 3));
}
// void nvnCommandBufferSetTiledCacheAction(NVNcommandBuffer *cmd, NVNtiledCacheAction action),
// SetTiledCacheTileSize(cmd, int width, int height)
NVN_API(nvnCommandBufferSetTiledCacheAction) { hint(c, NVN_HINT_TILED_CACHE_ACTION, (int32_t)arg_x(c, 1), 0, 0); }
NVN_API(nvnCommandBufferSetTiledCacheTileSize) {
    hint(c, NVN_HINT_TILED_CACHE_TILE_SIZE, (int32_t)arg_x(c, 1), (int32_t)arg_x(c, 2), 0);
}
