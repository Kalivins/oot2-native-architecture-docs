// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// vkpipecost: the measurements of spike S3 (architecture D12, D13, section 6.1).
// Shared declarations of vkpc_main.c (device, job file, timing, output),
// vkpc_build.c (pipeline libraries, monolithic pipelines, shader objects,
// pipeline cache and pipeline binaries) and vkpc_draw.c (per-draw cost of the
// descriptor layouts, bind cost, replay with and without a recorder thread).
//
// Windows only: threads, the high-resolution clock and the process memory
// counters come from the Win32 API.

#ifndef VKPC_H
#define VKPC_H

#define VK_NO_PROTOTYPES
#include "volk.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_PATH_LEN 1024
#define MAX_BINDS 96

// ---- options and the device ---------------------------------------------------

typedef struct {
    int validation;          // 0 off, 1 core validation
    int internal_cache;      // 1 the driver's own pipeline cache as shipped, 0 disabled (VK_KHR_pipeline_binary)
    int threads;             // workers for builds
    char device_filter[256];
    char out[MAX_PATH_LEN];  // per-item rows (TSV)
} Options;

extern Options opt;
extern VkInstance inst;
extern VkPhysicalDevice gpu;
extern VkDevice dev;
extern VkQueue queue;
extern uint32_t queue_family;
extern VkPhysicalDeviceProperties gpu_props;
extern VkPhysicalDeviceMemoryProperties mem_props;
extern unsigned val_errors, val_warnings;
extern int have_binaries;    // VK_KHR_pipeline_binary enabled

__attribute__((noreturn, format(printf, 2, 3))) void die(int code, const char *fmt, ...);
#define VK_CHECK(call)                                                                   \
    do {                                                                                 \
        VkResult r_ = (call);                                                            \
        if (r_ != VK_SUCCESS) die(3, "%s failed: %d (%s:%d)", #call, (int)r_, __FILE__, __LINE__); \
    } while (0)

void device_init(void);
void device_finish(void);  // waits idle, destroys the device and the instance

// ---- time, memory, threads -------------------------------------------------------

double now_ms(void);
double thread_cpu_ms(void);          // CPU time of the calling thread
void memory_mb(double *working_set, double *private_bytes, double *peak_working_set);
double available_mb(void);

typedef void (*WorkFn)(void *ctx, uint32_t item, uint32_t worker);
// Runs fn for items 0..n-1 on `threads` workers pulling from a shared counter;
// returns the wall time in milliseconds.
double run_parallel(uint32_t n, int threads, WorkFn fn, void *ctx);

typedef struct Thread Thread;
Thread *thread_start(void (*fn)(void *), void *arg);
void thread_join(Thread *t);

// ---- the job file ---------------------------------------------------------------

// One descriptor binding of a set layout: binding, VkDescriptorType, count, stages.
typedef struct {
    uint32_t binding;
    VkDescriptorType type;
    uint32_t count;
    VkShaderStageFlags stages;
} Bind;

typedef struct {
    Bind b[MAX_BINDS];
    uint32_t n;
} SetDesc;

typedef struct {
    char path[MAX_PATH_LEN];
    VkShaderStageFlagBits stage;
    uint32_t *code;  // NULL until module_load
    size_t size;
} Module;

// A pipeline to build: two modules, its per-draw set, its colour targets.
typedef struct {
    uint32_t vs, fs;      // module indices
    uint32_t set;         // set description index
    uint32_t targets;     // colour attachments
    uint32_t format;      // index into formats
} Pair;

typedef struct {
    Module *modules;
    uint32_t nmodules;
    SetDesc *sets;
    uint32_t nsets;
    Pair *pairs;
    uint32_t npairs;
    SetDesc global;          // the update-after-bind set of D13 (its set number is global_set)
    uint32_t global_set;     // set number of the global set; the per-draw set takes the other of 0 and 1
    uint32_t has_global;
    uint32_t push_bytes;     // push constant range, all stages
    uint32_t rebind;         // added to the set 0 bindings (except 0) of fragment modules at load
    VkFormat formats[8];
    uint32_t nformats;
    VkFormat depth_format;
    char words[64][64];      // `param NAME VALUE` lines, read by each mode
    char values[64][MAX_PATH_LEN];
    uint32_t nparams;
} Job;

void job_read(const char *path, Job *job);
// Reads a module's SPIR-V (and moves a fragment module's set 0 bindings by
// job->rebind); module_free releases it. Both are safe from several threads
// for different modules.
void module_load(const Job *job, Module *m);
void module_free(Module *m);
// Adds `add` to the Binding of every set 0 resource except binding 0 (the
// support buffer both stages share); returns how many bindings moved.
uint32_t spirv_rebind(uint32_t *words, size_t nwords, uint32_t add);
const char *job_param(const Job *job, const char *name, const char *fallback);
long job_param_int(const Job *job, const char *name, long fallback);
VkShaderModule module_create(const Module *m);
VkDescriptorSetLayout set_layout_create(const SetDesc *d, VkDescriptorSetLayoutCreateFlags flags,
                                        int update_after_bind);
uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want);

// ---- output --------------------------------------------------------------------

// Per-item rows: phase, what, index, milliseconds, worker, extra (TSV).
void out_open(void);
void out_row(const char *phase, const char *what, long index, double ms, int worker, const char *extra);
void out_close(void);

// Summary line on stdout: a JSON object built by the caller, closed by summary_end.
void summary_begin(const char *mode);
void summary_num(const char *key, double value);
void summary_str(const char *key, const char *value);
void summary_end(void);

// ---- modes ------------------------------------------------------------------------

int mode_build(Job *job, const char *phase);
int mode_cache(Job *job, const char *phase);
int mode_draws(Job *job);
int mode_replay(Job *job);

// Dynamic state shared by pipelines and command buffers (vkpc_build.c).
void dynamic_states(VkGraphicsPipelineLibraryFlagsEXT part, VkDynamicState *out, uint32_t *n);
void set_all_dynamic_state(VkCommandBuffer cb, uint32_t width, uint32_t height, uint32_t targets, int shader_objects);

// The pipeline of two modules (vkpc_build.c): method 0 monolithic, 1 libraries
// and a fast link, 2 libraries and an optimised link. Libraries it creates are
// appended to libs (at most 4 per call) so that the caller destroys them last.
VkPipeline pipeline_for(const Job *job, int method, VkShaderModule vs, VkShaderModule fs, VkPipelineLayout layout,
                        uint32_t targets, VkFormat format, VkPipeline *libs, uint32_t *nlibs);
// Unlinked vertex and fragment shader objects of two modules.
void shader_objects_for(const Job *job, const Module *vs, const Module *fs, const VkDescriptorSetLayout *sets,
                        uint32_t nsets, VkShaderEXT out[2]);

#endif
