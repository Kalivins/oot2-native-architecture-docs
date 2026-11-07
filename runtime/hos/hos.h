// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The Horizon layer under the SDK (architecture D9): the boot driver, the service manager and its sessions, and the
// host functions that answer the public nn:: entry points which would reach a service on the console. Shared by the
// files of runtime/hos, never included by generated code.
//
// Rule for every override here: queries and initialisation answer as a console would; a path that would need a real
// service (a data path, a network request) answers as a console without that service does, when that answer is known
// and recorded (offline), else it is a named fatal. Nothing returns a made-up success in silence: each override names
// its answer for the census with HOS_OVERRIDE.
//
// Start-up order (runtime/program/main.c): rt_mem_init, rt_kernel_init, rt_load_modules, nvn_procs_register,
// hos_init (the service SVCs, the boot driver's host functions), rt_dispatch_init, then rt_kernel_run(hos_boot_entry()).
#pragma once
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include "oot2/boot.h"
#include "oot2/kernel.h"
#include "oot2/loader.h"
#include "oot2/memory.h"
#include "oot2/program.h"
#include "oot2/runtime.h"

// ---- what an override answers, for the census (census.c)
//
// HOS_OVERRIDE(symbol, fn, status, why) is HOST_OVERRIDE(symbol, fn) plus a note the census prints for every override
// the run called: its status (one of the five below) and one line on the answer and its source. An override bound with
// a bare HOST_OVERRIDE counts as implemented, with no note.
#define HOS_IMPLEMENTED "implemented"   // does what the console's service does, for what the game asks
#define HOS_OFFLINE "offline"           // answers as a console with no network, no account online, no NFC reader
#define HOS_STUB "stub"                 // accepted with no effect, because the host has nothing to change
#define HOS_ABSENT "absent"             // answers that the thing does not exist (no add-on content, no font file)
// Not answered yet: the override is the named fatal of the call (hos_gap), so the run stops at the entry point the
// game called instead of later inside a service nothing serves. The census word of every named fatal.
#define HOS_GAP "named-fatal"

typedef struct HosNote {
    const char *symbol, *status, *why;
    const void *pad;   // 32 bytes, so linker padding between objects reads as empty notes (hostfn.c)
} HosNote;
#ifdef _WIN32
#define HOS_NOTE_SECTION section(".hosn$m")
#else
#define HOS_NOTE_SECTION section("oot2_hosn")
#endif
#define HOS_NOTE(sym, status, why)                                                                               \
    __attribute__((used, HOS_NOTE_SECTION, aligned(32))) static const HosNote RT_CAT(hos_note_, __COUNTER__) = { \
        sym, status, why, 0}
#define HOS_OVERRIDE(sym, fn, status, why) \
    HOST_OVERRIDE(sym, fn);                \
    HOS_NOTE(sym, status, why)

const HosNote *hos_note(const char *symbol);   // NULL when no override of that name has one

// ---- calling guest code from an override
// Run the guest function at fn with n (at most 8) integer arguments in x0.. on this thread and return its x0. The
// override's own pc and x30 are kept, so a named fatal after the call still names the override and its caller.
uint64_t hos_call(Cpu *c, uint64_t fn, int n, ...);
// The SDK function of that exact (mangled) name, found in the loaded modules; a named fatal of kind boot when no
// module defines it, since the runtime then cannot answer as the console does.
uint64_t hos_sdk(Cpu *c, const char *symbol);

// ---- guest memory and arguments
// A guest string of at most n - 1 bytes into buf (always terminated); "(null)" for address 0.
const char *hos_str(uint64_t addr, char *buf, size_t n);
// Where an override was called from: the guest address before x30, as rt_guest_name prints it.
const char *hos_caller(const Cpu *c, char *buf, int n);
// Write size bytes into guest memory at addr for an override's out parameter: 1, or 0 when the range is not guest
// memory the guest may write (the caller answers with an error, as the console's kernel would refuse the copy).
int hos_out(uint64_t addr, const void *src, uint64_t size);
int hos_in(void *dst, uint64_t addr, uint64_t size);

// ---- gaps: what the runtime does not answer yet
// The named fatal of a gap met by the guest: "FATAL: unimplemented <kind>: <message>", then the guest backtrace from
// x29, so the fatal names the SDK function whose path reached the gap. A run that stops is one census line.
void hos_gap(Cpu *c, const char *kind, const char *fmt, ...) __attribute__((noreturn, format(printf, 3, 4)));

// ---- kernel events for the SDK's system events
// Make the guest's nn::os::SystemEventType at ev a kernel event the host signals: a host event is created and its
// readable end attached through the SDK's own nn::os::AttachReadableHandleToSystemEvent, so the guest waits on it with
// the SDK's own code. Returns the writable end (hos_signal). The same ev gives the same event.
uint32_t hos_bind_system_event(Cpu *c, uint64_t ev, int autoclear);
void hos_signal(uint32_t writable);
// ov_audio.c: the audio stand-in's OpenAudioRenderer after the game's SystemEvent is bound: event is the writable end the
// renderer's clock thread signals once per frame, handle the guest address that gets the renderer's handle, param the
// guest address of the AudioRendererParameter. Split from the override so a test can pass a kernel event of its own.
void hos_audio_renderer_open(Cpu *c, uint32_t event, uint64_t handle, uint64_t param);
// How many clock threads the stand-in has started: one for the life of the process, however many renderers are opened.
int hos_audio_clock_threads(void);

// ov_codec.c: the AV1 video decoder service under the SDK's movie library (the movie player). Its submission-ready events are
// made with hos_bind_system_event; a test gives it a binder of its own (the synthetic module has no SDK to attach
// through). NULL restores hos_bind_system_event.
void hos_codec_set_event_binder(uint32_t (*bind)(Cpu *c, uint64_t ev, int autoclear));
// The same for the two nn::os calls the sync point events make (ClearSystemEvent, and InitializeMultiWaitHolder over a system
// event); NULL restores the SDK's own, found with hos_sdk.
void hos_codec_set_event_hooks(void (*clear)(Cpu *c, uint64_t ev), void (*init_holder)(Cpu *c, uint64_t holder, uint64_t ev));

// ---- the game folder (the folder oot2 prepare wrote, which holds the program): set once by hos_init
const char *hos_game_folder(void);
// The file the program glue lists under that description ("RomFS image", "RomFS index"), as a path the host opens;
// NULL when the build has none.
const char *hos_game_file(const char *what, char *buf, size_t n);

// ---- start-up: hos_init, hos_boot_entry and hos_census_write are declared in oot2/boot.h (main.c calls them)
// Counters the IPC layer and the overrides keep for the census.
void hos_census_service(const char *name, const char *status, const Cpu *c);
void hos_census_command(const char *service, const char *framing, uint32_t cmd, const char *status, const Cpu *c);

// ---- graphics bring-up (ov_gfx.c), for the NVN2 front end and a future presenter
// The guest's graphics allocator callbacks (nv::SetGraphicsAllocator and nv::SetGraphicsDevtoolsAllocator): guest
// function addresses of void *allocate(size_t size, size_t align, void *user), void release(void *ptr, void *user)
// and void *reallocate(void *ptr, size_t size, void *user), and the user pointer they take. 0 until the game sets them.
typedef struct HosNvAllocator { uint64_t allocate, release, reallocate, user; } HosNvAllocator;
// What nv::* recorded: the two allocators, the work memory of nv::InitializeGraphics (guest address and size, both 0
// when the game hands the driver no buffer) and whether nv::InitializeGraphics ran.
typedef struct HosNvState {
    HosNvAllocator graphics, devtools;
    uint64_t memory, memory_size;
    int initialized;
} HosNvState;
// The record itself, not a copy. The game sets it once during its graphics bring-up, before the NVN device exists, so
// the NVN layer reads it afterwards without a lock.
const HosNvState *hos_nv_state(void);
// The identity nn::vi::GetNativeWindow hands out for the one layer: a zeroed block in the kernel region, which the NVN
// layer recognises as the window. 0 until nn::vi::Initialize ran.
uint64_t hos_vi_native_window(void);
// The writable ends of the system events nn::vi::GetDisplayVsyncEvent bound (at most 4), copied to
// writable[0..max); returns how many. The virtual 60 Hz clock (oot2/vsync.h) signals each at every tick; nothing else
// signals them, a presenter included, so the game sees one vsync per tick whatever the monitor does.
int hos_vsync_events(uint32_t *writable, int max);

// ---- applet messages (ov_system.c): queue a message nn::oe::PopNotificationMessage hands out (focus, operation
// mode), and signal the notification event, for the window or lifecycle code that knows of one
void hos_oe_push_message(uint32_t message);

// Horizon results the overrides return (module | description << 9; Switchbrew, "Error codes").
#define HOS_R_OK 0u
// A generic failure for the online paths that are never served (module 162, description 1). The code a console returns
// offline is not recorded here; NXNative answers this one for every cut online feature, and a failure of any code ends
// the game's online path.
#define HOS_R_UNAVAILABLE 0x2a2u

// ---- between the files of runtime/hos
void boot_register(void);    // boot.c: the host start function and the three callbacks of nn::init::Start
void ipc_register(void);     // ipc.c: the service SVCs
void census_init(void);      // census.c: the stop hook
