// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The audio renderer's face for the services (runtime/hos/srv_audio.c) and for tests: open a renderer from its
// parameters, hand it the update blocks the SDK's client builds, start and stop it, close it, read its position.
// The renderer runs on a host thread of its own that holds no virtual core, one frame (sample_count samples at
// sample_rate) at a time on a grid of due times (runtime/audio/frameclock.c), and signals the game's frame event once
// per rendered frame. Nothing here waits for a device.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "oot2/audio_sink.h"

typedef struct AudioRenderer AudioRenderer;

// The parameters the client opens a renderer with (the first fields of its AudioRendererParameter, read from the wire
// by audio_config_read).
typedef struct AudioConfig {
    int32_t sample_rate, sample_count, mix_buffer_count, submix_count, voice_count, sink_count, effect_count,
        performance_frame_count, splitter_count, splitter_destination_count;
    int32_t rendering_device, execution_mode;
    uint32_t revision_tag;   // the raw tag the client wrote ("REV" and a byte above 0x30), kept as it came
} AudioConfig;

// Read the 0x34 bytes of the wire parameter into cfg and check that the renderer can take them: NULL, or why not.
const char *audio_config_read(const void *wire, uint32_t size, AudioConfig *cfg);
// The revision a raw tag names (the top byte minus 0x30, for the upper-case and the lower-case spelling alike).
int audio_revision_number(uint32_t tag);

// A renderer in the stopped state. event is the writable end of the kernel event the frame signals go to; the renderer
// owns it from here and closes it at audio_renderer_close.
// readable is the client's end of the same event, for the stall detector's view of who waits on it (0: none is watched).
AudioRenderer *audio_renderer_open(const AudioConfig *cfg, uint32_t event, uint32_t readable);
void audio_renderer_start(AudioRenderer *r);   // the grid starts at this instant; a second Start while running does nothing
void audio_renderer_stop(AudioRenderer *r);    // returns when no frame is in progress and none will signal again
void audio_renderer_close(AudioRenderer *r);   // stops, then frees; the event's writable end is closed
int audio_renderer_running(const AudioRenderer *r);
const AudioConfig *audio_renderer_config(const AudioRenderer *r);

// One RequestUpdate: the client's input block in (in_size bytes), the output block out (out_size bytes, written whole).
// 1 on success with *out_used the bytes of the output block's own total; 0 with the reason in why (a section that does
// not match the layout of the revision, with the numbers). Takes the renderer's lock, so it sees a whole frame or none.
int audio_renderer_update(AudioRenderer *r, const uint8_t *in, uint64_t in_size, uint8_t *out, uint64_t out_size,
                          uint32_t *out_used, char *why, size_t why_size);

// A snapshot of the renderer's counters. frames: frames rendered; signals: frame events signalled; samples: frames x sample_count; skipped: frames the
// grid passed without rendering (a late wake-up renders once); wave_buffers_released: released to the client so far;
// updates: RequestUpdate calls served; late_max_ns: the latest a frame's signal went out after its due time.
typedef struct AudioPosition {
    uint64_t frames, samples, skipped, signals;
    uint64_t wave_buffers_released, updates;
    uint64_t late_max_ns;
    uint64_t update_gap_max_ns;   // the longest time between two updates while running
    uint64_t signal_to_update_max_ns;   // the longest time from a frame's signal to the next update
    int running;
    // the sink's side, in frames: handed to it, taken by it (written less queued), waiting; the times its queue ran empty
    // while playing and the writes it refused for want of room; the frames the renderer therefore dropped
    uint64_t sink_written, sink_played, sink_queued, underruns, overruns, dropped_frames;
    uint64_t stalls;                // stall reports so far (first lines of episodes), see audio_watchdog_polled_ns
    uint64_t event_wait_max_ns;     // the longest wait on a renderer's frame event the watchdog has seen
} AudioPosition;
void audio_renderer_position(const AudioRenderer *r, AudioPosition *out);
// The sum over the renderers open now (one in practice), and the figures of the last one closed when none is open.
void audio_position(AudioPosition *out);

// The file a capture sink writes (a WAV, for listening): under the game folder or a work folder, never a repository. Set
// before the renderer opens; "" or NULL keeps the capture in memory only.
void audio_set_capture_path(const char *path);
// The sink the renderer writes to, for tests that read what a capture sink kept (oot2/audio_sink.h).
AudioSink *audio_renderer_sink(const AudioRenderer *r);
AudioRenderer *audio_renderer_nth(int n);   // the n-th renderer open now (0 first), or NULL; for tests
// The device's output volume (IAudioDevice's SetAudioDeviceOutputVolume), which scales the mixed output: 1.0 at first.
void audio_device_volume_set(float v);
float audio_device_volume(void);

// The grid's time source and wait, for tests, as the virtual display clock has (oot2/vsync.h): now() is the time in
// nanoseconds, wait_until() returns when now() has reached the deadline or after wake() was called. wake may be NULL
// when wait_until polls for the stop itself. Set before the first Start, NULL restores the host's clock and sleep.
typedef struct AudioClockDriver {
    uint64_t (*now)(void *ctx);
    void (*wait_until)(void *ctx, uint64_t deadline_ns);
    void (*wake)(void *ctx);
    void *ctx;
} AudioClockDriver;
void audio_clock_drive(const AudioClockDriver *d);

// A tick on the renderer's grid (and its driven clock) for another host service: fn is called every count samples at rate Hz,
// on a thread of its own that holds no virtual core, never early; a late wake-up calls it once. The sound mixer service
// (srv_audsmx.c) takes its streams' frames with it.
typedef struct AudioTicker AudioTicker;
AudioTicker *audio_ticker_start(uint32_t count, uint32_t rate, void (*fn)(void *ctx), void *ctx);
void audio_ticker_stop(AudioTicker *t);   // returns when fn is not running and will not run again

// The stall detector (runtime/audio/stall.c): a watchdog thread of its own, on its own clock, reports a guest thread that
// waits on the frame event, a wave buffer that is not released, or a missing update, for longer than the bound: by name,
// never by silence. The bound for the event wait and for an update is one second (200 frames of 5 ms); a wave buffer's is
// its own duration, the buffers queued ahead of it, and one second.
#define AUDIO_STALL_BOUND_NS 1000000000ull
// The watchdog's time source and wait, for tests, as audio_clock_drive has for the renderer's grid: separate, so a test can
// freeze the renderer and move the watchdog's time. Set before the first renderer opens, NULL restores the host's.
void audio_watchdog_drive(const AudioClockDriver *d);
// One poll now, on the calling thread, as if the watchdog's clock read now_ns: a test moves the driven clock and polls at the
// instants it means to look at, with nothing left to chance. A poll that would go back in time is ignored.
void audio_watchdog_poll_at(uint64_t now_ns);
