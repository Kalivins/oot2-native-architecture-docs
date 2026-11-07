// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The device (architecture D10): its builder, initialize and finalize, the modes the game sets on it, and
// nvnDeviceGetInteger answered with the console's values.
//
// Where the answers come from. NVN2 is compiled into the game's own sdk module, and its nvnDeviceGetInteger answers
// from a switch over the items that falls back on a table lookup; seventeen items read the device's GPU description
// instead, which only the running console has. Both the switch and the table were read in the game's own driver (NVN
// 22.2.8), and they agree item for item with nxnative's NVN2 table (dm/gfx/nvn/hle_nvn.c g_devinfo2 at 865cc35, MIT,
// read from Donkey Kong Bananza's NVN2 and SDK 21.4.4's) except two: the API minor version, 40 here and 39 there,
// which follows the SDK (oot-pc-recomp at 8e09aaa records that this game creates its graphics system only for 40 or
// more), and the vertex buffer bindings, 32 on the console, which nxnative answers as 16 because its host holds 16.
// Items 0xAA to 0xB2 are answered by the table alone, and every item past it reads 0, as the driver's lookup stores
// nothing there.
//
// The GPU-description items have no source for this console. Below 0x60, nxnative, DKB Recomp and oot-pc-recomp all
// answer the Tegra X1's value (nxnative g_devinfo, measured on the first Switch by asking each item), so that stand-in
// is answered here too and logged the first time each one is asked; the game asks for one of them, 0x18. The five
// above 0x60 have no answer anywhere and stop as a named fatal.
//
// The NVN_API bodies are ported from oot-pc-recomp runtime/nvn_api_core.c at 8e09aaa (GPL-3.0-or-later), as DKB Recomp
// carries it at 9d8fa02, ported there from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c at b75be8cf
// (MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile; licenses/dkb-xlate-MIT.txt). Changed here: the answer table
// (above), the depth and window origin modes kept for the back end from DeviceInitialize on, and no trace lines.
#include <string.h>

#include "nvn_api.h"

#define DEV_MAGIC MAGIC('D', 'E', 'V', 0)

enum { GPU = INT32_MIN };   // the console reads the item from its GPU description at run time

static const int32_t devinfo[] = {
    [0x00] = 231, [0x01] = 40, [0x02] = 14, [0x03] = 65536, [0x04] = 256, [0x05] = 8, [0x06] = 32, [0x07] = 4,
    [0x08] = 16, [0x09] = 32, [0x0a] = 16, [0x0b] = 4, [0x0c] = 4, [0x0d] = 4, [0x0e] = 32, [0x0f] = 32, [0x10] = 32,
    [0x11] = 256, [0x12] = 256, [0x13] = 4, [0x14] = 8, [0x15] = 512, [0x16] = 256, [0x17] = GPU, [0x18] = GPU,
    [0x19] = 4096, [0x1a] = 16384, [0x1b] = 16, [0x1c] = 1024, [0x1d] = 1024, [0x1e] = 64, [0x1f] = 1024,
    [0x20] = 2147483647, [0x21] = 65535, [0x22] = 65535, [0x23] = 8, [0x24] = 1048576, [0x25] = 4096, [0x26] = 16,
    [0x27] = 4096, [0x28] = 1, [0x29] = 1, [0x2a] = 1, [0x2b] = GPU, [0x2c] = 16, [0x2d] = GPU, [0x2e] = 201,
    [0x2f] = 201, [0x30] = 20, [0x31] = 0, [0x32] = 1, [0x33] = 8, [0x34] = 8, [0x35] = 4, [0x36] = 32, [0x37] = GPU,
    [0x38] = GPU, [0x39] = 128, [0x3a] = GPU, [0x3b] = 1, [0x3c] = 4, [0x3d] = GPU, [0x3e] = GPU, [0x3f] = GPU,
    [0x40] = GPU, [0x41] = GPU, [0x42] = 1, [0x43] = 1, [0x44] = 1, [0x45] = 1, [0x46] = 5, [0x47] = 1, [0x48] = 1,
    [0x49] = 1, [0x4a] = 1, [0x4b] = 1, [0x4c] = 1, [0x4d] = 16, [0x4e] = 2048, [0x4f] = 32, [0x50] = 4096,
    [0x51] = 65536, [0x52] = 65536, [0x53] = 256, [0x54] = 16384, [0x55] = 262144, [0x56] = 4096, [0x57] = 1,
    [0x58] = 4, [0x59] = 2, [0x5a] = 0, [0x5b] = 4096, [0x5c] = 16384, [0x5d] = 4096, [0x5e] = 128, [0x5f] = 32,
    [0x60] = 65535, [0x61] = 1, [0x62] = GPU, [0x63] = 256, [0x64] = 16, [0x65] = 65535, [0x66] = 65535, [0x67] = 256,
    [0x68] = 1, [0x69] = 16, [0x6a] = 16, [0x6b] = 16, [0x6c] = 1, [0x6d] = 1, [0x6e] = 1, [0x6f] = 256, [0x70] = 0,
    [0x71] = 16, [0x72] = 8, [0x73] = 16, [0x74] = 32, [0x75] = 31, [0x76] = 536870912, [0x77] = 16777216,
    [0x78] = 16777216, [0x79] = 1073741824, [0x7a] = 32, [0x7b] = 1, [0x7c] = 64, [0x7d] = GPU, [0x7e] = 1,
    [0x7f] = 2147483647, [0x80] = 32, [0x81] = 32, [0x82] = 4194303, [0x83] = 512, [0x84] = 256, [0x85] = 32,
    [0x86] = 16383, [0x87] = GPU, [0x88] = 8, [0x89] = 0, [0x8a] = 0, [0x8b] = 4, [0x8c] = 256, [0x8d] = 16,
    [0x8e] = 0, [0x8f] = 1, [0x90] = 31, [0x91] = 0, [0x92] = 64, [0x93] = 1, [0x94] = 58368, [0x95] = 24576,
    [0x96] = 0, [0x97] = 256, [0x98] = 262144, [0x99] = GPU, [0x9a] = GPU, [0x9b] = 0, [0x9c] = 201, [0x9d] = 14,
    [0x9e] = 201, [0x9f] = 20, [0xa0] = 163840, [0xa1] = 8, [0xa2] = 4, [0xa3] = 256, [0xa4] = 256, [0xa5] = 4,
    [0xa6] = 65536, [0xa7] = 201, [0xa8] = 5, [0xa9] = 1, [0xaa] = 0, [0xab] = 63, [0xac] = 1, [0xad] = 0, [0xae] = 1,
    [0xaf] = 0, [0xb0] = 4096, [0xb1] = 32768, [0xb2] = 32,
};
#define DEVINFO_ITEMS (sizeof devinfo / sizeof *devinfo)

// The Tegra X1's values of the GPU-description items below 0x60 (nxnative g_devinfo; the same in DKB Recomp's and
// oot-pc-recomp's tables): scratch scale factors, L2 size, texture layers and sizes, linear texture stride alignment,
// memory pool page size.
static const int32_t x1_standin[0x60] = {
    [0x17] = 2, [0x18] = 256, [0x2b] = 262144, [0x2d] = 2048, [0x37] = 128, [0x38] = 32, [0x3a] = 65536,
    [0x3d] = 16384, [0x3e] = 134217728, [0x3f] = 2048, [0x40] = 16384, [0x41] = 16384,
};

static int g_depth_mode, g_origin_mode;
int nvn_depth_mode(void) { return __atomic_load_n(&g_depth_mode, __ATOMIC_RELAXED); }
int nvn_window_origin_mode(void) { return __atomic_load_n(&g_origin_mode, __ATOMIC_RELAXED); }

// void nvnDeviceBuilderSetDefaults(NVNdeviceBuilder *b), void ...SetFlags(b, int flags), int ...GetFlags(b)
NVN_API(nvnDeviceBuilderSetDefaults) { memset(arg_p(c, 0), 0, sizeof(dev_builder_t)); }
NVN_API(nvnDeviceBuilderSetFlags) { ((dev_builder_t *)arg_p(c, 0))->flags = (uint32_t)arg_x(c, 1); }
NVN_API(nvnDeviceBuilderGetFlags) { ret_x(c, ((const dev_builder_t *)arg_p(c, 0))->flags); }

// NVNboolean nvnDeviceInitialize(NVNdevice *d, const NVNdeviceBuilder *b): NVN's defaults are depth mode
// NEAR_IS_MINUS_W and window origin LOWER_LEFT, both 0
NVN_API(nvnDeviceInitialize) {
    dev_t_ *o = arg_p(c, 0);
    *o = (dev_t_){.magic = DEV_MAGIC, .flags = ((const dev_builder_t *)arg_p(c, 1))->flags};
    __atomic_store_n(&g_depth_mode, NVN_DEPTH_MODE_NEAR_IS_MINUS_W, __ATOMIC_RELAXED);
    __atomic_store_n(&g_origin_mode, NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT, __ATOMIC_RELAXED);
    ret_x(c, 1);
}
// void nvnDeviceFinalize(NVNdevice *d)
NVN_API(nvnDeviceFinalize) { ((dev_t_ *)arg_p(c, 0))->magic = 0; }
// int nvnDeviceGetFlags(const NVNdevice *d)
NVN_API(nvnDeviceGetFlags) { ret_x(c, ((const dev_t_ *)arg_p(c, 0))->flags); }
// void nvnDeviceSetDebugLabel(NVNdevice *d, const char *label): for NVN's debugging tools only
NVN_API(nvnDeviceSetDebugLabel) { (void)c; }

// void nvnDeviceGetInteger(const NVNdevice *d, NVNdeviceInfo what, int *out)
NVN_API(nvnDeviceGetInteger) {
    uint32_t w = (uint32_t)arg_x(c, 1);
    int32_t *out = arg_p(c, 2);
    int32_t v = w < DEVINFO_ITEMS ? devinfo[w] : 0;
    if (v == GPU) {
        if (w >= sizeof x1_standin / sizeof *x1_standin || !x1_standin[w])
            rt_unimplemented(c, "nvn",
                             "nvnDeviceGetInteger item %#x: the console reads it from its GPU description and no "
                             "source gives its value",
                             w);
        v = x1_standin[w];
        static uint64_t said[2];
        uint64_t bit = 1ull << (w % 64);
        if (!(__atomic_fetch_or(&said[w / 64], bit, __ATOMIC_RELAXED) & bit))
            rt_log("nvn: nvnDeviceGetInteger item %#x answers %d, the Tegra X1's value: the console reads it from "
                   "its GPU description",
                   w, v);
    }
    *out = v;
}

// void nvnDeviceSetDepthMode(NVNdevice *d, NVNdepthMode mode), NVNdepthMode ...GetDepthMode(d)
NVN_API(nvnDeviceSetDepthMode) {
    uint32_t m = (uint32_t)arg_x(c, 1);
    ((dev_t_ *)arg_p(c, 0))->depth_mode = m;
    __atomic_store_n(&g_depth_mode, (int)m, __ATOMIC_RELAXED);
}
NVN_API(nvnDeviceGetDepthMode) { ret_x(c, ((const dev_t_ *)arg_p(c, 0))->depth_mode); }
// void nvnDeviceSetWindowOriginMode(NVNdevice *d, NVNwindowOriginMode mode), NVNwindowOriginMode ...GetWindowOriginMode
NVN_API(nvnDeviceSetWindowOriginMode) {
    uint32_t m = (uint32_t)arg_x(c, 1);
    ((dev_t_ *)arg_p(c, 0))->origin = m;
    __atomic_store_n(&g_origin_mode, (int)m, __ATOMIC_RELAXED);
}
NVN_API(nvnDeviceGetWindowOriginMode) { ret_x(c, ((const dev_t_ *)arg_p(c, 0))->origin); }

// void nvnDeviceInstallDebugCallback(NVNdevice *d, PFNNVNDEBUGCALLBACKPROC cb, void *data, NVNboolean enable): kept,
// never called, as no debug layer runs here
NVN_API(nvnDeviceInstallDebugCallback) {
    dev_t_ *o = arg_p(c, 0);
    o->dbg_cb = nvn_arg_bool(c, 3) ? arg_p(c, 1) : NULL;
    o->dbg_data = arg_p(c, 2);
}

// NVNboolean nvnDeviceRegisterFastClear{Color,Colori,Colorui,Depth,Stencil}[Index](NVNdevice *d, ...): a fast clear is
// a hint to the console's GPU; "not registered" keeps the ordinary clear, which gives the same picture
NVN_API(nvnDeviceRegisterFastClearColor) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearColori) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearColorui) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearDepth) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearStencil) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearColorIndex) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearColoriIndex) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearColoruiIndex) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearDepthIndex) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearStencilIndex) { ret_x(c, 0); }
