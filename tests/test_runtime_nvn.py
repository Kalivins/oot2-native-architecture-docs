# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the NVN2 front end: the seam (the entry-point table, the
bootstrap pair and the null back end), the device, memory pools, the GPU
address window and buffers (architecture D10, D14, section 2).

Builds tests/runtime/nvn_test.c with runtime/core and runtime/nvn against a
synthetic layout (one module of 0x8000 bytes), at -O1 and -O2 with
-march=x86-64-v3, then runs each case in its own process. A passing case must
print "ok <case>"; a failing case must end in exactly one FATAL line with the
expected exit code and text. Also pins the list of entry-point names
(runtime/nvn/nvn_procs.inc), checks that a body under a name the list does
not have fails to compile, and compares the device's answers with a table that
cites a source for each.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402

CORE = ROOT / "runtime" / "core"
NVN = ROOT / "runtime" / "nvn"
INCLUDE = ROOT / "runtime" / "include"
LIST = NVN / "nvn_procs.inc"
TEST_SOURCE = ROOT / "tests" / "runtime" / "nvn_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000

EXIT_FATAL, EXIT_UNIMPLEMENTED = 70, 71

# The list as taken from oot-pc-recomp at 8e09aaa (runtime/nvn_procs.inc and
# runtime/nvn_procs_nvn2.inc, merged and sorted): its size and the SHA-256 of
# the names joined by newlines. Changing the list means changing these two.
LIST_COUNT = 744
LIST_SHA256 = "3f0fa3853456a9e3c2d78d748f788f998cde182c41b98326ee2d59c04824303e"

# The entry points with a body: the bootstrap pair's, then the device's, the
# memory pools' and the buffers' (the device's command handles go with command
# buffers, its timestamps with queues), then the textures', the samplers', their
# pools' and the device's texture and sampler handles.
BODIES = {
    "nvnDeviceGetProcAddress",
    "nvnDeviceBuilderSetDefaults", "nvnDeviceBuilderSetFlags", "nvnDeviceBuilderGetFlags", "nvnDeviceInitialize",
    "nvnDeviceFinalize", "nvnDeviceGetFlags", "nvnDeviceSetDebugLabel", "nvnDeviceGetInteger",
    "nvnDeviceSetDepthMode", "nvnDeviceGetDepthMode", "nvnDeviceSetWindowOriginMode", "nvnDeviceGetWindowOriginMode",
    "nvnDeviceInstallDebugCallback",
    "nvnDeviceRegisterFastClearColor", "nvnDeviceRegisterFastClearColori", "nvnDeviceRegisterFastClearColorui",
    "nvnDeviceRegisterFastClearDepth", "nvnDeviceRegisterFastClearStencil", "nvnDeviceRegisterFastClearColorIndex",
    "nvnDeviceRegisterFastClearColoriIndex", "nvnDeviceRegisterFastClearColoruiIndex",
    "nvnDeviceRegisterFastClearDepthIndex", "nvnDeviceRegisterFastClearStencilIndex",
    "nvnMemoryPoolBuilderSetDefaults", "nvnMemoryPoolBuilderSetDevice", "nvnMemoryPoolBuilderSetStorage",
    "nvnMemoryPoolBuilderSetFlags", "nvnMemoryPoolBuilderGetDevice", "nvnMemoryPoolBuilderGetMemory",
    "nvnMemoryPoolBuilderGetSize", "nvnMemoryPoolBuilderGetFlags", "nvnMemoryPoolInitialize", "nvnMemoryPoolFinalize",
    "nvnMemoryPoolSetDebugLabel", "nvnMemoryPoolMap", "nvnMemoryPoolFlushMappedRange",
    "nvnMemoryPoolInvalidateMappedRange", "nvnMemoryPoolGetBufferAddress", "nvnMemoryPoolGetFlags",
    "nvnMemoryPoolGetSize", "nvnMemoryPoolMapVirtual",
    "nvnBufferBuilderSetDevice", "nvnBufferBuilderSetDefaults", "nvnBufferBuilderSetStorage",
    "nvnBufferBuilderGetDevice", "nvnBufferBuilderGetMemoryPool", "nvnBufferBuilderGetMemoryOffset",
    "nvnBufferBuilderGetSize", "nvnBufferInitialize", "nvnBufferFinalize", "nvnBufferSetDebugLabel",
    "nvnBufferGetAddress", "nvnBufferMap", "nvnBufferGetMemoryPool", "nvnBufferGetMemoryOffset", "nvnBufferGetSize",
    "nvnBufferFlushMappedRange", "nvnBufferInvalidateMappedRange",
    "nvnTextureBuilderSetDefaults", "nvnTextureBuilderSetDevice", "nvnTextureBuilderSetFlags",
    "nvnTextureBuilderSetTarget", "nvnTextureBuilderSetWidth", "nvnTextureBuilderSetHeight",
    "nvnTextureBuilderSetDepth", "nvnTextureBuilderSetSize1D", "nvnTextureBuilderSetSize2D",
    "nvnTextureBuilderSetSize3D", "nvnTextureBuilderSetLevels", "nvnTextureBuilderSetFormat",
    "nvnTextureBuilderSetSamples", "nvnTextureBuilderSetStride", "nvnTextureBuilderSetSwizzle",
    "nvnTextureBuilderSetDepthStencilMode", "nvnTextureBuilderSetStorage", "nvnTextureBuilderSetPackagedTextureLayout",
    "nvnTextureBuilderSetPackagedTextureData", "nvnTextureBuilderGetStorageSize",
    "nvnTextureBuilderGetStorageAlignment", "nvnTextureBuilderGetStorageClass", "nvnTextureBuilderGetZCullStorageSize",
    "nvnTextureBuilderGetDevice", "nvnTextureBuilderGetFlags", "nvnTextureBuilderGetTarget",
    "nvnTextureBuilderGetWidth", "nvnTextureBuilderGetHeight", "nvnTextureBuilderGetDepth",
    "nvnTextureBuilderGetLevels", "nvnTextureBuilderGetFormat", "nvnTextureBuilderGetSamples",
    "nvnTextureBuilderGetSwizzle", "nvnTextureBuilderGetDepthStencilMode", "nvnTextureBuilderGetStride",
    "nvnTextureBuilderGetMemoryPool", "nvnTextureBuilderGetMemoryOffset", "nvnTextureBuilderGetPackagedTextureData",
    "nvnTextureBuilderGetViewOffset", "nvnTextureInitialize", "nvnTextureFinalize", "nvnTextureSetDebugLabel",
    "nvnTextureGetFlags", "nvnTextureGetTarget", "nvnTextureGetWidth", "nvnTextureGetHeight", "nvnTextureGetDepth",
    "nvnTextureGetLevels", "nvnTextureGetFormat", "nvnTextureGetSamples", "nvnTextureGetSwizzle",
    "nvnTextureGetDepthStencilMode", "nvnTextureGetStride", "nvnTextureGetStorageSize", "nvnTextureGetStorageClass",
    "nvnTextureGetZCullStorageSize", "nvnTextureGetMemoryPool", "nvnTextureGetMemoryOffset",
    "nvnTextureGetTextureAddress", "nvnTextureGetDebugID", "nvnTextureGetViewOffset", "nvnTextureCompare",
    "nvnTextureViewSetDefaults", "nvnTextureViewSetLevels", "nvnTextureViewSetLayers", "nvnTextureViewSetFormat",
    "nvnTextureViewSetTarget", "nvnTextureViewSetSwizzle", "nvnTextureViewSetDepthStencilMode",
    "nvnTextureViewGetLevels", "nvnTextureViewGetLayers", "nvnTextureViewGetFormat", "nvnTextureViewGetTarget",
    "nvnTextureViewGetSwizzle", "nvnTextureViewGetDepthStencilMode", "nvnTextureViewCompare", "nvnTextureReadTexels",
    "nvnTextureReadTexelsStrided", "nvnTextureWriteTexels", "nvnTextureWriteTexelsStrided", "nvnTextureFlushTexels",
    "nvnTextureInvalidateTexels", "nvnSamplerBuilderSetDefaults", "nvnSamplerBuilderSetDevice",
    "nvnSamplerBuilderGetDevice", "nvnSamplerBuilderSetMinMagFilter", "nvnSamplerBuilderSetWrapMode",
    "nvnSamplerBuilderSetLodClamp", "nvnSamplerBuilderSetLodBias", "nvnSamplerBuilderSetMaxAnisotropy",
    "nvnSamplerBuilderSetLodSnap", "nvnSamplerBuilderSetCompare", "nvnSamplerBuilderSetReductionFilter",
    "nvnSamplerBuilderSetBorderColor", "nvnSamplerBuilderSetBorderColori", "nvnSamplerBuilderSetBorderColorui",
    "nvnSamplerBuilderGetMinMagFilter", "nvnSamplerBuilderGetWrapMode", "nvnSamplerBuilderGetLodClamp",
    "nvnSamplerBuilderGetLodBias", "nvnSamplerBuilderGetMaxAnisotropy", "nvnSamplerBuilderGetLodSnap",
    "nvnSamplerBuilderGetCompare", "nvnSamplerBuilderGetReductionFilter", "nvnSamplerBuilderGetBorderColor",
    "nvnSamplerBuilderGetBorderColori", "nvnSamplerBuilderGetBorderColorui", "nvnSamplerInitialize",
    "nvnSamplerFinalize", "nvnSamplerSetDebugLabel", "nvnSamplerGetMinMagFilter", "nvnSamplerGetWrapMode",
    "nvnSamplerGetLodClamp", "nvnSamplerGetLodBias", "nvnSamplerGetMaxAnisotropy", "nvnSamplerGetCompare",
    "nvnSamplerGetReductionFilter", "nvnSamplerGetBorderColor", "nvnSamplerGetBorderColori",
    "nvnSamplerGetBorderColorui", "nvnSamplerCompare", "nvnTexturePoolInitialize", "nvnSamplerPoolInitialize",
    "nvnTexturePoolFinalize", "nvnSamplerPoolFinalize", "nvnTexturePoolSetDebugLabel", "nvnSamplerPoolSetDebugLabel",
    "nvnTexturePoolGetMemoryPool", "nvnTexturePoolGetMemoryOffset", "nvnTexturePoolGetSize",
    "nvnSamplerPoolGetMemoryPool", "nvnSamplerPoolGetMemoryOffset", "nvnSamplerPoolGetSize",
    "nvnTexturePoolRegisterTexture", "nvnSamplerPoolRegisterSampler", "nvnSamplerPoolRegisterSamplerBuilder",
    "nvnDeviceGetTextureHandle", "nvnDeviceGetTexelFetchHandle", "nvnDeviceGetImageHandle",
    "nvnDeviceGetSeparateTextureHandle", "nvnDeviceGetSeparateSamplerHandle",
}
# Programs and state objects (runtime/nvn/nvn_program.c, nvn_state.c): the program entry points the game names, the
# builder's device and count, and every setter and getter of the eight state objects but programmable sample locations.
BODIES |= {
    "nvnProgramInitialize", "nvnProgramFinalize", "nvnProgramSetDebugLabel", "nvnProgramSetShaders",
    "nvnProgramSetShadersExt", "nvnProgramBuilderSetDefaults", "nvnProgramBuilderSetDevice",
    "nvnProgramBuilderGetDevice", "nvnProgramBuilderSetShaders", "nvnProgramBuilderGetShaderCount",
    "nvnProgramInitializeFromBuilder",
    "nvnBlendStateSetDefaults", "nvnBlendStateSetBlendTarget", "nvnBlendStateSetBlendFunc",
    "nvnBlendStateSetBlendEquation", "nvnBlendStateSetAdvancedMode", "nvnBlendStateSetAdvancedOverlap",
    "nvnBlendStateSetAdvancedPremultipliedSrc", "nvnBlendStateSetAdvancedNormalizedDst", "nvnBlendStateGetBlendTarget",
    "nvnBlendStateGetBlendFunc", "nvnBlendStateGetBlendEquation", "nvnBlendStateGetAdvancedMode",
    "nvnBlendStateGetAdvancedOverlap", "nvnBlendStateGetAdvancedPremultipliedSrc",
    "nvnBlendStateGetAdvancedNormalizedDst",
    "nvnColorStateSetDefaults", "nvnColorStateSetBlendEnable", "nvnColorStateSetLogicOp", "nvnColorStateSetAlphaTest",
    "nvnColorStateGetBlendEnable", "nvnColorStateGetLogicOp", "nvnColorStateGetAlphaTest",
    "nvnChannelMaskStateSetDefaults", "nvnChannelMaskStateSetChannelMask", "nvnChannelMaskStateGetChannelMask",
    "nvnMultisampleStateSetDefaults", "nvnMultisampleStateSetMultisampleEnable", "nvnMultisampleStateSetSamples",
    "nvnMultisampleStateSetAlphaToCoverageEnable", "nvnMultisampleStateSetAlphaToCoverageDither",
    "nvnMultisampleStateSetRasterSamples", "nvnMultisampleStateSetCoverageModulationMode",
    "nvnMultisampleStateSetCoverageTruncationEnable", "nvnMultisampleStateSetCoverageToColorEnable",
    "nvnMultisampleStateSetCoverageToColorOutput", "nvnMultisampleStateSetSampleLocationsEnable",
    "nvnMultisampleStateSetSampleLocationsGridEnable", "nvnMultisampleStateGetMultisampleEnable",
    "nvnMultisampleStateGetSamples", "nvnMultisampleStateGetAlphaToCoverageEnable",
    "nvnMultisampleStateGetAlphaToCoverageDither", "nvnMultisampleStateGetRasterSamples",
    "nvnMultisampleStateGetCoverageModulationMode", "nvnMultisampleStateGetCoverageTruncationEnable",
    "nvnMultisampleStateGetCoverageToColorEnable", "nvnMultisampleStateGetCoverageToColorOutput",
    "nvnMultisampleStateGetSampleLocationsEnable", "nvnMultisampleStateGetSampleLocationsGridEnable",
    "nvnPolygonStateSetDefaults", "nvnPolygonStateSetCullFace", "nvnPolygonStateSetFrontFace",
    "nvnPolygonStateSetPolygonMode", "nvnPolygonStateSetPolygonOffsetEnables", "nvnPolygonStateGetCullFace",
    "nvnPolygonStateGetFrontFace", "nvnPolygonStateGetPolygonMode", "nvnPolygonStateGetPolygonOffsetEnables",
    "nvnDepthStencilStateSetDefaults", "nvnDepthStencilStateSetDepthTestEnable",
    "nvnDepthStencilStateSetDepthWriteEnable", "nvnDepthStencilStateSetDepthFunc",
    "nvnDepthStencilStateSetStencilTestEnable", "nvnDepthStencilStateSetStencilFunc",
    "nvnDepthStencilStateSetStencilOp", "nvnDepthStencilStateGetDepthTestEnable",
    "nvnDepthStencilStateGetDepthWriteEnable", "nvnDepthStencilStateGetDepthFunc",
    "nvnDepthStencilStateGetStencilTestEnable", "nvnDepthStencilStateGetStencilFunc",
    "nvnDepthStencilStateGetStencilOp",
    "nvnVertexAttribStateSetDefaults", "nvnVertexAttribStateSetFormat", "nvnVertexAttribStateSetStreamIndex",
    "nvnVertexAttribStateGetFormat", "nvnVertexAttribStateGetStreamIndex",
    "nvnVertexStreamStateSetDefaults", "nvnVertexStreamStateSetStride", "nvnVertexStreamStateSetDivisor",
    "nvnVertexStreamStateGetStride", "nvnVertexStreamStateGetDivisor",
}

# Command buffers (runtime/nvn/nvn_cmd.c, nvn_cmd_ops.c): every command buffer entry point the game names, the rest
# of the object's memory and recording calls, and the device's FinalizeCommandHandle.
BODIES |= {
    "nvnCommandBufferAddCommandMemory", "nvnCommandBufferAddControlMemory", "nvnCommandBufferBarrier",
    "nvnCommandBufferBeginRecording", "nvnCommandBufferBindAccelerationStructuresDeferred",
    "nvnCommandBufferBindBlendState", "nvnCommandBufferBindChannelMaskState", "nvnCommandBufferBindColorState",
    "nvnCommandBufferBindDepthStencilState", "nvnCommandBufferBindImage", "nvnCommandBufferBindImages",
    "nvnCommandBufferBindMultisampleState", "nvnCommandBufferBindPolygonState", "nvnCommandBufferBindProgram",
    "nvnCommandBufferBindShadingRateImage", "nvnCommandBufferBindStorageBuffer",
    "nvnCommandBufferBindStorageBuffers", "nvnCommandBufferBindTexture", "nvnCommandBufferBindTextures",
    "nvnCommandBufferBindTexturesDeferred", "nvnCommandBufferBindUniformBuffer",
    "nvnCommandBufferBindVertexAttribState", "nvnCommandBufferBindVertexBuffer",
    "nvnCommandBufferBindVertexStreamState", "nvnCommandBufferCallCommands", "nvnCommandBufferClearBuffer",
    "nvnCommandBufferClearColor", "nvnCommandBufferClearDepthStencil", "nvnCommandBufferClearTexture",
    "nvnCommandBufferCopyBufferToTexture", "nvnCommandBufferCopyCommands", "nvnCommandBufferCopyTextureToBuffer",
    "nvnCommandBufferCopyTextureToTexture", "nvnCommandBufferDispatchCompute",
    "nvnCommandBufferDispatchComputeIndirect", "nvnCommandBufferDrawArrays", "nvnCommandBufferDrawArraysInstanced",
    "nvnCommandBufferDrawElements", "nvnCommandBufferDrawElementsBaseVertex",
    "nvnCommandBufferDrawElementsIndirect", "nvnCommandBufferDrawElementsInstanced",
    "nvnCommandBufferDrawMeshTasks", "nvnCommandBufferDrawMeshTasksIndirect", "nvnCommandBufferEndRecording",
    "nvnCommandBufferFinalize", "nvnCommandBufferGetCommandMemoryFree", "nvnCommandBufferGetCommandMemorySize",
    "nvnCommandBufferGetCommandMemoryUsed", "nvnCommandBufferGetControlMemoryFree",
    "nvnCommandBufferGetControlMemorySize", "nvnCommandBufferGetControlMemoryUsed",
    "nvnCommandBufferGetMemoryCallback", "nvnCommandBufferGetMemoryCallbackData", "nvnCommandBufferInitialize",
    "nvnCommandBufferIsRecording", "nvnCommandBufferPushComputeConstants", "nvnCommandBufferPushGraphicsConstants",
    "nvnCommandBufferReportCounter", "nvnCommandBufferReportValue", "nvnCommandBufferResetCounter",
    "nvnCommandBufferSetBlendColor", "nvnCommandBufferSetColorReductionEnable",
    "nvnCommandBufferSetColorReductionThresholds", "nvnCommandBufferSetCommandMemoryCallbackEnabled",
    "nvnCommandBufferSetComputeOverlapMode", "nvnCommandBufferSetComputeWorkloadPriority",
    "nvnCommandBufferSetConservativeRasterEnable", "nvnCommandBufferSetConservativeRasterMode",
    "nvnCommandBufferSetControlMemoryCallbackEnabled", "nvnCommandBufferSetCopyImageStride",
    "nvnCommandBufferSetCopyRowStride", "nvnCommandBufferSetDebugLabel", "nvnCommandBufferSetDepthBounds",
    "nvnCommandBufferSetDepthClamp", "nvnCommandBufferSetDepthRange", "nvnCommandBufferSetDepthRanges",
    "nvnCommandBufferSetGeometryProcessingMode", "nvnCommandBufferSetLineWidth",
    "nvnCommandBufferSetMemoryCallback", "nvnCommandBufferSetMemoryCallbackData", "nvnCommandBufferSetPointSize",
    "nvnCommandBufferSetPolygonOffsetClamp", "nvnCommandBufferSetPrimitiveRestart",
    "nvnCommandBufferSetRasterizerDiscard", "nvnCommandBufferSetRecordingMode", "nvnCommandBufferSetRenderEnable",
    "nvnCommandBufferSetRenderEnableConditional", "nvnCommandBufferSetRenderTargets",
    "nvnCommandBufferSetSampleMask", "nvnCommandBufferSetSamplerPool", "nvnCommandBufferSetScissor",
    "nvnCommandBufferSetScissors", "nvnCommandBufferSetShaderScratchMemory",
    "nvnCommandBufferSetShadingRatePaletteMode", "nvnCommandBufferSetStencilCullCriteria",
    "nvnCommandBufferSetStencilMask", "nvnCommandBufferSetStencilRef", "nvnCommandBufferSetStencilValueMask",
    "nvnCommandBufferSetTexturePool", "nvnCommandBufferSetTiledCacheAction",
    "nvnCommandBufferSetTiledCacheTileSize", "nvnCommandBufferSetViewport", "nvnCommandBufferSetViewports",
    "nvnCommandBufferShadingRateImagePalette", "nvnCommandBufferSignalEvent", "nvnCommandBufferUpdateUniformBuffer",
    "nvnCommandBufferWaitEvent", "nvnDeviceFinalizeCommandHandle",
}
# Queues, syncs, events and timestamps (runtime/nvn/nvn_queue.c), windows and QueuePresentTexture (nvn_window.c): every
# one of their entry points the game names or research 17's census met, the window's interval read back, and the
# device's current timestamp.
BODIES |= {
    "nvnQueueBuilderSetDefaults", "nvnQueueBuilderSetDevice", "nvnQueueBuilderSetFlags",
    "nvnQueueBuilderSetCommandMemorySize", "nvnQueueBuilderSetComputeMemorySize",
    "nvnQueueBuilderSetControlMemorySize", "nvnQueueBuilderGetQueueMemorySize", "nvnQueueBuilderSetQueueMemory",
    "nvnQueueBuilderSetPrimaryQueue", "nvnQueueInitialize", "nvnQueueFinalize", "nvnQueueSubmitCommands",
    "nvnQueueFlush", "nvnQueueFinish", "nvnQueueGetError", "nvnQueueFenceSync", "nvnQueueWaitSync",
    "nvnQueuePresentTexture", "nvnSyncInitialize", "nvnSyncFinalize", "nvnSyncWait",
    "nvnEventBuilderSetDefaults", "nvnEventBuilderSetStorage", "nvnEventInitialize", "nvnEventFinalize",
    "nvnEventGetValue", "nvnEventSignal", "nvnDeviceGetTimestampInNanoseconds",
    "nvnDeviceGetCurrentTimestampInNanoseconds",
    "nvnWindowBuilderSetDefaults", "nvnWindowBuilderSetDevice", "nvnWindowBuilderSetNativeWindow",
    "nvnWindowBuilderSetTextures", "nvnWindowBuilderSetNumActiveTextures", "nvnWindowBuilderSetPresentInterval",
    "nvnWindowInitialize", "nvnWindowFinalize", "nvnWindowSetCrop", "nvnWindowGetCrop",
    "nvnWindowSetNumActiveTextures", "nvnWindowSetPresentInterval", "nvnWindowGetPresentInterval",
    "nvnWindowAcquireTexture",
}

PASSING = ("table", "unknown", "unknown_many", "null_backend")

# case: (exit code, text the FATAL line must contain)
FAILING = {
    "unbodied": (EXIT_UNIMPLEMENTED, "FATAL: unimplemented nvn: nvnSyncInitializeFromFencedGLSync (pc 0x7ff"),
    "bootstrap_unregistered": (EXIT_FATAL, "FATAL: nvn: an entry point was asked for before nvn_procs_register"),
    "register_late": (EXIT_FATAL, "FATAL: nvn: nvn_procs_register after rt_dispatch_init"),
    "backend_partial": (EXIT_FATAL, "FATAL: gfx: back end partial lacks an operation"),
    "backend_late": (EXIT_FATAL, "FATAL: gfx: gfx_set_backend after rt_dispatch_init"),
}

UNKNOWN_LINE = '[rt] nvn: no entry point named "{}"; returning 0'

# nvnDeviceGetInteger: the console's answer to each item, grouped by where it comes from.
#   NVN2: the console's own answer, read in the switch and the table of nvnDeviceGetInteger in the game's NVN2 driver
#         (NVN 22.2.8, compiled into the game's sdk module), and the same value in nxnative's NVN2 table
#         (https://github.com/seedds/nxnative dm/gfx/nvn/hle_nvn.c, g_devinfo2, at 865cc35)
#   SDK:  the console's own answer, read in the game's driver only: nxnative's table says 39 for the API minor version
#         (0x01, which follows the SDK; oot-pc-recomp runtime/nvn_api_core.c at 8e09aaa notes that this game needs 40
#         or more) and 16 for the vertex buffer bindings (0x06, the number its host holds), and stops at 0xA9. Past
#         0xB2 the driver's lookup stores nothing, and the answer is 0.
#   X1:   an item the console reads from its GPU description. Below 0x60 nxnative (g_devinfo), DKB Recomp
#         (runtime/nvn_api_core.c, devinfo, at 9d8fa02) and oot-pc-recomp (the same table) answer the Tegra X1's value.
#   NONE: an item the console reads from its GPU description that no source gives: a named fatal.
NVN2 = {
    0x00: 231, 0x02: 14, 0x03: 65536, 0x04: 256, 0x05: 8, 0x07: 4, 0x08: 16, 0x09: 32, 0x0A: 16, 0x0B: 4, 0x0C: 4,
    0x0D: 4, 0x0E: 32, 0x0F: 32, 0x10: 32, 0x11: 256, 0x12: 256, 0x13: 4, 0x14: 8, 0x15: 512, 0x16: 256,
    0x19: 4096, 0x1A: 16384, 0x1B: 16, 0x1C: 1024, 0x1D: 1024, 0x1E: 64, 0x1F: 1024, 0x20: 2147483647, 0x21: 65535,
    0x22: 65535, 0x23: 8, 0x24: 1048576, 0x25: 4096, 0x26: 16, 0x27: 4096, 0x28: 1, 0x29: 1, 0x2A: 1, 0x2C: 16,
    0x2E: 201, 0x2F: 201, 0x30: 20, 0x31: 0, 0x32: 1, 0x33: 8, 0x34: 8, 0x35: 4, 0x36: 32, 0x39: 128, 0x3B: 1,
    0x3C: 4, 0x42: 1, 0x43: 1, 0x44: 1, 0x45: 1, 0x46: 5, 0x47: 1, 0x48: 1, 0x49: 1, 0x4A: 1, 0x4B: 1, 0x4C: 1,
    0x4D: 16, 0x4E: 2048, 0x4F: 32, 0x50: 4096, 0x51: 65536, 0x52: 65536, 0x53: 256, 0x54: 16384, 0x55: 262144,
    0x56: 4096, 0x57: 1, 0x58: 4, 0x59: 2, 0x5A: 0, 0x5B: 4096, 0x5C: 16384, 0x5D: 4096, 0x5E: 128, 0x5F: 32,
    0x60: 65535, 0x61: 1, 0x63: 256, 0x64: 16, 0x65: 65535, 0x66: 65535, 0x67: 256, 0x68: 1, 0x69: 16, 0x6A: 16,
    0x6B: 16, 0x6C: 1, 0x6D: 1, 0x6E: 1, 0x6F: 256, 0x70: 0, 0x71: 16, 0x72: 8, 0x73: 16, 0x74: 32, 0x75: 31,
    0x76: 536870912, 0x77: 16777216, 0x78: 16777216, 0x79: 1073741824, 0x7A: 32, 0x7B: 1, 0x7C: 64, 0x7E: 1,
    0x7F: 2147483647, 0x80: 32, 0x81: 32, 0x82: 4194303, 0x83: 512, 0x84: 256, 0x85: 32, 0x86: 16383, 0x88: 8,
    0x89: 0, 0x8A: 0, 0x8B: 4, 0x8C: 256, 0x8D: 16, 0x8E: 0, 0x8F: 1, 0x90: 31, 0x91: 0, 0x92: 64, 0x93: 1,
    0x94: 58368, 0x95: 24576, 0x96: 0, 0x97: 256, 0x98: 262144, 0x9B: 0, 0x9C: 201, 0x9D: 14, 0x9E: 201, 0x9F: 20,
    0xA0: 163840, 0xA1: 8, 0xA2: 4, 0xA3: 256, 0xA4: 256, 0xA5: 4, 0xA6: 65536, 0xA7: 201, 0xA8: 5, 0xA9: 1,
}
SDK = {
    0x01: 40, 0x06: 32, 0xAA: 0, 0xAB: 63, 0xAC: 1, 0xAD: 0, 0xAE: 1, 0xAF: 0, 0xB0: 4096, 0xB1: 32768, 0xB2: 32,
    **{item: 0 for item in range(0xB3, 0x110)},
}
X1 = {
    0x17: 2, 0x18: 256, 0x2B: 262144, 0x2D: 2048, 0x37: 128, 0x38: 32, 0x3A: 65536, 0x3D: 16384, 0x3E: 134217728,
    0x3F: 2048, 0x40: 16384, 0x41: 16384,
}
NONE = (0x62, 0x7D, 0x87, 0x99, 0x9A)
STANDIN_LINE = "[rt] nvn: nvnDeviceGetInteger item {:#x} answers {}, the Tegra X1's value"


def child_env() -> dict[str, str]:
    """The environment of a test program: census mode (OOT2_CENSUS, OOT2_CENSUS_FRAMES) is opt-in per case, never
    inherited from the shell that runs the suite (a gap is the named fatal here, not an answer of 0)."""
    return {k: v for k, v in os.environ.items() if not k.startswith("OOT2_CENSUS")}


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


def read_list() -> list[str]:
    names = []
    for line in LIST.read_text(encoding="utf-8").splitlines():
        if line.startswith("//") or not line:
            continue
        m = re.fullmatch(r"NVN_PROC\((nvn[A-Z]\w*)\)", line)
        if not m:
            raise AssertionError(f"{LIST.name}: not an NVN_PROC line: {line!r}")
        names.append(m.group(1))
    return names


def nvn_lines(stderr: str) -> list[str]:
    """The front end's log lines (guest memory logs its own at start-up)."""
    return [line for line in stderr.splitlines() if line.startswith("[rt] nvn: ")]


def compile_flags(cc: str) -> list[str]:
    return [cc, "-std=gnu11", "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", "-I", str(INCLUDE), "-I", str(NVN)]


class ListTest(unittest.TestCase):
    def test_list_is_sorted_unique_and_pinned(self) -> None:
        names = read_list()
        self.assertEqual(names, sorted(set(names)))
        self.assertEqual(len(names), LIST_COUNT)
        self.assertEqual(hashlib.sha256("\n".join(names).encode("ascii")).hexdigest(), LIST_SHA256)


class DeviceTableTest(unittest.TestCase):
    def test_every_item_has_one_source(self) -> None:
        groups = (set(NVN2), set(SDK), set(X1), set(NONE))
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                self.assertFalse(a & b)
        self.assertEqual(set().union(*groups), set(range(0x110)))


class NvnTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.cc = cc
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-nvn-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c")) + sorted(str(p) for p in NVN.glob("*.c"))
        # Windows: without high-entropy ASLR the host's own stacks stay below 4 GiB, out of the fixed module region
        # (runtime/core/memory.c, rt_mem_init); the program is linked the same way.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"nvn_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*compile_flags(cc), level, *extra, "-I", str(CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case, *args], capture_output=True, text=True, timeout=120,
                              env=child_env())

    def assert_passes(self, case: str, *args: str) -> list[subprocess.CompletedProcess]:
        runs = []
        for level in LEVELS:
            with self.subTest(level=level, case=case):
                done = self.run_case(level, case, *args)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                runs.append(done)
        return runs

    def assert_stops(self, case: str, code: int, text: str, *args: str) -> None:
        for level in LEVELS:
            with self.subTest(level=level, case=case, args=args):
                done = self.run_case(level, case, *args)
                self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                self.assertIn(text, done.stderr)
                self.assertEqual(done.stderr.count("FATAL"), 1, done.stderr)

    def test_passing_cases(self) -> None:
        for case in PASSING:
            self.assert_passes(case)

    def test_table_holds_exactly_the_list(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "table")
                rows = [line.split() for line in done.stdout.splitlines() if line.startswith("name ")]
                self.assertEqual([r[1] for r in rows], read_list())
                self.assertEqual({r[1] for r in rows if r[2] == "1"}, BODIES)
                self.assertIn(f"ok table {LIST_COUNT} {len(BODIES)}", done.stdout)

    def test_unknown_names_are_logged_once(self) -> None:
        done = self.run_case(LEVELS[0], "unknown")
        lines = nvn_lines(done.stderr)
        for name in ("nvnNotAName", "nvnDeviceGetProcAddresss", "(null)"):
            with self.subTest(name=name):
                self.assertEqual(lines.count(UNKNOWN_LINE.format(name)), 1, done.stderr)
        self.assertEqual(len(lines), 3, done.stderr)

    def test_unknown_names_past_the_cap_are_said_once(self) -> None:
        done = self.run_case(LEVELS[0], "unknown_many")
        lines = nvn_lines(done.stderr)
        logged = [f"nvnUnknown{i:02d}" for i in range(40) if UNKNOWN_LINE.format(f"nvnUnknown{i:02d}") in lines]
        self.assertEqual(logged, [f"nvnUnknown{i:02d}" for i in range(32)])
        self.assertEqual(lines.count("[rt] nvn: more than 32 unknown entry-point names; the rest are not logged"), 1)
        self.assertEqual(len(lines), 33, done.stderr)

    def test_fatal_cases(self) -> None:
        for case, (code, text) in FAILING.items():
            self.assert_stops(case, code, text)

    def check_body(self, name: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory(prefix="oot2-nvn-body-") as scratch:
            src = Path(scratch) / "body.c"
            src.write_text(f'#include "nvn_api.h"\nNVN_API({name}) {{ ret_x(c, 0); }}\n', encoding="ascii")
            return subprocess.run([*compile_flags(self.cc), "-fsyntax-only", str(src)], capture_output=True, text=True)

    def test_a_body_must_name_a_listed_entry_point(self) -> None:
        listed = self.check_body("nvnQueueFinish")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        unlisted = self.check_body("nvnNotAName")
        self.assertNotEqual(unlisted.returncode, 0)
        self.assertIn("NVN_ID_nvnNotAName", unlisted.stderr)

    # ---- the device

    def test_device(self) -> None:
        self.assert_passes("device")

    def test_device_answers(self) -> None:
        expected = {**NVN2, **SDK, **X1}
        for done in self.assert_passes("devinfo", *(f"{item:x}" for item in sorted(expected))):
            got = {int(m[1], 16): int(m[2]) for m in re.finditer(r"^item (0x[0-9a-f]+|0) (-?\d+)$", done.stdout, re.M)}
            self.assertEqual(got, expected)

    def test_device_stand_ins_are_logged_once(self) -> None:
        done = self.assert_passes("devinfo", "18", "18", "2b", "4")[0]
        lines = nvn_lines(done.stderr)
        for item in (0x18, 0x2B):
            said = [line for line in lines if line.startswith(STANDIN_LINE.format(item, X1[item]))]
            self.assertEqual(len(said), 1, done.stderr)
        self.assertEqual(len(lines), 2, done.stderr)

    def test_device_items_no_source_gives_stop(self) -> None:
        for item in NONE:
            text = f"FATAL: unimplemented nvn: nvnDeviceGetInteger item {item:#x}: the console reads it from its GPU"
            self.assert_stops("devinfo", EXIT_UNIMPLEMENTED, text, f"{item:x}")

    # ---- pools, buffers and the GPU address window

    def test_cpu_visible_buffer_address_is_the_guest_address(self) -> None:
        self.assert_passes("cpu_visible")

    def test_window_ranges_past_1024_and_reused(self) -> None:
        self.assert_passes("window")

    def test_virtual_pool_from_one_physical_base(self) -> None:
        done = self.assert_passes("virtual")[0]
        shape_prefix = "[rt] nvn: MemoryPoolMapVirtual virtual pool"
        shapes = [line for line in nvn_lines(done.stderr) if line.startswith(shape_prefix)]
        self.assertEqual(len(shapes), 3, done.stderr)
        self.assertIn("2 requests: [v+0 0x20000 <- 0x", shapes[0])
        self.assertTrue(shapes[0].endswith("; one physical run; 4 of 8 pages mapped"), shapes[0])
        self.assertIn("[v+0 0x40000 unmap]", shapes[1])
        self.assertTrue(shapes[1].endswith("; no mapping; 0 of 8 pages mapped"), shapes[1])
        refused = [line for line in nvn_lines(done.stderr) if "refused" in line]
        self.assertEqual(len(refused), 9, done.stderr)   # eight malformed pools and requests, one the back end refused

    def test_virtual_pool_from_two_runs_stops_with_its_shape(self) -> None:
        text = ("FATAL: unimplemented nvn: nvnMemoryPoolMapVirtual: request 1 would leave the pool's pages in two "
                "physical runs, which the descriptor path cannot express: virtual pool 0x")
        self.assert_stops("two_runs", EXIT_UNIMPLEMENTED, text)
        done = self.run_case(LEVELS[0], "two_runs")
        self.assertRegex(done.stderr, r"2 requests: \[v\+0 0x10000 <- 0x[0-9a-f]+\+0\] "
                                      r"\[v\+0x10000 0x10000 <- 0x[0-9a-f]+\+0x20000\]; two physical runs")

    def test_pool_lookup(self) -> None:
        self.assert_passes("lookup")

    def test_buffer_bench_runs(self) -> None:
        done = self.assert_passes("bench", "1000")[0]
        self.assertRegex(done.stdout, r"bench init [\d.]+ address [\d.]+ finalize [\d.]+ calls 1000")


if __name__ == "__main__":
    unittest.main()
