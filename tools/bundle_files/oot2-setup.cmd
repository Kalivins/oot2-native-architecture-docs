@rem SPDX-License-Identifier: GPL-3.0-or-later
@rem Copyright (C) 2026 The oot2-native contributors
@echo off
rem Opens the oot2-native setup window. Everything it needs is in this folder: no other install is required.
setlocal
set "OOT2_HOME=%~dp0"
set "PATH=%OOT2_HOME%llvm\bin;%OOT2_HOME%bin;%OOT2_HOME%python;%PATH%"
rem Nothing of another toolchain on this PC may decide what the bundle builds.
set "VULKAN_SDK="
set "VK_SDK_PATH="
set "DAV1D_DIR="
set "OPUS_DIR="
set "SDL3_DIR="
set "PYTHONPATH="
set "PYTHONHOME="
set "OOT2_CC="
set "OOT2_CXX="
set "OOT2_NINJA="
set "OOT2_XLATE="
set "OOT2_GLSLANG="
set "OOT2_IMGUI="
set "OOT2_RUNTIME_LIB="
start "" "%OOT2_HOME%python\pythonw.exe" "%OOT2_HOME%tools\oot2_setup.pyw" %*
