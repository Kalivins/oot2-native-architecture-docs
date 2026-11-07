@rem SPDX-License-Identifier: GPL-3.0-or-later
@rem Copyright (C) 2026 The oot2-native contributors
@echo off
rem The oot2 command line, with this folder's tools: oot2 prepare --dump FILE --dest FOLDER
setlocal
set "OOT2_HOME=%~dp0"
set "PATH=%OOT2_HOME%llvm\bin;%OOT2_HOME%bin;%OOT2_HOME%python;%PATH%"
set "VULKAN_SDK="
set "VK_SDK_PATH="
set "DAV1D_DIR="
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
"%OOT2_HOME%python\python.exe" -I -X utf8 "%OOT2_HOME%tools\oot2.py" %*
exit /b %ERRORLEVEL%
