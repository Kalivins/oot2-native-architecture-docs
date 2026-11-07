# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Power management for the length of an `oot2 prepare` run: the PC stays awake, and Windows does not throttle the
processes that do the work. A first run takes an hour or more; on a laptop that is on battery or a PC that sleeps after
a few idle minutes it would otherwise stop halfway or crawl on the efficiency cores.

- keep_awake(): SetThreadExecutionState with ES_SYSTEM_REQUIRED for as long as the returned guard is open. The state
  belongs to the calling thread, so the guard is made and closed on the thread that runs the steps. The display may
  still turn off. Nothing is asked of the system on other platforms.
- opt_out_of_throttling(handle): SetProcessInformation with ProcessPowerThrottling and the execution-speed bit
  controlled and not set, which is the documented way for a process to say that it must not run in the efficiency
  (EcoQoS) class. The run does not count on a parent's setting reaching its children, so install() wraps
  subprocess.Popen: every process this run starts gets the call right after it starts. Processes those start in
  turn (the compiler's own children, for instance) are not reached; they run at the class the system gives them.

Both calls fail quietly: an older Windows without the function, or a process that cannot be opened, is not a reason to
stop a run, and the run says nothing about it. (The calls report their result, which the tests read; Windows has no
query for the setting that works on this build, GetProcessInformation answers ERROR_INVALID_PARAMETER for it, so that
the tests can show that the system accepted the call and not what the system then does.)

Written for this project; no code of another project.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys

WIN = sys.platform == "win32"
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
PROCESS_POWER_THROTTLING = 4                 # PROCESS_INFORMATION_CLASS ProcessPowerThrottling
THROTTLING_VERSION = 1                       # PROCESS_POWER_THROTTLING_CURRENT_VERSION
EXECUTION_SPEED = 0x1                        # PROCESS_POWER_THROTTLING_EXECUTION_SPEED


class _ThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.SetThreadExecutionState.argtypes = [ctypes.c_ulong]
    k.SetThreadExecutionState.restype = ctypes.c_ulong
    k.GetCurrentProcess.restype = ctypes.c_void_p
    if hasattr(k, "SetProcessInformation"):
        k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        k.SetProcessInformation.restype = ctypes.c_int
    return k


class KeepAwake:
    """The PC is kept awake while this is open (a context manager; close() or leaving the block ends it)."""

    def __init__(self) -> None:
        self.active = False
        if WIN:
            try:
                self.active = bool(_kernel32().SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
            except (OSError, AttributeError):
                self.active = False

    def close(self) -> None:
        if self.active:
            try:
                _kernel32().SetThreadExecutionState(ES_CONTINUOUS)
            except (OSError, AttributeError):
                pass
            self.active = False

    def __enter__(self) -> "KeepAwake":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def keep_awake() -> KeepAwake:
    return KeepAwake()


def opt_out_of_throttling(handle: int | None = None) -> bool:
    """Tell Windows that the process (default: this one) must not be throttled; True when it accepted."""
    if not WIN:
        return False
    try:
        k = _kernel32()
        if not hasattr(k, "SetProcessInformation"):
            return False
        state = _ThrottlingState(THROTTLING_VERSION, EXECUTION_SPEED, 0)
        target = handle if handle is not None else k.GetCurrentProcess()
        return bool(k.SetProcessInformation(target, PROCESS_POWER_THROTTLING, ctypes.byref(state),
                                            ctypes.sizeof(state)))
    except (OSError, AttributeError, ValueError):
        return False


_installed = False
_original_init = subprocess.Popen.__init__


def _init_and_opt_out(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    handle = getattr(self, "_handle", None)
    if handle is not None:
        opt_out_of_throttling(int(handle))


def install() -> None:
    """Opt this process out now, and every process started with subprocess.Popen from now on. Safe to call twice."""
    global _installed
    if not WIN:
        return
    opt_out_of_throttling()
    if not _installed:
        subprocess.Popen.__init__ = _init_and_opt_out
        _installed = True


def uninstall() -> None:
    """Undo install()'s wrapper (the tests use it); the setting on this process stays."""
    global _installed
    if _installed:
        subprocess.Popen.__init__ = _original_init
        _installed = False


def wanted() -> bool:
    """Off with OOT2_NO_POWER_CONTROL=1, for a machine where the calls misbehave."""
    return os.environ.get("OOT2_NO_POWER_CONTROL", "") != "1"
