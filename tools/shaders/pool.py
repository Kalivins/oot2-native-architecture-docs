# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""A pool of translator children: jobs in, one reply per job out.

Each child is `oot2-xlate --serve`. The line protocol (xlate/host/Program.cs):

  request, one stdin line:  {"job": {...}}
  reply, one stdout line:   ok <key> <microseconds> <module> <description>
                                  both base64, whole bytes
                            fail <key> <microseconds> <message>

    The translator writes no file: the two parts ride in the reply line. The
    pool hands them to the caller's on_result, and writes <key>.spv and
    <key>.res.json into a job's out folder only when the caller gives one
    (tools that read the files back). The list run_pool returns holds the
    small metadata of every job and never the module bytes, which would add
    up to the whole shader pack (about 6 GB for the game) in memory.
  diagnostics go to stderr, which the pool appends to one log file per child.

The microseconds are measured inside the child around the translation alone,
so they leave out process start, pipes and the pool. The pool also records the
wall time per job as the parent saw it.

A child that dies or stops answering for --timeout seconds is replaced, and its
job is reported as `crash` or `timeout`. A memory guard samples the children's
working sets (Windows). While their sum is above the limit, a child that holds
more than its share is replaced by a fresh one before its next job (an idle
child keeps the memory of its largest job, so waiting alone could stall the
pool), and the others hold new jobs back, so a pool of twelve stays inside the
budget the spike sets (about 4 GB).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
from base64 import b64decode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

GIB = 1 << 30
PROTOCOL = 2                    # the line protocol spoken here (xlate/host/Host.cs)


@dataclass
class Result:
    key: str
    status: str            # ok, fail, crash, timeout
    micros: int = 0        # translation time inside the child
    wall_ms: float = 0.0   # request to reply, as the parent saw it
    spv_bytes: int = 0     # the module's bytes
    module: bytes = b""    # the SPIR-V module of an ok reply: given to on_result, never kept in the returned list
    description: bytes = b""   # the description of an ok reply: likewise
    message: str = ""


@dataclass
class PoolStats:
    peak_children_bytes: int = 0
    min_available_bytes: int = 0
    restarts: int = 0
    recycled: int = 0          # children replaced to give memory back (see run_pool)
    held_back_s: float = 0.0
    samples: int = 0
    notes: list[str] = field(default_factory=list)


def base64_decode(text: str, key: str) -> bytes:
    """One base64 part of an ok reply; a malformed one fails the job by name."""
    try:
        return b64decode(text, validate=True)
    except Exception as exc:
        raise ValueError(f"the reply for {key} is not base64: {exc}") from exc


def decode_ok(key: str, rest: str) -> tuple[bytes, bytes]:
    """The module and the description of an ok reply's rest. An ok reply that does not carry both, each non-empty,
    fails the job by name: it never becomes a success with empty bytes."""
    parts = rest.split(" ")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"the ok reply for {key} does not carry a module and a description ({len(parts) if rest else 0} "
                         "part(s)): a translator of another protocol, or a reply cut short")
    return base64_decode(parts[0], key), base64_decode(parts[1], key)


def write_pair(out: Path, key: str, module: bytes, description: bytes) -> None:
    """<key>.spv and <key>.res.json in out, each written to a .tmp name and renamed, the module first, so a reader
    never meets a half-written file."""
    out.mkdir(parents=True, exist_ok=True)
    spv, res = out / f"{key}.spv", out / f"{key}.res.json"
    spv_tmp, res_tmp = spv.with_name(spv.name + ".tmp"), res.with_name(res.name + ".tmp")
    try:
        spv_tmp.write_bytes(module)
        res_tmp.write_bytes(description)
        os.replace(spv_tmp, spv)
        os.replace(res_tmp, res)
    except OSError:
        for tmp in (spv_tmp, res_tmp):
            tmp.unlink(missing_ok=True)
        raise


_protocols: dict[tuple[str, ...], int | None] = {}


def child_protocol(cmd: Sequence[str]) -> int | None:
    """The line protocol the translator gives with --version (proto=N), asked once for each command; None when it
    does not say (a stand-in that has no --version)."""
    key = tuple(cmd)
    if key not in _protocols:
        found = None
        try:
            r = subprocess.run([*cmd, "--version"], capture_output=True, text=True, errors="replace", timeout=60,
                               stdin=subprocess.DEVNULL)
            for word in r.stdout.split():
                if word.startswith("proto=") and word[6:].isdigit():
                    found = int(word[6:])
        except (OSError, subprocess.TimeoutExpired):
            pass
        _protocols[key] = found
    return _protocols[key]


def parse_reply(line: str) -> tuple[str, str, int, str]:
    """Split a reply line into (status, key, micros, rest).

    For an ok reply, rest holds "<module> <description>", both base64;
    no space appears inside either, so the first space splits them.
    """
    parts = line.rstrip("\r\n").split(" ", 3)
    if len(parts) < 3 or parts[0] not in ("ok", "fail"):
        raise ValueError(f"malformed reply: {line[:200]!r}")
    rest = parts[3] if len(parts) > 3 else ""
    return parts[0], parts[1], int(parts[2]), rest


# --- memory sampling (Windows; elsewhere the guard is inactive) ---------------

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    class _MSX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    _psapi = ctypes.WinDLL("psapi")
    _k32 = ctypes.WinDLL("kernel32")

    def working_set(proc: subprocess.Popen) -> int:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        handle = wintypes.HANDLE(int(proc._handle))  # type: ignore[attr-defined]
        ok = _psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
        return int(pmc.WorkingSetSize) if ok else 0

    def available_ram() -> int:
        msx = _MSX()
        msx.dwLength = ctypes.sizeof(msx)
        return int(msx.ullAvailPhys) if _k32.GlobalMemoryStatusEx(ctypes.byref(msx)) else 0
else:
    def working_set(proc: subprocess.Popen) -> int:
        return 0

    def available_ram() -> int:
        return 0


class _Child:
    def __init__(self, cmd: list[str], log: Path):
        self.log = open(log, "ab")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, encoding="utf-8", bufsize=1)
        self.replies: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.replies.put(line)
        self.replies.put(None)

    def ask(self, request: str, timeout: float) -> str | None:
        """Send one request; return the reply line, None on death, '' on timeout."""
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(request + "\n")
            self.proc.stdin.flush()
        except OSError:
            return None
        try:
            return self.replies.get(timeout=timeout)
        except queue.Empty:
            return ""

    def close(self, kill: bool = False) -> None:
        try:
            if kill:
                self.proc.kill()
            elif self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            self.proc.wait()
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        self.log.close()


def run_pool(command: Sequence[str], jobs: Iterable[tuple[dict, Path]], workers: int, log_dir: Path,
             timeout: float = 300.0, memory_limit: int = 4 * GIB,
             on_result: Callable[[Result], None] | None = None) -> tuple[list[Result], PoolStats]:
    """Translate every (job, out folder) pair on `workers` children.

    `command` starts one translator, without its "--serve" argument: the
    executable alone for a NativeAOT build, or an interpreter and a script.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    base_cmd = [*map(str, command)]
    cmd = [*base_cmd, "--serve"]
    feed: Iterator[tuple[dict, Path]] = iter(jobs)
    first = next(feed, None)
    if first is not None:
        proto = child_protocol(base_cmd) if first[1] is not None else None
        if proto not in (None, PROTOCOL):
            raise ValueError(f"the translator {base_cmd[-1]} speaks protocol {proto}, which writes its own files; an "
                             f"out folder needs protocol {PROTOCOL}, whose replies carry the module and the description")
        feed = itertools.chain([first], feed)
    feed_lock = threading.Lock()
    results: list[Result] = []
    res_lock = threading.Lock()
    stats = PoolStats(min_available_bytes=available_ram())
    children: list[_Child | None] = [None] * workers
    over_limit = threading.Event()
    done = threading.Event()

    def monitor() -> None:
        while not done.is_set():
            total = sum(working_set(c.proc) for c in children if c is not None)
            avail = available_ram()
            with res_lock:
                stats.samples += 1
                stats.peak_children_bytes = max(stats.peak_children_bytes, total)
                if avail:
                    stats.min_available_bytes = min(stats.min_available_bytes or avail, avail)
            over_limit.set() if total > memory_limit else over_limit.clear()
            done.wait(0.5)

    def record(r: Result) -> None:
        # The caller's sink gets the bytes; the returned list keeps only the metadata.
        if on_result:
            on_result(r)
        kept = dataclasses.replace(r, module=b"", description=b"") if (r.module or r.description) else r
        with res_lock:
            results.append(kept)

    def worker(slot: int) -> None:
        child = _Child(cmd, log_dir / f"child{slot:02d}.log")
        children[slot] = child
        served = 0                      # jobs the current child has answered
        while True:
            # Over the limit, a child that has served jobs and holds more than its share
            # is replaced: an idle child keeps the memory its largest job needed, so
            # waiting for the sum to fall could wait forever. A fresh child is as small
            # as a child gets and never waits.
            while over_limit.is_set() and served:
                if working_set(child.proc) > memory_limit // workers:
                    child.close()
                    child = _Child(cmd, log_dir / f"child{slot:02d}.log")
                    children[slot] = child
                    served = 0
                    with res_lock:
                        stats.recycled += 1
                    break
                t0 = time.perf_counter()
                time.sleep(0.2)
                with res_lock:
                    stats.held_back_s += time.perf_counter() - t0
            with feed_lock:
                item = next(feed, None)
            if item is None:
                break
            job, out = item
            key = str(job.get("key", "?"))
            t0 = time.perf_counter()
            line = child.ask(json.dumps({"job": job}, separators=(",", ":")), timeout)
            wall = (time.perf_counter() - t0) * 1000.0
            cut = bool(line) and not line.endswith("\n")    # the child ended in the middle of its reply
            if line is None or line == "" or cut:
                record(Result(key, "timeout" if line == "" else "crash", wall_ms=wall,
                              message=f"child {slot} {'did not answer' if line == '' else 'exited'}"
                                      f"{' in the middle of its reply' if cut else ''}"))
                child.close(kill=True)
                child = _Child(cmd, log_dir / f"child{slot:02d}.log")
                children[slot] = child
                served = 0
                with res_lock:
                    stats.restarts += 1
                continue
            served += 1
            try:
                status, rkey, micros, rest = parse_reply(line)
            except ValueError as exc:
                record(Result(key, "fail", wall_ms=wall, message=str(exc)))
                continue
            if rkey == "-" and status == "fail":       # the translator could not read the key
                record(Result(key, "fail", micros, wall, message=rest))
                continue
            if rkey != key:
                record(Result(key, "fail", micros, wall, message=f"reply for {rkey}"))
                continue
            if status == "ok":
                try:
                    module, description = decode_ok(key, rest)
                except ValueError as exc:
                    record(Result(key, "fail", micros, wall, message=str(exc)))
                    continue
                if out is not None:
                    try:
                        write_pair(out, key, module, description)
                    except OSError as exc:
                        record(Result(key, "fail", micros, wall,
                                      message=f"the translator answered, but its output cannot be written to {out}: {exc}"))
                        continue
                record(Result(key, "ok", micros, wall, spv_bytes=len(module),
                              module=module, description=description))
            else:
                record(Result(key, "fail", micros, wall, message=rest))
        child.close()
        children[slot] = None

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    done.set()
    mon.join()
    return results, stats


if __name__ == "__main__":
    print("tools/shaders/pool.py is a module; tools/shaders/corpus.py drives it.", file=sys.stderr)
    sys.exit(2)
