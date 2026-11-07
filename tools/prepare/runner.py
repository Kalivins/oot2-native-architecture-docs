# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Which steps of `oot2 prepare` run, in what order, and the record kept of them (architecture D5).

Each step has a key: a fingerprint of everything its output depends on. That is
the step's own inputs (Step.key: files, options, the code of the tool doing the
work) and the keys of the steps it reads from (Step.needs). The runner folds the
upstream keys in itself, so a step cannot forget one. A step is skipped when the
state file records it with the same key and its output is still in place
(Step.current, a cheap check). Otherwise it runs. A step marked `always` (the
check) runs every time and is not recorded.

That one rule covers the three ways prepare is run:

- First run: nothing is recorded, every step runs.
- Resume: the steps that finished are recorded with their keys and are skipped,
  so the run starts at the first step that did not finish. A step's record is
  removed before it runs and written when it ends, so a step stopped halfway is
  never taken for a finished one.
- Update: a release whose tools changed changes the key of each step whose code
  changed, so that step and every step that needs it, directly or through
  another, run again; nothing upstream of it does, nor a step on another branch.

With --verify every kept output also gets its full check (Step.verify, which may
hash gigabytes). A step that fails it runs again; its dependents do not, because
their inputs did not change.

Keys are computed in order as the run reaches each step, so a step may compute
its key from facts an earlier step found (the check finds the dump's hash, and
the extraction's key is made from it).

Adapted from TotK Recomp installer/runner.py and the fingerprint helpers of
installer/steps.py at 0ae47a5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh,
licenses/totk-MIT.txt). Changed: upstream keys are folded in by the runner from
Step.needs instead of by each step; keys are computed as the run reaches each
step instead of all at the start; a step's record is dropped before it runs; no
transient steps, release field or update mode; plain text instead of JSON events.
"""

from __future__ import annotations

import datetime
import hashlib
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

from . import state as statefile

# Why a step runs, or does not, in the words the run prints.
REASONS = {
    "always": "runs every time",
    "new": "not done yet",
    "changed": "its inputs changed",
    "missing": "its output is missing",
    "damaged": "its output failed the full check",
    "current": "done, inputs unchanged",
}


class PrepareError(Exception):
    """A failure the player can act on. The message says what is wrong and what to do about it."""


class Step:
    id = ""
    needs: tuple[str, ...] = ()   # ids of earlier steps whose output this step reads
    always = False

    def key(self, ctx: "Context") -> str:
        """Fingerprint of this step's own inputs. The runner adds the keys of `needs`."""
        return ""

    def current(self, ctx: "Context", record: dict) -> bool:
        """True when the recorded output is still in place (cheap checks only)."""
        return True

    def verify(self, ctx: "Context", record: dict) -> bool:
        """True when the recorded output passes the full check (hashes). Only with --verify."""
        return True

    def run(self, ctx: "Context") -> dict | None:
        """Do the work. Returns fields to record with the step, such as "bytes" written."""
        return {}


class Reporter:
    """Plain lines with the time since the start; failures on stderr."""

    def __init__(self, out=None, err=None):
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self.t0 = time.monotonic()
        self.last: dict[str, float] = {}

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def say(self, text: str) -> None:
        print(f"[{self.elapsed():7.1f} s] {text}", file=self.out, flush=True)

    def progress(self, tag: str, text: str, every: float = 5.0, final: bool = False) -> None:
        """A progress line, at most one per `every` seconds for a tag, plus the first and the final one."""
        now = time.monotonic()
        last = self.last.get(tag)
        if last is not None and not final and now - last < every:
            return
        self.last[tag] = now
        self.say(text)

    def warning(self, text: str) -> None:
        self.say("warning: " + text)

    def error(self, text: str) -> None:
        print(f"oot2 prepare: {text}", file=self.err, flush=True)


class Context:
    def __init__(self, dest: Path, options, reporter: Reporter, state: dict | None = None, verify: bool = False):
        self.dest = Path(dest)
        self.options = options
        self.reporter = reporter
        self.fresh = state is None              # no state file: a first run in this folder
        self.state = state if state is not None else statefile.empty()
        self.verify = verify
        self.keys: dict[str, str] = {}
        self.reasons: dict[str, str] = {}
        self.facts: dict = {}                   # what a step found for the keys of later steps

    def record(self, step_id: str) -> dict | None:
        return self.state["steps"].get(step_id)

    def save(self) -> None:
        self.dest.mkdir(parents=True, exist_ok=True)
        statefile.save(self.dest, self.state)

    def decide(self, step: Step) -> str:
        """Compute the step's key and return why it runs, or "current" when it is kept."""
        own = step.key(self)
        self.keys[step.id] = fingerprint(step.id, own, *(f"{n}={self.keys[n]}" for n in step.needs))
        rec = self.record(step.id)
        if step.always:
            reason = "always"
        elif rec is None:
            reason = "new"
        elif rec.get("key") != self.keys[step.id]:
            reason = "changed"
        elif not step.current(self, rec):
            reason = "missing"
        elif self.verify and not step.verify(self, rec):
            self.reporter.warning(f"the output of step {step.id} failed its full check; it runs again")
            reason = "damaged"
        else:
            reason = "current"
        self.reasons[step.id] = reason
        return reason


def check_order(steps: list[Step]) -> None:
    """Step ids are unique and every step needs only steps before it. A programming error otherwise."""
    seen: set[str] = set()
    for s in steps:
        if not s.id or s.id in seen:
            raise ValueError(f"step id {s.id!r} is empty or used twice")
        for n in s.needs:
            if n not in seen:
                raise ValueError(f"step {s.id} needs {n}, which is not an earlier step")
        seen.add(s.id)


def run_steps(steps: list[Step], ctx: Context, only: Iterable[str] | None = None) -> str:
    """Run the steps in order. With `only`, run just those and say what the others would do.

    Returns "ok", "failed" or "cancelled"; the printed lines say why.
    """
    check_order(steps)
    only = set(only) if only is not None else None
    rep = ctx.reporter
    current = None
    try:
        for s in steps:
            current = s.id
            reason = ctx.decide(s)
            if reason == "current":
                rep.say(f"{s.id}: kept ({REASONS[reason]})")
                continue
            if only is not None and s.id not in only:
                rep.say(f"{s.id}: would run ({REASONS[reason]})")
                continue
            rep.say(f"{s.id}: started ({REASONS[reason]})")
            if not s.always and ctx.state["steps"].pop(s.id, None) is not None:
                ctx.save()
            t0 = rep.elapsed()
            fields = s.run(ctx) or {}
            seconds = round(rep.elapsed() - t0, 1)
            if not s.always:
                ctx.state["steps"][s.id] = {"key": ctx.keys[s.id], "finished": now_utc(), "seconds": seconds,
                                            **fields}
            ctx.save()
            rep.say(f"{s.id}: done in {seconds:.1f} s")
        current = None
        return "ok"
    except PrepareError as e:
        rep.error(f"{e}" if current is None else f"step {current}: {e}")
        name_failure(ctx, current)
        return "failed"
    except KeyboardInterrupt:
        rep.say(f"cancelled during step {current}; the steps finished before it are kept")
        return "cancelled"
    except Exception as e:
        rep.error(f"internal error in step {current}: {type(e).__name__}: {e}")
        traceback.print_exc(file=rep.err)
        name_failure(ctx, current)
        return "failed"


def name_failure(ctx: Context, step_id: str | None) -> None:
    """After a failure, say which step it was and where to read why: the lines above that start with the step's id,
    which the run printed as it went, and the step's own log under temp/logs when it keeps one."""
    if step_id is None:
        return
    logs = ctx.dest / "temp" / "logs"
    ctx.reporter.say(f"the step that failed is {step_id}. Read the lines above that start with \"{step_id}:\""
                       + (f" and the logs in {logs}" if logs.is_dir() else "") +
                       "; running oot2 prepare again resumes at this step.")


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path, progress=None) -> str:
    """SHA-256 hex of a file; progress(done, total) after each block."""
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb", buffering=0) as f:
        buf = bytearray(8 << 20)
        view = memoryview(buf)
        while n := f.readinto(buf):
            h.update(view[:n])
            done += n
            if progress:
                progress(done, total)
    return h.hexdigest()


def fingerprint(*parts) -> str:
    """SHA-256 over strings and the contents of files and folders (a missing path counts as missing)."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, Path):
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and "__pycache__" not in f.parts:
                        h.update(f.relative_to(p).as_posix().encode() + b"\0" + bytes.fromhex(file_sha256(f)))
            elif p.is_file():
                h.update(p.name.encode() + b"\0" + bytes.fromhex(file_sha256(p)))
            else:
                h.update(b"missing\0" + p.name.encode())
        else:
            h.update(str(p).encode())
        h.update(b"\1")
    return h.hexdigest()
