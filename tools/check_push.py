#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Pre-push guard: only commits made under the project identity, on top of the root, may leave this clone.

git runs .githooks/pre-push as `pre-push <remote> <url>` and writes one line per
pushed ref on standard input:

    <local ref> <local sha> <remote ref> <remote sha>

For every ref that is not a deletion, this refuses the push when

* a commit the remote lacks has an author or a committer other than IDENTITY
  (name and e-mail, both exact), or an annotated tag on the way has a tagger
  other than IDENTITY;
* the whole ancestry does not have exactly one parentless commit, or that commit
  is not the root pinned in this clone's `git config oot2.rootCommit`;
* the clone has no pinned root: an unpinned clone refuses everything, so a
  forgotten setup step cannot turn the guard off.

The identity rule reads only the commits the remote lacks, so that a contributor's
pull request, merged on GitHub with "Squash and merge", stays authored by the
contributor and does not make every later push refuse. A commit is held by the
remote when it is reachable from the tip the remote reported for the pushed ref
(the last field on standard input) or from a remote-tracking ref of the remote
being pushed to (refs/remotes/<remote>/*, as of the last fetch). A push to a bare
URL has no remote name, so there only the reported tip counts. The root rule
still reads the whole ancestry: what the remote holds does not change where the
history starts.

Setup, once per clone, after its root commit exists:

    git config core.hooksPath .githooks
    git config oot2.rootCommit "$(git rev-list --max-parents=0 HEAD)"

Exit codes: 0 allowed, 1 refused, 2 the check could not run (also a refusal for
git, and not a pass).
"""

from __future__ import annotations

import re
import subprocess
import sys

NAME = "RealZotto"
EMAIL = "185778380+RealZotto@users.noreply.github.com"
IDENTITY = f"{NAME} <{EMAIL}>"
ZERO = "0" * 40
FULL_SHA = re.compile(r"[0-9a-f]{40}")
REMOTE_NAME = re.compile(r"[A-Za-z0-9._-]+")
FIELD = "\x1f"


class CheckError(Exception):
    """The check could not run: exit 2."""


def git(*args: str) -> str:
    # --no-replace-objects: a replace ref must not rewrite the history this reads.
    done = subprocess.run(["git", "--no-replace-objects", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise CheckError(f"git {' '.join(args)}: {done.stderr.strip() or 'exit ' + str(done.returncode)}")
    return done.stdout


def commit_exists(sha: str) -> bool:
    done = subprocess.run(["git", "--no-replace-objects", "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True)
    return done.returncode == 0


def held_by_remote(remote: str | None, remote_sha: str) -> list[str]:
    """Revision arguments for what the remote already holds: the tip it reported for this ref, and the
    remote-tracking refs of the remote being pushed to. A remote name with a glob character is not used."""
    held: list[str] = []
    if remote_sha != ZERO and FULL_SHA.fullmatch(remote_sha) and commit_exists(remote_sha):
        held.append(remote_sha)
    if remote and REMOTE_NAME.fullmatch(remote) and remote in git("remote").split():
        held.append(f"--remotes={remote}")
    return held


def pinned_root() -> str | None:
    done = subprocess.run(["git", "config", "--get", "oot2.rootCommit"], capture_output=True, text=True)
    value = done.stdout.strip()
    if done.returncode != 0 or not value:
        return None
    if not FULL_SHA.fullmatch(value):
        raise CheckError(f"oot2.rootCommit is not a full commit id: {value!r}")
    return value


def tag_problems(ref: str, sha: str) -> tuple[list[str], str]:
    """Check the tagger of every tag object down to the commit; returns the problems and the commit id."""
    problems: list[str] = []
    seen = 0
    while git("cat-file", "-t", sha).strip() == "tag":
        seen += 1
        if seen > 16:
            raise CheckError(f"{ref}: tag chain deeper than 16")
        body = git("cat-file", "tag", sha)
        header = body.split("\n\n", 1)[0].splitlines()
        target = next((line[7:] for line in header if line.startswith("object ")), "")
        tagger = next((line for line in header if line.startswith("tagger ")), None)
        if tagger is None or not re.fullmatch(rf"tagger {re.escape(IDENTITY)} \d+ [+-]\d{{4}}", tagger):
            shown = "no tagger" if tagger is None else tagger[7:]
            problems.append(f"{ref}: tag {sha[:12]} tagger is '{shown}', not '{IDENTITY}'")
        if not FULL_SHA.fullmatch(target):
            raise CheckError(f"{ref}: tag {sha[:12]} has no readable target")
        sha = target
    if git("cat-file", "-t", sha).strip() != "commit":
        problems.append(f"{ref}: does not point to a commit")
        return problems, ""
    return problems, sha


def commit_problems(ref: str, commit: str, held: list[str]) -> list[str]:
    """The identity rule, over the commits reachable from `commit` that the remote does not hold."""
    problems: list[str] = []
    revisions = [commit, "--not", *held] if held else [commit]
    log = git("log", "-z", f"--format=%H{FIELD}%an{FIELD}%ae{FIELD}%cn{FIELD}%ce", *revisions)
    for record in log.split("\0"):
        if not record:
            continue
        sha, an, ae, cn, ce = record.split(FIELD)
        for role, name, mail in (("author", an, ae), ("committer", cn, ce)):
            if (name, mail) != (NAME, EMAIL):
                problems.append(f"{ref}: commit {sha[:12]} {role} is '{name} <{mail}>', not '{IDENTITY}'")
    return problems


def root_problems(ref: str, commit: str, pinned: str | None) -> list[str]:
    roots = git("rev-list", "--max-parents=0", commit).split()
    if pinned is None:
        return [f"{ref}: this clone has no pinned root; run: "
                f'git config oot2.rootCommit "$(git rev-list --max-parents=0 HEAD)"']
    if roots != [pinned]:
        shown = ", ".join(r[:12] for r in roots) or "none"
        return [f"{ref}: the history starts at {shown}, not at the pinned root {pinned[:12]}"]
    return []


def check_ref(ref: str, sha: str, pinned: str | None, remote: str | None, remote_sha: str) -> list[str]:
    problems, commit = tag_problems(ref, sha)
    if commit:
        problems += commit_problems(ref, commit, held_by_remote(remote, remote_sha))
        problems += root_problems(ref, commit, pinned)
    return problems


def main(argv: list[str], stdin) -> int:
    try:
        pinned: str | None = None
        pin_read = False
        problems: list[str] = []
        for line in stdin:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise CheckError(f"unexpected line from git: {line.strip()!r}")
            local_ref, local_sha, remote_ref, remote_sha = parts
            if local_sha == ZERO:  # a deletion sends no commit, so it never needs the pin
                continue
            if not pin_read:
                pinned, pin_read = pinned_root(), True
            problems += check_ref(remote_ref or local_ref, local_sha, pinned, argv[0] if argv else None, remote_sha)
    except CheckError as err:
        print(f"pre-push: cannot check: {err}", file=sys.stderr)
        print("pre-push: the push is refused rather than unchecked.", file=sys.stderr)
        return 2
    if problems:
        for text in problems:
            print(f"pre-push: {text}", file=sys.stderr)
        print(f"pre-push: refused. Every commit the remote lacks must be authored and committed by {IDENTITY}, "
              "and the history must start at the pinned root. Fix the author of a local commit with "
              "'git commit --amend --reset-author --no-edit', or carry a commit over with "
              "'git cherry-pick --no-commit <sha>' then 'git commit -C <sha> --reset-author'; "
              "never push around this hook.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], sys.stdin))
