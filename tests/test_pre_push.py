# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for .githooks/pre-push and tools/check_push.py.

Every refusal has a green counterpart that pushes. The repositories are
throwaway ones in a temporary directory, the remote is a bare repository on disk,
and the identities are synthetic: nothing here reaches a network or this repository.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NAME = "RealZotto"
EMAIL = "185778380+RealZotto@users.noreply.github.com"
GOOD = (NAME, EMAIL)
OTHER = ("Someone Else", "1234567+someone-else@users.noreply.github.com")
NAME_ONLY = ("Someone Else", EMAIL)  # right address, wrong name
EMAIL_ONLY = (NAME, "1234567+someone-else@users.noreply.github.com")  # right name, wrong address

AVAILABLE = shutil.which("git") is not None and shutil.which("sh") is not None


def run_git(cwd: Path, *args: str, env: dict | None = None, check: bool = True):
    full_env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    full_env.update(env or {})
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=full_env)
    if check and done.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {done.stdout}{done.stderr}")
    return done


def identity_env(author: tuple[str, str], committer: tuple[str, str]) -> dict:
    return {
        "GIT_AUTHOR_NAME": author[0], "GIT_AUTHOR_EMAIL": author[1],
        "GIT_COMMITTER_NAME": committer[0], "GIT_COMMITTER_EMAIL": committer[1],
    }


@unittest.skipUnless(AVAILABLE, "git and sh are required to run the hook")
class PrePushHook(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pp-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.remote = self.tmp / "remote.git"
        self.repo = self.tmp / "clone"
        self.remote.mkdir()
        self.repo.mkdir()
        run_git(self.remote, "init", "-q", "--bare", "-b", "main")
        run_git(self.repo, "init", "-q", "-b", "main")
        (self.repo / ".githooks").mkdir()  # only pre-push: pre-commit would need the content guard here too
        shutil.copy(ROOT / ".githooks" / "pre-push", self.repo / ".githooks" / "pre-push")
        (self.repo / "tools").mkdir()
        shutil.copy(ROOT / "tools" / "check_push.py", self.repo / "tools" / "check_push.py")
        (self.repo / ".githooks" / "pre-push").chmod(0o755)  # git skips a hook without the exec bit
        run_git(self.repo, "config", "core.hooksPath", ".githooks")
        run_git(self.repo, "config", "commit.gpgsign", "false")
        run_git(self.repo, "config", "tag.gpgsign", "false")
        run_git(self.repo, "remote", "add", "origin", self.remote.as_posix())
        self.counter = 0

    def commit(self, author=GOOD, committer=GOOD) -> str:
        self.counter += 1
        (self.repo / f"f{self.counter}.txt").write_text(f"{self.counter}\n")
        run_git(self.repo, "add", "-A")
        run_git(self.repo, "commit", "-q", "-m", f"commit {self.counter}", env=identity_env(author, committer))
        return run_git(self.repo, "rev-parse", "HEAD").stdout.strip()

    def pin(self, sha: str):
        run_git(self.repo, "config", "oot2.rootCommit", sha)

    def push(self, *refspecs: str):
        return run_git(self.repo, "push", "origin", *refspecs, check=False)

    def remote_refs(self) -> str:
        return run_git(self.remote, "for-each-ref", "--format=%(refname)").stdout.strip()

    def assertRefused(self, done, *needles: str):
        text = done.stdout + done.stderr
        self.assertNotEqual(done.returncode, 0, text)
        self.assertIn("pre-push", text)
        for needle in needles:
            self.assertIn(needle, text)

    def test_clean_history_on_the_pinned_root_is_pushed(self):
        root = self.commit()
        tip = self.commit()
        self.pin(root)
        done = self.push("main")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(run_git(self.remote, "rev-parse", "main").stdout.strip(), tip)

    def test_another_author_is_refused(self):
        root = self.commit()
        self.commit(author=OTHER)
        self.pin(root)
        self.assertRefused(self.push("main"), "author is 'Someone Else <")
        self.assertEqual(self.remote_refs(), "")

    def test_another_committer_is_refused(self):
        root = self.commit()
        self.commit(committer=OTHER)
        self.pin(root)
        self.assertRefused(self.push("main"), "committer is 'Someone Else <")
        self.assertEqual(self.remote_refs(), "")

    def test_the_name_and_the_address_are_both_checked(self):
        root = self.commit()
        self.pin(root)
        self.commit(author=NAME_ONLY, committer=NAME_ONLY)
        self.assertRefused(self.push("main"), "not 'RealZotto <")
        run_git(self.repo, "reset", "-q", "--hard", root)
        self.commit(author=EMAIL_ONLY, committer=EMAIL_ONLY)
        self.assertRefused(self.push("main"), "not 'RealZotto <")
        self.assertEqual(self.remote_refs(), "")

    def test_a_foreign_commit_under_good_ones_is_refused(self):
        foreign = self.commit(author=OTHER, committer=OTHER)
        self.commit()
        self.commit()
        self.pin(foreign)  # even a pinned foreign root does not make its author acceptable
        self.assertRefused(self.push("main"), "author is 'Someone Else <")
        self.assertEqual(self.remote_refs(), "")

    def test_a_history_that_does_not_descend_from_the_pinned_root_is_refused(self):
        root = self.commit()
        self.pin(root)
        run_git(self.repo, "checkout", "-q", "--orphan", "elsewhere")  # the index keeps the hook files
        self.commit()
        self.assertRefused(self.push("elsewhere"), "not at the pinned root")
        self.assertEqual(self.remote_refs(), "")

    def test_a_merge_of_two_roots_is_refused(self):
        root = self.commit()
        self.pin(root)
        run_git(self.repo, "checkout", "-q", "--orphan", "elsewhere")
        self.commit()
        run_git(self.repo, "checkout", "-q", "main")
        run_git(self.repo, "merge", "-q", "--allow-unrelated-histories", "--no-edit", "elsewhere",
                env=identity_env(GOOD, GOOD))
        self.assertRefused(self.push("main"), "not at the pinned root")
        self.assertEqual(self.remote_refs(), "")

    def test_a_clone_without_a_pinned_root_refuses_everything(self):
        self.commit()
        self.assertRefused(self.push("main"), "no pinned root")
        self.assertEqual(self.remote_refs(), "")

    def test_an_unreadable_pin_is_a_failure_to_run_not_a_pass(self):
        self.commit()
        self.pin("not-a-commit-id")
        self.assertRefused(self.push("main"), "cannot check")
        self.assertEqual(self.remote_refs(), "")

    def test_deleting_a_branch_is_allowed(self):
        root = self.commit()
        self.pin(root)
        self.assertEqual(self.push("main:refs/heads/topic").returncode, 0)
        run_git(self.repo, "config", "--unset", "oot2.rootCommit")  # a deletion sends no commit, so it needs no pin
        done = self.push(":refs/heads/topic")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(self.remote_refs(), "")

    def test_a_malformed_pin_does_not_stop_a_deletion(self):
        root = self.commit()
        self.pin(root)
        self.assertEqual(self.push("main:refs/heads/topic").returncode, 0)
        self.pin("not-a-commit-id")  # a deletion sends no commit, so it never reads the pin
        done = self.push(":refs/heads/topic")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(self.remote_refs(), "")

    def test_the_printed_setup_command_pins_the_root_and_the_first_push_then_goes_through(self):
        self.commit()
        done = self.push("main")
        self.assertRefused(done, "no pinned root")
        hint = next(line for line in done.stderr.splitlines() if "run: " in line).split("run: ", 1)[1]
        run = subprocess.run(["sh", "-c", hint], cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        done = self.push("main")  # the remote is empty: nothing like origin/main exists to read a root from
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(self.remote_refs(), "refs/heads/main")

    def test_an_annotated_tag_needs_the_same_tagger(self):
        root = self.commit()
        self.pin(root)
        run_git(self.repo, "tag", "-a", "-m", "good", "v-good", env=identity_env(GOOD, GOOD))
        run_git(self.repo, "tag", "-a", "-m", "other", "v-other", env=identity_env(GOOD, OTHER))
        done = self.push("v-other")
        self.assertRefused(done, "tagger is 'Someone Else <")
        self.assertEqual(self.remote_refs(), "")
        done = self.push("v-good")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(self.remote_refs(), "refs/tags/v-good")

    def test_a_tag_that_is_not_on_a_commit_is_refused(self):
        root = self.commit()
        self.pin(root)
        run_git(self.repo, "tag", "-a", "-m", "tree", "v-tree", "HEAD^{tree}", env=identity_env(GOOD, GOOD))
        self.assertRefused(self.push("v-tree"), "does not point to a commit")
        self.assertEqual(self.remote_refs(), "")

    def test_the_whole_ancestry_is_read_not_only_what_the_remote_lacks(self):
        root = self.commit()
        self.pin(root)
        self.assertEqual(self.push("main").returncode, 0)
        bad = self.commit(author=OTHER)
        run_git(self.repo, "config", "--unset", "core.hooksPath")  # let the bad commit reach the remote for the setup
        self.assertEqual(self.push("main").returncode, 0)
        run_git(self.repo, "config", "core.hooksPath", ".githooks")
        self.commit()
        self.assertRefused(self.push("main"), bad[:12])


if __name__ == "__main__":
    unittest.main()
