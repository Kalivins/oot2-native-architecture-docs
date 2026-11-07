# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""What `oot2 prepare` does with Dear ImGui, the library the runtime's menu is built on (tools/prepare/program.py,
tools/prepare/check.py, tools/fetch_imgui.py; the menu's first rows).

Prepare takes the folder --imgui or OOT2_IMGUI names as it is, refuses one that is not exactly the pin by name, and
otherwise lays out fetched/imgui itself when it is absent (git and network access: tests give it the real pin's files
from the fetched folder instead of the network, and count the fetches). The link step, and no other, builds the four
core sources into the runtime library with the folder, and its key holds the pin's contents hash so that a new pin
links again. The check step says where the library comes from and stops before the extraction when it cannot be had.

The tests that need the pinned files use the fetched folder (OOT2_IMGUI, else fetched/imgui) and skip without it, unless
OOT2_REQUIRE_IMGUI is set, as continuous integration sets it. The toolchain and key tests need clang, Ninja and the
Vulkan SDK's headers, like tests/test_build_game.py.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_build_game  # noqa: E402,F401  (puts tools/ on the path)
import fetch_imgui  # noqa: E402
from prepare import check  # noqa: E402
from prepare import program as steps  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, PrepareError, Reporter  # noqa: E402
from test_build_game import toolchain_or_skip  # noqa: E402
from test_fetch_imgui import fetched_or_skip  # noqa: E402


class Said:
    """A reporter that keeps what is said."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def say(self, text: str) -> None:
        self.lines.append(text)

    def warning(self, text: str) -> None:
        self.lines.append("warning: " + text)

    def progress(self, *args, **kwargs) -> None:
        pass


def ctx_with(imgui: Path | None, **options) -> argparse.Namespace:
    return argparse.Namespace(options=argparse.Namespace(imgui=imgui, **options), reporter=Said())


class ImguiFolderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.real = fetched_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-prepare-imgui-", ignore_cleanup_errors=True)
        cls.top = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        os.environ.pop("OOT2_IMGUI", None)
        self.addCleanup(self.env.stop)

    def test_a_named_folder_is_used_as_it_is_and_no_fetch_is_made(self) -> None:
        with mock.patch.object(fetch_imgui, "pinned_source", side_effect=AssertionError("fetched")):
            ctx = ctx_with(self.real)
            self.assertEqual(steps.imgui_folder(ctx), self.real)
            os.environ["OOT2_IMGUI"] = str(self.real)
            self.assertEqual(steps.imgui_folder(ctx_with(None)), self.real)   # the variable, when no option is given
        self.assertEqual(ctx.reporter.lines, [])

    def test_a_named_folder_that_is_not_the_pin_is_refused_by_name_and_never_fetched_into(self) -> None:
        empty = self.top / "empty"
        empty.mkdir()
        with mock.patch.object(fetch_imgui, "pinned_source", side_effect=AssertionError("fetched")):
            with self.assertRaisesRegex(PrepareError, r"(?s)Dear ImGui.*--imgui"):
                steps.imgui_folder(ctx_with(empty))
        self.assertEqual(list(empty.iterdir()), [])

    def test_an_absent_default_folder_is_laid_out_once(self) -> None:
        dest = self.top / "default" / "imgui"
        calls = []

        def source() -> dict[str, bytes]:
            calls.append(1)
            return fetch_imgui.read_folder(self.real)

        with mock.patch.object(fetch_imgui, "DEFAULT_DEST", dest), mock.patch.object(fetch_imgui, "pinned_source",
                                                                                      source):
            ctx = ctx_with(None)
            self.assertEqual(steps.imgui_folder(ctx), dest)
            self.assertEqual(fetch_imgui.problems(dest), [])
            self.assertEqual(steps.imgui_folder(ctx), dest)
        self.assertEqual(len(calls), 1, "the second call fetched again")
        self.assertTrue(any("fetching Dear ImGui" in line for line in ctx.reporter.lines), ctx.reporter.lines)

    def test_a_fetch_that_fails_stops_the_run_by_name_with_the_way_out(self) -> None:
        dest = self.top / "failing" / "imgui"
        failing = mock.Mock(side_effect=fetch_imgui.FetchError("no network"))
        with mock.patch.object(fetch_imgui, "DEFAULT_DEST", dest), mock.patch.object(fetch_imgui, "pinned_source",
                                                                                      failing):
            with self.assertRaisesRegex(PrepareError, r"no network.*fetch_imgui\.py.*--imgui"):
                steps.imgui_folder(ctx_with(None))
        self.assertFalse(dest.exists())

    def test_the_check_step_says_where_the_library_comes_from(self) -> None:
        self.assertEqual(check.imgui_note(ctx_with(self.real)), f"Dear ImGui {fetch_imgui.VERSION} in {self.real}")
        absent = self.top / "nothing-here" / "imgui"
        with mock.patch.object(fetch_imgui, "DEFAULT_DEST", absent):
            with mock.patch.object(shutil, "which", return_value="git"):
                note = check.imgui_note(ctx_with(None))
                self.assertIn("is fetched into", note)
                self.assertIn("git and network", note)
            with mock.patch.object(shutil, "which", return_value=None):   # the fetch is due and there is no git
                with self.assertRaisesRegex(PrepareError, r"no git was found.*--imgui"):
                    check.imgui_note(ctx_with(None))
        with self.assertRaisesRegex(PrepareError, r"not a folder|is missing|holds"):
            check.imgui_note(ctx_with(self.top / "not-a-fetch"))

    def test_the_link_key_holds_the_pin(self) -> None:
        before = steps.imgui_key()
        self.assertIn(fetch_imgui.CONTENTS_SHA256, before)
        with mock.patch.object(fetch_imgui, "CONTENTS_SHA256", "f" * 64):
            self.assertNotEqual(steps.imgui_key(), before)


class ToolchainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        toolchain_or_skip()
        cls.real = fetched_or_skip()

    def test_the_link_gets_the_folder_and_the_other_steps_do_not(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-prepare-tool-", ignore_cleanup_errors=True) as tmp:
            exe = sys.executable
            sdk = Path(tmp) / "SDK"
            for rel_name in ("Include/vulkan/vulkan_core.h", "Include/Volk/volk.c", "Include/Volk/volk.h"):
                (sdk / rel_name).parent.mkdir(parents=True, exist_ok=True)
                (sdk / rel_name).write_text("", encoding="utf-8")
            ctx = ctx_with(self.real, cc=exe, vulkan_sdk=sdk, dav1d_dir=None)
            with mock.patch.dict(os.environ, {"OOT2_GLSLANG": exe, "OOT2_NINJA": exe}):
                os.environ.pop("DAV1D_DIR", None)
                self.assertEqual(steps.toolchain(ctx, imgui=True).imgui, self.real)
                self.assertIsNone(steps.toolchain(ctx).imgui)

    def test_the_key_of_the_link_follows_the_pin(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-prepare-key-", ignore_cleanup_errors=True) as tmp:
            dest = Path(tmp)
            ctx = Context(dest, argparse.Namespace(cc=None, jobs=1, imgui=self.real), Reporter(sys.stdout, sys.stderr),
                          state=statefile.load(dest))
            before = steps.Link().key(ctx)
            self.assertEqual(steps.Link().key(ctx), before)
            with mock.patch.object(fetch_imgui, "CONTENTS_SHA256", "f" * 64):
                self.assertNotEqual(steps.Link().key(ctx), before)


if __name__ == "__main__":
    unittest.main()
