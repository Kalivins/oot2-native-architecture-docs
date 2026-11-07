# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of tools/fetch_imgui.py and of how Dear ImGui enters the build (research 107).

Dear ImGui is fetched at a pinned version and never committed. Without a network the tests use a pin over synthetic
files (the real table is only checked for its shape) and a local git repository for the part that reads blobs from
git's object store, so a line-ending setting on the machine cannot change a hash. Checked: a file with one changed byte
is refused by name and nothing is written; a missing, an added and a changed file are; the contents hash binds the
set; a folder laid out by the script is recognised, and a stale, tampered or foreign one is not; the blobs come out
exactly as git stores them under core.autocrlf=true; --print-pins prints a table that round-trips; the notice rows and
the licence texts exist for the pin; the guard and .gitignore agree on the folder, with the folder present and absent,
and a fetched file that gets committed turns the guard red; the Ninja plan compiles the four core sources with their own
rule and flags, beside the runtime's, and only when a folder is given; find_imgui refuses a folder that is not the pin.

A folder fetched for real (tools/fetch_imgui.py, then OOT2_IMGUI or fetched/imgui) is also read here: its files must
equal the pin and the notices must say what its files say. Without one those tests skip, unless OOT2_REQUIRE_IMGUI is
set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import build_game  # noqa: E402
import fetch_imgui as fi  # noqa: E402
import test_build_game as tbg  # noqa: E402
import test_check_content as tcc  # noqa: E402

GIT = shutil.which("git") is not None
HEX64 = re.compile(r"[0-9a-f]{64}")
# The files a build and a notice depend on: the licence, the core, the stb headers and the two back ends.
REQUIRED = {"LICENSE.txt", "imconfig.h", "imgui.h", "imgui_internal.h", "imstb_rectpack.h", "imstb_textedit.h",
            "imstb_truetype.h", "backends/imgui_impl_vulkan.cpp", "backends/imgui_impl_vulkan.h",
            "backends/imgui_impl_sdl3.cpp", "backends/imgui_impl_sdl3.h"}


def synthetic() -> dict[str, bytes]:
    """One distinct synthetic file for each path of the real pin (no byte of Dear ImGui)."""
    return {path: f"// synthetic {path}\n".encode() * 3 + bytes([i]) for i, path in enumerate(fi.FILES)}


def flip(data: bytes) -> bytes:
    return bytes([data[len(data) // 2] ^ 1]).join((data[:len(data) // 2], data[len(data) // 2 + 1:]))


class PinShape(unittest.TestCase):
    def test_the_pin_has_a_version_a_tag_a_commit_and_hashes(self) -> None:
        self.assertRegex(fi.VERSION, r"^\d+\.\d+\.\d+[a-z]?$")
        self.assertEqual(fi.TAG, "v" + fi.VERSION)
        self.assertRegex(fi.COMMIT, r"^[0-9a-f]{40}$")
        self.assertRegex(fi.CONTENTS_SHA256, HEX64.pattern)
        self.assertEqual(fi.pins(), {"version": fi.VERSION, "tag": fi.TAG, "commit": fi.COMMIT,
                                     "contents_sha256": fi.CONTENTS_SHA256})
        self.assertTrue(REQUIRED <= set(fi.FILES), sorted(REQUIRED - set(fi.FILES)))
        self.assertTrue(set(fi.CORE_SOURCES) <= set(fi.FILES))
        for path, digest in fi.FILES.items():
            with self.subTest(path=path):
                self.assertRegex(digest, HEX64.pattern)
        self.assertEqual(len(set(fi.FILES.values())), len(fi.FILES), "two files with one hash")

    def test_the_table_is_what_sha256sum_reads(self) -> None:
        lines = fi.SHA256SUMS.splitlines()
        self.assertEqual(len(lines), len(fi.FILES))
        for line in lines:
            self.assertRegex(line, r"^[0-9a-f]{64}  \S+$")


class Pinned(unittest.TestCase):
    """A pin over synthetic files, laid out in a temporary folder."""

    def setUp(self) -> None:
        self.files = synthetic()
        for name, value in (("FILES", {p: fi.sha256(d) for p, d in self.files.items()}),
                            ("CONTENTS_SHA256", fi.contents_sha256(self.files))):
            patcher = mock.patch.object(fi, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tmp = Path(tempfile.mkdtemp(prefix="oot2-imgui-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.dest = self.tmp / "fetched" / "imgui"

    def lay_out(self) -> None:
        fi.fetch(self.dest, lambda: dict(self.files))


class FetchTest(Pinned):
    def test_a_fetch_lays_out_every_file_byte_for_byte_and_records_the_pin(self) -> None:
        lines = fi.fetch(self.dest, lambda: dict(self.files))
        self.assertIn(fi.TAG, lines[0])
        for path, data in self.files.items():
            self.assertEqual((self.dest / path).read_bytes(), data)
        self.assertEqual(fi.problems(self.dest), [])
        self.assertTrue((self.dest / fi.RECORD).is_file())

    def test_a_file_with_one_changed_byte_is_refused_by_name_and_nothing_is_written(self) -> None:
        for path in list(self.files)[::4]:
            with self.subTest(path=path):
                bad = dict(self.files)
                bad[path] = flip(bad[path])
                with self.assertRaisesRegex(fi.FetchError, rf"{re.escape(path)}: SHA-256 [0-9a-f]{{64}}, expected"):
                    fi.fetch(self.dest, lambda: bad)
                self.assertFalse(self.dest.exists())
        self.lay_out()   # the green counterpart: the same call with the files as pinned is accepted

    def test_a_missing_and_an_added_file_are_refused_by_name(self) -> None:
        gone = dict(self.files)
        del gone["imgui.cpp"]
        with self.assertRaisesRegex(fi.FetchError, "imgui.cpp is missing"):
            fi.fetch(self.dest, lambda: gone)
        extra = dict(self.files, **{"imgui_demo.cpp": b"// not pinned\n"})
        with self.assertRaisesRegex(fi.FetchError, "imgui_demo.cpp is not one of the pinned files"):
            fi.fetch(self.dest, lambda: extra)
        self.assertFalse(self.dest.exists())

    def test_the_contents_hash_binds_the_set(self) -> None:
        with mock.patch.object(fi, "CONTENTS_SHA256", "0" * 64):
            with self.assertRaisesRegex(fi.FetchError, "as a whole"):
                fi.fetch(self.dest, lambda: dict(self.files))
        self.assertFalse(self.dest.exists())

    def test_a_folder_that_is_not_the_scripts_own_is_never_replaced(self) -> None:
        self.dest.mkdir(parents=True)
        (self.dest / "mine.txt").write_text("not yours to delete\n", encoding="utf-8")
        with self.assertRaisesRegex(fi.FetchError, "not a folder this script laid out"):
            fi.fetch(self.dest, lambda: dict(self.files))
        self.assertTrue((self.dest / "mine.txt").is_file())
        (self.dest.parent / "file").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(fi.FetchError, "not a folder this script laid out"):
            fi.fetch(self.dest.parent / "file", lambda: dict(self.files))

    def test_its_own_folder_is_replaced_and_an_empty_one_is_filled(self) -> None:
        self.dest.mkdir(parents=True)
        self.lay_out()
        (self.dest / "stray.txt").write_text("left over\n", encoding="utf-8")
        self.lay_out()
        self.assertFalse((self.dest / "stray.txt").exists())
        self.assertEqual(fi.problems(self.dest), [])

    def test_the_command_line_says_when_nothing_has_to_be_done(self) -> None:
        self.lay_out()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(fi, "pinned_source", side_effect=AssertionError):
            code = fi.main(["--dest", str(self.dest)])
        self.assertEqual(code, 0)
        self.assertIn("already holds Dear ImGui", out.getvalue())
        broken = dict(self.files)
        broken["imgui.h"] = flip(broken["imgui.h"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err), mock.patch.object(fi, "pinned_source", lambda: broken):
            self.assertEqual(fi.main(["--dest", str(self.tmp / "other")]), 1)
        self.assertIn("imgui.h: SHA-256", err.getvalue())


class ProblemsTest(Pinned):
    def test_a_laid_out_folder_has_no_problem(self) -> None:
        self.lay_out()
        self.assertEqual(fi.problems(self.dest), [])

    def test_a_tampered_stale_added_or_unrecorded_folder_does(self) -> None:
        self.assertIn("is not a folder", fi.problems(self.tmp / "nothing")[0])
        self.lay_out()
        victim = self.dest / "imgui_draw.cpp"
        victim.write_bytes(flip(victim.read_bytes()))
        self.assertIn("imgui_draw.cpp: SHA-256", fi.problems(self.dest)[0])
        self.lay_out()
        (self.dest / "extra.cpp").write_text("// added\n", encoding="utf-8")
        self.assertIn("extra.cpp is not one of the pinned files", fi.problems(self.dest)[0])
        self.lay_out()
        (self.dest / "imgui.h").unlink()
        self.assertIn("imgui.h is missing", fi.problems(self.dest)[0])
        self.lay_out()
        (self.dest / fi.RECORD).write_text('{"version": "0.0.1", "tag": "v0.0.1"}', encoding="utf-8")
        self.assertRegex(fi.problems(self.dest)[0], r"holds Dear ImGui v0\.0\.1, this tree pins " + fi.TAG)
        (self.dest / fi.RECORD).unlink()
        self.assertIn("fetch with tools/fetch_imgui.py", fi.problems(self.dest)[0])


@unittest.skipUnless(GIT, "git is required")
class BlobsFromGit(unittest.TestCase):
    """fetch_blobs against a local repository: blobs as git stores them, whatever core.autocrlf says here."""

    LF = b"line one\nline two\n"
    MIXED = b"\r\nkeeps\r\nits\nendings\r\n"

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="oot2-imgui-git-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = self.tmp / "src"
        self.src.mkdir()
        for args in (("init", "-q", "-b", "main"), ("config", "user.name", "Tester"),
                     ("config", "user.email", "tester@users.noreply.github.com"), ("config", "core.autocrlf", "false"),
                     ("config", "commit.gpgsign", "false"), ("config", "uploadpack.allowAnySHA1InWant", "true")):
            tcc.git_in(self.src, *args)
        (self.src / "sub").mkdir()
        (self.src / "a.txt").write_bytes(self.LF)
        (self.src / "sub" / "b.h").write_bytes(self.MIXED)
        tcc.git_in(self.src, "add", "-A")
        tcc.git_in(self.src, "commit", "-q", "-m", "synthetic")
        self.commit = tcc.git_in(self.src, "rev-parse", "HEAD")
        # From here on every git the script runs is told to rewrite line endings on a checkout, as a Windows setup does.
        env = mock.patch.dict(os.environ, {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.autocrlf",
                                           "GIT_CONFIG_VALUE_0": "true"})
        env.start()
        self.addCleanup(env.stop)
        url = mock.patch.object(fi, "URL", self.src.as_uri())
        url.start()
        self.addCleanup(url.stop)

    def test_blobs_come_out_as_git_stores_them(self) -> None:
        work = self.tmp / "work"
        work.mkdir()
        got = fi.fetch_blobs(self.commit, ["a.txt", "sub/b.h"], work)
        self.assertEqual(got, {"a.txt": self.LF, "sub/b.h": self.MIXED})
        # the setting does rewrite a checkout here, so the test can tell the two ways apart
        check = self.tmp / "checkout"
        tcc.git_in(self.tmp, "clone", "-q", str(self.src), str(check))
        self.assertEqual((check / "a.txt").read_bytes(), self.LF.replace(b"\n", b"\r\n"))

    def test_a_wrong_commit_and_a_missing_path_stop_by_name(self) -> None:
        work = self.tmp / "work"
        work.mkdir()
        with self.assertRaisesRegex(fi.FetchError, "git cat-file blob .*nothing.txt"):
            fi.fetch_blobs(self.commit, ["nothing.txt"], work)
        (self.tmp / "work2").mkdir()
        with self.assertRaisesRegex(fi.FetchError, "git fetch"):
            fi.fetch_blobs("0" * 40, ["a.txt"], self.tmp / "work2")

    def test_print_pins_prints_a_table_that_round_trips(self) -> None:
        table = {"a.txt": fi.sha256(self.LF), "sub/b.h": fi.sha256(self.MIXED)}
        out = io.StringIO()
        with mock.patch.object(fi, "FILES", table), contextlib.redirect_stdout(out):
            fi.print_pins(self.commit)
        text = out.getvalue()
        block = text.split('"""')[1].lstrip("\\\n")
        rows = {name: digest for digest, name in (line.split("  ", 1) for line in block.splitlines())}
        self.assertEqual(rows, table)
        self.assertIn(f'CONTENTS_SHA256 = "{fi.contents_sha256({"a.txt": self.LF, "sub/b.h": self.MIXED})}"', text)


class Notices(unittest.TestCase):
    """The notice rows and licence texts the pin needs, in THIRD_PARTY_NOTICES.md and licenses/."""

    @classmethod
    def setUpClass(cls) -> None:
        text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        cls.rows = [ln for ln in text.splitlines() if ln.startswith("| ")]

    def row(self, needle: str) -> str:
        found = [r for r in self.rows if needle in r]
        self.assertEqual(len(found), 1, f"{len(found)} notice rows mention {needle!r}; expected exactly one")
        return found[0]

    def test_dear_imgui_has_a_row_that_names_the_pin_the_script_and_the_licence(self) -> None:
        row = self.row("licenses/imgui-MIT.txt")
        for text in (fi.TAG, fi.COMMIT, fi.CONTENTS_SHA256, "tools/fetch_imgui.py", "https://github.com/ocornut/imgui",
                     "not in this repository", "MIT", "Omar Cornut"):
            self.assertIn(text, row)

    def test_the_stb_headers_have_a_row_that_says_which_alternative_is_relied_on(self) -> None:
        row = self.row("licenses/imgui-stb-MIT.txt")
        for text in ("imstb_rectpack.h", "imstb_textedit.h", "imstb_truetype.h", "alternative A", "Sean Barrett"):
            self.assertIn(text, row)

    def test_each_embedded_font_has_a_row_and_a_licence(self) -> None:
        clean, forever = self.row("licenses/proggyclean-MIT.txt"), self.row("licenses/proggyforever-MIT.txt")
        self.assertIn("ProggyClean.ttf", clean)
        self.assertIn("Tristan Grimmer", clean)
        self.assertIn("ProggyForever-Regular-minimal.ttf", forever)
        self.assertIn("Disco Hello", forever)

    def test_every_licence_a_row_links_is_in_the_repository_and_is_mit(self) -> None:
        for name in ("imgui-MIT.txt", "imgui-stb-MIT.txt", "proggyclean-MIT.txt", "proggyforever-MIT.txt"):
            with self.subTest(licence=name):
                text = (ROOT / "licenses" / name).read_text(encoding="utf-8")
                self.assertIn("Permission is hereby granted, free of charge", text)
                self.assertIn("THE SOFTWARE IS PROVIDED \"AS IS\"", text)
                self.assertRegex(text, r"Copyright \(c\) \d{4}")

    def test_the_imgui_licence_is_the_pinned_licence_file(self) -> None:
        data = (ROOT / "licenses" / "imgui-MIT.txt").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(fi.sha256(data), fi.FILES["LICENSE.txt"])


class GuardNames(unittest.TestCase):
    def test_the_default_folder_is_named_by_gitignore_and_by_the_guard(self) -> None:
        top = fi.DEFAULT_DEST.relative_to(ROOT).parts[0]
        ignored = {ln.strip().rstrip("/") for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
                   if ln.strip().endswith("/") and not ln.startswith("#")}
        self.assertIn(top, ignored)
        self.assertIn(top, tcc.cc.BLOCKED_DIRS)


@unittest.skipUnless(GIT, "git is required")
class GuardAndIgnore(unittest.TestCase):
    """The folder is ignored and blocked: the guard and .gitignore agree, with it present, absent and committed."""

    def setUp(self) -> None:
        self.repo = Path(tempfile.mkdtemp(prefix="oot2-imgui-guard-"))
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        for args in (("init", "-q", "-b", "main"), ("config", "user.name", "Tester"),
                     ("config", "user.email", "tester@users.noreply.github.com"), ("config", "commit.gpgsign", "false"),
                     ("config", "core.hooksPath", str(self.repo / ".no-hooks"))):
            tcc.git_in(self.repo, *args)
        shutil.copy(ROOT / ".gitignore", self.repo / ".gitignore")
        (self.repo / "ok.txt").write_text("plain text\n", encoding="utf-8")
        tcc.git_in(self.repo, "add", "-A")
        tcc.git_in(self.repo, "commit", "-q", "-m", "base")

    def run_cc(self, *argv: str) -> tuple[int, str]:
        done = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "check_content.py"), *argv], cwd=self.repo,
                              capture_output=True, text=True)
        return done.returncode, done.stdout

    def write_folder(self) -> None:
        folder = self.repo / fi.DEFAULT_DEST.relative_to(ROOT)
        (folder / "backends").mkdir(parents=True)
        # over both size limits, like the real imgui.cpp
        (folder / "imgui.cpp").write_bytes(b"int imgui_synthetic;\n" * 45000)
        (folder / "imgui.h").write_bytes(b"// synthetic\n")
        (folder / fi.RECORD).write_text("{}\n", encoding="utf-8")

    def test_the_whole_tree_passes_with_the_folder_absent_and_present(self) -> None:
        self.assertEqual(self.run_cc("--all")[0], 0)
        self.write_folder()
        code, out = self.run_cc("--all")
        self.assertEqual(code, 0, out)
        status = tcc.git_in(self.repo, "status", "--porcelain", "--untracked-files=all")
        self.assertNotIn("imgui", status, "git sees the fetched folder: it is not ignored")

    def test_a_fetched_file_that_is_committed_turns_the_guard_red(self) -> None:
        self.write_folder()
        rel = (fi.DEFAULT_DEST.relative_to(ROOT) / "imgui.h").as_posix()
        tcc.git_in(self.repo, "add", "-f", rel)
        code, out = self.run_cc("--staged")
        self.assertEqual(code, 1, out)
        self.assertIn("[blocked-dir]", out)
        self.assertIn("imgui.h", out)
        tcc.git_in(self.repo, "commit", "-q", "-m", "a fetched file")
        code, out = self.run_cc("--range", "HEAD~1..HEAD")
        self.assertEqual(code, 1, out)


class PlanWithImgui(unittest.TestCase):
    """What the Ninja file says about Dear ImGui, without a compiler (tools/build_game.py, plan)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="oot2-imgui-plan-")).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.rt = self.tmp / "rt"
        for name in ("ui/ui_standin.cpp", "core/x.c"):
            (self.rt / name).parent.mkdir(parents=True, exist_ok=True)
            (self.rt / name).write_text("", encoding="utf-8")
        self.imgui = self.tmp / "fetched" / "imgui"

    def plan(self, imgui: Path | None, **options) -> build_game.Plan:
        tool = dataclasses.replace(tbg.fake_tool(self.tmp), imgui=imgui)
        return build_game.plan(build_game.Build(self.tmp / "build", tool, runtime=self.rt, jobs=1, **options), [])

    def test_the_four_core_sources_are_built_with_their_own_rule_beside_the_runtimes(self) -> None:
        p = self.plan(self.imgui)
        lines = p.text.splitlines()
        objects = {obj.name: src for src, obj in p.rt_objects}
        for name in fi.CORE_SOURCES:
            stem = Path(name).stem
            self.assertEqual(objects[f"imgui-{stem}.o"], self.imgui / name)
            self.assertTrue(any(ln.startswith(f"build obj/runtime/imgui-{stem}.o: cxx_imgui ") and
                                ln.endswith(f"/{name}") for ln in lines), name)
        self.assertEqual(sum(1 for ln in lines if ": cxx_imgui " in ln), len(fi.CORE_SOURCES))
        archive = next(ln for ln in lines if ln.startswith("build ") and ": ar " in ln)
        for name in fi.CORE_SOURCES:
            self.assertIn(f"imgui-{Path(name).stem}.o", archive)
        self.assertIn("ui_ui_standin.o", archive)   # the runtime's own C++ stays where it was
        self.assertIn("rule cxx_imgui", lines)

    def test_flags_and_definitions_reach_the_cxx_files_and_not_the_c_files(self) -> None:
        lines = self.plan(self.imgui).text.splitlines()
        var = {ln.split(" = ", 1)[0]: ln.split(" = ", 1)[1] for ln in lines if re.match(r"^\w+ = ", ln)}
        defs = [f"-D{d}" for d in fi.DEFINES] + ["-DOOT2_WITH_IMGUI=1"]
        for name in ("cxxflags", "imguiflags"):
            for flag in defs:
                self.assertIn(flag, var[name].split(), f"{flag} missing from {name}")
            self.assertIn("-fno-exceptions", var[name].split())
            self.assertIn("-fno-rtti", var[name].split())
        self.assertIn("-w", var["imguiflags"].split())
        self.assertNotIn("-Wall", var["imguiflags"].split())
        self.assertIn("-Wall", var["cxxflags"].split())   # the runtime's own C++ keeps its warnings
        self.assertIn("fetched/imgui", var["cxxflags"])
        for name in ("rtflags", "genflags", "glueflags"):
            self.assertNotIn("IMGUI", var[name])

    def test_without_a_folder_or_with_a_prebuilt_library_nothing_of_it_is_planned(self) -> None:
        for p in (self.plan(None), self.plan(self.imgui, runtime_lib=self.tmp / "liboot2rt.a")):
            self.assertNotIn("cxx_imgui ", "\n".join(ln for ln in p.text.splitlines() if ln.startswith("build ")))
            self.assertNotIn("OOT2_WITH_IMGUI", p.text)
            self.assertFalse([o for _, o in p.rt_objects if o.name.startswith("imgui-")])


class FindImgui(Pinned):
    def setUp(self) -> None:
        super().setUp()
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(build_game.IMGUI_ENV, None)
        nowhere = mock.patch.object(fi, "DEFAULT_DEST", self.tmp / "default" / "imgui")
        nowhere.start()
        self.addCleanup(nowhere.stop)

    def test_no_folder_means_a_build_without_dear_imgui(self) -> None:
        self.assertIsNone(build_game.find_imgui())

    def test_a_named_folder_is_used_and_beats_the_variable_and_the_default(self) -> None:
        self.lay_out()
        os.environ[build_game.IMGUI_ENV] = str(self.tmp / "not there")
        self.assertEqual(build_game.find_imgui(self.dest), self.dest.resolve())
        os.environ[build_game.IMGUI_ENV] = str(self.dest)
        self.assertEqual(build_game.find_imgui(), self.dest.resolve())

    def test_the_default_folder_is_used_when_it_exists(self) -> None:
        fi.fetch(fi.DEFAULT_DEST, lambda: dict(self.files))
        self.assertEqual(build_game.find_imgui(), fi.DEFAULT_DEST.resolve())

    def test_a_folder_that_is_not_the_pin_is_refused_by_name_never_built(self) -> None:
        self.lay_out()
        victim = self.dest / "imgui_tables.cpp"
        victim.write_bytes(flip(victim.read_bytes()))
        with self.assertRaisesRegex(build_game.BuildError, r"cannot be used: imgui_tables\.cpp: SHA-256"):
            build_game.find_imgui(self.dest)
        os.environ[build_game.IMGUI_ENV] = str(self.dest)
        with self.assertRaisesRegex(build_game.BuildError, "tools/fetch_imgui.py"):
            build_game.find_imgui()
        fi.fetch(fi.DEFAULT_DEST, lambda: dict(self.files))   # a default folder that exists is checked too
        os.environ.pop(build_game.IMGUI_ENV)
        stale = fi.DEFAULT_DEST / "imgui.cpp"
        stale.write_bytes(flip(stale.read_bytes()))
        with self.assertRaisesRegex(build_game.BuildError, r"imgui\.cpp: SHA-256"):
            build_game.find_imgui()


def fetched_or_skip() -> Path:
    """The fetched folder (OOT2_IMGUI, else fetched/imgui), or a skip; a failure when OOT2_REQUIRE_IMGUI is set."""
    folder = build_game.find_imgui()
    if folder is None:
        if os.environ.get("OOT2_REQUIRE_IMGUI"):
            raise RuntimeError("OOT2_REQUIRE_IMGUI is set but no Dear ImGui folder was fetched (tools/fetch_imgui.py)")
        raise unittest.SkipTest("no fetched Dear ImGui (tools/fetch_imgui.py lays it out)")
    return folder


class FetchedFolder(unittest.TestCase):
    """A folder fetched for real: it is the pin, and the notices say what its files say."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = fetched_or_skip()

    def text(self, name: str) -> str:
        return (self.folder / name).read_text(encoding="utf-8")

    def test_the_folder_is_exactly_the_pin(self) -> None:
        self.assertEqual(fi.problems(self.folder), [])
        self.assertIn(f'#define IMGUI_VERSION       "{fi.VERSION}"', self.text("imgui.h"))

    def test_the_stb_headers_offer_the_mit_alternative_the_notice_relies_on(self) -> None:
        mine = (ROOT / "licenses" / "imgui-stb-MIT.txt").read_text(encoding="utf-8")
        mine = mine[mine.index("Copyright (c) 2017 Sean Barrett"):].strip()
        for name in ("imstb_rectpack.h", "imstb_textedit.h", "imstb_truetype.h"):
            with self.subTest(file=name):
                text = self.text(name)
                a = text.index("ALTERNATIVE A - MIT License") + len("ALTERNATIVE A - MIT License\n")
                self.assertEqual(text[a:text.index("---", a)].strip(), mine)
                self.assertIn("ALTERNATIVE B - Public Domain", text)

    def test_the_embedded_fonts_carry_the_notices_the_licence_files_hold(self) -> None:
        draw = self.text("imgui_draw.cpp")
        self.assertIn("// MIT License / Copyright (c) 2004, 2005 Tristan Grimmer", draw)
        self.assertIn("// MIT license / Copyright (c) 2026 Disco Hello, Copyright (c) 2019,2023 Tristan Grimmer", draw)
        clean = (ROOT / "licenses" / "proggyclean-MIT.txt").read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2004, 2005 Tristan Grimmer", clean)
        forever = (ROOT / "licenses" / "proggyforever-MIT.txt").read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2026 Disco Hello", forever)
        self.assertIn("Copyright (c) 2019,2023 Tristan Grimmer", forever)

    def test_the_licence_file_says_the_year_the_row_says(self) -> None:
        self.assertIn("Copyright (c) 2014-2026 Omar Cornut", self.text("LICENSE.txt"))
        self.assertIn("Copyright (c) 2014-2026 Omar Cornut", (ROOT / "licenses" / "imgui-MIT.txt").read_text())

    def test_the_core_sources_are_files_of_the_pin_and_none_of_them_is_the_demo(self) -> None:
        self.assertEqual(sorted(fi.CORE_SOURCES), ["imgui.cpp", "imgui_draw.cpp", "imgui_tables.cpp",
                                                   "imgui_widgets.cpp"])
        self.assertNotIn("imgui_demo.cpp", fi.FILES)   # the demo windows are not built: nothing here calls them


if __name__ == "__main__":
    unittest.main()
