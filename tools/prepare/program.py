# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The steps of `oot2 prepare` that turn the player's decoded modules into the program: lift, compile, link
(architecture D4 and section 13).

In the output folder:

  exefs/      the plain modules (main, subsdk0 to subsdk9, sdk) and main.npdm; the modules step writes them
              (modules.py), and oot2.exe loads and checks them at each start
  gen/        the lifted C (tools/lift/gen.py), about 1.2 GB for the game. It stays after the build: a later lift
              rewrites only the chunks whose C changed, so the compile after it is short. spin_waits.json beside it
              is the table of the game's spin waits (tools/lift/spin_waits.py), which the lift step computes from the
              modules before it lifts them, so that their back-edges go through rt_spin_gate
  build/      build.ninja, the objects and their SHA-256 manifest, the runtime library, the program glue
              (tools/build_game.py)
  romfs/      the packed RomFS (extract.py); the link bakes its two files into the glue, which oot2.exe checks
  oot2.exe    the program
  temp/logs/  the output of the lift and of Ninja

The keys follow what each output is made from. Lift: the modules and the lifter's code. Compile: the lift's key, the
build tool, the headers the generated C includes, the compiler and the flags. Link: the compile's key, the build tool,
the whole runtime (its shaders included), the compiler, the flags, the Vulkan headers, Volk and glslang, the pin of
Dear ImGui, and main.npdm. So an update of the tools that changed only the runtime runs the link alone, where Ninja compiles the
runtime files that changed and links again, without the 22 minutes of generated C (D4). Keys need the compiler and the
Vulkan SDK only; Ninja and the archiver are looked for by the check step and used when a step runs.

`--jobs N` sets the compile jobs (default: the CPU count, capped by the free memory); it is not part of a key.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import check
from .extract import IMAGE, IN_PLACE, INDEX, ROMFS
from .runner import Context, PrepareError, Step, file_sha256, fingerprint

TOOLS = Path(__file__).resolve().parent.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_game  # noqa: E402  (tools/build_game.py)
import bundle_env  # noqa: E402  (tools/bundle_env.py)
import fetch_imgui  # noqa: E402  (tools/fetch_imgui.py)
import fetch_sdl3  # noqa: E402  (tools/fetch_sdl3.py)
import layout  # noqa: E402  (tools/layout.py)

LIFTER = TOOLS / "lift"
EXEFS, GEN, BUILD = "exefs", "gen", "build"
SPIN_TABLE = "spin_waits.json"     # in gen/
GiB = 1 << 30


def python() -> list[str]:
    """The interpreter that runs this, for the lifter's commands, isolated when this is."""
    return [sys.executable, *(["-I"] if sys.flags.isolated else [])]


def exefs_modules(ctx: Context) -> list[Path]:
    """The plain modules in exefs/, in load order."""
    folder = ctx.dest / EXEFS
    return [folder / n for n in layout.ORDER if (folder / n).is_file()]


def compiler_id(ctx: Context) -> str:
    return build_game.tool_id(check.compiler(ctx))


def imgui_folder(ctx: Context) -> Path:
    """The fetched Dear ImGui that the runtime's menu is built on (runtime/ui, tools/fetch_imgui.py): the folder
    --imgui or OOT2_IMGUI names, used as it is and refused by name when it is not exactly the pinned layout, else
    fetched/imgui, laid out first when it is absent or another pin (git and network access; a folder laid out before
    is replaced, anything else is refused). The link step builds the four core sources into the runtime library, so
    oot2.exe carries the menu."""
    named = getattr(ctx.options, "imgui", None) or os.environ.get("OOT2_IMGUI")
    try:
        if named:
            return build_game.find_imgui(named)
        dest = fetch_imgui.DEFAULT_DEST
        if fetch_imgui.problems(dest):
            ctx.reporter.say(f"link: fetching Dear ImGui {fetch_imgui.VERSION} into {dest}")
            for line in fetch_imgui.fetch(dest):
                ctx.reporter.say(f"link: {line}")
        return dest
    except (build_game.BuildError, fetch_imgui.FetchError, OSError) as e:
        raise PrepareError(f"Dear ImGui, which the menu is built on: {e} (fetch it with tools/fetch_imgui.py, or "
                           "name a fetched folder with --imgui)") from None


runtime_lib = check.runtime_lib   # the prebuilt runtime library, or None (check.py finds it for both steps)


def imgui_key() -> str:
    """The part of the link step's key that Dear ImGui gives: the pin's contents hash, so that a new pin links again."""
    return f"imgui={fetch_imgui.CONTENTS_SHA256}"


def toolchain(ctx: Context, imgui: bool = False) -> build_game.Toolchain:
    """The build tools. With imgui the fetched Dear ImGui goes with them (the link step builds the runtime library, and
    the lifted code needs none); with a prebuilt runtime library there is no Dear ImGui to fetch, no Volk and no glslang."""
    prebuilt = runtime_lib(ctx) is not None
    try:
        return build_game.find_toolchain(check.compiler(ctx), vulkan_sdk=getattr(ctx.options, "vulkan_sdk", None),
                                         imgui=imgui_folder(ctx) if imgui and not prebuilt else None,
                                         dav1d_dir=getattr(ctx.options, "dav1d_dir", None),
                                         sdl3_dir=getattr(ctx.options, "sdl3_dir", None), prebuilt=prebuilt)
    except build_game.BuildError as e:
        raise PrepareError(str(e)) from None


def vulkan_key(ctx: Context) -> list:
    """The part of the link step's key that the Vulkan SDK gives (build_game.vulkan_inputs), or, with a prebuilt runtime
    library, the library itself: its bytes are what the headers, Volk, glslang and Dear ImGui became."""
    lib = runtime_lib(ctx)
    if lib is not None:
        return ["prebuilt-runtime:", lib, *build_game.dav1d_inputs(build_game.find_dav1d(
            getattr(ctx.options, "dav1d_dir", None)))]
    include = build_game.vulkan_include(getattr(ctx.options, "vulkan_sdk", None))
    kit = build_game.find_dav1d(getattr(ctx.options, "dav1d_dir", None))
    return build_game.vulkan_inputs(include, build_game.find_glslang(include)) + build_game.dav1d_inputs(kit)


def sdl3_key(ctx: Context) -> list:
    """The part of the link step's key that the SDL3 kit gives: its library and headers (build_game.sdl3_inputs), and the
    system libraries the link adds with it, so that a new kit or a new list links again."""
    try:
        kit = build_game.find_sdl3(getattr(ctx.options, "sdl3_dir", None))
    except build_game.BuildError as e:
        raise PrepareError(str(e)) from None
    return [*build_game.sdl3_inputs(kit), *(["sdl3 link libs: " + " ".join(fetch_sdl3.LINK_LIBS)] if kit else [])]


def log_lines(ctx: Context, name: str, progress):
    """A line callback that writes the tool's output to temp/logs/<name>.log and hands each line to progress."""
    path = ctx.dest / check.TEMP / "logs" / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    log = open(path, "w", encoding="utf-8", newline="\n")

    def on_line(line: str) -> None:
        log.write(line + "\n")
        progress(line)

    return on_line, log, path


def game_files(ctx: Context) -> list[build_game.GameFile]:
    """What oot2.exe checks at each start besides the modules and main.npdm (D4): the RomFS index by SHA-256, and the
    image the index reads from (23 GiB and more is too much to hash at each start; the index holds every file's hash
    and the image's size). In the copy form the image is romfs.bin, checked by size. In place (extract.py) it is the
    player's .nup at the path the extract step recorded, checked by size and by the SHA-256 of fixed ranges
    (build_game.sample_sha256), so a dump moved, replaced or changed stops the start with a named error. Both exist
    once the extract step has run; without them the program checks the modules alone. The runtime's file system finds
    the two by their descriptions, "RomFS index" and "RomFS image"."""
    index = (ctx.dest / ROMFS / INDEX).resolve()
    rec = ctx.record("extract") or {}
    if rec.get("romfs_form") == IN_PLACE:
        if not index.is_file():
            return []
        return [build_game.GameFile("RomFS index", index, True),
                build_game.GameFile("RomFS image", Path(rec["nup"]["path"]).resolve(), False, sampled=True)]
    image = (ctx.dest / ROMFS / IMAGE).resolve()
    if not (index.is_file() and image.is_file()):
        return []
    return [build_game.GameFile("RomFS index", index, True), build_game.GameFile("RomFS image", image, False)]


def romfs_key(ctx: Context) -> list:
    """The part of the link step's key that the RomFS gives: the index's contents and what the program checks of the
    image: its size, and in place its path and sampled SHA-256 too."""
    files = game_files(ctx)
    if not files:
        return ["romfs=none"]
    index, image = files
    try:
        size = image.path.stat().st_size
        sample = build_game.sample_sha256(image.path) if image.sampled else ""
    except OSError:
        size, sample = "missing", ""
    if image.sampled:
        return ["romfs:", index.path, f"romfs-in-place={image.path.as_posix()}:{size}:{sample}"]
    return ["romfs:", index.path, f"romfs-image-bytes={size}"]


def build_for(ctx: Context, imgui: bool = False) -> build_game.Build:
    jobs = getattr(ctx.options, "jobs", None) or 0
    return build_game.Build(ctx.dest / BUILD, toolchain(ctx, imgui), gen=ctx.dest / GEN, exefs=ctx.dest / EXEFS,
                            exe=ctx.dest / build_game.EXE_NAME, runtime=build_game.RUNTIME, files=game_files(ctx),
                            jobs=jobs, runtime_lib=runtime_lib(ctx))


def ninja_progress(ctx: Context, tag: str, what: str):
    def progress(line: str) -> None:
        m = build_game.NINJA_LINE.match(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            ctx.reporter.progress(tag, f"{tag}: {what} {done:,} of {total:,}", every=10.0, final=done == total)
    return progress


def mib(n: int) -> str:
    return f"{n / (1 << 20):,.1f} MiB"


class Lift(Step):
    id = "lift"
    needs = ("modules",)   # which writes exefs/

    def key(self, ctx: Context) -> str:
        # lift-2: the step searches the spin waits and lifts with their table (a lift-1 output has no spin gate).
        return fingerprint("lift-2", *exefs_modules(ctx), LIFTER, TOOLS / "layout.py")

    def current(self, ctx: Context, record: dict) -> bool:
        gen = ctx.dest / GEN
        return (gen / "lift_report.json").is_file() and (gen / SPIN_TABLE).is_file() and \
            all((gen / n / "manifest.json").is_file() for n in record.get("modules", ()))

    def verify(self, ctx: Context, record: dict) -> bool:
        try:
            build_game.lifted_modules(ctx.dest / GEN)
        except (build_game.BuildError, OSError, ValueError, KeyError):
            return False
        return True

    def run(self, ctx: Context) -> dict:
        mods = exefs_modules(ctx)
        if not any(p.name == "main" for p in mods):
            raise PrepareError(f"the plain modules are not in {ctx.dest / EXEFS}: the extraction step writes them "
                               "from the dump.")
        gen = ctx.dest / GEN
        rep = ctx.reporter
        names = ",".join(p.name for p in mods)
        table = gen / SPIN_TABLE
        table.parent.mkdir(parents=True, exist_ok=True)
        on_line, log, path = log_lines(ctx, "spin_waits", lambda line: rep.say(f"lift: spin waits: {line}"))
        try:
            search = build_game.run_measured([*python(), str(LIFTER / "spin_waits.py"), "--exefs",
                                              str(ctx.dest / EXEFS), "--out", str(table), "--modules", names],
                                             TOOLS.parent, on_line)
        finally:
            log.close()
        if search.code:
            raise PrepareError(f"the spin-wait search stopped with exit code {search.code}. The end of {path}:\n" +
                               "\n".join(search.tail[-15:]))
        waits = sum(len(v) for v in json.loads(table.read_text(encoding="utf-8")).values())
        # Private bytes are Windows's figure; elsewhere the only one is the resident set.
        peak = (f"{mib(search.peak_process)} private, {mib(search.peak_working_set)} resident" if build_game.WIN
                else f"{mib(search.peak_working_set)} resident")
        rep.say(f"lift: {waits:,} spin waits found in {search.seconds:.0f} s, peak {peak}")
        on_line, log, path = log_lines(ctx, "lift", lambda line: rep.progress(
            "lift", f"lift: {line}", every=10.0) if line.startswith("emitted ") else rep.say(f"lift: {line}"))
        cmd = [*python(), str(LIFTER / "gen.py"), "--exefs", str(ctx.dest / EXEFS), "--out", str(gen),
               "--modules", names, "--spin-waits", str(table)]
        try:
            m = build_game.run_measured(cmd, TOOLS.parent, on_line)
        finally:
            log.close()
        if m.code:
            raise PrepareError(f"the lift stopped with exit code {m.code}. The end of {path}:\n" + "\n".join(m.tail[-15:]))
        report = json.loads((gen / "lift_report.json").read_text(encoding="utf-8"))
        stats = [mod["stats"] for mod in report["modules"].values()]
        unsupported = sum(s.get("unsupported", 0) for s in stats)
        if unsupported:
            rep.warning(f"the lift met {unsupported:,} unsupported instructions; the game stops with a named fatal "
                        "where one runs")
        c_bytes = sum(s.get("c_bytes", 0) for s in stats)
        gates = report.get("spin_gates", 0)
        rep.say(f"lift: {report['total_functions']:,} functions, {gates:,} spin gates, {c_bytes / GiB:.2f} GiB of C, "
                f"peak {mib(m.peak_process)} per process" + (f", {mib(m.peak_all)} all processes" if m.peak_all else ""))
        return {"modules": [p.name for p in mods], "functions": report["total_functions"],
                "unsupported": unsupported, "spin_waits": waits, "spin_gates": gates,
                "spin_search_seconds": round(search.seconds, 1),
                "spin_search_peak_process": search.peak_process, "spin_search_peak_all": search.peak_all,
                "spin_search_peak_working_set": search.peak_working_set,
                "peak_process": m.peak_process, "peak_all": m.peak_all, "bytes": c_bytes}


class Compile(Step):
    id = "compile"
    needs = ("lift",)

    def key(self, ctx: Context) -> str:
        return fingerprint("compile-1", TOOLS / "build_game.py", build_game.RUNTIME / "include", compiler_id(ctx),
                           " ".join(build_game.GEN_FLAGS))

    def current(self, ctx: Context, record: dict) -> bool:
        obj = ctx.dest / BUILD / "obj"
        man = build_game.read_manifest(obj / build_game.MANIFEST)
        return len(man) >= record.get("objects", 1) and all((obj / name).is_file() for name in man)

    def verify(self, ctx: Context, record: dict) -> bool:
        obj = ctx.dest / BUILD / "obj"
        man = build_game.read_manifest(obj / build_game.MANIFEST)
        return bool(man) and all((obj / name).is_file() and file_sha256(obj / name) == sha
                                 for name, (sha, _, _) in man.items())

    def run(self, ctx: Context) -> dict:
        b = build_for(ctx)
        on_line, log, path = log_lines(ctx, "compile", ninja_progress(ctx, "compile", "files"))
        ctx.reporter.say(f"compile: {b.jobs} jobs, {b.tool.cc} ({build_game.tool_id(b.tool.cc)})")
        try:
            r = build_game.compile_generated(b, on_line)
        except build_game.BuildError as e:
            raise PrepareError(f"{e}\n(the whole output is in {path})") from None
        finally:
            log.close()
        ctx.reporter.say(f"compile: {r['objects']:,} objects in {r['seconds']:.0f} s on {r['jobs']} jobs, peak "
                         f"{mib(r['peak_process'])} per process" +
                         (f", {mib(r['peak_all'])} all processes" if r["peak_all"] else "") +
                         f"; {r['manifest']['verified']:,} objects verified, {r['manifest']['recompiled']} recompiled")
        return {"objects": r["objects"], "functions": r["functions"], "jobs": r["jobs"],
                "compile_seconds": r["seconds"], "peak_process": r["peak_process"], "peak_all": r["peak_all"],
                "bytes": r["bytes"]}


class Link(Step):
    id = "link"
    needs = ("compile",)

    def key(self, ctx: Context) -> str:
        prebuilt = runtime_lib(ctx) is not None
        return fingerprint("link-1", TOOLS / "build_game.py", build_game.RUNTIME / "include" if prebuilt else
                           build_game.RUNTIME, compiler_id(ctx),
                           " ".join(build_game.RT_FLAGS + build_game.LINK_FLAGS), ctx.dest / EXEFS / "main.npdm",
                           *romfs_key(ctx), *vulkan_key(ctx), *sdl3_key(ctx), "prebuilt" if prebuilt else imgui_key())

    def current(self, ctx: Context, record: dict) -> bool:
        exe = ctx.dest / build_game.EXE_NAME
        if not exe.is_file():
            return False
        st = exe.stat()
        return st.st_size == record.get("exe_bytes") and st.st_mtime_ns == record.get("exe_mtime_ns")

    def verify(self, ctx: Context, record: dict) -> bool:
        exe = ctx.dest / build_game.EXE_NAME
        return exe.is_file() and file_sha256(exe) == record.get("exe_sha256")

    def run(self, ctx: Context) -> dict:
        b = build_for(ctx, imgui=True)
        if b.runtime_lib:
            ctx.reporter.say(f"link: the prebuilt runtime library {b.runtime_lib} (menu library and shaders inside)")
        on_line, log, path = log_lines(ctx, "link", ninja_progress(ctx, "link", "files"))
        try:
            r = build_game.link_program(b, on_line)
        except build_game.BuildError as e:
            raise PrepareError(f"{e}\n(the whole output is in {path})") from None
        finally:
            log.close()
        npdm = r["baked"]["npdm"]
        if npdm is None:
            ctx.reporter.warning(f"no main.npdm in {ctx.dest / EXEFS}: the program starts with the default values")
        ctx.reporter.say(f"link: {b.exe.name} {mib(r['exe_bytes'])} (image {mib(r['image_bytes'])}) in "
                         f"{r['seconds']:.1f} s, peak {mib(r['peak_process'])}; {r['manifest']['verified']:,} "
                         f"objects verified, {r['manifest']['recompiled']} recompiled")
        lib = b.lib.stat().st_size if b.lib.is_file() and not b.runtime_lib else 0
        return {"exe_bytes": r["exe_bytes"], "exe_mtime_ns": r["exe_mtime_ns"], "exe_sha256": file_sha256(b.exe),
                "image_bytes": r["image_bytes"], "link_seconds": r["seconds"], "peak_process": r["peak_process"],
                "bytes": r["exe_bytes"] + lib}
