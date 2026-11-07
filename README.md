# oot2-native

A static recompilation of *The Legend of Zelda: Ocarina of Time* for Nintendo Switch 2 (title `0400486024DEC000`,
update 1.1.1) into a native Windows x86-64 program with its own NVN2 front end and a Vulkan renderer. The AArch64 code
is lifted to C, compiled on your machine and linked with a runtime that stands in for the console's operating system,
its graphics API and its services.

**Status.** Not playable yet. `oot2 prepare` builds the program from a dump; the program runs the game's start-up and
its first frames on a null graphics back end, with no window and no picture. The Vulkan back end, which shows the title
screen, is under construction (milestone M1: the title screen with its movie at 60 fps).
<!-- WAITING-FOR-M1: rewrite the status paragraph with what runs on the day of publication. -->

The repository holds tools and a runtime and nothing else: no game file, ever (see [Legal model](#legal-model)).
Design and decisions: [docs/architecture.md](docs/architecture.md).

## Requirements

- Windows 10 or 11, x86-64 with the `x86-64-v3` features (AVX2, BMI1, BMI2, FMA, F16C, LZCNT, MOVBE).
- A Vulkan 1.3 GPU. NVIDIA is the target of the first release (the mesh path assumes a subgroup size of 32).
  Reference machine: Ryzen 5 5600X, 16 GB RAM, GeForce RTX 3060 Ti 8 GB, driver 617.14.
- Your own dump of the game: one `.nup` file holding the base program and its update (about 25 GiB).
- About 38 GiB free on the disk that receives the build.
- Tools, until a prebuilt bundle exists: Python 3.14 with `numpy` and `capstone`; clang and clang++ from
  [LLVM-MinGW](https://github.com/mstorsjo/llvm-mingw) (x86_64; CI and the reference machine use 22.1.8), Ninja,
  git; the Vulkan headers, Volk and glslang, from the Vulkan SDK 1.4.357 or from `tools/fetch_vulkan.py`; the .NET 10
  SDK and the Visual Studio Build Tools (C++) to build the shader translator with `tools/build_xlate.py`. The movies
  need a dav1d kit from `tools/fetch_dav1d.py` (`DAV1D_DIR`); without one the check step says so.

## Build from your own dump

```bat
REM Without the Vulkan SDK: lay out the headers, Volk and glslang once.
py -3.14 tools/fetch_vulkan.py --dest vk
set VULKAN_SDK=vk

REM The shader translator.
py -3.14 tools/build_xlate.py
set OOT2_XLATE=build\xlate\publish\oot2-xlate.exe

REM Check the machine and the dump, then run every step. GAME.nup is your dump, OUT an empty or new folder.
py -3.14 tools/oot2.py prepare --dump GAME.nup --dest OUT --check
py -3.14 tools/oot2.py prepare --dump GAME.nup --dest OUT
```

The same with a window: `pyw -3.14 tools/oot2_setup.pyw` (or a double-click on it) asks for the dump and the folder,
checks the PC, runs the steps with their progress, and starts the game when it is prepared. It runs `oot2 prepare` for
you, so the tools above are still needed until the bundle exists; its logs are in `%LOCALAPPDATA%\oot2-native\logs`.

`--check` runs the first step alone: the dump, the disk, the CPU, the toolchain, the translator and Vulkan, and what
the other steps would do. A run that stops keeps its finished steps: start it again with the same `--dest`. The
program is `OUT\oot2.exe`. `tools/oot2.py` documents every option. On a 6-core desktop the lift and compile take about
half an hour and the shader translation about ten minutes; the pipeline step takes hours on a first run, and
`--skip-pipelines` leaves it out.

A prebuilt bundle (toolchain, translator and runtime library in one archive, so that the steps above need nothing
installed) is planned. It will contain no file derived from the game either.

## Legal model

This repository contains tools and a runtime, and it never contains and never accepts:

- game code, generated C or C++, or a disassembly of the game;
- shaders, shader caches, textures, models, audio, movies, fonts or text taken from the game;
- keys, firmware or tickets, or anything that decrypts the game or works around a protection measure;
- frame captures or screenshots of the game;
- addresses, offsets, program keys, module hashes or patch words read out of the game: none is committed today and
  none is accepted in a contribution. A game-specific fix is written as a structural match over the lifted code; a
  table of sites, if one is ever needed, is a separate decision of the maintainer.

Everyone builds from their own dump on their own machine, and the tools read it locally. Nothing derived from your dump
is uploaded. A content check runs before every commit (once you enable the hook) and in CI on every push and pull
request; see [CONTRIBUTING.md](CONTRIBUTING.md). Do not ask for game files, keys or dumps in issues, pull requests or
discussions.

The program you build, the `--dest` folder and every cache in it contain material derived from your copy of the game.
Keep them to yourself: do not share, upload or redistribute them, and do not attach them to issues or pull requests.

## Licence

oot2-native is licensed GPL-3.0-or-later: the text is in [LICENSE](LICENSE), and every source file written for the
project carries `SPDX-License-Identifier: GPL-3.0-or-later`. This licence covers the project's source and grants no right
in the game.

Parts that come from other projects keep their own licence and header (MIT, BSD, ISC, zlib or GPL-3.0-or-later, all
compatible with GPL-3.0-or-later). Each is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) with its source,
commit and modifications, and its licence text is in [licenses/](licenses/) or next to the files. Third-party code that
is not in the repository keeps its own licence and is listed in the same file: `oot2 prepare` downloads the ZBIC decoder
and Dear ImGui at pinned versions, and the scripts `tools/fetch_dav1d.py` and `tools/fetch_vulkan.py` lay out dav1d and
the Vulkan headers, Volk and glslang the same way, each checked by hash. The planned bundle will carry the compiled
parts with their licence texts.

## Credits

oot2-native is written for this project, with parts adapted from earlier recompilations of Switch and Switch 2 games
(listed in the notices) and strongly inspired by them:
[oot-pc-recomp](https://github.com/Chiuauaa/oot-pc-recomp) (Chiuauaa, philneitz), [DKB Recomp](https://github.com/Lucas-blip-png/dkb-pc-recomp),
[TotK Recomp](https://github.com/valentderah/totk-pc-recomp) (whose legal model this project follows),
[LASRecomp](https://github.com/peiikai/LASRecomp), [nxnative](https://github.com/seedds/nxnative),
[Lichii](https://github.com/zortos293/Lichii) and [Eglantine](https://github.com/Djihads80/Eglantine). Code taken from
any of them keeps its header and has a row in the notices.

## Disclaimer

oot2-native is an independent project. It is not affiliated with, endorsed by or connected to Nintendo. The Legend of
Zelda, Ocarina of Time, Nintendo Switch and Nintendo Switch 2 are trademarks of Nintendo. All other trademarks belong
to their owners.
