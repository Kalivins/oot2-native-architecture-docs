# oot2-native

A static recompilation of The Legend of Zelda: Ocarina of Time for Nintendo
Switch 2 (title `0400486024DEC000`, update 1.1.1) into a native Windows x86-64
program with a Vulkan renderer.

**Status: early. Nothing runs yet.** The architecture is being written. There is
no release, no build you can run, and no promise of a date. This page will say so
until that changes.

## No game files, ever

This repository contains tools and a runtime, and nothing else. It never
contains, and will never accept:

- game code, generated C or C++, or a disassembly of the game;
- shaders, shader caches, textures, models, audio, movies, fonts or text taken from the game;
- keys, firmware or tickets, or anything that decrypts the game or works around a protection measure;
- frame captures or screenshots of the game.

Everyone builds from their own dump, on their own machine. In outline, once the
tools exist:

1. You dump the game from your own console and your own copy.
2. The tools read that dump locally and produce everything derived from it
   (recompiled code, shader translations, caches) into ignored folders such as
   `build/` and `generated/`.
3. The tools compile and link those outputs with the runtime into a program on
   your disk.

The tools are designed so that nothing derived from your dump is uploaded, and
nothing a release of this project contains comes from the game. A content check
runs before every commit once you enable the hook, and in CI on every push and
pull request, to keep it that way; see [CONTRIBUTING.md](CONTRIBUTING.md).
Please do not ask for game files, keys or dumps in issues, pull requests or
discussions.

### The program you build is yours alone

The program you build, and the `build/` and `generated/` folders and any cache
next to it, contain material derived from your own copy of the game. Keep them
to yourself: do not share, upload or redistribute the built program, the
generated folders or any cache, and do not attach them to issues or pull
requests.

## Credits

oot2-native is written from scratch and strongly inspired by earlier work on
recompiling Switch and Switch 2 games. These projects showed what is possible
and how to approach it:

- [oot-pc-recomp](https://github.com/Chiuauaa/oot-pc-recomp) by Chiuauaa and philneitz (also [philneitz's fork](https://github.com/philneitz/oot-pc-recomp)), the first recompilation of this game.
- [DKB Recomp](https://github.com/Lucas-blip-png/dkb-pc-recomp) by Lucas-blip-png.
- [TotK Recomp](https://github.com/valentderah/totk-pc-recomp) by valentderah, whose legal model this project follows: tools and runtime only, built from your own dump.
- [LASRecomp](https://github.com/peiikai/LASRecomp) by peiikai.
- [nxnative](https://github.com/seedds/nxnative) by seedds.
- [Lichii](https://github.com/zortos293/Lichii) and [Eglantine](https://github.com/Djihads80/Eglantine), for their knowledge of how Ampere shaders behave.

If code from any of them is ever used here, it keeps its original header and gets
a row in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Disclaimer

oot2-native is an independent project. It is not affiliated with, endorsed by or
connected to Nintendo. The Legend of Zelda, Ocarina of Time, Nintendo Switch and
Nintendo Switch 2 are trademarks of Nintendo. All other trademarks belong to
their owners.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
