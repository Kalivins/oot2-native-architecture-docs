## What and why

Describe the problem first, then what changed.

## How it was tested

The tests you ran (`python -m unittest discover -s tests -v`, or the files that cover what you changed), and for
anything the program draws: commit built, graphics card and driver, Windows version. Text only, no screenshots.

## Checklist

- [ ] No game material is attached or included: no game files, dumps, keys, firmware, shaders, textures, audio, generated code, screenshots or captures.
- [ ] No address, offset, program key or module hash read out of the game is in a committed file.
- [ ] I wrote this change, or I have the right to submit it under GPL-3.0-or-later; new source files carry the SPDX and copyright lines.
- [ ] Code taken from another project keeps its header and has a row in THIRD_PARTY_NOTICES.md.
- [ ] I ran `tools/check_content.py --staged` (the pre-commit hook does it) and the tests that cover my change.
