# Contributing

Thank you for helping. Please read the one hard rule first.

## The one hard rule: nothing that came out of the game

Nothing that came out of the game goes into this project. That covers commits,
issues, pull requests, comments and attachments alike:

- no game files, dumps, keys, firmware or tickets;
- no recompiled or generated code, no disassembly of the game;
- no shaders, shader caches, textures, models, audio, movies, fonts or game text;
- **no screenshots, screen recordings or frame dumps of the game**, and no GPU
  capture files.

Evidence is text or hashes. To report what you saw, describe it in words, paste
the relevant lines of a log, or give identifiers and hashes (a function address,
a shader key, a SHA-256 of a file you own). If a maintainer needs more, they will
ask a question; they will not ask for a file.

Links to game files, keys or firmware get the issue or pull request closed and
the content removed.

### The content check

`tools/check_content.py` refuses game-derived file types and extensions, large
files, binary data, known file signatures, key files and key syntaxes, hex,
base64 and decimal dumps, source files that look recompiled (very large, or full
of functions named by address), and directories named for generated or extracted
data (`build`, `out`, `generated`, `extracted`, `captures` and the others that
`.gitignore` lists; a test keeps the two lists in step). It runs in two places:

- before each local commit, once you enable the hook (do this after cloning):

  ```sh
  git config core.hooksPath .githooks
  ```

- in CI on every push and pull request, over every commit in the push or pull
  request (not only the last one) and over the whole tree.

CI is the backstop, because the local hook only runs once you have enabled it.
A pull request can also edit the guard that checks it, so changes to `tools/`,
`tests/`, `.githooks/`, `.github/` and the allowlists are reviewed by the code
owners (`.github/CODEOWNERS`) with extra care.

Never bypass it with `--no-verify`. If it flags a file by mistake, change the
check in a separate pull request and say why. Exceptions (a logo, a reviewed list
of hashes) are explicit entries in the allowlists at the top of the script and are
reviewed like code.

Known false positives are accepted on purpose. A standard table (a CRC-32 table,
an S-box) looks like a dump, and so does a bare list of hashes with no labels.
Label hashes (`sha256`, `build_id`), put them in a Markdown table, or have the
file added to `ALLOW_BLOBS` in a reviewed change. Compiled SPIR-V built from this
project's own shader sources is binary data and would need an `ALLOW_BINARY`
entry, so build it in CI or at build time instead of committing it. Shaders
translated from the game are never committed in any form.

## Running the checks and the tests

The check needs Python 3.9 or newer and only the standard library; the lifter
and its tests also need numpy, and the runtime and lifter tests need clang. CI
runs the tests twice on every push and pull request: on Linux with Python 3.12
(`content-guard.yml`), and on Windows with Python 3.14 and llvm-mingw's clang,
after building the translator (`windows-build.yml`). Neither job can build the
game; every test input is synthetic.

```sh
python tools/check_content.py --staged         # what is about to be committed
python tools/check_content.py --range A..B     # every commit in a range
python tools/check_content.py --all            # the whole tracked tree
python tools/check_content.py --paths PATH...  # files or folders on disk
python -m unittest discover -s tests -v        # the unit tests
```

Every check has a test with a case that must be caught and a clean one that must
pass. Test fixtures are always synthetic: build the bytes inside the test, never
copy real game data. A new check is not finished until you have seen it fail
against the fault it is meant to catch.

## Licensing of your contribution

By submitting a contribution you state that you wrote it, or that you have the
right to submit it, and you license it under GPL-3.0-or-later, the licence of the
project. Source files carry an SPDX header
(`SPDX-License-Identifier: GPL-3.0-or-later`).

## Code from other projects

oot2-native is GPL-3.0-or-later as a whole, so code under GPL-3.0-compatible
licences (MIT, BSD, ISC, zlib, Apache-2.0, GPL-3.0-or-later) may be copied. Code
that is GPL-3.0-only is not accepted, because it cannot be combined with the
"or later" terms. Each borrowed file keeps its original header and gets a row in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): source, URL, commit, licence,
and whether it was modified. A file whose provenance is not recorded does not go
in.

No licence means no copy. A repository that states no licence is all rights
reserved, and none of its code can be used here, however much we admire it. Read
it, learn from it, write your own.

## Pull requests

- Keep them small and focused: one topic each.
- Describe the problem first, then what changed, then how you tested it (which
  commit, which GPU and driver, which Windows version).
- Commit messages are plain English sentences with a short subject line.
- Code, comments, commit messages and documentation are in English.
- Generated code is never edited by hand: change the generator.

## Reporting content that should not be here

If you see game material in the repository, an issue, a pull request or a
comment, do not repeat it and do not link to it. Report it privately, using
GitHub's private vulnerability reporting (the Security tab of the repository,
when it is enabled) or by contacting the maintainer, RealZotto, on GitHub, and
name the commit or the issue number. It will be removed, and the history
rewritten if needed.

## Reporting bugs

Include the commit you built, your GPU, driver and Windows version, the log text,
and the steps to reproduce. Describe what you see instead of attaching pictures
of the game.
