# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""One licence statement: the LICENSE text, the documents and every file header say the same thing.

The project is GPL-3.0-or-later. Parts taken from other projects keep their own licence, and each has a row in
THIRD_PARTY_NOTICES.md with its licence text in licenses/. A public recompilation was challenged over exactly this
(conflicting statements in its own files), so the statement is checked, not trusted.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPDX = "SPDX-License-Identifier: GPL-3.0-or-later"
COPYRIGHT = "Copyright (C) 2026 The oot2-native contributors"
HEADER_LINES = 8
# Fail closed: every tracked file must carry the header unless it is data (these suffixes and names) or exempt below.
DATA_SUFFIXES = {".md", ".txt", ".ivf"}
DATA_NAMES = {".gitignore", ".gitattributes", "LICENSE", "CODEOWNERS"}
# Words that mark a sentence as being about other projects' code, the only place another licence may be named.
THIRD_PARTY_WORDS = ("other project", "compatible", "keeps its own", "keep their own", "borrowed", "third-party")
OTHER_LICENCES = re.compile(r"\b(MIT|BSD|Apache|ISC|zlib|GPL-2|AGPL|LGPL)\b")
# "released under X", "licensed X", "you license it under X": X must be the project's licence.
STATEMENT = re.compile(r"\b(?:licensed|released|distributed|available|dual-licensed)\s+(?:under\s+|as\s+)?(?:the\s+)?"
                       r"([A-Za-z0-9][\w.\-]*)|\blicen[sc]e\s+(?:it\s+)?under\s+(?:the\s+)?([A-Za-z0-9][\w.\-]*)", re.I)
# Folders and files that keep their own licence or are data, with the reason.
EXEMPT_PREFIXES = {
    "xlate/third_party/": "vendored, each component keeps its own licence file (THIRD_PARTY_NOTICES.md)",
    "licenses/": "licence texts of other projects",
    ".github/ISSUE_TEMPLATE/": "issue forms: data, no code",
}
EXEMPT_FILES = {
    "runtime/include/oot2/nvn.h": "kept verbatim from upstream, whose MIT header it carries (THIRD_PARTY_NOTICES.md)",
}
DOCUMENTS = ("README.md", "CONTRIBUTING.md", "THIRD_PARTY_NOTICES.md", ".github/PULL_REQUEST_TEMPLATE.md")


def tracked() -> list[str]:
    done = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if done.returncode != 0:
        raise unittest.SkipTest("not a git checkout: the tracked files cannot be listed")
    return [p for p in done.stdout.decode("utf-8").split("\0") if p]


def exempt(path: str) -> bool:
    return path in EXEMPT_FILES or any(path.startswith(p) for p in EXEMPT_PREFIXES)


def is_source(path: str) -> bool:
    return Path(path).suffix not in DATA_SUFFIXES and Path(path).name not in DATA_NAMES


def head(path: str) -> str:
    with open(ROOT / path, encoding="utf-8", errors="replace") as fh:
        return "".join(line for _, line in zip(range(HEADER_LINES), fh))


class LicenceStatement(unittest.TestCase):
    def test_license_file_is_the_gpl_3_text(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("GNU GENERAL PUBLIC LICENSE\n", text)
        self.assertIn("Version 3, 29 June 2007", text)
        self.assertIn("END OF TERMS AND CONDITIONS", text)
        self.assertTrue(text.lstrip().startswith("GNU GENERAL PUBLIC LICENSE"), "the file must open with the GPL")

    def test_every_source_file_states_the_project_licence(self):
        missing = []
        for path in tracked():
            if is_source(path) and not exempt(path):
                top = head(path)
                if SPDX not in top or COPYRIGHT not in top:
                    missing.append(path)
        self.assertEqual(missing, [], "files without the SPDX and copyright lines in their first lines")

    def test_no_file_outside_third_party_names_another_licence_identifier(self):
        other = []
        for path in tracked():
            if exempt(path) or not is_source(path):
                continue
            for line in head(path).splitlines():
                m = re.search(r"SPDX-License-Identifier:\s*(\S.*?)\s*(\*/|-->)?\s*$", line)
                if m and m.group(1) != "GPL-3.0-or-later":
                    other.append((path, m.group(1)))
        self.assertEqual(other, [])

    def test_exemptions_exist_and_are_still_needed(self):
        files = set(tracked())
        for path in EXEMPT_FILES:
            self.assertIn(path, files, f"exempt file {path} is gone: drop its entry")
            self.assertNotIn(SPDX, head(path), f"{path} now carries the project header: drop its exemption")
        for prefix in EXEMPT_PREFIXES:
            self.assertTrue(any(p.startswith(prefix) for p in files), f"exempt folder {prefix} is empty")

    def document(self, doc: str) -> str:
        """The text of a document that speaks about the project's own licence (the notices: before their table)."""
        text = (ROOT / doc).read_text(encoding="utf-8")
        return text.split("\n| File |", 1)[0] if doc == "THIRD_PARTY_NOTICES.md" else text

    def test_each_document_states_the_exact_licence_sentence(self):
        sentences = {
            "README.md": "oot2-native is licensed GPL-3.0-or-later",
            "CONTRIBUTING.md": "you license it under GPL-3.0-or-later",
            "THIRD_PARTY_NOTICES.md": "oot2-native is licensed GPL-3.0-or-later (see [LICENSE](LICENSE))",
            ".github/PULL_REQUEST_TEMPLATE.md": "under GPL-3.0-or-later",
        }
        self.assertEqual(set(sentences), set(DOCUMENTS))
        for doc, sentence in sentences.items():
            self.assertIn(sentence, " ".join(self.document(doc).split()), doc)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        section = readme.split("## Licence", 1)[1].split("\n## ", 1)[0]
        self.assertIn("[LICENSE](LICENSE)", section)
        self.assertIn("THIRD_PARTY_NOTICES.md", section)

    def test_no_document_licenses_the_project_under_anything_else(self):
        for doc in DOCUMENTS:
            text = " ".join(self.document(doc).split())
            for m in STATEMENT.finditer(text):
                word = (m.group(1) or m.group(2)).rstrip(".,;:)")
                self.assertEqual(word, "GPL-3.0-or-later", f"{doc}: {m.group(0)!r}")
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if OTHER_LICENCES.search(sentence):
                    self.assertTrue(any(w in sentence.lower() for w in THIRD_PARTY_WORDS),
                                    f"{doc} names another licence outside a sentence about other projects: {sentence!r}")

    def test_every_licence_text_is_listed_and_every_listed_text_exists(self):
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        listed = set(re.findall(r"licenses/([A-Za-z0-9._-]+\.txt)", notices))
        present = {Path(p).name for p in tracked() if p.startswith("licenses/")}
        self.assertEqual(listed - present, set(), "licence texts the notices link but the repository lacks")
        self.assertEqual(present - listed, set(), "licence texts no row of the notices links")

    def test_every_vendored_component_ships_its_licence_file(self):
        files = set(tracked())
        for licence in ("xlate/third_party/ryujinx/LICENSE.txt", "xlate/third_party/ryujinx/src/Spv.Generator/LICENSE"):
            self.assertIn(licence, files)
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn("xlate/third_party/ryujinx/", notices)


if __name__ == "__main__":
    unittest.main()
