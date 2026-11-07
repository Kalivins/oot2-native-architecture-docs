# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/check_content.py.

Every rule has a red fixture (must be caught) and a green counterpart (must
pass). All fixtures are synthetic and built in a temporary directory: magic
bytes are assembled here, never read from real game files.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_content", ROOT / "tools" / "check_content.py")
cc = importlib.util.module_from_spec(_spec)
sys.modules["check_content"] = cc
_spec.loader.exec_module(cc)

TEXT = b"plain text\n"
HEX16 = "0123456789abcdef"


def run_main(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cc.main(argv)
    return code, out.getvalue()


class PathsCase(unittest.TestCase):
    """Rule tests: write fixtures to a temporary directory, check it with --paths."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cc-paths-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def check(self, files: dict[str, bytes]) -> tuple[int, str]:
        shutil.rmtree(self.tmp)
        self.tmp.mkdir()
        for rel, data in files.items():
            path = self.tmp / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return run_main(["--paths", str(self.tmp)])

    def red(self, files: dict[str, bytes], rule: str):
        code, out = self.check(files)
        self.assertEqual(code, 1, out)
        self.assertIn(f"[{rule}]", out)

    def green(self, files: dict[str, bytes]):
        code, out = self.check(files)
        self.assertEqual(code, 0, out)


class PathRules(PathsCase):
    def test_game_suffixes(self):
        for suffix in (".nso", ".nro", ".nca", ".nsp", ".xci", ".nup", ".bntx", ".bfsha", ".sarc",
                       ".szs", ".zs", ".byml", ".bwav", ".bars", ".zst", ".bin", ".keys", ".zip"):
            with self.subTest(suffix=suffix):
                self.red({f"x/data{suffix}": TEXT}, "blocked-suffix")
        self.green({"x/data.txt": TEXT, "x/data.md": TEXT, "x/data.py": TEXT})

    def test_names(self):
        for name in ("main", "rtld", "sdk", "subsdk0", "main.npdm", "prod." + "keys",
                     "title." + "keys", "header" + "_key"):
            with self.subTest(name=name):
                self.red({f"exe/{name}": TEXT}, "blocked-name")
        self.green({"exe/main.c": TEXT, "exe/sdk_notes.txt": TEXT, "exe/rtld.md": TEXT})

    def test_directories(self):
        for directory in sorted(cc.BLOCKED_DIRS) + ["ExeFS", "Generated", "cmake-build-debug"]:
            with self.subTest(directory=directory):
                self.red({f"a/{directory}/notes.txt": TEXT}, "blocked-dir")
                self.red({f"{directory}/deep/er/f.c": TEXT}, "blocked-dir")
        self.green({"a/dump_tool/notes.txt": TEXT, "a/romfs-format/notes.txt": TEXT,
                    "a/builder/notes.txt": TEXT, "a/output/notes.txt": TEXT})

    # Directory patterns of .gitignore that are tooling, not game material.
    TOOLING_DIRS = {"logs", ".cache", "__pycache__", ".venv", "venv", ".pytest_cache",
                    ".mypy_cache", ".ruff_cache", ".vs", ".idea", ".vscode"}

    @unittest.skipUnless((ROOT / ".gitignore").exists(), "needs the repository's .gitignore")
    def test_guard_and_gitignore_agree_on_directories(self):
        patterns = [ln.strip() for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
                    if ln.strip().endswith("/") and not ln.startswith("#")]
        names = {ln.rstrip("/") for ln in patterns}
        self.assertIn("generated", names)
        for name in sorted(names):
            with self.subTest(gitignored=name):
                covered = (name in cc.BLOCKED_DIRS or name in self.TOOLING_DIRS
                           or (name.endswith("*") and name[:-1] in cc.BLOCKED_DIR_PREFIXES))
                self.assertTrue(covered, f"'{name}/' is ignored but neither blocked nor listed as tooling")
        for name in sorted(cc.BLOCKED_DIRS):
            with self.subTest(blocked=name):
                self.assertIn(name, names, f"'{name}' is blocked but missing from .gitignore")

    def test_generated_code_heuristics(self):
        big = b"int x;\n" * (cc.MAX_SOURCE_BYTES // 7 + 10)
        self.assertLess(len(big), cc.MAX_BYTES)
        self.red({"src/big.c": big}, "generated-code")
        self.green({"src/big.txt": big, "src/small.c": b"int x;\n" * 1000})
        names = "".join(f"void sub_71{n:08x}(void);\n" for n in range(cc.ADDRESS_NAMES_LIMIT)).encode()
        self.red({"src/out.cpp": names}, "generated-code")
        self.green({"notes/functions.md": names, "src/out.cpp": names[: names.rindex(b"void sub")]})
        with mock.patch.object(cc, "ALLOW_BLOBS", {"src/big.c"}):
            self.green({"src/big.c": big})

    def test_lifter_names_from_the_project_base(self):
        """The lifter names functions f_<address> from 0x70_0000_0000, not from the Switch 1 base."""
        limit = cc.ADDRESS_NAMES_LIMIT
        lifted = "".join(f"void f_70{n:08x}(Cpu *c);\n" for n in range(limit)).encode()
        self.red({"src/excerpt.c": lifted}, "generated-code")
        mixed = "".join(f"void f_7{n % 2}{n:08x}(Cpu *c);\n" for n in range(limit)).encode()
        self.red({"src/excerpt.c": mixed}, "generated-code")  # both bases count toward one total
        self.green({"src/excerpt.c": lifted[: lifted.rindex(b"void f_")]})  # one short of the limit
        short = "".join(f"void f_70{n:07x}(Cpu *c);\n" for n in range(limit)).encode()
        self.green({"src/excerpt.c": short, "notes/functions.md": lifted})  # 9 digits; not a source file

    def test_media_suffixes(self):
        for suffix in (".png", ".jpg", ".webp", ".gif", ".dds", ".ktx", ".ico", ".wav", ".ogg", ".mp4"):
            with self.subTest(suffix=suffix):
                self.red({f"a/pic{suffix}": TEXT}, "media-suffix")
        self.green({"a/pic.svg": b"<svg/>\n", "a/pic.txt": TEXT})

    def test_size_limit(self):
        line = b"just some words on a line\n"
        big = line * (cc.MAX_BYTES // len(line) + 10)
        self.red({"a/big.txt": big}, "size")
        self.green({"a/ok.txt": line * 1000})

    def test_binary_content(self):
        self.red({"a/tool.dat": b"abc\0def"}, "binary")
        self.red({"a/tool.dat": b"caf\xe9 not utf-8 \xff\xfe"}, "binary")
        self.green({"a/tool.dat": "café is utf-8\n".encode()})

    def test_generated_marker(self):
        marker = b"GENERATED_FROM_" + b"GAME_CODE"
        self.red({"a/out.c": b"// " + marker + b"\nint x;\n"}, "generated-marker")
        self.green({"a/out.c": b"// written by hand\nint x;\n"})


class MagicRules(PathsCase):
    PAD = b"x" * 64

    def fixtures(self) -> dict[str, bytes]:
        pad = self.PAD
        return {
            "NSO0": b"NSO0" + pad,
            "NRO0": b"\0" * 0x10 + b"NRO0" + pad,
            "MOD0 header area": b"\0" * 0x80 + b"MOD0" + pad,
            "NCA after 0x200": b"\0" * 0x200 + b"NCA3" + pad,
            "PFS0": b"PFS0" + pad,
            "HFS0": b"HFS0" + pad,
            "BNTX": b"BNTX" + pad,
            "BNSH": b"BNSH" + pad,
            "BFSHA": b"BFSHA" + pad,
            "SARC": b"SARC" + pad,
            "Yaz0": b"Yaz0" + pad,
            "zstd frame": bytes([0x28, 0xB5, 0x2F, 0xFD]) + pad,
            "zstd skippable": bytes([0x55, 0x2A, 0x4D, 0x18]) + pad,
            "ZBIC": b"ZBIC" + pad,
            "BYML little": b"YB\x07\x00" + pad,
            "BYML big": b"BY\0\x04" + pad,
            "BFRES": b"FRES" + pad,
            "BWAV": b"BWAV" + pad,
            "BARS": b"BARS" + pad,
            "ASTC": bytes([0x13, 0xAB, 0xA1, 0x5C]) + pad,
            "DDS": b"DDS " + pad,
            "PNG": b"\x89PNG\r\n\x1a\n" + pad,
            "JPEG": bytes([0xFF, 0xD8, 0xFF, 0xE0]) + pad,
            "WebP": b"RIFF\x10\0\0\0WEBP" + pad,
            "GIF": b"GIF89a" + pad,
            "KTX": b"\xabKTX 11\xbb\r\n\x1a\n" + pad,
            "KTX2": b"\xabKTX 20\xbb\r\n\x1a\n" + pad,
        }

    def test_each_signature_is_caught_under_an_innocent_name(self):
        for label, data in self.fixtures().items():
            with self.subTest(magic=label):
                self.red({"notes/readme.txt": data}, "magic")

    def test_same_words_in_ordinary_text_pass(self):
        for word in ("NSO0", "NRO0", "MOD0", "PFS0", "HFS0", "BNTX", "BNSH", "BFSHA", "SARC", "Yaz0",
                     "ZBIC", "FRES", "BWAV", "BARS", "DDS ", "GIF89a", "YB", "NCA3", "HEAD"):
            with self.subTest(word=word):
                body = f"The format {word} is described in the notes.\n".encode()
                self.green({"notes/readme.txt": body})
        code, out = self.check({"notes/readme.txt": b"\0" * 0x80 + b"nothing here" + self.PAD})
        self.assertEqual(code, 1, out)  # NUL bytes alone are the binary rule, not a signature
        self.assertIn("[binary]", out)
        self.assertNotIn("[magic]", out)

    def test_signature_inside_a_text_file_header_area_passes(self):
        self.green({"notes/readme.txt": b"# MOD0 and NCA3 appear in prose\n" * 40})

    def test_allowlist_exempts_media_never_game_formats(self):
        png = self.fixtures()["PNG"] + b"\0"
        self.red({"logo.png": png}, "magic")
        with mock.patch.object(cc, "ALLOW_BINARY", {"logo.png"}):
            self.green({"logo.png": png})
            self.red({"logo.png": b"NSO0" + b"\0" * 64}, "magic")
        with mock.patch.object(cc, "ALLOW_BINARY", {"blob.dat"}):
            self.green({"blob.dat": b"abc\0def"})
        self.red({"blob.dat": b"abc\0def"}, "binary")

    def test_the_allowlist_holds_the_av1_test_clips_and_only_them(self):
        """Ticket 98: three short AV1 clips of generated pictures (tests/fixtures/av1) are the allowlist's first entries.
        Each passes under its own name and is a binary file under any other; a game signature in a file of that name is
        still refused."""
        root = Path(__file__).resolve().parent.parent
        self.assertEqual(sorted(cc.ALLOW_BINARY), [f"tests/fixtures/av1/{n}.ivf" for n in
                                                   ("clip_1920x1080", "clip_200x126", "clip_201x125")])
        for name in sorted(cc.ALLOW_BINARY):
            with self.subTest(name=name):
                data = (root / name).read_bytes()
                self.assertTrue(data.startswith(b"DKIF"), name)   # an IVF file
                self.assertLess(len(data), 64 * 1024, name)
                self.green({name: data})
                self.red({name.replace("tests/fixtures/av1/", "tests/fixtures/other/"): data}, "binary")
                self.red({name: b"NSO0" + b"\0" * 64}, "magic")


class KeyRules(PathsCase):
    def test_key_assignment(self):
        for name in ("aes_kek_generation_source", "titlekek_source_00", "master_key_00", "header" + "_key"):
            for digits in (32, 64):
                with self.subTest(name=name, digits=digits):
                    line = f"{name} = {HEX16 * 4}\n"[: len(name) + 3 + digits] + "\n"
                    self.red({"a/notes.txt": line.encode()}, "key-line")
        self.red({"a/notes.txt": f'  "title_key": "{HEX16 * 2}",\n'.encode()}, "key-line")

    def test_hashes_and_other_lengths_pass(self):
        sha = HEX16 * 4
        self.green({"a/notes.txt": f"sha256 = {sha}\nsha256: {sha}\nmd5 = {sha[:32]}\n".encode()})
        self.green({"a/notes.txt": f"build_id = {sha[:32]}\nimage_hash = {sha}\n".encode()})
        ids = f'{{\n  "main": "{sha}",\n  "main.npdm": "{sha}",\n  "rtld": "{sha[:32]}"\n}}\n'
        self.green({"a/ids.json": ids.encode()})  # ExeFS module names label build ids
        table = f'{{"main.patched.bin": "{sha}"}}\n'.encode()
        self.red({"a/hashes.json": table}, "key-line")  # an arbitrary label looks like a key...
        with mock.patch.object(cc, "ALLOW_BLOBS", {"a/hashes.json"}):
            self.green({"a/hashes.json": table})  # ...unless a reviewer allowlisted the file
            self.red({"a/hashes.json": b"-----BEGIN " + b"EC PRIVATE KEY-----\n"}, "private-key")
        self.green({"a/notes.txt": f"master_key = {HEX16 * 2}ab\nmaster_key = {HEX16 * 2}"[:-1].encode() + b"\n"})
        self.green({"a/notes.txt": b"the key = a secret that is not written here\n# key notes\n"})

    def test_key_syntaxes_that_used_to_slip_through(self):
        key = HEX16 * 2
        cases = {
            "crlf": f"title_key = {key}\r\n",
            "crlf in a block": f"x = 1\r\nmaster_key_00 = {key}\r\ny = 2\r\n",
            "trailing hash comment": f"master_key_01 = {key}  # from the console\n",
            "trailing slash comment": f"master_key_02 = {key} // dumped\n",
            "define": f"#define TITLEKEK 0x{key}\n",
            "define with comment": f"#define TITLEKEK 0x{key} // kek\n",
            "quoted c string": f'const char *title_key = "{key}";\n',
            "c array declaration": f'static const char aes_key[] = "{key}";\n',
            "spaces only": f"aes_key_source {key}\n",
            "0x prefix": f"master_key_03 = 0x{key}\n",
            "64 digits, crlf": f"title_kek_source: {HEX16 * 4}\r\n",
        }
        for label, line in cases.items():
            with self.subTest(case=label):
                self.red({"a/notes.txt": line.encode()}, "key-line")

    def test_hash_like_lines_stay_green(self):
        sha, key = HEX16 * 4, HEX16 * 2
        lines = [
            f"{sha}  some_file.bin", f"sha256sum: {sha}", f"Build ID: {key}", f"Build ID {key}",
            f"nso_build_id = {key}", f"UUID = {key}", f"uuid: {key}", f"title_id = {key}",
            f"shader_7: 0x{HEX16}", f"| sub_7100001234 | 0x{HEX16} |",
            f"#define BUILD_HASH 0x{sha}", f"const char *build_id = \"{key}\";",
        ]
        for line in lines:
            with self.subTest(line=line):
                self.green({"a/notes.txt": (line + "\n").encode()})
        self.green({"a/notes.txt": ("the id " + key + "\n").encode()})  # spaces only, not a key name

    def test_private_key_block(self):
        self.red({"a/id.txt": ("-----BEGIN " + "RSA PRIVATE KEY-----\nabc\n").encode()}, "private-key")
        self.green({"a/id.txt": ("-----BEGIN " + "PUBLIC KEY-----\nabc\n").encode()})


class BlobRules(PathsCase):
    def test_contiguous_hex(self):
        run = HEX16 * 12  # 192 digits
        self.red({"a/t.txt": f"data = {run}\n".encode()}, "hex-blob")
        self.green({"a/t.txt": f"data = {run[:191]}\n".encode()})

    def test_contiguous_base64(self):
        raw = base64.b64encode(bytes(range(256))).decode()
        self.assertGreater(len(raw), 300)
        self.red({"a/t.txt": f"blob = {raw}\n".encode()}, "base64-blob")
        self.green({"a/t.txt": f"blob = {raw[:255]}\n".encode()})

    def test_hex_dump_lines(self):
        def table(rows: int) -> bytes:
            lines = [", ".join(f"0x{(r * 31 + c * 7) % 256 | 0x0a:02x}" for c in range(16)) + "," for r in range(rows)]
            return ("\n".join(lines) + "\n").encode()

        self.red({"a/t.c": table(cc.HEX_TABLE_LINES)}, "hex-blob")
        self.red({"a/t.c": table(cc.HEX_DENSE_LINES)}, "hex-blob")  # the whole file is the table
        self.green({"a/t.c": table(cc.HEX_DENSE_LINES - 1)})
        mixed = table(cc.HEX_TABLE_LINES - 1) + b"x = 1;\n" * 60  # long, but not a streak and not dense
        self.green({"a/t.c": mixed})
        decimal = "\n".join(" ".join(str(10 + n) for n in range(40)) for _ in range(30)) + "\n"
        self.green({"a/t.txt": decimal.encode()})

    def test_wrapped_base64_lines(self):
        raw = base64.b64encode(bytes(range(256)) * 8).decode()
        for width in (64, 44, 32, 26):
            lines = [raw[i:i + width] for i in range(0, len(raw) - width, width)]
            self.assertGreater(len(lines), cc.B64_BLOCK_LINES)
            with self.subTest(width=width):
                self.red({"a/t.txt": ("\n".join(lines[: cc.B64_BLOCK_LINES]) + "\n").encode()}, "base64-blob")
                self.green({"a/t.txt": ("\n".join(lines[: cc.B64_BLOCK_LINES - 1]) + "\n").encode()})
        paths = "\n".join(f"src/renderer/vulkan/device_{n}" for n in range(30)) + "\n"
        words = "\n".join("abcdefghijklmnopqrstuvwxyz" for _ in range(30)) + "\n"
        self.green({"a/paths.txt": paths.encode(), "a/words.txt": words.encode()})

    @staticmethod
    def sample(n: int) -> bytes:
        return bytes((i * 73 + 11) % 256 for i in range(n))

    def test_common_dump_formats(self):
        data = self.sample(16 * 40)

        def chunks(width):
            return [data[i:i + width] for i in range(0, len(data), width)]

        xxd_i = "\n".join("  " + ", ".join(f"0x{b:02x}" for b in c) + "," for c in chunks(12)) + "\n"
        hexdump_c = "\n".join(
            f"{n * 16:08x}  " + " ".join(f"{b:02x}" for b in c[:8]) + "  " + " ".join(f"{b:02x}" for b in c[8:])
            + "  |" + "".join(chr(b) if 32 <= b < 127 else "." for b in c) + "|"
            for n, c in enumerate(chunks(16))) + "\n"
        xxd_plain = "\n".join(
            f"{n * 16:08x}: " + " ".join(c[i:i + 2].hex() for i in range(0, 16, 2)) + "  "
            + "".join(chr(b) if 32 <= b < 127 else "." for b in c) for n, c in enumerate(chunks(16))) + "\n"
        xxd_p = "\n".join(c.hex() for c in chunks(30)) + "\n"
        fifteen = "\n".join(" ".join(f"{b:02x}" for b in c) for c in chunks(15)) + "\n"
        escaped = "\n".join('  "' + "".join(f"\\x{b:02x}" for b in c) + '"' for c in chunks(16)) + "\n"
        for label, text in {"xxd -i": xxd_i, "hexdump -C": hexdump_c, "xxd": xxd_plain, "xxd -p": xxd_p,
                            "15 bytes a line": fifteen, "escaped string": escaped}.items():
            with self.subTest(format=label):
                self.red({"a/t.txt": text.encode()}, "hex-blob")

    def test_many_short_initialisers_and_decimal_arrays(self):
        def pairs(n: int) -> bytes:
            return "".join("{ 0x%02x, 0x%02x },\n" % ((i * 7) % 256, (i * 13) % 256) for i in range(n)).encode()

        limit_lines = cc.BYTE_TOKEN_LIMIT // 2
        self.red({"a/t.c": pairs(limit_lines + 1)}, "hex-blob")
        self.green({"a/t.c": pairs(limit_lines)})
        numbers = [(i * 37 + 5) % 256 for i in range(300)]
        self.red({"a/t.c": ("static const unsigned char t[] = {" + ", ".join(map(str, numbers)) + "};\n").encode()},
                 "hex-blob")
        rows = ",\n".join("    " + ", ".join(map(str, numbers[i:i + 16])) for i in range(0, 300, 16))
        self.red({"a/t.c": rows.encode()}, "hex-blob")
        self.green({"a/t.c": ", ".join(map(str, numbers[:255])).encode()})  # one short of 256
        self.green({"a/t.c": ", ".join(["0"] * 400).encode()})  # a zero-filled initialiser
        self.green({"a/t.c": ", ".join(str(1000 + n) for n in range(300)).encode()})  # not byte values

    def test_near_misses_stay_fast(self):
        """A pattern that backtracks exponentially turns the pre-commit hook into a hang.

        It runs in a subprocess with a timeout, so a regression fails instead of hanging the suite.
        """
        program = (
            "import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('cc', sys.argv[1])\n"
            "cc = importlib.util.module_from_spec(spec)\n"
            "sys.modules['cc'] = cc\n"
            "spec.loader.exec_module(cc)\n"
            "near = ', '.join(str((i * 37 + 5) % 256) for i in range(255))\n"
            "shapes = [near, near.replace(', ', ',\\n    '), '1 ' * 3000, 'a' * 100000, '0x12, ' * 5000,\n"
            "          'ab ' * 20000, ('x = ' + 'ab' * 15 + '\\n') * 3000, 'name ' * 2000,\n"
            "          'const char *const char *' * 400, 'f' + '_700000012' * 10000]\n"
            "for text in shapes:\n"
            "    cc.blob_findings(text); cc.key_findings(text); cc.source_findings('a/t.c', len(text), text)\n"
            "print('done')\n"
        )
        try:
            done = subprocess.run([sys.executable, "-I", "-c", program, str(ROOT / "tools" / "check_content.py")],
                                  capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            self.fail("a rule backtracks catastrophically on a near miss (30 s timeout)")
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "done"), done.stderr)

    def test_labelled_hash_lists_and_crypto_tables(self):
        crc = []
        for n in range(256):
            value = n
            for _ in range(8):
                value = (value >> 1) ^ (0xEDB88320 if value & 1 else 0)
            crc.append(value)
        table = "static const uint32_t crc_table[256] = {\n" + "".join(
            "    " + ", ".join(f"0x{v:08x}" for v in crc[i:i + 4]) + ",\n" for i in range(0, 256, 4)) + "};\n"
        self.red({"src/crc.c": table.encode()}, "hex-blob")  # a known false positive...
        with mock.patch.object(cc, "ALLOW_BLOBS", {"src/crc.c"}):
            self.green({"src/crc.c": table.encode()})  # ...with a documented way out
        sums = "".join(f"{(HEX16 * 4)[:-2]}{n:02x}  file_{n}.txt\n" for n in range(40))
        table_md = "| function | shader hash |\n| --- | --- |\n" + "".join(
            f"| sub_71{n:08x} | 0x{n:016x} |\n" for n in range(60))
        self.green({"a/SHA256SUMS": sums.encode(), "a/functions.md": table_md.encode()})

    def test_blob_allowlist(self):
        data = f"data = {HEX16 * 12}\n".encode()
        self.red({"a/t.txt": data}, "hex-blob")
        with mock.patch.object(cc, "ALLOW_BLOBS", {"a/t.txt"}):
            self.green({"a/t.txt": data})


GIT_AVAILABLE = shutil.which("git") is not None


def git_in(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return done.stdout.strip()


@unittest.skipUnless(GIT_AVAILABLE, "git is required")
class GitModes(unittest.TestCase):
    """Mode tests: throwaway repositories, never this one."""

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp(prefix="cc-git-"))
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        git_in(self.repo, "init", "-q", "-b", "main")
        git_in(self.repo, "config", "user.name", "Tester")
        git_in(self.repo, "config", "user.email", "tester@users.noreply.github.com")
        git_in(self.repo, "config", "core.hooksPath", str(self.repo / ".no-hooks"))
        git_in(self.repo, "config", "commit.gpgsign", "false")

    def write(self, rel: str, data: bytes):
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def run_cc(self, *argv: str) -> tuple[int, str]:
        done = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "check_content.py"), *argv],
                              cwd=self.repo, capture_output=True, text=True)
        return done.returncode, done.stdout

    def commit(self, message: str) -> str:
        git_in(self.repo, "add", "-A")
        git_in(self.repo, "commit", "-q", "-m", message)
        return git_in(self.repo, "rev-parse", "HEAD")

    BAD = b"NSO0" + b"\0" * 32

    def stage_entry(self, mode: str, rel: str, data: bytes):
        """Put a blob in the index with an explicit mode, so a symlink needs no operating system support."""
        done = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=self.repo, input=data,
                              capture_output=True, check=True)
        git_in(self.repo, "update-index", "--add", "--cacheinfo", f"{mode},{done.stdout.decode().strip()},{rel}")

    def test_type_change_is_checked_in_staged_and_range(self):
        self.stage_entry("120000", "tc.txt", b"target.txt")
        git_in(self.repo, "commit", "-q", "-m", "a symlink")
        base = git_in(self.repo, "rev-parse", "HEAD")
        self.stage_entry("100644", "tc.txt", self.BAD)  # the symlink becomes a regular file
        self.assertTrue(git_in(self.repo, "diff", "--cached", "--name-status").startswith("T"))
        code, out = self.run_cc("--staged")
        self.assertEqual(code, 1, out)
        self.assertIn("tc.txt: [magic]", out)
        git_in(self.repo, "commit", "-q", "-m", "now a file")
        code, out = self.run_cc("--range", f"{base}..HEAD")
        self.assertEqual(code, 1, out)
        self.assertIn("[magic]", out)

    def test_force_added_ignored_directories_are_refused(self):
        self.write(".gitignore", b"generated/\nbuild/\ncaptures/\n")
        self.write("ok.txt", TEXT)
        for rel in ("generated/f.c", "build/a.c", "captures/a.txt", "extracted/a.txt", "shader-cache/a.txt"):
            self.write(rel, TEXT)
            git_in(self.repo, "add", "-f", rel)
        code, out = self.run_cc("--staged")
        self.assertEqual(code, 1, out)
        self.assertEqual(out.count("[blocked-dir]"), 5, out)

    def test_staged_red_and_green(self):
        self.write("ok.txt", TEXT)
        git_in(self.repo, "add", "ok.txt")
        self.assertEqual(self.run_cc("--staged")[0], 0)
        self.write("sample.txt", self.BAD)
        git_in(self.repo, "add", "sample.txt")
        code, out = self.run_cc("--staged")
        self.assertEqual(code, 1, out)
        self.assertIn("sample.txt: [magic]", out)

    def test_staged_reads_the_index_not_the_working_tree(self):
        self.write("a.txt", self.BAD)
        git_in(self.repo, "add", "a.txt")
        self.write("a.txt", TEXT)  # cleaned on disk after staging
        self.assertEqual(self.run_cc("--staged")[0], 1)
        git_in(self.repo, "add", "a.txt")
        self.assertEqual(self.run_cc("--staged")[0], 0)
        self.write("a.txt", self.BAD)  # dirty on disk only
        self.assertEqual(self.run_cc("--staged")[0], 0)

    def test_range_checks_every_commit_not_only_the_tip(self):
        self.write("a.txt", TEXT)
        base = self.commit("base")
        self.write("keys.txt", ("title_key = " + HEX16 * 2 + "\n").encode())
        bad = self.commit("add a key line")
        (self.repo / "keys.txt").unlink()
        tip = self.commit("remove it again")
        code, out = self.run_cc("--range", f"{base}..{tip}")
        self.assertEqual(code, 1, out)
        self.assertIn("[key-line]", out)
        self.assertIn(bad[:10], out)
        self.assertEqual(self.run_cc("--all")[0], 0)  # the tip alone is clean
        self.assertEqual(self.run_cc("--range", f"{bad}..{tip}")[0], 0)  # the bad commit is outside
        self.assertEqual(self.run_cc("--range", tip)[0], 1)  # a single rev checks all history

    def test_range_green_on_clean_history(self):
        self.write("a.txt", TEXT)
        base = self.commit("one")
        self.write("b.txt", TEXT)
        tip = self.commit("two")
        code, out = self.run_cc("--range", f"{base}..{tip}")
        self.assertEqual((code, "OK" in out), (0, True), out)

    def test_all_sees_untracked_but_not_ignored(self):
        self.write(".gitignore", b"ignored/\n")
        self.write("a.txt", TEXT)
        self.commit("base")
        self.write("ignored/x.txt", self.BAD)
        self.assertEqual(self.run_cc("--all")[0], 0)
        self.write("untracked.txt", self.BAD)
        self.assertEqual(self.run_cc("--all")[0], 1)

    def test_failures_to_run_are_not_passes(self):
        self.write("a.txt", TEXT)
        self.commit("base")
        self.assertEqual(self.run_cc("--range", "no-such-rev..HEAD")[0], 2)
        outside = Path(tempfile.mkdtemp(prefix="cc-nogit-"))
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        done = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "check_content.py"), "--staged"],
                              cwd=outside, capture_output=True, text=True)
        self.assertEqual(done.returncode, 2, done.stdout)
        self.assertEqual(run_main(["--paths", str(outside / "missing")])[0], 2)

    def test_pre_commit_hook_blocks_a_bad_commit(self):
        if shutil.which("sh") is None:
            self.skipTest("sh is required to run the hook")
        shutil.copytree(ROOT / ".githooks", self.repo / ".githooks")
        (self.repo / "tools").mkdir()
        shutil.copy(ROOT / "tools" / "check_content.py", self.repo / "tools" / "check_content.py")
        (self.repo / ".githooks" / "pre-commit").chmod(0o755)  # git skips a hook without the exec bit
        git_in(self.repo, "config", "core.hooksPath", ".githooks")
        self.write("ok.txt", TEXT)
        git_in(self.repo, "add", "ok.txt")
        git_in(self.repo, "commit", "-q", "-m", "clean commit passes the hook")
        self.write("sample.txt", self.BAD)
        git_in(self.repo, "add", "sample.txt")
        blocked = subprocess.run(["git", "commit", "-q", "-m", "must be refused"], cwd=self.repo,
                                 capture_output=True, text=True)
        self.assertNotEqual(blocked.returncode, 0, blocked.stdout + blocked.stderr)
        self.assertIn("[magic]", blocked.stdout + blocked.stderr)


@unittest.skipUnless(GIT_AVAILABLE and (ROOT / ".git").exists(), "needs a git checkout of the repository")
class RepositoryItself(unittest.TestCase):
    def test_all_passes_on_this_repository(self):
        done = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "check_content.py"), "--all"],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK", done.stdout)


if __name__ == "__main__":
    unittest.main()
