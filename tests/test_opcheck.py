# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/shaders/opcheck.py, the corpus comparison of two translations.

A pass folder is faked here: a results.tsv, modules assembled from words (see test_spvstats.py) and a
description each. The checks that guard the report: the sizes are summed per stage and kind, an
atomic missing from the candidate shows as lost for the right kind, a read that moved shows as moved,
a job the candidate fails and the reference does not shows as failed, and a candidate equal to the
reference reports nothing.

Run: python -m unittest discover -s tests -p test_opcheck.py -v
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "shaders"))

import opcheck  # noqa: E402
from test_spvstats import Asm, Func, Types  # noqa: E402


def module(atomic: bool, read_first: bool = True) -> bytes:
    a = Asm()
    t = Types(a)
    f = Func(a, t)
    if atomic:
        f.atomic_add(t.sb, t.scope)
    if read_first:
        x = f.load(t.wg)
        f.store(t.wg, t.scope)
    else:
        f.store(t.wg, t.scope)
        x = f.load(t.wg)
    f.store(t.wg, x).done()
    return a.module()


class Folder:
    """A pass folder with results.tsv, spv/<key>.spv and spv/<key>.res.json."""

    def __init__(self, root: Path, name: str, jobs: dict[str, tuple[str, bytes | None, bool]]):
        self.path = root / name
        (self.path / "spv").mkdir(parents=True)
        with open(self.path / "results.tsv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(("key", "status", "micros", "wall_ms", "spv_bytes", "message"))
            for key, (stage, data, raw) in jobs.items():
                w.writerow((key, "ok" if data else "fail", 0, 0, len(data or b""), "" if data else "translate: crash"))
                if data:
                    (self.path / "spv" / f"{key}.spv").write_bytes(data)
                    (self.path / "spv" / f"{key}.res.json").write_text(json.dumps({"raw_access": {"page_table": raw}}), encoding="utf-8")


class OpcheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.split = self.root / "split"
        self.split.mkdir()
        jobs = [{"key": k, "stage": s} for k, s in (("a", "ps"), ("b", "ps"), ("c", "cs"))]
        (self.split / "jobs.jsonl").write_text("\n".join(json.dumps(j) for j in jobs) + "\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def stats(self, folder: Folder, tensor: str | None = None):
        args = argparse.Namespace(split=self.split, work=folder.path.parent, pass_name=folder.path.name, workers=1, limit=None, tensor_keys=tensor)
        self.assertEqual(opcheck.cmd_stats(args), 0)
        return opcheck.read_stats(folder.path), opcheck.read_status(folder.path)

    def test_a_candidate_equal_to_the_reference_reports_nothing(self):
        jobs = {"a": ("ps", module(True), False), "b": ("ps", module(False), True), "c": ("cs", module(True), False)}
        ref, rs = self.stats(Folder(self.root, "ref", jobs))
        cand, cs = self.stats(Folder(self.root, "cand", jobs))
        s = opcheck.summarise(ref, cand, rs, cs, set())
        self.assertEqual((s["modules_compared"], s["lost"], s["failed_in_candidate_only"]), (3, {}, {}))
        self.assertEqual(s["moved_reads_lower_bound"], {"modules": {}, "reads": {}})
        self.assertEqual(s["size"]["all/all"]["bytes_ratio"], 1.0)

    def test_kinds_come_from_the_description_and_the_tensor_list(self):
        jobs = {"a": ("ps", module(True), False), "b": ("ps", module(False), True), "c": ("cs", module(True), False)}
        tensor = self.root / "tensor.txt"
        tensor.write_text("c\n", encoding="utf-8")
        ref, _ = self.stats(Folder(self.root, "ref", jobs), str(tensor))
        self.assertEqual({k: v["kind"] for k, v in ref.items()}, {"a": "plain", "b": "raw", "c": "tensor"})

    def test_lost_moved_failed_and_size_are_reported(self):
        ref_jobs = {"a": ("ps", module(True), False), "b": ("ps", module(False, True), True), "c": ("cs", module(True), False)}
        cand_jobs = {"a": ("ps", module(False), False),               # lost its atomic
                     "b": ("ps", module(False, False), True),         # its read crossed a store
                     "c": ("cs", None, False)}                        # failed
        ref, rs = self.stats(Folder(self.root, "ref", ref_jobs))
        cand, cs = self.stats(Folder(self.root, "cand", cand_jobs))
        s = opcheck.summarise(ref, cand, rs, cs, {"c"})
        self.assertEqual(s["modules_compared"], 2)
        self.assertEqual(s["lost"]["atomics"], {"modules": {"plain": 1, "all": 1}, "operations": {"plain": 1, "all": 1}})
        self.assertEqual(s["moved_reads_lower_bound"], {"modules": {"raw": 1, "all": 1}, "reads": {"raw": 1, "all": 1}})
        self.assertEqual(s["comparison_unavailable"], {"plain": 1})     # the module that lost a write
        self.assertEqual(s["failed_in_candidate_only"], {"only_product100": 1})
        self.assertLess(s["size"]["ps/plain"]["bytes_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
