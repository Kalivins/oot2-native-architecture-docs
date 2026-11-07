# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the Vulkan back end on a real device (runtime/gpu, runtime/include/oot2/gpu_vk.h), through
tests/runtime/vk_test.c: the device check of every device, the device opened with every item of oot2/vk_required.h,
1,000 empty replayed submissions from two threads mixed with the back end's own, a fence wait that times out and a later
one that succeeds, finish, the device-loss report on an injected loss, and the backing map (oot2/backing.h): reads and
writes through the resolver into each kind of backing, pool_fini held by a submission in flight, and 4,660 pools made
and finalized with device and host memory left flat.

The program is built like tests/test_runtime_gpu.py's (build_program), at -O2. Without a Vulkan device, or without a
device that has every item (the laptop's GTX 1060), the tests that need one skip, unless OOT2_REQUIRE_GPU is set, in the
manner of OOT2_REQUIRE_XLATE for the translator's tests; continuous integration has no GPU and sets neither. The device
check itself runs wherever a loader and a device exist, and prints what each device lacks.

The submissions run with the validation layer and synchronization validation (OOT2_VK_VALIDATION=sync) and require no
validation error; the layer comes with the Vulkan SDK, which these tests need anyway. Third-party implicit layers are
kept out (case_env).

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXIT_FATAL, ROOT, build_program, case_env, clang_or_skip, fatal_lines, vulkan_or_skip

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_test.c"
LEVEL = "-O2"
EXIT_NO_DEVICE = 3


def require_gpu() -> bool:
    return bool(os.environ.get("OOT2_REQUIRE_GPU"))


class VkDeviceTest(unittest.TestCase):
    """vk_test.c on this machine's Vulkan devices."""

    @classmethod
    def setUpClass(cls) -> None:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-vk-", ignore_cleanup_errors=True)
        cls.exe = build_program(cc, include, glslang, TEST_SOURCE, Path(cls.tmp.name), levels=(LEVEL,))[LEVEL]
        done = cls.run_case("devices")
        cls.devices = done
        if done.returncode == EXIT_NO_DEVICE:
            why = f"no Vulkan device here ({done.stdout.strip()})"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but there is {why}")
            raise unittest.SkipTest(why)
        if done.returncode != 0:
            raise RuntimeError(f"the device check failed:\n{done.stdout}{done.stderr}")
        m = re.search(r"^usable (\d+)$", done.stdout, re.M)
        cls.usable = int(m.group(1)) if m else 0

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    @classmethod
    def run_case(cls, case: str, *args: str, env: dict[str, str] | None = None,
                 timeout: int = 120) -> subprocess.CompletedProcess:
        return subprocess.run([str(cls.exe), case, *args], capture_output=True, text=True, timeout=timeout,
                              env=case_env(env))

    def need_usable_device(self) -> None:
        if self.usable:
            return
        why = "no device here has every item the renderer requires (see test_the_device_check_lists_what_each_lacks)"
        if require_gpu():
            self.fail(f"OOT2_REQUIRE_GPU is set but {why}")
        self.skipTest(why)

    def assert_clean(self, done: subprocess.CompletedProcess, case: str) -> None:
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"ok {case}", done.stdout)
        self.assertEqual(fatal_lines(done.stderr), [])
        self.assertIn("validation errors 0 ", done.stdout)

    def test_the_device_check_lists_what_each_lacks(self) -> None:
        rows = [line.split("\t") for line in self.devices.stdout.splitlines() if line.startswith("device ")]
        self.assertTrue(rows, self.devices.stdout)
        for row in rows:
            have, total = (int(x) for x in row[3].split(" of "))
            self.assertEqual(have == total, len(row) == 4, row)   # a device lacks something exactly when it says what
        print("\n" + self.devices.stdout, end="")   # the record for the research note (the laptop's GTX 1060)

    def test_the_device_opens_with_every_item(self) -> None:
        self.need_usable_device()
        done = self.run_case("open", env={"OOT2_VK_VALIDATION": "sync"})
        self.assert_clean(done, "open")
        m = re.search(r"opened (.+), (\d+) of (\d+) items", done.stdout)
        self.assertTrue(m and m.group(2) == m.group(3), done.stdout)
        self.assertRegex(done.stderr, r"gfx: vulkan: .* device made with the \d+ items' features and \d+ extensions")
        print(f"\n{done.stdout}", end="")

    def test_submissions_from_two_threads_complete_in_order_with_their_effects_first(self) -> None:
        self.need_usable_device()
        done = self.run_case("submits", env={"OOT2_VK_VALIDATION": "sync"}, timeout=300)
        self.assert_clean(done, "submits")
        self.assertRegex(done.stdout, r"values 1100 \(replayed 1000, internal 100, effects run 100\), done 1100, "
                                      r"device counter 1100")
        print(f"\n{done.stdout}", end="")

    def test_a_fence_wait_times_out_and_a_later_one_succeeds_and_finish_leaves_nothing(self) -> None:
        self.need_usable_device()
        done = self.run_case("wait", env={"OOT2_VK_VALIDATION": "sync"})
        self.assert_clean(done, "wait")
        self.assertRegex(done.stdout, r"finish left values 21, counter 21")
        print(f"\n{done.stdout}", end="")

    def test_the_backing_map_reads_and_writes_each_kind_through_the_resolver(self) -> None:
        self.need_usable_device()
        done = self.run_case("backing", env={"OOT2_VK_VALIDATION": "sync"}, timeout=300)
        self.assert_clean(done, "backing")
        for line in ("an import over uncommitted pages is refused; a 4 KiB import of a committed page then succeeds",
                     "imported: the GPU read the CPU's words and the CPU read the GPU's, on touched and untouched pages",
                     "VRAM extent: written and read back",
                     "virtual over a physical run: through the window's extent and through an import of guest memory",
                     "an import across a 64 MiB window boundary (two reservation pieces): written and seen",
                     "the heap shrank under 4 imports: decommitted, and the imports retired at the next sync"):
            self.assertIn(line, done.stdout)
        print(f"\n{done.stdout}", end="")

    def test_pool_fini_with_a_submission_in_flight_returns_only_after_it_is_done(self) -> None:
        self.need_usable_device()
        done = self.run_case("backing_fini", env={"OOT2_VK_VALIDATION": "sync"})
        self.assert_clean(done, "backing_fini")
        self.assertIn("pool_fini held while the submission was in flight", done.stdout)
        print(f"\n{done.stdout}", end="")

    def test_4660_pools_made_and_finalized_leave_device_and_host_memory_flat(self) -> None:
        self.need_usable_device()
        done = self.run_case("backing_flat", env={"OOT2_VK_VALIDATION": "on"}, timeout=900)
        self.assert_clean(done, "backing_flat")
        self.assertEqual(len(re.findall(r"^round \d+: ", done.stdout, re.M)), 3, done.stdout)
        print(f"\n{done.stdout}", end="")

    def assert_loss_report(self, case: str, where: str) -> str:
        with tempfile.TemporaryDirectory(prefix="oot2-loss-") as d:
            done = self.run_case(case, d)
            self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
            self.assertNotIn("returned", done.stdout)
            fatal = fatal_lines(done.stderr)
            self.assertEqual(len(fatal), 1, done.stderr)
            m = re.fullmatch(rf"FATAL: gfx: vulkan: the device was lost \(VK_ERROR_DEVICE_LOST from {re.escape(where)}\);"
                             r" the report is (.+\.txt)", fatal[0])
            self.assertTrue(m, fatal[0])
            report = Path(m.group(1))
            self.assertEqual(report.parent.resolve(), (Path(d) / "logs").resolve())
            text = report.read_text(encoding="utf-8")
            self.assertIn(f"lost at: {where} returned VK_ERROR_DEVICE_LOST", text)
            self.assertRegex(text, r"device: .+ \(index \d+\), vendor 0x[0-9a-f]+, device 0x[0-9a-f]+, Vulkan 1\.")
            self.assertIn('fault: "injected by the test: no real fault"', text)
            self.assertIn("fault address 0: read invalid at 0x7f0012345678, precision 0x1000 "
                          "(in [0x7f0012345000, 0x7f0012346000))", text)
            self.assertIn("fault address 1: instruction pointer (fault) at 0x1234", text)
            self.assertIn('fault vendor record 0: "injected vendor record", code 0xdead, data 0xbeef', text)
            self.assertIn("checkpoints: the last 2 the queue reached", text)
            self.assertIn("checkpoint 0: stage 0x1, value 41, replayed submission, start of its commands", text)
            self.assertIn("checkpoint 1: stage 0x2000, value 40, internal submission, end of its commands", text)
            self.assertRegex(text, r"heap 0( \(device local\))?: \d+ MiB, usage \d+ MiB of a budget of \d+ MiB")
            binary = report.with_suffix(".bin")
            if "with the vendor's binary" in done.stderr:   # the device offers the vendor's binary
                self.assertIn("fault vendor binary: 200 bytes, saved beside this report", text)
                data = binary.read_bytes()
                self.assertEqual(len(data), 200)
                self.assertEqual(data[100:], bytes(i ^ 0x5A for i in range(100, 200)))
                self.assertIn("fault vendor binary header: version one", text)
            else:
                self.assertIn("fault vendor binary: none (the device offers none)", text)
                self.assertFalse(binary.exists())
            self.assertIn("gfx: vulkan: the device was lost: \"injected by the test: no real fault\"", done.stderr)
            return text

    def test_a_lost_device_at_submit_writes_the_report_and_stops_by_name(self) -> None:
        self.need_usable_device()
        text = self.assert_loss_report("loss_submit", "vkQueueSubmit2")
        self.assertRegex(text, r"timeline: values handed out to 4, issued to 3, done to 3; the device's counter 3")
        self.assertIn("being submitted: value 4", text)
        print(f"\n{text}", end="")

    def test_a_lost_device_on_the_completion_thread_writes_the_report_and_stops_by_name(self) -> None:
        self.need_usable_device()
        text = self.assert_loss_report("loss_completion", "vkWaitSemaphores (the completion thread)")
        self.assertRegex(text, r"timeline: values handed out to 1, issued to 1, done to 0")
        self.assertIn("in flight: values 1 to 1", text)

    def test_a_device_named_by_the_setting_that_is_not_there_stops_the_start_by_name(self) -> None:
        self.need_usable_device()
        done = self.run_case("refuse", env={"OOT2_VK_DEVICE": "no such device"})
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertRegex(fatal_lines(done.stderr)[0], r"^FATAL: gfx: vulkan: OOT2_VK_DEVICE=no such device, and no "
                                                      r"device's name contains it: 0 ")

    def test_a_device_without_every_item_is_refused_with_what_it_lacks(self) -> None:
        rows = [line.split("\t") for line in self.devices.stdout.splitlines() if line.startswith("device ")]
        lacking = [r for r in rows if len(r) > 4]
        if not lacking:
            self.skipTest("every device here has every item")
        index = lacking[0][0].split()[1]
        done = self.run_case("refuse", env={"OOT2_VK_DEVICE": index})
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        fatal = fatal_lines(done.stderr)
        self.assertEqual(len(fatal), 1, done.stderr)
        self.assertIn(f"OOT2_VK_DEVICE={index} names a device the renderer cannot use: device {index} ", fatal[0])
        self.assertIn(" lacks: ", fatal[0])
        print(f"\n{fatal[0]}", end="")


if __name__ == "__main__":
    unittest.main()
