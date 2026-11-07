# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Differential tests of the lifter (tools/lift) on synthetic instruction streams. No game bytes.

Random AArch64 programs (tests/lift/aarch64_ref.py) are assembled by clang's own assembler, wrapped in a synthetic
module (tests/lift/synth.py), lifted by tools/lift/gen.py, compiled with runtime/core and tests/lift/driver.c for the
target generated code uses (-O1 -march=x86-64-v3), and run; every register, the stack pointer, eight V registers and
the memory they touched must equal what the reference model computes from the instructions' definitions. The same
programs are lifted three ways (default, no register locals, no callee-saved trust), so the register and flag
planners (regs.py, flags.py) are checked against the plain lifting as well as against the model.

The D3 cases lift a synthetic sdk that defines setjmp, longjmp and __cxa_throw, and a main that imports setjmp and
longjmp through PLT stubs, and run them: a longjmp lands back in its setjmp's frame with the setjmp's callee-saved
registers, a longjmp to a buffer whose function returned is refused, a throw stops the run with its exit code, and
a setjmp through a pointer is the named fatal.

Needs clang for x86-64 and for the AArch64 assembler (OOT2_CC, else clang on PATH). Without it the tests skip,
unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "tools" / "lift" / "gen.py"
CORE = ROOT / "runtime" / "core"
DRIVER = ROOT / "tests" / "lift" / "driver.c"
sys.path.insert(0, str(ROOT / "tests" / "lift"))

import aarch64_ref as ref  # noqa: E402
import synth  # noqa: E402

MAIN_BASE = 0x7000000000
SEEDS = (11, 22, 33)
CASES_PER_TOP = 4
CONFIGS = {"default": [], "no-locals": ["--reg-locals", "0"], "no-trust": ["--no-callee-saved-trust"]}
EXE = ".exe" if os.name == "nt" else ""
HEADER = ".text\n.p2align 2\n.word 0\n.word 0\n"

D3_SDK = HEADER + """.p2align 4
sj_def:
    brk #0x51
    ret
lj_def:
    brk #0x52
    ret
throw_def:
    brk #0x53
    ret
f_jump:
    stp x29, x30, [sp, #-32]!
    stp x19, x20, [sp, #16]
    mov x19, x0
    mov x20, #5
    bl sj_def
    cbnz w0, 1f
    mov x20, #7
    mov x0, x19
    mov x1, #42
    bl g_jump
    mov x0, #99
    b 2f
1:  add x0, x0, x20
2:  ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret
g_jump:
    stp x29, x30, [sp, #-16]!
    mov x19, #1234
    mov x20, #4321
    bl lj_def
    brk #1
stale_set:
    stp x29, x30, [sp, #-16]!
    bl sj_def
    ldp x29, x30, [sp], #16
    ret
stale_jump:
    stp x29, x30, [sp, #-32]!
    str x19, [sp, #16]
    mov x19, x0
    bl stale_set
    mov x0, x19
    mov x1, #3
    bl lj_def
    brk #2
thrower:
    stp x29, x30, [sp, #-16]!
    mov x1, #0
    bl throw_def
    brk #3
sj_pointer:
    stp x29, x30, [sp, #-16]!
    adr x9, sj_def
    blr x9
    ldp x29, x30, [sp], #16
    ret
edges:
    mov x2, #5
    movn x3, #0
    cmp x2, x2
    adcs x4, x2, x3
    cset x0, cs
    cmp x2, x2
    cset x5, hi
    orr x0, x0, x5, lsl #1
    cset x5, ls
    orr x0, x0, x5, lsl #2
    mov x6, #1
    mov x7, #2
    scvtf d0, x6
    scvtf d1, x7
    fcmp d0, d1
    cset x5, mi
    orr x0, x0, x5, lsl #3
    cset x5, lt
    orr x0, x0, x5, lsl #4
    movn w3, #0
    cmp x2, x2
    adcs w4, wzr, w3
    cset x5, eq
    orr x0, x0, x5, lsl #5
    ret
"""
D3_SDK_EXPORTS = {"setjmp": "sj_def", "_setjmp": "sj_def", "longjmp": "lj_def", "__cxa_throw": "throw_def",
                  "f_jump": "f_jump", "g_jump": "g_jump", "stale_set": "stale_set", "stale_jump": "stale_jump",
                  "thrower": "thrower", "sj_pointer": "sj_pointer", "edges": "edges"}

D3_MAIN = HEADER + """.p2align 4
plt_setjmp:
    .word 0, 0, 0, 0
plt_longjmp:
    .word 0, 0, 0, 0
h_jump:
    stp x29, x30, [sp, #-32]!
    stp x19, x20, [sp, #16]
    mov x19, x0
    mov x20, #6
    bl plt_setjmp
    cbnz w0, 1f
    mov x20, #8
    mov x0, x19
    mov x1, #40
    bl k_jump
    mov x0, #99
    b 2f
1:  add x0, x0, x20
2:  ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret
k_jump:
    stp x29, x30, [sp, #-16]!
    mov x20, #555
    bl plt_longjmp
    brk #4
"""


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


def random_value(rng: random.Random) -> int:
    return rng.choice((rng.getrandbits(64), rng.getrandbits(64), rng.getrandbits(32), rng.getrandbits(8), 0,
                       ref.M64, 1 << 63, (1 << 63) - 1, 1 << 31, 0x7FFFFFFF, 1, 2))


def function_text(c_text: str, addr: int) -> str:
    """The C definition of f_<addr> in a chunk."""
    m = re.search(rf"^(?:__attribute__\(\(noinline\)\) )?void f_{addr:x}\(Cpu \*c\) \{{.*?^\}}$", c_text,
                  re.S | re.M)
    if not m:
        raise AssertionError(f"f_{addr:x} not found")
    return m.group(0)


class Lifted:
    """Modules lifted into a temporary folder and linked with the driver."""

    def __init__(self, cc: str, work: Path, modules: dict[str, bytes], args: list[str], level: str = "-O1"):
        exefs, out = work / "exefs", work / "out"
        exefs.mkdir(parents=True)
        for name, data in modules.items():
            (exefs / name).write_bytes(data)
        done = subprocess.run([sys.executable, "-I", str(GEN), "--exefs", str(exefs), "--out", str(out),
                               "--modules", ",".join(modules), "--jobs", "1", *args],
                              capture_output=True, text=True, timeout=600)
        if done.returncode != 0:
            raise RuntimeError(f"gen.py failed:\n{done.stdout}{done.stderr}")
        self.out = out
        self.report = json.loads((out / "lift_report.json").read_text(encoding="utf-8"))
        objs = []
        for c in sorted(out.glob("*/*.c")):
            obj = c.with_suffix(".o")
            run = subprocess.run([cc, "-std=gnu11", level, "-march=x86-64-v3", "-w", "-I", str(ROOT / "runtime" /
                                  "include"), "-c", str(c), "-o", str(obj)], capture_output=True, text=True)
            if run.returncode != 0:
                raise RuntimeError(f"compiling {c.name} failed:\n{run.stderr}")
            objs.append(str(obj))
        self.exe = work / f"driver{EXE}"
        run = subprocess.run([cc, "-std=gnu11", "-O1", "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror",
                              "-I", str(ROOT / "runtime" / "include"), "-I", str(CORE), "-I", str(out), str(DRIVER),
                              *sorted(str(p) for p in CORE.glob("*.c")), *objs, "-o", str(self.exe)],
                             capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"linking the driver failed:\n{run.stderr}")

    def chunks(self, module: str) -> str:
        return "\n".join(p.read_text(encoding="utf-8") for p in sorted((self.out / module).glob(f"{module}_0*.c")))

    def call(self, addr: int, x1: int = 0) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exe), "call", hex(addr), hex(x1)], capture_output=True, text=True,
                              timeout=60)


class LiftDiffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.cc = cc
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-lift-", ignore_cleanup_errors=True)
        cls.work = Path(cls.tmp.name)
        # D3: a synthetic sdk and main, lifted once.
        sdk_text, sdk_labels = synth.assemble(cc, D3_SDK, cls.work)
        main_text, main_labels = synth.assemble(cc, D3_MAIN, cls.work)
        sdk = synth.Module(sdk_text, {sym: sdk_labels[lbl] for sym, lbl in D3_SDK_EXPORTS.items()})
        main = synth.Module(main_text, {"h_jump": main_labels["h_jump"], "k_jump": main_labels["k_jump"]},
                            plt={"setjmp": main_labels["plt_setjmp"], "longjmp": main_labels["plt_longjmp"]})
        mods = {"main": main.build(), "sdk": sdk.build()}
        # -O1 is the level generated code is built at; at -O0 every C local lives in the frame, so a register kept
        # in a local across a longjmp shows its stale value instead of the one the host longjmp restored.
        cls.d3 = Lifted(cc, cls.work / "d3", mods, [])
        cls.builds = {"-O1": cls.d3, "-O0": Lifted(cc, cls.work / "d3-O0", mods, [], "-O0")}
        cls.sdk_labels, cls.main_labels = sdk_labels, main_labels
        layout = json.loads((cls.d3.out / "layout.json").read_text(encoding="utf-8"))
        cls.base = {m["name"]: int(m["base"], 16) for m in layout}

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    # ---- random programs against the reference model

    def test_random_programs_match_the_model(self) -> None:
        progs = [ref.Program(seed) for seed in SEEDS]
        source = HEADER + "".join(p.assembly() for p in progs)
        text, labels = synth.assemble(self.cc, source, self.work)
        exports = {p.name(i): labels[p.name(i)] for p in progs for i in range(len(p.funcs))}
        nso = synth.Module(text, exports).build()

        rng = random.Random(2026)
        cases = []   # (program, top, initial state arguments)
        for p in progs:
            for top in p.tops:
                for _ in range(CASES_PER_TOP):
                    x = [random_value(rng) for _ in range(31)]
                    v = [rng.getrandbits(128) for _ in range(8)]
                    buf = bytes(rng.getrandbits(8) for _ in range(ref.BUF_SIZE))
                    cases.append((p, top, x, rng.getrandbits(4) << 28, v, buf))
        blob = bytearray(struct.pack("<I", len(cases)))
        expected = []
        for p, top, x, nzcv, v, buf in cases:
            blob += struct.pack("<Q", MAIN_BASE + labels[p.name(top)])
            blob += struct.pack("<31Q", *x) + struct.pack("<Q", nzcv)
            blob += b"".join(val.to_bytes(16, "little") for val in v) + buf
            st = ref.State(x, nzcv, v, buf)
            p.run(top, st)
            regs = list(st.x[:31])
            regs[20] -= ref.BUF
            regs[21] -= ref.BUF
            expected.append((regs, st.sp - ref.STACK_TOP, st.vr[:8], bytes(st.buf)))
        self.assertGreater(len(cases), 50)

        for config, args in CONFIGS.items():
            with self.subTest(config=config):
                lifted = Lifted(self.cc, self.work / f"rand-{config}", {"main": nso}, args)
                stats = lifted.report["modules"]["main"]["stats"]
                self.assertEqual(stats["unsupported"], 0)
                if config == "default":
                    self.assertGreater(stats["local_reg_functions"], 0)
                    self.assertGreater(stats["local_flag_functions"], 0)
                cases_in, cases_out = lifted.out / "cases.bin", lifted.out / "results.bin"
                cases_in.write_bytes(blob)
                run = subprocess.run([str(lifted.exe), "cases", str(cases_in), str(cases_out)], capture_output=True,
                                     text=True, timeout=120)
                self.assertEqual(run.returncode, 0, run.stderr)
                data = cases_out.read_bytes()
                size = 8 * 31 + 8 + 16 * 8 + ref.BUF_SIZE
                self.assertEqual(len(data), size * len(cases))
                for k, ((p, top, *_), (regs, sp, vr, buf)) in enumerate(zip(cases, expected)):
                    rec = data[k * size:(k + 1) * size]
                    got = list(struct.unpack_from("<31Q", rec))
                    got_sp, = struct.unpack_from("<q", rec, 248)
                    got_v = [int.from_bytes(rec[256 + 16 * i:272 + 16 * i], "little") for i in range(8)]
                    where = f"{config}: {p.name(top)}, case {k}"
                    for r in range(31):
                        self.assertEqual(got[r], regs[r] & ref.M64, f"{where}: x{r}")
                    self.assertEqual(got_sp, sp, f"{where}: sp")
                    for i in range(8):
                        self.assertEqual(got_v[i], vr[i], f"{where}: v{i}")
                    self.assertEqual(rec[384:], buf, f"{where}: memory")

    # ---- D3: setjmp, longjmp, landing pads, replaced definitions

    def sdk(self, label: str) -> int:
        return self.base["sdk"] + self.sdk_labels[label]

    def main(self, label: str) -> int:
        return self.base["main"] + self.main_labels[label]

    def test_longjmp_lands_with_the_setjmp_registers(self) -> None:
        # x0 = 42 (the longjmp value) + x20 as the setjmp saw it (5), not as the code changed it later (7).
        for level, built in self.builds.items():
            with self.subTest(level=level):
                run = built.call(self.sdk("f_jump"))
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("x0=2f sp=0", run.stdout)
                self.assertIn("longjmp 1: jmp_buf", run.stderr)   # the run-time log of long jumps

    def test_setjmp_and_longjmp_through_plt_imports(self) -> None:
        # main imports both through PLT stubs; the stubs would dispatch through an unbound GOT, so this passes only
        # if the lifter compiled both calls itself: 40 + 6.
        for level, built in self.builds.items():
            with self.subTest(level=level):
                run = built.call(self.main("h_jump"))
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("x0=2e sp=0", run.stdout)

    def test_longjmp_to_a_returned_setjmp_is_refused(self) -> None:
        for level, built in self.builds.items():
            with self.subTest(level=level):
                run = built.call(self.sdk("stale_jump"))
                self.assertEqual(run.returncode, 70, run.stdout + run.stderr)
                self.assertIn("no live setjmp filled it", run.stderr)

    def test_throw_stops_the_run(self) -> None:
        for level, built in self.builds.items():
            with self.subTest(level=level):
                run = built.call(self.sdk("thrower"))
                self.assertEqual(run.returncode, 72, run.stdout + run.stderr)
                self.assertIn("C++ throw of (no type)", run.stderr)

    def test_setjmp_through_a_pointer_is_named(self) -> None:
        for level, built in self.builds.items():
            with self.subTest(level=level):
                run = built.call(self.sdk("sj_pointer"))
                self.assertEqual(run.returncode, 71, run.stdout + run.stderr)
                self.assertIn("setjmp called through a pointer", run.stderr)

    def test_edge_conditions(self) -> None:
        # bit 0: ADCS of 5 + all ones + carry sets C; bits 1, 2: HI false and LS true on equal operands; bits 3, 4:
        # FCMP 1.0 < 2.0 gives MI and LT; bit 5: 32-bit ADCS of 0 + 0xffffffff + 1 gives zero.
        for level, built in self.builds.items():
            with self.subTest(level=level):
                run = built.call(self.sdk("edges"))
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("x0=3d sp=0", run.stdout)

    def test_generated_shape_of_d3_functions(self) -> None:
        c = self.d3.chunks("sdk")
        f = function_text(c, self.sdk("f_jump"))
        self.assertTrue(f.startswith("__attribute__((noinline)) "))
        self.assertIn("c->x[0] = RT_SETJMP(c, c->x[0]);", f)
        self.assertNotIn("musttail", f)
        self.assertEqual(f.count("return;"), f.count("RT_JMP_LEAVE(c); return;"))
        self.assertGreater(f.count("RT_JMP_LEAVE(c); return;"), 0)
        self.assertNotIn("uint64_t x", f)   # no register locals
        self.assertNotIn("Flags F", f)      # no local flags
        g = function_text(c, self.sdk("g_jump"))
        self.assertIn("rt_longjmp(c, c->x[0], (uint32_t)c->x[1]);", g)
        self.assertNotIn("uint64_t x", g)
        self.assertIn('rt_replaced(c, "setjmp");', function_text(c, self.sdk("sj_def")))
        self.assertIn("rt_longjmp(c, c->x[0], (uint32_t)c->x[1]);", function_text(c, self.sdk("lj_def")))
        self.assertIn('rt_replaced(c, "__cxa_throw");', function_text(c, self.sdk("throw_def")))
        self.assertNotIn("rt_brk(c, 0x51", c)   # the replaced bodies are not lifted

    def test_report_lists_sites_and_trust(self) -> None:
        sdk = self.d3.report["modules"]["sdk"]
        main = self.d3.report["modules"]["main"]
        self.assertEqual(sdk["stats"]["setjmp_sites"], 2)    # f_jump, stale_set
        self.assertEqual(sdk["stats"]["longjmp_sites"], 2)   # g_jump, stale_jump
        self.assertEqual(sdk["stats"]["replaced_definitions"], 3)
        self.assertEqual(main["stats"]["setjmp_sites"], 1)
        self.assertEqual(main["stats"]["longjmp_sites"], 1)
        how = {(e["symbol"], e["how"]) for e in main["special_targets"]}
        self.assertEqual(how, {("setjmp", "plt"), ("longjmp", "plt")})
        fns = {int(e["function"][:-3], 16): e for e in sdk["d3_functions"] if "trust" in e}
        self.assertEqual(fns[self.sdk("f_jump")]["setjmp_sites"], [f"{self.sdk('f_jump') + 16:#x}ull"])
        self.assertTrue(all(e["trust"].startswith("off") for e in fns.values()))


if __name__ == "__main__":
    unittest.main()
