#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The expected image of one draw, computed on the CPU from the programs' own code.

The vertex program runs once per vertex (sass_eval), the triangles are set up
with Vulkan's rules (viewport transform, pixel centres at +0.5, an upper-left
origin, vertex positions snapped to the device's sub-pixel grid), and the
pixel program runs once per covered pixel centre.

IPA follows the hardware's model, not Vulkan's: the interpolation of every
attribute is linear in screen space. IPA.PASS of a perspective attribute
returns the screen-linear interpolation of attribute/w, of a screen-linear
attribute the screen-linear interpolation of the attribute itself, and of
position W (a[0x7c]) the screen-linear interpolation of 1/w; the pixel
program's own code then multiplies by w where it wants a perspective-correct
value. The modes come from the pixel program's header (sass_eval.parse_sph).

A pixel is compared only when the answer does not depend on rounding: its
centre must lie at least `edge_margin` pixels inside one triangle (else it is
an edge pixel, skipped), and no texture it samples with nearest filtering may
have a different texel within `texel_margin` texels of the coordinate, nor a
cube face it could equally have chosen (else it is ambiguous, skipped). Both
counts are reported.

IPA.CONSTANT (flat) reads the first vertex of the triangle, Vulkan's default
provoking vertex. A vertex program reads its vertex index at a[0x2fc] (the
vertex's position in the draw, which is not indexed) and its instance index at
a[0x2f8] (0: one instance).

Textures have one level, so an explicit or implicit level of detail selects
the same texels. Cube faces follow the Vulkan specification's face selection
(major axis, then s and t on the face), the same table the hardware's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import sass_eval as se

OUTSIDE, EDGE, AMBIGUOUS, KILLED, COLOR = "outside", "edge", "ambiguous", "killed", "color"


# Cube face selection (Vulkan specification, "Cube Map Face Selection"): face, then sc, tc and ma by axis.
_CUBE = {(0, 1): (0, lambda x, y, z: (-z, -y)), (0, -1): (1, lambda x, y, z: (z, -y)),
         (1, 1): (2, lambda x, y, z: (x, z)), (1, -1): (3, lambda x, y, z: (x, -z)),
         (2, 1): (4, lambda x, y, z: (x, -y)), (2, -1): (5, lambda x, y, z: (-x, -y))}


@dataclass
class Texture:
    width: int
    height: int
    texels: list[tuple[float, float, float, float]]  # layer (or slice, or face) major, then row-major, row 0 at v = 0
    filter: str = "nearest"
    address: str = "clamp"
    kind: str = "2d"  # 2d, 2darray, 3d or cube
    layers: int = 1  # array layers, depth, or 6 faces

    def texel(self, layer: int, x: int, y: int) -> tuple[float, float, float, float]:
        return self.texels[(layer * self.height + y) * self.width + x]

    def _axis(self, c: float, n: int, margin: float) -> tuple[int, set[int]]:
        """The texel index nearest filtering takes along one axis, and every index within margin texels of it."""
        def index(t: float) -> int:
            i = math.floor(t)
            return i % n if self.address == "repeat" else min(max(i, 0), n - 1)
        t = c * n
        return index(t), {index(t - margin), index(t), index(t + margin)}

    def sample(self, coords: list[float], layer: int | None, margin: float) -> tuple[tuple[float, float, float, float], bool]:
        """The nearest texel's RGBA, and whether a texel of another value lies within margin of the coordinate
        (or a cube face the hardware could equally have chosen)."""
        if self.filter != "nearest" or self.address not in ("clamp", "repeat"):
            raise se.Unsupported(f"texture {self.filter}/{self.address}")
        if any(math.isnan(c) or math.isinf(c) for c in coords):
            return (0.0, 0.0, 0.0, 0.0), True
        tie = False
        if self.kind in ("2d", "2darray"):
            uv, slices = coords[:2], [min(max(layer or 0, 0), self.layers - 1)]
            near_slices = slices
        elif self.kind == "3d":
            uv = coords[:2]
            k, near = self._axis(coords[2], self.layers, margin)
            slices, near_slices = [k], sorted(near)
        elif self.kind == "cube":
            mags = sorted((abs(c), a) for a, c in enumerate(coords[:3]))
            axis = mags[2][1]
            ma = coords[axis]
            if ma == 0:
                return (0.0, 0.0, 0.0, 0.0), True
            tie = mags[1][0] >= mags[2][0] * (1 - 1e-5)
            face, st = _CUBE[(axis, 1 if ma > 0 else -1)]
            sc, tc = st(*coords[:3])
            uv = [0.5 * (sc / abs(ma) + 1), 0.5 * (tc / abs(ma) + 1)]
            slices = near_slices = [face]
        else:
            raise se.Unsupported(f"a {self.kind} texture")
        (x, near_x), (y, near_y) = self._axis(uv[0], self.width, margin), self._axis(uv[1], self.height, margin)
        value = self.texel(slices[0], x, y)
        others = {self.texel(s, i, j) for s in near_slices for i in near_x for j in near_y}
        return value, tie or len(others) > 1


@dataclass
class Draw:
    width: int
    height: int
    vs: list[se.Instr]
    ps: list[se.Instr]
    ps_sph: dict
    vertices: list[dict[int, int]]  # per vertex: input attribute address -> word
    triangles: list[tuple[int, int, int]]
    vs_cbank: Callable[[int, int], int]
    ps_cbank: Callable[[int, int], int]
    gload: Callable[[int, int], bytes]
    textures: dict[int, Texture]  # the pixel program's, by the TEX handle
    subpixel_bits: int = 8
    edge_margin: float = 0.02
    texel_margin: float = 1e-3
    # Deliberate errors, for counter-runs of the comparison (see first_draw.py).
    swap_uv: bool = False
    swap_interpolation: bool = False
    tex_sources_from_ra: bool = False  # every TEX source from Ra on (sass_eval.Machine)
    vs_textures: dict[int, Texture] = field(default_factory=dict)  # the vertex program's, by the TEX handle


@dataclass
class Expected:
    kind: list[str]
    rgba: list[tuple[float, float, float, float] | None]
    vertex_outputs: list[dict[int, int]] = field(default_factory=list)


def shade_vertices(draw: Draw) -> list[dict[int, int]]:
    outs = []
    def tex(handle: int, dim: str, coords: list, layer, lod):
        # A vertex sample sets a whole triangle; a coordinate near a texel of another value cannot be told apart, and refuses.
        t = draw.vs_textures.get(handle)
        if t is None:
            raise se.Unsupported(f"no vertex texture for handle {handle:#x}")
        _check_dim(t, dim, handle)
        texel, ambiguous = t.sample(coords, layer, draw.texel_margin)
        if ambiguous:
            raise se.Unsupported(f"vertex texture {handle:#x} sampled where nearest filtering is ambiguous ({coords})")
        return texel

    for index, attrs in enumerate(draw.vertices):
        def ald(a: int, attrs=attrs, index=index) -> int:
            if a == 0x2FC:
                return index
            if a == 0x2F8:
                return 0
            return attrs.get(a, 0)
        env = se.Env(cbank=draw.vs_cbank, gload=draw.gload, ald=ald, tex=tex)
        outs.append(se.Machine(draw.vs, env, draw.tex_sources_from_ra).run().stores)
    return outs


_DIMS = {"2D": "2d", "ARRAY_2D": "2darray", "3D": "3d", "CUBE": "cube"}


def _check_dim(t: Texture, dim: str, handle: int) -> None:
    if _DIMS.get(dim) != t.kind:
        raise se.Unsupported(f"texture {handle:#x} is {t.kind}, the program samples it as {dim}")


def _screen(draw: Draw, out: dict[int, int]) -> tuple[float, float, float, float]:
    x, y, z, w = (se.bits_to_float(out.get(0x70 + 4 * k, 0)) for k in range(4))
    if not (w > 0 and abs(x) <= w and abs(y) <= w and 0 <= z <= w):
        raise se.Unsupported(f"a vertex outside the view volume needs clipping, which the reference does not do: {(x, y, z, w)}")
    sx = (x / w + 1) * draw.width / 2
    sy = (y / w + 1) * draw.height / 2
    q = 1 << draw.subpixel_bits
    return round(sx * q) / q, round(sy * q) / q, z / w, w


def render(draw: Draw) -> Expected:
    vout = shade_vertices(draw)
    screen = [_screen(draw, o) for o in vout]
    out_regs = se.pixel_output_registers(draw.ps_sph)
    if not out_regs:
        raise se.Unsupported("the pixel program writes no colour")
    tri_data = []
    for tri in draw.triangles:
        p = [screen[i] for i in tri]
        area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1])
        if area == 0:
            continue
        tri_data.append((tri, p, area))

    n = draw.width * draw.height
    kind = [OUTSIDE] * n
    rgba: list = [None] * n
    for py in range(draw.height):
        for px in range(draw.width):
            cx, cy = px + 0.5, py + 0.5
            hit = None
            near = False
            for tri, p, area in tri_data:
                b, dist = _barycentrics(p, area, cx, cy)
                if min(dist) >= draw.edge_margin:
                    hit = (tri, p, b)
                    break
                if min(dist) > -draw.edge_margin:
                    near = True
            k = py * draw.width + px
            if hit is None:
                kind[k] = EDGE if near else OUTSIDE
                continue
            kind[k], rgba[k] = _shade_pixel(draw, vout, hit, cx, cy, out_regs)
    return Expected(kind, rgba, vout)


def _barycentrics(p, area, cx, cy):
    """Screen-space barycentrics of a point, and its distance in pixels inside each edge."""
    b, dist = [], []
    for i in range(3):
        a, c = p[(i + 1) % 3], p[(i + 2) % 3]
        e = (c[0] - a[0]) * (cy - a[1]) - (c[1] - a[1]) * (cx - a[0])
        b.append(e / area)
        length = math.hypot(c[0] - a[0], c[1] - a[1])
        dist.append((e if area > 0 else -e) / length)
    return b, dist


def _shade_pixel(draw: Draw, vout, hit, cx, cy, out_regs):
    tri, p, b = hit
    imap = draw.ps_sph["imap_generic"]
    ambiguous = [False]

    def ipa(addr: int) -> int:
        if addr == 0x70:
            return se.float_to_bits(cx)
        if addr == 0x74:
            return se.float_to_bits(cy)
        if addr == 0x78:
            return se.float_to_bits(sum(b[i] * p[i][2] for i in range(3)))
        if addr == 0x7C:
            return se.float_to_bits(sum(b[i] / p[i][3] for i in range(3)))
        if 0x80 <= addr < 0x280:
            vec, comp = (addr - 0x80) >> 4, (addr >> 2) & 3
            mode = imap[vec][comp]
            vals = [se.bits_to_float(vout[i].get(addr, 0)) for i in tri]
            if draw.swap_interpolation and mode in (se.IMAP_PERSPECTIVE, se.IMAP_SCREEN_LINEAR):
                mode = se.IMAP_PERSPECTIVE + se.IMAP_SCREEN_LINEAR - mode
            if mode == se.IMAP_PERSPECTIVE:
                return se.float_to_bits(sum(b[i] * vals[i] / p[i][3] for i in range(3)))
            if mode == se.IMAP_SCREEN_LINEAR:
                return se.float_to_bits(sum(b[i] * vals[i] for i in range(3)))
            if mode == se.IMAP_CONSTANT:
                return se.float_to_bits(vals[0])  # Vulkan's provoking vertex: the first
            raise se.Unsupported(f"IPA of attribute {addr:#x}, which the program header marks unused")
        raise se.Unsupported(f"IPA of attribute {addr:#x}")

    def ipa_flat(addr: int) -> int:
        if not 0x80 <= addr < 0x280:
            raise se.Unsupported(f"IPA.CONSTANT of attribute {addr:#x}")
        return vout[tri[0]].get(addr, 0)  # Vulkan's provoking vertex: the first

    def tex(handle: int, dim: str, coords: list, layer, lod):
        t = draw.textures.get(handle)
        if t is None:
            raise se.Unsupported(f"no texture for handle {handle:#x}")
        _check_dim(t, dim, handle)
        if draw.swap_uv:
            coords = [coords[1], coords[0], *coords[2:]]
        texel, amb = t.sample(coords, layer, draw.texel_margin)
        ambiguous[0] |= amb
        return texel

    env = se.Env(cbank=draw.ps_cbank, gload=draw.gload, ipa=ipa, ipa_flat=ipa_flat, tex=tex)
    res = se.Machine(draw.ps, env, draw.tex_sources_from_ra).run()
    if ambiguous[0]:
        return AMBIGUOUS, None
    if res.killed:
        return KILLED, None
    color = tuple(se.bits_to_float(res.regs[out_regs[(0, c)]]) if (0, c) in out_regs else 0.0 for c in range(4))
    return COLOR, color
