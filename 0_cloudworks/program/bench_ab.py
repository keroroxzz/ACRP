"""Quick A/B of two CloudWorks shaders at the default camera.

    python bench_ab.py                                  # HSE vs Next
    python bench_ab.py shader.glsl shader_HSE_Next.glsl

Writes results/ab_<name>.png plus a side-by-side results/ab_compare.png.
For the proper quality gate use scene_gate.py; this is just the fast loop.

Run with the py3.7 GL interpreter.
"""
from __future__ import annotations
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import cw_gl as G

CAMERA = (25000.0, 0.0, 0.0)
RAY_BIAS = 0.40


def main():
    names = sys.argv[1:] or ["shader_HSE.glsl", "shader_HSE_Next.glsl"]
    tgt = G.Target()
    print("GPU: %s   %dx%d" % (tgt.renderer, G.W, G.H))
    if not os.path.isdir(G.OUT):
        os.makedirs(G.OUT)

    shots = []
    for n in names:
        try:
            p = G.program(G.place(G.load(n), CAMERA, RAY_BIAS))
        except RuntimeError as e:
            print("=== COMPILE FAIL %s ===\n%s" % (n, e))
            continue
        ms, img = tgt.draw(p)
        shot = G.tonemap(img)
        shots.append(("%s  %.2f ms" % (n, ms), shot))
        print("%-24s %8.2f ms" % (n, ms))
        Image.fromarray((shot * 255).astype(np.uint8)).save(
            os.path.join(G.OUT, "ab_%s.png" % n.replace(".glsl", "")))

    if len(shots) == 2:
        print("mean abs diff %.5f" % np.abs(shots[0][1] - shots[1][1]).mean())
        im = Image.new("RGB", (G.W * 2 + 10, G.H + 18), (18, 18, 18))
        d = ImageDraw.Draw(im)
        for i, (label, x) in enumerate(shots):
            im.paste(Image.fromarray((x * 255).astype(np.uint8)), (i * (G.W + 10), 18))
            d.text((i * (G.W + 10) + 4, 4), label, fill=(230, 230, 230))
        im.save(os.path.join(G.OUT, "ab_compare.png"))
        print("-> results/ab_compare.png")


main()
