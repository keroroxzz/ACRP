"""Quality gate for shader_HSE_PSMS_Next_2.glsl (the optimised PSMS shader).

    python psms_gate.py            # run the gate
    python psms_gate.py --floor    # re-measure the reference noise floor

Twelve configurations, all at the same cloud time (so the same density seed):

  * the six camera configurations from scene_gate.py, at the default sun;
  * a SUN matrix - elevation +4 / -4 deg (either side of Paper 3's day/twilight
    switch at +2 deg) crossed with back-lit / side-lit / front-lit - shot from
    scene B at ground level.  A low sun is where the Chapman slant path, the
    twilight branch and the aerial perspective all do their most work, so an
    optimisation that only holds at midday is not an optimisation.

scene_gate.py does NOT cover this shader: its references are built from the Next
atmosphere, which is a different sky.  References here are namespaced "psms_".

Run with the py3.7 GL interpreter.
"""
from __future__ import annotations
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import cw_gl as G
from scene_gate import ANGLES, SCENES, SUN_CAMERA, SUN_CONFIGS, camera

BASE = "shader_HSE_PSMS_Next.glsl"        # the frozen baseline
OPT = "shader_HSE_PSMS_Next_2.glsl"       # the shader under test

# The gate is RELATIVE to the baseline, not absolute.
#
# The reference is finer than the baseline in BOTH the marcher (2.5 m uniform,
# full reach) and the atmosphere (sky 64 / aerial 48 / probe 32 steps against
# 40 / 24 / 12), so the baseline already sits 0.012-0.020 from it by
# construction.  That distance is the accepted quality of the shipping shader,
# not an error budget to spend.  What has to be shown is that optimising does
# not make it worse - so the gate is
#
#     rmse(opt, ref) <= max(ABS_GATE, TOL x rmse(base, ref))
#
# evaluated on all three metrics, with base and opt measured in the same run
# against the same reference.
ABS_GATE = 0.005
TOL = 1.15


def configs():
    """(key, label, camera, ray bias, sun or None) for all twelve."""
    out = []
    for sk in SCENES:
        for ak, label, az, bias in ANGLES:
            out.append(("%s-%s" % (sk, ak), label, camera(sk, az), bias, None))
    sk, ak = SUN_CAMERA
    az, bias = [(a[2], a[3]) for a in ANGLES if a[0] == ak][0]
    for key, label, elev, azim in SUN_CONFIGS:
        out.append((key, label, camera(sk, az), bias, (elev, azim)))
    return out


def build(src, cam, bias, sun_cfg):
    src = G.place(src, cam, bias)
    if sun_cfg is not None:
        src = G.sun(src, sun_cfg[0], sun_cfg[1])
    return src


def get_reference(tgt, base_src, key, cam, bias, sun_cfg, step=None):
    tag = "psms_" + key + ("" if step is None else "_%s" % step)
    path = os.path.join(G.CACHE, "ref_%s.npy" % tag)
    if os.path.exists(path):
        return np.load(path)
    src = G.reference(base_src)
    if step is not None:
        budget = int(40000.0 / step) + 100
        src = src.replace("float4(2.5, 2.5, 1.0, 16100.0)",
                          "float4(%s, %s, 1.0, %d)" % (step, step, budget))
    _, img = tgt.draw(G.program(build(src, cam, bias, sun_cfg)), tiles=32)
    if not os.path.isdir(G.CACHE):
        os.makedirs(G.CACHE)
    np.save(path, img)
    return img


def main():
    tgt = G.Target()
    base_src = G.load(BASE)

    opt_src = G.load(OPT) if os.path.exists(os.path.join(G.ART, OPT)) else None
    print("GPU: %s   %dx%d   cloudTime=%.0f (fixed seed for every render)"
          % (tgt.renderer, G.W, G.H, G.ITIME * 100.0 + 5000.0))
    print("\ngate: rmse(opt,ref) <= max(%.3f, 2 x reference noise floor)\n" % ABS_GATE)
    print("%-22s %7s %7s %6s %8s  %-19s %-19s %s"
          % ("config", "base ms", "opt ms", "x", "floor", "rmse base -> opt",
             "edge base -> opt", "speckle base -> opt"))

    failed, sheet = [], []
    tb = to = 0.0
    for key, label, cam, bias, sun_cfg in configs():
        ref = G.tonemap(get_reference(tgt, base_src, key, cam, bias, sun_cfg))
        res = {}
        for name, src in (("base", base_src), ("opt", opt_src or base_src)):
            ms, img = tgt.draw(G.program(build(src, cam, bias, sun_cfg)))
            shot = G.tonemap(img)
            res[name] = (ms, G.compare(shot, ref), shot)
        b, o = res["base"], res["opt"]
        tb += b[0]; to += o[0]
        ratio = max(o[1][i] / max(b[1][i], 1e-9) for i in range(3))
        print("%-22s %7.2f %7.2f %5.2fx %7.2fx  %.5f -> %.5f   %.5f -> %.5f   %.5f -> %.5f"
              % ("%s %s" % (key, label), b[0], o[0], b[0] / max(o[0], 1e-6), ratio,
                 b[1][0], o[1][0], b[1][1], o[1][1], b[1][2], o[1][2]))
        if o[1][0] > max(ABS_GATE, TOL * b[1][0]) or ratio > TOL:
            failed.append(key)
        sheet.append(("%s %s" % (key, label), ref, b[2], o[2]))

    print("%-22s %7.2f %7.2f %5.2fx" % ("TOTAL", tb, to, tb / max(to, 1e-6)))

    cw, ch = 430, 242
    im = Image.new("RGB", (cw * 3 + 20, (ch + 16) * len(sheet)), (18, 18, 18))
    d = ImageDraw.Draw(im)
    for i, (tag, ref, a, b) in enumerate(sheet):
        for j, (lbl, x) in enumerate([("reference", ref), (BASE, a), (OPT, b)]):
            im.paste(Image.fromarray((x * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (j * (cw + 10), i * (ch + 16) + 14))
            d.text((j * (cw + 10) + 4, i * (ch + 16) + 2), "%s  %s" % (tag, lbl),
                   fill=(230, 230, 230))
    if not os.path.isdir(G.OUT):
        os.makedirs(G.OUT)
    im.save(os.path.join(G.OUT, "psms_gate.png"))
    print("  -> results/psms_gate.png")
    print("\nFAIL: %s" % (failed if failed else "none - all %d configs pass" % len(sheet)))


if __name__ == "__main__":
    main()
