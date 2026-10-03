"""Quality/perf gate for shader_HSE_Next.glsl over six camera configurations.

    python scene_gate.py            # run the gate
    python scene_gate.py --floor    # re-measure the reference noise floor

Two scenes x three angles, every render at the same cloud time (so the same
density seed), each compared against a converged uniform-march reference.

Scene locations come from hse_field.scan() - A is the flattest wedge found,
B is the one with the most separated mature towers.

  THE GATE IS NOT AN ABSOLUTE NUMBER.  A uniform 2.5 m reference is itself
  only converged to ~0.012 RMSE in the top-down views, so an absolute 0.005
  threshold there is measuring the reference's own noise.  FLOOR below is
  |ref@5m - ref@2.5m| with the reach matched, measured with --floor, and the
  gate is max(0.005, 2 x floor).

Run with the py3.7 GL interpreter.
"""
from __future__ import annotations
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import cw_gl as G

SCENES = {"A": ("cloud sea", (57500.0, 57500.0)),
          "B": ("cumulonimbus", (57500.0, 22500.0))}
# key, label, camera z (None = use IN_LAYER_Z), eye-ray z bias
ANGLES = [("g", "ground,up", 0.0, 0.40),
          ("h", "in-layer,level", None, 0.02),
          ("t", "high,down", 9000.0, -0.35)]
IN_LAYER_Z = {"A": 500.0,     # inside the 300-700 m deck
              "B": 2500.0}    # in a gap between the towers

# Sun matrix, exercised from one camera (scene B, ground looking up - the
# richest of the six).  Azimuth is measured from the view direction: 0 = the
# camera faces the sun, 180 = the sun is behind the camera, 90 = screen-left.
#
# Elevation straddles Paper 3's day/twilight switch, which sits at a solar
# zenith of 88 deg = +2 deg elevation.  The post-sunset value is -1, not the
# more obvious -4: CloudWorks' own LightSource fades the sun out over
# z in [-0.075, -0.025], so by -4 deg the light is already down to 3% and the
# frame is nearly black - it would exercise the twilight branch on an image with
# nothing in it.  At -1 deg the sun is below the horizon (so Paper 3 is on the
# twilight branch) while CloudWorks still gives full light.
#   key, label, elevation deg, azimuth deg
SUN_CONFIGS = [("preset-back",  "presunset back-lit",   4.0,   0.0),
               ("preset-side",  "presunset side-lit",   4.0,  90.0),
               ("preset-front", "presunset front-lit",  4.0, 180.0),
               ("postset-back", "postsunset back-lit", -1.0,   0.0),
               ("postset-side", "postsunset side-lit", -1.0,  90.0),
               ("postset-front","postsunset front-lit",-1.0, 180.0)]
SUN_CAMERA = ("B", "g")     # which scene/angle the sun matrix is shot from

# |ref@5m - ref@2.5m|, both reach-matched.  Regenerate with --floor.
FLOOR = {"A-g": 0.00052, "A-h": 0.00143, "A-t": 0.01241,
         "B-g": 0.00178, "B-h": 0.00460, "B-t": 0.01146}
ABS_GATE = 0.005


def camera(scene_key, cam_z):
    cx, cy = SCENES[scene_key][1]
    return (cx, cy, IN_LAYER_Z[scene_key] if cam_z is None else cam_z)


def get_reference(tgt, next_src, key, cam, bias, step=None, tag_prefix=""):
    """Render (and cache) the converged reference for one configuration.

    tag_prefix namespaces the cache.  It matters: shader_HSE_PSMS_Next has a
    different ATMOSPHERE, so its ground truth is a different image, and a cache
    keyed on the configuration alone would silently hand it the old sky's
    reference and report nonsense.
    """
    tag = key if step is None else "%s_%s" % (key, step)
    tag = tag_prefix + tag
    path = os.path.join(G.CACHE, "ref_%s.npy" % tag)
    if os.path.exists(path):
        return np.load(path)
    src = G.reference(next_src)
    if step is not None:            # coarser variant, reach kept the same
        budget = int(40000.0 / step) + 100
        src = src.replace("float4(2.5, 2.5, 1.0, 16100.0)",
                          "float4(%s, %s, 1.0, %d)" % (step, step, budget))
    _, img = tgt.draw(G.program(G.place(src, cam, bias)), tiles=24)
    if not os.path.isdir(G.CACHE):
        os.makedirs(G.CACHE)
    np.save(path, img)
    return img


def measure_floor(tgt, next_src):
    print("reference noise floor  |ref@5m - ref@2.5m|, both reach-matched:")
    for sk in SCENES:
        for ak, label, az, bias in ANGLES:
            key, cam = "%s-%s" % (sk, ak), camera(sk, az)
            fine = G.tonemap(get_reference(tgt, next_src, key, cam, bias))
            coarse = G.tonemap(get_reference(tgt, next_src, key, cam, bias, step=5.0))
            print('    "%s": %.5f,   # %s' % (key, G.compare(coarse, fine)[0], label))


def main():
    tgt = G.Target()
    next_src = G.load("shader_HSE_Next.glsl")
    if "--floor" in sys.argv:
        measure_floor(tgt, next_src)
        return
    hse_src = G.load("shader_HSE.glsl")

    print("GPU: %s   %dx%d   cloudTime=%.0f (fixed seed for every render)"
          % (tgt.renderer, G.W, G.H, G.ITIME * 100.0 + 5000.0))
    print("\ngate: rmse(Next,ref) <= max(%.3f, 2 x reference noise floor)" % ABS_GATE)
    print("      AND Next no worse than shader_HSE on rmse / edge / speckle\n")
    print("%-24s %6s %6s %6s %8s  %-19s %-19s %s"
          % ("config", "HSEms", "Nxtms", "x", "floor", "rmse HSE -> Next",
             "edge HSE -> Next", "speckle HSE -> Next"))

    failed = []
    if not os.path.isdir(G.OUT):
        os.makedirs(G.OUT)
    for sk in SCENES:
        sheet = []
        for ak, label, az, bias in ANGLES:
            key, cam = "%s-%s" % (sk, ak), camera(sk, az)
            ref = G.tonemap(get_reference(tgt, next_src, key, cam, bias))
            res = {}
            for name, src in (("HSE", hse_src), ("Next", next_src)):
                ms, img = tgt.draw(G.program(G.place(src, cam, bias)))
                shot = G.tonemap(img)
                res[name] = (ms, G.compare(shot, ref), shot)
            h, n = res["HSE"], res["Next"]
            gate = max(ABS_GATE, 2.0 * FLOOR[key])
            print("%-24s %6.2f %6.2f %5.2fx %8.5f  %.5f -> %.5f   %.5f -> %.5f   %.5f -> %.5f"
                  % ("%s %s" % (key, label), h[0], n[0], h[0] / n[0], FLOOR[key],
                     h[1][0], n[1][0], h[1][1], n[1][1], h[1][2], n[1][2]))
            if n[1][0] > gate or any(n[1][i] > h[1][i] for i in range(3)):
                failed.append(key)
            sheet.append(("%s %s" % (key, label), ref, h[2], n[2]))

        cw, ch = 520, 292
        im = Image.new("RGB", (cw * 3 + 20, (ch + 20) * 3), (18, 18, 18))
        d = ImageDraw.Draw(im)
        for i, (tag, ref, a, b) in enumerate(sheet):
            for j, (lbl, x) in enumerate([("reference", ref), ("shader_HSE", a),
                                          ("shader_HSE_Next", b)]):
                im.paste(Image.fromarray((x * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                         (j * (cw + 10), i * (ch + 20) + 18))
                d.text((j * (cw + 10) + 4, i * (ch + 20) + 4), "%s   %s" % (tag, lbl),
                       fill=(230, 230, 230))
        name = "scene_%s_%s.png" % (sk, SCENES[sk][0].replace(" ", "_"))
        im.save(os.path.join(G.OUT, name))
        print("  -> results/%s" % name)

    print("\nFAIL: %s" % (failed if failed else "none - all 6 configs pass"))


if __name__ == "__main__":       # bench_2k.py imports the scene table from here
    main()
