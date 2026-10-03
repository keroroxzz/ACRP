"""Compare every CloudWorks variant on the twelve psms_gate configurations.

    python bench_all.py            # 2560x1440
    python bench_all.py --800      # 800x450

All five are measured INTERLEAVED per configuration - all five back to back on
the same camera before moving on - so they see the same GPU clock state.  Only
ratios inside one run are meaningful (see bench_2k.py).

Comparability, stated plainly because two of these are not like-for-like:

  * shader.glsl is the 2021 pre-HSE snapshot.  place() rewrites its +X-forward
    eye ray onto the +Y convention and its cloud-time offset is normalised here
    from -2000 to +5000, so it renders the same cloud field as the rest - but it
    still has no HSE cloud-top events, so it is drawing LESS.  Its number is the
    "before" scale, not a fair rival.
  * the two PSMS shaders additionally run a full Paper 3 atmosphere and light
    the clouds from it.  They are doing strictly more work than the three
    non-PSMS ones; the interesting comparison is PSMS-vs-PSMS and
    non-PSMS-vs-non-PSMS.

Run with the py3.7 GL interpreter.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys

import cw_gl as G
from psms_gate import build, configs

VARIANTS = [
    ("cloudworks-2021", "shader.glsl",                  "pre-HSE snapshot, no cloud-top events"),
    ("HSE",             "shader_HSE.glsl",              "+ hashed spatiotemporal cloud tops"),
    ("HSE-Next",        "shader_HSE_Next.glsl",         "+ band-limited adaptive marcher"),
    ("HSE-PSMS",        "shader_HSE_PSMS_Next.glsl",    "+ Paper 3 atmosphere (baseline)"),
    ("HSE-PSMS-Next",   "shader_HSE_PSMS_Next_2.glsl",  "+ gather interpolation"),
]


def normalise(src):
    """Make every variant draw the same thing, so the timings compare like for like.

    Two differences in the 2021 shader would otherwise wreck the comparison:

      * its cloud time is offset -2000 where the HSE line uses +5000, so it
        would render a different cloud field;
      * its getViewRay() falls through to a mouse-rotation block, because
        NO_MOUSE is never defined - and the hardcoded "mouse" there is a
        function of iResolution, so the camera comes out ROTATED by about 80
        degrees.  Half the frame ended up as ground, the cloud march exited
        immediately, and it measured 31x faster than everything else.  The HSE
        line returns the ray directly (its own copy of that block is dead code
        after an unconditional `return d;`), so match that.
    """
    src = src.replace("float time = gameTime(iTime*100.0)-2000.0;",
                      "float time = gameTime(iTime*100.0)+5000.0;")
    return src.replace("    #ifdef NO_MOUSE\n    return d;\n    #endif", "    return d;")


def main():
    w, h = (800, 450) if "--800" in sys.argv else (2560, 1440)
    tgt = G.Target(w, h)
    print("GPU: %s   %dx%d   cloudTime=%.0f\n" % (tgt.renderer, w, h, G.ITIME * 100.0 + 5000.0))

    srcs = [(n, normalise(G.load(f)), d) for n, f, d in VARIANTS]
    cfgs = configs()
    progs = {}
    for n, s, _ in srcs:
        for key, lb, cam, bias, sun in cfgs:
            progs[(n, key)] = G.program(build(s, cam, bias, sun))

    names = [n for n, _, _ in srcs]
    print("%-16s" % "config" + "".join("%16s" % n for n in names))
    totals = dict((n, 0.0) for n in names)
    rows = {}
    for key, lb, cam, bias, sun in cfgs:
        r = {}
        for n in names:                       # interleaved: same clock state
            ms, _ = tgt.draw(progs[(n, key)], frames=11, warm=1.0)
            r[n] = ms
            totals[n] += ms
        rows[key] = r
        print("%-16s" % key + "".join("%16.2f" % r[n] for n in names))
    print("%-16s" % "TOTAL" + "".join("%16.2f" % totals[n] for n in names))

    ref = totals[names[0]]
    print("\n%-16s" % "vs 2021" + "".join("%15.2fx" % (ref / totals[n]) for n in names))
    print("%-16s" % "vs HSE" + "".join("%15.2fx" % (totals["HSE"] / totals[n]) for n in names))

    print("\nwhat each one adds:")
    for n, _, d in srcs:
        print("  %-16s %-42s %8.1f ms" % (n, d, totals[n]))

    try:
        print("\n" + subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,clocks_event_reasons.active",
             "--format=csv,noheader"]).decode().strip() + "   (after the run)")
    except Exception:
        pass

    out = {"resolution": [w, h],
           "totals_ms": dict((n, round(v, 2)) for n, v in totals.items()),
           "per_config": dict((k, dict((n, round(v, 2)) for n, v in r.items()))
                              for k, r in rows.items())}
    if not os.path.isdir(G.OUT):
        os.makedirs(G.OUT)
    json.dump(out, open(os.path.join(G.OUT, "perf_all_variants.json"), "w"), indent=1)
    print("-> results/perf_all_variants.json")


main()
