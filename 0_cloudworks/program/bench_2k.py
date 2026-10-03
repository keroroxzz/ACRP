"""2560x1440 performance benchmark for shader_HSE_Next.glsl.

    python bench_2k.py                                   # Next vs shader_HSE
    python bench_2k.py --against OTHER.glsl              # A/B two versions
    python bench_2k.py --shader X.glsl --against Y.glsl  # A/B two arbitrary files
    python bench_2k.py --against BASE.glsl --record LABEL # append to the ledger

To extract a past version to A/B against:
    git show <commit>:articles/0_cloudworks/shader_HSE_Next.glsl > _base.glsl

Uses the same six camera configurations as scene_gate.py, so a perf number
always has a matching quality number.  The headline figure is the SUM over the
six configs - optimising one camera at the expense of another is not progress.

Two measurement rules, both learned the hard way on this GPU:

  * the two shaders are INTERLEAVED per configuration.  Running all of A then
    all of B let the clock state drift ~14% between the halves, which is larger
    than most of the wins being chased.
  * ONLY the ratio measured inside a single interleaved run is meaningful.  The
    same shader measured 536 ms in one session and 580 ms in another (~8%), so
    the ledger never compares absolute times across runs; every entry records
    the gain against a pinned baseline shader measured in the SAME run.  Pass
    that baseline with --against when recording.

Per-pixel work does not depend on resolution here (no screen-space derivatives,
no mip selection), so the 800x450 quality gate stays representative.

Run with the py3.7 GL interpreter.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time

import cw_gl as G
from scene_gate import ANGLES, SCENES, camera

WIDTH, HEIGHT = 2560, 1440
LEDGER = os.path.join(G.OUT, "perf_2k_psms.json" if "--psms" in sys.argv
                      else "perf_2k.json")


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def gpu_state():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,clocks_event_reasons.active",
             "--format=csv,noheader"]).decode().strip()
    except Exception:
        return "unavailable"
    return out


def main():
    tgt = G.Target(WIDTH, HEIGHT)
    print("GPU: %s   %dx%d" % (tgt.renderer, WIDTH, HEIGHT))

    other = arg("--against")
    base_name = other if other else "shader_HSE.glsl"
    base_src = (open(other, encoding="utf-8").read() if other
                else G.load("shader_HSE.glsl"))
    cur = arg("--shader")
    cur_name = cur if cur else "shader_HSE_Next.glsl"
    next_src = open(cur, encoding="utf-8").read() if cur else G.load("shader_HSE_Next.glsl")

    # --psms switches to psms_gate's twelve configurations, which add the sun
    # matrix (low sun x back/side/front lit).  Those are where the Chapman slant
    # path and the twilight branch cost the most, so a PSMS perf number measured
    # only at midday would be misleading.
    if "--psms" in sys.argv:
        import psms_gate
        cfgs = [(k, cam, bias, sun) for k, lb, cam, bias, sun in psms_gate.configs()]
        prep = lambda src, cam, bias, sun: psms_gate.build(src, cam, bias, sun)
    else:
        cfgs = [("%s-%s" % (sk, ak), camera(sk, az), bias, None)
                for sk in SCENES for ak, label, az, bias in ANGLES]
        prep = lambda src, cam, bias, sun: G.place(src, cam, bias)
    progs = {}
    for tag, src in (("base", base_src), ("next", next_src)):
        for key, cam, bias, sun in cfgs:
            progs[(tag, key)] = G.program(prep(src, cam, bias, sun))

    print("\n%-8s %12s %12s %9s" % ("config", base_name[:12], "Next ms", "gain"))
    totals, rows = {"base": 0.0, "next": 0.0}, {}
    for key, cam, bias, sun in cfgs:
        r = {}
        for tag in ("base", "next"):            # interleaved: same clock state
            ms, _ = tgt.draw(progs[(tag, key)], frames=11, warm=1.0)
            r[tag] = ms
            totals[tag] += ms
        rows[key] = r["next"]
        print("%-14s %12.2f %12.2f %8.1f%%"
              % (key, r["base"], r["next"], 100.0 * (r["base"] - r["next"]) / r["base"]))
    print("%-14s %12.2f %12.2f %8.1f%%"
          % ("TOTAL", totals["base"], totals["next"],
             100.0 * (totals["base"] - totals["next"]) / totals["base"]))
    print("\n" + gpu_state() + "   (immediately after the run; check for throttle flags)")

    label = arg("--record")
    if not label:
        return
    ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    entry = {"label": label, "date": time.strftime("%Y-%m-%d %H:%M"),
             "resolution": [WIDTH, HEIGHT],
             "baseline": base_name,
             "baseline_ms": round(totals["base"], 2),
             "total_ms": round(totals["next"], 2),
             "gain_vs_baseline_pct": round(
                 100.0 * (totals["base"] - totals["next"]) / totals["base"], 2),
             "per_config": dict((k, round(v, 2)) for k, v in rows.items())}
    ledger.append(entry)
    if not os.path.isdir(G.OUT):
        os.makedirs(G.OUT)
    json.dump(ledger, open(LEDGER, "w"), indent=1)
    print("\nrecorded '%s': %.2f ms vs %.2f ms baseline, %.2f%% faster"
          % (label, entry["total_ms"], entry["baseline_ms"], entry["gain_vs_baseline_pct"]))


main()
