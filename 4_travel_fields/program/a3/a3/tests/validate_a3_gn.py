from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from a3.kernel.a3_gn_kernel import (
    intersect_box,
    make_demo_medium,
    merged_stream_transactions,
    streaming_transactions,
    trace,
)


def random_ray(medium, rng):
    center = 0.5*(medium.bmin+medium.bmax)
    if rng.random() < 0.35:
        ro = rng.uniform(medium.bmin, medium.bmax)
    else:
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        ro = center+d*1.5*np.linalg.norm(medium.bmax-medium.bmin)
    target = rng.uniform(medium.bmin, medium.bmax)
    rd = target-ro; rd /= np.linalg.norm(rd)
    return ro, rd


def main() -> None:
    seed = 20260718
    rng = np.random.default_rng(seed)
    medium = make_demo_medium(seed)

    stream_mismatches = 0
    max_time_error = 0.0
    tau_repeat_error = 0.0
    segments = []
    reloads = []
    event_errors = []
    newton_iterations = []

    rays = 240
    tested = 0
    for _ in range(rays):
        ro, rd = random_ray(medium, rng)
        hit, te, tx = intersect_box(ro, rd, medium.bmin, medium.bmax)
        if not hit or tx <= te:
            continue
        tested += 1
        a = list(streaming_transactions(medium, ro, rd, te, tx))
        b = merged_stream_transactions(medium, ro, rd, te, tx)
        if len(a) != len(b):
            stream_mismatches += 1
        else:
            for xa, xb in zip(a, b):
                max_time_error = max(max_time_error, abs(xa.t-xb.t))
                if xa.axis_masks != xb.axis_masks:
                    stream_mismatches += 1
                    break

        result = trace(medium, ro, rd)
        repeat = trace(medium, ro, rd)
        tau_repeat_error = max(tau_repeat_error, abs(result.total_tau-repeat.total_tau))
        segments.append(result.segments)
        reloads.append(result.octave_reloads)

        if result.total_tau > 1e-10:
            target = 0.5*result.total_tau
            event = trace(medium, ro, rd, target_tau=target)
            event2 = trace(medium, ro, rd, target_tau=target)
            event_errors.append(abs(event.event_t-event2.event_t))
            newton_iterations.append(event.terminal_newton_iterations)

    summary = {
        "seed": seed,
        "rays_tested": tested,
        "stream_mismatches": stream_mismatches,
        "max_stream_time_error": max_time_error,
        "max_repeat_tau_error": tau_repeat_error,
        "max_repeat_event_error": float(max(event_errors) if event_errors else 0.0),
        "mean_segments": float(np.mean(segments)),
        "mean_octave_reloads": float(np.mean(reloads)),
        "mean_newton_iterations": float(np.mean(newton_iterations)),
        "p95_newton_iterations": float(np.percentile(newton_iterations, 95)),
        "all_gates_passed": bool(
            stream_mismatches == 0
            and max_time_error < 1e-11
            and tau_repeat_error < 1e-12
            and (max(event_errors) if event_errors else 0.0) < 1e-12
        ),
    }

    frozen = ROOT / "a3" / "tests" / "a3_gn_frozen_gate.json"
    frozen.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
