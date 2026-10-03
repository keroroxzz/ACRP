"""Render the float32 transport-validation JSON into a Markdown report.

Run:  python a3/harness/float32_transport_report.py results/float32_transport_summary.json \
          --out reports/float32_transport_validation.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x):
    return "%.2e" % x if isinstance(x, float) else str(x)


def render(data: dict) -> str:
    L = []
    L.append("# Full-Float32 Transactional Transport Validation (handoff Sec.18)\n")
    L.append("Ground truth: **%s**. Rays per category: **%d**; categories: %s.\n"
             % (data["ground_truth"], data["rays_per_category"], ", ".join(data["categories"])))
    L.append("Error is decomposed into three independent sources:\n")
    L.append("- **ray_quantization** — f32 ray (origin+direction), f64 pipeline "
             "(`mixed64`) vs the true f64 ray. The physically different ray a GPU sees.\n")
    L.append("- **DDA-clock** — `clock32_*` (f32 boundary times, f64 integration), "
             "scored *pipeline-only* against the same f32 ray, reproducing the frozen convention.\n")
    L.append("- **full-f32** — `full32_*` (f32 GL5 + f32 terminal Newton, tol 1e-6). The new "
             "Sec.18 measurement.\n")
    L.append("\nTopology mismatches are counted **separately** from transport error "
             "(Invariant Sec.19.12): a reordered tie is transport-neutral.\n")

    for fkey, case in data["cases"].items():
        L.append("\n## %s  (rho_max = %.3f)\n" % (fkey, case["rho_max"]))
        rq = case["ray_quantization"]
        L.append("**Ray quantization (mixed64 vs true f64):** "
                 "max|dtau| %s, max|d event| %s, topology mismatches %d / %d rays.\n"
                 % (_fmt(rq["max_tau_abs_error"]), _fmt(rq["max_event_distance_error"]),
                    rq["topology_mismatches"], rq["rays"]))
        L.append("\n| mode | scope | p99 dtau | max dtau | p99 d_evt | max d_evt | topo | newton (mean/max) | fb |")
        L.append("|---|---|---|---|---|---|---:|---|---:|")
        for mode, P in case["pipeline"].items():
            for scope_key, scope_label in (("end_to_end", "vs true f64"),
                                           ("pipeline_only", "vs f32 ray")):
                s = P[scope_key]
                L.append("| %s | %s | %s | %s | %s | %s | %d | %.2f / %d | %d |" % (
                    mode, scope_label,
                    _fmt(s["p99_tau_abs_error"]), _fmt(s["max_tau_abs_error"]),
                    _fmt(s["p99_event_distance_error"]), _fmt(s["max_event_distance_error"]),
                    s["topology_mismatches"],
                    P["mean_newton_iterations"], P["max_newton_iterations"], P["newton_fallbacks"]))
    L.append("\n## Findings\n")
    L.append("1. **Ray quantization dominates end-to-end transport error.** Feeding the ray "
             "in f32 (a physically different ray) contributes max|dtau| up to ~1e-4 and event "
             "error up to ~1e-3, orders above any pipeline-precision effect; the end-to-end "
             "columns track the ray_quantization row almost exactly. This is the binding limit "
             "for an f32 GPU pipeline and reinforces Invariant Sec.19.7 (form local coordinates "
             "before f32 conversion) -- but even perfect local coords cannot beat the f32 ray.\n")
    L.append("2. **Pipeline precision is small and bounded.** Pipeline-only (same f32 ray): the "
             "f32 DDA clock adds ~7e-7 in tau across all F; full-f32 GL5+Newton adds ~1.5e-6 (F16) "
             "growing to ~7e-6 (F128) as the segment count rises. Terminal Newton stays ~2 iters "
             "with **zero fallbacks** once the spatial tolerance is set to an f32-reachable 1e-6.\n")
    L.append("3. **Topology mismatch is transport-neutral.** Up to a third of rays change the f32 "
             "event list, yet total tau and event position stay within the small pipeline error -- "
             "confirming the handoff claim that float32 boundary perturbations change the event "
             "list while preserving transport (Invariant Sec.19.12).\n")
    L.append("4. **div vs cross ordering is a topology-only choice.** Cross-multiplication changes "
             "the tie-order mismatch counts but leaves tau/event error identical; it does not "
             "reduce transport error, which is dominated by ray quantization.\n")
    L.append("\n**Modeling note.** `clock32` computes the box entry/exit (integration domain) in "
             "f32, so its pipeline-only tau (~7e-7) is larger than a pure ordering-only isolation "
             "(which would keep the domain in f64). This is the more GPU-realistic choice; the "
             "true-f64 reference and the independent GL12 oracle both remain the ground-truth "
             "hierarchy.\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    data = json.loads(args.json.read_text(encoding="utf-8"))
    text = render(data)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print("wrote", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
