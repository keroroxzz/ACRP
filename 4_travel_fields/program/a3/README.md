# A3 Analytic Procedural Cloud Transport — Claude Research Package

Prepared for Brian Tu on 2026-07-18. Updated with the complete A3-GN kernel.

## Start here

1. Read `reports/A3_research_handoff_for_Claude.md`.
2. Read `a3/kernel/a3_gn_kernel.py`: this is the complete reusable A3-GN reference kernel.
3. Run `python a3/tests/validate_a3_gn.py`.
4. Compare the output with `a3/tests/a3_gn_frozen_gate.json`.
5. Use `core/multiscale_A1_incremental_reference.py` and the older reports as independent historical/oracle material.

## Complete A3-GN kernel

`a3/kernel/a3_gn_kernel.py` contains:

- streaming single finest-grid DDA;
- authoritative integer next-plane indices;
- axis-specific octave crossing transactions;
- atomic cached-cell/corner updates;
- exact GL5 full-segment optical-depth integration;
- one combined degree-9 terminal density polynomial build;
- degree-10 optical-depth polynomial construction;
- safeguarded Newton terminal inversion;
- an independent merged multi-octave stream oracle;
- a deterministic demo-medium constructor.

The public high-level entry point is:

```python
result = trace(medium, ray_origin, ray_direction, target_tau=None)
```

Passing `target_tau` returns the first free-path event; omitting it returns total optical depth.

## Frozen validation gate

The bundled gate currently validates 240 deterministic rays:

- streaming DDA equals the independent merged-stream oracle with zero mismatches;
- maximum stream boundary-time difference is zero;
- deterministic repeated optical-depth error is zero;
- deterministic repeated event-position error is zero;
- mean segment count is 64.225;
- mean octave reload count is 119.5625;
- mean terminal Newton iterations is approximately 3.0083.

These are implementation gates, not claims of GPU performance.

## Core invariants

- integer finest-plane indices are authoritative;
- form volume-local coordinates before float32 conversion;
- do not accumulate `t_max += t_delta` in the exact path;
- do not reclassify cells from approximate event times;
- carry crossed-axis information and commit octave cell updates transactionally;
- consume full cubic-fade common segments with GL5;
- build the terminal degree-9 density polynomial once;
- invert the degree-10 optical-depth clock with safeguarded Newton;
- preserve float64 as the ground-truth hierarchy;
- report transport error separately from event-topology mismatch.

## Directory layout

- `a3/kernel/`: complete current A3-GN kernel.
- `a3/tests/`: frozen gate and validation runner.
- `core/`: earlier substantive reusable Python sources.
- `experiments/result_only/`: execution-marker files whose reports/JSON are authoritative but whose full harness source was not persisted.
- `reports/`: handoff and experiment interpretation.
- `results/`: machine-readable JSON summaries.
- `archive/`: LaTeX/PDF research checkpoints.
- `image_benchmark/`: image-space pilot output and arrays.
- `MANIFEST.csv`: SHA-256 hashes, sizes, and source classifications.

## Current research roadmap

`Archive v1 -> A3 exact traversal -> transport-aware LOD -> free-path preservation -> camera/sun asymmetric policies -> multiple scattering -> nonlinear operator algebra -> procedural-field compiler`

Engineering branch:

`A3-GN -> transactional float32 traversal -> full float32 validation -> GPU kernel`
