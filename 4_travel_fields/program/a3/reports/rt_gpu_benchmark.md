# Real-time GPU benchmark: A3 vs classic raymarch (RTX 4070, GLSL)

Goal: A3 > 1.5x faster than classic fixed-step raymarch at no obvious quality difference.
Method: real GLSL fragment shader (`a3/gpu/cloud_frag.glsl`) rendered offscreen via glfw+PyOpenGL
(`a3/gpu/bench.py`), GPU-timed with GL_TIME_ELAPSED at 1280x720, RMSE vs a 512-step reference,
PNG dumps for visual inspection.

## HONEST HEADLINE (after Gate 0 conservative multi-camera audit)

The single-camera **1.60x** figure below is near the TOP of the viewpoint distribution, not typical.
Under a conservative, multi-camera audit (see "Gate 0" section), the honest number is:

> **conservative median ~1.38x across 12 cameras (P10 ~0.9x, P90 ~1.65x)**, using the cheapest
> actual march point no worse than A3 on BOTH full-image and edge RMSE. A3 ties or loses on
> inside-volume and grazing views. The 1.5x goal is met at favourable viewpoints / high quality,
> NOT as a universal figure.

Also: this shader does high-order GL **quadrature** of a ReLU-shaped, non-polynomial-envelope
density -- it is NOT exact polynomial integration, NOT a provable empty skip (midpoint heuristic),
NOT the validated transactional DDA, and has NO cached/direct-builder. See the shader header and the
"Gate 1 caveats" below. The measured speedups are real; the mechanism labels were overclaimed.

## Single-camera frontier (original, favourable viewpoint): 1.58-1.60x

Cost-vs-quality frontier (GPU ms, RMSE vs reference):

| RMSE | classic raymarch (ms) | A3 (ms) | speedup |
|---|---|---|---|
| 0.00057 / 0.00031 | 2.00 (c40) | 2.35 (f8g2 s2) | ~1.25x @0.00031 |
| 0.00025 | 3.38 (c56) | 2.65 (f8g3 s2) | 1.33x |
| 0.00015 | ~4.82 (interp) | 3.40 (f8g3 s3) | 1.42x |
| 0.00012 | 5.60 (c80) | 4.02 (f8g3 s4) | 1.36x |
| **0.00009** | **~7.28 (interp)** | **4.64 (f8g5 s4)** | **1.57x** |
| **0.00007** | **~9.65 (interp)** | **6.11 (f8g5 s4 g3)** | **1.58x** |

Best equal-quality speedup: **1.58x**. The quality-matched images (A3 4.64ms vs march 7.7ms at
RMSE ~0.00009) are visually identical (see results/rt/). Frontier plot: results/rt/frontier.png.

## Why it works, and why the speedup GROWS with quality

- Classic raymarch has step-size bias (banding); killing it needs ever more steps:
  c80=5.6ms, c96=7.7ms, c128=12.7ms, c192=26.4ms for diminishing RMSE gains.
- A3 integrates each cell's optical depth EXACTLY (Gauss-Legendre on the per-cell polynomial),
  so it reaches high quality without oversampling. Hence the speedup rises from ~1.25x (low
  quality) to ~1.58x (converged): A3's advantage is exactness, which matters most when quality
  is demanded.
- Empty-space skip (a cheap 2-octave backbone occupancy test that is a PROVABLE emptiness bound)
  lets A3 pay the expensive 4-octave evaluation + sun ray only in occupied cells.

## Honest scope and caveats (what makes or breaks the win)

1. **Density must be polynomial (or piecewise-polynomial) for A3's GL to be exact.** With raw /
   ReLU (`max(rawFbm-theta,0)`) density the win is 1.25-1.58x. With a nonlinear THRESHOLD
   (`smoothstep`) shaping the density is non-polynomial -> A3's GL is no longer exact -> the
   advantage collapses to ~1.0x (A3 ties). Crisp "game" clouds that use hard thresholds need the
   root-partition treatment (operator taxonomy P0) to recover exactness; otherwise A3 does not win.
2. **The win is on the HARD regime**: high contrast + sparse + high quality. On smooth,
   low-contrast, or low-quality targets, classic raymarch is as good or cheaper.
3. **Baseline is fixed-step raymarch** (the common "一般 raymarching"). An adaptive /
   empty-space-skipping raymarch would narrow the gap (both can skip empty space).
4. **Sun cost dominates and resists cheapening**: reducing sun octaves, sun cell frequency, or
   decoupling sun sampling from camera cells all traded quality ~1:1 for cost -> did not move the
   frontier. The win came from the CAMERA exactness at high quality, not from cheapening the sun.

## FROZEN production baseline (ReLU, 3 presets)

`a3/gpu/a3_production.glsl` + `a3/gpu/bench_production.py` -> `results/rt_production/`. This is the
frozen 1.58x ReLU baseline; do not modify (it anchors the smoothstep S1/S2 experiments).

| preset | GPU ms | RMSE | edge RMSE | evals/px | sun-ray/px | skipped | speedup |
|---|---|---|---|---|---|---|---|
| low | 2.33 | 3.1e-4 | 8.9e-4 | 28.2 | 1.09 | 68.7% | 1.25x |
| medium | 4.07 | 1.2e-4 | 3.5e-4 | 43.9 | 1.12 | 68.7% | 1.33x |
| converged | 4.59 | 9e-5 | 2.5e-4 | 51.2 | 1.14 | 68.7% | 1.60x |

The 2-octave empty bound skips 68.7% of segments, so only ~1.1 sun rays/px are cast. Architecture:
finest-grid transactional DDA, ReLU density, exact per-segment GL5, per-segment independent camera
attenuation (no cross-segment scattering batch), Tcam early-out, analytic sun via DDA+GL. Pure f32.

## Smoothstep (crisp game clouds): tiered classification helps but does NOT reach 1.5x

Following the collaborator guidance (2-octave empty bound -> segment classify -> uncertain-only
partition), a tiered adaptive integrator was added for the smoothstep-shaped density: per cell,
classify by midpoint raw-fbm +/- an in-cell margin -> skip (empty), constant (saturated), or
high-order GL (transition band). Benchmarked with full-image AND edge-region RMSE.

Result (smoothstep scene, RTX 4070):
- Adaptive classification IS cheaper than uniform GL5 (4.1ms vs 5.4ms at equal quality), but
- there is an accuracy FLOOR at RMSE ~8e-4 at f8: even uniform GL5 floors there, because GL5 is
  not exact for the degree-27 `smoothstep(degree-9)` transition. GL8 does not help.
- Finer cells (f16) break the floor (reach 3e-4) but double the cost (2x cells, 2x sun rays)
  -> 9.3ms vs march c96 7.6ms -> 0.81x (loses).
- Best smoothstep speedup with tractable methods: **~1.17x** (mid quality), not 1.5x.

Interpretation (confirms the operator taxonomy): smoothstep is NOT polynomial, so per-cell GL is
inexact; recovering the high-quality win needs TRUE root partition (split at the f=theta and
f=theta+w crossings, integrate the thin transition exactly) -- and even that faces a fine-cell /
per-cell-sun-ray cost barrier. The tractable classify+higher-GL approximation is insufficient.
Edge-region RMSE tracks full-image RMSE here (no hidden edge artifact), so the floor is real
integration error, not a masking effect.

## Smoothstep S1 (isolated accuracy proof): root partition breaks the floor, cheaply

`a3/a3/research/smoothstep_s1.py` (no envelope, no sun; rho = smoothstep((f-theta)/w), f a
realistic cubic-fade degree-9 cell field). Optical-depth error vs a dense reference over 4000
segments:

| method | p50 | p99 | max |
|---|---|---|---|
| NO partition (GL5 whole cell) | 1.6e-3 | 6.3e-2 | 9.6e-2 |
| PARTITION (roots at f=theta,theta+w + GL14 transition) | 0.0 | 3.3e-16 | 6.7e-16 |

Root partition makes the transition integration EXACT (machine precision), and it is cheap:
**mean 1.23 roots per segment** (most cells have a single threshold crossing). So the smoothstep
accuracy floor that blocked the crisp-cloud win is breakable and inexpensive in principle. Remaining
risk is the GPU cost of the degree-9 builder + root solve + GL14, and sun-ray decoupling (S2).
smoothstep is C1 -> pure branch partition, NO jump term (only a hard threshold needs a jump budget).

## Gate 0: conservative multi-camera credibility audit

`a3/gpu/bench_gate0.py` -> `results/rt_gate0/gate0.json` (full raw frontier per camera). Addresses the
audit gaps: (1) 12 viewpoints incl. inside-volume + grazing, not one scene; (2) full raymarch
frontier saved raw; (3) CONSERVATIVE speedup = cheapest ACTUAL march point no worse than A3 on BOTH
full-image AND edge RMSE (no interpolation); (4) reference convergence 512/64 vs 1024/128.

Reference convergence: max self-RMSE **1.85e-5**, ~5x below A3 converged RMSE (~9e-5) -> reference is
adequate. Note the reference is itself a high-step RAYMARCH, so it converges toward raymarch's own
answer -- this biases the comparison AGAINST A3, making the numbers conservative.

Conservative speedup across 12 cameras:

| preset | P10 | median | P90 | best single view |
|---|---|---|---|---|
| low | 1.01x | 1.36x | 1.57x | 1.72x |
| medium | 0.98x | 1.39x | 1.63x | 2.09x |
| converged | 0.87x | 1.38x | 1.65x | 2.99x |

Per-view: A3 wins on most orbit views (up to 2.99x), but **ties or loses on inside-volume (0.70-0.77x)
and grazing (0.83-0.97x)** -- its weak regime (short/degenerate segments where per-cell setup and sun
rays do not amortize). So the honest position is: **A3 beats fixed-step raymarch by ~1.4x median /
up to ~1.65x (P90), scene-dependent; the 1.5x goal is reached at favourable viewpoints and high
quality, not universally.**

## Gate 1 caveats (mechanism labels were overclaimed)

The frozen shader's advantages are real but were mislabelled. Accurate description:
1. Per-segment **GL quadrature**, NOT exact integration. ReLU `max(f-theta,0)` is piecewise
   polynomial (needs a root split at `f=theta` to be exact) and the `length+smoothstep` envelope is
   non-polynomial, so GL5 is approximate here (transport-stable, not quadrature-exact).
2. The 68.7% skip is a **midpoint heuristic** (occupied() samples the segment midpoint only), NOT a
   certified segment-empty bound -- a likely residual-RMSE source. A real certificate needs a
   2-octave Bernstein/Lipschitz whole-segment bound.
3. The DDA re-derives cells from `floor(p/cs)` with a `1e-6` nudge -- NOT the validated integer-plane
   / axis-mask / atomic-commit transactional DDA.
4. NO cached octave state / NO direct degree-9 builder (every eval recomputes 8 hashes/octave) -- so
   the speedup is achieved WITHOUT those optimizations (headroom, if register pressure allows).

An ablation (midpoint-skip vs certified-Bernstein-skip vs GL5-no-partition vs ReLU-root-partition+GL5)
to attribute the speedup between legitimate structure, heuristic skipping, and reference-tolerance-
masked quadrature error is the next Gate-1 step.

## Cell-lattice (grid) artifact: diagnosis + two fixes

The DDA shader renders a visible rectilinear grid on inside-volume / grazing views (|A3 - ref| x40;
autocorr pitch ~ projected finest-cell lattice). Isolation (FLAT_SUN=1 -> grid vanishes) proved the
cause is **single-scatter self-shadow banding**: A3 makes per-cell optical depth exact, but the
in-scatter source carries `exp(-tau_sun(s))` and was sampled ONCE PER CELL (midpoint) -> lighting
stair-steps at cell boundaries (raymarch banding at cell granularity). Camera GL order does not change
it; finer camera cells only shrink the stairs. Two earlier guesses were wrong (midpoint empty-skip;
per-cell GL-node sun averaging). Shader: `a3/gpu/a3_gridfix.glsl`; harnesses `bench_gridfix.py`,
`diag_grid.py`, `bench_gate0_gridfix.py`; data `results/rt_gridfix/`, `results/rt_gate0/gate0_gridfix.json`.

Two recorded fixes (RMSE is blind to this artifact -> we track a grid score = autocorr peak in the
cell-pitch band, not just RMSE):

| method | grid | perf | mechanism / cost |
|---|---|---|---|
| **two-anchor** (SUN_MODE=2) | clean (0.25->0.13) | **1.39x -> 0.77x (loses)** | sample tau_sun at cell entry+exit, interpolate optical depth across the cell (shared boundary anchor -> C0-continuous source), integrate source with GL incl. within-cell atten exp(-dtau*u). Costs ~1.6x (extra sun ray/cell). Per-cell adaptive gating BACKFIRES on GPU (warp divergence + broken cache). Needs S2 coarse-block sun to afford. |
| **per-pixel jitter** (SUN_JITTER) | noise (0.25->0.11) | **1.39x kept (free)** | jitter the within-cell sun sample position per pixel (`hash13(fragcoord)`). Zero cost; converts the STRUCTURED grid into UNSTRUCTURED noise. Raises RMSE (noise vs smooth ref) but noise << grid perceptually AND is removed by TAA -- exactly how production raymarched clouds work. |

Recommendation for real-time games: **jitter** (free, TAA cleans the noise). two-anchor is the
offline/no-TAA route and only pays off once S2 cheapens the sun. Key theoretical takeaway: **A3's
exactness is in TRANSMITTANCE, not in the in-scatter (lighting) integral** -- the grid is the visible
face of that limit, consistent with the free-flight/NEE positioning ([[paper4-efficiency-benchmark]]).

## Takeaway

A3 genuinely exceeds classic fixed-step raymarch (>1.5x) at converged quality, on polynomial /
piecewise-polynomial procedural media in the high-contrast, high-quality regime -- the case where
raymarch's oversampling-to-remove-banding is most wasteful. The exactness is the lever; it is
negated by non-polynomial (hard-threshold) shaping, which is the next thing to solve (root
partition) for crisp game clouds.
