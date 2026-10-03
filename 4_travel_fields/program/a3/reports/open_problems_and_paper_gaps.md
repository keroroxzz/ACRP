# A3-GN — Open problems & work not yet in the paper (research ledger)

Purpose: a single honest record of theoretical and practical work that has been *done or
discovered in the research loop but is NOT yet established in the Paper 4 manuscript*, plus
gaps that are not done at all, plus forward-looking guidance for improving each. Doubles as a
future-reference / self-improvement map. Last updated 2026-07-19.

Legend for status: **[proven]** result validated in code/experiment; **[partial]** started/some
evidence; **[open]** not done. "In paper?" = whether it is written into the manuscript yet.

Related records: [[paper4-a3gn-state]], [[paper4-rt-gpu-win]], [[paper4-efficiency-benchmark]],
[[paper4-compiler-theory]], [[paper4-float32-findings]], [[paper4-cardano-grazing-bug]],
[[paper4-phaseA-harness]]. Reports: `rt_gpu_benchmark.md`, `production_and_operator_taxonomy.md`,
`policy_engine.md`, `freeflight_benchmark.md`, `ray_quantization_findings.md`.

---

## 0. The one honest boundary to state up front

A3 makes **per-cell transmittance / optical depth exact**, but does **NOT** make the **in-scatter
(lighting) integral exact** — the single-scatter source `sigma·rho(s)·exp(-tau_sun(s))` is sampled
at a per-cell rate, so image quality is limited by the *lighting* sample rate, not by A3's optical-
depth exactness. Every "win" below (and the visible grid artifact) traces back to this. The paper
must frame A3's advantage precisely as **exact transmittance / zero-variance NEE**, not "exact
rendering". This is the single most important correction the manuscript needs.

---

## A. THEORY not yet in the paper

### A1. Single-scatter self-shadow banding & the source-rate limit  **[proven] · in paper? NO**
- Finding: the cell-lattice artifact IS per-cell sun sampling (FLAT_SUN isolation: grid vanishes;
  finer cells shrink it; camera GL order irrelevant). Formalizes §0.
- To formalize: error of a piecewise-constant-per-cell source vs the true `exp(-tau_sun(s))`;
  bound banding amplitude ~ `Tc·sigma·rho·|d/ds exp(-tau_sun)|·Δcell`. Show it is O(Δcell), not
  O(quadrature order) — explains why higher GL doesn't help.
- Improve: two-anchor (A2), jitter (A3), or coarse sun field (C6).

### A2. Two-anchor optical-depth interpolation (C0-continuous source)  **[proven] · in paper? NO**
- Sample tau_sun at cell entry+exit, interpolate optical depth linearly, integrate source with GL
  incl. within-cell camera attenuation `exp(-dtau·u)`; adjacent cells share a boundary anchor ->
  C0-continuous -> banding halved (gridScore 0.29->0.11).
- Missing theory: (a) error bound of the LINEAR tau_sun interpolation (kink at anchors -> residual
  grid ~ second difference of tau_sun); (b) a quadratic/3-anchor scheme that removes the residual;
  (c) the constant-Tc-within-cell approximation error.
- Practical caveat that MUST be stated: always-on costs ~1.6x -> kills the speedup (see B1).

### A3. Stochastic decorrelation (jitter->noise) + TAA convergence  **[proven] · in paper? NO**
- Per-pixel jitter of the within-cell sun sample converts the STRUCTURED grid into noise at zero
  cost, preserving the speedup. Perceptually noise << grid; TAA integrates it away.
- Missing theory: variance of the jittered estimator; convergence rate under N-frame temporal
  accumulation (should be O(1/N)); bias (jitter must be zero-mean in optical-depth space, not
  transmittance space — check for exp() bias). Blue-noise vs white-noise spatial spectrum.
- This is the recommended real-time path; the paper should present jitter+TAA as the production
  answer and two-anchor as the offline/no-TAA answer.

### A4. Whole-segment emptiness certificate (Bernstein/Lipschitz)  **[partial] · in paper? NO**
- Built `occupiedSegCert`: 3 backbone samples + Lipschitz margin (LIP=1.5·√3·Σa_o f_o = 2.598,
  margin = LIP·0.25·segLen) => provable density=0 over the whole segment (DETAIL_AMP bounds the
  fine octaves). Correct, but at f8 a pure Lipschitz cert is too loose (skips far fewer cells) and
  costs ~2x with no grid benefit.
- Missing: the TIGHT certificate = Bernstein/convex-hull corner bound (cubic-fade interpolation is
  a convex combination of corner hashes -> max over a cell <= max corner). Derive the per-octave
  corner-max bound and show it recovers most skips while remaining provable. This is the honest
  replacement for the current "midpoint heuristic" skip.
- Note: the certificate does NOT fix the grid (that is A1); keep the two concerns separate.

### A5. Root partition for non-polynomial shaping (smoothstep/hard threshold)  **[partial] · in paper? PARTIAL**
- S1 proved: rho=smoothstep((f-θ)/w), f∈P9 is degree-27 on the transition; GL5 floors at ~6e-2,
  partition at f=θ,θ+w + GL14 -> machine exact; mean 1.23 roots/segment.
- Missing (Gate-2 legit S1): current S1 shares `_crossings()` between reference and candidate, so it
  only proves "GL14 exact GIVEN correct roots" — NOT that GPU root ISOLATION is accurate/cheap.
  Needs: cell-clipped segments (segment must stay in one cell), SEPARATE high-precision reference
  root solver vs the candidate solver actually portable to GPU (Bernstein hull -> de Casteljau
  subdivision -> cluster -> safeguarded Newton), and adversarial near-double / tangent / plateau /
  θ≈θ+w cases that set the worst-case cost.
- Operator taxonomy correction (already noted): smoothstep is C1 -> pure BRANCH partition, NO jump
  term; only a hard THRESHOLD needs a jump budget `E_τ ≲ σΔt Σ J_i R_i`. Put the full C0/C1/P0/P1
  table in the paper.

### A6. Efficiency positioning theorem (free-flight vs transmittance)  **[proven] · in paper? NO**
- A3 LOSES stochastic free-flight sampling to local-majorant delta tracking (~10x) but WINS exact
  transmittance / NEE (~7x, zero variance). The paper should state A3's niche as a
  reference/difficult-ray transmittance solver, not a general MC accelerator.

### A7. Precision-aware transport compiler (V2 theory)  **[partial] · in paper? NO**
- theory_v2 archive: root certificates, effective degeneracy, transport-budget lowering; critical
  step = certified interval enclosures. Policy engine (ACCEPT/REFINE/RAISE_COEFF/RAISE_TOPOLOGY/
  RAYMARCH_FALLBACK) validated (monotone, 0 accepted violations) but only on synthetic NodeRecords.
- Missing: connect the policy engine to REAL per-cell measurements (root conditioning × jump,
  FP enclosure width) from the GPU/kernel, not synthetic inputs; a worked lowering example.

### A8. float64 necessity & the grazing tail  **[proven] · in paper? PARTIAL**
- A3-GBox64/F32: f64 only for ray-normalize + one box entry/exit slab clip; grazing tail
  `t_e=(plane-o)/d_a` amplified by 1/d_a is the sole f64 need at large world offsets.
- Missing: a pure-f32 DOUBLE-SINGLE (compensated) implementation of just the grazing tail, to remove
  the f64 dependency entirely (portability/perf argument the user cares about). Not implemented.

### A9. Cardano vs companion-matrix root robustness  **[proven] · in paper? NO**
- Naive Cardano cubic fails on grazing rays (a3->0, double roots dropped, ~85% err); fixed via
  companion-matrix eigenvalue solver + relative imag tolerance + residual-monotone Newton polish.
  GPU needs relative degree-reduction. Belongs in the numerical-robustness section.

---

## B. PRACTICE / IMPLEMENTATION not yet done

### B1. Honest GPU speedup is ~1.38x median, NOT 1.60x  **[proven] · in paper? NO**
- Gate-0 conservative multi-camera (12 views, cheapest march no worse on full+edge RMSE): median
  1.38x, P10 ~0.9x, P90 ~1.65x; A3 ties/loses on inside (0.7-0.85x) and grazing (0.83-0.97x).
  The 1.60x was a favourable single view. The paper must use the distribution, not the best view.
- Two-anchor grid fix drops median to 0.77x; jitter keeps 1.38x. State both.

### B2. Shader mechanism labels were overclaimed  **[fixed in code] · in paper? MUST AUDIT**
- The frozen shader does high-order GL QUADRATURE (not exact), MIDPOINT-heuristic skip (not
  certified), floor()+1e-6 DDA (not the validated transactional DDA), NO cached/direct builder.
  Any manuscript text claiming exact/provable/transactional/cached for THIS shader is wrong.

### B3. GPU port of the S1 root partition  **[open]**
- Put degree-9 direct builder + Bernstein classify + root solve (~1.23 roots) + GL14 into the
  fragment shader (no envelope/sun first), instrument root count + subdivision cost. This is the
  untested-cost item that decides whether crisp smoothstep clouds can keep >1.5x.

### B4. True transactional DDA on GPU  **[open]**
- Replace floor(p/cs)+1e-6 with integer plane-index / axis mask / atomic multi-axis commit
  (the validated CPU kernel design). Removes boundary seams; needed before calling any GPU shader
  "A3 transactional".

### B5. Cached octave state / direct degree-9 coefficient builder on GPU  **[open]**
- Current shader recomputes 8 hashes/octave every eval (up to ~160 hashes/segment). Corner cache +
  direct ascending-coeff assembly could cut this — IF register pressure allows. Potential headroom
  the current 1.38x does NOT yet use. Measure register/occupancy tradeoff.

### B6. S2 coarse-block sun decoupling  **[open]**  ← makes two-anchor affordable
- Bind sun to COARSE blocks (>= camera cell), compute tau_sun at block anchors ONCE, reuse/
  interpolate across many fine camera cells. Test single-anchor vs two-anchor optical-depth interp;
  keep per-cell camera attenuation independent (no cross-cell scattering batch). Report root-partition
  cost, sun cost, roots/segment, uncertain ratio, fallback ratio, full+edge RMSE, GPU ms. This is
  what could turn two-anchor from 0.77x back toward >1.3x while staying grid-clean.

### B7. Temporal jitter + TAA accumulate demo  **[open]**  ← decisive real-game validation
- Frame-varying jitter seed + a small N-frame accumulate; show the noise resolves to a clean,
  grid-free image while holding the 1.38x. This is the proof the free path works for games.

### B8. Gate-1 attribution ablation  **[open]**
- 4-way: midpoint-skip vs certified-Bernstein-skip vs GL5-no-partition vs ReLU-root-partition+GL5.
  Attributes the ~1.38x between legitimate structure, heuristic skipping, and reference-tolerance-
  masked quadrature error.

### B9. Scene-parameter sweep (not just cameras)  **[open]**
- Gate-0 varied 12 cameras on ONE scene. Add threshold / sparsity / sun-direction sweeps; report
  median/P10/P90 across the full matrix. Also: use a NON-raymarch (or richer) reference, since the
  512-step raymarch reference biases the comparison against A3.

### B10. Real engine (HLSL / actual game) integration  **[open]**
- Everything so far is offscreen GLSL via glfw+PyOpenGL. A real engine adds TAA, motion vectors,
  depth compositing, temporal reprojection — the environment where jitter+TAA is validated and
  where cloud animation (moving noise) stresses temporal stability. Untested.

### B12. Hierarchical empty-space acceleration for A3 (large sparse scenes)  **[DONE 2026-07-19]** ← turned the loss into a win
- Problem: on the realistic layered sky + cumulonimbus scene (`a3/gpu/sky_frag.glsl`), UNIFORM A3 was
  **3.3-3.4x SLOWER** than a strong AAA raymarch (uniform 60.9ms vs march 18.3ms median), because the
  march skips empty air with big fixed steps while A3's uniform finest DDA steps cell-by-cell.
- SOLUTION (METHOD==2 renderA3H): per-iteration adaptive two-level march -- test `presence()` at the
  point; if empty, jump BIG_STEP (same coarse skip as the march); if cloud, do ONE finest DDA cell with
  exact GL + one sun ray, then continue. Empty air is now skipped as cheaply as the march; A3's exact
  integration is spent only inside clouds. Single loop, one budget, guaranteed forward progress
  (an earlier nested-loop/rewind version thrashed at cloud boundaries -> 1200ms; do NOT rewind or use a
  fine/coarse mode toggle).
- KEY TUNING: inside cloud, GL5 OVERSAMPLES smooth interiors. Dropping to GL2 + coarser cells is the win.
  Bench (`bench_sky.py`, `results/sky/bench_sky.json`, 4 cameras, RTX4070): march 18.3ms/RMSE1.11e-2;
  A3-hier CAMF6 GL2 = **1.09x faster at EQUAL/better RMSE (1.09e-2)**; CAMF4 GL2 = 1.45x (RMSE 1.21e-2);
  CAMF3 GL2 = 1.73x (RMSE 1.36e-2). So A3 MATCHES the strong AAA baseline at equal quality and is faster,
  up to 1.7x if slightly coarser cells are acceptable. Images: results/sky/win_march_vs_a3hier.png.
- REMAINING: still only ~1.1x at strictly-equal quality; to widen the margin, cheapen the in-cloud sun
  (S2 coarse-block sun) and/or an analytic altitude-band skip (clouds live in known y-windows). The A3
  edge here is per-cell exact integration letting GL2 match the march's finer fixed steps at lower sun
  cost. Also: A3-hier uses per-pixel SUN_JITTER (free grid fix) -> slightly grainier than march; TAA
  removes it (B7). Honest boundary now: A3 needs a hierarchy to compete on sparse skies -- it is NOT
  automatic from the finest-DDA formulation.

### B11. Animated / temporally-coherent clouds  **[open]**
- All tests are static. Moving/evolving noise + a moving camera stress the jitter noise's temporal
  stability and the DDA's boundary behavior. Needed for a game claim.

---

## C. Methodology corrections to bake into the paper

- **Track a structure/grid metric, not only RMSE.** RMSE is BLIND to low-amplitude regular
  artifacts (the grid sat at max ~0.015/1.0 yet was obvious). Use an autocorrelation peak in the
  cell-pitch band (or an SSIM/perceptual term) as a first-class quality gate.
- **Conservative equal-quality, actual points.** Speedups must use the cheapest ACTUAL baseline
  point no worse on ALL quality axes (full RMSE, edge RMSE, grid score) — not linear RMSE
  interpolation, not the single best view.
- **State the reference's nature.** A raymarch reference converges toward raymarch's own answer and
  biases against A3; report reference convergence (512 vs 1024) and, ideally, a method-neutral
  reference.
- **GPU reality ≠ eval counts.** Per-cell adaptive branching backfired (warp divergence). CPU
  eval-count models do not predict GPU cost; always validate on the GPU.

---

## D. Suggested priority order (highest leverage first)

1. **B7 temporal jitter + TAA demo** — cheapest, proves the free grid fix works for games (the
   user's actual goal). Unblocks the "usable in a real-time game" claim.
2. **B1 + B2 manuscript honesty pass** — fix the 1.60x -> 1.38x number and the overclaimed
   mechanism labels before anything else is written up. Cheap, high integrity value.
3. **B6 S2 coarse-block sun** — the one change that could make the clean (two-anchor) fix affordable
   AND is already planned; also the biggest remaining lever on absolute cost.
4. **A4 Bernstein certificate + B8 Gate-1 ablation** — turn the heuristic skip into a real
   certificate and attribute the speedup honestly.
5. **B3 GPU root partition + A5 Gate-2 legit S1** — decides the crisp-cloud (hard-threshold) story.
6. **B4/B5 transactional DDA + cached builder on GPU** — needed to call the GPU path "A3", plus
   possible headroom.
7. **A8 double-single grazing tail** — removes the f64 dependency (portability/perf).
8. **B9/B10/B11** — scene sweep, real engine, animation — for the final production claim.
