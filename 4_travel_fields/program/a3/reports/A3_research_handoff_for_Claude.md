# A3 / Analytic Procedural Cloud Transport Research Handoff

**Recipient:** Claude  
**Prepared for:** Brian Tu  
**Date:** 2026-07-18  
**Purpose:** Synchronize the current mathematical, numerical, and engineering state of the research so work can continue without re-deriving or contradicting validated results.

---

# 1. Research Objective

The project investigates an analytic or semi-analytic alternative to fixed-step ray marching for procedural participating media, especially large-scale volumetric clouds.

The central goal is not currently “replace production ray marching everywhere.” The project is organized around three stronger and more defensible objectives:

1. **Exact reference transport solver** for procedural cloud media.
2. **Theoretically clean free-path and transmittance computation** without step-size bias.
3. **A path toward hybrid or engineering-optimized implementations** when exact traversal can be made practical.

The current exact backbone is called **A3-GN**:

\[
\boxed{
\text{Streaming finest-grid DDA}
\rightarrow
\text{octave crossing transactions}
\rightarrow
\text{GL5 full-segment integration}
\rightarrow
\text{terminal polynomial build}
\rightarrow
\text{safeguarded Newton}
}
\]

---

# 2. Research Roadmap

Keep this roadmap explicit and restate it every few interactions to avoid drift:

\[
\boxed{
\text{Archive v1}
\rightarrow
\text{A3 exact traversal}
\rightarrow
\text{transport-aware LOD}
\rightarrow
\text{free-path preservation}
\rightarrow
\text{camera/sun asymmetric policies}
\rightarrow
\text{multiple scattering}
\rightarrow
\text{operator algebra/compiler}
}
\]

The engineering branch currently is:

\[
\boxed{
\text{A3-GN exact kernel}
\rightarrow
\text{single finest-grid DDA}
\rightarrow
\text{transactional float32 traversal}
\rightarrow
\text{full float32 validation}
\rightarrow
\text{GPU realization}
}
\]

Do not let the project collapse into parameter tuning only. The priority remains theory, invariants, transport correctness, and reusable compiler-like rules.

---

# 3. Mathematical Foundation

## 3.1 Ray flow

State space:

\[
\mathcal X=\mathbb R^3\times S^2
\]

Ray flow:

\[
\Phi_t(p,\omega)=(p+t\omega,\omega)
\]

For a deterministic hit countdown \(T\):

\[
T(\Phi_s x)=T(x)-s
\]

before the first hit.

## 3.2 Optical-depth clock

For extinction \(\sigma_t\):

\[
A_x(t)=\int_0^t \sigma_t(\Phi_s x)\,ds
\]

with cocycle law:

\[
A(x,s+t)=A(x,s)+A(\Phi_sx,t)
\]

For free-path sampling:

\[
\xi=-\log(1-U)
\]

and the event time is:

\[
T_\xi=\inf\{t:A_x(t)\ge\xi\}
\]

The optical-depth countdown state is the correct stochastic analogue of geometric distance-to-hit.

## 3.3 Cubic-fade value noise

For one coordinate:

\[
F_0=3r_0^2-2r_0^3
\]

\[
F_1=d(6r_0-6r_0^2)
\]

\[
F_2=d^2(3-6r_0)
\]

\[
F_3=-2d^3
\]

Along a ray, each faded coordinate is cubic in ray parameter.

For trilinear interpolation in faded coordinates:

\[
\rho=d_{000}
+\Delta_xF_x+\Delta_yF_y+\Delta_zF_z
+\Delta_{xy}F_xF_y+\Delta_{xz}F_xF_z+\Delta_{yz}F_yF_z
+\Delta_{xyz}F_xF_yF_z
\]

Therefore:

\[
\deg\rho\le9
\]

and:

\[
\deg A\le10
\]

If corners are nonnegative, cubic fade is convex interpolation, so:

\[
\rho\ge0,\qquad A'(t)\ge0
\]

Thus optical depth is monotone and terminal inversion is well-posed.

---

# 4. Exact Segment Integration

For one common-refinement segment, the multiscale sum of cubic-fade octaves is still degree at most 9:

\[
\rho(t)=\sum_\ell a_\ell\rho_\ell(t)
\]

The octave count increases partition complexity, not polynomial degree.

Therefore 5-point Gauss-Legendre is exact:

\[
\boxed{\text{GL5 exactly integrates degree-9 density}}
\]

This is the core reason A3 can consume a complete common segment exactly without constructing the full polynomial each time.

---

# 5. A3-GN Terminal Event Solver

The correct event workflow is:

1. Consume complete common segments with GL5.
2. Find the segment containing the optical-depth target.
3. Build the combined degree-9 density polynomial once.
4. Integrate it to a degree-10 optical-depth polynomial.
5. Solve with safeguarded Newton.

Validated result:

- 500 terminal events
- 5 octaves
- P99 distance error versus 28-step GL5 bisection:
  \[
  1.33\times10^{-10}
  \]
- Maximum:
  \[
  1.46\times10^{-10}
  \]
- Mean Newton iterations: 3.04
- P95: 4
- Maximum: 4
- Safeguard fallbacks: 0

Repeated GL5 bisection required approximately 700 scalar octave-density evaluations per terminal solve. The intended polynomial/Newton terminal solve needs 5 octave polynomial builds plus short Horner evaluations.

Important lesson: do not benchmark A3 using repeated GL5 bisection and then treat that as the final A3 cost.

---

# 6. Multiscale Exact Traversal

## 6.1 A0/A1/A2/A3 history

- **A0:** rebuild all active octave data at every segment.
- **A1:** attempted polynomial shifting in world distance; unstable at high frequencies due to \(f^k\) coefficient scaling.
- **A2:** antiderivative subtraction; unstable due to cancellation.
- **A3:** exact common-refinement traversal with GL5; successful.

A critical boundary ownership rule was found:

If a local grid coordinate is numerically close to an integer plane, perturb in the sign of the ray direction in local grid coordinates before selecting the cell.

This reduced previously catastrophic optical-depth error from order 1 to approximately \(10^{-12}\).

## 6.2 A3 benchmark

400 rays, frequencies:

\[
[1,2,4,8,16]
\]

Results:

- P99 optical-depth error:
  \[
  4.79\times10^{-12}
  \]
- Maximum:
  \[
  5.68\times10^{-12}
  \]
- Mean common segments: 64.35
- A0 mean octave rebuilds: 321.74
- A3 mean reloads: 124.83
- Reload reduction: 61.2%
- All tests passed

---

# 7. Major Engineering Breakthrough: Single Finest-Grid DDA

Because octave frequencies are dyadic:

\[
1,2,4,8,16,\ldots,F
\]

the common-refinement partition is exactly the finest grid.

Instead of maintaining one DDA per octave, traverse only the finest grid.

For finest plane index \(k\), octave frequency \(f\) crosses when:

\[
k\bmod(F/f)=0
\]

Since \(F/f\) is a power of two, crossing detection can use bit tests or trailing-zero logic.

Validated against the previous multi-octave merged event stream:

- 1028 rays
- 28 adversarial rays
- 0 event-list mismatches
- 0 mask mismatches
- 0 reload-count differences
- Maximum boundary-time difference: 0

DDA clock state:

- Previous conceptual state:
  \[
  3L
  \]
- New state:
  \[
  3
  \]

For 5 octaves:

\[
15\rightarrow3
\]

an 80% reduction in clock state.

The octave cell/corner states remain \(O(L)\), but independent octave boundary clocks and runtime stream merging are eliminated.

---

# 8. Streaming Incremental DDA

The event list was then made fully streaming.

State:

- current finest cell index per axis
- integer next-plane index per axis
- three boundary clocks
- ray direction signs
- octave cell/corner states

Memory:

\[
O(1)\text{ traversal state}+O(L)\text{ octave state}
\]

Validation:

- 2027 rays
- 27 adversarial rays
- 0 stream mismatches
- 0 mask mismatches
- 0 event-count mismatches
- maximum boundary-time difference: 0

## Critical numerical rule

Do not update:

\[
t_{\max}\mathrel{+}=\Delta t
\]

Repeated accumulation can separate theoretically simultaneous diagonal crossings.

Instead make the integer plane index authoritative:

\[
k_a\leftarrow k_a+\operatorname{sign}(\omega_a)
\]

and recompute:

\[
t_{\max,a}
=
\frac{k_a/F-o_a}{\omega_a}
\]

The design principle is:

\[
\boxed{
\text{integer topology}
\rightarrow
\text{floating geometry}
}
\]

not the reverse.

---

# 9. Float32 Findings

## 9.1 World coordinates are unsafe

Tested:

\[
F_{\max}=16,32,64,128
\]

and world offsets:

\[
0,\ 10^3,\ 10^5,\ 10^7
\]

Three pipelines:

1. large world origin directly in float32;
2. subtract two separately quantized float32 world values;
3. form local coordinates in higher precision, then convert to float32.

The first two are effectively equivalent and fail badly at large offsets.

Correct interface:

\[
\boxed{
\mathbf o_{\text{local},32}
=
\operatorname{float32}
\left(
\mathbf o_{\text{world},64}
-
\mathbf o_{\text{volume},64}
\right)
}
\]

Do not use:

\[
\operatorname{float32}(\mathbf o_{\text{world}})
-
\operatorname{float32}(\mathbf o_{\text{volume}})
\]

Large-world information may already be lost before subtraction.

## 9.2 Float32 event-list identity is too strict

At high frequencies, pure float32 boundary times can split or merge near-tie events. Event topology may differ from float64 even when transport error is tiny.

Therefore distinguish:

\[
\text{event-list identity}
\]

from:

\[
\text{transport-equivalent partition}
\]

---

# 10. Critical Bug Found and Fixed

An earlier cached-state tie-envelope benchmark produced huge optical-depth errors of order 1.

This was a harness bug.

The bad implementation reclassified the new octave cell from the approximate float32 event time. If the time lay slightly before the true grid plane, it selected the old cell and created a long stale-state interval.

Correct rule:

\[
\boxed{
\text{Never infer topology from approximate event time after the crossing is known.}
}
\]

Instead, each event carries an axis-specific octave transaction.

For every crossed axis and affected octave:

\[
\text{cell}_{\ell,a}
\leftarrow
\text{cell}_{\ell,a}
+
\operatorname{sign}(\omega_a)
\]

Then reload the corresponding corners.

This must be committed atomically for all axes in the tie group.

---

# 11. Transactional Float32 Traversal

Each event must retain:

- crossed axis
- integer plane index
- octave crossing mask
- ray direction sign

An event is not just a scalar time and a combined octave mask. Axis information is required to update the correct cell coordinates.

Correct state update:

\[
\boxed{
\text{axis-specific integer crossing transaction}
}
\]

This is the central float32 engineering result.

## Focused validation

For \(F_{\max}=16\):

- maximum optical-depth error:
  \[
  4.44\times10^{-15}
  \]
- maximum median free-path event-distance error:
  \[
  3.56\times10^{-7}
  \]

For \(F_{\max}=128\):

- maximum optical-depth error:
  \[
  4.88\times10^{-15}
  \]
- maximum median free-path event-distance error:
  \[
  1.34\times10^{-6}
  \]

This was a small focused pilot, not yet a large statistical validation.

Interpretation:

- Integer transactional updates protect cached state.
- Float32 time perturbation affects event position slightly.
- Total optical depth can remain essentially exact in the tested cases.
- The next validation must use more rays, several target quantiles, and full float32 arithmetic.

---

# 12. Tie Envelopes

A possible coalescing rule was studied:

\[
\varepsilon_t
=
\min\left(
\varepsilon_{\mathrm{ULP}},
\frac{\varepsilon_\tau}{\sigma_t\rho_{\max}}
\right)
\]

If crossings fall inside the envelope, merge them into one transaction.

The local conservative bound is:

\[
\Delta\tau
\le
\sigma_t\rho_{\max}\Delta t
\]

Important nuance:

- This bound controls optical depth over the collapsed span.
- It does not protect against downstream stale-state error unless all merged axis/octave updates are committed transactionally.
- A broad ULP envelope may merge more events than necessary.
- A transport-bounded envelope can be more conservative.

Current evidence is promising but not yet a full proof or production policy.

---

# 13. Transport-Aware LOD Results

## 13.1 Raw cutoff is biased

Dropping high-frequency octaves directly removes mean extinction and causes strong optical-depth and transmittance bias.

## 13.2 B1 mean compensation

For cutoff \(K\):

\[
\widetilde\rho_K
=
\sum_{\ell<K}\rho_\ell
+
\sum_{\ell\ge K}\mu_\ell
\]

The exact cell average of cubic-fade interpolation equals the arithmetic mean of its corners.

Results:

- K3 raw P95 \(\tau\) error: 0.2935
- K3 mean-compensated P95: 0.04693
- K4 raw P95: 0.09524
- K4 mean-compensated P95: 0.01324

Strong conclusion: most raw-cutoff error is systematic mean-extinction loss.

## 13.3 B2 global second-cumulant correction

Model:

\[
\operatorname{Var}[\tau_R\mid L]\approx aL+bL^2
\]

and effective correction:

\[
\tau_{\mathrm{low}}+\mu_R-\frac12v_R
\]

Result: negligible improvement over B1, sometimes slightly worse.

Conclusion: global path-length variance correction is not useful enough.

## 13.4 B3 local residual mean hierarchy

Piecewise residual-mean grids produced some improvement but poor memory/cost ratio.

Not production-worthy in current form.

## 13.5 B4 conditional mean under modulation

Model:

\[
\rho(x)=B(x)+m(x)R(x)
\]

Approximation:

\[
\rho_{\mathrm{LOD}}(x)=B(x)+m(x)\mathbb E[R]
\]

Compared with global mean, conditional mean improved:

- P95 optical-depth error by about 28.6%
- P99 transmittance error by about 30%

This is the first clearly useful graph-structured transport LOD rule.

## 13.6 B5 nonlinear operator

Model:

\[
\rho(x)=\max(0,B(x)-\lambda R(x))
\]

Compared:

- raw cutoff
- mean substitution
- distribution-aware conditional mean:
  \[
  h(B)=\mathbb E_R[(B-\lambda R)_+]
  \]

Distribution-aware correction gave modest but real improvement over mean substitution and reduced signed bias.

Operator algebra result:

If \(f\in\mathcal P_n\), then:

\[
\max(0,f)
\]

is piecewise polynomial, with partition augmented by roots of \(f=0\).

---

# 14. Negative Results That Must Not Be Forgotten

1. **C residual bounding approach** was abandoned as the main line. Conservative residual bounds force finest refinement and naive recursion repeats active-octave work.
2. **A1 world-distance polynomial shifting** is unstable at high frequencies.
3. **A2 antiderivative subtraction** is unstable due to cancellation.
4. **Global second-cumulant LOD correction** is not worthwhile.
5. **Dense local residual-mean grids** have poor memory/cost efficiency.
6. **Cross multiplication alone** does not recover lost float32 ray geometry.
7. **FP64 clock with already-quantized float32 ray state** improves timing but cannot restore information already lost.
8. **Reclassifying cells from approximate event time** is incorrect and can create huge stale-state errors.
9. Do not trust the old failed A/B/C comparison report unless rerun.
10. Do not treat the earlier repeated-GL5 image benchmark as the final A3 cost.

---

# 15. Image-Space Benchmark Status

A first low-resolution paired camera/sun single-scattering benchmark was run.

Methods:

- A3 exact
- K4 plus mean compensation
- ray marching with 24, 48, 96 steps

The current unoptimized Python A3 was much more expensive than ray marching, and ray marching was accurate in that smooth scene.

Interpretation:

- A3 is not yet a universal production replacement.
- The scene did not stress thin features or adversarial sampling.
- A3 remains valuable as a reference and difficult-ray solver.
- The benchmark used repeated GL5 terminal bisection and overstated the intended A3 terminal cost.

Future image benchmarks should include:

- thin cloud sheets
- grazing rays
- sparse narrow peaks
- sub-step features
- moving-camera phase changes
- sun-shadow leakage
- temporal flicker

---

# 16. Current Reliable Claims

The following claims are well supported:

1. Cubic-fade value noise restricted to one cell becomes degree at most 9 along a ray.
2. Its optical depth is degree at most 10.
3. GL5 exactly integrates one common segment.
4. A multiscale linear octave sum does not increase segment polynomial degree.
5. A3 exact traversal is numerically validated in float64.
6. Terminal safeguarded Newton converges rapidly and accurately.
7. Dyadic multiscale common refinement can be traversed by one finest-grid DDA.
8. The single-DDA event stream exactly matches the multi-octave merged stream in float64.
9. Streaming constant-memory traversal is valid.
10. Integer plane index must be authoritative.
11. Local coordinates must be formed before float32 quantization.
12. Cached octave states must be updated transactionally from integer crossing data.
13. Event-list mismatch does not necessarily imply meaningful transport error.
14. Mean extinction compensation is the strongest simple LOD correction found so far.
15. Conditional expectation is the correct direction for graph-structured LOD.

---

# 17. Claims Not Yet Established

Do not overclaim the following:

1. A3 is faster than optimized production ray marching.
2. Full float32 A3 is statistically validated.
3. Float32 topology is exactly preserved at high frequency.
4. Tie-envelope coalescing is fully proven for all cases.
5. GPU register pressure and divergence are acceptable.
6. Multiple scattering has been integrated.
7. Camera/sun asymmetric LOD has been validated in full images.
8. Nonlinear operator transfer-function surrogates are complete.
9. Full procedural graph compiler exists.
10. Current CPU operation counts predict GPU wall time.

---

# 18. Immediate Next Experiment

The next rigorous experiment should be:

## Full float32 transactional transport validation

Use:

- \(F_{\max}=16,32,64,128\)
- several hundred rays
- random, exact diagonal, near-diagonal, axis-aligned, near-axis, grazing
- local ray origin supplied directly
- float32 density evaluation
- float32 GL5
- float32 terminal polynomial and Newton
- integer axis/octave transactions
- immediate, ULP, and transport-bounded tie policies

Measure:

\[
|\Delta\tau|
\]

\[
|\Delta T|
\]

\[
|t_{\xi,32}-t_{\xi,64}|
\]

for targets at:

\[
20\%,\ 50\%,\ 80\%
\]

of total optical depth, plus random exponential targets.

Also record:

- event count
- topology mismatch
- merged groups
- collapsed length
- state reload count
- hit/escape mismatch
- Newton iterations
- fallback count

The most important distinction is:

\[
\boxed{
\text{topology correctness}
\quad\text{versus}\quad
\text{transport correctness}
}
\]

---

# 19. Recommended Implementation Invariants

Claude should preserve these invariants in all future code:

1. **Integer plane index is authoritative.**
2. **No cumulative \(t_{\max}+=\Delta t\) in the exact path.**
3. **Never classify a post-crossing cell from approximate event time.**
4. **Every event retains crossed-axis information.**
5. **Octave state updates are transactional.**
6. **All tied axes commit before the next nonzero segment.**
7. **Local coordinates are formed before float32 conversion.**
8. **Full segments use GL5.**
9. **Terminal segment builds the polynomial only once.**
10. **Terminal root solve uses safeguarded Newton.**
11. **Float64 remains the reference hierarchy.**
12. **Any approximate float32 mode must report transport error, not only topology mismatch.**

---

# 20. Key Artifacts

All files are under `/mnt/data`.

## Archive

- `Analytic_Multiscale_Cloud_Traversal_Archive_v1.tex`
- `Analytic_Multiscale_Cloud_Traversal_Archive_v1.pdf`
- `Analytic_Multiscale_Cloud_Traversal_Archive_v1.zip`

## Exact single-cell and volume references

- `cubic_fade_degree10_reference.py`
- `cubic_fade_degree10_validation_report.md`
- `cubic_fade_degree10_volume_reference.py`
- `cubic_fade_degree10_volume_validation_report.md`

## Float32 studies

- `cubic_fade_degree10_float32_study.py`
- `cubic_fade_degree10_whole_volume_float32_final_report.md`
- `A3_float32_streaming_dda_robustness.md`
- `A3_mixed_precision_event_ordering.md`

## Multiscale A3

- `multiscale_A1_incremental_reference.py`
- `multiscale_A3_GL5_report.md`
- `multiscale_A3_GL5_summary.json`

## A3 engineering

- `A3_terminal_newton_optimization.md`
- `A3_nested_dyadic_single_dda.md`
- `A3_streaming_incremental_dda.md`
- `A3_transport_safe_tie_envelope_fixed.md`
- `A3_transactional_event_validation.md`

## Transport-aware LOD

- `transport_aware_mean_compensation_pilot.md`
- `transport_aware_cumulant_pilot.md`
- `transport_aware_local_mean_pilot.md`
- `transport_aware_conditional_mean_pilot.md`
- `transport_aware_relu_operator_pilot.md`

## Image-space pilot

- `single_scattering_image_benchmark.md`
- `single_scattering_image_benchmark.png`
- `single_scattering_image_benchmark.json`

---

# 21. Suggested Initial Prompt for Claude

Copy the following into Claude together with this report:

> You are joining an ongoing research project on exact and transport-aware rendering of procedural participating media. Treat the attached handoff as authoritative unless a result is explicitly marked preliminary or invalid. Preserve the research roadmap and restate it every few interactions. Do not restart from GLSL or generic ray marching. Continue from the current A3-GN architecture: streaming single finest-grid DDA, integer axis/octave crossing transactions, GL5 exact full-segment integration, and one terminal degree-9 polynomial build followed by safeguarded degree-10 Newton inversion. Your immediate task is to design and implement a statistically meaningful full-float32 transactional transport validation harness, while keeping float64 as the ground-truth hierarchy. Distinguish event-topology mismatch from optical-depth, transmittance, and event-position error. Do not reclassify cells from approximate event times.

---

# 22. Final Status

The exact float64 theory and traversal architecture are in strong condition.

The most important recent engineering result is:

\[
\boxed{
\text{single finest-grid DDA}
+
\text{integer axis/octave transactions}
}
\]

This converts A3 from a multi-stream theoretical traversal into a compact streaming kernel architecture.

The open problem is no longer whether exact multiscale traversal is mathematically possible. It is now:

\[
\boxed{
\text{How much of the exact topology must be retained in float32
to preserve transport quantities within a controlled error budget?}
}
\]

That is the correct next research question.
