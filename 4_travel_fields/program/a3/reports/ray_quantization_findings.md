# Ray-Quantization Transport Error: a Grazing-Ray Tail with a One-Line Predictor

Follow-up to the Sec.18 float32 harness, which found the **f32 ray** (a physically
different ray) dominates transport error, with a max non-monotonic in F. This experiment
resolves whether that limit is fundamental or concentrated in identifiable rays.

Ground truth: true-f64 pipeline (== kernel). Error metric: `|dtau|` between the f32-ray
(`mixed64`: f32 ray, f64 pipeline) and the true-f64 ray. Geometry predictor:
`graze = min_a |rd_a|` (small => nearly tangent to a cell face).

## Result 1 — the limit is entirely a grazing-ray tail

Per-category `|dtau|` distribution (F in {16,64,128}, 40 rays/category; representative F64):

| category | p50 | p90 | p99 | max | median graze |
|---|---|---|---|---|---|
| random | 2.3e-7 | 6.5e-7 | 9.9e-7 | 1.2e-6 | 0.20 |
| axis_aligned | 3.0e-8 | 1.2e-7 | 1.3e-7 | 1.4e-7 | 0.00 |
| near_axis | 3.1e-8 | 5.4e-8 | 1.0e-7 | 1.2e-7 | 5.8e-5 |
| exact_diagonal | 3.1e-7 | 5.4e-7 | 5.4e-7 | 5.4e-7 | 0.58 |
| near_diagonal | 2.6e-7 | 7.5e-7 | 1.1e-6 | 1.3e-6 | 0.58 |
| **grazing** | **8.1e-8** | **5.4e-5** | **3.5e-4** | **4.3e-4** | 1.0e-4 |

Every non-grazing category is bounded at max ~1.4e-6, F-independent. Grazing has the same
tiny **median** (~8e-8) but a heavy tail: only its top ~10% explode. Every worst-5 ray at
every F is a grazing ray with `graze ~ 1e-4`. Mechanism: f32 direction quantization slides a
near-tangent ray across cell faces, clipping different high-density cells.

## Result 2 — a one-line f64-fallback predictor caps the error

420 mixed rays at F64; apply f64 fallback when `graze < threshold`:

| criterion | rays kept | max \|dtau\| kept |
|---|---|---|
| all rays | 420 | 9.32e-4 |
| graze >= 1e-4 | 244 | 7.08e-4 |
| graze >= 3e-4 | 220 | 4.63e-5 |
| **graze >= 1e-3** | 211 | **1.42e-6** |
| graze >= 3e-3 | 208 | 1.42e-6 |
| graze >= 1e-2 | 206 | 1.42e-6 |

`graze = min_a|rd_a| >= 1e-3` cleanly caps the f32-ray transport error at **1.42e-6** (the
pipeline-precision floor); tightening beyond 1e-3 gives nothing. The predictor is a single
min-abs test on the (already-normalized) direction — negligible cost.

(The ~50% "fallback" fraction here is an artifact of the adversarial ray mix — a full sixth
of rays are deliberately grazing, plus axis/near-axis. In a realistic scene near-tangent
rays are rare, so the true fallback fraction is small. Note exactly axis-aligned rays have
graze=0 yet small error, so the rule is safe-but-conservative: it over-flags the cheap
axis-aligned rays, never under-flags a bad one.)

## Result 3 — tightened guard: "exactly one small direction component"

`min_a|rd_a|` conflates grazing (one tiny component, dangerous) with axis-like rays (two
tiny components, safe), so it over-flags. Requiring the SECOND-smallest component to be
large isolates true grazing (F64+F128, 540 rays, worst |dtau| 1.55e-3):

| predictor | flagged | kept max\|dtau\| |
|---|---|---|
| `m0 < 1e-3` (naive) | 49.8% | 2.98e-5 (misses 3) |
| `m0 < 1e-2` (pure) | 50.0% | 1.63e-6 |
| **`m0 < 1e-2 AND m1 > 1e-2`** | **16.7%** | **1.63e-6** |

`m0 = min_a|rd_a|`, `m1` = 2nd-smallest. The two-threshold "exactly one small component"
rule gives the same safety as pure `m0<1e-2` while flagging **3x fewer** rays. The worst ray
(`m0=4.7e-5, m1=1.75e-2`) is a *double-graze*, so `m1>1e-2` (not `>0.1`) is needed to catch it.

## Result 4 — precision ablation: the minimal high-precision region is the WHOLE decision path

Grazing-heavy rays (F64, 135 rays), each precision component raised in isolation vs the
true-f64 reference:

| mode | max\|dtau\| | tail(>1.5e-6) |
|---|---|---|
| full32 (all f32) | 1.62e-3 | 24 |
| A: f64 ray only | 1.62e-3 | 25 |
| B: f64 box-clip only | 1.62e-3 | 24 |
| C: f64 DDA-decision only | 1.62e-3 | 23 |
| D: f64 box + f64 DDA (f32 ray) | 1.62e-3 | 23 |
| **F: f64 ray + box + DDA (f32 arith)** | **4.29e-6** | 15 |

Raising ANY single component is useless: a grazing side-of-face decision is flipped by the
f32 ray value OR the f32 boundary arithmetic, independently. Only raising the ENTIRE discrete
-decision path together (ray origin/direction + box clipping + DDA plane-times) to f64 kills
the tail (1.6e-3 -> 4.3e-6, ~380x). f32 GL5/Newton then contributes only a residual ~4e-6 on
grazing rays. In particular **D does not suffice** (it keeps the f32 ray).

### Result 5 — generalization: decision64 caps the tail to the f32 floor everywhere

Sweep (one axis at a time from F=64, world=3, sigma=1.2, sparsity=0.35):

| config | full32 tail | decision64 tail | f32 floor |
|---|---|---|---|
| F=16 / 32 / 64 / 128 | 6.1e-4 / 7.2e-4 / 4.4e-4 / 8.8e-5 | 2.4e-6 / 3.0e-6 / 2.6e-6 / 6.0e-6 | 1.4 / 2.6 / 1.9 / 2.9 e-6 |
| world 2 / 4 | 2.1e-4 / 3.1e-5 | 1.5e-6 / 7.3e-6 | 1.2e-6 / 3.2e-6 |
| sigma 0.5 / 3.0 | 1.9e-4 / 1.1e-3 | 1.5e-6 / 1.1e-5 | 1.0e-6 / 5.6e-6 |
| sparsity 0.2 / 0.6 | 4.5e-4 / 1.6e-4 | 3.8e-6 / 6.5e-6 | 2.8e-6 / 1.5e-6 |

`decision64_tail` stays within ~2x of the f32 arithmetic floor in EVERY config while
`full32_tail` is 1-3 orders larger. The floor scales gently with sigma_t and F (f32
accumulation) and decision64 tracks it. So the architecture generalizes with no
threshold-universality caveat.

### Result 6 — box-vs-grid: the tail is a box entry-time shift, not a grid error

Worst 12 rays: **12 box-grazing, 0 grid-grazing.** For every one, the internal crossing
stream is BIT-IDENTICAL between f32 and f64 (same token sequence, same crossing count,
first-divergence = None) and the entry/exit face is unchanged; the entire error is a shift
in the box entry time `t_entry` of ~1e-4..9e-4. Mechanism: for a grazing ray whose grazing
axis is the entry face, `t_entry = (plane - o)/d_graze` amplifies the f32 ray error by
`1/d_graze`, moving the integration-domain start (dtau ~ rho * d_t_entry). The transactional
integer DDA keeps the internal cell sequence exact even in f32.

This refines the minimal region: since the internal grid is f32-robust, only the **ray + box
entry/exit clip** need f64. Mode G (f64 ray + f64 box clip; f32 internal DDA + f32 arith)
matches decision64: full32 tail 1.62e-3 -> G 5.26e-6 ~ decision64 4.29e-6. So the DDA plane-time
arithmetic can stay f32; f64 is required only for the ray and the one box-clip per ray.

### Scope boundary (important)

The "0% internal-grid / box-only f64" conclusion is established for the **frozen, continuous,
shared-corner cubic-fade A3 primitive**. Stress tests on **discontinuous high-contrast
hashed-cell fields** DO produce internal cell-sequence mismatch and large transport error.
So the minimal-precision result must be stated as:

- **continuous A3 primitive: box-only f64** (f64 ray + f64 box entry/exit clip; rest f32);
- **partition-sensitive operators** (hard threshold, ReLU branch, sparse-brick activation,
  discontinuous LOD): may require raising internal topology to higher precision.

### Theory (updated): entry-time amplification, not internal bifurcation

The tail is the box entry-time, not an internal face-order bifurcation:

    t_entry = (p - o_a)/d_a ,
    |d t_entry| <~ (|d o_a| + |t_entry| |d d_a|) / |d_a| ,
    d tau ~ sigma_t * rho(t_entry) * d t_entry ,

amplified by the entry-face grazing measure g_box = |n_entry . d| (small -> large shift).
Because the fix is global and fixed-cost, g_box is kept as a diagnostic, not a runtime guard.

### Two viable architectures
- **Guarded**: full-f32 everywhere + per-ray guard `m0<1e-2 & m1>1e-2` -> f64 fallback (~17%
  flagged in an adversarial mix; far less in a realistic scene). Per-ray branch -> GPU divergence.
- **Global decision-layer f64**: run the ray + DDA + clipping *decision path* in f64 always
  (cheap: a few divisions per crossing), keep the expensive GL5/Newton in f32. No per-ray
  branch, no divergence; kills the tail globally to ~4e-6. Likely the better GPU choice.

## Result 7 — cost closure: the f64 box clip is near-free

The f64 work is fixed at 36 ops/ray (ray normalize + one slab clip); all per-segment work
(f32 DDA + per-octave build + GL5 + terminal Newton) scales with N_seg*L. With FP64 charged a
throughput penalty r (consumer ~32-64, datacenter ~2), and N_seg/L from real traces:

    C_f64clip / C_total  (camera / shadow):
      F16 : 1.79% / 6.72% (r=32),  3.51% / 12.59% (r=64)
      F32 : 0.75% / 2.91%,          1.49% / 5.65%
      F64 : 0.34% / 1.33%,          0.67% / 2.62%
      F128: 0.14% / 0.57%,          0.29% / 1.14%

Camera rays: <1% for F>=32, <=3.5% worst case. Shadow rays: <6% for F>=32. Only the pathology
(short shadow ray + F16 + consumer FP64=1/64) reaches ~12% -- there a double-single (f32x2)
entry-time emulation is cheaper than full f64. Confirmed near-free (typically <1%).
(a3/results/cost_proxy.json)

## Result 8 — direct coefficient-assembly closure

Segment-normalized direct degree-9/degree-10 assembly (no Vandermonde surrogate), vs an f64
normalized reference, over terminal segments at F in {16,64,128}:

| variant | coeff abs | coeff rel | dtau(1) | d_event | it | fb |
|---|---|---|---|---|---|---|
| norm_direct_f32 (Horner) | 1.21e-6 | 2.73e-6 | 2.8e-8 | 3.3e-8 | 2.07 | 0 |
| norm_direct_f32 (compensated) | 1.21e-6 | 2.73e-6 | 2.8e-8 | 3.3e-8 | 2.07 | 0 |
| norm_cast_f32 (f64->cast) | 5.9e-8 | 5.8e-8 | 9.7e-9 | 3.3e-8 | 2.07 | 0 |
| world_direct_f32 | - | - | 2.4e-8 | 9.9e-7 | 1.70 | 0 |

Normalized direct-f32 + ordinary Horner suffices (event 3.3e-8, ~2 iters, 0 fallbacks).
Compensated Horner is bit-identical here (marginal only for long/cancellation segments).
World-distance parametrization is ~30x worse on the event -> normalized s in [0,1] is the
right choice. Direct f32 == f64-cast for the event. The terminal assembly (3.3e-8) is far
below the full-pipeline residual (~4e-6), so it is not the bottleneck. (a3/results/direct_assembly.json)

## Formal mainline

    A3-GN-64 reference  ->  A3-GBox64/F32 production (f64 ray + f64 box clip; rest f32)
      ->  cost closure (<1% typical)  ->  direct coefficient closure (normalized direct f32)
      ->  GPU implementation

## Conclusion (answers handoff Sec.22)

Full-float32 A3-GN preserves transport to ~1e-6 for all rays **except near-tangent ones**.
The exact topology that must be retained in higher precision is specifically the
**grazing-ray face-crossing decision** (rays with `min_a|rd_a| < ~1e-3`), not the general
tie structure. Flag those for an f64 (or higher-precision boundary) path; everything else
runs full-f32 within the pipeline-precision budget. This makes f32 A3-GN transport-viable
with a trivial, well-characterized guard rather than a global precision or tie-envelope policy.
