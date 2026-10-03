# A3-GBox64/F32 Production Policy + Operator Taxonomy

## (a) Frozen production policy and regression gate

**Policy (`a3_gn_precision.trace_production`, config `A3-GBox64/F32`):**
f64 ray normalize + f64 box entry/exit slab clip; internal transactional DDA, direct
segment-normalized degree-9 assembly, degree-10 optical-depth Newton -- all f32. This is the
minimal high-precision region for the CONTINUOUS shared-corner cubic-fade A3 primitive.

**Gate (`a3/tests/validate_production.py`)** -- TRANSPORT-based, not event-list-identity
(handoff Sec.9.2). Over 6 media (F in {16,64,128}, sigma in {1.2,3.0}, sparsity in
{0.35,0.5,0.65} incl. plateau/low-density) x {camera, box-grazing, long} rays it asserts:

- max |dtau| / sigma_t  < 1.5e-5   (observed ~5-8e-6; NORMALIZED by sigma_t because the f32
  floor scales with the optical-depth magnitude, tau ~ sigma_t -- raw |dtau| reached 1.6e-5 at
  sigma=3.0, i.e. 5.4e-6 per unit sigma, so an unnormalized absolute tolerance is wrong)
- max |d event|         < 1.5e-5   (observed ~3e-6; distance, gated absolutely)
- zero hit/escape mismatch
- terminal Newton iterations bounded (<= 16; observed <= 3)

**Transaction-list divergence is reported, NOT asserted zero.** Box-only production (f32
internal DDA) differs from the f64 crossing list in ~50% of rays, but this is
transport-neutral (Sec.9.2: transport-equivalent partition != event-list identity). Requiring
zero list-identity would force `decision64` (f64 internal DDA) at higher per-segment cost;
the continuous primitive does not need it. The gate forbids regression to world-distance
polynomials or Vandermonde fitting by construction (production uses the normalized direct
builder only).

## (b) Operator taxonomy validation

Three operator classes applied to the continuous density rho(t), under a controlled ray
perturbation eps (models the representation error). Transport error
E = sigma_t * integral |op(rho_pert) - op(rho_orig)| dt, bound = Lipschitz part +
jump part sigma_t * sum_i J_i |B_i| (J_i = jump magnitude, B_i = branch-flip interval).

| eps | ReLU E_p50 | hard E_p50 | brick/sparse E_p50 | all E <= bound |
|---|---|---|---|---|
| 1e-4 | 7.6e-4 | 1.5e-3 | 7.2e-2 | yes |
| 1e-3 | 6.8e-3 | 1.7e-2 | 6.1e-1 | yes |
| 1e-2 | 5.6e-2 | 1.8e-1 | 1.05e0 | yes |

**Findings:**
- **ReLU (value-continuous, 1-Lipschitz):** E scales linearly with eps (~10x/decade),
  jump-independent, and stays under the pure Lipschitz bound. Branch-sequence differences do
  NOT hurt transport -> box-only64 is sufficient.
- **hard / masked threshold (jump-sensitive analytic operator):** E ~ 2-3x ReLU; the excess is
  exactly the jump contribution sigma_t * J |B| with J ~ theta (masked: J ~ rho). Error is
  governed by the uncertain root-cluster width |B| times the jump, not by exact multiplicity.
  Compiler policy: cluster budget -> accept / refine / raise precision.
- **brick / sparse activation (representation-sensitive):** E is ~100x ReLU because the branch
  is on discrete cell identity, flipping at every perturbed cell boundary (J ~ rho each). A
  continuous / Lipschitz bound is INSUFFICIENT; this class needs branch-root partition or
  higher internal-topology precision -- a value-continuous bound cannot certify it.

**Validated inequality:** E_tau,jump <= sigma_t * sum_i J_i |B_i| holds as the jump part of the
total bound for all classes; ReLU is bounded by the continuous (Lipschitz) part alone.

**Compiler distinction (confirmed):**

    value-continuous operator (ReLU)      -> box-only64, Lipschitz certificate
    jump-sensitive operator (hard/masked) -> root-cluster budget (accept/refine/raise)
    representation-sensitive (sparse/brick)-> internal-topology precision / branch-root partition

Artifacts: a3/tests/validate_production.py, a3/harness/partition_sensitivity.py,
a3/results/partition_sensitivity.json.
