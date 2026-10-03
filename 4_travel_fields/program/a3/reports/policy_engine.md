# Precision-Aware Transport Lowering Policy Engine

Compiler layer that decides how to lower a procedural operator to a precision/partition policy,
DECOUPLED from the frozen A3-GBox64/F32 production primitive (`trace_production` is never touched).

- Engine: `a3/a3/compiler/policy.py`
- Validation: `a3/a3/tests/validate_policy.py`

## Interface

Input: node record `N = (op_class, sigma_t, V_min, V_max, L, E_repr, E_continuous, E_event,
clusters, repr_boundary)` + tolerances `(eps_tau_norm, eps_T, eps_hit)`.
Output: `Decision(action, op_class, E_continuous, E_jump, E_event, E_total, eps_tau, certified,
reason)` where action in
`{ACCEPT_F32, REFINE_ROOTS, RAISE_COEFF_PRECISION, RAISE_TOPOLOGY_PRECISION, RAYMARCH_FALLBACK}`.

Transport budget: `E_total = E_continuous + sigma_t * sum_i J_i w_i`; `eps_tau = eps_tau_norm *
sigma_t`. Refinement floor uses cluster `R_floor` (best width after maximal refinement).

## Lowering rules (by continuity class)

- **C0/C1 value-continuous** (ReLU, abs, clamp, min, max): jump-free; `E_cont <= eps_tau` ->
  ACCEPT_F32 (Lipschitz certificate), else RAISE_COEFF_PRECISION.
- **P0 analytic-roots** (hard/masked threshold, CSG): `E_total <= eps_tau` -> ACCEPT; else if
  refinable to budget (`E_cont + sigma sum J R_floor <= eps_tau`) -> REFINE_ROOTS; else
  RAISE_COEFF_PRECISION.
- **P1 representation-sensitive** (sparse brick, cell hash, LOD): `E_total <= eps_tau` -> ACCEPT
  (transport-budget principle); else enumerable boundary -> RAISE_TOPOLOGY_PRECISION; black-box
  -> RAYMARCH_FALLBACK.
- Event-distance failure (`E_event > eps_T`) escalates to at least RAISE_COEFF_PRECISION.

## Requirements met

1. Decisions use uncertainty-aware clusters `(w, J, R_floor)`, never exact multiplicity;
   `m_eff` is a diagnostic label only.
2. **Monotone**: property-tested over 4000 random nodes x 6 hardening axes (tighter eps_tau /
   eps_T, larger E_continuous / E_event / cluster width / jump) -> 0 de-escalations.
3. Three separate tolerances (eps_tau_norm, eps_T, eps_hit).
4. `certified` flag propagated from cluster enclosures; heuristic-radius decisions are flagged
   `certified=False` ("empirical policy") -- the hook for a Bernstein/interval enclosure backend.
5. P1 split: enumerable boundary -> topology precision; black-box -> raymarch fallback.
6. Full decision trace (class, E_continuous, E_jump, E_event, action, reason).
7. **Zero accepted violations** (primary bar): oracle confusion matrix over 3000 cases with
   conservative estimates -> 0 accepted violations; unnecessary escalation 3.1% (secondary).

## Validation result

    7/7 regression gates passed
    4000/4000 monotone (0 violations)
    oracle: 3000 cases, 0 accepted violations, 3.1% unnecessary escalation
    POLICY ENGINE VALIDATION PASSED

## Open (next)

- Replace heuristic cluster radii with certified interval enclosures (Bernstein exclusion +
  interval Newton/Krawczyk + Taylor-model radii) so P0/P1 decisions carry `certified=True`.
- Connect to the frozen A3 primitive: derive node records from the real degree-9/10 density
  polynomials produced by `trace_production`, and run the oracle confusion matrix against actual
  A3 transport ground truth.
- Cost/quality closure: predicted vs actual cost per action; reduce unnecessary fallback.
