"""Regression + property + oracle validation for the lowering policy engine.

1. Fixed regression gates: each operator scenario maps to its required action.
2. Monotonicity property test: hardening any input (tighter tolerance, larger continuous /
   event / cluster-width / jump) never de-escalates the action.
3. Oracle confusion matrix: cases carry a KNOWN true transport error and a CONSERVATIVE node
   record (estimates >= truth); the engine must produce ZERO accepted violations, and we
   measure the (secondary) unnecessary-escalation / fallback rates.

Run:  python a3/tests/validate_policy.py   (exit nonzero on failure)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from a3.compiler.policy import (  # noqa: E402
    OperatorClass, Action, RootCluster, Tolerances, NodeRecord, decide,
)

SIGMA = 1.2


def _node(op_class, *, E_cont=0.0, E_event=0.0, clusters=(), boundary="", Vmax=1.0):
    return NodeRecord(op_class=op_class, sigma_t=SIGMA, V_min=0.0, V_max=Vmax,
                      L=1.0, E_repr=1e-7, E_continuous=E_cont, E_event=E_event,
                      clusters=tuple(clusters), repr_boundary=boundary)


def regression_gates():
    tol = Tolerances(eps_tau_norm=1e-5, eps_T=1e-5)
    eps_tau = tol.eps_tau_norm * SIGMA  # 1.2e-5
    cases = []
    # 1. ReLU (value-continuous), tiny continuous error -> ACCEPT
    cases.append(("relu_accept", _node(OperatorClass.C1_CONTINUOUS, E_cont=1e-7),
                  Action.ACCEPT_F32))
    # 2. well-separated hard roots: tiny w -> jump error << budget -> ACCEPT
    c = RootCluster(w=1e-7, J=0.5, R_floor=1e-8, certified=True)
    cases.append(("hard_separated_accept", _node(OperatorClass.P0_ANALYTIC_ROOTS,
                  E_cont=1e-7, clusters=[c]), Action.ACCEPT_F32))
    # 3. near-multiple hard roots: E_total>budget but refinable to budget -> REFINE
    c = RootCluster(w=5e-5, J=0.5, R_floor=1e-7, certified=True)   # jump 0.5*5e-5*1.2=3e-5>1.2e-5
    cases.append(("hard_near_multiple_refine", _node(OperatorClass.P0_ANALYTIC_ROOTS,
                  E_cont=1e-7, clusters=[c]), Action.REFINE_ROOTS))
    # 4. cluster budget still over after maximal refine -> RAISE_COEFF
    c = RootCluster(w=5e-4, J=0.8, R_floor=5e-5, certified=True)  # best 0.8*5e-5*1.2=4.8e-5>1.2e-5
    cases.append(("hard_refine_insufficient_raise", _node(OperatorClass.P0_ANALYTIC_ROOTS,
                  E_cont=1e-7, clusters=[c]), Action.RAISE_COEFF_PRECISION))
    # 5. sparse brick, enumerable boundary, over budget -> RAISE_TOPOLOGY
    c = RootCluster(w=1e-3, J=1.0, R_floor=1e-3, certified=True)
    cases.append(("sparse_enumerable_topology", _node(OperatorClass.P1_REPRESENTATION,
                  E_cont=1e-7, clusters=[c], boundary="enumerable"), Action.RAISE_TOPOLOGY_PRECISION))
    # 6. sparse black-box, over budget -> FALLBACK
    cases.append(("sparse_blackbox_fallback", _node(OperatorClass.P1_REPRESENTATION,
                  E_cont=1e-2, boundary="blackbox"), Action.RAYMARCH_FALLBACK))
    # 7. continuous ok on tau but event error too large -> RAISE_COEFF
    cases.append(("event_failure_raise", _node(OperatorClass.C1_CONTINUOUS,
                  E_cont=1e-7, E_event=1e-3), Action.RAISE_COEFF_PRECISION))

    fails = []
    for name, node, expected in cases:
        d = decide(node, tol)
        if d.action != expected:
            fails.append("%s: got %s expected %s" % (name, d.action.name, expected.name))
    return fails, len(cases)


def monotonicity(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(n):
        oc = OperatorClass(rng.integers(4))
        nc = rng.integers(0, 3)
        clusters = [RootCluster(w=10 ** rng.uniform(-7, -2), J=rng.uniform(0, 1),
                                R_floor=10 ** rng.uniform(-8, -4),
                                certified=bool(rng.integers(2))) for _ in range(nc)]
        clusters = [RootCluster(w=max(c.w, c.R_floor), J=c.J, R_floor=c.R_floor,
                                certified=c.certified) for c in clusters]
        boundary = rng.choice(["enumerable", "blackbox"])
        base = _node(oc, E_cont=10 ** rng.uniform(-8, -3), E_event=10 ** rng.uniform(-8, -3),
                     clusters=clusters, boundary=boundary)
        tol = Tolerances(eps_tau_norm=10 ** rng.uniform(-6, -4), eps_T=10 ** rng.uniform(-6, -4))
        a0 = decide(base, tol).action
        # harden one axis at a time; action must not decrease
        harder = []
        harder.append((base, Tolerances(tol.eps_tau_norm * 0.3, tol.eps_T)))          # tighter tau
        harder.append((base, Tolerances(tol.eps_tau_norm, tol.eps_T * 0.3)))          # tighter event
        h = NodeRecord(**{**base.__dict__, "E_continuous": base.E_continuous * 3})
        harder.append((h, tol))
        h = NodeRecord(**{**base.__dict__, "E_event": base.E_event * 5})
        harder.append((h, tol))
        if clusters:
            wc = [RootCluster(w=c.w * 4, J=c.J, R_floor=c.R_floor, certified=c.certified) for c in clusters]
            h = NodeRecord(**{**base.__dict__, "clusters": tuple(wc)})
            harder.append((h, tol))
            jc = [RootCluster(w=c.w, J=min(1.0, c.J * 3 + 0.1), R_floor=c.R_floor, certified=c.certified) for c in clusters]
            h = NodeRecord(**{**base.__dict__, "clusters": tuple(jc)})
            harder.append((h, tol))
        for hn, ht in harder:
            if int(decide(hn, ht).action) < int(a0):
                fails += 1
    return fails, n


def confusion_matrix(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    counts = dict(accept=0, accepted_violation=0, escalate=0, unnecessary_escalation=0,
                  fallback=0)
    for _ in range(n):
        tol = Tolerances(eps_tau_norm=10 ** rng.uniform(-6, -4), eps_T=1e-4)
        eps_tau = tol.eps_tau_norm * SIGMA
        family = rng.choice(["relu", "hard", "sparse"])
        if family == "relu":
            true_err = SIGMA * 10 ** rng.uniform(-9, -5)  # continuous, small
            node = _node(OperatorClass.C1_CONTINUOUS, E_cont=true_err * rng.uniform(1.0, 1.3))
        elif family == "hard":
            w_true = 10 ** rng.uniform(-7, -3)
            J = rng.uniform(0.2, 1.0)
            true_err = SIGMA * J * w_true
            w_cons = w_true * rng.uniform(1.0, 1.5)             # conservative upper bound
            c = RootCluster(w=w_cons, J=J, R_floor=10 ** rng.uniform(-8, -5), certified=True)
            node = _node(OperatorClass.P0_ANALYTIC_ROOTS, E_cont=1e-8, clusters=[c])
        else:  # sparse representation-sensitive
            w_true = 10 ** rng.uniform(-5, -2)
            J = rng.uniform(0.3, 1.0)
            true_err = SIGMA * J * w_true
            c = RootCluster(w=w_true * rng.uniform(1.0, 1.4), J=J,
                            R_floor=w_true, certified=True)
            node = _node(OperatorClass.P1_REPRESENTATION, E_cont=1e-8, clusters=[c],
                         boundary=rng.choice(["enumerable", "blackbox"]))
        d = decide(node, tol)
        fits = true_err <= eps_tau
        if d.action == Action.ACCEPT_F32:
            counts["accept"] += 1
            if not fits:
                counts["accepted_violation"] += 1
        else:
            counts["escalate"] += 1
            if d.action == Action.RAYMARCH_FALLBACK:
                counts["fallback"] += 1
            if fits:
                counts["unnecessary_escalation"] += 1
    return counts, n


def main() -> int:
    print("=== policy regression gates ===")
    fails, ncase = regression_gates()
    for f in fails:
        print("  FAIL", f)
    print("  %d/%d gates passed" % (ncase - len(fails), ncase))

    print("=== monotonicity property ===")
    mfail, mn = monotonicity()
    print("  %d/%d monotone (violations %d)" % (mn - mfail, mn, mfail))

    print("=== oracle confusion matrix ===")
    counts, cn = confusion_matrix()
    print("  cases=%d accept=%d escalate=%d fallback=%d" % (cn, counts["accept"],
          counts["escalate"], counts["fallback"]))
    print("  ACCEPTED VIOLATIONS = %d   (must be 0)" % counts["accepted_violation"])
    print("  unnecessary escalations = %d (%.1f%% of escalations, secondary)"
          % (counts["unnecessary_escalation"],
             100.0 * counts["unnecessary_escalation"] / max(counts["escalate"], 1)))

    ok = (len(fails) == 0 and mfail == 0 and counts["accepted_violation"] == 0)
    print("POLICY ENGINE VALIDATION PASSED" if ok else "POLICY ENGINE VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
