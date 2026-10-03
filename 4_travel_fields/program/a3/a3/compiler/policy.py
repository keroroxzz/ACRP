"""Precision-aware transport lowering policy engine (compiler layer).

Decoupled from the frozen A3-GBox64/F32 production primitive: this engine only DECIDES how
to lower a procedural operator (partition / raise precision / fall back); it never modifies
`trace_production`. Given a node analysis record and a requested transport tolerance, it emits

    ACCEPT_F32 | REFINE_ROOTS | RAISE_COEFF_PRECISION | RAISE_TOPOLOGY_PRECISION | RAYMARCH_FALLBACK

Design requirements (collaborator-specified):
  1. Decisions use uncertainty-aware root clusters (width w, jump J, refinable floor R_floor),
     NOT exact multiplicity (unstable under quantization). m_eff is a diagnostic label only.
  2. MONOTONE: tightening tolerance, or growing representation error / cluster width / jump,
     can only keep or escalate the action (ACCEPT<REFINE<RAISE_COEFF<RAISE_TOPO<FALLBACK).
  3. THREE separate tolerances: eps_tau_norm (= eps_tau/sigma_t), eps_T (event distance),
     eps_hit -- a small optical-depth error does not guarantee event-distance stability.
  4. Hard/masked (P0) budgets must use CERTIFIED cluster enclosures; decisions built on
     heuristic (non-certified) radii are flagged certified=False ("empirical policy").
  5. Representation-sensitive (P1): enumerable discrete boundary -> RAISE_TOPOLOGY_PRECISION;
     black-box discontinuity -> RAYMARCH_FALLBACK.
  6. Every decision emits a trace (operator class, E_continuous, E_jump, E_event, action, reason).
  7. Success bar: zero ACCEPTED violation (accepted case whose true error exceeds tolerance),
     verified by an oracle confusion matrix; unnecessary fallback is a secondary concern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple


class OperatorClass(IntEnum):
    C0_SMOOTH = 0        # add, scale, smooth modulation, polynomial composition
    C1_CONTINUOUS = 1    # ReLU, abs, clamp, min, max (continuous, nonsmooth)
    P0_ANALYTIC_ROOTS = 2  # hard / masked threshold, analytic CSG branches
    P1_REPRESENTATION = 3  # sparse brick, cell hash, discontinuous LOD, black-box lookup


class Action(IntEnum):
    # Escalation ladder: strictly ordered so monotonicity = non-decreasing index.
    ACCEPT_F32 = 0
    REFINE_ROOTS = 1
    RAISE_COEFF_PRECISION = 2
    RAISE_TOPOLOGY_PRECISION = 3
    RAYMARCH_FALLBACK = 4


@dataclass(frozen=True)
class RootCluster:
    """An uncertainty-aware branch cluster (not an exact root)."""
    w: float           # current certified/estimated uncertain width (world-distance units)
    J: float           # jump magnitude of the operator across this branch
    R_floor: float     # width achievable after maximal refinement at current coeff precision
    certified: bool = False  # True iff w and R_floor come from a certified interval enclosure
    m_eff: int = 1     # effective degeneracy order -- DIAGNOSTIC ONLY, never a correctness gate


@dataclass(frozen=True)
class Tolerances:
    eps_tau_norm: float   # optical-depth tolerance normalized by sigma_t (eps_tau / sigma_t)
    eps_T: float          # free-path event-distance tolerance
    eps_hit: float = 0.0  # hit-classification tolerance (0 => must match exactly)


@dataclass(frozen=True)
class NodeRecord:
    op_class: OperatorClass
    sigma_t: float
    V_min: float
    V_max: float
    L: float                       # Lipschitz / local variation bound
    E_repr: float                  # representation (ray/coeff) error scale
    E_continuous: float            # continuous transport error estimate (optical-depth, absolute)
    E_event: float                 # event-distance error estimate (distance units)
    clusters: Tuple[RootCluster, ...] = ()   # P0 branch clusters
    repr_boundary: str = ""        # P1: "enumerable" | "blackbox"


@dataclass(frozen=True)
class Decision:
    action: Action
    op_class: OperatorClass
    E_continuous: float
    E_jump: float
    E_event: float
    E_total: float
    eps_tau: float
    certified: bool
    reason: str


def _jump_error(sigma_t: float, clusters, use_floor: bool) -> float:
    return sigma_t * sum(c.J * (c.R_floor if use_floor else c.w) for c in clusters)


def decide(node: NodeRecord, tol: Tolerances) -> Decision:
    """Emit a monotone lowering decision with a full error trace."""
    sigma = node.sigma_t
    eps_tau = tol.eps_tau_norm * sigma           # absolute optical-depth budget
    E_cont = node.E_continuous
    E_jump = _jump_error(sigma, node.clusters, use_floor=False)
    E_jump_best = _jump_error(sigma, node.clusters, use_floor=True)
    E_total = E_cont + E_jump
    E_total_best = E_cont + E_jump_best
    certified = all(c.certified for c in node.clusters) if node.clusters else True
    event_ok = node.E_event <= tol.eps_T

    # event failure is escalated to at least a coefficient/terminal precision raise
    event_action = Action.ACCEPT_F32 if event_ok else Action.RAISE_COEFF_PRECISION

    oc = node.op_class
    if oc in (OperatorClass.C0_SMOOTH, OperatorClass.C1_CONTINUOUS):
        if E_cont <= eps_tau:
            tau_action = Action.ACCEPT_F32
            reason = "value-continuous: continuous error within budget (Lipschitz certificate)"
        else:
            tau_action = Action.RAISE_COEFF_PRECISION
            reason = "value-continuous: continuous/coeff error exceeds budget -> raise precision"
    elif oc == OperatorClass.P0_ANALYTIC_ROOTS:
        if E_total <= eps_tau:
            tau_action = Action.ACCEPT_F32
            reason = "analytic jump: continuous+jump budget satisfied"
        elif E_total_best <= eps_tau:
            tau_action = Action.REFINE_ROOTS
            reason = "analytic jump: refinable root clusters can reach budget"
        else:
            tau_action = Action.RAISE_COEFF_PRECISION
            reason = "analytic jump: even maximal refinement insufficient -> raise coeff precision"
    else:  # P1 representation-sensitive
        if E_total <= eps_tau:
            tau_action = Action.ACCEPT_F32
            reason = "representation-sensitive but transport budget satisfied"
        elif node.repr_boundary == "enumerable":
            tau_action = Action.RAISE_TOPOLOGY_PRECISION
            reason = "representation-sensitive, enumerable boundary -> resolve internal topology"
        else:
            tau_action = Action.RAYMARCH_FALLBACK
            reason = "representation-sensitive, black-box discontinuity -> raymarch fallback"

    action = Action(max(int(tau_action), int(event_action)))
    if action == Action.RAISE_COEFF_PRECISION and tau_action != Action.RAISE_COEFF_PRECISION:
        reason = reason + "; event-distance error exceeds eps_T -> raise terminal precision"

    return Decision(action=action, op_class=oc, E_continuous=E_cont, E_jump=E_jump,
                    E_event=node.E_event, E_total=E_total, eps_tau=eps_tau,
                    certified=certified, reason=reason)
