"""travel: iso-surface / nonlinear-operator cell traversal reference.

SCOPE NOTE (2026-07-18): this package models the DPDF/Gemini *signed-lattice
iso-surface* medium ``d(t) = max(0, V(t) - tau)`` -- NOT the authoritative A3-GN volume
algorithm (which uses nonnegative density, a raw quartic optical depth, and no
root-finding; see ``a3/`` and the A3 handoff). Within the A3 project this maps to the
Sec.13.6 "B5 nonlinear operator" line ``rho = max(0, B - lambda R)``, whose operator
algebra is exactly "max(0, f) is piecewise polynomial, partitioned by the roots of
f = 0" -- which is what the cubic root partition here implements. The grazing-root
robustness finding (companion-matrix vs naive Cardano) belongs to this line.

It remains a CPU high-precision reference and self-validation loop for that cellwise
event computation, independent of the ``psrt`` spherical radiative-transfer package.

Core idea (see ``articles/4. Reasearch_Route.md``):

    fixed-step ray marching  ->  cell traversal (DDA) + analytic integration / root inversion

For a surface, the event is the first root of the clearance polynomial ``C(t)``.
For a volume, the event is the generalized inverse of a monotone optical-depth clock.

Phase A modules:
    poly        -- monomial-basis polynomial utilities (eval, derivative, integral)
    trilinear   -- trilinear density cell: corners -> coefficients -> ray density poly
    opticaldepth-- analytic optical depth A(t) with iso-threshold clamping
    reference   -- independent high-precision reference (adaptive quadrature)
    inversion   -- monotone generalized inverse T_xi (safeguarded Newton + bisection)
    dda         -- 3D voxel traversal (Amanatides-Woo)
    harness     -- random generators, property-based checks, worst-case ray search
"""

__all__ = [
    "poly",
    "trilinear",
    "opticaldepth",
    "reference",
    "inversion",
    "dda",
    "harness",
]
