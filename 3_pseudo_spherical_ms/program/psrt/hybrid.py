"""
Hybrid pseudo-spherical multiple-scattering solver (Route A).

Two branches, switched on the solar zenith angle, both validated to < 5% relative
error on the diffuse flux profile against the converged spherical SOS reference:

  * daytime  (theta_s <~ switch): pseudo-spherical two-stream (Eddington diffuse
    closure driven by the spherical Chapman beam) with a single-scattering (TMS)
    correction that replaces the crude Eddington single scatter with the exact one.
      F = F_TS_full - F_TS_single + F_single_exact
  * twilight (theta_s >~ switch): exact single scatter + k orders of multiple
    scatter on a COARSE angular grid (the twilight diffuse field is MS-dominated
    and smooth, so few ordinates suffice). SS-corrected to keep single scatter exact.
      F = F_single_exact(fine) + (F_coarse_k - F_coarse_single)

The Eddington closure fails in twilight because the field is anisotropic there;
the twilight branch uses real orders of scattering instead. See validation_phase3.
"""
import math
import numpy as np
from typing import Tuple

from psrt.vector import Vec3
from psrt.atmosphere import Atmosphere
from psrt.angular import DOMGrid
from psrt.phase import rayleigh_phase
from psrt.beam import solar_transmittance
from psrt.transport.spherical_vec import spherical_transport_solver_vec
from psrt.sos import solve_sos
from psrt.twostream import solve_twostream_beam


class HybridFluxResult:
    """Diffuse radial flux profiles on the radial grid (index 0 = surface)."""
    def __init__(self, tau, F_up, F_down, branch):
        self.tau = tau          # vertical optical depth at each layer [M]
        self.F_up = F_up        # diffuse upward (escaping) flux [M]
        self.F_down = F_down    # diffuse downward flux [M]
        self.branch = branch    # "daytime" | "twilight"


def diffuse_fluxes(L: np.ndarray, grid: DOMGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Radial up/down diffuse flux from a DOM radiance field on the polar axis.

    The transport convention integrates upstream along +d, so L[k,d] is the
    radiance PROPAGATING along -d; upward-propagating (escaping) photons are the
    looking-directions with mu = d.z < 0, downward-propagating have d.z > 0.
    """
    mu = np.array([d.z for d in grid.directions])
    w = grid.weights
    up = mu < 0.0
    dn = mu > 0.0
    F_up = (L[:, up] * (-mu[up]) * w[up]).sum(axis=1)
    F_down = (L[:, dn] * (mu[dn]) * w[dn]).sum(axis=1)
    return F_up, F_down


def _sos_fluxes(atmosphere, r_grid, sun_dir, omega0, max_orders,
                grid, N_sub, T_s, tol=1e-6):
    """Run SOS to `max_orders` and return (F_up, F_down) diffuse flux profiles."""
    def transport(r, J, gd, atm, lt, lb):
        return spherical_transport_solver_vec(r, J, gd, atm, lt, lb, sun_dir, N_sub=N_sub)

    L, _ = solve_sos(
        r_grid=r_grid, grid=grid, atmosphere=atmosphere,
        phase_function=rayleigh_phase, transport_solver=transport,
        T_s=T_s, sun_dir=sun_dir, solar_flux=1.0, omega0=omega0,
        max_orders=max_orders, tol=tol,
    )
    return diffuse_fluxes(L, grid)


def _solar_transmittance_profile(atmosphere, radii, sun_dir, N=128):
    return np.array([
        solar_transmittance(Vec3(0.0, 0.0, float(r)), sun_dir, atmosphere, N=N)
        for r in radii
    ])


def solar_zenith_angle(sun_dir: Vec3) -> float:
    """Solar zenith angle (radians) for the on-axis column (local vertical = +z)."""
    return math.acos(max(-1.0, min(1.0, sun_dir.z)))


def daytime_flux(atmosphere: Atmosphere, r_grid: np.ndarray, sun_dir: Vec3,
                 omega0: float, N_mu: int = 16, N_phi: int = 32, N_sub: int = 32,
                 n_tau: int = 400, g: float = 0.0) -> HybridFluxResult:
    """Daytime branch: pseudo-spherical two-stream + single-scattering correction."""
    R, H = atmosphere.radius, atmosphere.scale_height
    grid = DOMGrid(N_mu=N_mu, N_phi=N_phi)
    tau = np.array([atmosphere.optical_depth_vertical(r) for r in r_grid])
    Ts = _solar_transmittance_profile(atmosphere, r_grid, sun_dir)

    # exact single scatter (SOS, one order) on the fine grid
    Fu_1, Fd_1 = _sos_fluxes(atmosphere, r_grid, sun_dir, omega0, 1, grid, N_sub, Ts)

    # two-stream on a uniform tau grid; beam = spherical Chapman transmittance
    tau0 = tau[0]
    tg = np.linspace(max(tau.min(), tau0 * math.exp(-18.0)), tau0, n_tau)
    r_of_tau = R + H * np.log(tau0 / tg)
    Ts_tg = _solar_transmittance_profile(atmosphere, r_of_tau, sun_dir)
    ts_full = solve_twostream_beam(tg, Ts_tg, omega0, g=g)              # all orders
    ts_1 = solve_twostream_beam(tg, Ts_tg, omega0, g=g, lam2=3.0)       # single scatter

    def onto_r(arr):
        return np.interp(tau[::-1], tg, arr)[::-1]

    Fu = onto_r(ts_full.F_up) - onto_r(ts_1.F_up) + Fu_1
    Fd = onto_r(ts_full.F_down) - onto_r(ts_1.F_down) + Fd_1
    return HybridFluxResult(tau, Fu, Fd, "daytime")


def twilight_flux(atmosphere: Atmosphere, r_grid: np.ndarray, sun_dir: Vec3,
                  omega0: float, k_orders: int = 6, N_mu: int = 16, N_phi: int = 32,
                  coarse: Tuple[int, int] = (6, 12), N_sub: int = 32) -> HybridFluxResult:
    """Twilight branch: exact single scatter (fine) + k MS orders on a coarse grid."""
    grid_fine = DOMGrid(N_mu=N_mu, N_phi=N_phi)
    grid_coarse = DOMGrid(N_mu=coarse[0], N_phi=coarse[1])
    tau = np.array([atmosphere.optical_depth_vertical(r) for r in r_grid])
    Ts = _solar_transmittance_profile(atmosphere, r_grid, sun_dir)

    Fu_1f, Fd_1f = _sos_fluxes(atmosphere, r_grid, sun_dir, omega0, 1, grid_fine, N_sub, Ts)
    Fu_ck, Fd_ck = _sos_fluxes(atmosphere, r_grid, sun_dir, omega0, k_orders, grid_coarse, N_sub, Ts)
    Fu_1c, Fd_1c = _sos_fluxes(atmosphere, r_grid, sun_dir, omega0, 1, grid_coarse, N_sub, Ts)

    Fu = Fu_1f + (Fu_ck - Fu_1c)
    Fd = Fd_1f + (Fd_ck - Fd_1c)
    return HybridFluxResult(tau, Fu, Fd, "twilight")


def hybrid_flux(atmosphere: Atmosphere, r_grid: np.ndarray, sun_dir: Vec3,
                omega0: float, switch_deg: float = 88.0, **kwargs) -> HybridFluxResult:
    """Full hybrid: pick the branch by solar zenith angle (switch ~88 deg)."""
    theta_s = math.degrees(solar_zenith_angle(sun_dir))
    if theta_s < switch_deg:
        day_keys = ("N_mu", "N_phi", "N_sub", "n_tau", "g")
        return daytime_flux(atmosphere, r_grid, sun_dir, omega0,
                            **{k: v for k, v in kwargs.items() if k in day_keys})
    twi_keys = ("k_orders", "N_mu", "N_phi", "coarse", "N_sub")
    return twilight_flux(atmosphere, r_grid, sun_dir, omega0,
                        **{k: v for k, v in kwargs.items() if k in twi_keys})
