"""
Phase 3 validation: the hybrid pseudo-spherical multiple-scattering solver.

Gates the two branches (daytime PseudoTS+SS-correction, twilight exact-single +
coarse-grid MS) and the switch against the converged spherical SOS reference.
"""
import os
import csv
import json
import math
import numpy as np

from psrt.vector import Vec3
from psrt.atmosphere import Atmosphere
from psrt.angular import DOMGrid
from psrt.phase import rayleigh_phase
from psrt.beam import solar_transmittance
from psrt.transport.spherical_vec import spherical_transport_solver_vec
from psrt.sos import solve_sos
from psrt.twostream import solve_twostream_beam
from psrt.hybrid import hybrid_flux, daytime_flux, twilight_flux, diffuse_fluxes, solar_zenith_angle

R_EARTH = 6371.0
H_EARTH = R_EARTH / 796.0


def _setup(theta_deg, tau0, omega0):
    atm = Atmosphere(radius=R_EARTH, scale_height=H_EARTH, rho0=1.0, extinction_coeff=tau0 / H_EARTH)
    r_grid = R_EARTH + np.linspace(0.0, 20.0 * H_EARTH, 64)
    th = math.radians(theta_deg)
    sun = Vec3(math.sin(th), 0.0, math.cos(th))
    return atm, r_grid, sun


def _truth_fu(atm, r_grid, sun, omega0):
    grid = DOMGrid(16, 32)
    Ts = np.array([solar_transmittance(Vec3(0, 0, r), sun, atm, 128) for r in r_grid])
    def sph(r, J, gd, a, lt, lb):
        return spherical_transport_solver_vec(r, J, gd, a, lt, lb, sun, N_sub=32)
    L, _ = solve_sos(r_grid=r_grid, grid=grid, atmosphere=atm, phase_function=rayleigh_phase,
                     transport_solver=sph, T_s=Ts, sun_dir=sun, solar_flux=1.0, omega0=omega0,
                     max_orders=80, tol=1e-6)
    Fu, _ = diffuse_fluxes(L, grid)
    return Fu


def _eta(Fu, Fu_tot):
    sig = Fu_tot > 1e-4 * Fu_tot.max()
    return float(np.linalg.norm((Fu - Fu_tot)[sig]) / np.linalg.norm(Fu_tot[sig]))


def _monolithic_twostream_fu(atm, r_grid, sun, omega0, n_tau=400):
    """Plain pseudo-spherical two-stream (no SS-correction), for comparison."""
    tau = np.array([atm.optical_depth_vertical(r) for r in r_grid])
    tau0 = tau[0]
    tg = np.linspace(max(tau.min(), tau0 * math.exp(-18.0)), tau0, n_tau)
    r_of = R_EARTH + H_EARTH * np.log(tau0 / tg)
    Ts_tg = np.array([solar_transmittance(Vec3(0, 0, r), sun, atm, 128) for r in r_of])
    ts = solve_twostream_beam(tg, Ts_tg, omega0)
    return np.interp(tau[::-1], tg, ts.F_up)[::-1]


def run_validation():
    print("=" * 80)
    print("                      PSRT Phase 3 Validation Framework                         ")
    print("=" * 80)
    out_dir = os.path.join("results", "phase3")
    os.makedirs(out_dir, exist_ok=True)
    gates = {}
    rows = []
    omega0 = 0.9

    # ---- Gate 4A: daytime branch accuracy ----
    print("\nGate 4A: daytime branch (PseudoTS + SS-correction) accuracy...")
    day_ok = True
    for th in (0.0, 30.0, 60.0):
        atm, r_grid, sun = _setup(th, 1.0, omega0)
        Fu_tot = _truth_fu(atm, r_grid, sun, omega0)
        res = daytime_flux(atm, r_grid, sun, omega0)
        e = _eta(res.F_up, Fu_tot)
        day_ok = day_ok and (e < 0.05)
        rows.append(["daytime", th, e])
        print(f"  theta_s={th:>4}: eta_C(F_up) = {e*100:5.2f}%")
    gates["Gate 4A: Daytime branch < 5%"] = "PASS" if day_ok else "FAIL"

    # ---- Gate 4B: twilight branch accuracy ----
    print("\nGate 4B: twilight branch (exact single + k=6 coarse MS) accuracy...")
    twi_ok = True
    for th in (90.0, 92.0, 95.0):
        atm, r_grid, sun = _setup(th, 1.0, omega0)
        Fu_tot = _truth_fu(atm, r_grid, sun, omega0)
        res = twilight_flux(atm, r_grid, sun, omega0, k_orders=6, coarse=(6, 12))
        e = _eta(res.F_up, Fu_tot)
        twi_ok = twi_ok and (e < 0.05)
        rows.append(["twilight", th, e])
        print(f"  theta_s={th:>4}: eta_C(F_up) = {e*100:5.2f}%")
    gates["Gate 4B: Twilight branch < 5%"] = "PASS" if twi_ok else "FAIL"

    # ---- Gate 4C: twilight branch decisively beats plain two-stream ----
    print("\nGate 4C: twilight branch beats plain two-stream (theta_s=92)...")
    atm, r_grid, sun = _setup(92.0, 1.0, omega0)
    Fu_tot = _truth_fu(atm, r_grid, sun, omega0)
    e_hy = _eta(twilight_flux(atm, r_grid, sun, omega0).F_up, Fu_tot)
    e_ts = _eta(_monolithic_twostream_fu(atm, r_grid, sun, omega0), Fu_tot)
    gate4c = e_hy < 0.25 * e_ts
    gates["Gate 4C: Twilight beats two-stream"] = "PASS" if gate4c else "FAIL"
    print(f"  hybrid twilight = {e_hy*100:.2f}%   plain two-stream = {e_ts*100:.2f}%  "
          f"(ratio {e_hy/e_ts:.2f})")

    # ---- Gate 4D: switch picks the right branch ----
    print("\nGate 4D: branch switch by solar zenith angle...")
    _, _, sun_day = _setup(30.0, 1.0, omega0)
    _, _, sun_twi = _setup(92.0, 1.0, omega0)
    atmd, rgd, _ = _setup(30.0, 1.0, omega0)
    b_day = hybrid_flux(atmd, rgd, sun_day, omega0, switch_deg=88.0).branch
    b_twi = hybrid_flux(atmd, rgd, sun_twi, omega0, switch_deg=88.0).branch
    gate4d = (b_day == "daytime") and (b_twi == "twilight")
    gates["Gate 4D: Branch switch correct"] = "PASS" if gate4d else "FAIL"
    print(f"  theta_s=30 -> {b_day} ; theta_s=92 -> {b_twi}")

    # ---- Gate 4E: SS-correction improves the daytime branch ----
    print("\nGate 4E: SS-correction improves daytime vs plain two-stream (theta_s=30)...")
    atm, r_grid, sun = _setup(30.0, 1.0, omega0)
    Fu_tot = _truth_fu(atm, r_grid, sun, omega0)
    e_corr = _eta(daytime_flux(atm, r_grid, sun, omega0).F_up, Fu_tot)
    e_mono = _eta(_monolithic_twostream_fu(atm, r_grid, sun, omega0), Fu_tot)
    gate4e = e_corr < e_mono
    gates["Gate 4E: SS-correction helps daytime"] = "PASS" if gate4e else "FAIL"
    print(f"  SS-corrected = {e_corr*100:.2f}%   plain two-stream = {e_mono*100:.2f}%")

    with open(os.path.join(out_dir, "gate_results.json"), "w") as f:
        json.dump(gates, f, indent=4)
    with open(os.path.join(out_dir, "hybrid_error.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["branch", "theta_s", "eta_Fup"])
        wr.writerows(rows)

    print("\n" + "=" * 80)
    print("Phase 3 Validation Summary:")
    print("=" * 80)
    for g, s in gates.items():
        print(f"  {g:<44} : {s}")
    print("=" * 80)


if __name__ == "__main__":
    run_validation()
