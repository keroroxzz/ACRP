import os
import math
import csv
import json
import numpy as np
from psrt.vector import Vec3
from psrt.ray import Ray
from psrt.atmosphere import Atmosphere
from psrt.angular import DOMGrid
from psrt.phase import rayleigh_phase, isotropic_phase, henyey_greenstein_phase
from psrt.phase_function import cos_scattering_angle
from psrt.transform import global_to_local_direction
from psrt.transport.plane import plane_transport_solver
from psrt.transport.spherical import spherical_transport_solver
from psrt.sos import solve_sos
from psrt.twostream import solve_twostream_eddington
from psrt.twilight import compute_twilight_indicator, twilight_hybrid_switch
from psrt.single_scatter import spherical_L1_physical
from psrt.beam import solar_transmittance

def run_validation():
    print("================================================================================")
    print("                      PSRT Phase 2 Validation Framework                         ")
    print("================================================================================")
    
    phase2_dir = os.path.join("results", "phase2")
    os.makedirs(phase2_dir, exist_ok=True)
    
    gate_json_path = os.path.join(phase2_dir, "gate_results.json")
    flux_csv_path = os.path.join(phase2_dir, "flux_conservation.csv")
    twostream_csv_path = os.path.join(phase2_dir, "twostream_comparison.csv")
    
    R_earth = 6371.0
    H_earth = 8.0
    
    gates_status = {}
    
    # --------------------------------------------------------------------------
    # BLOCK 3A: Angular & Phase Foundation
    # --------------------------------------------------------------------------
    print("Running Block 3A: Angular & Phase Foundation...")
    grid = DOMGrid(N_mu=16, N_phi=32)
    
    # Gate 3A-0: DOM weight normalization
    w_sum = sum(grid.weights)
    err_3A0 = abs(w_sum - 4.0 * math.pi)
    gate3A0_pass = err_3A0 < 1e-12
    gates_status['Gate 3A-0: DOM Weights Normalization'] = 'PASS' if gate3A0_pass else 'FAIL'
    print(f"  - Gate 3A-0 (DOM Weights Sum): {w_sum:.12f} (Err: {err_3A0:.2e})")
    
    # Gate 3A-1: Phase function probability normalization (Rayleigh)
    # Integral over 4*pi should equal 1.0. We test for a direction cosine at index 0.
    d_o = grid.directions[0]
    sum_p = 0.0
    for d_idx, d_i in enumerate(grid.directions):
        cos_theta = cos_scattering_angle(d_i, d_o)
        p_val = rayleigh_phase(cos_theta)
        sum_p += p_val * grid.weights[d_idx]
    err_3A1 = abs(sum_p - 1.0)
    gate3A1_pass = err_3A1 < 1e-12
    gates_status['Gate 3A-1: Phase Probability Normalization'] = 'PASS' if gate3A1_pass else 'FAIL'
    print(f"  - Gate 3A-1 (Rayleigh Phase Sum): {sum_p:.12f} (Err: {err_3A1:.2e})")
    
    # Gate 3A-2: Phase reciprocity
    # p(w_i, w_o) = p(w_o, w_i)
    d_test1 = Vec3(0.5, 0.2, 0.8)
    d_test2 = Vec3(-0.1, 0.9, 0.3)
    p_io = rayleigh_phase(cos_scattering_angle(d_test1, d_test2))
    p_oi = rayleigh_phase(cos_scattering_angle(d_test2, d_test1))
    err_3A2 = abs(p_io - p_oi)
    gate3A2_pass = err_3A2 < 1e-12
    gates_status['Gate 3A-2: Phase Reciprocity'] = 'PASS' if gate3A2_pass else 'FAIL'
    print(f"  - Gate 3A-2 (Phase reciprocity): Err: {err_3A2:.2e}")
    
    # Gate 3A-3: DOM Reciprocity Matrix symmetry
    # P_ij = p(w_i, w_j) should be symmetric.
    P_matrix = np.zeros((len(grid.directions), len(grid.directions)))
    for i, d_i in enumerate(grid.directions):
        for j, d_j in enumerate(grid.directions):
            cos_theta = cos_scattering_angle(d_i, d_j)
            P_matrix[i, j] = rayleigh_phase(cos_theta)
    err_3A3 = np.max(np.abs(P_matrix - P_matrix.T))
    gate3A3_pass = err_3A3 < 1e-12
    gates_status['Gate 3A-3: DOM Reciprocity Matrix'] = 'PASS' if gate3A3_pass else 'FAIL'
    print(f"  - Gate 3A-3 (DOM Reciprocity Matrix symmetry): MaxDiff: {err_3A3:.2e}")
    
    # --------------------------------------------------------------------------
    # BLOCK 3B: Plane-Parallel SOS Reference Solver
    # --------------------------------------------------------------------------
    print("\nRunning Block 3B: Plane-Parallel SOS Reference Solver...")
    X_plane = 100000.0
    R_plane = X_plane * H_earth
    # tau0 = 1.0 (extinction_coeff = 1/H_earth)
    atm_plane = Atmosphere(radius=R_plane, scale_height=H_earth, rho0=1.0, extinction_coeff=1.0/H_earth)
    
    r_grid = R_plane + np.linspace(0.0, 20.0 * H_earth, 256)
    z_grid = r_grid - R_plane
    taus = np.array([atm_plane.optical_depth_vertical(r) for r in r_grid])
    tau0 = taus[0]
    
    # Sun zenith theta_s = 45 deg => mu_s = cos(45)
    theta_s = math.radians(45.0)
    mu_s = math.cos(theta_s)
    sun_dir = Vec3(math.sin(theta_s), 0.0, mu_s)
    
    T_s_plane = np.exp(-taus / mu_s)
    
    # Run Plane SOS solver with Rayleigh scattering, albedo = 1.0 (conservative)
    L_total_plane, history = solve_sos(
        r_grid=r_grid,
        grid=grid,
        atmosphere=atm_plane,
        phase_function=rayleigh_phase,
        transport_solver=plane_transport_solver,
        T_s=T_s_plane,
        sun_dir=sun_dir,
        solar_flux=1.0,
        omega0=1.0,
        max_orders=100,
        tol=1e-8
    )
    
    # Compute boundary fluxes for Plane SOS
    F_up_plane = np.zeros(len(r_grid))
    F_down_plane = np.zeros(len(r_grid))
    
    for k in range(len(r_grid)):
        up_sum = 0.0
        down_sum = 0.0
        for d_idx, d_vec in enumerate(grid.directions):
            mu_ij = d_vec.z
            w_ij = grid.weights[d_idx]
            L_val = L_total_plane[k, d_idx]
            if mu_ij > 0.0:
                up_sum += L_val * mu_ij * w_ij
            elif mu_ij < 0.0:
                down_sum += L_val * abs(mu_ij) * w_ij
        F_up_plane[k] = up_sum
        F_down_plane[k] = down_sum
        
    F_top_diffuse = F_up_plane[-1]
    F_bottom_diffuse = F_down_plane[0]
    F_in = 1.0 * mu_s
    F_direct = F_in * math.exp(-tau0 / mu_s)
    
    F_out = F_top_diffuse + F_bottom_diffuse + F_direct
    err_energy = abs(F_out - F_in) / F_in
    gate3B0_pass = err_energy < 1e-3
    gates_status['Gate 3B-0: Plane SOS Energy Conservation'] = 'PASS' if gate3B0_pass else 'FAIL'
    
    gate3B1_pass = len(history) < 100 or history[-1] < 1e-6
    gates_status['Gate 3B-1: SOS Convergence Rate'] = 'PASS' if gate3B1_pass else 'FAIL'
    
    print(f"  - Gate 3B-0 (Energy Conservation):")
    print(f"    F_in:          {F_in:.8f}")
    print(f"    F_top(diff):   {F_top_diffuse:.8f}")
    print(f"    F_bottom(diff):{F_bottom_diffuse:.8f}")
    print(f"    F_direct:      {F_direct:.8f}")
    print(f"    F_out (total): {F_out:.8f} (Err: {err_energy:.2e})")
    print(f"  - Gate 3B-1 (SOS convergence loops): {len(history)} (Last delta: {history[-1]:.2e})")
    
    # Save flux profile
    with open(flux_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['height_z', 'tau', 'F_up', 'F_down'])
        for k in range(len(r_grid)):
            writer.writerow([z_grid[k], taus[k], F_up_plane[k], F_down_plane[k]])
            
    # Gate 3B-2: Analytic Isotropic Single Scatter (Gate 3H)
    # Formula conditions: observer at TOA (k=-1), no surface reflection, sun zenith mu_s=1.0, albedo omega0 = 0.8
    mu_s_zenith = 1.0
    sun_zenith = Vec3(0.0, 0.0, 1.0)
    T_s_zenith = np.exp(-taus / mu_s_zenith)
    omega0_test = 0.8
    
    # First order isotropic source function: J_1 = omega0 * F0 * T_s * (1/(4*pi))
    J_1_iso = np.zeros((len(r_grid), len(grid.directions)))
    for d_idx in range(len(grid.directions)):
        # isotropic phase function value is 1/(4*pi)
        J_1_iso[:, d_idx] = omega0_test * 1.0 * T_s_zenith * isotropic_phase(1.0)
        
    L_1_iso = plane_transport_solver(
        r_grid=r_grid,
        J=J_1_iso,
        grid=grid,
        atmosphere=atm_plane,
        L_top=np.zeros(len(grid.directions)),
        L_bottom=np.zeros(len(grid.directions))
    )
    
    # Pick an upward direction to verify (e.g. zenith angle cosine mu = 0.617)
    test_d_idx = -1
    for d_idx, d_vec in enumerate(grid.directions):
        if d_vec.z > 0.5:
            test_d_idx = d_idx
            break
    mu_test = grid.directions[test_d_idx].z
    
    # Analytic formula: L_1_analytic
    L_1_analytic = (omega0_test * 1.0 / (4.0 * math.pi)) * (mu_s_zenith / (mu_test + mu_s_zenith)) * (
        1.0 - math.exp(-tau0 * (1.0 / mu_s_zenith + 1.0 / mu_test))
    )
    L_1_sos_val = L_1_iso[-1, test_d_idx]
    err_3B2 = abs(L_1_sos_val - L_1_analytic) / L_1_analytic
    gate3B2_pass = err_3B2 < 1e-3
    gates_status['Gate 3B-2: Analytic Isotropic Single Scatter'] = 'PASS' if gate3B2_pass else 'FAIL'
    print(f"  - Gate 3B-2 (Analytic Isotropic Single Scatter vs SOS):")
    print(f"    L1_SOS:      {L_1_sos_val:.8e}")
    print(f"    L1_analytic:  {L_1_analytic:.8e} (RelErr: {err_3B2:.2e})")
    
    # --------------------------------------------------------------------------
    # BLOCK 3C: Spherical SOS & Continuous Alignment
    # --------------------------------------------------------------------------
    print("\nRunning Block 3C: Spherical SOS & Continuous Alignment...")
    # Set extinction_coeff = 0.1 so that vertical optical thickness is moderate (tau0 = 0.8)
    atm_earth = Atmosphere(radius=R_earth, scale_height=H_earth, rho0=1.0, extinction_coeff=0.1)
    r_grid_earth = R_earth + np.linspace(0.0, 20.0 * H_earth, 64)
    
    sun_dir_earth = Vec3(0.0, 0.0, 1.0)  # sun at zenith
    T_s_earth = np.zeros(len(r_grid_earth))
    for k in range(len(r_grid_earth)):
        pos_k = Vec3(0.0, 0.0, r_grid_earth[k])
        T_s_earth[k] = solar_transmittance(pos_k, Vec3(0.0, 0.0, 1.0), atm_earth, N=128)
        
    sph_solver_wrapped = lambda r, J_arr, gd, atm, lt, lb: spherical_transport_solver(
        r, J_arr, gd, atm, lt, lb, sun_dir_earth
    )
    
    L_total_sph, _ = solve_sos(
        r_grid=r_grid_earth,
        grid=grid,
        atmosphere=atm_earth,
        phase_function=rayleigh_phase,
        transport_solver=sph_solver_wrapped,
        T_s=T_s_earth,
        sun_dir=sun_dir_earth,
        solar_flux=1.0,
        omega0=1.0,
        max_orders=1,
        tol=1e-8
    )
    
    view_idx = -1
    for d_idx, d_vec in enumerate(grid.directions):
        if d_vec.z > 0.5:
            view_idx = d_idx
            break
            
    view_dir = grid.directions[view_idx]
    # L_sos IS physical radiance (the phase function is normalized to integrate to 1,
    # so J_1 = omega0*F0*Ts*p carries no extra 1/4pi). Compare directly against the
    # now-corrected physical single-scatter integral -- a non-circular check.
    L1_sos = L_total_sph[0, view_idx]
    
    # Phase 1 ground truth single scatter continuous integration
    L1_single = spherical_L1_physical(
        camera_pos=Vec3(0.0, 0.0, R_earth),
        view_dir=view_dir,
        sun_dir=Vec3(0.0, 0.0, 1.0),
        atmosphere=atm_earth,
        N=256,
        solar_flux=1.0,
        omega0=1.0
    )
    
    err_3CA = abs(L1_sos - L1_single) / L1_single
    gate3CA_pass = err_3CA < 5e-2
    gates_status['Gate 3C-A: L1 Alignment (Same angles)'] = 'PASS' if gate3CA_pass else 'FAIL'
    
    # Full DOM L1 alignment
    sos_avg_L1 = np.mean(L_total_sph[0, :])
    single_vals = []
    for d_vec in grid.directions:
        if d_vec.z > 0.0:
            val = spherical_L1_physical(Vec3(0.0, 0.0, R_earth), d_vec, Vec3(0.0, 0.0, 1.0), atm_earth, N=128, solar_flux=1.0, omega0=1.0)
            single_vals.append(val)
        else:
            single_vals.append(0.0)
    single_avg_L1 = np.mean(single_vals)
    
    err_3CB = abs(sos_avg_L1 - single_avg_L1) / single_avg_L1 if single_avg_L1 > 0 else abs(sos_avg_L1 - single_avg_L1)
    gate3CB_pass = err_3CB < 5e-2
    gates_status['Gate 3C-B: L1 Alignment (Full DOM)'] = 'PASS' if gate3CB_pass else 'FAIL'
    
    print(f"  - Gate 3C-A (Same angles consistency):")
    print(f"    L1_SOS (norm): {L1_sos:.8e}")
    print(f"    L1_single:     {L1_single:.8e} (RelErr: {err_3CA:.2e})")
    print(f"  - Gate 3C-B (Full DOM average consistency):")
    print(f"    DOM Avg SOS:   {sos_avg_L1:.8e}, Single Avg: {single_avg_L1:.8e} (RelErr: {err_3CB:.2e})")
    
    # --------------------------------------------------------------------------
    # BLOCK 3D: Approximation Models
    # --------------------------------------------------------------------------
    print("\nRunning Block 3D: Approximation Models...")
    # Gate 3D-0: Two-stream vs Plane SOS
    ts_res = solve_twostream_eddington(
        tau_grid=np.linspace(0.0, tau0, len(r_grid)),
        mu_s=mu_s,
        solar_flux=1.0,
        omega0=1.0,
        g=0.0,
        As=0.0
    )
    
    F_top_ts = ts_res.F_up[0]
    F_surface_ts = ts_res.F_down[-1]
    
    err_F_top = abs(F_top_ts - F_top_diffuse) / F_top_diffuse
    err_F_surf = abs(F_surface_ts - F_bottom_diffuse) / F_bottom_diffuse
    
    gate3D0_pass = (err_F_top < 0.05) and (err_F_surf < 0.05)
    gates_status['Gate 3D-0: Two-Stream vs Plane SOS'] = 'PASS' if gate3D0_pass else 'FAIL'
    
    print(f"  - Gate 3D-0 (Two-stream vs Plane SOS fluxes):")
    print(f"    Two-Stream F_top: {F_top_ts:.8f} (RelDiff: {err_F_top:.2e})")
    print(f"    Two-Stream F_surf:{F_surface_ts:.8f} (RelDiff: {err_F_surf:.2e})")
    
    # Save TwoStream comparison data
    with open(twostream_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['height_z', 'tau', 'F_up_SOS', 'F_down_SOS', 'F_up_TwoStream', 'F_down_TwoStream'])
        for k in range(len(r_grid)):
            ts_idx = len(r_grid) - 1 - k
            writer.writerow([z_grid[k], taus[k], F_up_plane[k], F_down_plane[k], ts_res.F_up[ts_idx], ts_res.F_down[ts_idx]])
            
    # Gate 3D-1: Twilight Switch validation (Gate 3E)
    pos_surf = Vec3(0.0, 0.0, R_earth)
    th_s_92 = math.radians(92.0)
    sun_92 = Vec3(math.sin(th_s_92), 0.0, math.cos(th_s_92))
    
    lambda_tw = compute_twilight_indicator(pos_surf, sun_92, atm_earth)
    mode = twilight_hybrid_switch(pos_surf, sun_92, atm_earth, threshold=20.0)
    
    gate3D1_pass = (lambda_tw > 20.0) and (mode == 'SOS')
    gates_status['Gate 3D-1: Twilight Hybrid Switch'] = 'PASS' if gate3D1_pass else 'FAIL'
    print(f"  - Gate 3D-1 (Twilight indicator & switch):")
    print(f"    Twilight indicator (theta_s=92): {lambda_tw:.2f}")
    print(f"    Hybrid switch mode:              {mode} (expected: SOS)")
    
    # --------------------------------------------------------------------------
    # BLOCK 3E: Numerical Convergence
    # --------------------------------------------------------------------------
    print("\nRunning Block 3E: Numerical Convergence...")
    
    # Gate 3E-0: Radial grid refinement & Richardson Extrapolation (Gate 3I)
    # Compute Plane SOS on M=128 and M=256
    def run_sos_M(M_points):
        r_grid_M = R_plane + np.linspace(0.0, 20.0 * H_earth, M_points)
        taus_M = np.array([atm_plane.optical_depth_vertical(r) for r in r_grid_M])
        T_s_M = np.exp(-taus_M / mu_s)
        L_M, _ = solve_sos(
            r_grid=r_grid_M, grid=grid, atmosphere=atm_plane, phase_function=rayleigh_phase,
            transport_solver=plane_transport_solver, T_s=T_s_M, sun_dir=sun_dir,
            solar_flux=1.0, omega0=1.0, max_orders=40, tol=1e-8
        )
        # return upward escaping radiance in zenith test direction
        return L_M[-1, test_d_idx]
        
    L_128 = run_sos_M(128)
    L_256 = run_sos_M(256)
    
    # Richardson Extrapolation for second-order convergence:
    L_inf = L_256 + (L_256 - L_128) / 3.0
    err_richardson = abs(L_inf - L_256) / L_inf
    gate3E0_pass = err_richardson < 1e-3
    gates_status['Gate 3E-0: Grid Richardson Extrapolation'] = 'PASS' if gate3E0_pass else 'FAIL'
    
    print(f"  - Gate 3E-0 (Richardson Extrapolation):")
    print(f"    L_128:        {L_128:.8e}")
    print(f"    L_256:        {L_256:.8e}")
    print(f"    L_inf:        {L_inf:.8e}")
    print(f"    Rel Truncation Error: {err_richardson:.2e}")
    
    # Gate 3E-1: DOM Angular Convergence (Gate 3J)
    # We compare N_mu=8, N_phi=16 (DOM 128) vs N_mu=16, N_phi=32 (DOM 512)
    # on a moderate optical depth (tau0 = 0.5 => extinction = 0.5/H) on M=32 grid for speed.
    grid_128 = DOMGrid(N_mu=8, N_phi=16)
    grid_512 = DOMGrid(N_mu=16, N_phi=32)
    
    atm_conv = Atmosphere(radius=R_plane, scale_height=H_earth, rho0=1.0, extinction_coeff=0.5/H_earth)
    r_grid_conv = R_plane + np.linspace(0.0, 20.0 * H_earth, 32)
    taus_conv = np.array([atm_conv.optical_depth_vertical(r) for r in r_grid_conv])
    T_s_conv = np.exp(-taus_conv / mu_s)
    
    def run_conv_flux(grid_obj, phase_func):
        L_res, _ = solve_sos(
            r_grid=r_grid_conv, grid=grid_obj, atmosphere=atm_conv, phase_function=phase_func,
            transport_solver=plane_transport_solver, T_s=T_s_conv, sun_dir=sun_dir,
            solar_flux=1.0, omega0=1.0, max_orders=30, tol=1e-8
        )
        # Compute upward flux at the top of the atmosphere
        F_up_top = 0.0
        for d_idx, d_vec in enumerate(grid_obj.directions):
            if d_vec.z > 0.0:
                F_up_top += L_res[-1, d_idx] * d_vec.z * grid_obj.weights[d_idx]
        return F_up_top
        
    print(f"  - Gate 3E-1 (DOM Angular Convergence):")
    # Test A: Isotropic
    F_128_iso = run_conv_flux(grid_128, isotropic_phase)
    F_512_iso = run_conv_flux(grid_512, isotropic_phase)
    err_iso = abs(F_128_iso - F_512_iso) / F_512_iso if F_512_iso > 0 else 0
    print(f"    Isotropic: DOM128: {F_128_iso:.8f}, DOM512: {F_512_iso:.8f} (RelDiff: {err_iso:.2e})")
    
    # Test B: Rayleigh
    F_128_ray = run_conv_flux(grid_128, rayleigh_phase)
    F_512_ray = run_conv_flux(grid_512, rayleigh_phase)
    err_ray = abs(F_128_ray - F_512_ray) / F_512_ray if F_512_ray > 0 else 0
    print(f"    Rayleigh:  DOM128: {F_128_ray:.8f}, DOM512: {F_512_ray:.8f} (RelDiff: {err_ray:.2e})")
    
    # Test C: Henyey-Greenstein (g=0.5)
    hg_05 = lambda cos_t: henyey_greenstein_phase(cos_t, 0.5)
    F_128_hg = run_conv_flux(grid_128, hg_05)
    F_512_hg = run_conv_flux(grid_512, hg_05)
    err_hg = abs(F_128_hg - F_512_hg) / F_512_hg if F_512_hg > 0 else 0
    print(f"    HG(g=0.5): DOM128: {F_128_hg:.8f}, DOM512: {F_512_hg:.8f} (RelDiff: {err_hg:.2e})")
    
    gate3E1_pass = (err_iso < 5e-3) and (err_ray < 5e-3) and (err_hg < 5e-2)
    gates_status['Gate 3E-1: DOM Angular Convergence'] = 'PASS' if gate3E1_pass else 'FAIL'
    
    # --------------------------------------------------------------------------
    # SAVE & PRINT RESULTS
    # --------------------------------------------------------------------------
    with open(gate_json_path, 'w') as f:
        json.dump(gates_status, f, indent=4)
        
    print("\n" + "=" * 80)
    print("Phase 2 Validation Summary:")
    print("=" * 80)
    for gate, status in gates_status.items():
        print(f"  {gate:<40} : {status}")
    print("=" * 80)

if __name__ == "__main__":
    run_validation()
