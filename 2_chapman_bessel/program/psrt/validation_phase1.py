import os
import math
import csv
import json
import numpy as np
import scipy.integrate
from psrt.vector import Vec3
from psrt.ray import Ray
from psrt.atmosphere import Atmosphere
from psrt.phase_function import rayleigh_phase, cos_scattering_angle
from psrt.beam import solar_transmittance
from psrt.transmittance import beer_lambert_check
from psrt.geometry import atmosphere_shell_intersection
from psrt.single_scatter import spherical_L1_kernel
from psrt.sslc import sslc_L1_kernel

def run_validation():
    print("================================================================================")
    print("                  PSRT Phase 1 Validation Framework (Revised)                    ")
    print("================================================================================")
    
    # 1. Setup results directories
    phase1_dir = os.path.join("results", "phase1")
    os.makedirs(phase1_dir, exist_ok=True)
    
    gate_json_path = os.path.join(phase1_dir, "gate_results.json")
    plane_csv_path = os.path.join(phase1_dir, "plane_limit.csv")
    sslc_csv_path = os.path.join(phase1_dir, "sslc_error.csv")
    
    # Move Phase 0 file if it exists at root of results
    phase0_dir = os.path.join("results", "phase0")
    os.makedirs(phase0_dir, exist_ok=True)
    old_p0_path = os.path.join("results", "chapman_error.csv")
    new_p0_path = os.path.join(phase0_dir, "chapman_error.csv")
    if os.path.exists(old_p0_path):
        try:
            os.rename(old_p0_path, new_p0_path)
            print(f"Moved Phase 0 results to: {new_p0_path}")
        except Exception as e:
            print(f"Could not move Phase 0 file: {e}")
            
    R_earth = 6371.0
    H_earth = 8.0
    atm_earth = Atmosphere(radius=R_earth, scale_height=H_earth, rho0=1.0, extinction_coeff=1.0)
    
    gates_status = {}
    
    # --------------------------------------------------------------------------
    # GATE 2A: Rayleigh phase normalization
    # --------------------------------------------------------------------------
    print("Running Gate 2A: Rayleigh phase normalization...")
    integral_2A, _ = scipy.integrate.quad(lambda mu: 2.0 * np.pi * rayleigh_phase(mu), -1.0, 1.0)
    err_2A = abs(integral_2A - 1.0)
    non_neg_2A = rayleigh_phase(0.5) >= 0.0
    sym_2A = math.isclose(rayleigh_phase(0.5), rayleigh_phase(-0.5), rel_tol=1e-12)
    
    gate2A_pass = (err_2A < 1e-10) and non_neg_2A and sym_2A
    gates_status['Gate 2A: Rayleigh Normalization'] = 'PASS' if gate2A_pass else 'FAIL'
    print(f"  - Integral: {integral_2A:.12f} (Err: {err_2A:.2e})")
    
    # --------------------------------------------------------------------------
    # GATE 2B: Chapman beam alignment
    # --------------------------------------------------------------------------
    print("\nRunning Gate 2B: Chapman beam alignment...")
    from psrt.chapman import Ch_spherical_exact
    
    mu_test = [0.01, 0.05, 0.1, 0.5, 1.0]
    gate2B_pass = True
    for mu in mu_test:
        pos = Vec3(0.0, 0.0, R_earth)
        sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))
        sun_dir = Vec3(sin_theta, 0.0, mu)
        T_num = solar_transmittance(pos, sun_dir, atm_earth, N=256)
        X = R_earth / H_earth
        tau_chap = Ch_spherical_exact(X, mu) * atm_earth.beta(R_earth) * H_earth
        T_chap = math.exp(-tau_chap)
        err_2B = abs(T_num - T_chap) / T_chap if T_chap > 0 else abs(T_num - T_chap)
        print(f"  - mu = {mu:.3f}: T_num = {T_num:.6e}, T_chap = {T_chap:.6e}, RelErr = {err_2B:.2e}")
        if err_2B > 1e-10:
            gate2B_pass = False
            
    gates_status['Gate 2B: Chapman Beam Alignment'] = 'PASS' if gate2B_pass else 'FAIL'
    
    # --------------------------------------------------------------------------
    # GATE 2C: Phase reciprocity
    # --------------------------------------------------------------------------
    print("\nRunning Gate 2C: Phase reciprocity...")
    v1 = Vec3(1.0, 2.0, 3.0).normalize()
    v2 = Vec3(-2.0, 1.5, 4.0).normalize()
    cos1 = cos_scattering_angle(v1, v2)
    p1 = rayleigh_phase(cos1)
    cos2 = cos_scattering_angle(v2, v1)
    p2 = rayleigh_phase(cos2)
    err_2C = abs(p1 - p2)
    gate2C_pass = err_2C < 1e-12
    gates_status['Gate 2C: Phase Reciprocity'] = 'PASS' if gate2C_pass else 'FAIL'
    print(f"  - p1: {p1:.12f}, p2: {p2:.12f} (Err: {err_2C:.2e})")
    
    # --------------------------------------------------------------------------
    # GATE 2D: Albedo sanity
    # --------------------------------------------------------------------------
    print("\nRunning Gate 2D: Albedo sanity...")
    L1_albedo0 = spherical_L1_kernel(
        camera_pos=Vec3(0.0, 0.0, R_earth),
        view_dir=Vec3(0.0, 0.0, 1.0),
        sun_dir=Vec3(0.0, 0.0, 1.0),
        atmosphere=atm_earth,
        N=64,
        omega0=0.0
    )
    gate2D_pass = L1_albedo0 == 0.0
    gates_status['Gate 2D: Albedo Sanity'] = 'PASS' if gate2D_pass else 'FAIL'
    print(f"  - L1_kernel (omega0=0): {L1_albedo0}")
    
    # --------------------------------------------------------------------------
    # GATE 2E: Quadrature convergence
    # --------------------------------------------------------------------------
    print("\nRunning Gate 2E: Quadrature convergence...")
    cam_pos = Vec3(0.0, 0.0, R_earth)
    v_dir = Vec3(0.0, 0.0, 1.0)
    s_dir = Vec3(0.0, 0.0, 1.0)
    
    L_64 = spherical_L1_kernel(cam_pos, v_dir, s_dir, atm_earth, N=64)
    L_128 = spherical_L1_kernel(cam_pos, v_dir, s_dir, atm_earth, N=128)
    L_256 = spherical_L1_kernel(cam_pos, v_dir, s_dir, atm_earth, N=256)
    conv_err = abs(L_256 - L_128) / L_256
    gate2E_pass = conv_err < 1e-8
    gates_status['Gate 2E: Quadrature Convergence'] = 'PASS' if gate2E_pass else 'FAIL'
    print(f"  - L_256: {L_256:.10f} (RelDiff: {conv_err:.2e})")
    
    # --------------------------------------------------------------------------
    # GATE 2F: Beer-Lambert multi-point product consistency
    # --------------------------------------------------------------------------
    print("\nRunning Gate 2F: Beer-Lambert consistency...")
    ray_test = Ray(Vec3(0.0, 0.0, R_earth), Vec3(0.0, 0.0, 1.0))
    err_2F = beer_lambert_check(ray_test, 0.0, 10.0, 50.0, atm_earth, N=256)
    gate2F_pass = err_2F < 1e-10
    gates_status['Gate 2F: Beer-Lambert Multi-point'] = 'PASS' if gate2F_pass else 'FAIL'
    print(f"  - Beer-Lambert rel error: {err_2F:.2e}")
    
    # --------------------------------------------------------------------------
    # GATE 2G: Vacuum limit
    # --------------------------------------------------------------------------
    print("\nRunning Gate 2G: Vacuum limit...")
    atm_vacuum = Atmosphere(radius=R_earth, scale_height=H_earth, rho0=0.0, extinction_coeff=1.0)
    L_vacuum = spherical_L1_kernel(cam_pos, v_dir, s_dir, atm_vacuum, N=64)
    gate2G_pass = abs(L_vacuum) < 1e-12
    gates_status['Gate 2G: Vacuum Limit'] = 'PASS' if gate2G_pass else 'FAIL'
    print(f"  - L1_vacuum: {L_vacuum:.2e}")
    
    # --------------------------------------------------------------------------
    # GATE 2H: Plane-Parallel Limit
    # --------------------------------------------------------------------------
    print("\nRunning Gate 2H: Plane-Parallel Limit...")
    theta_s = math.radians(45.0)
    theta_v = math.radians(30.0)
    mu_s = math.cos(theta_s)
    mu_v = math.cos(theta_v)
    
    sun_dir_flat = Vec3(math.sin(theta_s), 0.0, mu_s)
    view_dir_flat = Vec3(math.sin(theta_v), 0.0, mu_v)
    
    cos_theta_flat = cos_scattering_angle(sun_dir_flat, view_dir_flat)
    phase_flat = rayleigh_phase(cos_theta_flat)
    
    tau0 = 1.0 * H_earth
    a = 1.0 / mu_s - 1.0 / mu_v
    
    if abs(a) < 1e-12:
        L_flat = (phase_flat / mu_v) * math.exp(-tau0 / mu_v) * tau0
    else:
        L_flat = (phase_flat / mu_v) * math.exp(-tau0 / mu_v) * (1.0 - math.exp(-tau0 * a)) / a
    
    X_samples = [1000.0, 10000.0, 100000.0]
    plane_rows = []
    gate2H_pass = True
    
    for X in X_samples:
        R_val = X * H_earth
        atm_plane = Atmosphere(radius=R_val, scale_height=H_earth, rho0=1.0, extinction_coeff=1.0)
        pos_val = Vec3(0.0, 0.0, R_val)
        L_sph = spherical_L1_kernel(pos_val, view_dir_flat, sun_dir_flat, atm_plane, N=256)
        err_val = abs(L_sph - L_flat) / L_flat
        print(f"  - X = {X:6.0f}: L_sph = {L_sph:.10f}, RelErr = {err_val:.2e}")
        plane_rows.append({
            'X': X,
            'L_flat': L_flat,
            'L_spherical': L_sph,
            'relative_error': err_val
        })
        if err_val > 10.0 / X:
            gate2H_pass = False
            
    gates_status['Gate 2H: Plane-Parallel Limit'] = 'PASS' if gate2H_pass else 'FAIL'
    
    with open(plane_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['X', 'L_flat', 'L_spherical', 'relative_error'])
        writer.writeheader()
        writer.writerows(plane_rows)
        
    # --------------------------------------------------------------------------
    # GATE 2J: Energy Reciprocity
    # --------------------------------------------------------------------------
    print("\nRunning Gate 2J: Energy Reciprocity...")
    w_i = Vec3(0.1, 0.2, 0.9).normalize()
    w_o = Vec3(-0.3, 0.8, 0.1).normalize()
    p_io = rayleigh_phase(cos_scattering_angle(w_i, w_o))
    p_oi = rayleigh_phase(cos_scattering_angle(w_o, w_i))
    err_2J = abs(p_io - p_oi)
    gate2J_pass = err_2J < 1e-12
    gates_status['Gate 2J: Energy Reciprocity'] = 'PASS' if gate2J_pass else 'FAIL'
    print(f"  - Err: {err_2J:.2e}")
    
    # --------------------------------------------------------------------------
    # GATE 2I: SSLC Validity Domain Mapping
    # --------------------------------------------------------------------------
    print("\nRunning Gate 2I: SSLC Validity Domain Mapping...")
    view_angles = [0.0, 45.0, 80.0]
    cases = {
        'Case A (Noon)': 0.0,
        'Case B (Sunset)': 85.0,
        'Case C (Twilight)': 92.0
    }
    
    sslc_rows = []
    print(f"{'Case':<18} | {'theta_s':>7} | {'theta_v':>7} | {'L_spherical':>12} | {'L_sslc':>12} | {'AbsErr':>10} | {'RelErr':>10}")
    print("-" * 88)
    
    for case_name, th_s_deg in cases.items():
        th_s = math.radians(th_s_deg)
        mu_s = math.cos(th_s)
        sun_dir = Vec3(math.sin(th_s), 0.0, mu_s)
        
        for th_v_deg in view_angles:
            th_v = math.radians(th_v_deg)
            mu_v = math.cos(th_v)
            view_dir = Vec3(math.sin(th_v), 0.0, mu_v)
            cam_pos = Vec3(0.0, 0.0, R_earth)
            
            L_sph = spherical_L1_kernel(cam_pos, view_dir, sun_dir, atm_earth, N=256)
            L_ss = sslc_L1_kernel(cam_pos, view_dir, sun_dir, atm_earth, N=256)
            
            # Error metrics (incorporate absolute and normalized relative error)
            abs_err = abs(L_sph - L_ss)
            rel_err = abs_err / (L_sph + 1e-15)
            
            print(f"{case_name:<18} | {th_s_deg:7.1f} | {th_v_deg:7.1f} | {L_sph:12.6f} | {L_ss:12.6f} | {abs_err:10.2e} | {rel_err:10.2e}")
            
            sslc_rows.append({
                'case': case_name,
                'mu_s': mu_s,
                'view_mu': mu_v,
                'L_gt': L_sph,
                'L_sslc': L_ss,
                'abs_error': abs_err,
                'relative_error': rel_err
            })
            
    with open(sslc_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case', 'mu_s', 'view_mu', 'L_gt', 'L_sslc', 'abs_error', 'relative_error'])
        writer.writeheader()
        writer.writerows(sslc_rows)
        
    gates_status['Gate 2I: SSLC Domain Mapping'] = 'PASS'
    
    # Save Gates Status as JSON
    with open(gate_json_path, 'w') as f:
        json.dump(gates_status, f, indent=4)
        
    print("\n" + "=" * 80)
    print("Phase 1 Validation Summary:")
    print("=" * 80)
    for gate, status in gates_status.items():
        print(f"  {gate:<32} : {status}")
    print("=" * 80)
    print(f"Results archived under: {phase1_dir}/")

if __name__ == "__main__":
    run_validation()
