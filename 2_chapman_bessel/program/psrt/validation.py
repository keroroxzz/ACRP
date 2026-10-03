import os
import math
import csv
import numpy as np
from psrt.vector import Vec3
from psrt.ray import Ray
from psrt.atmosphere import Atmosphere
from psrt.opticaldepth import integrate_optical_depth
from psrt.chapman import Ch_spherical_exact, Ch_exact, Ch_erfcx

def run_validation():
    print("================================================================================")
    print("                      PSRT Phase 0 Validation Framework                         ")
    print("================================================================================")
    
    # 1. Setup results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "chapman_error.csv")
    
    # 2. Parameters
    R = 6371.0  # Planetary radius (km)
    X_samples = [100.0, 800.0, 8000.0]
    mu_samples = [0.0, 0.001, 0.005, 0.01, 0.02, 0.035, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    csv_rows = []
    
    # Print headers
    print(f"{'X':>6} | {'mu':>6} | {'Ch_num':>12} | {'Ch_sph_ex':>12} | {'Ch_exact':>12} | {'Ch_erfcx':>12} | {'Err_1A':>10} | {'Err_1B_ex':>10} | {'Err_1B_erf':>10}")
    print("-" * 115)
    
    gate0_passed = True
    gate1a_passed = True
    
    for X in X_samples:
        H = R / X
        # Create Atmosphere
        atm = Atmosphere(radius=R, scale_height=H, rho0=1.0, extinction_coeff=1.0)
        r_top = atm.r_top
        
        for mu in mu_samples:
            # Origin on the surface (0, 0, R)
            origin = Vec3(0.0, 0.0, R)
            
            # Direction vector: omega = (sqrt(1 - mu^2), 0, mu)
            sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))
            direction = Vec3(sin_theta, 0.0, mu)
            
            ray = Ray(origin, direction)
            
            # --- GATE 0: Geometry check ---
            # Analytical intersection with sphere r_top:
            term_under_sqrt = R * R * mu * mu + r_top * r_top - R * R
            s_analytic = -R * mu + math.sqrt(max(0.0, term_under_sqrt))
            
            # Numerical intersection
            intersections = ray.intersect_sphere(r_top)
            if len(intersections) == 0:
                print(f"Error: Ray did not intersect boundary r_top = {r_top} for X={X}, mu={mu}")
                gate0_passed = False
                s_numeric = 0.0
            else:
                s_numeric = intersections[0]
                geom_err = abs(s_numeric - s_analytic) / s_analytic if s_analytic > 0 else abs(s_numeric - s_analytic)
                if geom_err > 1e-12:
                    print(f"Gate 0 Failed: Geometry relative error = {geom_err:.2e} for X={X}, mu={mu}")
                    gate0_passed = False
            
            # --- GATE 1A: Ray integration vs Ch_spherical_exact ---
            tau_64 = integrate_optical_depth(ray, 0.0, s_numeric, atm, N=64)
            tau_128 = integrate_optical_depth(ray, 0.0, s_numeric, atm, N=128)
            tau_256 = integrate_optical_depth(ray, 0.0, s_numeric, atm, N=256)
            
            # Convergence check
            if tau_256 > 0.0:
                rel_conv = abs(tau_256 - tau_128) / tau_256
            else:
                rel_conv = abs(tau_256 - tau_128)
                
            conv_ok = (rel_conv < 1e-10) or (abs(tau_256 - tau_128) < 1e-12)
            if not conv_ok:
                print(f"Warning: Quadrature did not converge for X={X}, mu={mu}. Rel diff = {rel_conv:.2e}")
            
            # Chapman numeric value = tau / (beta(R) * H) = tau / H
            Ch_num = tau_256 / H
            
            # Exact spherical Chapman value (ground truth physics)
            Ch_sph_ex = Ch_spherical_exact(X, mu)
            
            # Relative error for Gate 1A
            err_1A = abs(Ch_num - Ch_sph_ex) / Ch_sph_ex
            if err_1A > 1e-8:
                print(f"Gate 1A Warning: Rel error = {err_1A:.2e} for X={X}, mu={mu} exceeds 1e-8!")
                gate1a_passed = False
                
            # --- GATE 1B: Taylor approximations ---
            Ch_ex = Ch_exact(X, mu)
            Ch_erf = Ch_erfcx(X, mu)
            
            # Error of asymptotic second-order Chapman integral relative to spherical exact
            err_1B_ex = abs(Ch_ex - Ch_sph_ex) / Ch_sph_ex
            # Error of erfcx approximation relative to spherical exact
            err_1B_erf = abs(Ch_erf - Ch_sph_ex) / Ch_sph_ex
            
            # Print row
            print(f"{X:6.1f} | {mu:6.3f} | {Ch_num:12.6f} | {Ch_sph_ex:12.6f} | {Ch_ex:12.6f} | {Ch_erf:12.6f} | {err_1A:10.2e} | {err_1B_ex:10.2e} | {err_1B_erf:10.2e}")
            
            csv_rows.append({
                'X': X,
                'mu': mu,
                's_top': s_numeric,
                'tau_num': tau_256,
                'Ch_spherical_exact': Ch_sph_ex,
                'Ch_exact': Ch_ex,
                'Ch_erfcx': Ch_erf,
                'err_gate1a': err_1A,
                'err_gate1b_exact': err_1B_ex,
                'err_gate1b_erfcx': err_1B_erf
            })
            
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'X', 'mu', 's_top', 'tau_num', 'Ch_spherical_exact', 
            'Ch_exact', 'Ch_erfcx', 'err_gate1a', 'err_gate1b_exact', 'err_gate1b_erfcx'
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print("-" * 115)
    print(f"Results written to: {csv_path}")
    print("-" * 115)
    
    # Validation summary
    print("\nSummary:")
    print(f"Gate 0 (Geometry Check):                 {'PASS' if gate0_passed else 'FAIL'}")
    print(f"Gate 1A (Ray vs Spherical Exact):        {'PASS' if gate1a_passed else 'FAIL'}")
    
    if gate0_passed and gate1a_passed:
        print("\nAll Phase 0 Gates passed successfully!")
    else:
        print("\nSome Phase 0 Gates failed. Please check the logs.")

if __name__ == "__main__":
    run_validation()
