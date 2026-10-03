import math
import numpy as np
from psrt.angular import DOMGrid
from psrt.atmosphere import Atmosphere

def plane_transport_solver(
    r_grid: np.ndarray,
    J: np.ndarray,
    grid: DOMGrid,
    atmosphere: Atmosphere,
    L_top: np.ndarray,
    L_bottom: np.ndarray
) -> np.ndarray:
    """
    Solve the plane-parallel formal transport equation on the radial/height grid
    using the Piecewise Linear Exponential (Short Characteristics) integrator.
    
    Parameters:
        r_grid: 1D array of radial coordinates [M]
        J: source function array [M, num_directions] (phase integral only)
        grid: DOMGrid angular grid
        atmosphere: Atmosphere instance
        L_top: incoming downward boundary [num_directions] at grid[-1]
        L_bottom: incoming upward boundary [num_directions] at grid[0]
    Returns:
        L: radiance field [M, num_directions]
    """
    M = len(r_grid)
    num_dirs = len(grid.directions)
    L = np.zeros((M, num_dirs))
    
    # Precompute vertical optical depths
    taus = np.array([atmosphere.optical_depth_vertical(r) for r in r_grid])
    
    for d_idx, d_vec in enumerate(grid.directions):
        mu = d_vec.z
        
        if mu > 0.0:
            # ------------------------------------------------------------------
            # UPWARD PROPAGATION (mu > 0): Sweep from bottom (k=0) to top (k=M-1)
            # ------------------------------------------------------------------
            L[0, d_idx] = L_bottom[d_idx]
            for k in range(1, M):
                # dtau > 0 since taus is decreasing with height
                dtau = taus[k-1] - taus[k]
                Y = dtau / mu
                
                # Piecewise linear integration coefficients
                if Y < 1e-5:
                    w0 = Y * (1.0 - 0.5 * Y)
                    w1 = Y * Y * (0.5 - Y / 3.0)
                else:
                    exp_Y = math.exp(-Y)
                    w0 = 1.0 - exp_Y
                    w1 = 1.0 - (1.0 + Y) * exp_Y
                    
                coef_k = w0 - w1 / Y
                coef_prev = w1 / Y
                
                # Add emission along the shell layer
                I_k = J[k, d_idx] * coef_k + J[k-1, d_idx] * coef_prev
                L[k, d_idx] = L[k-1, d_idx] * math.exp(-Y) + I_k
                
        elif mu < 0.0:
            # ------------------------------------------------------------------
            # DOWNWARD PROPAGATION (mu < 0): Sweep from top (k=M-1) to bottom (k=0)
            # ------------------------------------------------------------------
            mu_abs = -mu
            L[M-1, d_idx] = L_top[d_idx]
            for k in range(M - 2, -1, -1):
                # dtau > 0 since taus is increasing downwards
                dtau = taus[k] - taus[k+1]
                Y = dtau / mu_abs
                
                # Piecewise linear integration coefficients
                if Y < 1e-5:
                    w0 = Y * (1.0 - 0.5 * Y)
                    w1 = Y * Y * (0.5 - Y / 3.0)
                else:
                    exp_Y = math.exp(-Y)
                    w0 = 1.0 - exp_Y
                    w1 = 1.0 - (1.0 + Y) * exp_Y
                    
                coef_k = w0 - w1 / Y
                coef_next = w1 / Y
                
                # Add emission along the shell layer
                I_k = J[k, d_idx] * coef_k + J[k+1, d_idx] * coef_next
                L[k, d_idx] = L[k+1, d_idx] * math.exp(-Y) + I_k
        else:
            # mu == 0.0 (horizontal path: decoupled flat plane boundary)
            L[:, d_idx] = 0.0
            
    return L
