import numpy as np
from typing import Callable, List, Tuple
from psrt.vector import Vec3
from psrt.angular import DOMGrid
from psrt.atmosphere import Atmosphere
from psrt.phase_function import cos_scattering_angle
from psrt.source import compute_phase_matrix, scattering_source

def solve_sos(
    r_grid: np.ndarray,
    grid: DOMGrid,
    atmosphere: Atmosphere,
    phase_function: Callable[[float], float],
    transport_solver: Callable[[np.ndarray, np.ndarray, DOMGrid, Atmosphere, np.ndarray, np.ndarray], np.ndarray],
    T_s: np.ndarray,  # pre-calculated solar transmittance [M]
    sun_dir: Vec3,  # global sun direction vector
    solar_flux: float = 1.0,
    omega0: float = 1.0,
    max_orders: int = 50,
    tol: float = 1e-8
) -> Tuple[np.ndarray, List[float]]:
    """
    Solve the radiative transfer equation using Successive Orders of Scattering (SOS).
    
    Parameters:
        r_grid: radial coordinates array [M]
        grid: DOMGrid angular grid
        atmosphere: Atmosphere instance
        phase_function: phase function callback
        transport_solver: formal transport solver callback
        T_s: pre-calculated solar transmittance [M]
        sun_dir: sun direction vector
        solar_flux: solar incident flux F0
        omega0: single scattering albedo
        max_orders: maximum scattering order iterations
        tol: relative convergence tolerance
    Returns:
        L_total: converged total radiance field [M, num_directions]
        convergence_history: list of relative changes per order
    """
    M = len(r_grid)
    num_dirs = len(grid.directions)
    
    # 1. Precompute Phase Function Matrix
    phase_matrix = compute_phase_matrix(grid, phase_function)
    
    # 2. Setup boundary arrays for higher orders (no external source input)
    L_top_boundary = np.zeros(num_dirs)
    L_bottom_boundary = np.zeros(num_dirs)
    
    # 3. Calculate First Order Scattering Source J_1 [M, num_directions]
    # J_1 = omega0 * F0 * T_s(r) * p(-sun_dir . view_dir)
    J_1 = np.zeros((M, num_dirs))
    for d_idx, d_vec in enumerate(grid.directions):
        cos_theta = cos_scattering_angle(sun_dir, d_vec)
        p_val = phase_function(cos_theta)
        
        # J_1[k] = omega0 * F0 * T_s[k] * p
        J_1[:, d_idx] = omega0 * solar_flux * T_s * p_val
        
    # 4. First Order Transport Integration
    # L_1 = transport(J_1)
    L_1 = transport_solver(r_grid, J_1, grid, atmosphere, L_top_boundary, L_bottom_boundary)
    
    L_total = L_1.copy()
    L_prev = L_1.copy()
    
    convergence_history = []
    
    # 5. Successive Orders loop
    for order in range(2, max_orders + 1):
        # S_n source term (phase integral only)
        J_n = scattering_source(L_prev, grid, phase_matrix, omega0)
        
        # L_n transport solution
        L_n = transport_solver(r_grid, J_n, grid, atmosphere, L_top_boundary, L_bottom_boundary)
        
        # L2 norm convergence check
        norm_L_total = np.linalg.norm(L_total)
        norm_L_n = np.linalg.norm(L_n)
        
        rel_change = norm_L_n / (norm_L_total + 1e-12)
        convergence_history.append(float(rel_change))
        
        L_total += L_n
        L_prev = L_n.copy()
        
        if rel_change < tol:
            break
            
    return L_total, convergence_history
