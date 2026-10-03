import math
import numpy as np
from psrt.vector import Vec3
from psrt.angular import DOMGrid
from psrt.atmosphere import Atmosphere
from psrt.ray import Ray
from psrt.quadrature import gauss_legendre_rules
from psrt.geometry import atmosphere_shell_intersection
from psrt.transform import global_to_local_direction

def trilinear_interpolate_J(
    r_val: float,
    mu_val: float,
    phi_val: float,
    r_grid: np.ndarray,
    grid: DOMGrid,
    J_reshaped: np.ndarray
) -> float:
    """
    Perform trilinear interpolation of source function J on [r, mu, phi] space.
    J_reshaped: [M, N_mu, N_phi]
    """
    M = len(r_grid)
    N_mu = grid.N_mu
    N_phi = grid.N_phi
    
    # 1. Radial index and interpolation weight
    if r_val <= r_grid[0]:
        k0, k1, tr = 0, 0, 0.0
    elif r_val >= r_grid[-1]:
        k0, k1, tr = M - 1, M - 1, 0.0
    else:
        k = np.searchsorted(r_grid, r_val)
        k0, k1 = k - 1, k
        tr = (r_val - r_grid[k0]) / (r_grid[k1] - r_grid[k0])
        
    # 2. Zenith angle index and weight
    if mu_val <= grid.mu[0]:
        i0, i1, tm = 0, 0, 0.0
    elif mu_val >= grid.mu[-1]:
        i0, i1, tm = N_mu - 1, N_mu - 1, 0.0
    else:
        i = np.searchsorted(grid.mu, mu_val)
        i0, i1 = i - 1, i
        tm = (mu_val - grid.mu[i0]) / (grid.mu[i1] - grid.mu[i0])
        
    # 3. Azimuth angle index and weight (periodic in [0, 2*pi))
    phi_val = phi_val % (2.0 * math.pi)
    d_phi = grid.w_phi  # spacing between uniform azimuth points
    idx_phi = phi_val / d_phi
    j0 = int(math.floor(idx_phi)) % N_phi
    j1 = (j0 + 1) % N_phi
    tp = idx_phi - math.floor(idx_phi)
    
    # 4. Trilinear interpolation sum
    val = (1.0 - tr) * (1.0 - tm) * (1.0 - tp) * J_reshaped[k0, i0, j0] + \
          (1.0 - tr) * (1.0 - tm) * tp         * J_reshaped[k0, i0, j1] + \
          (1.0 - tr) * tm         * (1.0 - tp) * J_reshaped[k0, i1, j0] + \
          (1.0 - tr) * tm         * tp         * J_reshaped[k0, i1, j1] + \
          tr         * (1.0 - tm) * (1.0 - tp) * J_reshaped[k1, i0, j0] + \
          tr         * (1.0 - tm) * tp         * J_reshaped[k1, i0, j1] + \
          tr         * tm         * (1.0 - tp) * J_reshaped[k1, i1, j0] + \
          tr         * tm         * tp         * J_reshaped[k1, i1, j1]
          
    return float(val)

def spherical_transport_solver(
    r_grid: np.ndarray,
    J: np.ndarray,
    grid: DOMGrid,
    atmosphere: Atmosphere,
    L_top: np.ndarray,  # placeholder, unused for Gate 3A (set to 0)
    L_bottom: np.ndarray, # placeholder, unused for Gate 3A (set to 0)
    sun_dir: Vec3,  # global sun direction vector
    N_sub: int = 64
) -> np.ndarray:
    """
    Solve the spherical formal transport equation on the radial coordinates
    using O(N) cumulative path integration.
    
    Parameters:
        r_grid: 1D array of radial coordinates [M]
        J: source function array [M, num_directions] (phase integral only)
        grid: DOMGrid angular grid
        atmosphere: Atmosphere instance
        L_top, L_bottom: boundaries (ignored for Gate 3A black boundary condition)
        sun_dir: global sun direction vector (Vec3)
    Returns:
        L: radiance field [M, num_directions]
    """
    M = len(r_grid)
    num_dirs = len(grid.directions)
    L = np.zeros((M, num_dirs))
    
    # Reshape J for easy trilinear interpolation
    J_reshaped = J.reshape((M, grid.N_mu, grid.N_phi))
    sun_vec = sun_dir
    
    # Iterate through all height levels
    for k in range(M):
        r_k = r_grid[k]
        pos = Vec3(0.0, 0.0, r_k)
        
        # Iterate through all DOM direction vectors
        for d_idx, d_vec in enumerate(grid.directions):
            ray = Ray(pos, d_vec)
            
            # Intersection with boundary shell R to r_top
            intersect_res = atmosphere_shell_intersection(ray, atmosphere)
            if not intersect_res['valid']:
                L[k, d_idx] = 0.0
                continue
                
            s0 = intersect_res['entry_distance']
            s1 = intersect_res['exit_distance']
            
            if s1 <= 0.0:
                L[k, d_idx] = 0.0
                continue
                
            # Perform numerical line integration along ray from s=0 to s1
            nodes, weights = gauss_legendre_rules(0.0, s1, N_sub)
            
            # Precompute positions, norms, and local beta coefficients along nodes
            x_nodes = []
            rs = []
            betas = []
            for s in nodes:
                x = ray.position(s)
                x_nodes.append(x)
                r_s = x.norm()
                rs.append(r_s)
                # Extinction is 0 outside the atmosphere bounds
                if r_s >= atmosphere.radius and r_s <= atmosphere.r_top:
                    betas.append(atmosphere.beta(r_s))
                else:
                    betas.append(0.0)
                    
            # Compute cumulative path optical depth along nodes: tau(0, s_i)
            taus_v = np.zeros(N_sub)
            p_norm = pos.norm()
            beta_start = atmosphere.beta(p_norm) if (p_norm >= atmosphere.radius and p_norm <= atmosphere.r_top) else 0.0
            
            # First node
            taus_v[0] = 0.5 * (beta_start + betas[0]) * nodes[0]
            # Subsequent nodes
            for idx in range(1, N_sub):
                ds = nodes[idx] - nodes[idx-1]
                taus_v[idx] = taus_v[idx-1] + 0.5 * (betas[idx-1] + betas[idx]) * ds
                
            # Integrate the formal solution equation
            integral = 0.0
            for idx in range(N_sub):
                r_s = rs[idx]
                if r_s < atmosphere.radius or r_s > atmosphere.r_top:
                    continue
                    
                beta = betas[idx]
                Tv = math.exp(-taus_v[idx])
                
                # Calculate local angles at the scattering point x
                mu_local, phi_local = global_to_local_direction(x_nodes[idx], d_vec, sun_vec)
                
                # Interpolate source function J at this node
                J_val = trilinear_interpolate_J(r_s, mu_local, phi_local, r_grid, grid, J_reshaped)
                
                integral += weights[idx] * beta * Tv * J_val
                
            L[k, d_idx] = float(integral)
            
    return L
