import math
import numpy as np
from typing import Callable
from psrt.angular import DOMGrid
from psrt.vector import Vec3

def compute_phase_matrix(grid: DOMGrid, phase_function: Callable[[float], float]) -> np.ndarray:
    """
    Precompute the phase function matrix P[k, m] = p(direction_m . direction_k).
    Size is [num_directions, num_directions].
    """
    dirs = grid.directions
    N = len(dirs)
    P = np.zeros((N, N))
    
    # Extract coordinates for vectorized dot product
    coords = np.array([[d.x, d.y, d.z] for d in dirs])
    
    # Matrix of dot products between all directions
    # dot_matrix[k, m] = dir_k . dir_m
    dot_matrix = coords @ coords.T
    
    # Clip to [-1.0, 1.0] to prevent math.acos/sqrt errors in phase functions
    dot_matrix = np.clip(dot_matrix, -1.0, 1.0)
    
    # Evaluate phase function on the entire dot matrix
    # Vectorize the callback if it isn't already
    v_phase = np.vectorize(phase_function)
    P = v_phase(dot_matrix)
    
    return P

def scattering_source(L_prev: np.ndarray, grid: DOMGrid, phase_matrix: np.ndarray, omega0: float) -> np.ndarray:
    """
    Compute the generalized scattering source term J_n (phase integral only, decoupled from beta):
    J_n(k) = omega0 * sum_m L_prev(m) * P[k, m] * w_m / (4 * pi)
    
    Supports:
        L_prev: 1D array [num_directions] or 2D array [num_radii, num_directions]
    Returns:
        J_n: same shape as L_prev
    """
    weights = grid.weights  # 1D array of weights [num_directions]
    
    # Vectorized integration:
    # J_n = (omega0 / 4pi) * (L_prev * weights) @ P_matrix^T
    # Note: phase_matrix is symmetric, so P^T is same as P.
    
    if L_prev.ndim == 1:
        weighted_L = L_prev * weights
        J_n = omega0 * (weighted_L @ phase_matrix)
    elif L_prev.ndim == 2:
        # weights broadcast along axis 0
        weighted_L = L_prev * weights[np.newaxis, :]
        J_n = omega0 * (weighted_L @ phase_matrix.T)
    else:
        raise ValueError("L_prev must be a 1D or 2D array.")
        
    return J_n
