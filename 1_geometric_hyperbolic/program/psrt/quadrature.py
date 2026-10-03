import numpy as np
from typing import Tuple

def gauss_legendre_rules(a: float, b: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Gauss-Legendre quadrature nodes and weights mapped to the interval [a, b].
    Parameters:
        a: start of integration interval
        b: end of integration interval
        N: number of quadrature points
    Returns:
        nodes: numpy array of mapped quadrature nodes
        weights: numpy array of mapped quadrature weights
    """
    # Get standard Legendre nodes and weights on [-1, 1]
    t, w = np.polynomial.legendre.leggauss(N)
    
    # Map to [a, b]
    half_width = 0.5 * (b - a)
    mid_point = 0.5 * (b + a)
    
    nodes = half_width * t + mid_point
    weights = half_width * w
    
    return nodes, weights
