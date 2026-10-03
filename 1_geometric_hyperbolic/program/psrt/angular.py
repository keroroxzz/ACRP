import math
import numpy as np
from typing import List, Tuple
from psrt.vector import Vec3

class DOMGrid:
    """
    A Discrete Ordinates Method (DOM) angular grid.
    Parameters:
        N_mu: number of zenith angle cosine points (Gauss-Legendre)
        N_phi: number of azimuth angle points (uniform)
    """
    def __init__(self, N_mu: int = 16, N_phi: int = 32):
        self.N_mu = N_mu
        self.N_phi = N_phi
        
        # 1. Zenith angle cosines and weights on [-1, 1] using Gauss-Legendre
        # np.polynomial.legendre.leggauss returns points in ascending order
        self.mu, self.w_mu = np.polynomial.legendre.leggauss(N_mu)
        
        # 2. Azimuth angles and uniform weights on [0, 2*pi)
        self.phi = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)
        self.w_phi = 2.0 * np.pi / N_phi
        
        # 3. Compile full 2D direction vectors and weights
        self.directions: List[Vec3] = []
        self.weights: List[float] = []
        
        for i in range(N_mu):
            mu_val = self.mu[i]
            sin_theta = math.sqrt(max(0.0, 1.0 - mu_val * mu_val))
            for j in range(N_phi):
                phi_val = self.phi[j]
                
                # Cartesian direction
                dx = sin_theta * math.cos(phi_val)
                dy = sin_theta * math.sin(phi_val)
                dz = mu_val
                
                # We enforce normalization
                v = Vec3(dx, dy, dz).normalize()
                
                self.directions.append(v)
                self.weights.append(float(self.w_mu[i] * self.w_phi))
                
        self.weights = np.array(self.weights)
                
    def verify_normalization(self, tol: float = 1e-12) -> bool:
        """Verify that the sum of weights equals 4*pi."""
        w_sum = sum(self.weights)
        return abs(w_sum - 4.0 * math.pi) < tol
