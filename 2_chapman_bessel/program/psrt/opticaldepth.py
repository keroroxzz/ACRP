import numpy as np
from psrt.ray import Ray
from psrt.atmosphere import Atmosphere
from psrt.quadrature import gauss_legendre_rules

def integrate_optical_depth(ray: Ray, d_start: float, d_end: float, atmosphere: Atmosphere, N: int) -> float:
    """
    Integrate optical depth along the ray from s = d_start to s = d_end.
    tau = integral_{d_start}^{d_end} beta(|ray.position(s)|) ds
    using N-point Gauss-Legendre quadrature.
    """
    if d_start >= d_end:
        return 0.0
        
    nodes, weights = gauss_legendre_rules(d_start, d_end, N)
    
    # Calculate position and extinction coefficient beta at each node
    # Ray.position(s) returns a Vec3, and Vec3.norm() computes the radius r.
    radii = [ray.position(s).norm() for s in nodes]
    betas = np.array([atmosphere.beta(r) for r in radii])
    
    # Numerical integration sum
    tau = np.sum(betas * weights)
    return float(tau)
