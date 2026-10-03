import math
from psrt.ray import Ray
from psrt.atmosphere import Atmosphere
from psrt.opticaldepth import integrate_optical_depth

def ray_transmittance(ray: Ray, s_start: float, s_end: float, atmosphere: Atmosphere, N: int) -> float:
    """Calculate transmittance exp(-tau) along a ray section."""
    tau = integrate_optical_depth(ray, s_start, s_end, atmosphere, N)
    return math.exp(-tau)

def beer_lambert_check(ray: Ray, sA: float, sB: float, sC: float, atmosphere: Atmosphere, N: int) -> float:
    """
    Verify Beer-Lambert multiplicativity for points along a ray: sA < sB < sC.
    T(sA -> sC) should equal T(sA -> sB) * T(sB -> sC).
    Returns the relative error.
    """
    # Force sorting
    sA, sB, sC = sorted([sA, sB, sC])
    
    t_AC = ray_transmittance(ray, sA, sC, atmosphere, N)
    t_AB = ray_transmittance(ray, sA, sB, atmosphere, N)
    t_BC = ray_transmittance(ray, sB, sC, atmosphere, N)
    
    product = t_AB * t_BC
    
    if product > 0.0:
        return abs(t_AC - product) / product
    return abs(t_AC - product)
