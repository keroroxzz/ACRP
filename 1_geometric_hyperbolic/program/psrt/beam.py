import math
from psrt.vector import Vec3
from psrt.ray import Ray
from psrt.atmosphere import Atmosphere
from psrt.opticaldepth import integrate_optical_depth

def solar_transmittance(position: Vec3, sun_direction: Vec3, atmosphere: Atmosphere, N: int = 256) -> float:
    """
    Calculate the solar transmittance Ts = exp(-tau_s) from a scattering position
    along the sun_direction (pointing towards the sun) to space.
    If the path is blocked by the planetary core, returns 0.0.
    """
    R = atmosphere.radius
    r_top = atmosphere.r_top
    
    # 1. Planetary core occlusion check
    # If the point is physically underground, transmittance is zero.
    if position.norm() < R - 1e-7:
        return 0.0
        
    ray = Ray(position, sun_direction)
    
    # Find intersections with the planet core
    planet_hits = ray.intersect_sphere(R)
    # If there is a valid forward intersection distance greater than epsilon, the sun is occluded.
    for s in planet_hits:
        if s > 1e-7:
            return 0.0
            
    # 2. Integrate optical depth to the top of the atmosphere
    top_hits = ray.intersect_sphere(r_top)
    if len(top_hits) == 0:
        # If the ray never hits the atmosphere outer boundary, we must be outside the atmosphere
        # and pointing away from it. Optical depth is zero.
        return 1.0
        
    s_top = top_hits[0]
    
    # Integrate optical depth from s=0 to s_top
    tau = integrate_optical_depth(ray, 0.0, s_top, atmosphere, N)
    
    return math.exp(-tau)
