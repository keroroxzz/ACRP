from psrt.vector import Vec3
from psrt.ray import Ray
from psrt.atmosphere import Atmosphere

def compute_twilight_indicator(position: Vec3, sun_dir: Vec3, atmosphere: Atmosphere) -> float:
    """
    Calculate the twilight indicator:
    Lambda_tw = l_sun / H
    where l_sun is the distance along the solar vector from the scattering point
    to the planetary core intersection. If there is no planetary intersection, l_sun = 0.
    """
    R = atmosphere.radius
    H = atmosphere.scale_height
    
    # If point is inside the planet, we are fully in shadow
    if position.norm() < R - 1e-7:
        return 1e10 / H
        
    ray = Ray(position, sun_dir)
    hits = ray.intersect_sphere(R)
    
    valid_hits = [s for s in hits if s > 1e-7]
    if len(valid_hits) > 0:
        l_sun = min(valid_hits)
    else:
        l_sun = 0.0
        
    return l_sun / H

def twilight_hybrid_switch(position: Vec3, sun_dir: Vec3, atmosphere: Atmosphere, threshold: float = 20.0) -> str:
    """
    Determine whether to use 'SSLC' or 'SOS' based on the twilight indicator.
    Returns 'SSLC' if Lambda_tw <= threshold, else 'SOS'.
    """
    lambda_tw = compute_twilight_indicator(position, sun_dir, atmosphere)
    if lambda_tw <= threshold:
        return "SSLC"
    return "SOS"
