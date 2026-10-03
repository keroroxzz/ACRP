from typing import Dict, Any
from psrt.vector import Vec3
from psrt.ray import Ray
from psrt.atmosphere import Atmosphere

def chord_distance(x1: Vec3, x2: Vec3) -> float:
    """Calculate Euclidean distance between two points."""
    return (x2 - x1).norm()

def chord_ray(x1: Vec3, x2: Vec3) -> Ray:
    """Generate a Ray starting at x1 and pointing towards x2."""
    direction = x2 - x1
    return Ray(x1, direction)

def atmosphere_shell_intersection(ray: Ray, atmosphere: Atmosphere, tol: float = 1e-7) -> Dict[str, Any]:
    """
    Calculate entry and exit distances s0 and s1 where the camera ray interacts with the atmosphere.
    Returns:
      {
         'valid': bool,
         'entry_distance': float,
         'exit_distance': float,
         'terminated_by_planet': bool
      }
    """
    R = atmosphere.radius
    r_top = atmosphere.r_top
    r0 = ray.origin.norm()
    
    # Setup default return
    invalid_result = {
        'valid': False,
        'entry_distance': 0.0,
        'exit_distance': 0.0,
        'terminated_by_planet': False
    }
    
    # 1. Underground observer
    if r0 < R - tol:
        return invalid_result
        
    ints_top = ray.intersect_sphere(r_top, tol)
    ints_surf = ray.intersect_sphere(R, tol)
    
    # 2. Observer inside or on the surface of the atmosphere
    if r0 <= r_top + tol:
        # entry_distance is 0 (current position)
        entry_dist = 0.0
        
        # Check if ray hits the planet core
        # Filter surface intersections greater than zero
        valid_surf_hits = [s for s in ints_surf if s > tol]
        if len(valid_surf_hits) > 0:
            # Ray is pointing downwards and hits the surface
            exit_dist = min(valid_surf_hits)
            return {
                'valid': True,
                'entry_distance': entry_dist,
                'exit_distance': exit_dist,
                'terminated_by_planet': True
            }
        else:
            # Ray points towards the sky and exits through the top boundary
            valid_top_hits = [s for s in ints_top if s > tol]
            exit_dist = valid_top_hits[0] if len(valid_top_hits) > 0 else 0.0
            return {
                'valid': True,
                'entry_distance': entry_dist,
                'exit_distance': exit_dist,
                'terminated_by_planet': False
            }
            
    # 3. Observer outside the atmosphere (space camera)
    else:
        # To intersect the atmosphere, the ray must intersect the outer boundary r_top twice
        valid_top_hits = [s for s in ints_top if s > tol]
        if len(valid_top_hits) < 2:
            return invalid_result
            
        s_in = valid_top_hits[0]
        s_out = valid_top_hits[1]
        
        # Check if the path inside the atmosphere hits the planetary surface
        # A valid planet surface hit must occur AFTER entering the atmosphere (s > s_in)
        valid_surf_hits = [s for s in ints_surf if s > s_in + tol]
        if len(valid_surf_hits) > 0:
            exit_dist = min(valid_surf_hits)
            return {
                'valid': True,
                'entry_distance': s_in,
                'exit_distance': exit_dist,
                'terminated_by_planet': True
            }
        else:
            return {
                'valid': True,
                'entry_distance': s_in,
                'exit_distance': s_out,
                'terminated_by_planet': False
            }
