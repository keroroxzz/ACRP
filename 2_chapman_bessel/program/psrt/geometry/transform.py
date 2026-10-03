import math
from typing import Tuple
from psrt.vector import Vec3

def get_local_frame(position: Vec3, sun_direction: Vec3) -> Tuple[Vec3, Vec3, Vec3]:
    """
    Generate local orthonormal basis (x_local, y_local, z_local) at position.
    z_local: local surface normal (outward radial unit vector).
    x_local: projection of sun_direction onto local horizon, normalized.
    y_local: cross product of z_local and x_local.
    """
    r = position.norm()
    if r == 0.0:
        # Fallback at center
        z_local = Vec3(0.0, 0.0, 1.0)
    else:
        z_local = position / r
        
    # Project sun_direction onto horizon: x_proj = sun - (sun . z) * z
    sun_dot_z = sun_direction.dot(z_local)
    x_proj = sun_direction - sun_dot_z * z_local
    
    x_norm = x_proj.norm()
    if x_norm > 1e-7:
        x_local = x_proj / x_norm
    else:
        # Sun is at zenith/nadir. Pick an arbitrary orthogonal vector.
        # Pick another vector to cross
        temp = Vec3(1.0, 0.0, 0.0)
        if abs(temp.dot(z_local)) > 0.99:
            temp = Vec3(0.0, 1.0, 0.0)
        x_proj = temp - temp.dot(z_local) * z_local
        x_local = x_proj.normalize()
        
    # y_local = z_local x x_local
    # Cross product: (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
    cross_x = z_local.y * x_local.z - z_local.z * x_local.y
    cross_y = z_local.z * x_local.x - z_local.x * x_local.z
    cross_z = z_local.x * x_local.y - z_local.y * x_local.x
    y_local = Vec3(cross_x, cross_y, cross_z).normalize()
    
    return x_local, y_local, z_local

def global_to_local_direction(position: Vec3, omega_global: Vec3, sun_direction: Vec3) -> Tuple[float, float]:
    """
    Convert global direction vector to local (mu, phi) angles.
    mu: cosine of local zenith angle.
    phi: local azimuth angle (in radians, from [0, 2*pi)).
    """
    x_local, y_local, z_local = get_local_frame(position, sun_direction)
    
    mu = omega_global.dot(z_local)
    # Clip mu to prevent precision issues
    mu = max(-1.0, min(1.0, mu))
    
    dx = omega_global.dot(x_local)
    dy = omega_global.dot(y_local)
    
    phi = math.atan2(dy, dx)
    if phi < 0.0:
        phi += 2.0 * math.pi
        
    return mu, phi

def local_to_global_direction(position: Vec3, mu: float, phi: float, sun_direction: Vec3) -> Vec3:
    """Convert local (mu, phi) angles back to global direction vector."""
    x_local, y_local, z_local = get_local_frame(position, sun_direction)
    
    sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))
    
    omega_global = (sin_theta * math.cos(phi)) * x_local + (sin_theta * math.sin(phi)) * y_local + mu * z_local
    return omega_global.normalize()
