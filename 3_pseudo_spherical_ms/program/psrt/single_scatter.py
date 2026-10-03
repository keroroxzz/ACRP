import math
import numpy as np
from psrt.vector import Vec3
from psrt.ray import Ray
from psrt.atmosphere import Atmosphere
from psrt.quadrature import gauss_legendre_rules
from psrt.opticaldepth import integrate_optical_depth
from psrt.phase_function import rayleigh_phase, cos_scattering_angle
from psrt.beam import solar_transmittance
from psrt.geometry import atmosphere_shell_intersection

def spherical_L1_kernel(camera_pos: Vec3, view_dir: Vec3, sun_dir: Vec3, atmosphere: Atmosphere, N: int, omega0: float = 1.0) -> float:
    """
    Compute the normalized spherical single scattering kernel L1_tilde:
    L1_tilde = omega0 * integral_{s0}^{s1} beta(r(s)) * Ts(x(s)) * Tv(s) * p(cos_theta) ds
    """
    ray = Ray(camera_pos, view_dir)
    intersect_res = atmosphere_shell_intersection(ray, atmosphere)
    
    if not intersect_res['valid']:
        return 0.0
        
    s0 = intersect_res['entry_distance']
    s1 = intersect_res['exit_distance']
    
    if s1 <= s0:
        return 0.0
        
    nodes, weights = gauss_legendre_rules(s0, s1, N)
    
    # Calculate scattering angle cosine (constant along the view direction for parallel sun rays)
    cos_theta = cos_scattering_angle(sun_dir, view_dir)
    phase_val = rayleigh_phase(cos_theta)
    
    integral = 0.0
    for s, w in zip(nodes, weights):
        x = ray.position(s)
        r = x.norm()
        
        beta = atmosphere.beta(r)
        Ts = solar_transmittance(x, sun_dir, atmosphere, N=128) # use 128 points for inner solar path for speed and accuracy
        
        # Tv(s) is the transmittance along the sightline from s0 (atmosphere entry) to s
        tau_v = integrate_optical_depth(ray, s0, s, atmosphere, N=128)
        Tv = math.exp(-tau_v)
        
        integral += w * beta * Ts * Tv * phase_val
        
    return omega0 * float(integral)

def spherical_L1_physical(camera_pos: Vec3, view_dir: Vec3, sun_dir: Vec3, atmosphere: Atmosphere, N: int, solar_flux: float = 1.0, omega0: float = 1.0) -> float:
    """
    Compute the physical single scattering radiance L1:
        L1 = solar_flux * L1_tilde

    The phase function p is normalized so that the DOM sum p*w = 1, i.e.
    the integral of p over solid angle is 1 (p already carries the 1/4pi).
    Hence the beam first-scatter source is omega0 * F0 * Ts * p and the
    transported radiance is F0 * kernel -- there is NO extra 1/(4*pi) factor.
    (This matches the SOS radiance L_sos and closes the spherical energy budget;
    the previous 1/(4*pi) here was a bug that under-reported L1 by 4*pi.)
    """
    kernel = spherical_L1_kernel(camera_pos, view_dir, sun_dir, atmosphere, N, omega0)
    return solar_flux * kernel
