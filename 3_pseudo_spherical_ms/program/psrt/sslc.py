import math
from psrt.vector import Vec3
from psrt.ray import Ray
from psrt.atmosphere import Atmosphere
from psrt.chapman import Ch_erfcx
from psrt.quadrature import gauss_legendre_rules
from psrt.phase_function import rayleigh_phase, cos_scattering_angle
from psrt.geometry import atmosphere_shell_intersection

def chapman_tau_to_space(position: Vec3, direction: Vec3, atmosphere: Atmosphere) -> float:
    """
    Compute the analytical Chapman optical depth to space:
    tau_inf = Ch_erfcx(X, mu) * beta(r) * H
    If direction points towards the planetary interior (mu < 0), returns a large number (1e10).
    """
    R = atmosphere.radius
    H = atmosphere.scale_height
    r = position.norm()
    
    if r < R - 1e-7:
        return 1e10
        
    n = position / r
    mu = n.dot(direction.normalize())
    
    if mu < 0.0:
        return 1e10
        
    X = r / H
    Ch = Ch_erfcx(X, mu)
    
    beta = atmosphere.beta(r)
    return Ch * beta * H

def sslc_L1_kernel(camera_pos: Vec3, view_dir: Vec3, sun_dir: Vec3, atmosphere: Atmosphere, N: int, omega0: float = 1.0) -> float:
    """
    Calculate the normalized single scattering L1 kernel using the Single Scattering Local Chapman (SSLC) approximation:
    L1_sslc = omega0 * integral_{s0}^{s1} beta(r(s)) * Ts_sslc * Tv_sslc * p(cos_theta) ds
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
    
    cos_theta = cos_scattering_angle(sun_dir, view_dir)
    phase_val = rayleigh_phase(cos_theta)
    
    integral = 0.0
    for s, w in zip(nodes, weights):
        x = ray.position(s)
        r = x.norm()
        
        beta = atmosphere.beta(r)
        
        # Solar transmittance using local Chapman to space
        tau_s = chapman_tau_to_space(x, sun_dir, atmosphere)
        Ts = math.exp(-tau_s)
        
        # View path transmittance using local Chapman difference to space
        # Since camera view_dir is pointing towards the sky, we project along view_dir to avoid negative mu.
        tau_v = abs(chapman_tau_to_space(camera_pos, view_dir, atmosphere) - chapman_tau_to_space(x, view_dir, atmosphere))
        Tv = math.exp(-tau_v)
        
        integral += w * beta * Ts * Tv * phase_val
        
    return omega0 * float(integral)
