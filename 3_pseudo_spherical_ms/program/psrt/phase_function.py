import numpy as np
from psrt.vector import Vec3

def rayleigh_phase(cos_theta: float) -> float:
    """
    Compute Rayleigh phase function value for cos(theta).
    p(cos_theta) = 3 / (16 * pi) * (1 + cos_theta^2)
    This is normalized such that the integral over 4*pi solid angle is 1.0.
    """
    return (3.0 / (16.0 * np.pi)) * (1.0 + cos_theta * cos_theta)

def cos_scattering_angle(sun_dir: Vec3, view_dir: Vec3) -> float:
    """
    Calculate the cosine of the scattering angle.
    cos_theta = (-sun_dir) . view_dir
    where:
      sun_dir: unit vector pointing from the scattering point towards the sun.
      view_dir: unit vector pointing from the scattering point towards the observer.
    """
    return -sun_dir.dot(view_dir)
