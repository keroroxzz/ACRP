import math
import numpy as np

def isotropic_phase(cos_theta: float) -> float:
    """Isotropic phase function: p(cos_theta) = 1 / (4 * pi)"""
    return 1.0 / (4.0 * np.pi)

def rayleigh_phase(cos_theta: float) -> float:
    """Rayleigh phase function: p(cos_theta) = 3 / (16 * pi) * (1 + cos_theta^2)"""
    return (3.0 / (16.0 * np.pi)) * (1.0 + cos_theta * cos_theta)

def henyey_greenstein_phase(cos_theta: float, g: float) -> float:
    """
    Henyey-Greenstein phase function:
    p(cos_theta) = 1 / (4 * pi) * (1 - g^2) / (1 + g^2 - 2 * g * cos_theta)^1.5
    """
    if abs(g) < 1e-6:
        return isotropic_phase(cos_theta)
    denom = 1.0 + g * g - 2.0 * g * cos_theta
    return (1.0 / (4.0 * np.pi)) * (1.0 - g * g) / (denom ** 1.5)
