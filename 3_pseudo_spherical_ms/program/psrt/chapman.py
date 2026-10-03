import math
import numpy as np
import scipy.integrate
import scipy.special

def Ch_spherical_exact(X: float, mu: float) -> float:
    """
    Compute the exact spherical path integral (no Taylor expansion):
    Ch_spherical_exact(X, mu) = integral_0_inf exp(-X * (sqrt(1 + 2*u*mu/X + u^2/X^2) - 1)) du
    using scipy.integrate.quad.
    """
    integrand = lambda u: math.exp(-X * (math.sqrt(max(0.0, 1.0 + 2.0 * u * mu / X + u * u / (X * X))) - 1.0))
    val, _ = scipy.integrate.quad(integrand, 0.0, np.inf)
    return val

def Ch_exact(X: float, mu: float) -> float:
    """
    Compute the exact Chapman integral (with asymptotic second-order Taylor expansion):
    Ch_exact(X, mu) = integral_0_inf exp(-mu * u - u^2 / (2 * X)) du
    using scipy.integrate.quad.
    """
    integrand = lambda u: math.exp(-mu * u - u * u / (2.0 * X))
    val, _ = scipy.integrate.quad(integrand, 0.0, np.inf)
    return val

def Ch_erfcx(X: float, mu: float) -> float:
    """
    Compute the asymptotic erfcx Chapman approximation:
    Ch_erfcx(X, mu) = sqrt(pi * X / 2) * erfcx(sqrt(X / 2) * mu)
    """
    arg = math.sqrt(0.5 * X) * mu
    return math.sqrt(0.5 * math.pi * X) * scipy.special.erfcx(arg)
