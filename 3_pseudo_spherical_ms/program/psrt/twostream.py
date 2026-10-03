import math
import numpy as np

class TwoStreamResult:
    """Structure to hold Two-Stream solver results."""
    def __init__(self, J: np.ndarray, F_up: np.ndarray, F_down: np.ndarray, taus: np.ndarray):
        self.J = J            # Mean intensity [Nt]
        self.F_up = F_up      # Upward flux [Nt]
        self.F_down = F_down  # Downward flux [Nt]
        self.taus = taus      # Optical depth grid [Nt]

def solve_twostream_eddington(
    tau_grid: np.ndarray,  # 1D array of optical depths from 0 to tau_g [Nt]
    mu_s: float,           # sun zenith cosine
    solar_flux: float,     # F0
    omega0: float,         # albedo
    g: float = 0.0,        # asymmetry parameter (default Rayleigh/isotropic = 0.0)
    As: float = 0.0        # surface albedo (default black = 0.0)
) -> TwoStreamResult:
    """
    Solve Eddington's BVP using the Thomas tridiagonal algorithm.
    d^2J / dtau^2 = lambda^2 * J - S0 * exp(-tau/mu_s)
    
    Where:
        lambda^2 = 3 * (1 - omega0) * (1 - omega0 * g)
        S0 = 0.75 * omega0 * F0 * (1 + g * (1 - omega0) * mu_s) (Note: for g=0, S0 = 0.75 * omega0 * F0)
    """
    Nt = len(tau_grid)
    J = np.zeros(Nt)
    
    # Grid spacings
    dtau = tau_grid[1] - tau_grid[0]
    
    # Eddington coefficients
    lam2 = 3.0 * (1.0 - omega0) * (1.0 - omega0 * g)
    S0 = 0.75 * omega0 * solar_flux * (1.0 + g * (1.0 - omega0) * mu_s)
    
    # Set up tridiagonal system: a[i] * J[i-1] + b[i] * J[i] + c[i] * J[i+1] = r[i]
    a = np.zeros(Nt)
    b = np.zeros(Nt)
    c = np.zeros(Nt)
    r = np.zeros(Nt)
    
    # 1. Top Boundary condition: J(0) - 2/3 * dJ/dtau = 0
    # Using first order forward difference: J_0 - 2/3 * (J_1 - J_0)/dtau = 0
    # => (3*dtau + 2) * J_0 - 2 * J_1 = 0
    b[0] = 3.0 * dtau + 2.0
    c[0] = -2.0
    r[0] = 0.0
    
    # 2. Bottom Boundary condition:
    # F+(tau_g) = As * (F-(tau_g) + mu_s * F0 * exp(-tau_g/mu_s))
    # Where:
    # F+ = pi * (J + 2/3 * dJ/dtau)
    # F- = pi * (J - 2/3 * dJ/dtau)
    # Let dJ/dtau at end be back-difference: (J_n - J_{n-1})/dtau
    # => pi*(J_n + 2/3*(J_n - J_{n-1})/dtau) = As * pi*(J_n - 2/3*(J_n - J_{n-1})/dtau) + As * mu_s * F0 * exp(-tau_g/mu_s)
    # Collect coefficients:
    # J_{n-1} * [-2/3/dtau * (1 - As)] + J_n * [1 + 2/3/dtau * (1 - As) - As] = As * mu_s * F0 * exp(-tau_g/mu_s) / pi
    tau_g = tau_grid[-1]
    direct_term = As * mu_s * solar_flux * math.exp(-tau_g / mu_s) / np.pi
    
    factor = (2.0 / 3.0) / dtau * (1.0 - As)
    a[-1] = -factor
    b[-1] = 1.0 + factor - As
    r[-1] = direct_term
    
    # 3. Internal points
    # (J_{i-1} - 2*J_i + J_{i+1}) / dtau^2 - lam2 * J_i = -S0 * exp(-tau_i/mu_s)
    # => J_{i-1} - (2 + lam2 * dtau^2) * J_i + J_{i+1} = -S0 * dtau^2 * exp(-tau_i/mu_s)
    for i in range(1, Nt - 1):
        a[i] = 1.0
        b[i] = -(2.0 + lam2 * dtau * dtau)
        c[i] = 1.0
        r[i] = -S0 * dtau * dtau * math.exp(-tau_grid[i] / mu_s)
        
    # 4. Solve tridiagonal system using Thomas Algorithm
    c_prime = np.zeros(Nt)
    r_prime = np.zeros(Nt)
    
    c_prime[0] = c[0] / b[0]
    r_prime[0] = r[0] / b[0]
    
    for i in range(1, Nt):
        denom = b[i] - a[i] * c_prime[i-1]
        if i < Nt - 1:
            c_prime[i] = c[i] / denom
        r_prime[i] = (r[i] - a[i] * r_prime[i-1]) / denom
        
    J[-1] = r_prime[-1]
    for i in range(Nt - 2, -1, -1):
        J[i] = r_prime[i] - c_prime[i] * J[i+1]
        
    # 5. Compute fluxes
    F_up = np.zeros(Nt)
    F_down = np.zeros(Nt)
    
    # Estimate dJ/dtau using finite differences
    dJ = np.zeros(Nt)
    dJ[0] = (J[1] - J[0]) / dtau
    dJ[-1] = (J[-1] - J[-2]) / dtau
    for i in range(1, Nt - 1):
        dJ[i] = 0.5 * (J[i+1] - J[i-1]) / dtau
        
    for i in range(Nt):
        F_up[i] = J[i] + (2.0 / 3.0) * dJ[i]
        F_down[i] = J[i] - (2.0 / 3.0) * dJ[i]

    return TwoStreamResult(J, F_up, F_down, tau_grid)


def solve_twostream_beam(
    tau_grid: np.ndarray,   # uniform optical-depth grid, ascending, index 0 = top
    beam: np.ndarray,       # direct-beam transmittance profile at each tau layer
    omega0: float,          # single-scattering albedo
    g: float = 0.0,         # asymmetry parameter
    lam2: float = None,     # eigenvalue^2; default 3*(1-w0)(1-w0 g). Pass 3.0 for the
                            # single-scatter variant (pure extinction, no re-scatter gain).
    solar_flux: float = 1.0,
) -> TwoStreamResult:
    """
    Eddington two-stream BVP driven by an ARBITRARY beam-source transmittance
    profile ``beam`` (rather than the plane sec law exp(-tau/mu_s)). This is the
    pseudo-spherical driver: pass the spherical Chapman solar transmittance as
    ``beam`` so grazing / twilight geometry is handled correctly.

    Solves  d^2 J/dtau^2 = lam2 * J - S0 * beam(tau),  S0 = 0.75*omega0*F0,
    with a black surface (As=0) and the diffusion top boundary J - 2/3 J' = 0.
    Requires a UNIFORM tau grid. With lam2=3 (pure extinction) it returns the
    single-scattered field used by the single-scattering (TMS) correction.
    """
    Nt = len(tau_grid)
    dtau = tau_grid[1] - tau_grid[0]
    if lam2 is None:
        lam2 = 3.0 * (1.0 - omega0) * (1.0 - omega0 * g)
    S0 = 0.75 * omega0 * solar_flux

    a = np.zeros(Nt); b = np.zeros(Nt); c = np.zeros(Nt); r = np.zeros(Nt)
    # top diffusion BC: (3*dtau + 2) J_0 - 2 J_1 = 0
    b[0] = 3.0 * dtau + 2.0; c[0] = -2.0
    # bottom black-surface BC (As=0)
    factor = (2.0 / 3.0) / dtau
    a[-1] = -factor; b[-1] = 1.0 + factor
    for i in range(1, Nt - 1):
        a[i] = 1.0
        b[i] = -(2.0 + lam2 * dtau * dtau)
        c[i] = 1.0
        r[i] = -S0 * dtau * dtau * beam[i]

    # Thomas algorithm
    cp = np.zeros(Nt); rp = np.zeros(Nt)
    cp[0] = c[0] / b[0]; rp[0] = r[0] / b[0]
    for i in range(1, Nt):
        denom = b[i] - a[i] * cp[i - 1]
        if i < Nt - 1:
            cp[i] = c[i] / denom
        rp[i] = (r[i] - a[i] * rp[i - 1]) / denom
    J = np.zeros(Nt)
    J[-1] = rp[-1]
    for i in range(Nt - 2, -1, -1):
        J[i] = rp[i] - cp[i] * J[i + 1]

    dJ = np.gradient(J, tau_grid)
    F_up = J + (2.0 / 3.0) * dJ
    F_down = J - (2.0 / 3.0) * dJ
    return TwoStreamResult(J, F_up, F_down, tau_grid)
