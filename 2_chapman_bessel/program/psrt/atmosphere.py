import math

class Atmosphere:
    """
    An exponential atmosphere model.
    Parameters:
        radius (R): Planetary radius.
        scale_height (H): Atmospheric scale height.
        rho0 (rho0): Reference surface density.
        extinction_coeff (sigma_ext): Extinction cross section.
    """
    def __init__(self, radius: float, scale_height: float, rho0: float, extinction_coeff: float):
        self.radius = float(radius)
        self.scale_height = float(scale_height)
        self.rho0 = float(rho0)
        self.extinction_coeff = float(extinction_coeff)

    @property
    def r_top(self) -> float:
        """Top boundary radius of the atmosphere, defined at R + 100*H to avoid truncation errors."""
        return self.radius + 100.0 * self.scale_height

    def density(self, r: float) -> float:
        """Compute atmospheric density at distance r from the planetary center."""
        # Standard exponential atmosphere: rho(r) = rho0 * exp(-(r-R)/H)
        # To avoid negative overflow at extremely large r, clip or handle properly.
        # Since r_top is R + 100H, exp(-100) is fine.
        h = r - self.radius
        # We can clip h to avoid negative values (under planetary surface) if needed, but in Ray integration
        # we assume we start at the surface or above.
        return self.rho0 * math.exp(-h / self.scale_height)

    def beta(self, r: float) -> float:
        """Compute local extinction coefficient beta(r) = sigma_ext * density(r)."""
        return self.extinction_coeff * self.density(r)

    def optical_depth_vertical(self, r: float) -> float:
        """
        Compute analytical vertical optical depth from radius r to infinity.
        For an exponential atmosphere, this is exactly beta(r) * H.
        """
        return self.beta(r) * self.scale_height
