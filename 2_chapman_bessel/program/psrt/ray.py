import math
from typing import List, Optional, Dict
from psrt.vector import Vec3

class Ray:
    """A Ray parameterized by origin and unit direction."""
    __slots__ = ('origin', 'direction')

    def __init__(self, origin: Vec3, direction: Vec3):
        self.origin = origin
        # Enforce direction to be unit vector
        self.direction = direction.normalize()

    def position(self, s: float) -> Vec3:
        """Calculate position at distance s along the ray."""
        return self.origin + self.direction * s

    def intersect_sphere(self, radius: float, tol: float = 1e-10) -> List[float]:
        """
        Calculate intersection distances s >= 0 with a sphere of given radius centered at (0,0,0).
        Returns a sorted list of distances.
        """
        x0 = self.origin
        omega = self.direction
        
        # Coefficients of quadratic equation: s^2 + 2*s*(x0.dot(omega)) + (x0.dot(x0) - R^2) = 0
        x0_dot_omega = x0.dot(omega)
        c = x0.dot(x0) - radius * radius
        
        discriminant_quarter = x0_dot_omega * x0_dot_omega - c
        
        if discriminant_quarter < -tol:
            return []
        
        if abs(discriminant_quarter) < tol:
            # Single tangent intersection
            s = -x0_dot_omega
            if s >= -tol:
                return [max(0.0, s)]
            return []
            
        sqrt_disc = math.sqrt(discriminant_quarter)
        s1 = -x0_dot_omega - sqrt_disc
        s2 = -x0_dot_omega + sqrt_disc
        
        intersections = []
        for s in (s1, s2):
            if s >= -tol:
                intersections.append(max(0.0, s))
                
        return sorted(list(set(intersections)))

    def intersect_shell(self, r_inner: float, r_outer: float, tol: float = 1e-10) -> Dict[str, List[float]]:
        """
        Calculate intersection distances s >= 0 with the shell boundaries.
        Returns a dictionary with 'inner' and 'outer' intersection distance lists.
        """
        if r_inner > r_outer:
            r_inner, r_outer = r_outer, r_inner
            
        inner_intersections = self.intersect_sphere(r_inner, tol)
        outer_intersections = self.intersect_sphere(r_outer, tol)
        
        return {
            'inner': inner_intersections,
            'outer': outer_intersections
        }
