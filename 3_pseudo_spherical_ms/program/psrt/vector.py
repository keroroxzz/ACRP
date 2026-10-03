import math

class Vec3:
    """A 3D Cartesian vector class."""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other: 'Vec3') -> 'Vec3':
        if not isinstance(other, Vec3):
            return NotImplemented
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        if not isinstance(other, Vec3):
            return NotImplemented
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vec3':
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'Vec3':
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'Vec3':
        if scalar == 0.0:
            raise ZeroDivisionError("Cannot divide Vec3 by zero.")
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: 'Vec3') -> float:
        """Calculate dot product of two vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        """Calculate Euclidean L2 norm of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> 'Vec3':
        """Return a normalized unit vector. Raises ValueError if norm is zero."""
        n = self.norm()
        if n == 0.0:
            raise ValueError("Cannot normalize a zero vector.")
        return self / n

    def __repr__(self) -> str:
        return f"Vec3({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y) and math.isclose(self.z, other.z)
