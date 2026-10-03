# PSRT: Pseudo-Spherical Radiative Transfer Reference Solver

PSRT is a research-grade reference solver for validating Chapman spherical attenuation, single scattering, multiple scattering, pseudo-spherical approximations, and two-stream closures.

Currently, **Phase 0** has been fully implemented and verified. This phase establishes a high-precision geometry and optical depth integration engine, validated against exact Chapman integrals.

## Phase 0 Architecture

- **`psrt/vector.py`**: 3D Cartesian vector mathematics (`Vec3`).
- **`psrt/ray.py`**: Ray parametrization and analytical intersection solvers for spheres and shells.
- **`psrt/geometry.py`**: Coordinate transforms, distance, and chord ray utilities.
- **`psrt/atmosphere.py`**: Exponential atmosphere representation with dynamic cutoff boundaries.
- **`psrt/quadrature.py`**: Gauss-Legendre quadrature mapping.
- **`psrt/opticaldepth.py`**: Numerical line-integration of extinction coefficients along rays.
- **`psrt/chapman.py`**: Exact Chapman numerical integral calculations and asymptotic erfcx approximations.
- **`psrt/validation.py`**: Automated gate-verification suite.

## Mathematical Definitions

### Zenith Angle Cosine
$$\mu = \vec{n}_{\text{surface}} \cdot \vec{\omega}$$
where $\vec{n}_{\text{surface}} = \vec{x}/|\vec{x}|$ is the outward radial unit normal at the start of the path, and $\vec{\omega}$ is the ray direction.

### Exact Chapman Integral
$$Ch_{\text{exact}}(X, \mu) = \int_0^\infty \exp\left(-\mu u - \frac{u^2}{2X}\right) du$$
where $X = R / H$ and $u = s / H$.

### erfcx Chapman Approximation
$$Ch_{\text{erfcx}}(X, \mu) = \sqrt{\frac{\pi X}{2}} \text{erfcx}\left(\sqrt{\frac{X}{2}} \mu\right)$$

---

## Getting Started

### 1. Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Tests
Run automated unit tests and gate checks:
```bash
python -m pytest
```

### 3. Run Validation Pipeline
Execute the full validation pipeline to run Gate 0, Gate 1A, and Gate 1B checks. This will output a numerical error report and save results to `results/chapman_error.csv`:
```bash
python -m psrt.validation
```
