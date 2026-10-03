"""
Vectorized spherical formal-transport solver.

This is a numpy-vectorized, bit-for-bit faithful reimplementation of
``psrt.transport.spherical.spherical_transport_solver``. It reproduces the
exact same algorithm (same Gauss-Legendre nodes on [0, s1], same trapezoidal
cumulative optical depth seeded from the origin, same local-frame construction,
same trilinear interpolation with the same edge clamping) so that it can be
validated pointwise against the original ("golden") solver before adoption.

The only difference is that the per-(radius, direction, node) inner work is
expressed as numpy array operations instead of Python loops over Vec3 objects.
"""
import numpy as np
from psrt.vector import Vec3
from psrt.angular import DOMGrid
from psrt.atmosphere import Atmosphere

# tolerance used by atmosphere_shell_intersection (passed down to intersect_sphere)
_SHELL_TOL = 1e-7


def spherical_transport_solver_vec(
    r_grid: np.ndarray,
    J: np.ndarray,
    grid: DOMGrid,
    atmosphere: Atmosphere,
    L_top: np.ndarray,   # unused (black boundary), kept for signature parity
    L_bottom: np.ndarray,  # unused (black boundary), kept for signature parity
    sun_dir: Vec3,
    N_sub: int = 64,
) -> np.ndarray:
    M = len(r_grid)
    dirs = grid.directions
    D = len(dirs)
    N_mu = grid.N_mu
    N_phi = grid.N_phi

    R = atmosphere.radius
    r_top = atmosphere.r_top
    H = atmosphere.scale_height
    ext = atmosphere.extinction_coeff
    rho0 = atmosphere.rho0

    r_grid = np.asarray(r_grid, dtype=float)
    d_vec = np.array([[d.x, d.y, d.z] for d in dirs], dtype=float)  # [D,3]
    dz = d_vec[:, 2]
    sx, sy, sz = float(sun_dir.x), float(sun_dir.y), float(sun_dir.z)

    # ---- 1. Exit distance s1[k,d] via ray-sphere (observer always inside atm) ----
    rk = r_grid                                   # [M]
    b = rk[:, None] * dz[None, :]                 # [M,D]  (x0.omega, x0=(0,0,rk))
    c_R = rk * rk - R * R                          # [M]
    c_top = rk * rk - r_top * r_top                # [M]  (<=0)

    disc_R = b * b - c_R[:, None]                  # [M,D]
    disc_top = b * b - c_top[:, None]              # [M,D]  (>0)

    sq_R = np.sqrt(np.clip(disc_R, 0.0, None))
    sq_top = np.sqrt(np.clip(disc_top, 0.0, None))

    # planet-surface hit: smallest positive root > tol, only when real
    r_small = -b - sq_R
    r_large = -b + sq_R
    has_real_R = disc_R >= -_SHELL_TOL
    surf_exit = np.where(
        has_real_R & (r_small > _SHELL_TOL), r_small,
        np.where(has_real_R & (r_large > _SHELL_TOL), r_large, np.inf),
    )
    hit_planet = np.isfinite(surf_exit)

    top_exit = -b + sq_top                         # positive root through top shell
    s1 = np.where(hit_planet, surf_exit, top_exit)  # [M,D]
    s1 = np.where(s1 > 0.0, s1, 0.0)

    # ---- 2. Gauss-Legendre nodes on [0, s1] ----
    t_std, w_std = np.polynomial.legendre.leggauss(N_sub)   # on [-1,1]
    # mapped node = 0.5*s1*(t+1), weight = 0.5*s1*w
    half = 0.5 * s1                                 # [M,D]
    s = half[:, :, None] * (t_std[None, None, :] + 1.0)   # [M,D,N]
    w = half[:, :, None] * w_std[None, None, :]           # [M,D,N]

    # ---- 3. Node positions and local extinction ----
    dx = d_vec[:, 0][None, :, None]                 # [1,D,1]
    dy = d_vec[:, 1][None, :, None]
    dzc = d_vec[:, 2][None, :, None]
    x_x = s * dx
    x_y = s * dy
    x_z = rk[:, None, None] + s * dzc
    r_s = np.sqrt(x_x * x_x + x_y * x_y + x_z * x_z)  # [M,D,N]

    in_atm = (r_s >= R) & (r_s <= r_top)
    # clip the exponent's lower bound before exp: nodes with r_s < R are masked
    # out below anyway, but evaluating exp on their (large positive) argument
    # would overflow. Clipping keeps the masked branch finite; results unchanged.
    h_over_H = np.clip((r_s - R) / H, -50.0, None)
    beta = np.where(in_atm, ext * rho0 * np.exp(-h_over_H), 0.0)
    beta_start = ext * rho0 * np.exp(-(rk - R) / H)   # [M] (rk in atmosphere)

    # ---- 4. Cumulative trapezoidal optical depth seeded from s=0 ----
    inc = np.empty_like(beta)
    inc[:, :, 0] = 0.5 * (beta_start[:, None] + beta[:, :, 0]) * s[:, :, 0]
    ds = s[:, :, 1:] - s[:, :, :-1]
    inc[:, :, 1:] = 0.5 * (beta[:, :, :-1] + beta[:, :, 1:]) * ds
    tau_v = np.cumsum(inc, axis=-1)
    Tv = np.exp(-tau_v)

    # ---- 5. Local (mu, phi) of the (constant) ray direction at each node ----
    zx = x_x / r_s
    zy = x_y / r_s
    zz = x_z / r_s
    sun_dot_z = sx * zx + sy * zy + sz * zz
    xpx = sx - sun_dot_z * zx
    xpy = sy - sun_dot_z * zy
    xpz = sz - sun_dot_z * zz
    xn = np.sqrt(xpx * xpx + xpy * xpy + xpz * xpz)
    deg = xn <= 1e-7

    # fallback basis when sun is (anti)parallel to local vertical
    use_y = np.abs(zx) > 0.99
    # temp = (1,0,0): proj = (1 - zx^2, -zx*zy, -zx*zz)
    fbx_x = 1.0 - zx * zx
    fbx_y = -zx * zy
    fbx_z = -zx * zz
    # temp = (0,1,0): proj = (-zy*zx, 1 - zy^2, -zy*zz)
    fby_x = -zy * zx
    fby_y = 1.0 - zy * zy
    fby_z = -zy * zz
    fb_x = np.where(use_y, fby_x, fbx_x)
    fb_y = np.where(use_y, fby_y, fbx_y)
    fb_z = np.where(use_y, fby_z, fbx_z)
    fb_n = np.sqrt(fb_x * fb_x + fb_y * fb_y + fb_z * fb_z)
    fb_n = np.where(fb_n > 0.0, fb_n, 1.0)

    xn_safe = np.where(deg, 1.0, xn)
    xlx = np.where(deg, fb_x / fb_n, xpx / xn_safe)
    xly = np.where(deg, fb_y / fb_n, xpy / xn_safe)
    xlz = np.where(deg, fb_z / fb_n, xpz / xn_safe)

    # y_local = z_local x x_local
    ylx = zy * xlz - zz * xly
    yly = zz * xlx - zx * xlz
    ylz = zx * xly - zy * xlx

    mu = dx * zx + dy * zy + dzc * zz
    mu = np.clip(mu, -1.0, 1.0)
    dloc_x = dx * xlx + dy * xly + dzc * xlz
    dloc_y = dx * ylx + dy * yly + dzc * ylz
    phi = np.arctan2(dloc_y, dloc_x)
    phi = np.where(phi < 0.0, phi + 2.0 * np.pi, phi)

    # ---- 6. Trilinear interpolation of J on (r, mu, phi) ----
    J_res = J.reshape((M, N_mu, N_phi))
    mu_grid = grid.mu           # ascending [N_mu]
    w_phi = grid.w_phi

    # radial index/weight with edge clamping identical to the scalar version
    k = np.searchsorted(r_grid, r_s, side="left")
    kk = np.clip(k, 1, M - 1)
    k0 = kk - 1
    k1 = kk
    denom_r = (r_grid[k1] - r_grid[k0])
    tr = (r_s - r_grid[k0]) / denom_r
    below_r = r_s <= r_grid[0]
    above_r = r_s >= r_grid[-1]
    k0 = np.where(below_r, 0, np.where(above_r, M - 1, k0))
    k1 = np.where(below_r, 0, np.where(above_r, M - 1, k1))
    tr = np.where(below_r | above_r, 0.0, tr)

    # zenith index/weight
    i = np.searchsorted(mu_grid, mu, side="left")
    ii = np.clip(i, 1, N_mu - 1)
    i0 = ii - 1
    i1 = ii
    denom_m = (mu_grid[i1] - mu_grid[i0])
    tm = (mu - mu_grid[i0]) / denom_m
    below_m = mu <= mu_grid[0]
    above_m = mu >= mu_grid[-1]
    i0 = np.where(below_m, 0, np.where(above_m, N_mu - 1, i0))
    i1 = np.where(below_m, 0, np.where(above_m, N_mu - 1, i1))
    tm = np.where(below_m | above_m, 0.0, tm)

    # azimuth index/weight (periodic)
    idx_phi = (phi % (2.0 * np.pi)) / w_phi
    fl = np.floor(idx_phi)
    j0 = (fl.astype(np.int64)) % N_phi
    j1 = (j0 + 1) % N_phi
    tp = idx_phi - fl

    def gather(ka, ia, ja):
        return J_res[ka, ia, ja]

    c000 = gather(k0, i0, j0)
    c001 = gather(k0, i0, j1)
    c010 = gather(k0, i1, j0)
    c011 = gather(k0, i1, j1)
    c100 = gather(k1, i0, j0)
    c101 = gather(k1, i0, j1)
    c110 = gather(k1, i1, j0)
    c111 = gather(k1, i1, j1)

    omtr, omtm, omtp = 1.0 - tr, 1.0 - tm, 1.0 - tp
    J_val = (
        omtr * omtm * omtp * c000 + omtr * omtm * tp * c001
        + omtr * tm * omtp * c010 + omtr * tm * tp * c011
        + tr * omtm * omtp * c100 + tr * omtm * tp * c101
        + tr * tm * omtp * c110 + tr * tm * tp * c111
    )

    # ---- 7. Assemble the formal-solution integral ----
    integrand = w * beta * Tv * J_val    # nodes outside atmosphere have beta=0
    L = np.sum(integrand, axis=-1)       # [M,D]
    return L
