"""NumPy port of HSELowCloudTop() from the CloudWorks shaders.

Used only to pick reproducible test scenes: the field decides where the low
cloud deck is flat ("cloud sea") and where several cumulonimbus towers happen
to be mature at a given cloud time.  It is not part of the renderer, and it
does not need to be bit-exact with the GPU - only close enough to locate
features.

Run directly to scan for scene locations:  python hse_field.py
"""
from __future__ import annotations
import numpy as np

f32 = np.float32

CELL_SIZE = 8000.0
MAX_RISE = 7000.0
MIN_RADIUS = 5000.0
MAX_RADIUS = 10000.0
TIME_SCALE = 0.00008
EXISTENCE = 0.38          # cells below this have no event at all


def _frac(x):
    return np.float32(x - np.floor(x))


def _hash12(cx, cy):
    p3 = _frac(np.stack([cx, cy, cx], -1) * f32(0.1031))
    yzx = np.stack([p3[..., 1], p3[..., 2], p3[..., 0]], -1)
    p3 = p3 + (p3 * (yzx + f32(33.33))).sum(-1, keepdims=True)
    return _frac((p3[..., 0] + p3[..., 1]) * p3[..., 2])


def _hash22(cx, cy):
    p3 = _frac(np.stack([cx, cy, cx], -1) * np.array([0.1031, 0.1030, 0.0973], f32))
    yzx = np.stack([p3[..., 1], p3[..., 2], p3[..., 0]], -1)
    p3 = p3 + (p3 * (yzx + f32(33.33))).sum(-1, keepdims=True)
    return np.stack([_frac((p3[..., 0] + p3[..., 1]) * p3[..., 2]),
                     _frac((p3[..., 0] + p3[..., 2]) * p3[..., 1])], -1)


def _smoother(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (x * 6 - 15) + 10)


def _life(u):
    return _smoother(u / 0.20) * (1.0 - _smoother((u - 0.70) / 0.30))


def _radius_life(u):
    return 0.52 + (1.0 - 0.52) * _smoother((u - 0.08) / 0.55)


def event_rise(px, py, t, cx, cy):
    """Rise contributed by the event owned by cell (cx, cy)."""
    exists = _hash12(cx + f32(19.71), cy + f32(19.71))
    r0 = _hash22(cx + f32(7.13), cy + f32(7.13))
    r1 = _hash22(cx + f32(23.91), cy + f32(23.91))
    r2 = _hash22(cx + f32(51.37), cy + f32(51.37))

    ctr = np.stack([(cx + 0.25 + 0.5 * r0[..., 0]) * CELL_SIZE,
                    (cy + 0.25 + 0.5 * r0[..., 1]) * CELL_SIZE], -1)
    u = _frac(t * TIME_SCALE + _hash12(cx + f32(37.11), cy + f32(37.11)))
    life = _life(u)
    ctr = ctr + (r1 * 2 - 1) * (650.0 * (u * 2 - 1)[..., None])

    rad = (MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * r1[..., 1]) * _radius_life(u)
    aspect = 0.68 + 0.32 * r2[..., 1]
    swap = r0[..., 0] < 0.5
    rx = np.where(swap, rad, rad * aspect)
    ry = np.where(swap, rad * aspect, rad)

    qx = (px - ctr[..., 0]) / np.maximum(rx, 1.0)
    qy = (py - ctr[..., 1]) / np.maximum(ry, 1.0)
    r_sq = qx * qx + qy * qy
    bump = np.clip(1.0 - r_sq, 0, None) ** 3
    height = 700.0 + (MAX_RISE - 700.0) * r2[..., 0]

    return np.where((exists < EXISTENCE) | (life <= 1e-4) | (r_sq >= 1.0),
                    0.0, height * life * bump)


def roof_rise(px, py, t):
    """max over the fixed 3x3 neighbourhood, as the shader does."""
    bx, by = np.floor(px / CELL_SIZE), np.floor(py / CELL_SIZE)
    r = np.zeros_like(px)
    for j in (-1, 0, 1):
        for i in (-1, 0, 1):
            r = np.maximum(r, event_rise(px, py, t, bx + i, by + j))
    return r


def scan(t, half_angle=1.0, near=2000.0, far=25000.0, step=2500.0, extent=60000.0):
    """Score candidate camera positions by what the view wedge contains.

    The test cameras look along +Y (getViewRay returns vec3(-uv.x, 1/fov, ...)),
    with roughly a 58 degree horizontal half angle at fov 1.6.

    Returns a list of (cx, cy, max_rise, n_tower_groups, strong_fraction).
    """
    ang = np.linspace(-half_angle, half_angle, 41)
    rng = np.linspace(near, far, 48)
    A, R = np.meshgrid(ang, rng, indexing="ij")
    dx, dy = -A, np.full_like(A, 0.625)
    n = np.sqrt(dx * dx + dy * dy)
    dx, dy = dx / n, dy / n

    rows = []
    for cx in np.arange(-10000.0, extent, step):
        for cy in np.arange(-10000.0, extent, step):
            rise = roof_rise((cx + dx * R).astype(f32), (cy + dy * R).astype(f32), f32(t))
            strong = rise > 3500.0
            occupied = strong.any(1).astype(int)          # per angular bin
            groups = int((occupied[1:] - occupied[:-1] == 1).sum() + (1 if occupied[0] else 0))
            rows.append((cx, cy, float(rise.max()), groups, float(strong.mean())))
    return rows


if __name__ == "__main__":
    from cw_gl import ITIME
    t = ITIME * 100.0 + 5000.0                # gameTime(iTime*100) + 5000
    rows = np.array(scan(t))
    print("cloudTime = %.0f" % t)
    print("\nflattest (cloud-sea candidates)   cx        cy     maxrise groups frac")
    for r in rows[np.argsort(rows[:, 2])][:5]:
        print("   %9.0f %9.0f  %6.0f   %d   %.3f" % tuple(r))
    tw = rows[rows[:, 3] >= 3]
    print("\nmulti-tower (cumulonimbus candidates)")
    for r in tw[np.argsort(-tw[:, 4])][:5]:
        print("   %9.0f %9.0f  %6.0f   %d   %.3f" % tuple(r))
