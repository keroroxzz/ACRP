"""Phase-H2 cell-statistics and transmittance-closure benchmark.

The empirical reference is E[exp(-k*rho)] over the same 4^3 samples used to
build the H1 hierarchy. B1 keeps only the mean. B2 adds the variance cumulant.
This is a cell-distribution experiment, not yet a full ray radiance closure.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import glfw
import numpy as np
from OpenGL.GL import *

import bench_h1 as h1


HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(HERE)
OUT = os.path.join(ARTICLE, "results", "h2")
SHADER = os.path.join(HERE, "sky_frag.glsl")
K_VALUES = (0.5, 1.0, 2.0, 4.0, 8.0)
SEGMENT_LENGTHS_KM = (0.03, 0.06, 0.125, 0.25, 0.5, 1.0)


def percentile(values, q):
    if values.size == 0:
        return None
    return float(np.percentile(values.astype(np.float64), q))


def error_stats(estimate, reference, mask):
    rel = np.abs(estimate-reference) / np.maximum(reference, 1e-4)
    values = rel[mask]
    return {
        "count": int(values.size),
        "mean_relative_error": (
            float(values.mean()) if values.size else None
        ),
        "p50_relative_error": percentile(values, 50),
        "p95_relative_error": percentile(values, 95),
        "max_relative_error": (
            float(values.max()) if values.size else None
        ),
    }


def read_texture(texture):
    nx, ny, nz = h1.GRID_DIMS
    glBindTexture(GL_TEXTURE_3D, texture)
    raw = glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT)
    return np.frombuffer(raw, np.float32).reshape(ny, nz, nx, 4).copy()


def main():
    fragment = open(SHADER, encoding="utf-8").read()
    shader_hash = hashlib.sha256(fragment.encode("utf-8")).hexdigest()
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    for key, value in (
        (glfw.VISIBLE, glfw.FALSE),
        (glfw.CONTEXT_VERSION_MAJOR, 3),
        (glfw.CONTEXT_VERSION_MINOR, 3),
        (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
    ):
        glfw.window_hint(key, value)
    window = glfw.create_window(256, 256, "research7-h2", None, None)
    if window is None:
        raise RuntimeError("OpenGL context creation failed")
    glfw.make_context_current(window)
    vao = glGenVertexArrays(1)
    os.makedirs(OUT, exist_ok=True)

    records = []
    arrays = {}
    total_build_ms = 0.0
    moment_texture, moment_build_ms = h1.build_macro_texture(
        fragment, vao, cumulants=True
    )
    moments = read_texture(moment_texture)
    glDeleteTextures([moment_texture])
    moment_mean, moment_variance, kappa3, kappa4 = [
        moments[..., i] for i in range(4)
    ]
    total_build_ms += moment_build_ms
    for k in K_VALUES:
        texture, build_ms = h1.build_macro_texture(
            fragment, vao, closure_k=k
        )
        total_build_ms += build_ms
        data = read_texture(texture)
        maximum, mean, variance, reference = [
            data[..., i] for i in range(4)
        ]
        active = maximum > 0.001
        fill = mean / np.maximum(maximum, 1e-6)
        normalized_variance = variance / np.maximum(mean*mean, 1e-6)
        interior = (
            active & (maximum > 0.02)
            & (fill >= 0.25) & (normalized_variance <= 2.0)
        )
        boundary = active & (~interior)
        cumulant_risk = 0.5*k*k*variance
        analytic = interior & (cumulant_risk <= 0.1)
        fallback = active & (~analytic)

        b0 = np.exp(-k*maximum)
        b1 = np.exp(-k*mean)
        b2 = np.clip(np.exp(-k*mean+0.5*k*k*variance), 0.0, 1.0)
        b3 = np.clip(
            np.exp(
                -k*moment_mean + 0.5*k*k*moment_variance
                - (k**3)*kappa3/6.0
            ),
            0.0, 1.0,
        )
        b4 = np.clip(
            np.exp(
                -k*moment_mean + 0.5*k*k*moment_variance
                - (k**3)*kappa3/6.0 + (k**4)*kappa4/24.0
            ),
            0.0, 1.0,
        )
        item = {
            "k": k,
            "gpu_build_ms": build_ms,
            "active_cells": int(active.sum()),
            "interior_cells": int(interior.sum()),
            "boundary_cells": int(boundary.sum()),
            "analytic_cells": int(analytic.sum()),
            "fallback_cells": int(fallback.sum()),
            "analytic_fraction_of_active": float(
                analytic.sum()/max(int(active.sum()), 1)
            ),
            "b0_max_density": {
                "interior": error_stats(b0, reference, interior),
                "boundary": error_stats(b0, reference, boundary),
            },
            "b1_mean": {
                "interior": error_stats(b1, reference, interior),
                "boundary": error_stats(b1, reference, boundary),
                "analytic": error_stats(b1, reference, analytic),
            },
            "b2_variance": {
                "interior": error_stats(b2, reference, interior),
                "boundary": error_stats(b2, reference, boundary),
                "analytic": error_stats(b2, reference, analytic),
            },
            "b3_third_cumulant": {
                "interior": error_stats(b3, reference, interior),
                "analytic": error_stats(b3, reference, analytic),
            },
            "b4_fourth_cumulant": {
                "interior": error_stats(b4, reference, interior),
                "analytic": error_stats(b4, reference, analytic),
            },
        }
        records.append(item)
        key = ("k%.1f" % k).replace(".", "_")
        arrays[key+"_indices"] = np.argwhere(active).astype(np.int16)
        arrays[key+"_stats"] = np.stack(
            [maximum[active], mean[active], variance[active],
             reference[active], b0[active], b1[active], b2[active],
             fill[active], normalized_variance[active],
             interior[active], analytic[active], cumulant_risk[active],
             kappa3[active], kappa4[active], b3[active], b4[active]],
            axis=-1,
        ).astype(np.float32)
        print(
            "k=%3.1f cells=%6d interior=%5d analytic=%4d  "
            "analytic B1=%7s B2=%7s B3=%7s B4=%7s build=%7.3f ms" %
            (
                k, active.sum(), interior.sum(), analytic.sum(),
                (
                    "%.4f" % item["b1_mean"]["analytic"]["p95_relative_error"]
                    if analytic.any() else "fallback"
                ),
                (
                    "%.4f" %
                    item["b2_variance"]["analytic"]["p95_relative_error"]
                    if analytic.any() else "fallback"
                ),
                (
                    "%.4f" %
                    item["b3_third_cumulant"]["analytic"]["p95_relative_error"]
                    if analytic.any() else "fallback"
                ),
                (
                    "%.4f" %
                    item["b4_fourth_cumulant"]["analytic"]["p95_relative_error"]
                    if analytic.any() else "fallback"
                ),
                build_ms,
            )
        )
        glDeleteTextures([texture])

    analytic_p95 = [
        r["b2_variance"]["analytic"]["p95_relative_error"] for r in records
    ]
    b2_better = [
        (
            r["analytic_cells"] == 0
            or r["b2_variance"]["analytic"]["p95_relative_error"]
            < r["b1_mean"]["analytic"]["p95_relative_error"]
        )
        for r in records
    ]
    higher_better = [
        (
            r["analytic_cells"] == 0
            or (
                r["b4_fourth_cumulant"]["analytic"]["p95_relative_error"]
                < r["b2_variance"]["analytic"]["p95_relative_error"]
            )
        )
        for r in records
    ]
    result = {
        "phase": "H2",
        "experiment": "cell-density transmittance moments",
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": glGetString(GL_RENDERER).decode("utf-8", "replace"),
        "shader_sha256": shader_hash,
        "grid_dimensions_xyz": list(h1.GRID_DIMS),
        "samples_per_cell": 64,
        "cumulant_gpu_build_ms": moment_build_ms,
        "total_gpu_build_ms": total_build_ms,
        "classification": {
            "active": "densityMax > 0.001",
            "interior": (
                "max > 0.02, mean/max >= 0.25, variance/mean^2 <= 2"
            ),
            "boundary": "active and not interior",
            "analytic": "interior and 0.5*k^2*variance <= 0.1",
            "fallback": "active and not analytic",
        },
        "gates": {
            "analytic_b2_p95_relative_error_under_2pct_or_fallback": all(
                value is None or value < 0.02 for value in analytic_p95
            ),
            "b2_p95_better_than_b1_all_scales": all(b2_better),
            "b4_p95_better_than_b2_all_scales": all(higher_better),
            "boundary_fallback_defined": True,
            "full_ray_validation": False,
        },
        "optical_scales": records,
        "physical_segment_sweep": [
            {
                "length_km": length,
                "optical_scale": 7.0*2.4*length,
                "accepted_interior_cells": int(
                    (interior & (
                        0.5*(7.0*2.4*length)**2*variance <= 0.1
                    )).sum()
                ),
                "accepted_fraction_of_interior": float(
                    (interior & (
                        0.5*(7.0*2.4*length)**2*variance <= 0.1
                    )).sum()/max(int(interior.sum()), 1)
                ),
            }
            for length in SEGMENT_LENGTHS_KM
        ],
        "limitations": [
            "Empirical reference uses within-cell density samples, not full rays.",
            "Third/fourth cumulants are measured in a separate offline pass.",
            "Boundary cells are classified for fallback, not analytically closed.",
        ],
    }
    with open(os.path.join(OUT, "h2_cell_closure.json"), "w",
              encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(os.path.join(OUT, "h2_cell_records.npz"), **arrays)
    glfw.terminate()


if __name__ == "__main__":
    main()
