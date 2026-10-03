# Transport-Aware LOD Pilot: Local Residual Mean Hierarchy

- Rays: `700`
- Frequencies: `[1, 2, 4, 8, 16]`

Each omitted high-frequency tail is replaced by a piecewise-constant
residual mean field. R1, R2, and R4 denote 1, 2, and 4 macro voxels
per world unit along each axis.

## K=3

| Method | Voxels | Mean abs tau error | P95 abs tau error | P99 T error |
|---|---:|---:|---:|---:|
| `global` | 1 | 1.765e-02 | 4.813e-02 | 1.292e-02 |
| `R1` | 27 | 1.730e-02 | 4.491e-02 | 1.287e-02 |
| `R2` | 216 | 1.678e-02 | 4.246e-02 | 1.250e-02 |
| `R4` | 1728 | 1.507e-02 | 3.852e-02 | 1.117e-02 |

## K=4

| Method | Voxels | Mean abs tau error | P95 abs tau error | P99 T error |
|---|---:|---:|---:|---:|
| `global` | 1 | 5.778e-03 | 1.459e-02 | 4.167e-03 |
| `R1` | 27 | 5.755e-03 | 1.484e-02 | 4.253e-03 |
| `R2` | 216 | 5.712e-03 | 1.448e-02 | 4.391e-03 |
| `R4` | 1728 | 5.491e-03 | 1.346e-02 | 3.915e-03 |

## Interpretation

- Global mean compensation is the zero-memory baseline.
- A local hierarchy is useful only if residual energy varies coherently
  at scales coarser than the omitted octaves.
- If finer local grids do not improve held-out ray error, the residual
  is already self-averaging and storage should not be spent on local means.