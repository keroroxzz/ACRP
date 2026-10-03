# Research 7 Agent Instructions

This directory is the active workspace for Research 7: scale-separated hybrid
rendering of kilometer-scale procedural clouds.

## Governing document

Read `ROADMAP.md` before doing research or implementation work. Treat its phase
gates as hypotheses to test, not as achieved results.

## Current phase

Phase H0: establish a strong, stable production raymarch baseline before
implementing A3-Lite or transport closures.

## Source policy

- Research 7 code, measurements, notes, and generated outputs belong under this
  directory.
- The root `a3/` tree is the live upstream implementation. It may be read and
  benchmarked, but Research 7 must record the exact source revision or file hash
  when using it.
- Never modify the frozen references:
  `a3/gpu/a3_production.glsl` and `a3/a3/kernel/a3_gn_kernel.py`.
- Do not edit completed-paper archive code to change Research 7 behavior.
- Keep GLSL source ASCII-only and always open GLSL files with UTF-8 encoding.

## Measurement policy

- Use `GL_TIME_ELAPSED`, not CPU wall time, for GPU pass timings.
- Warm the GPU with continuous draws for about 1.5 seconds before measurement.
- Report multiple views using median, p95, range, and worst case. Do not promote
  a favorable single-view result.
- Preserve `glReadPixels` orientation by vertically flipping saved images.
- Label design targets separately from measured results.
- Compare quality against a clean, non-jittered high-sample reference.
- Preserve a strong production raymarch baseline; do not weaken it to improve a
  reported hybrid speedup.

## Research hygiene

- Record commands, environment, GPU, shader hash, parameters, and raw JSON.
- Keep train/validation/test splits separated by procedural configuration when
  the MLP dataset phase begins.
- Commit only when the user asks.

