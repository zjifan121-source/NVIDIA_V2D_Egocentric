"""B1 registration-ambiguity test (V4_BUILD_SPEC.md §13.2): the scorer registers our mesh onto the scan
from 25 PCA starts and keeps the start with the lowest symmetric mean distance; a near-symmetric mesh
can win flipped (public: ep13's cup 165 degrees, ep31's sweeper 179, ep0's pot 146), and the whole scene
is aligned from that object's frame 0. Here the same recipe registers our mesh onto our own rest scan
(GT-free): margin = the best distinct runner-up's cost - the best cost (m); a small margin flags an
anchor the scorer may flip. Distinct: more than DISTINCT_DEG apart (for a body of revolution, its axis,
the mesh's z, more than DISTINCT_DEG apart: turns about the axis are the same pose).

The kit's helpers (tools/track3/score_local.py -> metric_code/track_3/AUC.py: _surface_moments,
_sample_mesh_surface, _initial_rotations, _kabsch, _symmetric_mean_distance and its constants) with one
change: scale fixed (both are metric; the kit's RMS-radius rescale would shrink our mesh to a partial's
radius).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

DISTINCT_DEG = 30.0  # default: a runner-up this far from the best is another pose


def angle_deg(a: np.ndarray, b: np.ndarray, axis_only: bool) -> float:
    """Angle between two rotations; axis_only: between the images of the mesh's z axis (column 2)."""
    if axis_only:
        return float(np.degrees(np.arccos(np.clip(a[:, 2] @ b[:, 2], -1, 1))))
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def margin(mesh_v: np.ndarray, mesh_f: np.ndarray, scan_v: np.ndarray, scan_f: np.ndarray, seed: int,
           revolution: bool = False) -> dict:
    """{best_cost_m, runner_up_cost_m, margin_m, runner_up_deg, starts}: our mesh onto the rest scan."""
    from .team import scorer
    M = scorer().metric_module("AUC")
    cc, _ = M._surface_moments(np.asarray(mesh_v, float), np.asarray(mesh_f))
    rc, _ = M._surface_moments(np.asarray(scan_v, float), np.asarray(scan_f))
    cp = M._sample_mesh_surface(np.asarray(mesh_v, float), np.asarray(mesh_f), M._ALIGNMENT_SAMPLES, seed) - cc
    rp = M._sample_mesh_surface(np.asarray(scan_v, float), np.asarray(scan_f), M._ALIGNMENT_SAMPLES, seed) - rc
    tree = cKDTree(rp)
    results = []
    for start in M._initial_rotations(rp, cp, M._ALIGNMENT_STARTS):  # the kit's _align_rigid, every start kept
        rotation = start.copy()
        translation = rp.mean(axis=0) - rotation @ cp.mean(axis=0)
        previous = math.inf
        for _ in range(M._ICP_ITERATIONS):
            moved = cp @ rotation.T + translation
            distances, indices = tree.query(moved, k=1)
            keep = max(3, int(math.ceil(len(distances) * M._TRIM_FRACTION)))
            chosen = np.argpartition(distances, keep - 1)[:keep] if keep < len(distances) else np.arange(len(distances))
            step_r, step_t = M._kabsch(moved[chosen], rp[indices[chosen]])
            rotation, translation = step_r @ rotation, step_r @ translation + step_t
            error = float(np.mean(distances[chosen]))
            if math.isfinite(previous) and abs(previous - error) <= 1e-8 * max(previous, 1.0):
                break
            previous = error
        results.append((rotation, float(M._symmetric_mean_distance(rp, cp @ rotation.T + translation))))
    best_r, best = min(results, key=lambda x: x[1])
    other = [(r, s) for r, s in results if angle_deg(best_r, r, revolution) > DISTINCT_DEG]
    if not other:
        return {"best_cost_m": best, "runner_up_cost_m": None, "margin_m": None, "runner_up_deg": None,
                "starts": len(results)}
    run_r, runner = min(other, key=lambda x: x[1])
    return {"best_cost_m": best, "runner_up_cost_m": runner, "margin_m": runner - best,
            "runner_up_deg": angle_deg(best_r, run_r, revolution), "starts": len(results)}
