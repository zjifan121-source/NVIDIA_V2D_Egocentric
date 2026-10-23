"""Canonical mesh per object (V3_BUILD_SPEC.md §11 A2.2): every clip's partial scan of one object
(s3a2_fuse_scan), registered and completed by the object's symmetry; videos only (no scan, no GT).

    uv run --project path_b python path_b/run.py canonical --object white_pot --out <run> [--episodes ...]

1. Partials: <run>/episode_*/s3a2_fuse_scan/<object>/partial_scan.ply (metres, each clip's v3 world,
   the object at its frame-0 placement). Reference: the largest surface; every other partial onto it by
   FPFH + RANSAC (best of 3 seeds), point-to-point ICP (two clips often share only the top, whose
   outline only point-to-point uses; it slides slowly: 300 iterations), then point-to-plane to finish;
   scale fixed; kept if fitness >= 0.3.
2. Fused: the registered partials' points, one 4 mm voxel grid (each partial is a TSDF average already).
3. Symmetry (roster, spec §11): revolution (white_pot, white_pot_lid, blue_cup, beige_cup,
   water_pitcher): the axis starts vertical (they rest upright) and is refined through robust circle
   fits per 5 mm height slice of the side wall; completion revolves the profile (4 mm cells in the
   radius-height plane seen over at least half the azimuth seen at their height: walls and caps). Mirror (mini_dust_pan,
   wooden_piece_1/2): the vertical plane maximising the mirrored cloud's overlap (points within 5 mm),
   normals every 5 degrees tilted up to 10, offsets within 3 cm; completion mirrors the cloud. Others
   (plastic_dish_rack, mini_sweeper, wooden_spoon): mirror if its overlap >= 0.6, else the fused cloud.
   Observed points are kept, so handles and spouts stay where they were seen.
4. The table side is never seen: a flat cap of points at the table plane (z = 0 of the reference clip's
   world) under the object's lowest band closes it.
5. Screened Poisson (depth 8) on the oriented points; density-trimmed only if it stays watertight;
   decimated to <= 20,000 faces. Where the clips leave a side unseen, Poisson inflates it: if its mesh
   leaks or outgrows the cloud (extents outside 0.8-1.25x), the voxel solid instead (4 mm occupancy,
   closing, filled, smoothed field, marching cubes, the voxel coarsened until <= 20,000 faces):
   watertight by construction, tight to what was seen.
Frame: the reference clip's world, shifted so the origin is the footprint centre on the table (z up).
Out: <run>/canonical/<object>/mesh.glb, mesh.obj, canonical.json (clips and registration, symmetry fit,
coverage before completion, watertight, world-axis extents vs the completed cloud (bar 0.8-1.25) and vs
the fused partial; coverage < 0.6 flagged).
The learned completion fallbacks (SymmCompletion, AdaPoinTr, ForeHOI) need torch and their checkpoints:
not wired; canonical.json says when one would have been the next step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import scan as S

REVOLUTION = {"white_pot", "white_pot_lid", "blue_cup", "beige_cup", "water_pitcher"}
MIRROR = {"mini_dust_pan", "wooden_piece_1", "wooden_piece_2"}
MIN_FITNESS = 0.3
SLICE = 0.005  # m
NEAR = 0.005  # m: overlap and coverage distance
MIRROR_CONFIDENT = 0.6
MAX_FACES = 20000
POISSON_DEPTH = 8
COVERAGE_FLAG = 0.6


# ---------------------------------------------------------------- 1-2: partials onto one reference
def area(mesh: o3d.geometry.TriangleMesh) -> float:
    return float(mesh.get_surface_area()) if len(mesh.triangles) else 0.0


def register(partials: list[o3d.geometry.TriangleMesh]) -> tuple[int, list[dict]]:
    """Reference index and, per partial, its move onto the reference (None if it did not fit)."""
    ref = int(np.argmax([area(m) for m in partials]))
    target = S.mesh_points(partials[ref], 20000)
    out = []
    for i, m in enumerate(partials):
        if i == ref:
            out.append({"T": np.eye(4), "fitness": 1.0, "rmse_m": 0.0, "kept": True})
            continue
        source = S.mesh_points(m, 20000)
        best = None
        for seed in range(3):
            T0 = S.global_start(source, target, seed=seed)
            T = S.icp(source, target, T0, max_dist=0.02, iterations=300, point_to_point=True)["T"]  # outlines
            r = S.icp(source, target, T, max_dist=0.005)
            if best is None or r["fitness"] > best["fitness"]:
                best = r
        out.append(best | {"kept": best["fitness"] >= MIN_FITNESS})
    return ref, out


def fused_cloud(partials: list, moves: list[dict]) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    for m, mv in zip(partials, moves):
        if mv["kept"]:
            pc += S.mesh_points(m, 40000).transform(mv["T"])
    return pc.voxel_down_sample(S.VOXEL)


# ---------------------------------------------------------------- 3: symmetry
def circle(xy: np.ndarray, rounds: int = 3) -> tuple[np.ndarray, float, np.ndarray]:
    """Robust algebraic circle fit: centre (2,), radius, inlier mask (residual <= 5 mm)."""
    keep = np.ones(len(xy), bool)
    for _ in range(rounds):
        A = np.c_[2 * xy[keep], np.ones(keep.sum())]
        b = (xy[keep] ** 2).sum(axis=1)
        (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
        r = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0)))
        resid = np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - r)
        keep = resid <= max(NEAR, np.percentile(resid, 60))
    return np.array([cx, cy]), r, keep


def basis(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def fit_revolution(points: np.ndarray, normals: np.ndarray, iterations: int = 4) -> dict:
    """Axis and radius profile. Starts vertical: every revolution object of the roster rests upright
    (a partial view skews the principal axes, so symmetry.axis would not do). The circle fits use the
    side wall only (normals across the axis): a cap's points fill the circle and pull its centre."""
    a, c = np.array([0.0, 0.0, 1.0]), points.mean(axis=0)
    all_points = points
    slices = []
    for _ in range(iterations):
        points = all_points[np.abs(normals @ a) < 0.5]
        u, v = basis(a)
        h = (points - c) @ a
        centres, weights, slices = [], [], []
        for lo in np.arange(h.min(), h.max(), SLICE):
            sel = (h >= lo) & (h < lo + SLICE)
            if sel.sum() < 20:
                continue
            q = points[sel] - c
            xy = np.c_[q @ u, q @ v]
            (cx, cy), r, inl = circle(xy)
            centres.append(c + (lo + SLICE / 2) * a + cx * u + cy * v)
            weights.append(inl.sum())
            slices.append({"h": float(lo + SLICE / 2), "r": r, "inliers": int(inl.sum()), "points": int(sel.sum())})
        if len(centres) < 3:
            break
        P, w = np.asarray(centres), np.asarray(weights, float)
        m = (P * w[:, None]).sum(axis=0) / w.sum()
        _, _, vt = np.linalg.svd((P - m) * np.sqrt(w)[:, None], full_matrices=False)
        a_new = vt[0] * np.sign(vt[0] @ a)
        c, a = m, a_new / np.linalg.norm(a_new)
    return {"axis": a, "point": c, "slices": len(slices)}


def generatrix(points: np.ndarray, normals: np.ndarray, fit: dict, azimuth_share: float = 0.5) -> dict:
    """The surface's profile in the (radius, height) plane about the axis: 4 mm cells seen over at least
    azimuth_share of the azimuth range seen at their height (walls and caps; a handle or spout, seen
    over a narrow azimuth, stays out), each with its mean radius, height and normal (radial and axial
    parts). Per height, because a cap seen from above spans every azimuth while the wall below it
    shows only the side facing the camera."""
    a, c = fit["axis"], fit["point"]
    u, v = basis(a)
    q = points - c
    h = q @ a
    rho = q - np.outer(h, a)
    r = np.linalg.norm(rho, axis=1)
    phi_bin = ((np.arctan2(rho @ v, rho @ u) + np.pi) // np.radians(10)).astype(int)
    h_bin = (h // S.VOXEL).astype(int)
    seen = {}
    for hb, pb in zip(h_bin, phi_bin):
        seen.setdefault(hb, set()).add(pb)
    radial = rho / np.maximum(r, 1e-9)[:, None]
    n_r, n_a = (normals * radial).sum(axis=1), normals @ a
    cells = {}
    for k, key in enumerate(zip((r // S.VOXEL).astype(int), h_bin)):
        cells.setdefault(key, []).append(k)
    rows, resid = [], []
    for (_, hb), idx in cells.items():
        idx = np.asarray(idx)
        if len(idx) < 5 or len(np.unique(phi_bin[idx])) < azimuth_share * len(seen[hb]):
            continue
        nr, na = n_r[idx].mean(), n_a[idx].mean()
        norm = np.hypot(nr, na) or 1.0
        rows.append((r[idx].mean(), h[idx].mean(), nr / norm, na / norm))
        resid.append(np.abs(r[idx] - r[idx].mean()))
    profile = np.asarray(rows)
    return {"profile": profile, "cells": len(rows), "kept_share": float(sum(map(len, resid)) / max(len(points), 1)),
            "residual_m": float(np.median(np.concatenate(resid))) if resid else None}


def revolve(fit: dict, gen: dict) -> tuple[np.ndarray, np.ndarray]:
    a, c = fit["axis"], fit["point"]
    u, v = basis(a)
    pts, nrm = [], []
    for r, h, nr, na in gen["profile"]:
        steps = max(12, int(2 * np.pi * max(r, S.VOXEL) / S.VOXEL))
        t = np.linspace(0, 2 * np.pi, steps, endpoint=False)
        ring = np.outer(np.cos(t), u) + np.outer(np.sin(t), v)
        pts.append(c + h * a + r * ring)
        nrm.append(nr * ring + na * a)
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(pts), np.concatenate(nrm)


def reflect(points: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    return points - 2 * ((points @ n) - d)[:, None] * n


def overlap(points: np.ndarray, tree: cKDTree, n: np.ndarray, d: float) -> float:
    return float(np.mean(tree.query(reflect(points, n, d))[0] <= NEAR))


def fit_mirror(points: np.ndarray) -> dict:
    """The vertical plane n.x = d maximising the mirrored overlap: horizontal normals every 5 degrees,
    tilted up to 10 degrees, offsets within 3 cm of the centroid; then a fine offset. Vertical only:
    a resting object's mirror plane stands upright, and a horizontal one would fold the seen top onto
    the unseen bottom (the cap closes that side)."""
    rng = np.random.default_rng(0)
    sample = points[rng.choice(len(points), min(4000, len(points)), replace=False)]
    tree = cKDTree(points)
    c = points.mean(axis=0)
    best = {"overlap": -1.0}
    for yaw in np.radians(np.arange(0, 180, 5)):
        horizontal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        side = np.cross([0, 0, 1.0], horizontal)
        for tilt in np.radians([-10, -5, 0, 5, 10]):
            n = Rotation.from_rotvec(side * tilt).apply(horizontal)
            for shift in np.arange(-0.03, 0.0301, 0.004):
                d = float(n @ c + shift)
                o = overlap(sample, tree, n, d)
                if o > best["overlap"]:
                    best = {"overlap": o, "normal": n, "d": d}
    n, d = best["normal"], best["d"]
    for shift in np.arange(-0.004, 0.0041, 0.001):
        o = overlap(sample, tree, n, d + shift)
        if o > best["overlap"]:
            best = {"overlap": o, "normal": n, "d": d + shift}
    return best


# ---------------------------------------------------------------- 4-5: closing and meshing
def bottom_cap(points: np.ndarray, band: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Points on the table plane under the object's lowest band (the side the camera never sees)."""
    from scipy.spatial import ConvexHull, Delaunay
    low = points[points[:, 2] < points[:, 2].min() + band][:, :2]
    if len(low) < 10:
        return np.zeros((0, 3)), np.zeros((0, 3))
    hull = Delaunay(low[ConvexHull(low).vertices])
    lo, hi = low.min(axis=0), low.max(axis=0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], S.VOXEL), np.arange(lo[1], hi[1], S.VOXEL))
    grid = np.c_[gx.ravel(), gy.ravel()]
    grid = grid[hull.find_simplex(grid) >= 0]
    z = min(0.0, float(points[:, 2].min()))
    return np.c_[grid, np.full(len(grid), z)], np.tile([0, 0, -1.0], (len(grid), 1))


def poisson(points: np.ndarray, normals: np.ndarray) -> o3d.geometry.TriangleMesh:
    pc = S.cloud(points, normals).voxel_down_sample(S.VOXEL)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=POISSON_DEPTH)
    labels, counts, _ = mesh.cluster_connected_triangles()  # Poisson can leave small closed bubbles
    keep = int(np.argmax(np.asarray(counts)))
    vertex_keep = np.zeros(len(mesh.vertices), bool)
    vertex_keep[np.asarray(mesh.triangles)[np.asarray(labels) == keep].ravel()] = True
    mesh.remove_triangles_by_mask(np.asarray(labels) != keep)
    density = np.asarray(density)[vertex_keep]
    mesh.remove_unreferenced_vertices()
    trimmed = o3d.geometry.TriangleMesh(mesh)
    trimmed.remove_vertices_by_mask(np.asarray(density) < np.quantile(np.asarray(density), 0.01))
    mesh = trimmed if trimmed.is_watertight() else mesh
    if len(mesh.triangles) > MAX_FACES:  # may open it: close_surface then takes the voxel solid
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=MAX_FACES)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def voxel_solid(points: np.ndarray, close: int = 2) -> o3d.geometry.TriangleMesh:
    """A watertight surface around the points: occupancy, morphological closing, interior filled, the
    field smoothed (Gaussian, 0.75 voxel), marching cubes at 0.5. Smoothing the field, not the mesh:
    a smoothed mesh folds thin parts into themselves. Not decimated (quadric decimation opens it):
    the voxel grows from 4 mm until the mesh has <= MAX_FACES faces."""
    from scipy import ndimage
    from skimage.measure import marching_cubes
    voxel = S.VOXEL
    for _ in range(6):
        lo = points.min(axis=0) - (close + 2) * voxel
        idx = np.floor((points - lo) / voxel).astype(int)
        occ = np.zeros(idx.max(axis=0) + close + 3, bool)
        occ[tuple(idx.T)] = True
        occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=close))
        field = ndimage.gaussian_filter(np.pad(occ, 2).astype(np.float32), 0.75)
        verts, faces, _, _ = marching_cubes(field, level=0.5)
        if len(faces) <= MAX_FACES:
            break
        voxel *= 1.05 * np.sqrt(len(faces) / MAX_FACES)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector((verts - 2) * voxel + lo),
                                     o3d.utility.Vector3iVector(faces[:, ::-1].astype(np.int32)))
    mesh.compute_vertex_normals()
    return mesh


def close_surface(points: np.ndarray, normals: np.ndarray) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Poisson, unless it leaks or balloons (world-axis extents vs the cloud outside 0.8-1.25): then
    the voxel solid. Which one and why go into canonical.json."""
    mesh = poisson(points, normals)
    ratio = np.ptp(np.asarray(mesh.vertices), axis=0) / np.ptp(points, axis=0)
    if mesh.is_watertight() and np.all((ratio >= 0.8) & (ratio <= 1.25)):
        return mesh, {"closing": "screened Poisson"}
    why = ("not watertight" if not mesh.is_watertight() else "") + \
          (f"{' and ' if not mesh.is_watertight() else ''}ballooned (extents x{ratio.round(2).tolist()})"
           if not np.all((ratio >= 0.8) & (ratio <= 1.25)) else "")
    return voxel_solid(points), {"closing": "voxel solid", "poisson": why}


def build(name: str, partials: list[o3d.geometry.TriangleMesh]) -> tuple[o3d.geometry.TriangleMesh, dict]:
    ref, moves = register(partials)
    fused = fused_cloud(partials, moves)
    pts, nrm = np.asarray(fused.points), np.asarray(fused.normals)
    report = {"reference": ref, "moves": [{k: v for k, v in m.items()} for m in moves], "fused_points": len(pts)}
    kind = "revolution" if name in REVOLUTION else "mirror" if name in MIRROR else "none"
    extra = [np.zeros((0, 3))], [np.zeros((0, 3))]
    if kind == "revolution":
        fit = fit_revolution(pts, nrm)
        gen = generatrix(pts, nrm, fit)
        rp, rn = revolve(fit, gen)
        extra = [rp], [rn]
        report["symmetry"] = {"kind": kind, "axis": fit["axis"], "point": fit["point"], "slices": fit["slices"],
                              "profile_cells": gen["cells"], "points_on_profile": gen["kept_share"],
                              "radial_residual_m": gen["residual_m"]}
    else:
        fit = fit_mirror(pts)
        use = kind == "mirror" or fit["overlap"] >= MIRROR_CONFIDENT
        if use:
            n, d = fit["normal"], fit["d"]
            extra = [reflect(pts, n, d)], [nrm - 2 * (nrm @ n)[:, None] * n]
        report["symmetry"] = {"kind": "mirror" if use else "none", "roster": kind, "normal": fit["normal"],
                              "d": fit["d"], "overlap": fit["overlap"], "applied": bool(use)}
    cap_p, cap_n = bottom_cap(np.concatenate([pts, *extra[0]]))
    all_p = np.concatenate([pts, *extra[0], cap_p])
    all_n = np.concatenate([nrm, *extra[1], cap_n])
    mesh, closing = close_surface(all_p, all_n)
    report.update(closing)
    # canonical frame: the footprint centre on the table, z up
    v = np.asarray(mesh.vertices)
    origin = np.array([*((v[:, :2].min(axis=0) + v[:, :2].max(axis=0)) / 2), 0.0])
    mesh.translate(-origin)
    surface = np.asarray(S.mesh_points(mesh, 20000).points) + origin
    coverage = float(np.mean(cKDTree(pts).query(surface)[0] <= NEAR))
    world = np.asarray(mesh.vertices) + origin
    ratio_cloud, ratio_fused = np.ptp(world, axis=0) / np.ptp(all_p, axis=0), np.ptp(world, axis=0) / np.ptp(pts, axis=0)
    report.update(world_to_canonical=-origin, coverage_before_completion=coverage, cap_points=len(cap_p),
                  watertight=bool(mesh.is_watertight()), faces=len(mesh.triangles),
                  extents_cm=(np.ptp(world, axis=0) * 100).round(1),
                  # bar against the completed cloud (Poisson must not balloon); against the fused partial
                  # completion legitimately doubles an unseen side, so that ratio is information only
                  extents_vs_completed=ratio_cloud.round(3), extents_vs_fused=ratio_fused.round(3),
                  extents_ok=bool(np.all((ratio_cloud >= 0.8) & (ratio_cloud <= 1.25))),
                  flags=(["coverage < 0.6"] if coverage < COVERAGE_FLAG else [])
                  + (["not watertight"] if not mesh.is_watertight() else []),
                  fallbacks="not wired (SymmCompletion / AdaPoinTr / ForeHOI need torch and checkpoints)"
                  + ("; next step for this object" if coverage < COVERAGE_FLAG or not mesh.is_watertight() else ""))
    return mesh, report


def to_json(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))


def main(name: str, out: Path, episodes: list[int] | None = None) -> Path:
    import trimesh
    found = sorted(p for p in out.glob(f"episode_*/s3a2_fuse_scan/{name}/partial_scan.ply")
                   if episodes is None or int(p.parts[-4][len("episode_"):]) in episodes)
    partials, used = [], []
    for p in found:
        m = o3d.io.read_triangle_mesh(str(p))
        if len(m.triangles):
            m.compute_vertex_normals()
            partials.append(m)
            used.append(p)
    if not partials:
        raise SystemExit(f"no partial scan of {name} under {out}/episode_*/s3a2_fuse_scan/")
    mesh, report = build(name, partials)
    dest = out / "canonical" / name
    dest.mkdir(parents=True, exist_ok=True)
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
    tm.export(dest / "mesh.obj")
    tm.export(dest / "mesh.glb")
    report = {"object": name, "clips": [str(p.relative_to(out)) for p in used], **report}
    (dest / "canonical.json").write_text(json.dumps(report, indent=1, default=to_json) + "\n")
    print(f"canonical {name}: {len(used)} clips, {report['faces']} faces, watertight {report['watertight']}, "
          f"coverage {report['coverage_before_completion']:.2f}, {report['symmetry']['kind']} -> {dest}", flush=True)
    return dest
