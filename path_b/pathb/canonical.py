"""Canonical mesh per object (V3_BUILD_SPEC.md §11 A2.2): every clip's partial scan of one object
(s3a2_fuse_scan), registered and completed by the object's symmetry; videos only (no scan, no GT).

    uv run --project path_b python path_b/run.py canonical --object white_pot --out <run> [--episodes ...]

1. Partials: <run>/episode_*/s3a2_fuse_scan/<object>/partial_scan.ply (metres, each clip's v3 world,
   the object at its frame-0 placement), each fused from >= MIN_CLIP_FRAMES frames (all when none is).
   Reference: the largest surface; every other partial onto it by
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

MAX_TILT_DEG = 5.0  # default: a resting body of revolution stands within this of the table normal
WALL_MARGIN = 1.15  # default: profile cells beyond this x the wall circles' (90th percentile) radius are a
# handle or spout: at a handle's height little else is seen, so the azimuth test alone let ep41's pot
# handle be revolved into a 49 cm disc
REVOLUTION = {"white_pot", "white_pot_lid", "blue_cup", "beige_cup", "water_pitcher"}
MIRROR = {"mini_dust_pan", "wooden_piece_1", "wooden_piece_2"}
MIN_FITNESS = 0.3
SLICE = 0.005  # m
NEAR = 0.005  # m: overlap and coverage distance
MIRROR_CONFIDENT = 0.6
MAX_FACES = 20000
POISSON_DEPTH = 8
COVERAGE_FLAG = 0.6
MIN_CLIP_FRAMES = 10  # default: a partial fused from fewer frames is left out (s3a2_fuse_scan's WARN bar)


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
def circle(xy: np.ndarray, rounds: int = 3, tries: int = 300, max_r: float = 0.3) -> tuple[np.ndarray, float, np.ndarray]:
    """Robust circle fit: centre (2,), radius, inlier mask (residual <= 5 mm). RANSAC over triples (the
    most points within NEAR of one circle of radius <= max_r), then algebraic refits on the inliers. A
    trimmed algebraic fit alone settled on a circle through a pot's wall and its handle (ep41: the
    revolved mesh came out 48 cm across)."""
    rng = np.random.default_rng(0)
    keep, best = np.ones(len(xy), bool), -1
    for _ in range(tries if len(xy) >= 3 else 0):
        p = xy[rng.choice(len(xy), 3, replace=False)]
        A = np.c_[2 * p, np.ones(3)]
        try:
            cx, cy, c = np.linalg.solve(A, (p ** 2).sum(axis=1))
        except np.linalg.LinAlgError:
            continue
        r = np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0))
        if not 0 < r <= max_r:
            continue
        inl = np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - r) <= NEAR
        if inl.sum() > best:
            best, keep = int(inl.sum()), inl
    if keep.sum() < 3:
        keep = np.ones(len(xy), bool)
    for _ in range(rounds):
        A = np.c_[2 * xy[keep], np.ones(keep.sum())]
        b = (xy[keep] ** 2).sum(axis=1)
        (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
        r = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0)))
        resid = np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - r)
        keep = resid <= NEAR
        if keep.sum() < 3:
            break
    return np.array([cx, cy]), r, keep


def basis(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def fit_revolution(points: np.ndarray, normals: np.ndarray, iterations: int = 4) -> dict:
    """Axis and radius profile. Starts vertical: every revolution object of the roster rests upright
    (a partial view skews the principal axes, so symmetry.axis would not do). The circle fits use the
    side wall only (normals across the axis): a cap's points fill the circle and pull its centre. The
    line through the slices' centres may tilt the axis by at most MAX_TILT_DEG; beyond that it stays
    vertical (ep41's white pot, 11 cm tall, seen from one side with its 35 cm handle: the line tilted
    30 degrees and the revolved mesh came out 39 x 35 x 34 cm)."""
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
        a_new /= np.linalg.norm(a_new)
        tilt = float(np.degrees(np.arccos(np.clip(a_new[2], -1, 1))))
        capped = tilt > MAX_TILT_DEG
        c, a = m, (np.array([0.0, 0.0, 1.0]) if capped else a_new)
    radii = [x["r"] for x in slices if x["inliers"] >= 20]
    return {"axis": a, "point": c, "slices": len(slices), "fitted_tilt_deg": tilt if centres else None,
            "tilt_capped": bool(centres) and capped,
            "wall_radius_m": float(np.percentile(radii, 90)) if radii else None}


def generatrix(points: np.ndarray, normals: np.ndarray, fit: dict, azimuth_share: float = 0.5,
               max_radius: float | None = None) -> dict:
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
        if max_radius is not None and r[idx].mean() > max_radius:  # a handle or spout: kept as observed
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
def watertight(mesh: o3d.geometry.TriangleMesh) -> bool:
    """Closed 2-manifold, as trimesh counts it (the spec's S3a bar); Open3D's is_watertight also fails
    any self-intersection, which a voxel solid of a thin or smeared cloud can have and the simulator does
    not mind."""
    import trimesh
    return bool(len(mesh.triangles)) and trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                                        process=False).is_watertight


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
    # one thread: Open3D's OpenMP next to another library's (sklearn, via the scorer) crashed in
    # Poisson on macOS, and a single thread is deterministic
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=POISSON_DEPTH, n_threads=1)
    labels, counts, _ = mesh.cluster_connected_triangles()  # Poisson can leave small closed bubbles
    keep = int(np.argmax(np.asarray(counts)))
    vertex_keep = np.zeros(len(mesh.vertices), bool)
    vertex_keep[np.asarray(mesh.triangles)[np.asarray(labels) == keep].ravel()] = True
    mesh.remove_triangles_by_mask(np.asarray(labels) != keep)
    density = np.asarray(density)[vertex_keep]
    mesh.remove_unreferenced_vertices()
    trimmed = o3d.geometry.TriangleMesh(mesh)
    trimmed.remove_vertices_by_mask(np.asarray(density) < np.quantile(np.asarray(density), 0.01))
    mesh = trimmed if watertight(trimmed) else mesh
    if len(mesh.triangles) > MAX_FACES:  # may open it: close_surface then takes the voxel solid
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=MAX_FACES)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


SOLID_SHARE = 0.5  # default: Poisson's volume over the voxel solid's below this is hollow
ENCLOSED_SHARE = 0.2  # default: an interior this large (vs the dilated shell) makes a closed object


def enclosed(occ: np.ndarray) -> np.ndarray:
    """Voxels with occupied voxels on all six axis directions: the inside of a closed shell, even one
    with holes (a voxel in line with a hole is missed, the rest are not); never the open side of a
    sheet, cup, pot or rack."""
    inside = ~occ
    for axis in range(3):
        before = np.maximum.accumulate(occ, axis=axis)
        after = np.flip(np.maximum.accumulate(np.flip(occ, axis), axis=axis), axis)
        inside &= before & after
    return inside


def voxel_solid(points: np.ndarray, close: int = 2) -> o3d.geometry.TriangleMesh:
    """A watertight surface around the points: occupancy, morphological closing, interior filled, the
    field smoothed (Gaussian, 0.75 voxel), marching cubes at 0.5. Filling: the occupancy dilated by
    `close`, its holes and the voxels it encloses along all six axis directions (`enclosed`) filled,
    then eroded back by `close`, when that interior is >= ENCLOSED_SHARE of the dilated shell (a closed
    object); else (open: a sheet, a cup, a rack) the plain closing, then fill_holes. A closing seals
    gaps in solids, not in a sparse surface (its erosion thins the dilated sheet back, holes and all):
    on a sparse cloud fill_holes then filled nothing and marching cubes made a hollow solid (a synthetic
    box enclosed 13 % of its hull). Smoothing the field, not the mesh:
    a smoothed mesh folds thin parts into themselves. Not decimated (quadric decimation opens it):
    the voxel grows from 4 mm until the mesh has <= MAX_FACES faces."""
    from scipy import ndimage
    from skimage.measure import marching_cubes
    voxel = S.VOXEL
    for _ in range(6):
        lo = points.min(axis=0) - (close + 2) * voxel
        idx = np.floor((points - lo) / voxel).astype(int)
        raw = np.zeros(idx.max(axis=0) + close + 3, bool)
        raw[tuple(idx.T)] = True
        grown = ndimage.binary_dilation(raw, iterations=close)
        interior = (ndimage.binary_fill_holes(grown) | enclosed(grown)) & ~grown
        if interior.sum() >= ENCLOSED_SHARE * grown.sum():  # a closed object: fill, then erode back
            occ = ndimage.binary_erosion(grown | interior, iterations=close) | raw
        else:  # open (a sheet, a cup, a rack): the plain closing
            occ = ndimage.binary_fill_holes(ndimage.binary_closing(raw, iterations=close))
        field = ndimage.gaussian_filter(np.pad(occ, 2).astype(np.float32), 0.75)
        verts, faces, _, _ = marching_cubes(field, level=0.5)
        if len(faces) <= MAX_FACES:
            break
        voxel *= 1.05 * np.sqrt(len(faces) / MAX_FACES)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector((verts - 2) * voxel + lo),
                                     o3d.utility.Vector3iVector(faces[:, ::-1].astype(np.int32)))
    mesh.compute_vertex_normals()
    return mesh


def volume(mesh: o3d.geometry.TriangleMesh) -> float:
    import trimesh
    return float(abs(trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False).volume))


def close_surface(points: np.ndarray, normals: np.ndarray) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Poisson, unless it leaks, balloons (world-axis extents vs the cloud outside 0.8-1.25) or is
    hollow (it encloses less than SOLID_SHARE of the voxel solid's volume: Poisson can wrap the
    partials' surface twice, a closed shell a few mm thick, watertight and of the right size but a
    fraction of the mass; a synthetic box's came out at 0.13): then the voxel solid. Which one and why
    go into canonical.json."""
    mesh = poisson(points, normals)
    ratio = np.ptp(np.asarray(mesh.vertices), axis=0) / np.ptp(points, axis=0)
    solid = voxel_solid(points)
    share = volume(mesh) / max(volume(solid), 1e-12) if watertight(mesh) else 0.0
    why = [w for w, bad in (("not watertight", not watertight(mesh)),
                            (f"ballooned (extents x{ratio.round(2).tolist()})", not np.all((ratio >= 0.8) & (ratio <= 1.25))),
                            (f"hollow (volume x{share:.2f} of the voxel solid's)", watertight(mesh) and share < SOLID_SHARE))
           if bad]
    if not why:
        return mesh, {"closing": "screened Poisson", "volume_vs_voxel_solid": share}
    return solid, {"closing": "voxel solid", "poisson": " and ".join(why)}


def build(name: str, partials: list[o3d.geometry.TriangleMesh]) -> tuple[o3d.geometry.TriangleMesh, dict]:
    ref, moves = register(partials)
    fused = fused_cloud(partials, moves)
    pts, nrm = np.asarray(fused.points), np.asarray(fused.normals)
    report = {"reference": ref, "moves": [{k: v for k, v in m.items()} for m in moves], "fused_points": len(pts)}
    kind = "revolution" if name in REVOLUTION else "mirror" if name in MIRROR else "none"
    extra = [np.zeros((0, 3))], [np.zeros((0, 3))]
    if kind == "revolution":
        fit = fit_revolution(pts, nrm)
        cap = fit["wall_radius_m"] * WALL_MARGIN if fit.get("wall_radius_m") else None
        gen = generatrix(pts, nrm, fit, max_radius=cap)
        rp, rn = revolve(fit, gen)
        extra = [rp], [rn]
        report["symmetry"] = {"kind": kind, "axis": fit["axis"], "point": fit["point"], "slices": fit["slices"],
                              "fitted_tilt_deg": fit.get("fitted_tilt_deg"), "tilt_capped": fit.get("tilt_capped"),
                              "wall_radius_m": fit.get("wall_radius_m"), "profile_radius_cap_m": cap,
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
                  watertight=watertight(mesh), faces=len(mesh.triangles),
                  extents_cm=(np.ptp(world, axis=0) * 100).round(1),
                  # bar against the completed cloud (Poisson must not balloon); against the fused partial
                  # completion legitimately doubles an unseen side, so that ratio is information only
                  extents_vs_completed=ratio_cloud.round(3), extents_vs_fused=ratio_fused.round(3),
                  extents_ok=bool(np.all((ratio_cloud >= 0.8) & (ratio_cloud <= 1.25))),
                  flags=(["coverage < 0.6"] if coverage < COVERAGE_FLAG else [])
                  + (["not watertight"] if not watertight(mesh) else []),
                  fallbacks="not wired (SymmCompletion / AdaPoinTr / ForeHOI need torch and checkpoints)"
                  + ("; next step for this object" if coverage < COVERAGE_FLAG or not watertight(mesh) else ""))
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
    found = sorted(p for p in out.glob(f"episode_*/s3a2_fuse_scan/{name}/partial_scan*.ply")  # rest before, after
                   if episodes is None or int(p.parts[-4][len("episode_"):]) in episodes)
    def frames(p: Path) -> int:  # how many frames the fuse integrated into this partial
        f = p.parent / "fuse_report.json"
        if not f.is_file():
            return MIN_CLIP_FRAMES
        parts = json.loads(f.read_text())["partials"]
        return len(next((v["fused"] for v in parts.values() if v["file"] == p.name), []))
    enough = [p for p in found if frames(p) >= MIN_CLIP_FRAMES]
    skipped = [str(p.relative_to(out)) for p in found if p not in enough] if enough else []
    partials, used = [], []
    for p in enough or found:  # ep43's pot: touched at frame 7, a 1-frame rest-before partial of 45 vertices
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
    report = {"object": name, "clips": [str(p.relative_to(out)) for p in used], "clips_left_out": skipped,
              "min_clip_frames": MIN_CLIP_FRAMES, **report}
    (dest / "canonical.json").write_text(json.dumps(report, indent=1, default=to_json) + "\n")
    print(f"canonical {name}: {len(used)} clips, {report['faces']} faces, watertight {report['watertight']}, "
          f"coverage {report['coverage_before_completion']:.2f}, {report['symmetry']['kind']} -> {dest}", flush=True)
    return dest
