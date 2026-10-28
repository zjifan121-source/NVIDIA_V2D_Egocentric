"""Canonical mesh per object (V3_BUILD_SPEC.md §11 A2.2): every clip's partial scan of one object
(s3a2_fuse_scan), registered and completed by the object's symmetry; videos only (no scan, no GT).

    uv run --project path_b python path_b/run.py canonical --object white_pot --out <run> [--episodes ...]

1. Partials: <run>/episode_*/s3a2_fuse_scan/<object>/partial_scan.ply (metres, each clip's v3 world,
   the object at its frame-0 placement), each passing s3a2_fuse_scan's held-out depth check (trimmed
   RMS <= HELD_OUT_RMS_MM, its WARN bar; V4 §13.1 A1: table patches fused as the object measured
   0.77-1.25 m and, as the largest surface, became the reference) and fused from >= MIN_CLIP_FRAMES
   frames (all passing ones when none is). Reference: the largest surface; every other partial onto it by
   FPFH + RANSAC (best of 3 seeds), point-to-point ICP (two clips often share only the top, whose
   outline only point-to-point uses; it slides slowly: 300 iterations), then point-to-plane to finish;
   scale fixed; kept if fitness >= 0.3. Then (V4 §13.1 A1) a revolution object's clips are turned about
   the pool's axis so their asymmetric feature (the points farthest outside the wall: a pot's handle)
   points where the pool's does, and left out when it points more than FEATURE_AGREE_DEG away (a body
   of revolution registers at any turn: ep27's and ep42's pot clips came out 120 degrees from nine
   others, a pot with two handles); last, a clip whose points lie a median > POOL_DIST from the rest of
   the pool (the other kept clips and their symmetry completion) is left out (never the reference).
2. Fused: the registered partials' points, one 4 mm voxel grid (each partial is a TSDF average already).
3. Symmetry (roster, spec §11): revolution (white_pot, white_pot_lid, blue_cup, beige_cup,
   water_pitcher): the axis starts vertical (they rest upright) and is refined through robust circle
   fits per 5 mm height slice of the side wall; the fit is kept within MAX_TILT_DEG of the table normal,
   the table normal is used up to AXIS_REJECT_DEG, and beyond that the revolution is rejected and the
   cloud closed as seen (V4 §13.1 A1: a fit 86-89 degrees off, forced vertical, revolved a table patch
   into a 40-180 cm disc); completion revolves the profile (4 mm cells in the
   radius-height plane seen over at least half the azimuth seen at their height: walls and caps). Mirror (mini_dust_pan,
   wooden_piece_1/2): the vertical plane maximising the mirrored cloud's overlap (points within 5 mm),
   normals every 5 degrees tilted up to 10, offsets within 3 cm; completion mirrors the cloud. Others
   (plastic_dish_rack, mini_sweeper, wooden_spoon): mirror if its overlap >= 0.6, else the fused cloud.
   Observed points are kept, so handles and spouts stay where they were seen.
4. The table side is never seen: a flat cap of points at the table plane (z = 0 of the reference clip's
   world) under the object's lowest band closes it.
5. Screened Poisson (depth 8) on the oriented points; density-trimmed only if it stays watertight;
   decimated to <= 20,000 faces. Where the clips leave a side unseen, Poisson inflates it: if its mesh
   leaks or outgrows the cloud (extents outside 0.8-1.25x), the voxel solid instead (2 mm occupancy,
   closing, filled, smoothed field, marching cubes, stray pieces dropped, the voxel coarsened until
   <= 100,000 faces; not decimated, which opens it): watertight by construction, tight to what was seen.
Pools (V4 §13.1 A3): `--versus <split>` also pools that split's clips alone and keeps the pool whose mesh
explains more of the clips (each partial of the larger pool registered onto each mesh as s3a2_register
does; mean share), the other in canonical/<object>/alternative/.
Frame: the reference clip's world, shifted so the origin is the footprint centre on the table (z up).
Out: <run>/canonical/<object>/mesh.glb, mesh.obj, canonical.json (clips and registration, each clip's
gates and the share of it the mesh explains (within 5 mm, normals agreeing; s3a2_register's bar 0.8),
the clips left out before registration and why, symmetry fit, coverage before completion, watertight,
world-axis extents vs the completed cloud (bar 0.8-1.25) and vs the fused partial; coverage < 0.6 flagged).
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
AXIS_REJECT_DEG = 20.0  # spec §13.1 A1: a fitted axis further than this from the table normal is rejected
FEATURE_AGREE_DEG = 30.0  # spec §13.1 A1: a clip whose handle points further than this from the pool's is left out
FEATURE_MIN = 30  # default: points outside the wall (and >= 1 % of the clip) that make a feature
POOL_DIST = 0.003  # m, spec §13.1 A1: a clip a median further than this from the rest of the pool is left out
HELD_OUT_RMS_MM = 10.0  # s3a2_fuse_scan's WARN bar on a partial's surface vs held-out stereo depth
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
    30 degrees and the revolved mesh came out 39 x 35 x 34 cm). A fit beyond AXIS_REJECT_DEG, or no
    fit (no side wall about the table normal), is `rejected`: the cloud is not upright (V4 §13.1 A1)."""
    a, c = np.array([0.0, 0.0, 1.0]), points.mean(axis=0)
    all_points = points
    slices, tilt, capped = [], None, False
    for _ in range(iterations):
        points = all_points[np.abs(normals @ a) < 0.5]
        if len(points) < 20:  # no side wall about this axis (a cup lying over, seen from above)
            break
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
    return {"axis": a, "point": c, "slices": len(slices), "fitted_tilt_deg": tilt,
            "tilt_capped": tilt is not None and capped, "rejected": tilt is None or tilt > AXIS_REJECT_DEG,
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


SEATED_ABOVE = 0.03  # default: an object whose lowest seen point is this far above the table rests on another
# one (the lid on the pot's rim: ~11 cm); below it, an object on the table whose lowest centimetre went
# unseen (mask erosion, occlusion) is still closed down to the table


def bottom_cap(points: np.ndarray, band: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Points on the table plane under the object's lowest band (the side the camera never sees); at
    the lowest band itself when the object rests on another one (V4 §13.1 A5: ep0's lid sits on the
    pot, and a cap on the table closed it into a 12.7 cm column; the scan's lid is 4.7 cm tall)."""
    from scipy.spatial import ConvexHull, Delaunay
    low = points[points[:, 2] < points[:, 2].min() + band][:, :2]
    if len(low) < 10:
        return np.zeros((0, 3)), np.zeros((0, 3))
    hull = Delaunay(low[ConvexHull(low).vertices])
    lo, hi = low.min(axis=0), low.max(axis=0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], S.VOXEL), np.arange(lo[1], hi[1], S.VOXEL))
    grid = np.c_[gx.ravel(), gy.ravel()]
    grid = grid[hull.find_simplex(grid) >= 0]
    bottom = float(points[:, 2].min())
    z = bottom if bottom > SEATED_ABOVE else min(0.0, bottom)
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
SOLID_VOXEL = 0.002  # default: the voxel solid's grid; at 4 mm the pot's walls came out ~1 cm thick, the clips
# a median 3.9 mm off it (explain-share 0.38 -> 0.66 at 2 mm, CD-O 0.59 -> 0.41 cm on the public pot)
SOLID_SIGMA = 0.5  # default: smoothing (voxels)
SOLID_STEPS = ((1, 1), (2, 2), (2, 1))  # default: (dilate, erode) iterations of an open object's occupancy with the
# 3x3x3 cube, thinnest first; the first the cloud cannot see through (see_through) is kept. The cloud lies on
# fused_cloud's 4 mm grid: the 6-neighbour closing of 1 (until 2026-10-27) left ep40's dust pan a net of 816 pieces
# that 27 % of its mask's rays passed through on static frames (CD-O and explain-share, measured on points, did not
# see it), and no 6-neighbour closing seals a thin sheet (its erosion reopens the gaps across it). The cube's step
# reaches the grid's diagonals: (1, 1) seals the dust pan, the pot and a synthetic 1 mm sheet
SEE_THROUGH_BAR = 0.05  # default: rays across the cloud's surface that may pass through the mesh (ep40's dust pan:
# 0.33 -> 27 % of its static mask rays missed, 0.14 -> 12 %, 0.10 -> 6 %, <= 0.034 -> none)
SEE_PAIR, SEE_REACH = 0.006, 0.015  # default: a ray through two points this close; no hit this far either side
CUBE = np.ones((3, 3, 3), bool)
FLOAT_AREA = 2e-4  # m^2, default: closed pieces smaller than this are dropped: stray voxels (<= 1.2 cm^2 on the pooled
# dust pan and pot), not a part seen apart from the body (a synthetic pot's handle: pieces of 13-17 cm^2, 1 % of it)
SOLID_MAX_FACES = 100_000  # default: the voxel solid is not decimated (fast_simplification opens it); the
# packer budgets the submitted copy itself


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


def voxel_solid(points: np.ndarray) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """A watertight surface around the points: occupancy, morphological closing, interior filled, the
    field smoothed, marching cubes at 0.5. Filling: the occupancy dilated by `close`, its holes and the
    voxels it encloses along all six axis directions (`enclosed`) filled, then eroded back by `close`,
    when that interior is >= ENCLOSED_SHARE of the dilated shell (a closed object); else (open: a sheet,
    a cup, a rack) the plain closing, then fill_holes. A closing seals
    gaps in solids, not in a sparse surface (its erosion thins the dilated sheet back, holes and all):
    on a sparse cloud fill_holes then filled nothing and marching cubes made a hollow solid (a synthetic
    box enclosed 13 % of its hull). Smoothing the field, not the mesh:
    a smoothed mesh folds thin parts into themselves. Not decimated (quadric decimation and
    fast_simplification open it): the voxel grows until the mesh fits the face cap. Closed or open is
    decided on the 4 mm grid (closing 2: seals a shell 6-8 mm sparse); a closed object keeps that solid
    (<= MAX_FACES), an open one is rebuilt on the SOLID_VOXEL grid (the first of SOLID_STEPS the cloud
    cannot see through, smoothing SOLID_SIGMA, <= SOLID_MAX_FACES), whose walls follow the clips (the pot's:
    1 cm thick at 4 mm). Either way, closed pieces smaller than FLOAT_AREA are dropped.
    Returns the mesh and how it was made (canonical.json)."""
    mesh, closed = grid_solid(points, S.VOXEL, 2, 0.75, MAX_FACES)
    if closed:
        return mesh, {"solid": "closed (4 mm grid, closing 2)"}
    tried, meshes = [], []
    for dilate, erode in SOLID_STEPS:
        meshes.append(grid_solid(points, SOLID_VOXEL, dilate, SOLID_SIGMA, SOLID_MAX_FACES, erode=erode)[0])
        tried.append({"dilate": dilate, "erode": erode, "see_through": see_through(meshes[-1], points)})
        if tried[-1]["see_through"] <= SEE_THROUGH_BAR:
            break
    # none passes (a rack's wires leave real gaps: 0.07-0.18): the least see-through, thinnest on ties
    kept = min(range(len(tried)), key=lambda i: (tried[i]["see_through"] > SEE_THROUGH_BAR, tried[i]["see_through"]))
    return meshes[kept], {"solid": "open", "steps": tried, "kept": tried[kept]}


def see_through(mesh: o3d.geometry.TriangleMesh, points: np.ndarray, rays: int = 4000) -> float:
    """The share of rays across the cloud's surface that pass through the mesh: each through the midpoint
    of two neighbouring points (<= SEE_PAIR apart: the surface between samples, where a voxel net has its
    holes), along the local normal, no hit within SEE_REACH of it on either side. Cloud only (no camera):
    on ep40's dust pan it ranks the meshes as the mask rays they let through on static frames do."""
    if not len(mesh.triangles) or len(points) < 10:
        return 1.0
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * S.VOXEL, max_nn=30))
    normals = np.asarray(pc.normals)
    rng = np.random.default_rng(0)
    i = rng.choice(len(points), min(rays, len(points)), replace=False)
    dist, j = cKDTree(points).query(points[i], k=7)
    pick = rng.integers(1, j.shape[1], len(i))
    near = dist[np.arange(len(i)), pick] <= SEE_PAIR
    if not near.any():
        return 1.0
    mid = (points[i] + points[j[np.arange(len(i)), pick]])[near] / 2
    n = normals[i][near]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    hit = scene.cast_rays(o3d.core.Tensor(np.concatenate([mid - SEE_REACH * n, n], axis=1).astype(np.float32)))
    return float(np.mean(~(hit["t_hit"].numpy() <= 2 * SEE_REACH)))


def grid_solid(points: np.ndarray, voxel: float, close: int, sigma: float, max_faces: int,
               erode: int | None = None) -> tuple[o3d.geometry.TriangleMesh, bool]:
    """voxel_solid on one grid: the mesh and whether the occupancy enclosed an interior (closed). With
    `erode`, open whatever the occupancy encloses: dilated by `close`, eroded by `erode` (the 3x3x3 cube)."""
    from scipy import ndimage
    from skimage.measure import marching_cubes
    for _ in range(6):
        lo = points.min(axis=0) - (close + 2) * voxel
        idx = np.floor((points - lo) / voxel).astype(int)
        raw = np.zeros(idx.max(axis=0) + close + 3, bool)
        raw[tuple(idx.T)] = True
        grown = ndimage.binary_dilation(raw, iterations=close)
        interior = (ndimage.binary_fill_holes(grown) | enclosed(grown)) & ~grown
        closed = erode is None and bool(interior.sum() >= ENCLOSED_SHARE * grown.sum())
        if closed:  # a closed object: fill, then erode back
            occ = ndimage.binary_erosion(grown | interior, iterations=close) | raw
        else:  # open (a sheet, a cup, a rack): dilated, eroded (a closing when equal), holes filled
            occ = ndimage.binary_fill_holes(ndimage.binary_erosion(grown, iterations=close) if erode is None else
                                            ndimage.binary_erosion(ndimage.binary_dilation(raw, CUBE, iterations=close),
                                                                   CUBE, iterations=erode))
        field = ndimage.gaussian_filter(np.pad(occ, 2).astype(np.float32), sigma)
        # a sheet one voxel thick (an open object closed as seen) smooths below 0.5 everywhere: a thin
        # closed shell around it at half its peak
        verts, faces, _, _ = marching_cubes(field, level=min(0.5, 0.5 * float(field.max())))
        if len(faces) <= max_faces:
            break
        voxel *= 1.05 * np.sqrt(len(faces) / max_faces)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector((verts - 2) * voxel + lo),
                                     o3d.utility.Vector3iVector(faces[:, ::-1].astype(np.int32)))
    labels, _, areas = mesh.cluster_connected_triangles()  # each piece is closed: dropping one keeps it watertight
    labels, areas = np.asarray(labels), np.asarray(areas)
    if len(areas) > 1:
        mesh.remove_triangles_by_mask(areas[labels] < min(FLOAT_AREA, areas.max()))
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh, closed


def pieces(mesh: o3d.geometry.TriangleMesh) -> int:
    """Connected pieces (triangles sharing an edge): one for a rigid object."""
    return len(np.unique(np.asarray(mesh.cluster_connected_triangles()[0]))) if len(mesh.triangles) else 0


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
    solid, how = voxel_solid(points)
    share = volume(mesh) / max(volume(solid), 1e-12) if watertight(mesh) else 0.0
    why = [w for w, bad in (("not watertight", not watertight(mesh)),
                            (f"ballooned (extents x{ratio.round(2).tolist()})", not np.all((ratio >= 0.8) & (ratio <= 1.25))),
                            (f"hollow (volume x{share:.2f} of the voxel solid's)", watertight(mesh) and share < SOLID_SHARE))
           if bad]
    if not why:
        return mesh, {"closing": "screened Poisson", "volume_vs_voxel_solid": share}
    return solid, {"closing": "voxel solid", "poisson": " and ".join(why), "voxel_solid": how}


def complete(name: str, pts: np.ndarray, nrm: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """The roster symmetry's completion of a fused cloud: the points and normals it adds, and its report."""
    kind = "revolution" if name in REVOLUTION else "mirror" if name in MIRROR else "none"
    none = np.zeros((0, 3))
    if kind == "revolution":
        fit = fit_revolution(pts, nrm)
        if fit["rejected"]:
            return none, none, {"kind": kind, "applied": False, "axis": fit["axis"], "point": fit["point"],
                                "fitted_tilt_deg": fit["fitted_tilt_deg"], "wall_radius_m": fit["wall_radius_m"],
                                "rejected": "no side wall about the table normal to fit an axis: closed as seen"
                                if fit["fitted_tilt_deg"] is None else
                                f"axis {fit['fitted_tilt_deg']:.0f} deg from the table normal "
                                f"(> {AXIS_REJECT_DEG:.0f}): closed as seen"}
        cap = fit["wall_radius_m"] * WALL_MARGIN if fit.get("wall_radius_m") else None
        gen = generatrix(pts, nrm, fit, max_radius=cap)
        rp, rn = revolve(fit, gen)
        return rp, rn, {"kind": kind, "applied": True, "axis": fit["axis"], "point": fit["point"],
                        "slices": fit["slices"], "fitted_tilt_deg": fit.get("fitted_tilt_deg"),
                        "tilt_capped": fit.get("tilt_capped"), "wall_radius_m": fit.get("wall_radius_m"),
                        "profile_radius_cap_m": cap, "profile_cells": gen["cells"],
                        "points_on_profile": gen["kept_share"], "radial_residual_m": gen["residual_m"]}
    fit = fit_mirror(pts)
    use = kind == "mirror" or fit["overlap"] >= MIRROR_CONFIDENT
    rp, rn = (reflect(pts, fit["normal"], fit["d"]), nrm - 2 * (nrm @ fit["normal"])[:, None] * fit["normal"]) \
        if use else (none, none)
    return rp, rn, {"kind": "mirror" if use else "none", "roster": kind, "normal": fit["normal"], "d": fit["d"],
                    "overlap": fit["overlap"], "applied": bool(use)}


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def circular_mean(deg) -> float:
    return float(np.degrees(np.angle(np.exp(1j * np.radians(np.asarray(deg, float))).mean())))


def feature_azimuth(points: np.ndarray, fit: dict) -> float | None:
    """Azimuth (degrees about the fitted axis) of a body of revolution's asymmetric feature in one clip:
    the farthest fifth of its points outside WALL_MARGIN x the wall radius (a pot's handle); None when
    fewer than FEATURE_MIN of them (or 1 % of the clip) are there."""
    a, c = fit["axis"], fit["point"]
    u, v = basis(a)
    q = points - c
    rho = q - np.outer(q @ a, a)
    r = np.linalg.norm(rho, axis=1)
    far = r > WALL_MARGIN * fit["wall_radius_m"]
    if far.sum() < max(FEATURE_MIN, 0.01 * len(points)):
        return None
    tip = far & (r >= np.quantile(r[far], 0.8))
    return circular_mean(np.degrees(np.arctan2(rho[tip] @ v, rho[tip] @ u)))


def turn_about(fit: dict, deg: float) -> np.ndarray:
    """4x4: a turn by deg about the fitted axis."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.radians(deg) * np.asarray(fit["axis"])).as_matrix()
    T[:3, 3] = fit["point"] - T[:3, :3] @ fit["point"]
    return T


def align_features(clouds: list, moves: list[dict], ref: int, fit: dict) -> dict:
    """V4 §13.1 A1 for a body of revolution: the feature azimuth most clips agree with (within
    FEATURE_AGREE_DEG; ties: the reference's), each clip within it turned onto it about the axis, each
    clip beyond it left out. Needs two clips agreeing on it. Updates moves and clouds in place."""
    seen = {i: x for i, m in enumerate(moves) if m["kept"]
            and (x := feature_azimuth(np.asarray(clouds[i].points), fit)) is not None}
    if len(seen) < 2:
        return {"clips_with_feature": len(seen)}

    def members(x):
        return [i for i, y in seen.items() if abs(wrap(y - x)) <= FEATURE_AGREE_DEG]
    best = max(seen, key=lambda i: (len(members(seen[i])), i == ref))
    pivot = seen[best]
    if len(members(pivot)) < 2:  # no two clips agree: no majority to turn or drop a clip by (ep11/ep12's cup:
        for i, x in seen.items():  # rack wires in one clip's mask read as a handle)
            moves[i]["feature_azimuth_deg"] = x
        return {"clips_with_feature": len(seen), "consensus_azimuth_deg": None}
    consensus = wrap(pivot + circular_mean([wrap(seen[i] - pivot) for i in members(pivot)]))
    for i, x in seen.items():
        off = wrap(x - consensus)
        moves[i].update(feature_azimuth_deg=x, feature_off_deg=off)
        if abs(off) > FEATURE_AGREE_DEG:
            moves[i].update(kept=False, reason=f"feature (handle) {off:+.0f} deg from the pool's")
        else:
            moves[i]["T"] = turn_about(fit, -off) @ np.asarray(moves[i]["T"])
            clouds[i].transform(turn_about(fit, -off))
    return {"clips_with_feature": len(seen), "consensus_azimuth_deg": consensus}


def pool_distances(name: str, clouds: list, moves: list[dict], ref: int) -> None:
    """V4 §13.1 A1: each kept clip's median distance to the rest of the pool (the other kept clips'
    points and their symmetry completion); beyond POOL_DIST it is left out. Every distance is measured
    before any clip goes; only with three clips kept or more; never the reference."""
    kept = [i for i, m in enumerate(moves) if m["kept"]]
    if len(kept) < 3:
        return
    rng = np.random.default_rng(0)
    dist = {}
    for i in kept:
        rest = o3d.geometry.PointCloud()
        for j in kept:
            if j != i:
                rest += clouds[j]
        rest = rest.voxel_down_sample(S.VOXEL)
        p, n = np.asarray(rest.points), np.asarray(rest.normals)
        ep, _, _ = complete(name, p, n)
        mine = np.asarray(clouds[i].points)
        mine = mine[rng.choice(len(mine), min(5000, len(mine)), replace=False)]
        dist[i] = float(np.median(cKDTree(np.concatenate([p, ep])).query(mine)[0]))
    for i, d in dist.items():
        moves[i]["pool_distance_mm"] = d * 1000
        if d > POOL_DIST and i != ref:
            moves[i].update(kept=False, reason=f"median {d * 1000:.1f} mm from the rest of the pool")


def build(name: str, partials: list[o3d.geometry.TriangleMesh]) -> tuple[o3d.geometry.TriangleMesh, dict]:
    ref, moves = register(partials)
    for m in moves:
        if not m["kept"]:
            m["reason"] = f"registration fitness {m['fitness']:.2f} < {MIN_FITNESS}"
    clouds = [S.mesh_points(p, 20000).transform(np.asarray(m["T"])) for p, m in zip(partials, moves)]
    gates = {}
    if name in REVOLUTION:
        pre = fused_cloud(partials, moves)
        fit = fit_revolution(np.asarray(pre.points), np.asarray(pre.normals))
        if not fit["rejected"] and fit.get("wall_radius_m"):
            gates["feature"] = align_features(clouds, moves, ref, fit)
    pool_distances(name, clouds, moves, ref)
    fused = fused_cloud(partials, moves)
    pts, nrm = np.asarray(fused.points), np.asarray(fused.normals)
    report = {"reference": ref, "moves": moves, "gates": gates, "fused_points": len(pts)}
    ep, en, report["symmetry"] = complete(name, pts, nrm)
    cap_p, cap_n = bottom_cap(np.concatenate([pts, ep]))
    all_p = np.concatenate([pts, ep, cap_p])
    all_n = np.concatenate([nrm, en, cap_n])
    mesh, closing = close_surface(all_p, all_n)
    report.update(closing)
    # canonical frame: the footprint centre on the table, z up
    v = np.asarray(mesh.vertices)
    origin = np.array([*((v[:, :2].min(axis=0) + v[:, :2].max(axis=0)) / 2), 0.0])
    mesh.translate(-origin)
    placed = S.mesh_points(mesh, 20000).translate(origin)
    surface = np.asarray(placed.points)
    coverage = float(np.mean(cKDTree(pts).query(surface)[0] <= NEAR))
    for c, m in zip(clouds, moves):  # the share of each clip the mesh explains (s3a2_register's measure)
        m["explained"] = S.icp(c, placed, np.eye(4), max_dist=NEAR, iterations=0)["fitness"]
    explained = [m["explained"] for m in moves if m["kept"]]
    report["explained_min_kept"] = min(explained) if explained else None
    world = np.asarray(mesh.vertices) + origin
    ratio_cloud, ratio_fused = np.ptp(world, axis=0) / np.ptp(all_p, axis=0), np.ptp(world, axis=0) / np.ptp(pts, axis=0)
    report.update(world_to_canonical=-origin, coverage_before_completion=coverage, cap_points=len(cap_p),
                  watertight=watertight(mesh), faces=len(mesh.triangles), pieces=pieces(mesh),
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


def pool(name: str, out: Path, episodes: list[int] | None = None) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """The canonical mesh and its report from the partials of `episodes` (all when None)."""
    found = sorted(p for p in out.glob(f"episode_*/s3a2_fuse_scan/{name}/partial_scan*.ply")  # rest before, after
                   if episodes is None or int(p.parts[-4][len("episode_"):]) in episodes)
    def row(p: Path) -> dict:  # this partial's line in its fuse report (frames fused, held-out depth RMS)
        f = p.parent / "fuse_report.json"
        if not f.is_file():
            return {}
        return next((v for v in json.loads(f.read_text())["partials"].values() if v["file"] == p.name), {})

    def rms_mm(p: Path) -> float | None:
        r = (row(p).get("held_out_rms") or {}).get("trimmed_rms_m")
        return None if r is None else r * 1000

    checked = [p for p in found if (r := rms_mm(p)) is not None and r <= HELD_OUT_RMS_MM]
    skipped = [{"clip": str(p.relative_to(out)), "reason": "no held-out depth check" if rms_mm(p) is None
                else f"held-out depth RMS {rms_mm(p):.0f} mm > {HELD_OUT_RMS_MM:.0f}"} for p in found if p not in checked]
    if not checked:  # never an empty pool: the partial that comes closest
        known = [p for p in found if rms_mm(p) is not None]
        checked = [min(known, key=rms_mm)] if known else found
        skipped = [s for s in skipped if s["clip"] not in {str(p.relative_to(out)) for p in checked}]
    enough = [p for p in checked if len(row(p).get("fused", [])) >= MIN_CLIP_FRAMES or not row(p)]
    if enough:
        skipped += [{"clip": str(p.relative_to(out)), "reason": f"fused from {len(row(p).get('fused', []))} frames "
                     f"< {MIN_CLIP_FRAMES}"} for p in checked if p not in enough]
    partials, used = [], []
    for p in enough or checked:  # ep43's pot: touched at frame 7, a 1-frame rest-before partial of 45 vertices
        m = o3d.io.read_triangle_mesh(str(p))
        if len(m.triangles):
            m.compute_vertex_normals()
            partials.append(m)
            used.append(p)
    if not partials:
        raise SystemExit(f"no partial scan of {name} under {out}/episode_*/s3a2_fuse_scan/")
    mesh, report = build(name, partials)
    return mesh, {"object": name, "clips": [str(p.relative_to(out)) for p in used], "clips_left_out": skipped,
                  "min_clip_frames": MIN_CLIP_FRAMES, "held_out_rms_bar_mm": HELD_OUT_RMS_MM, **report}


def explains(mesh: o3d.geometry.TriangleMesh, report: dict, out: Path, clips: list[str]) -> list[float]:
    """Each clip's partial registered onto the canonical mesh as s3a2_register registers it (from this
    pool's move when it kept the clip, else global starts); the share of the partial the mesh explains."""
    from .stages.s3a2_register import canonical_move, register
    placed = o3d.geometry.TriangleMesh(mesh)
    placed.compute_vertex_normals()
    shares = []
    for c in clips:
        partial = o3d.io.read_triangle_mesh(str(out / c))
        partial.compute_vertex_normals()
        shares.append(register(placed, partial, canonical_move(report, c))["partial_explained"])
    return shares


def write(mesh: o3d.geometry.TriangleMesh, report: dict, dest: Path) -> None:
    import trimesh
    dest.mkdir(parents=True, exist_ok=True)
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
    tm.export(dest / "mesh.obj")
    tm.export(dest / "mesh.glb")
    (dest / "canonical.json").write_text(json.dumps(report, indent=1, default=to_json) + "\n")


def main(name: str, out: Path, episodes: list[int] | None = None, versus: list[int] | None = None,
         labels: tuple[str, str] = ("pool", "versus")) -> Path:
    """Open3D on one thread, as in pathb/invert.solve: its multithreaded ICP sums in a different order each run."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1, user_api="openmp"):
        return _main(name, out, episodes, versus, labels)


def _main(name: str, out: Path, episodes: list[int] | None, versus: list[int] | None,
          labels: tuple[str, str]) -> Path:
    import shutil
    mesh, report = pool(name, out, episodes)
    dest = out / "canonical" / name
    shutil.rmtree(dest / "alternative", ignore_errors=True)  # an earlier comparison's loser
    if versus is not None:  # V4 §13.1 A3: keep the pool that explains more of the larger pool's clips
        try:
            alt_mesh, alt_report = pool(name, out, versus)
        except SystemExit:  # no clip of the object in that split (wooden_spoon has no public clip)
            alt_mesh = None
        if alt_mesh is not None:
            clips = report["clips"] if len(report["clips"]) >= len(alt_report["clips"]) else alt_report["clips"]
            mine, theirs = explains(mesh, report, out, clips), explains(alt_mesh, alt_report, out, clips)
            choice = {labels[0]: {"clips": len(report["clips"]), "mean_explained": float(np.mean(mine))},
                      labels[1]: {"clips": len(alt_report["clips"]), "mean_explained": float(np.mean(theirs))},
                      "judged_on": len(clips)}
            if np.mean(theirs) > np.mean(mine):
                mesh, report, alt_mesh, alt_report = alt_mesh, alt_report, mesh, report
                choice["kept"] = labels[1]
            else:
                choice["kept"] = labels[0]
            report["pool_choice"], alt_report["pool_choice"] = choice, choice
            write(alt_mesh, alt_report, dest / "alternative")
        else:
            report["pool_choice"] = {"kept": labels[0], "note": f"no {labels[1]} clip"}
    write(mesh, report, dest)
    skipped = report["clips_left_out"]
    kept = sum(m["kept"] for m in report["moves"])
    if "pool_choice" in report:
        print(f"canonical {name}: pool choice {report['pool_choice']}", flush=True)
    print(f"canonical {name}: {kept} of {len(report['clips'])} clips kept ({len(skipped)} left out before), {report['faces']} "
          f"faces, watertight {report['watertight']}, extents {report['extents_cm'].tolist()} cm, explained >= "
          f"{report['explained_min_kept']:.2f}, coverage {report['coverage_before_completion']:.2f}, "
          f"{report['symmetry']['kind']}{'' if report['symmetry']['applied'] else ' (not applied)'} -> {dest}", flush=True)
    return dest
