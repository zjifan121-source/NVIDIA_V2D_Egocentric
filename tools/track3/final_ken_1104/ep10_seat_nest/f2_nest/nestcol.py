#!/usr/bin/env python3
"""nestcol.py -- ep10 fix F2 (plan: C:/v2d_demo/diag/ep10_fix/ep10_diagnosis_plan.json, plan.fixes[1]).

Collision-only nest proxy, GT-free (inputs = the scene's processed *_visual.stl + the scene parquet + its support USD):
  beige (inner cup)  : beige_col.stl = visual mesh minus the 4 Vicon marker spheres (cavity kept); CoACD as usual.
  blue  (container)  : blue_col_000..191.stl = explicit convex wall slabs on the OUTER surface, 8 height bands x 24
                       azimuths, t = 1.5 mm, plus a monotone rim flare that keeps >= 1 mm from the beige CoACD hulls in
                       the coaxial end-nest pose at every twist; blue_floor.stl = 3 mm convex slab whose top sits at the
                       median end-rest nest depth (f124-160 of the scene reference) minus 2 mm.
Visual meshes, score_input meshes and the parquet are never written.  Geometry helpers are reused from
~/v2d_ops/kenc/diag/ep10_plan/nestplan.py (axis_frame, radial profiles, interp_prof, slab construction, surf_pts,
max_depth, pen, coax, nest), shell.py and flare.py.

Subcommands (all take --scene SCENE_DIR --work WORK_DIR):
  prep    marker-free beige -> WORK/beige_col.stl, WORK/prep.json, WORK/orch/ (test URDFs for the orchard CoACD run)
  slabs   needs WORK/hulls_beige.npz (orchard) -> flare + WORK/blue_col_*.stl, blue_floor.stl, WORK/slabs.json
  check   needs WORK/hulls_blue.npz (orchard) -> gt_free_check a, b, d -> WORK/checks.json
  settle  MuJoCo no-hands settle (gt_free_check c) -> WORK/settle.json      (needs the mujoco module)
  apply   write the collision files into SCENE, back up + edit both *_rigid.urdf, NESTCOL.json
--emulate_f1 SK10A2_JSON (dry runs on bob1 only): applies the planner's A2 seat simulation (nestplan.py) to the blue
z and the left-hand joints on f45-125 so the dry run sees an F1-like reference.  Never used on k10_ep10_nest.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as Rot

NAMES = ("blue_cup", "beige_cup")          # roster ep10: slot 0 = blue_cup, slot 1 = beige_cup
HB, NPH = 0.005, 24                         # profile bins: 0.5 cm bands x 24 azimuths (as nestplan.py)
NB_SLAB, NPHI_SLAB = 8, 24                  # 8 bands x 24 azimuths = 192 wall slabs
T_SLAB = 0.0015                             # slab thickness 1.5 mm
MARKER_DET = 0.006                          # marker vertex: > 6 mm beyond the per-height median radius
MARKER_GROW = 0.001                         # grow the marker region while > 1 mm beyond the marker-free wall
CLEAR = 0.001                               # flare clearance 1 mm
FLOOR_T = 0.003                             # floor slab 3 mm
FLOOR_BELOW = 0.002                         # floor top = median end nest depth - 2 mm
ZT = 1.0                                    # levelled table plane (blue support top), only for --emulate_f1
COACD_DEFAULT = dict(threshold=0.05, max_convex_hull=-1, preprocess_mode="auto")


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- scene I/O
def scene_files(scene):
    proc = os.path.join(scene, "assets", "human_motion_data", "ego_recon", "processed")
    pqs = glob.glob(os.path.join(proc, "sequence_id=*", "robot_name=*", "*.parquet"))
    if len(pqs) != 1:
        raise SystemExit(f"expected exactly one parquet under {proc}, got {pqs}")
    out = dict(scene=scene, proc=proc, parquet=pqs[0], obj={})
    for nm in NAMES:
        u = [x for x in glob.glob(os.path.join(proc, "*_rigid.urdf")) if x.endswith(f"_{nm}_rigid.urdf")]
        if len(u) != 1:
            raise SystemExit(f"{nm}: expected one *_rigid.urdf, got {u}")
        prefix = os.path.basename(u[0])[: -len("_rigid.urdf")]
        out["obj"][nm] = dict(urdf=u[0], prefix=prefix, stl=os.path.join(proc, prefix + "_visual.stl"),
                              objfile=os.path.join(proc, prefix + ".obj"))
    usd = glob.glob(os.path.join(scene, "assets", "human_motion_data", "ego_recon", "reconstructed_stage", "*.usda"))
    out["usd"] = usd[0] if usd else None
    return out


def load_ref(parquet, emulate_f1=None):
    t = pq.read_table(parquet).to_pylist()[0]
    names = list(t["object_body_names"])
    order = [next(i for i, n in enumerate(names) if n.endswith(nm)) for nm in NAMES]
    P = np.array(t["object_body_position"], float)[:, order]
    Q = np.array(t["object_body_wxyz"], float)[:, order]
    N = len(P)
    R = Rot.from_quat(Q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(N, 2, 3, 3)
    ref = dict(t=t, names=names, order=order, P=P, Q=Q, R=R, N=N,
               JR=np.array(t["mano_right_joints"], float), JL=np.array(t["mano_left_joints"], float),
               urdf_paths=list(t["object_urdf_paths"]), mesh_paths=list(t["object_mesh_paths"]), emulated=None)
    if emulate_f1:
        ref["emulated"] = emulate_f1
    return ref


def apply_emulated_f1(ref, sk10a2_json, Vb):
    """Dry-run only: the planner's A2 seat simulation from nestplan.py (gate on the depth-observed visible top)."""
    d2 = json.load(open(sk10a2_json))
    obs = d2["observed"]
    N = ref["N"]; P = ref["P"]; R = ref["R"]
    p95 = np.array([obs[str(f)]["blue_cup"]["p95"] if obs[str(f)].get("blue_cup") else np.nan for f in range(N)])
    med5 = np.array([np.nanmedian(p95[max(0, f - 2):f + 3]) for f in range(N)])
    rest = float(np.nanmedian(p95[:36]))
    gate = np.abs(med5 - rest) <= 1.0
    low = np.array([(Vb @ R[f, 0].T + P[f, 0])[:, 2].min() for f in range(N)])
    dz_raw = np.where(gate & (low > ZT + 0.003), ZT - low, 0.0)
    dz = np.convolve(np.pad(dz_raw, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    dz[np.abs(dz) < 0.002] = 0.0
    ref["P"] = P.copy(); ref["P"][:, 0, 2] += dz
    JL = ref["JL"].copy()
    for f in range(45, min(126, N)):
        JL[f, :, 2] += dz[f]
    ref["JL"] = JL
    ref["emulated_dz_cm"] = {int(f): round(float(dz[f]) * 100, 2) for f in range(0, N, 5)}


# ----------------------------------------------------------------------------------------------- geometry (nestplan)
def axis_frame(V, R0):
    """nestplan.py axis_frame: PCA axis refined by circle fits of horizontal slices; bottom = lowest point on axis."""
    c = V - V.mean(0)
    w, U = np.linalg.eigh(c.T @ c)
    a = U[:, np.argmax(w)]
    if (R0 @ a)[2] < 0:
        a = -a
    for _ in range(3):
        e1 = np.cross(a, [1, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 1e-3 else np.cross(a, [0, 1, 0])
        e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
        h = V @ a; lo, hi = h.min(), h.max(); cents = []
        for s0 in np.linspace(lo + 0.1 * (hi - lo), hi - 0.15 * (hi - lo), 14):
            sel = np.abs(h - s0) < 0.004
            if sel.sum() < 50:
                continue
            x, y = V[sel] @ e1, V[sel] @ e2
            A = np.c_[2 * x, 2 * y, np.ones_like(x)]
            sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
            cents.append(sol[0] * e1 + sol[1] * e2 + s0 * a)
        cents = np.array(cents); cc = cents.mean(0)
        w2, U2 = np.linalg.eigh((cents - cc).T @ (cents - cc))
        an = U2[:, np.argmax(w2)]; a = an if an @ a > 0 else -an
    h = (V - cc) @ a
    bottom = cc + a * h.min()
    e1 = np.cross(a, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    return dict(a=a, b=bottom, H=float(h.max() - h.min()), e1=e1, e2=e2)


def cyl(X, A):
    d = np.asarray(X, float) - A["b"]
    h = d @ A["a"]; x, y = d @ A["e1"], d @ A["e2"]
    return h, np.hypot(x, y), np.arctan2(y, x)


def fill_rows(arr):
    arr = arr.copy()
    for j in range(arr.shape[1]):
        col = arr[:, j]; ok = ~np.isnan(col)
        if ok.any():
            idx = np.arange(len(col)); arr[:, j] = np.interp(idx, idx[ok], col[ok])
    return arr


def profiles(mesh, A, seed):
    """nestplan.py radial profiles: per (0.5 cm band, 24 azimuth bins) min radius and p99 radius of 300k samples."""
    S = trimesh.sample.sample_surface(mesh, 300000, seed=seed)[0]
    h, r, ph = cyl(S, A)
    nh = int(np.ceil(A["H"] / HB)) + 1
    ih = np.clip((h / HB).astype(int), 0, nh - 1); ip = ((ph + np.pi) / (2 * np.pi) * NPH).astype(int) % NPH
    rmin = np.full((nh, NPH), np.nan); rmax = np.full((nh, NPH), np.nan)
    for i in range(nh):
        for j in range(NPH):
            s = (ih == i) & (ip == j)
            if s.sum() >= 3:
                rmin[i, j] = r[s].min(); rmax[i, j] = np.percentile(r[s], 99)
    return fill_rows(rmin), fill_rows(rmax)


def interp_prof(arr, h, ph):
    nh = arr.shape[0]
    h = np.asarray(h, float); ph = np.asarray(ph, float)
    i = np.clip(h / HB, 0, nh - 1 - 1e-6); j = (((ph + np.pi) / (2 * np.pi) * NPH).astype(int)) % NPH
    i0 = np.floor(i).astype(int); f = i - i0
    a0 = arr[i0, j]; a1 = arr[np.minimum(i0 + 1, nh - 1), j]
    out = a0 * (1 - f) + a1 * f
    return np.where(np.isnan(out), np.nanmedian(arr), out)


def hull_eq(v):
    return ConvexHull(v).equations


def surf_pts(hs, n_per=40, seed=0, n_single=4000):
    """nestplan.py surf_pts: hull vertices + area-weighted samples of every hull's surface."""
    rng = np.random.default_rng(seed); out = []
    for v, eq in hs:
        ch = ConvexHull(v); tri = v[ch.simplices]
        a = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        n = n_per if len(hs) > 1 else n_single
        idx = rng.choice(len(tri), n, p=a / a.sum()); u, w = rng.random(n), rng.random(n)
        m = u + w > 1; u[m], w[m] = 1 - u[m], 1 - w[m]
        out += [v, tri[idx, 0] + u[:, None] * (tri[idx, 1] - tri[idx, 0]) + w[:, None] * (tri[idx, 2] - tri[idx, 0])]
    return np.concatenate(out)


def max_depth(pts, hs):
    """nestplan.py max_depth: deepest point-in-convex-hull depth (distance to the nearest face)."""
    best = 0.0
    for v, eq in hs:
        lo, hi = v.min(0) - 0.002, v.max(0) + 0.002
        sel = np.all((pts > lo) & (pts < hi), 1)
        if not sel.any():
            continue
        s_ = pts[sel] @ eq[:, :3].T + eq[:, 3]
        best = max(best, float((-s_.max(1)).max()))
    return best


def count_inside(pts, hs, tol=0.0):
    """number of points inside any convex hull (signed distance < -tol)."""
    inside = np.zeros(len(pts), bool)
    for v, eq in hs:
        lo, hi = v.min(0), v.max(0)
        sel = np.all((pts > lo) & (pts < hi), 1)
        if not sel.any():
            continue
        s_ = pts[sel] @ eq[:, :3].T + eq[:, 3]
        inside[np.nonzero(sel)[0][s_.max(1) < -tol]] = True
    return inside


# ----------------------------------------------------------------------------------------------- marker-free beige
def boundary_loops(faces):
    E = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    es = set(map(tuple, E.tolist()))
    nxt = {}
    for a, b in E.tolist():
        if (b, a) not in es:
            nxt.setdefault(a, []).append(b)
    if any(len(v) != 1 for v in nxt.values()):
        raise RuntimeError("non-simple hole boundary (a vertex with >1 outgoing boundary edge)")
    loops, seen = [], set()
    for s in list(nxt):
        if s in seen:
            continue
        loop = [s]; seen.add(s); c = nxt[s][0]
        while c != s:
            if c in seen:
                raise RuntimeError("boundary chain does not close")
            loop.append(c); seen.add(c); c = nxt[c][0]
        loops.append(loop)
    return loops


def ear_clip(P2):
    """triangulate a simple CCW polygon (n,2); returns index triples (CCW) or None."""
    idx = list(range(len(P2)))
    tris = []

    def a2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    while len(idx) > 3:
        n = len(idx); best = None
        Pi = P2[idx]
        for k in range(n):
            i0, i1, i2 = (k - 1) % n, k, (k + 1) % n
            a, b, c = Pi[i0], Pi[i1], Pi[i2]
            ar = a2(a, b, c)
            if ar <= 1e-16:
                continue
            others = np.ones(n, bool); others[[i0, i1, i2]] = False
            p = Pi[others]
            if len(p):
                d1 = (b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0])
                d2 = (c[0] - b[0]) * (p[:, 1] - b[1]) - (c[1] - b[1]) * (p[:, 0] - b[0])
                d3 = (a[0] - c[0]) * (p[:, 1] - c[1]) - (a[1] - c[1]) * (p[:, 0] - c[0])
                if np.any((d1 >= 0) & (d2 >= 0) & (d3 >= 0)):
                    continue
            # prefer fat ears: ratio area / longest edge^2
            q = ar / max(np.sum((a - c) ** 2), 1e-18)
            if best is None or q > best[0]:
                best = (q, k)
        if best is None:
            return None
        k = best[1]
        tris.append((idx[(k - 1) % n], idx[k], idx[(k + 1) % n]))
        idx.pop(k)
    tris.append(tuple(idx))
    return tris


def seg_intersections(P2):
    """number of proper intersections between non-adjacent edges of a closed polygon (n,2)."""
    n = len(P2); A_ = P2; B_ = np.roll(P2, -1, 0)
    cnt = 0

    def orient(p, q, r_):
        return (q[..., 0] - p[..., 0]) * (r_[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r_[..., 0] - p[..., 0])
    for i in range(n):
        a, b = A_[i], B_[i]
        j = np.arange(i + 2, n)
        if i == 0:
            j = j[j != n - 1]
        if not len(j):
            continue
        c, d = A_[j], B_[j]
        o1 = orient(a[None], b[None], c); o2 = orient(a[None], b[None], d)
        o3 = orient(c, d, a[None]); o4 = orient(c, d, b[None])
        cnt += int(np.sum((o1 * o2 < 0) & (o3 * o4 < 0)))
    return cnt


def refine(tris, pts, boundary, levels):
    """red-green midpoint refinement of a chart triangulation; boundary edges are never split."""
    pts = [p for p in pts]
    for _ in range(levels):
        mid = {}

        def mp(a, b):
            key = (min(a, b), max(a, b))
            if key in boundary:
                return None
            if key not in mid:
                pts.append(0.5 * (pts[a] + pts[b])); mid[key] = len(pts) - 1
            return mid[key]
        new = []
        for t in tris:
            a, b, c = t
            ms = [mp(a, b), mp(b, c), mp(c, a)]
            ns = sum(m is not None for m in ms)
            if ns == 3:
                mab, mbc, mca = ms
                new += [(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]
            elif ns == 0:
                new.append(t)
            else:
                for rot in range(3):
                    tt = (t[rot], t[(rot + 1) % 3], t[(rot + 2) % 3]); mm = ms[rot:] + ms[:rot]
                    if ns == 2 and mm[0] is None:
                        a_, b_, c_ = tt; mbc, mca = mm[1], mm[2]
                        new += [(a_, b_, mbc), (a_, mbc, mca), (mca, mbc, c_)]; break
                    if ns == 1 and mm[1] is not None:
                        a_, b_, c_ = tt; m_ = mm[1]
                        new += [(a_, b_, m_), (a_, m_, c_)]; break
        tris = new
    return tris, np.array(pts)


def marker_free_beige(mesh, A, rmaxG, rminG, dil_list=(0.002, 0.003, 0.004, 0.006), levels=3):
    """Remove the Vicon marker spheres from the beige visual mesh.
    1) marker vertices: > 6 mm beyond the per-height median radius (the plan's rule); each sphere is grown over
       vertices that stay > 1 mm beyond the marker-free wall.
    2) cut: those faces plus every outer-side face (r > mid-wall) within DIL of the sphere footprint in the
       (arc length, height) chart, so the overhang under the sphere goes too; small islands and bow-ties dropped.
    3) each hole (a simple polygon in the chart) is ear-clipped, refined 3x (boundary edges kept) and its interior
       vertices are lifted onto the marker-free outer wall (per-height median radius).  The cavity (inner wall) and
       the rest of the surface are untouched; topology stays a closed 2-manifold."""
    from scipy.spatial import cKDTree
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.faces)
    h, r, ph = cyl(V, A)
    medG = np.nanmedian(rmaxG, 1)
    mk = rmaxG > medG[:, None] + MARKER_DET
    rmaxG_nm = np.where(mk, medG[:, None], rmaxG)
    ib = np.clip((h / HB).astype(int), 0, len(medG) - 1)
    core = r > medG[ib] + MARKER_DET
    rwall = interp_prof(rmaxG_nm, h, ph)
    rmid = 0.5 * (rwall + interp_prof(rminG, h, ph))
    grow_ok = r > rwall + MARKER_GROW
    nbr = [[] for _ in range(len(V))]
    for a, b in mesh.edges_unique.tolist():
        nbr[a].append(b); nbr[b].append(a)
    region = core.copy(); stack = list(np.nonzero(core)[0])
    while stack:
        v = stack.pop()
        for u in nbr[v]:
            if not region[u] and grow_ok[u]:
                region[u] = True; stack.append(u)
    ridx = np.nonzero(region)[0]
    lab = -np.ones(len(V), int); nm_ = 0
    for s in ridx:
        if lab[s] >= 0:
            continue
        st = [s]; lab[s] = nm_
        while st:
            v = st.pop()
            for u in nbr[v]:
                if region[u] and lab[u] < 0:
                    lab[u] = nm_; st.append(u)
        nm_ += 1
    Rref = float(np.median(medG[-6:]))
    hgrid = (np.arange(len(medG)) + 0.5) * HB

    def wrap(x):
        return np.angle(np.exp(1j * x))
    last_err = None
    for DIL in dil_list:
        try:
            cut = region.copy()
            for c in range(nm_):
                comp = lab == c
                phc = float(np.angle(np.mean(np.exp(1j * ph[comp]))))
                tree = cKDTree(np.c_[wrap(ph[comp] - phc) * Rref, h[comp]])
                cand = np.nonzero((r > rmid) & (np.abs(wrap(ph - phc)) < 0.7))[0]
                dd, _ = tree.query(np.c_[wrap(ph[cand] - phc) * Rref, h[cand]])
                cut[cand[dd < DIL]] = True
            drop = cut[F].any(1)
            n_island = 0; n_dilate = 0
            for _ in range(30):
                keepF = F[~drop]
                km = trimesh.Trimesh(V, keepF, process=False)
                labels = trimesh.graph.connected_component_labels(km.face_adjacency, node_count=len(keepF))
                big = np.bincount(labels).argmax()
                if (labels != big).any():
                    kidx = np.nonzero(~drop)[0]; drop[kidx[labels != big]] = True; n_island += int((labels != big).sum()); continue
                E = np.vstack([keepF[:, [0, 1]], keepF[:, [1, 2]], keepF[:, [2, 0]]])
                es = set(map(tuple, E.tolist()))
                outdeg = {}
                for a_, b_ in E.tolist():
                    if (b_, a_) not in es:
                        outdeg[a_] = outdeg.get(a_, 0) + 1
                bad = [v for v, c_ in outdeg.items() if c_ > 1]
                if not bad:
                    break
                badm = np.zeros(len(V), bool); badm[bad] = True
                drop |= badm[F].any(1); n_dilate += 1
            keepF = F[~drop]
            loops = boundary_loops(keepF)
            if len(loops) != nm_:
                raise RuntimeError(f"{len(loops)} hole loops for {nm_} markers")
            newV = [V]; nV = len(V); fill = []; loop_info = []
            for loop in loops:
                lv = np.array(loop[::-1])                  # fill runs against the kept boundary
                phc = float(np.angle(np.mean(np.exp(1j * ph[lv]))))
                P2 = np.c_[wrap(ph[lv] - phc) * Rref, h[lv]]
                area = 0.5 * np.sum(P2[:, 0] * np.roll(P2[:, 1], -1) - np.roll(P2[:, 0], -1) * P2[:, 1])
                if area <= 0:
                    raise RuntimeError("hole polygon is not CCW in the outward chart")
                nx = seg_intersections(P2)
                if nx:
                    raise RuntimeError(f"hole polygon self-intersects ({nx})")
                tri = ear_clip(P2)
                if tri is None:
                    raise RuntimeError("ear clipping failed")
                nb_ = len(lv)
                bnd = set((min(i, (i + 1) % nb_), max(i, (i + 1) % nb_)) for i in range(nb_))
                tri2, pts2 = refine(tri, list(P2), bnd, levels)
                extra = pts2[nb_:]
                eph = phc + extra[:, 0] / Rref; eh = extra[:, 1]
                er = np.interp(eh, hgrid, medG)
                E3 = A["b"][None] + eh[:, None] * A["a"][None] + er[:, None] * (np.cos(eph)[:, None] * A["e1"][None] + np.sin(eph)[:, None] * A["e2"][None])
                gidx = np.r_[lv, nV + np.arange(len(extra))]
                newV.append(E3); nV += len(extra)
                fill += [(gidx[a], gidx[b], gidx[c]) for a, b, c in tri2]
                loop_info.append(dict(n_boundary=int(nb_), az_deg=round(float(np.degrees(phc)), 1),
                                      h_cm=[round(float(P2[:, 1].min()) * 100, 2), round(float(P2[:, 1].max()) * 100, 2)],
                                      r_boundary_cm=[round(float(r[lv].min()) * 100, 2), round(float(r[lv].max()) * 100, 2)],
                                      fill_faces=len(tri2), interior_vertices=int(len(extra))))
            out = trimesh.Trimesh(np.vstack(newV), np.vstack([keepF, np.array(fill, int)]), process=False)
            out.remove_unreferenced_vertices()
            if not (out.is_watertight and out.is_winding_consistent):
                raise RuntimeError("result not watertight / winding-consistent")
            info = dict(method="cut sphere + overhang, chart ear-clip fill refined x%d, lifted to the per-height median outer radius" % levels,
                        dilation_mm=DIL * 1000, markers=nm_, core_vertices=int(core.sum()), region_vertices=int(region.sum()),
                        cut_vertices=int(cut.sum()), faces_removed=int(drop.sum()), islands_dropped=n_island, bowtie_dilations=n_dilate,
                        fill_faces=len(fill), holes=loop_info, marker_bins=int(mk.sum()),
                        region_h_cm=[round(float(h[region].min()) * 100, 2), round(float(h[region].max()) * 100, 2)],
                        region_rmax_cm=round(float(r[region].max()) * 100, 2),
                        watertight=bool(out.is_watertight), winding_consistent=bool(out.is_winding_consistent),
                        euler=int(out.euler_number), volume_cm3=round(float(out.volume) * 1e6, 3),
                        orig_volume_cm3=round(float(mesh.volume) * 1e6, 3), n_vertices=int(len(out.vertices)), n_faces=int(len(out.faces)),
                        failed_attempts=last_err)
            return out, info, rmaxG_nm
        except RuntimeError as e:
            last_err = (last_err or []) + [f"DIL {DIL * 1000:.0f} mm: {e}"]
    raise RuntimeError(f"marker removal failed: {last_err}")


def radius_by_band(pts, A, h0=0.0, h1=None, step=0.005):
    h, r, ph = cyl(pts, A)
    h1 = A["H"] if h1 is None else h1
    out = {}
    for lo in np.arange(h0, h1, step):
        s = (h >= lo) & (h < lo + step)
        if s.any():
            out[round(float(lo) * 100, 1)] = float(r[s].max())
    return out


# ----------------------------------------------------------------------------------------------- npz hull sets
def load_hullset(npz):
    """orchard output: piece names + per piece list of (vertices, faces)."""
    z = np.load(npz, allow_pickle=False)
    names = [str(x) for x in z["names"]]
    out = {}
    for i, nm in enumerate(names):
        n = int(z[f"p{i}_n"])
        out[nm] = [(z[f"p{i}_v{k}"].astype(float), z[f"p{i}_f{k}"].astype(int)) for k in range(n)]
    return out


def as_hs(hulls):
    return [(v, hull_eq(v)) for v, f in hulls]


def write_stl(path, pts):
    m = trimesh.convex.convex_hull(np.asarray(pts, float))
    m.export(path, file_type="stl")
    return m


def nest(A, G, Pb, Rb, Pg, Rg):
    aw = Rb @ A["a"]; bw = Rb @ A["b"] + Pb
    gbw = Rg @ G["b"] + Pg; gaw = Rg @ G["a"]
    d = (gbw - bw) @ aw
    rad = np.linalg.norm((gbw - bw) - d * aw)
    tilt = np.degrees(np.arccos(np.clip(aw @ gaw, -1, 1)))
    return d, rad, tilt


def coax(A, G, d, theta=0.0):
    v = np.cross(G["a"], A["a"]); s = np.linalg.norm(v); c = G["a"] @ A["a"]
    R0 = np.eye(3) if s < 1e-9 else Rot.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()
    Rt = Rot.from_rotvec(A["a"] * theta).as_matrix() @ R0
    return A["b"] + A["a"] * d - Rt @ G["b"], Rt


# ----------------------------------------------------------------------------------------------- common setup
def setup(args):
    sf = scene_files(args.scene)
    meshes = {nm: trimesh.load(sf["obj"][nm]["stl"], force="mesh") for nm in NAMES}   # process=True as newton loads
    ref = load_ref(sf["parquet"])
    if args.emulate_f1:
        apply_emulated_f1(ref, args.emulate_f1, np.asarray(meshes["blue_cup"].vertices))
    AX = {nm: axis_frame(np.asarray(meshes[nm].vertices), ref["R"][0, k]) for k, nm in enumerate(NAMES)}
    return sf, meshes, ref, AX


def prof_cache(args, meshes, AX):
    p = os.path.join(args.work, "profiles.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["rminB"], z["rmaxB"], z["rminG"], z["rmaxG"]
    rminB, rmaxB = profiles(meshes["blue_cup"], AX["blue_cup"], seed=11)
    rminG, rmaxG = profiles(meshes["beige_cup"], AX["beige_cup"], seed=12)
    np.savez(p, rminB=rminB, rmaxB=rmaxB, rminG=rminG, rmaxG=rmaxG)
    return rminB, rmaxB, rminG, rmaxG


def ax_json(A):
    return {k: (np.round(v, 7).tolist() if isinstance(v, np.ndarray) else v) for k, v in A.items()}


URDF_TEST = """<?xml version="1.0" ?>
<robot name="{name}">
   <link name="object">
      <inertial>
         <origin xyz="0 0 0" rpy="0 0 0"/>
         <mass value="0.3"/>
         <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
{collisions}
   </link>
</robot>
"""
COLL = """      <collision>
         <origin rpy="0 0 0" xyz="0 0 0"/>
         <geometry>
            <mesh filename="./{fn}" scale="1 1 1"/>
         </geometry>
      </collision>"""


def cmd_prep(args):
    os.makedirs(args.work, exist_ok=True); orch = os.path.join(args.work, "orch"); os.makedirs(orch, exist_ok=True)
    if args.reuse:
        sf = scene_files(args.scene)
        old = json.load(open(os.path.join(args.reuse, "prep.json")))
        now = {nm: md5(sf["obj"][nm]["stl"]) for nm in NAMES}
        if now != old["visual_stl_md5"]:
            raise SystemExit(f"--reuse refused: visual STL md5 {now} != {old['visual_stl_md5']}")
        for fn in ("beige_col.stl", "profiles.npz", "hulls_beige.npz"):
            src_ = os.path.join(args.reuse, fn)
            if os.path.exists(src_):
                shutil.copy2(src_, os.path.join(args.work, fn))
        shutil.copy2(os.path.join(args.work, "beige_col.stl"), os.path.join(orch, "beige_col.stl"))
        open(os.path.join(orch, "beige_test.urdf"), "w").write(URDF_TEST.format(name="beige_test", collisions=COLL.format(fn="beige_col.stl")))
        old.update(scene=args.scene, parquet=sf["parquet"], parquet_md5=md5(sf["parquet"]), emulate_f1=args.emulate_f1,
                   reused_from=os.path.abspath(args.reuse), beige_col_md5_check=md5(os.path.join(args.work, "beige_col.stl")))
        if old["beige_col_md5_check"] != old["beige_col_md5"]:
            raise SystemExit("reused beige_col.stl md5 mismatch")
        jdump(old, os.path.join(args.work, "prep.json"))
        print("prep: reused", args.reuse, "visual md5", now, "beige_col md5", old["beige_col_md5"])
        return
    sf, meshes, ref, AX = setup(args)
    rminB, rmaxB, rminG, rmaxG = prof_cache(args, meshes, AX)
    G = AX["beige_cup"]
    col, info, rmaxG_nm = marker_free_beige(meshes["beige_cup"], G, rmaxG, rminG)
    p = os.path.join(args.work, "beige_col.stl")
    col.export(p, file_type="stl")
    chk = trimesh.load(p, force="mesh")                    # reload exactly as newton/trimesh will
    info["reload"] = dict(watertight=bool(chk.is_watertight), n_vertices=int(len(chk.vertices)), n_faces=int(len(chk.faces)))
    # marker-free radius per 0.5 cm band (visual surface vs collision surface) above 10 cm
    Sv = trimesh.sample.sample_surface(meshes["beige_cup"], 300000, seed=21)[0]
    Sc = trimesh.sample.sample_surface(chk, 300000, seed=22)[0]
    info["rmax_by_band_cm_visual"] = {k: round(v * 100, 2) for k, v in radius_by_band(Sv, G, 0.10).items()}
    info["rmax_by_band_cm_col"] = {k: round(v * 100, 2) for k, v in radius_by_band(Sc, G, 0.10).items()}
    info["outer_r_cm_by_band_nomarker_median"] = [round(float(np.nanmedian(x)) * 100, 2) for x in rmaxG_nm]
    shutil.copy(p, os.path.join(orch, "beige_col.stl"))
    open(os.path.join(orch, "beige_test.urdf"), "w").write(URDF_TEST.format(name="beige_test", collisions=COLL.format(fn="beige_col.stl")))
    # bob1-style URDFs with the scene's visual STLs, to prove the cache keys match the training's
    for nm in NAMES:
        fn = os.path.basename(sf["obj"][nm]["stl"])
        shutil.copy(sf["obj"][nm]["stl"], os.path.join(orch, fn))
        open(os.path.join(orch, f"visual_{nm}.urdf"), "w").write(URDF_TEST.format(name=f"visual_{nm}", collisions=COLL.format(fn=fn)))
    res = dict(scene=args.scene, parquet=sf["parquet"], parquet_md5=md5(sf["parquet"]), emulate_f1=args.emulate_f1,
               axis=dict((nm, ax_json(AX[nm])) for nm in NAMES),
               height_cm={nm: round(AX[nm]["H"] * 100, 2) for nm in NAMES},
               visual_stl_md5={nm: md5(sf["obj"][nm]["stl"]) for nm in NAMES},
               beige_col=info, beige_col_md5=md5(p))
    jdump(res, os.path.join(args.work, "prep.json"))
    print(json.dumps({k: v for k, v in info.items() if k not in ("rmax_by_band_cm_visual", "rmax_by_band_cm_col", "outer_r_cm_by_band_nomarker_median")}, indent=0))
    print("visual rmax by band >10cm", info["rmax_by_band_cm_visual"])
    print("col    rmax by band >10cm", info["rmax_by_band_cm_col"])


# ----------------------------------------------------------------------------------------------- slabs + flare + floor
def end_nest_depth(ref, AX, f0=124, f1=160):
    arr = np.array([nest(AX["blue_cup"], AX["beige_cup"], ref["P"][f, 0], ref["R"][f, 0], ref["P"][f, 1], ref["R"][f, 1])
                    for f in range(ref["N"])])
    f1 = min(f1, ref["N"] - 1)
    seg = arr[f0:f1 + 1, 0]
    return float(np.median(seg)), float(seg.min()), arr


def slab_points(A, rmaxB, flare_b, i, j, t=T_SLAB):
    """nestplan.blue_slabs construction (+ flare at the band boundaries): 8 corner points of slab (band i, az j)."""
    H = A["H"]; hs = np.linspace(0, H, NB_SLAB + 1)
    th0 = -np.pi + 2 * np.pi * j / NPHI_SLAB; th1 = th0 + 2 * np.pi / NPHI_SLAB
    pts = []
    for k, hh in ((i, hs[i]), (i + 1, hs[i + 1])):
        for th in (th0 + 1e-3, th1 - 1e-3):
            ro_ = float(interp_prof(rmaxB, [min(hh, H - 1e-4)], [th])[0]) + flare_b[k]
            for rr in (ro_, ro_ - t):
                pts.append(A["b"] + hh * A["a"] + rr * (np.cos(th) * A["e1"] + np.sin(th) * A["e2"]))
    return np.array(pts)


def inner_radius_lin(A, rmaxB, flare_b, h, t=T_SLAB):
    """min over the 24 slabs of the slab inner-face radius at height h (chord mid-point, linear in h inside a band)."""
    H = A["H"]; hs = np.linspace(0, H, NB_SLAB + 1)
    i = int(np.clip(np.searchsorted(hs, h, side="right") - 1, 0, NB_SLAB - 1))
    u = (h - hs[i]) / (hs[i + 1] - hs[i])
    half = (2 * np.pi / NPHI_SLAB - 2e-3) / 2
    best = 1e9
    for j in range(NPHI_SLAB):
        th = -np.pi + 2 * np.pi * (j + 0.5) / NPHI_SLAB
        r0 = float(interp_prof(rmaxB, [min(hs[i], H - 1e-4)], [th])[0]) + flare_b[i] - t
        r1 = float(interp_prof(rmaxB, [min(hs[i + 1], H - 1e-4)], [th])[0]) + flare_b[i + 1] - t
        best = min(best, (r0 * (1 - u) + r1 * u) * np.cos(half))
    return best


def hull_rmax_at(hulls, G, hq):
    """exact max distance from the beige axis of the union of convex hulls cut by the plane h = hq (beige frame):
    cross-section vertices are hull-edge / plane intersections (plus vertices lying on the plane)."""
    out = np.full(len(hq), -1.0)
    for v, f in hulls:
        hv, _, _ = cyl(v, G)
        E = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
        a, b = E[:, 0], E[:, 1]
        ha, hb_ = hv[a], hv[b]
        lo, hi = np.minimum(ha, hb_), np.maximum(ha, hb_)
        for k, hh in enumerate(hq):
            if hh < hv.min() or hh > hv.max():
                continue
            sel = (lo <= hh) & (hi >= hh)
            if not sel.any():
                continue
            aa, bb = a[sel], b[sel]
            den = hv[bb] - hv[aa]
            t = np.where(np.abs(den) > 1e-12, (hh - hv[aa]) / np.where(np.abs(den) > 1e-12, den, 1), 0.0)
            P = v[aa] + t[:, None] * (v[bb] - v[aa])
            _, rr, _ = cyl(P, G)
            out[k] = max(out[k], float(rr.max()))
    return out


def flare_profile(A, G, rmaxB, beige_hulls, d_floor, step=0.001):
    """Minimum monotone outward flare that keeps >= CLEAR (1 mm) between the slab inner faces and the beige CoACD
    hulls in the coaxial pose with the beige bottom on the floor, at the worst twist (beige max radius over all
    azimuths vs the smallest slab inner-face radius over all azimuths).  The plan's monotone profile ('fine') is then
    realised with the slab geometry (linear between the 9 band edges) by a small LP: minimise the sum of the
    band-edge flares subject to the linear interpolation >= fine profile everywhere and monotone, >= 0."""
    from scipy.optimize import linprog
    H = A["H"]; hs = np.linspace(0, H, NB_SLAB + 1)
    zero = np.zeros(NB_SLAB + 1)
    half = (2 * np.pi / NPHI_SLAB - 2e-3) / 2
    hh = np.arange(max(d_floor, 0.0), H + 1e-9, step)
    rb = hull_rmax_at(beige_hulls, G, hh - d_floor)
    ok = rb > 0
    hh, rb = hh[ok], rb[ok]
    have = np.array([inner_radius_lin(A, rmaxB, zero, h) for h in hh])
    need = (rb + CLEAR - have) / np.cos(half)
    fine = np.maximum.accumulate(np.maximum(need, 0.0))
    rows = [(float(a), float(b), float(c), float(d)) for a, b, c, d in zip(hh, rb, have, need)]
    nv = NB_SLAB + 1
    Aub, bub = [], []
    for h, req in zip(hh, fine):
        if req <= 0:
            continue
        i = int(np.clip(np.searchsorted(hs, h, side="right") - 1, 0, NB_SLAB - 1)); u = (h - hs[i]) / (hs[i + 1] - hs[i])
        row = np.zeros(nv); row[i] = -(1 - u); row[i + 1] = -u
        Aub.append(row); bub.append(-req)
    for k in range(NB_SLAB):                       # monotone: f_k - f_{k+1} <= 0
        row = np.zeros(nv); row[k] = 1; row[k + 1] = -1; Aub.append(row); bub.append(0.0)
    if len(bub) > NB_SLAB:
        lp = linprog(np.ones(nv), A_ub=np.array(Aub), b_ub=np.array(bub), bounds=[(0, None)] * nv, method="highs")
        if not lp.success:
            raise RuntimeError(f"flare LP failed: {lp.message}")
        f = np.maximum(lp.x, 0.0) + 1e-6 * (lp.x > 1e-9)
    else:
        f = zero.copy()
    f = np.maximum.accumulate(f)
    return f, rows, fine


def floor_points(A, rmaxB, flare_b, d_floor, n=48, margin=0.0003):
    rin = min(inner_radius_lin(A, rmaxB, flare_b, d_floor), inner_radius_lin(A, rmaxB, flare_b, d_floor - FLOOR_T))
    rf = rin - margin
    pts = []
    for hh in (d_floor - FLOOR_T, d_floor):
        for th in np.linspace(-np.pi, np.pi, n, endpoint=False):
            pts.append(A["b"] + hh * A["a"] + rf * (np.cos(th) * A["e1"] + np.sin(th) * A["e2"]))
    return np.array(pts), rf


def cmd_slabs(args):
    sf, meshes, ref, AX = setup(args)
    rminB, rmaxB, rminG, rmaxG = prof_cache(args, meshes, AX)
    A, G = AX["blue_cup"], AX["beige_cup"]
    hb = load_hullset(os.path.join(args.work, "hulls_beige.npz"))
    beige_hulls = hb[sorted(hb)[0]] if len(hb) == 1 else hb["beige_col"]
    d_med, d_min, arr = end_nest_depth(ref, AX)
    d_end = d_med if args.d_end_cm is None else args.d_end_cm / 100
    d_floor = d_end - FLOOR_BELOW
    flare_b, rows, fine = flare_profile(A, G, rmaxB, beige_hulls, d_floor)
    orch = os.path.join(args.work, "orch"); os.makedirs(orch, exist_ok=True)
    pieces = []
    for i in range(NB_SLAB):
        for j in range(NPHI_SLAB):
            k = i * NPHI_SLAB + j
            fn = f"blue_col_{k:03d}.stl"
            m = write_stl(os.path.join(args.work, fn), slab_points(A, rmaxB, flare_b, i, j))
            shutil.copy(os.path.join(args.work, fn), os.path.join(orch, fn))
            pieces.append(dict(file=fn, band=i, az=j, n_vertices=int(len(m.vertices)), volume_mm3=round(float(m.volume) * 1e9, 2)))
    fp, rf = floor_points(A, rmaxB, flare_b, d_floor)
    m = write_stl(os.path.join(args.work, "blue_floor.stl"), fp)
    shutil.copy(os.path.join(args.work, "blue_floor.stl"), os.path.join(orch, "blue_floor.stl"))
    pieces.append(dict(file="blue_floor.stl", n_vertices=int(len(m.vertices)), volume_mm3=round(float(m.volume) * 1e9, 2)))
    colls = "\n".join(COLL.format(fn=p["file"]) for p in pieces)
    open(os.path.join(orch, "blue_test.urdf"), "w").write(URDF_TEST.format(name="blue_test", collisions=colls))
    hs = np.linspace(0, A["H"], NB_SLAB + 1)
    res = dict(scene=args.scene, emulate_f1=args.emulate_f1,
               d_end_source="override" if args.d_end_cm is not None else "scene reference: beige bottom above blue bottom on the blue axis, median f124-160",
               d_end_cm=round(d_end * 100, 3), d_end_min_cm_f124_160=round(d_min * 100, 3), d_floor_top_cm=round(d_floor * 100, 3),
               floor=dict(top_cm=round(d_floor * 100, 3), bottom_cm=round((d_floor - FLOOR_T) * 100, 3), radius_cm=round(rf * 100, 3), n_sides=48),
               slab=dict(t_mm=T_SLAB * 1000, bands=NB_SLAB, azimuths=NPHI_SLAB, band_edges_cm=[round(float(x) * 100, 3) for x in hs]),
               flare_mm_at_band_edges=dict(zip([round(float(x) * 100, 2) for x in hs], [round(float(x) * 1000, 2) for x in flare_b])),
               flare_fine_mm_every_5mm=[(round(r[0] * 100, 1), round(r[1] * 100, 2), round(r[2] * 100, 2), round(r[3] * 1000, 2), round(float(fv) * 1000, 2))
                                        for r, fv in list(zip(rows, fine))[::5]],
               flare_fine_cols="blue h cm, beige hull max r cm (worst twist, exact cross-section), unflared slab inner r cm (chord mid, min over azimuth), need mm, monotone flare mm",
               flare_starts_cm=(round(float(rows[int(np.argmax(fine > 0))][0]) * 100, 2) if (fine > 0).any() else None),
               flare_at_rim_mm=round(float(flare_b[-1]) * 1000, 2), grasp_band_7_8p7cm_flare_mm=[
                   round(float(np.interp(h, hs, flare_b)) * 1000, 3) for h in (0.070, 0.087)],
               pieces=pieces, n_pieces=len(pieces),
               nest_ref_cm_every5=[(f, round(arr[f, 0] * 100, 2), round(arr[f, 1] * 100, 2), round(arr[f, 2], 1)) for f in range(0, ref["N"], 5)])
    jdump(res, os.path.join(args.work, "slabs.json"))
    print(json.dumps({k: res[k] for k in ("d_end_cm", "d_end_min_cm_f124_160", "d_floor_top_cm", "floor", "flare_mm_at_band_edges",
                                           "flare_starts_cm", "flare_at_rim_mm", "grasp_band_7_8p7cm_flare_mm", "n_pieces")}, indent=0))
    for r in res["flare_fine_mm_every_5mm"]:
        print("  flare row", r)


# ----------------------------------------------------------------------------------------------- checks a, b, d
def slab_index_hulls(hb_set):
    """blue hull set (orchard) -> list of (piece file, [hull (v, eq)])."""
    out = []
    for nm in sorted(hb_set):
        out.append((nm, as_hs(hb_set[nm])))
    return out


def through_wall_pen(A, pieces, gpts_blue_local, floor_top, floor_r):
    """Honest thin-wall penetration: for each beige point (blue local frame) cast a ray from the blue axis (same
    height) outward through the point; if it enters a wall piece before reaching the point, the excess radius is
    the penetration (also catches points that already passed through the wall).  Floor: points below the floor top
    inside the floor radius.  Returns max penetration (m) and the arg info."""
    h, r, ph = cyl(gpts_blue_local, A)
    c = A["b"][None, :] + h[:, None] * A["a"][None, :]
    u = gpts_blue_local - c; nu = np.linalg.norm(u, axis=1); u = u / np.maximum(nu, 1e-12)[:, None]
    best = 0.0; arg = None
    for nm, hs in pieces:
        if nm.startswith("blue_floor"):
            continue
        for v, eq in hs:
            hv, rv, phv = cyl(v, A)
            lo_h, hi_h = hv.min(), hv.max()
            dph = np.angle(np.exp(1j * (ph - np.angle(np.mean(np.exp(1j * phv))))))
            sel = (h >= lo_h) & (h <= hi_h) & (np.abs(dph) < 0.4) & (r > rv.min() - 0.02)
            if not sel.any():
                continue
            idx = np.nonzero(sel)[0]
            n_ = eq[:, :3]; d_ = eq[:, 3]
            nu_ = u[idx] @ n_.T                        # (m, F)
            nc = c[idx] @ n_.T + d_                    # (m, F)
            with np.errstate(divide="ignore", invalid="ignore"):
                s = -nc / nu_
            lower = np.where(nu_ < -1e-12, s, -np.inf).max(1)
            upper = np.where(nu_ > 1e-12, s, np.inf).min(1)
            inside0 = np.all(np.where(np.abs(nu_) <= 1e-12, nc <= 0, True), 1)
            hit = (lower <= upper) & inside0 & (upper > 0)
            excess = np.where(hit, r[idx] - np.maximum(lower, 0), -1)
            k = int(np.argmax(excess))
            if excess[k] > best:
                best = float(excess[k]); arg = dict(piece=nm, h_cm=round(float(h[idx[k]]) * 100, 2), r_cm=round(float(r[idx[k]]) * 100, 2),
                                                    az_deg=round(float(np.degrees(ph[idx[k]])), 1))
    fl = (r < floor_r) & (h < floor_top)
    fpen = float((floor_top - h[fl]).max()) if fl.any() else 0.0
    return best, arg, fpen


def cmd_check(args):
    sf, meshes, ref, AX = setup(args)
    rminB, rmaxB, rminG, rmaxG = prof_cache(args, meshes, AX)
    A, G = AX["blue_cup"], AX["beige_cup"]
    sl = json.load(open(os.path.join(args.work, "slabs.json")))
    orch_sum = json.load(open(os.path.join(args.work, "orch_summary.json")))
    hbset = load_hullset(os.path.join(args.work, "hulls_blue.npz"))
    hgset = load_hullset(os.path.join(args.work, "hulls_beige.npz"))
    beige_hulls = hgset["beige_col"] if "beige_col" in hgset else hgset[sorted(hgset)[0]]
    HG = as_hs(beige_hulls)
    pieces = slab_index_hulls(hbset)
    HBall = [x for _, hs in pieces for x in hs]
    res = dict(scene=args.scene, emulate_f1=args.emulate_f1, parquet=sf["parquet"], parquet_md5=md5(sf["parquet"]))
    # ---- a) CoACD
    a = dict(per_piece=[], fail=[])
    for p in orch_sum["pieces"]:
        if not p["file"].startswith("blue_"):
            continue
        m = trimesh.load(os.path.join(args.work, p["file"]), force="mesh")
        hv = np.concatenate([v for v, f in hbset[p["file"]]])
        bb = float(np.abs(hv.min(0) - m.bounds[0]).max() + 0 * 1) * 1000
        bb = max(bb, float(np.abs(hv.max(0) - m.bounds[1]).max()) * 1000)
        ok = (p["n_hulls"] == 1) and bb <= 0.2 and p.get("auto_equals_off", True)
        a["per_piece"].append((p["file"], p["n_hulls"], round(bb, 4), p.get("auto_equals_off")))
        if not ok:
            a["fail"].append(p["file"])
    a["n_blue_pieces"] = len(a["per_piece"]); a["all_one_hull_within_0.2mm"] = len(a["fail"]) == 0
    a["max_bbox_diff_mm"] = max(x[2] for x in a["per_piece"])
    # beige: max hull radius above 10 cm vs marker-free visual surface (+1 mm)
    gpts = surf_pts(HG, n_per=400, seed=5)
    col = trimesh.load(os.path.join(args.work, "beige_col.stl"), force="mesh")
    Sc = trimesh.sample.sample_surface(col, 400000, seed=22)[0]
    hull_b = radius_by_band(gpts, G, 0.10); col_b = radius_by_band(Sc, G, 0.10)
    rows = []; worst = -1e9
    for k in sorted(hull_b):
        if k in col_b:
            ex = (hull_b[k] - col_b[k]) * 1000; worst = max(worst, ex)
            rows.append((k, round(hull_b[k] * 100, 3), round(col_b[k] * 100, 3), round(ex, 2)))
    a["beige"] = dict(n_hulls=len(beige_hulls), rows_h_hullmax_colmax_excess_mm=rows, worst_excess_mm=round(worst, 2),
                      pass_le_1mm=bool(worst <= 1.0), auto_equals_off=next((p.get("auto_equals_off") for p in orch_sum["pieces"] if p["file"] == "beige_col.stl"), None),
                      overall_max_r_cm=round(float(cyl(gpts, G)[1].max()) * 100, 3))
    a["pass"] = bool(a["all_one_hull_within_0.2mm"] and a["beige"]["pass_le_1mm"])
    a["per_piece"] = a["per_piece"][:3] + [["..."]] + a["per_piece"][-2:]
    res["a"] = a
    print("a)", json.dumps({k: v for k, v in a.items() if k != "per_piece"})[:1500], flush=True)
    # ---- b) static vs the reference f105-160 + coaxial drop at all twists
    floor_top = sl["floor"]["top_cm"] / 100; floor_r = sl["floor"]["radius_cm"] / 100
    pb_pts = surf_pts(HBall, n_per=30, seed=1)
    gpts2 = surf_pts(HG, n_per=200, seed=2)
    P, R = ref["P"], ref["R"]
    rows = []; worst = 0.0; worstp = 0.0
    for f in range(105, min(161, ref["N"])):
        g_in_b = ((gpts2 @ R[f, 1].T + P[f, 1]) - P[f, 0]) @ R[f, 0]
        b_in_g = ((pb_pts @ R[f, 0].T + P[f, 0]) - P[f, 1]) @ R[f, 1]
        plan_pen = max(max_depth(g_in_b, HBall), max_depth(b_in_g, HG))
        tw, arg, fpen = through_wall_pen(A, pieces, g_in_b, floor_top, floor_r)
        d, rad, tilt = nest(A, G, P[f, 0], R[f, 0], P[f, 1], R[f, 1])
        rows.append(dict(f=f, nest_cm=round(d * 100, 2), radial_cm=round(rad * 100, 2), tilt_deg=round(tilt, 1),
                         plan_pen_mm=round(plan_pen * 1000, 2), wall_pen_mm=round(tw * 1000, 2), floor_pen_mm=round(fpen * 1000, 2), at=arg))
        worst = max(worst, tw, fpen); worstp = max(worstp, plan_pen)
    # upward shift (along the blue axis) the reference beige needs to be free, on a few frames
    def free_shift(f):
        lo, hi = 0.0, 0.16
        def pen_at(s):
            Pg = P[f, 1] + (R[f, 0] @ A["a"]) * s
            g_in_b = ((gpts2 @ R[f, 1].T + Pg) - P[f, 0]) @ R[f, 0]
            tw, _, fp = through_wall_pen(A, pieces, g_in_b, floor_top, floor_r)
            return max(tw, fp)
        if pen_at(0.0) <= 0.0005:
            return 0.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if pen_at(mid) > 0.0005:
                lo = mid
            else:
                hi = mid
        return hi
    shifts = {f: round(free_shift(f) * 100, 2) for f in (105, 110, 115, 120, 125, 130, 140, 150, 160) if f < ref["N"]}
    # diagnostic: same frames with the beige's tilt relative to the blue removed (beige rotated about its own bottom point
    # onto the blue axis direction; its depth, radial offset and twist kept) -> how much of the penetration is the
    # reference's relative tilt
    co_rows = []
    for f in range(105, min(161, ref["N"])):
        aw = R[f, 0] @ A["a"]; gaw = R[f, 1] @ G["a"]; gbw = R[f, 1] @ G["b"] + P[f, 1]
        v = np.cross(gaw, aw); s = np.linalg.norm(v)
        Rm = np.eye(3) if s < 1e-9 else Rot.from_rotvec(v / s * np.arctan2(s, gaw @ aw)).as_matrix()
        Rgc = Rm @ R[f, 1]; Pgc = gbw - Rgc @ G["b"]
        g_in_b = ((gpts2 @ Rgc.T + Pgc) - P[f, 0]) @ R[f, 0]
        tw, arg, fpen = through_wall_pen(A, pieces, g_in_b, floor_top, floor_r)
        dd, rad, tilt = nest(A, G, P[f, 0], R[f, 0], Pgc, Rgc)
        co_rows.append(dict(f=f, rel_tilt_removed_deg=round(float(np.degrees(np.arctan2(s, gaw @ aw))), 2), nest_cm=round(dd * 100, 2),
                            radial_cm=round(rad * 100, 2), wall_pen_mm=round(tw * 1000, 2), floor_pen_mm=round(fpen * 1000, 2),
                            blue_tilt_vertical_deg=round(float(np.degrees(np.arccos(np.clip(aw[2], -1, 1)))), 2),
                            beige_tilt_vertical_deg=round(float(np.degrees(np.arccos(np.clip(gaw[2], -1, 1)))), 2)))

    # coaxial drop at 24 twists
    def coax_stop(theta, tol=0.0005):
        lo, hi = -0.03, A["H"] + 0.02
        for _ in range(16):
            mid = 0.5 * (lo + hi); Pt, Rt = coax(A, G, mid, theta)
            g_in_b = gpts2 @ Rt.T + Pt
            tw, _, fp = through_wall_pen(A, pieces, g_in_b, floor_top, floor_r)
            if max(tw, fp) > tol:
                lo = mid
            else:
                hi = mid
        return hi
    twists = np.radians(np.arange(0, 360, 15))
    stops = [round(coax_stop(th) * 100, 3) for th in twists]
    # beige hull lowest point above its own mesh bottom (the stop is measured for the mesh bottom)
    hull_bottom = float(cyl(np.concatenate([v for v, e in HG]), G)[0].min())
    b = dict(frames=rows, max_wall_or_floor_pen_mm=round(worst * 1000, 2), max_plan_pen_mm=round(worstp * 1000, 2),
             frames_wall_pen_gt_2mm=[r_["f"] for r_ in rows if max(r_["wall_pen_mm"], r_["floor_pen_mm"]) > 2.0],
             free_shift_cm=shifts, coax_stop_cm_by_15deg=stops, floor_top_cm=sl["floor"]["top_cm"],
             beige_hull_bottom_below_mesh_bottom_mm=round(-hull_bottom * 1000, 2),
             coax_stop_minus_floor_cm=[round(s - sl["floor"]["top_cm"], 3) for s in stops])
    b["diag_rel_tilt_removed"] = dict(rows=co_rows[::5], max_pen_mm=round(max(max(r_["wall_pen_mm"], r_["floor_pen_mm"]) for r_ in co_rows), 2),
                                      frames_gt_2mm=[r_["f"] for r_ in co_rows if max(r_["wall_pen_mm"], r_["floor_pen_mm"]) > 2.0],
                                      note="diagnostic only: beige rotated about its bottom point onto the blue axis (relative tilt removed); depth, radial offset, twist kept")
    b["pass_static_le_2mm"] = bool(worst <= 0.002)
    b["pass_coax"] = bool(all(abs(s - sl["floor"]["top_cm"] + hull_bottom * 100) <= 0.2 for s in stops))
    b["pass"] = bool(b["pass_static_le_2mm"] and b["pass_coax"])
    res["b"] = b
    print("b)", json.dumps({k: v for k, v in b.items() if k != "frames"}), flush=True)
    for r_ in rows[::5]:
        print("   ", r_, flush=True)
    # ---- d) hands
    JL, JR = ref["JL"], ref["JR"]
    left_new, left_vis, right_new, right_old = [], [], [], []
    old = None
    if args.bob1_beige_hulls and os.path.exists(args.bob1_beige_hulls):
        z = np.load(args.bob1_beige_hulls); old = [(z[f"v{i}"].astype(float), hull_eq(z[f"v{i}"].astype(float))) for i in range(int(z["n"]))]
    for f in range(ref["N"]):
        lb = (JL[f] - P[f, 0]) @ R[f, 0]
        left_new.append(int(count_inside(lb, HBall).sum()))
        hh, rr, pp = cyl(lb, A)
        rin = interp_prof(rminB, hh, pp); rout = interp_prof(rmaxB, hh, pp)
        left_vis.append(int(((hh > 0) & (hh < A["H"]) & (rr >= rin) & (rr <= rout)).sum()))
        rg = (JR[f] - P[f, 1]) @ R[f, 1]
        right_new.append(int(count_inside(rg, HG).sum()))
        if old is not None:
            right_old.append(int(count_inside(rg, old).sum()))
    d = dict(left_joints_in_new_blue_total=int(sum(left_new)), left_joints_in_blue_visual_wall_total=int(sum(left_vis)),
             left_frames_new=[f for f, c in enumerate(left_new) if c], left_frames_visual=[f for f, c in enumerate(left_vis) if c],
             right_joints_in_new_beige_total=int(sum(right_new)),
             right_joints_in_bob1_beige_total=int(sum(right_old)) if old is not None else None,
             right_frames_new=[f for f, c in enumerate(right_new) if c],
             right_frames_bob1=[f for f, c in enumerate(right_old) if c] if old is not None else None)
    # also the blue envelope check for the right hand vs the new blue pieces (risk note: thumb on the rim f93-102)
    rb_new = []
    for f in range(ref["N"]):
        rb = (JR[f] - P[f, 0]) @ R[f, 0]
        rb_new.append(int(count_inside(rb, HBall).sum()))
    d["right_joints_in_new_blue_total"] = int(sum(rb_new)); d["right_frames_in_new_blue"] = [f for f, c in enumerate(rb_new) if c]
    d["pass_left"] = bool(d["left_joints_in_new_blue_total"] <= d["left_joints_in_blue_visual_wall_total"])
    d["pass_right"] = bool(old is not None and d["right_joints_in_new_beige_total"] <= d["right_joints_in_bob1_beige_total"])
    d["pass"] = bool(d["pass_left"] and d["pass_right"])
    res["d"] = d
    print("d)", json.dumps(d), flush=True)
    jdump(res, os.path.join(args.work, "checks.json"))


# ----------------------------------------------------------------------------------------------- c) MuJoCo settle
def parse_support(usd_path):
    txt = open(usd_path).read()
    cyls = []
    for blk in re.split(r"def Cylinder", txt)[1:]:
        r_ = float(re.search(r"double radius = ([0-9.eE+-]+)", blk).group(1))
        h_ = float(re.search(r"double height = ([0-9.eE+-]+)", blk).group(1))
        c_ = [float(x) for x in re.search(r"xformOp:translate = \(([^)]*)\)", blk).group(1).split(",")]
        cyls.append((h_, r_, c_[0], c_[1], c_[2]))
    return cyls


def cmd_settle(args):
    import mujoco
    sf, meshes, ref, AX = setup(args)
    A, G = AX["blue_cup"], AX["beige_cup"]
    sl = json.load(open(os.path.join(args.work, "slabs.json")))
    hbset = load_hullset(os.path.join(args.work, "hulls_blue.npz"))
    hgset = load_hullset(os.path.join(args.work, "hulls_beige.npz"))
    HBv = [v for nm in sorted(hbset) for v, f in hbset[nm]]
    HGv = [v for v, f in (hgset["beige_col"] if "beige_col" in hgset else hgset[sorted(hgset)[0]])]
    sup = parse_support(sf["usd"]) if sf["usd"] else []
    P, Q, R = ref["P"], ref["Q"], ref["R"]
    floor_top = sl["floor"]["top_cm"] / 100
    old = None
    if args.bob1_beige_hulls and args.bob1_blue_hulls:
        zb = np.load(args.bob1_blue_hulls); zg = np.load(args.bob1_beige_hulls)
        old = ([zb[f"v{i}"].astype(float) for i in range(int(zb["n"]))], [zg[f"v{i}"].astype(float) for i in range(int(zg["n"]))])

    def run(blue_hulls, beige_hulls, Pb, Qb, Pg, Qg, T=2.0, dt=0.002):
        assets, bodies = [], []
        objs = (("blue", blue_hulls, Pb, Qb),) + ((("beige", beige_hulls, Pg, Qg),) if beige_hulls is not None else ())
        for nm, hl, pos, q in objs:
            geoms = []
            for i, v in enumerate(hl):
                assets.append(f'<mesh name="{nm}{i}" vertex="{" ".join(f"{x:.7f}" for x in v.ravel())}"/>')
                geoms.append(f'<geom type="mesh" mesh="{nm}{i}" friction="1 0.005 0.0001" condim="3"/>')
            bodies.append(f'<body name="{nm}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}"><freejoint/>'
                          f'<inertial pos="0 0 0" mass="0.3" diaginertia="0.001 0.001 0.001"/>{"".join(geoms)}</body>')
        sups = "".join(f'<geom type="cylinder" size="{r_} {h_ / 2}" pos="{c0} {c1} {c2}" friction="1 0.005 0.0001"/>' for h_, r_, c0, c1, c2 in sup)
        xml = (f'<mujoco><compiler inertiafromgeom="false"/><option timestep="{dt}" gravity="0 0 -9.81"/><size memory="512M"/>'
               f'<asset>{"".join(assets)}</asset><worldbody>{sups}{"".join(bodies)}</worldbody></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        Rb0 = Rot.from_quat([Qb[1], Qb[2], Qb[3], Qb[0]]).as_matrix()
        ncon, rows, dmin = [], [], 1e9
        marks = {int(round(t / dt)) for t in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0)}
        for k in range(int(T / dt) + 1):
            if k > 0:
                mujoco.mj_step(m, d)
            ncon.append(int(d.ncon))
            pb, Rb = d.xpos[1].copy(), d.xmat[1].reshape(3, 3).copy()
            if beige_hulls is not None:
                pg, Rg = d.xpos[2].copy(), d.xmat[2].reshape(3, 3).copy()
            else:
                pg, Rg = pb, Rb
            dd, rad, tilt = nest(A, G, pb, Rb, pg, Rg)
            dmin = min(dmin, dd)
            if k in marks:
                rows.append(dict(t=round(k * dt, 2), nest_cm=round(dd * 100, 2), radial_cm=round(rad * 100, 2), tilt_rel_deg=round(tilt, 2),
                                 beige_tilt_vertical_deg=round(float(np.degrees(np.arccos(np.clip((Rg @ G["a"])[2], -1, 1)))), 2),
                                 blue_move_mm=round(float(np.linalg.norm(pb - Pb)) * 1000, 2),
                                 blue_tilt_change_deg=round(float(np.degrees(np.arccos(np.clip((Rb @ A["a"]) @ (Rb0 @ A["a"]), -1, 1)))), 2),
                                 blue_tilt_vertical_deg=round(float(np.degrees(np.arccos(np.clip((Rb @ A["a"])[2], -1, 1)))), 2),
                                 blue_bottom_z_cm=round(float((Rb @ A["b"] + pb)[2] - ZT) * 100, 2)))
        return dict(rows=rows, ncon_max=int(max(ncon)), ncon_mean=round(float(np.mean(ncon)), 1), ncon_last=int(ncon[-1]), nest_min_cm=round(dmin * 100, 2),
                    n_geoms=int(m.ngeom), blue_final_pos=pb.tolist(), blue_final_axis=(Rb @ A["a"]).tolist())
    out = dict(scene=args.scene, emulate_f1=args.emulate_f1, support=sup, floor_top_cm=sl["floor"]["top_cm"], runs={})
    up = np.array([0, 0, 0.02])
    for f in (140, 100):
        if f >= ref["N"]:
            continue
        lab = f"f{f}: blue at ref pose, beige released 2 cm above its ref pose"
        d0, rad0, tilt0 = nest(A, G, P[f, 0], R[f, 0], P[f, 1] + up, R[f, 1])
        r_ = run(HBv, HGv, P[f, 0], Q[f, 0], P[f, 1] + up, Q[f, 1])
        r_["start_nest_cm"] = round(d0 * 100, 2); r_["start_radial_cm"] = round(rad0 * 100, 2); r_["start_tilt_rel_deg"] = round(tilt0, 2)
        r0 = run(HBv, None, P[f, 0], Q[f, 0], None, None)          # control: the blue alone from the same pose
        r_["blue_alone_final"] = r0["rows"][-1]
        r_["blue_move_vs_alone_mm"] = round(float(np.linalg.norm(np.array(r_["blue_final_pos"]) - np.array(r0["blue_final_pos"]))) * 1000, 2)
        r_["blue_axis_vs_alone_deg"] = round(float(np.degrees(np.arccos(np.clip(np.dot(r_["blue_final_axis"], r0["blue_final_axis"]), -1, 1)))), 2)
        r_["ref_blue_tilt_vertical_deg"] = round(float(np.degrees(np.arccos(np.clip((R[f, 0] @ A["a"])[2], -1, 1)))), 2)
        out["runs"][lab] = r_
        print("##", lab, json.dumps(r_), flush=True)
    # coaxial variant (beige upright in the blue, 2 cm above the floor-contact depth), f140 blue pose
    f = 140 if ref["N"] > 140 else ref["N"] - 1
    Pt, Rt = coax(A, G, floor_top + 0.02, 0.0)
    Pg = R[f, 0] @ Pt + P[f, 0]; Rg = R[f, 0] @ Rt
    qg = Rot.from_matrix(Rg).as_quat(); qg = np.array([qg[3], qg[0], qg[1], qg[2]])
    r_ = run(HBv, HGv, P[f, 0], Q[f, 0], Pg, qg)
    out["runs"]["coaxial: beige 2 cm above the floor depth, blue f140"] = r_
    print("## coaxial", json.dumps(r_), flush=True)
    if old is not None:
        Pt, Rt = coax(A, G, 0.16, 0.0)                 # today's hulls: drop onto the rim from 16 cm
        Pg = R[f, 0] @ Pt + P[f, 0]; Rg = R[f, 0] @ Rt
        qg = Rot.from_matrix(Rg).as_quat(); qg = np.array([qg[3], qg[0], qg[1], qg[2]])
        r_ = run(old[0], old[1], P[f, 0], Q[f, 0], Pg, qg)
        out["runs"]["today (bob1 CoACD hulls): coaxial drop from 16 cm, blue f140"] = r_
        print("## today", json.dumps(r_), flush=True)
    c = dict()
    ok = True
    for lab in [k for k in out["runs"] if k.startswith("f")]:
        r_ = out["runs"][lab]; last = r_["rows"][-1]
        tests = dict(nest_5p2_5p9=bool(5.2 <= last["nest_cm"] <= 5.9), tilt_rel_lt_3deg=bool(last["tilt_rel_deg"] < 3),
                     blue_move_vs_alone_lt_3mm=bool(r_["blue_move_vs_alone_mm"] < 3), blue_upright_lt_5deg=bool(last["blue_tilt_vertical_deg"] < 5),
                     no_tunnelling=bool(last["nest_cm"] >= floor_top * 100 - 0.3 and last["radial_cm"] < 1.0),
                     contacts_le_150=bool(r_["ncon_max"] <= 150))
        p_ = all(tests.values())
        c[lab] = dict(pass_=bool(p_), tests=tests, final=last, ncon_max=r_["ncon_max"], nest_min_transient_cm=r_["nest_min_cm"],
                      blue_move_raw_mm=last["blue_move_mm"], blue_move_vs_alone_mm=r_["blue_move_vs_alone_mm"],
                      ref_blue_tilt_vertical_deg=r_["ref_blue_tilt_vertical_deg"], blue_alone_move_mm=r_["blue_alone_final"]["blue_move_mm"])
        ok = ok and p_
    out["c"] = dict(per_run=c, pass_=bool(ok),
                    criteria=("final (t=2 s) nest 5.2-5.9 cm; beige tilt vs blue axis < 3 deg; blue moves < 3 mm relative to a blue-alone "
                              "settle from the same pose (the raw move also contains the reference blue's own fall to upright) and stays "
                              "upright (< 5 deg from vertical); no tunnelling (final nest >= floor top - 3 mm and beige still inside, radial "
                              "< 1 cm; transient minimum reported); contacts per step <= 150"))
    jdump(out, os.path.join(args.work, "settle.json"))
    print("c)", json.dumps(out["c"]), flush=True)


# ----------------------------------------------------------------------------------------------- apply
def cmd_apply(args):
    if os.path.basename(os.path.normpath(args.scene)) != args.target:
        raise SystemExit(f"apply refused: scene basename != --target {args.target!r}")
    sf = scene_files(args.scene)
    proc = sf["proc"]
    bk = os.path.join(args.scene, "col_backup")
    if os.path.exists(os.path.join(args.scene, "NESTCOL.json")):
        raise SystemExit("NESTCOL.json already exists in the scene -> refusing to apply twice")
    os.makedirs(bk, exist_ok=False)
    sl = json.load(open(os.path.join(args.work, "slabs.json")))
    before = {}
    for p in glob.glob(os.path.join(proc, "*")) + glob.glob(os.path.join(args.scene, "score_input", "meshes", "*", "*")):
        if os.path.isfile(p):
            before[os.path.relpath(p, args.scene)] = md5(p)
    for nm in NAMES:
        shutil.copy2(sf["obj"][nm]["urdf"], os.path.join(bk, os.path.basename(sf["obj"][nm]["urdf"])))
    files = ["beige_col.stl"] + [p["file"] for p in sl["pieces"]]
    for fn in files:
        dst = os.path.join(proc, fn)
        if os.path.exists(dst):
            raise SystemExit(f"{dst} exists -> refusing to overwrite")
        shutil.copy2(os.path.join(args.work, fn), dst)
    coll_re = re.compile(r"\s*<collision>.*?</collision>", re.S)
    for nm, colfiles in (("beige_cup", ["beige_col.stl"]), ("blue_cup", [p["file"] for p in sl["pieces"]])):
        u = sf["obj"][nm]["urdf"]; txt = open(u).read()
        if len(coll_re.findall(txt)) != 1 or "<visual>" not in txt:
            raise SystemExit(f"{u}: unexpected URDF structure")
        new = coll_re.sub("\n" + "\n".join(COLL.format(fn=fn) for fn in colfiles), txt, count=1)
        tmp = u + ".tmp_nestcol"
        open(tmp, "w").write(new)
        os.replace(tmp, u)
    after = {}
    for p in glob.glob(os.path.join(proc, "*")) + glob.glob(os.path.join(args.scene, "score_input", "meshes", "*", "*")):
        if os.path.isfile(p):
            after[os.path.relpath(p, args.scene)] = md5(p)
    changed = sorted(k for k in before if before[k] != after.get(k))
    added = sorted(k for k in after if k not in before)
    # parquet URDF paths
    t = pq.read_table(sf["parquet"]).to_pylist()[0]
    up = list(t["object_urdf_paths"])
    up_ok = all(os.path.realpath(x) in (os.path.realpath(sf["obj"][nm]["urdf"]) for nm in NAMES) for x in up) and len(up) == 2
    prep = json.load(open(os.path.join(args.work, "prep.json")))
    nc = dict(fix="F2 ep10 nest collision proxy (plan.fixes[1], C:/v2d_demo/diag/ep10_fix/ep10_diagnosis_plan.json)",
              rule=("beige collision = visual mesh minus the 4 Vicon marker spheres (vertices > 6 mm beyond the per-height median radius, "
                    "grown while > 1 mm beyond the marker-free wall, holes closed flat on the wall), cavity kept, CoACD as usual; "
                    "blue collision = 192 explicit convex wall slabs on the outer surface (8 bands x 24 azimuths, t = 1.5 mm) with a "
                    "monotone rim flare (>= 1 mm clearance from the beige CoACD hulls, coaxial end-nest pose, worst twist) + a 3 mm "
                    "floor slab whose top is at the median end-rest nest depth (f124-160) minus 2 mm. Visual, score_input meshes and "
                    "the parquet unchanged."),
              d_end_cm=sl["d_end_cm"], d_end_min_cm=sl["d_end_min_cm_f124_160"], floor=sl["floor"],
              flare_mm_at_band_edges=sl["flare_mm_at_band_edges"], flare_fine=sl["flare_fine_mm_every_5mm"], flare_at_rim_mm=sl["flare_at_rim_mm"],
              pieces=dict(beige=1, blue=len(sl["pieces"]), blue_slabs=len(sl["pieces"]) - 1, blue_floor=1),
              beige_col=dict(md5=md5(os.path.join(proc, "beige_col.stl")), info={k: v for k, v in prep["beige_col"].items() if not k.startswith("rmax_by_band")}),
              backup_dir=bk, urdf_backup_md5={os.path.basename(p): md5(p) for p in glob.glob(os.path.join(bk, "*.urdf"))},
              urdf_new_md5={os.path.basename(sf["obj"][nm]["urdf"]): md5(sf["obj"][nm]["urdf"]) for nm in NAMES},
              files_changed=changed, files_added=len(added),
              parquet=sf["parquet"], parquet_md5=md5(sf["parquet"]), parquet_object_urdf_paths=up, parquet_urdf_paths_point_at_edited=bool(up_ok),
              written=time.strftime("%Y-%m-%d %H:%M:%S %Z"), tool="~/v2d_ops/kenc/diag/ep10_fix/nestcol.py",
              rules_note="The collision proxy differs from the submitted mesh (thin walls, flare, floor, no markers) - rules question for Ken.")
    jdump(nc, os.path.join(args.scene, "NESTCOL.json"))
    print(json.dumps(dict(changed=changed, added=len(added), parquet_urdf_ok=up_ok, urdf_paths=up), indent=0))


def cmd_ecmp(args):
    """gt_free_check e: md5 of score_input meshes, processed *_visual.stl and *.obj of SCENE vs REF_SCENE (F1)."""
    out = dict(scene=args.scene, ref_scene=args.ref_scene, files=[], all_identical=True)
    a, b = scene_files(args.scene), scene_files(args.ref_scene)
    pairs = []
    for p in sorted(glob.glob(os.path.join(args.scene, "score_input", "meshes", "*", "*"))):
        pairs.append((p, os.path.join(args.ref_scene, os.path.relpath(p, args.scene)), "score_input"))
    for nm in NAMES:
        pairs.append((a["obj"][nm]["stl"], b["obj"][nm]["stl"], "visual_stl"))
        pairs.append((a["obj"][nm]["objfile"], b["obj"][nm]["objfile"], "processed_obj"))
    for pa, pb_, kind in pairs:
        ma = md5(pa) if os.path.exists(pa) else None
        mb = md5(pb_) if os.path.exists(pb_) else None
        rec = dict(kind=kind, file=os.path.relpath(pa, args.scene), ref_file=os.path.relpath(pb_, args.ref_scene), md5=ma, ref_md5=mb,
                   identical=bool(ma is not None and ma == mb))
        if kind == "processed_obj" and ma and mb and ma != mb:
            # the processed OBJ names its scene-specific .mtl; compare with that line normalised
            ta = re.sub(rb"(?m)^mtllib .*$", b"mtllib X", open(pa, "rb").read())
            tb = re.sub(rb"(?m)^mtllib .*$", b"mtllib X", open(pb_, "rb").read())
            rec["identical_except_mtllib_name"] = bool(ta == tb)
        out["files"].append(rec)
        ok = rec["identical"] or rec.get("identical_except_mtllib_name", False)
        out["all_identical"] = bool(out["all_identical"] and ok)
    jdump(out, os.path.join(args.work, "ecmp.json"))
    print(json.dumps(out, indent=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "slabs", "check", "settle", "apply", "ecmp"])
    ap.add_argument("--reuse", default=None, help="prep: reuse beige_col.stl/profiles/hulls from this dry-run work dir")
    ap.add_argument("--target", default=None, help="apply: required basename of the scene to modify")
    ap.add_argument("--ref_scene", default=None, help="ecmp: the F1 scene to compare against")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--emulate_f1", default=None)
    ap.add_argument("--d_end_cm", type=float, default=None)
    ap.add_argument("--bob1_beige_hulls", default=None)
    ap.add_argument("--bob1_blue_hulls", default=None)
    args = ap.parse_args()
    t0 = time.time()
    dict(prep=cmd_prep, slabs=cmd_slabs, check=cmd_check, settle=cmd_settle, apply=cmd_apply, ecmp=cmd_ecmp)[args.cmd](args)
    print(f"[{args.cmd}] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
