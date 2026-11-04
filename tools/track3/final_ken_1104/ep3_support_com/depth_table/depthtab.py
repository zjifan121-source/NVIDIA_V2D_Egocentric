"""depthtab.py -- GT-free: real table surface from our FoundationStereo depth vs the sim support plane and
the resting objects' mesh bottoms, in the scene (loader/sim) world.

usage: python depthtab.py <scene dir> <ken_c episode dir> <out dir>
Depth encoding (as measmesh_fp.py / bsdf_track_prep.py): depth_m = 65535/raw - 1 for 0 < raw < 65535.
Camera in sim world = A @ bundle_cleaned c2w (A from render_review.estimate_A).
"""
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.expanduser("~/v2d_ops/kenc/review/tool"))
import render_review as rr  # noqa: E402

T0 = time.time()
SC, REC, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
H, W = 1520, 2028
rng = np.random.default_rng(0)


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


# ------------------------------------------------------------------ scene, bundle, A, support
pqp = rr.find_parquet(SC)
P = rr.load_scene_parquet(pqp)
N = P["N"]
B = rr.load_bundle(os.path.join(SC, "bundle_cleaned", "result.npz"))
proc_dir = os.path.dirname(os.path.dirname(os.path.dirname(pqp)))
names = B["names"]
objs = []
for k, nm in enumerate(names):
    mp = P["mesh_paths"][k]
    mpath = next(c for c in [mp, os.path.join(proc_dir, os.path.basename(mp))] if os.path.isfile(c))
    V, F = rr.load_obj(mpath)
    R = rr.wxyz_to_R(P["wxyz"][:, k]).reshape(N, 3, 3)
    T = rr.to_T(R, P["pos"][:, k])
    lo, hi = V.min(0), V.max(0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    objs.append(dict(name=nm, mesh=mpath, V=V, F=F, T=T, corners=corners))
A, res = rr.estimate_A([o["T"] for o in objs], [B["objT"][k] for k in range(len(objs))],
                       [o["corners"] for o in objs])
c2w = A[None] @ B["c2w"]
fx, fy, cx, cy = B["K"][0]
usd = sorted(glob.glob(os.path.join(SC, "assets/human_motion_data/ego_recon/reconstructed_stage", "*.usda")))
sups = rr.parse_usda(usd[0])
sup = max(sups, key=lambda s: s["area"])
Z_SUP = sup["top_z"]
log(f"A t={A[:3, 3].round(4)} rot={rr.rot_angle_deg(np.eye(3), A[:3, :3]):.3f} deg res max "
    f"{np.nanmax(res) * 1000:.3f} mm; support {sup['name']} top {Z_SUP:.4f} area {sup['area']:.3f}")

for o in objs:
    Pw = np.einsum("nij,vj->nvi", o["T"][:, :3, :3], o["V"]) + o["T"][:, None, :3, 3]
    z = Pw[..., 2]
    o["bottom"] = z.min(1)
    low = z < (o["bottom"][:, None] + 0.003)
    o["xy_low"] = (Pw[..., :2] * low[..., None]).sum(1) / low.sum(1)[:, None]
    o["centroid"] = Pw.mean(1)
    o["step_mm"] = np.r_[0, np.linalg.norm(np.diff(o["centroid"], axis=0), axis=1) * 1000]
    del Pw

PHASES = {"white_pot": {"start": list(range(0, 49, 3)), "end": list(range(252, 313, 4))},
          "wooden_spoon": {"start": list(range(0, 37, 3)), "end": list(range(276, 313, 4))}}
for o in objs:
    for ph, fr in PHASES[o["name"]].items():
        log(f"{o['name']} {ph}: frames {fr[0]}..{fr[-1]} bottom z {np.round(o['bottom'][fr], 4).tolist()} "
            f"max centroid step {o['step_mm'][fr].max():.2f} mm")
frames = sorted(set(f for o in objs for f in sum(PHASES[o["name"]].values(), [])))


# ------------------------------------------------------------------ helpers
def read_depth(i):
    raw = np.asarray(Image.open(f"{REC}/white_pot/depth/{i:06d}.png")).astype(np.float64)
    ok = (raw > 0) & (raw < 65535)
    return np.where(ok, 65535.0 / np.maximum(raw, 1.0) - 1.0, np.nan)


def read_mask(obj, k, i):
    p = f"{REC}/{obj}/masks/{k}/{i:06d}.png"
    if not os.path.isfile(p):
        return np.zeros((H, W), bool)
    return np.asarray(Image.open(p).convert("L")) > 0


def zbuffer(Vc, F):
    """camera-space vertices -> z-buffer (H, W) by dense barycentric splatting (<= 0.5 px spacing)."""
    zb = np.full(H * W, np.inf)
    ok = Vc[:, 2] > 0.05
    Fk = F[ok[F].all(1)]
    uv = np.stack([fx * Vc[:, 0] / np.where(ok, Vc[:, 2], 1) + cx, fy * Vc[:, 1] / np.where(ok, Vc[:, 2], 1) + cy], 1)
    tri = uv[Fk]
    L = np.max(np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=2), axis=1)
    s = np.clip(np.ceil(L / 0.5).astype(int), 1, 400)
    for sv in np.unique(s):
        sel = np.where(s == sv)[0]
        ii, jj = np.meshgrid(np.arange(sv + 1), np.arange(sv + 1), indexing="ij")
        m = ii + jj <= sv
        b1, b2 = ii[m] / sv, jj[m] / sv
        b0 = 1 - b1 - b2
        per = len(b0)
        step = max(1, 3_000_000 // per)
        for c0 in range(0, len(sel), step):
            ff = Fk[sel[c0:c0 + step]]
            Vs = (b0[None, :, None] * Vc[ff[:, 0]][:, None] + b1[None, :, None] * Vc[ff[:, 1]][:, None]
                  + b2[None, :, None] * Vc[ff[:, 2]][:, None]).reshape(-1, 3)
            u = np.round(fx * Vs[:, 0] / Vs[:, 2] + cx).astype(np.int64)
            v = np.round(fy * Vs[:, 1] / Vs[:, 2] + cy).astype(np.int64)
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            np.minimum.at(zb, v[inb] * W + u[inb], Vs[inb, 2])
    return zb.reshape(H, W)


def fit_plane(Pts, thr=0.006, iters=500):
    """robust near-horizontal plane z = a x + b y + c. Returns (a, b, c), inlier mask, rms (m)."""
    n = len(Pts)
    if n < 50:
        return None, None, None
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        nv = np.cross(Pts[idx[1]] - Pts[idx[0]], Pts[idx[2]] - Pts[idx[0]])
        nn = np.linalg.norm(nv)
        if nn < 1e-12:
            continue
        nv /= nn
        if abs(nv[2]) < 0.94:  # within ~20 deg of horizontal
            continue
        d = np.abs((Pts - Pts[idx[0]]) @ nv)
        cnt = int((d < thr).sum())
        if best is None or cnt > best[0]:
            best = (cnt, nv, Pts[idx[0]])
    if best is None:
        return None, None, None
    inl = np.abs((Pts - best[2]) @ best[1]) < thr
    for _ in range(3):
        X = np.c_[Pts[inl, 0], Pts[inl, 1], np.ones(inl.sum())]
        coef, *_ = np.linalg.lstsq(X, Pts[inl, 2], rcond=None)
        r = Pts[:, 2] - (coef[0] * Pts[:, 0] + coef[1] * Pts[:, 1] + coef[2])
        inl = np.abs(r) < thr
    rms = float(np.sqrt(np.mean(r[inl] ** 2)))
    return coef, inl, rms


def plane_z(coef, xy):
    return coef[0] * xy[..., 0] + coef[1] * xy[..., 1] + coef[2]


def tilt(coef):
    g = np.hypot(coef[0], coef[1])
    return float(np.degrees(np.arctan(g))), float(np.degrees(np.arctan2(coef[1], coef[0])))


uu, vv = np.meshgrid(np.arange(W), np.arange(H))
SUB = (slice(0, H, 2), slice(0, W, 2))
dil = lambda m, r: ndimage.binary_dilation(m, iterations=r)  # noqa: E731
ero = lambda m, r: ndimage.binary_erosion(m, iterations=r)  # noqa: E731

# ------------------------------------------------------------------ per frame
glob_pts = []
ann = {o["name"]: {} for o in objs}
objchk = {o["name"]: {} for o in objs}
align = {}
for i in frames:
    Z = read_depth(i)
    hands = np.zeros((H, W), bool)
    for ob in ("white_pot", "wooden_spoon"):
        for k in (2, 3):
            hands |= read_mask(ob, k, i)
    hands_d = dil(hands, 20)
    sam = {o["name"]: read_mask(o["name"], 1, i) for o in objs}
    w2c = np.linalg.inv(c2w[i])
    zbs = {}
    for o in objs:
        Vw = o["V"] @ o["T"][i, :3, :3].T + o["T"][i, :3, 3]
        Vc = Vw @ w2c[:3, :3].T + w2c[:3, 3]
        zbs[o["name"]] = zbuffer(Vc, o["F"])
    excl = hands_d.copy()
    for o in objs:
        excl |= dil(sam[o["name"]] | np.isfinite(zbs[o["name"]]), 12)
    # back-project table candidates (every 2nd px)
    Zs = Z[SUB]
    ok = np.isfinite(Zs) & ~excl[SUB]
    u, v, zz = uu[SUB][ok], vv[SUB][ok], Zs[ok]
    Pc = np.stack([(u - cx) / fx * zz, (v - cy) / fy * zz, zz], 1)
    Pw = Pc @ c2w[i, :3, :3].T + c2w[i, :3, 3]
    cen = np.mean([o["xy_low"][i] for o in objs], 0)
    win = (np.abs(Pw[:, 2] - Z_SUP) < 0.08) & (np.linalg.norm(Pw[:, :2] - cen, axis=1) < 0.45)
    Pwin = Pw[win]
    if len(Pwin) > 30000:
        Pg = Pwin[rng.choice(len(Pwin), 30000, replace=False)]
    else:
        Pg = Pwin
    glob_pts.append(np.c_[Pg, np.full(len(Pg), i)])
    # local annuli
    for o in objs:
        Vw = o["V"] @ o["T"][i, :3, :3].T + o["T"][i, :3, 3]
        other = [q for q in objs if q is not o]
        d_own = cKDTree(Vw[::max(1, len(Vw) // 4000), :2]).query(Pwin[:, :2])[0]
        d_oth = np.min([cKDTree((q["V"] @ q["T"][i, :3, :3].T + q["T"][i, :3, 3])[:, :2]).query(Pwin[:, :2])[0]
                        for q in other], axis=0) if other else np.full(len(Pwin), 9.0)
        sel = (d_own > 0.02) & (d_own < 0.12) & (d_oth > 0.02)
        ann[o["name"]][i] = np.c_[Pwin[sel], d_own[sel]]
    # object depth check
    for o in objs:
        zb = zbs[o["name"]]
        others = [zbs[q["name"]] for q in objs if q is not o]
        front = np.ones((H, W), bool)
        for zo in others:
            front &= ~(zo < zb)
        rend = np.isfinite(zb)
        pix = ero(sam[o["name"]], 5) & ero(rend, 5) & front & ~hands_d & np.isfinite(Z)
        inter = (sam[o["name"]] & rend).sum()
        union = (sam[o["name"]] | rend).sum()
        Vw = o["V"] @ o["T"][i, :3, :3].T + o["T"][i, :3, 3]
        cc = (Vw.mean(0) @ w2c[:3, :3].T + w2c[:3, 3])
        ray_w = c2w[i, :3, :3] @ (cc / np.linalg.norm(cc))
        e = dict(n=int(pix.sum()), iou=round(float(inter / max(union, 1)), 3),
                 cam_dist_m=round(float(np.linalg.norm(cc)), 3), ray_z=round(float(ray_w[2]), 3))
        if pix.sum() > 30:
            dlt = (Z - zb)[pix]
            e["dz_mm_p25_p50_p75"] = (np.percentile(dlt, [25, 50, 75]) * 1000).round(1).tolist()
            # height change if the object were moved along its camera ray by the median depth difference
            e["implied_dz_world_cm"] = round(float(np.median(dlt) * np.linalg.norm(cc) / cc[2] * ray_w[2] * 100), 2)
            # top vs bottom half of the visible pixels (image rows) -> tilt/shape hints
            rows = vv[pix]
            med_r = np.median(rows)
            e["dz_mm_upper_rows"] = round(float(np.median(dlt[rows < med_r]) * 1000), 1)
            e["dz_mm_lower_rows"] = round(float(np.median(dlt[rows >= med_r]) * 1000), 1)
            # pixels where SAM says object but our mesh is absent (missing parts): their depth -> world z
            miss = ero(sam[o["name"]], 3) & ~dil(rend, 3) & ~hands_d & np.isfinite(Z) & front
            if miss.sum() > 20:
                zm = Z[miss]
                Pm = np.stack([(uu[miss] - cx) / fx * zm, (vv[miss] - cy) / fy * zm, zm], 1) @ c2w[i, :3, :3].T + c2w[i, :3, 3]
                e["sam_not_mesh_n"] = int(miss.sum())
                e["sam_not_mesh_world_z_p10_p50_p90"] = np.percentile(Pm[:, 2], [10, 50, 90]).round(4).tolist()
            # lowest real object points: SAM-mask pixels back-projected -> lowest world z (p2) vs mesh bottom
            sm = ero(sam[o["name"]], 3) & ~hands_d & np.isfinite(Z) & front
            if sm.sum() > 50:
                zs_ = Z[sm]
                Ps = np.stack([(uu[sm] - cx) / fx * zs_, (vv[sm] - cy) / fy * zs_, zs_], 1) @ c2w[i, :3, :3].T + c2w[i, :3, 3]
                e["sam_obj_world_z_p2_p50_p98"] = np.percentile(Ps[:, 2], [2, 50, 98]).round(4).tolist()
                e["mesh_visible_world_z_top"] = round(float(Vw[:, 2].max()), 4)
        objchk[o["name"]][i] = e
    # depth/image registration check on frame 0 (pot SAM boundary vs depth gradient)
    if i == 0:
        gz = np.hypot(*np.gradient(np.nan_to_num(Z, nan=np.nanmedian(Z))))
        m = sam["white_pot"]
        bd = m & ~ero(m, 1)
        ys, xs = np.nonzero(bd)
        sc = {}
        for du in range(-10, 11, 2):
            for dv in range(-10, 11, 2):
                y2, x2 = ys + dv, xs + du
                okk = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
                sc[(du, dv)] = float(np.mean(gz[y2[okk], x2[okk]]))
        bestk = max(sc, key=sc.get)
        align = dict(best_shift_px=list(bestk), grad_at_best=round(sc[bestk] * 1000, 2),
                     grad_at_zero=round(sc[(0, 0)] * 1000, 2))
        log(f"depth/SAM-boundary registration at f0: {align}")
    log(f"frame {i}: table pts {len(Pwin)}; " + "; ".join(
        f"{o['name']} ann {len(ann[o['name']][i])} obj {objchk[o['name']][i].get('n')} dz50 "
        f"{objchk[o['name']][i].get('dz_mm_p25_p50_p75', [None, None])[1]} iou {objchk[o['name']][i]['iou']}"
        for o in objs))

# ------------------------------------------------------------------ global plane
G = np.concatenate(glob_pts)
coefG, inlG, rmsG = fit_plane(G[:, :3], thr=0.008, iters=800)
tG, azG = tilt(coefG)
per_frame_off = {}
for i in frames:
    s = (G[:, 3] == i) & inlG
    if s.sum() > 100:
        per_frame_off[i] = round(float(np.median(G[s, 2] - plane_z(coefG, G[s, :2]))) * 100, 3)
# also per-frame plane, to show frame-to-frame consistency of the depth table under the objects
glob_res = dict(n=int(len(G)), inliers=int(inlG.sum()), rms_mm=round(rmsG * 1000, 2), coef=coefG.round(6).tolist(),
                tilt_deg=round(tG, 3), uphill_azimuth_deg_of_gradient=round(azG, 1),
                per_frame_median_offset_cm=per_frame_off)
log(f"GLOBAL depth table plane: tilt {tG:.2f} deg (gradient azimuth {azG:.0f}), rms {rmsG * 1000:.1f} mm, "
    f"inliers {inlG.sum()}/{len(G)}")

# ------------------------------------------------------------------ per object / phase
out = dict(scene=SC, recon=REC, depth="FoundationStereo (ken_c_scored white_pot/depth, 65535/raw-1 m)",
           A_t=A[:3, 3].round(4).tolist(), A_rot_deg=float(rr.rot_angle_deg(np.eye(3), A[:3, :3])),
           support=dict(name=sup["name"], top_z=Z_SUP, area_m2=sup["area"]), registration_f0=align,
           global_plane=glob_res, objects={})
for o in objs:
    nm = o["name"]
    eo = {}
    for ph, fr in PHASES[nm].items():
        pts = np.concatenate([np.c_[ann[nm][i][:, :3], np.full(len(ann[nm][i]), i)] for i in fr if len(ann[nm][i])])
        coef, inl, rms = fit_plane(pts[:, :3], thr=0.006, iters=600)
        xy = o["xy_low"][fr]
        bot = o["bottom"][fr]
        zt_local = plane_z(coef, xy)
        zt_glob = plane_z(coefG, xy)
        # per-frame: median annulus z residual vs the phase plane, and a per-frame-only plane under the object
        pf = {}
        for i in fr:
            a = ann[nm][i]
            if len(a) < 200:
                pf[i] = None
                continue
            c_i, in_i, rms_i = fit_plane(a[:, :3], thr=0.006, iters=300)
            if c_i is None:
                pf[i] = None
                continue
            zt_i = float(plane_z(c_i, o["xy_low"][i]))
            pf[i] = dict(n=int(in_i.sum()), table_under_obj_z=round(zt_i, 4),
                         float_vs_depth_table_cm=round((o["bottom"][i] - zt_i) * 100, 2),
                         rms_mm=round(rms_i * 1000, 1), tilt_deg=round(tilt(c_i)[0], 2))
        tl, azl = tilt(coef)
        oc = [objchk[nm][i] for i in fr if "dz_mm_p25_p50_p75" in objchk[nm][i]]
        eo[ph] = dict(
            frames=f"{fr[0]}-{fr[-1]} (every {fr[1] - fr[0]})",
            annulus_points=int(len(pts)), inliers=int(inl.sum()), rms_mm=round(rms * 1000, 2),
            local_plane_tilt_deg=round(tl, 2), local_plane_gradient_azimuth_deg=round(azl, 1),
            mesh_bottom_z_median=round(float(np.median(bot)), 4),
            depth_table_under_obj_z_local=round(float(np.median(zt_local)), 4),
            depth_table_under_obj_z_global=round(float(np.median(zt_glob)), 4),
            sim_support_z=round(Z_SUP, 4),
            float_vs_depth_table_local_cm=round(float(np.median(bot - zt_local)) * 100, 2),
            float_vs_depth_table_global_cm=round(float(np.median(bot - zt_glob)) * 100, 2),
            float_vs_sim_support_cm=round(float(np.median(bot - Z_SUP)) * 100, 2),
            depth_table_minus_sim_support_cm=round(float(np.median(zt_local - Z_SUP)) * 100, 2),
            per_frame=pf,
            object_depth_check=dict(
                frames_used=len(oc),
                dz_mm_median_of_frame_medians=round(float(np.median([c["dz_mm_p25_p50_p75"][1] for c in oc])), 1) if oc else None,
                dz_mm_frame_medians=[c["dz_mm_p25_p50_p75"][1] for c in oc],
                implied_world_dz_cm_median=round(float(np.median([c["implied_dz_world_cm"] for c in oc])), 2) if oc else None,
                upper_rows_dz_mm=round(float(np.median([c["dz_mm_upper_rows"] for c in oc])), 1) if oc else None,
                lower_rows_dz_mm=round(float(np.median([c["dz_mm_lower_rows"] for c in oc])), 1) if oc else None,
                iou_median=round(float(np.median([c["iou"] for c in oc])), 3) if oc else None,
                sam_obj_lowest_world_z_p2_median=round(float(np.median([c["sam_obj_world_z_p2_p50_p98"][0] for c in oc if "sam_obj_world_z_p2_p50_p98" in c])), 4) if oc else None,
                n_pix_median=int(np.median([c["n"] for c in oc])) if oc else 0))
        log(f"{nm} {ph}: mesh bottom {eo[ph]['mesh_bottom_z_median']:.4f} | depth table local "
            f"{eo[ph]['depth_table_under_obj_z_local']:.4f} global {eo[ph]['depth_table_under_obj_z_global']:.4f} | "
            f"support {Z_SUP:.4f} || float vs depth table {eo[ph]['float_vs_depth_table_local_cm']:+.2f} cm "
            f"(global {eo[ph]['float_vs_depth_table_global_cm']:+.2f}), vs support {eo[ph]['float_vs_sim_support_cm']:+.2f} cm, "
            f"depth table - support {eo[ph]['depth_table_minus_sim_support_cm']:+.2f} cm; local tilt {tl:.2f} deg rms "
            f"{rms * 1000:.1f} mm; obj depth dz50 {eo[ph]['object_depth_check']['dz_mm_median_of_frame_medians']} mm "
            f"-> implied world dz {eo[ph]['object_depth_check']['implied_world_dz_cm_median']} cm")
    out["objects"][nm] = eo
out["per_frame_object_check"] = {nm: {str(k): v for k, v in d.items()} for nm, d in objchk.items()}
json.dump(out, open(os.path.join(OUT, "ep3_depthtab.json"), "w"), indent=1, default=float)
np.savez_compressed(os.path.join(OUT, "ep3_table_points.npz"), G=G.astype(np.float32),
                    **{f"ann_{nm}_{i}": a.astype(np.float32) for nm, d in ann.items() for i, a in d.items()})
log("DONE")
