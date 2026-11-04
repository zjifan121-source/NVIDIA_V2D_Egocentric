"""followup.py -- per-vertex clearance of each resting mesh above the depth-table plane, object tilt vs the
real table normal, and object xy positions. Reuses depthtab.py outputs (ep3_table_points.npz, ep3_depthtab.json).
usage: python followup.py <scene dir> <out dir>"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/v2d_ops/kenc/review/tool"))
import render_review as rr  # noqa: E402

SC, OUT = sys.argv[1:3]
rng = np.random.default_rng(0)
J = json.load(open(os.path.join(OUT, "ep3_depthtab.json")))
Z = np.load(os.path.join(OUT, "ep3_table_points.npz"))
pqp = rr.find_parquet(SC)
P = rr.load_scene_parquet(pqp)
N = P["N"]
proc_dir = os.path.dirname(os.path.dirname(os.path.dirname(pqp)))
names = ["white_pot", "wooden_spoon"]
PH = {"white_pot": {"start": list(range(0, 49, 3)), "end": list(range(252, 313, 4))},
      "wooden_spoon": {"start": list(range(0, 37, 3)), "end": list(range(276, 313, 4))}}


def fit_plane(Pts, thr=0.006, iters=600):
    n = len(Pts)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        nv = np.cross(Pts[idx[1]] - Pts[idx[0]], Pts[idx[2]] - Pts[idx[0]])
        nn = np.linalg.norm(nv)
        if nn < 1e-12:
            continue
        nv /= nn
        if abs(nv[2]) < 0.94:
            continue
        cnt = int((np.abs((Pts - Pts[idx[0]]) @ nv) < thr).sum())
        if best is None or cnt > best[0]:
            best = (cnt, nv, Pts[idx[0]])
    inl = np.abs((Pts - best[2]) @ best[1]) < thr
    for _ in range(3):
        X = np.c_[Pts[inl, 0], Pts[inl, 1], np.ones(inl.sum())]
        coef, *_ = np.linalg.lstsq(X, Pts[inl, 2], rcond=None)
        r = Pts[:, 2] - (X @ coef if False else coef[0] * Pts[:, 0] + coef[1] * Pts[:, 1] + coef[2])
        inl = np.abs(r) < thr
    return coef


cG = np.array(J["global_plane"]["coef"])
nG = np.array([-cG[0], -cG[1], 1.0]) / np.linalg.norm([-cG[0], -cG[1], 1.0])
res = dict(global_normal=nG.round(5).tolist())
# support hull: global depth plane height over the sim support disk (centre / extremes)
usd = sorted(glob.glob(os.path.join(SC, "assets/human_motion_data/ego_recon/reconstructed_stage", "*.usda")))
sup = max(rr.parse_usda(usd[0]), key=lambda s: s["area"])
hull = sup["hull_xy"]
cen = hull.mean(0)
zh = cG[0] * hull[:, 0] + cG[1] * hull[:, 1] + cG[2]
res["support"] = dict(top_z=sup["top_z"], centre_xy=cen.round(4).tolist(), radius_m=round(float(np.linalg.norm(hull - cen, axis=1).max()), 3),
                      depth_plane_minus_support_cm_at_centre=round(float((cG[0] * cen[0] + cG[1] * cen[1] + cG[2] - sup["top_z"]) * 100), 2),
                      depth_plane_minus_support_cm_min_max_over_rim=[round(float((zh.min() - sup["top_z"]) * 100), 2), round(float((zh.max() - sup["top_z"]) * 100), 2)])
for k, nm in enumerate(names):
    mp = P["mesh_paths"][k]
    mpath = next(c for c in [mp, os.path.join(proc_dir, os.path.basename(mp))] if os.path.isfile(c))
    V, F = rr.load_obj(mpath)
    R = rr.wxyz_to_R(P["wxyz"][:, k]).reshape(N, 3, 3)
    t = P["pos"][:, k]
    up_b = R[0].T @ np.array([0, 0, 1.0])  # body axis that is sim-vertical at frame 0
    # body axis of the flattest side: normal of the lowest-vertex patch at frame 0 (fit plane to verts within 5 mm)
    W0 = V @ R[0].T + t[0]
    low0 = W0[:, 2] < W0[:, 2].min() + 0.005
    eo = {}
    for ph, fr in PH[nm].items():
        pts = np.concatenate([Z[f"ann_{nm}_{i}"][:, :3] for i in fr if f"ann_{nm}_{i}" in Z.files])
        c = fit_plane(pts.astype(float))
        nL = np.array([-c[0], -c[1], 1.0]) / np.linalg.norm([-c[0], -c[1], 1.0])
        rows = []
        for i in fr:
            Wv = V @ R[i].T + t[i]
            h = Wv[:, 2] - (c[0] * Wv[:, 0] + c[1] * Wv[:, 1] + c[2])  # height above the local depth plane
            hs = np.sort(h)
            up_w = R[i] @ up_b
            rows.append(dict(f=i, clearance_min_cm=hs[0] * 100, clearance_p1pct_cm=hs[max(0, len(hs) // 100)] * 100,
                             verts_below_table=int((h < 0).sum()), deepest_cm=hs[0] * 100,
                             up_tilt_vs_table_deg=float(np.degrees(np.arccos(np.clip(abs(up_w @ nL), -1, 1)))),
                             up_tilt_vs_sim_z_deg=float(np.degrees(np.arccos(np.clip(abs(up_w[2]), -1, 1)))),
                             xy=(t[i][:2]).round(4).tolist(), bottom_z=float(Wv[:, 2].min())))
        md = lambda key: round(float(np.median([r[key] for r in rows])), 2)  # noqa: E731
        eo[ph] = dict(local_coef=c.round(6).tolist(), local_tilt_vs_sim_z_deg=round(float(np.degrees(np.arccos(nL[2]))), 2),
                      clearance_min_cm_median=md("clearance_min_cm"), clearance_p1pct_cm_median=md("clearance_p1pct_cm"),
                      verts_below_table_median=int(np.median([r["verts_below_table"] for r in rows])),
                      up_tilt_vs_table_deg_median=md("up_tilt_vs_table_deg"), up_tilt_vs_sim_z_deg_median=md("up_tilt_vs_sim_z_deg"),
                      xy_first=rows[0]["xy"], xy_last=rows[-1]["xy"],
                      per_frame=[{k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in r.items()} for r in rows])
        print(nm, ph, json.dumps({k2: v for k2, v in eo[ph].items() if k2 != "per_frame"}))
    res[nm] = eo
    print(nm, "start->end xy shift cm", round(float(np.linalg.norm(np.array(eo["end"]["xy_first"]) - np.array(eo["start"]["xy_last"])) * 100), 1))
print("support", res["support"])
json.dump(res, open(os.path.join(OUT, "ep3_followup.json"), "w"), indent=1, default=float)
print("FOLLOWUP_DONE")
