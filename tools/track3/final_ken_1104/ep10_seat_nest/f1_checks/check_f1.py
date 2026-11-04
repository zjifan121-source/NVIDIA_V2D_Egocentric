"""k10 F1 GT-free checks a, b, c, f (+ contact windows) of k10_ep10_seat against k10_ep10_ctl.
usage (retarget venv python): check_f1.py <ctl scene> <seat scene> <dryrun_p10.json> <blue_toprise.json> <out.json>"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as Rot

CTL, SEAT, DRY, TOPR, OUTJ = sys.argv[1:6]
ZT = 1.0
RES = {"ctl": CTL, "seat": SEAT}


def load(scene):
    f = sorted(glob.glob(os.path.join(scene, "assets/human_motion_data/ego_recon/processed/*/robot_name=sharpa_wave/*.parquet")))[0]
    t = pq.read_table(f).to_pylist()[0]
    rep = json.load(open(os.path.join(scene, "report.json")))
    return f, t, rep


fc, tc, rc = load(CTL)
fs, ts, rs = load(SEAT)
cols = list(tc.keys())
RES["parquet_columns"] = cols
N = len(tc["object_body_position"])
names_col = next((c for c in cols if "body" in c and "name" in c), None)
body_names = [str(x) for x in tc[names_col]] if names_col else ["blue_cup", "beige_cup"]
RES["body_names"] = body_names
iB = next(i for i, n in enumerate(body_names) if n.endswith("blue_cup"))
iG = next(i for i, n in enumerate(body_names) if n.endswith("beige_cup"))


def poses(t):
    P = np.array(t["object_body_position"], float)
    Q = np.array(t["object_body_wxyz"], float)
    R = Rot.from_quat(Q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(N, -1, 3, 3)
    return P, R


Pc, Rc = poses(tc)
Ps, Rs = poses(ts)

# ---------------- (a) report.json
dry = json.load(open(DRY))
dz_build = np.array(rs["cleaning"]["p10_dz_cm"])
dz_dry = np.array(dry["dz_cm"])
dd = np.abs(dz_build - dz_dry)
a = {"p10_dz_build_vs_dryrun_maxabs_cm": round(float(dd.max()), 3),
     "p10_dz_build_vs_dryrun_frames_over_0.1cm": [int(f) for f in np.flatnonzero(dd > 0.1)],
     "p10_dz_build_vs_dryrun_frames_over_0.3cm": [int(f) for f in np.flatnonzero(dd > 0.3)],
     "p10_rest_hold_frames": rs["cleaning"].get("p10_rest_hold_frames"),
     "p10_frames_changed": rs["cleaning"].get("p10_frames_changed"),
     "p10_dz_min_cm": rs["cleaning"].get("p10_dz_min_cm"),
     "ctl_attached": {k: v for k, v in rc["cleaning"].items() if k.startswith("attached_") or k.endswith("_attached_frames")},
     "seat_attached": {k: v for k, v in rs["cleaning"].items() if k.startswith("attached_") or k.endswith("_attached_frames")},
     "rest_hold_ctl": {k: v for k, v in rc["cleaning"].items() if k.startswith("rest_hold")},
     "rest_hold_seat": {k: v for k, v in rs["cleaning"].items() if k.startswith("rest_hold")},
     "respin_ctl": {k: v for k, v in rc["cleaning"].items() if k.startswith("h5_respin") or k.startswith("h5_spin")},
     "respin_seat": {k: v for k, v in rs["cleaning"].items() if k.startswith("h5_respin") or k.startswith("h5_spin")},
     "attach_rule": [rc["cleaning"].get("attach_rule"), rs["cleaning"].get("attach_rule")],
     "output_check_ctl": rc.get("output_check"), "output_check_seat": rs.get("output_check")}
a["dz_table_every5_cm"] = {int(f): [round(float(dz_build[f]), 2), round(float(dz_dry[f]), 2)] for f in range(0, N, 5)}
RES["a"] = a

# ---------------- (b) parquet diff vs ctl
b = {}
dpos = np.abs(Ps - Pc)
ang = np.degrees(np.array([[Rot.from_matrix(Rc[f, k].T @ Rs[f, k]).magnitude() for k in range(Pc.shape[1])] for f in range(N)]))
b["beige_pos_maxabs_m"] = float(dpos[:, iG].max())
b["beige_rot_max_deg"] = float(ang[:, iG].max())
b["blue_xy_maxabs_m"] = float(dpos[:, iB, :2].max())
b["blue_z_change_cm_min_max"] = [round(float((Ps[:, iB, 2] - Pc[:, iB, 2]).min()) * 100, 3),
                                 round(float((Ps[:, iB, 2] - Pc[:, iB, 2]).max()) * 100, 3)]
b["blue_rot_change_max_deg"] = round(float(ang[:, iB].max()), 3)
# twist/swing of the blue rotation change about its symmetry axis (mesh PCA axis, body frame)
meshB = trimesh.load(sorted(glob.glob(os.path.join(SEAT, "assets/human_motion_data/ego_recon/processed/*blue_cup.obj")))[0],
                     force="mesh", process=False)
VB = np.asarray(meshB.vertices)
smp = trimesh.sample.sample_surface(meshB, 20000, seed=0)[0]
w, U = np.linalg.eigh(np.cov((smp - smp.mean(0)).T))
sep = [min(abs(w[j] - w[(j + 1) % 3]), abs(w[j] - w[(j + 2) % 3])) for j in range(3)]
axB = U[:, int(np.argmax(sep))]
tw, sw = [], []
for f in range(N):
    q = Rot.from_matrix(Rc[f, iB].T @ Rs[f, iB]).as_quat()
    th = 2 * np.arctan2(float(q[:3] @ axB), float(q[3]))
    th = (th + np.pi) % (2 * np.pi) - np.pi
    tw.append(np.degrees(abs(th)))
    swing = Rot.from_matrix(Rc[f, iB].T @ Rs[f, iB]) * Rot.from_rotvec(-axB * th)
    sw.append(np.degrees(swing.magnitude()))
b["blue_rot_change_twist_about_axis_max_deg"] = round(float(max(tw)), 3)
b["blue_rot_change_swing_max_deg"] = round(float(max(sw)), 4)
b["blue_rot_changed_frames"] = (lambda fr: [int(fr[0]), int(fr[-1]), int(len(fr))] if len(fr) else [])(np.flatnonzero(ang[:, iB] > 1e-3))


def hand_changes(key):
    xa, xb = np.array(tc[key], float), np.array(ts[key], float)
    per = np.abs(xa - xb).reshape(N, -1).max(1)
    fr = np.flatnonzero(per > 1e-5)
    return per, fr


for side in ("left", "right"):
    per, fr = hand_changes(f"mano_{side}_joints")
    b[f"{side}_mano_joints_changed_frames"] = [int(fr[0]), int(fr[-1]), int(len(fr))] if len(fr) else []
    b[f"{side}_mano_joints_max_change_cm"] = round(float(per.max()) * 100, 2)
    runs = []
    if len(fr):
        s = fr[0]
        for x, y in zip(fr[:-1], fr[1:]):
            if y != x + 1:
                runs.append([int(s), int(x)])
                s = y
        runs.append([int(s), int(fr[-1])])
    b[f"{side}_mano_joints_changed_runs"] = runs
    rk = next((c for c in cols if c.startswith(f"robot_{side}") and "frames" in c and "error" not in c), None)
    if rk:
        per2, fr2 = hand_changes(rk)
        b[f"{side}_{rk}_changed_frames"] = [int(fr2[0]), int(fr2[-1]), int(len(fr2))] if len(fr2) else []
# other columns changed (summary)
other = {}
for c in cols:
    if c in ("object_body_position", "object_body_wxyz") or c.startswith("mano_left_joints") or c.startswith("mano_right_joints"):
        continue
    try:
        xa, xb = np.array(tc[c], float), np.array(ts[c], float)
        if xa.shape != xb.shape:
            other[c] = "shape differs"
            continue
        d = np.abs(xa - xb)
        if d.size and np.nanmax(d) > 1e-5:
            per = d.reshape(xa.shape[0], -1).max(1) if xa.ndim > 1 else d
            fr = np.flatnonzero(per > 1e-5)
            other[c] = {"maxabs": float(np.nanmax(d)), "frames": [int(fr[0]), int(fr[-1]), int(len(fr))] if len(fr) else []}
    except (TypeError, ValueError):
        if json.dumps(tc[c], default=str) != json.dumps(ts[c], default=str):
            other[c] = "non-numeric differs"
b["other_columns_changed"] = other
RES["b"] = b

# ---------------- contact label windows
cw = {}
for lab, t in (("ctl", tc), ("seat", ts)):
    for side in ("left", "right"):
        C = np.array(t[f"mano_{side}_object_contact_part_ids"])
        vals, cnt = np.unique(C, return_counts=True)
        cw[f"{lab}_{side}_values"] = {str(v): int(c) for v, c in zip(vals, cnt)}
        nocon = vals[np.argmax(cnt)]
        for v in vals:
            if v == nocon:
                continue
            fr = np.flatnonzero((C == v).any(axis=1)) if C.ndim > 1 else np.flatnonzero(C == v)
            cw[f"{lab}_{side}_part{v}"] = [int(fr[0]), int(fr[-1]), int(len(fr))] if len(fr) else []
RES["contact_windows"] = cw

# ---------------- (c) blue lowest vertex vs its support top
sup = None
try:
    from pxr import Usd, UsdGeom
    usda = glob.glob(os.path.join(SEAT, "assets/human_motion_data/ego_recon/reconstructed_stage/*_support.usda"))[0]
    stage = Usd.Stage.Open(usda)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    boxes = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Cylinder):
            bx = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            boxes.append((np.array(bx.GetMin()), np.array(bx.GetMax()), str(prim.GetPath())))
    RES["support_boxes"] = [[b_[2], [round(float(x), 4) for x in b_[0]], [round(float(x), 4) for x in b_[1]]] for b_ in boxes]
    w0 = VB @ Rs[0, iB].T + Ps[0, iB]
    cen = w0[:, :2].mean(0)
    under = [hi[2] for lo, hi, _ in boxes if np.all(lo[:2] <= cen) and np.all(cen <= hi[:2])]
    sup = float(max(under)) if under else None
except Exception as e:  # noqa: BLE001
    RES["support_error"] = repr(e)
low_s = np.array([(VB @ Rs[f, iB].T + Ps[f, iB])[:, 2].min() for f in range(N)])
low_c = np.array([(VB @ Rc[f, iB].T + Pc[f, iB])[:, 2].min() for f in range(N)])
if sup is None:
    sup = float(low_s[0])
    RES["support_note"] = "support top from the blue's frame-0 lowest vertex (USD read failed)"
gs, gc = (low_s - sup) * 100, (low_c - sup) * 100
w = slice(55, N)
c = {"support_top_z_m": round(sup, 5), "blue_f0_seating_cm": round(float(gs[0]), 3),
     "seat_f55_160_min_max_cm": [round(float(gs[w].min()), 2), round(float(gs[w].max()), 2)],
     "ctl_f55_160_min_max_cm": [round(float(gc[w].min()), 2), round(float(gc[w].max()), 2)],
     "seat_frames_f55_160_outside_3mm": [int(f) for f in range(55, N) if abs(gs[f]) > 0.3],
     "seat_all_frames_min_cm": round(float(gs.min()), 3), "seat_all_frames_argmin": int(gs.argmin()),
     "seat_frames_below_minus3mm": [int(f) for f in range(N) if gs[f] < -0.3],
     "seat_gap_cm_every5": {int(f): round(float(gs[f]), 2) for f in range(0, N, 5)},
     "ctl_gap_cm_every5": {int(f): round(float(gc[f]), 2) for f in range(0, N, 5)}}
topr = json.load(open(TOPR))
rise = np.array([np.nan if topr["rise_cm"][str(f)] is None else topr["rise_cm"][str(f)] for f in range(N)])
rise5 = np.array([np.nanmedian(rise[max(0, f - 2):f + 3]) for f in range(N)])
c["seat_gap_minus_observed_rise5_f55_160_cm"] = [round(float((gs - np.maximum(rise5, 0))[w].min()), 2),
                                                round(float((gs - np.maximum(rise5, 0))[w].max()), 2)]
c["outside_frames_observed_rise5_cm"] = {int(f): [round(float(gs[f]), 2), round(float(rise5[f]), 2)]
                                        for f in c["seat_frames_f55_160_outside_3mm"]}
RES["c"] = c
RES["blue_gap_series_cm"] = {"seat": [round(float(x), 3) for x in gs], "ctl": [round(float(x), 3) for x in gc]}
# reference blue top (rim) minus observed p95 (check e numbers from per-frame tops; sk10a2 re-run gives the same obs)
top_s = np.array([(VB @ Rs[f, iB].T + Ps[f, iB])[:, 2].max() for f in range(N)])
top_c = np.array([(VB @ Rc[f, iB].T + Pc[f, iB])[:, 2].max() for f in range(N)])
p95 = np.array([np.nan if topr["p95_cm"][str(f)] is None else topr["p95_cm"][str(f)] for f in range(N)])
es, ec = (top_s - ZT) * 100 - p95, (top_c - ZT) * 100 - p95
RES["e_from_parquet"] = {"seat_f55_160_min_max_cm": [round(float(np.nanmin(es[w])), 2), round(float(np.nanmax(es[w])), 2)],
                         "ctl_f55_160_min_max_cm": [round(float(np.nanmin(ec[w])), 2), round(float(np.nanmax(ec[w])), 2)],
                         "seat_frames_f55_160_outside_1cm": [int(f) for f in range(55, N) if abs(es[f]) > 1.0],
                         "seat_every5": {int(f): round(float(es[f]), 2) for f in range(0, N, 5)},
                         "ctl_every5": {int(f): round(float(ec[f]), 2) for f in range(0, N, 5)},
                         "seat_series_cm": [round(float(x), 3) for x in es],
                         "ctl_series_cm": [round(float(x), 3) for x in ec],
                         "rise5_cm": [round(float(x), 3) for x in rise5],
                         "note": "reference blue top = max z of the posed blue mesh minus 1.0 m; observed = blue_toprise p95"}

# ---------------- (f) right-hand MANO joints inside the blue wall envelope (nestplan.py in_blue)
V = VB


def axis_frame(V, R0):
    cc0 = V - V.mean(0)
    w_, U_ = np.linalg.eigh(cc0.T @ cc0)
    a_ = U_[:, np.argmax(w_)]
    if (R0 @ a_)[2] < 0:
        a_ = -a_
    for _ in range(3):
        e1 = np.cross(a_, [1, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 1e-3 else np.cross(a_, [0, 1, 0])
        e1 /= np.linalg.norm(e1); e2 = np.cross(a_, e1)
        h = V @ a_; lo, hi = h.min(), h.max(); cents = []
        for s0 in np.linspace(lo + 0.1 * (hi - lo), hi - 0.15 * (hi - lo), 14):
            sel = np.abs(h - s0) < 0.004
            if sel.sum() < 50:
                continue
            x, y = V[sel] @ e1, V[sel] @ e2
            A_ = np.c_[2 * x, 2 * y, np.ones_like(x)]
            sol = np.linalg.lstsq(A_, x * x + y * y, rcond=None)[0]
            cents.append(sol[0] * e1 + sol[1] * e2 + s0 * a_)
        cents = np.array(cents); cc = cents.mean(0)
        w2, U2 = np.linalg.eigh((cents - cc).T @ (cents - cc))
        an = U2[:, np.argmax(w2)]; a_ = an if an @ a_ > 0 else -an
    h = (V - cc) @ a_
    bottom = cc + a_ * h.min()
    e1 = np.cross(a_, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(a_, e1)
    return dict(a=a_, b=bottom, H=float(h.max() - h.min()), e1=e1, e2=e2)


AX = axis_frame(V, Rs[0, iB])
HB, NPH = 0.005, 24


def cyl(Vb):
    d = Vb - AX["b"]
    h = d @ AX["a"]; x, y = d @ AX["e1"], d @ AX["e2"]
    return h, np.hypot(x, y), np.arctan2(y, x)


S = trimesh.sample.sample_surface(meshB, 300000, seed=11)[0]
h, r, ph = cyl(S)
nh = int(np.ceil(AX["H"] / HB)) + 1
ih = np.clip((h / HB).astype(int), 0, nh - 1); ip = ((ph + np.pi) / (2 * np.pi) * NPH).astype(int) % NPH
rmin = np.full((nh, NPH), np.nan); rmax = np.full((nh, NPH), np.nan)
for i in range(nh):
    for j in range(NPH):
        s = (ih == i) & (ip == j)
        if s.sum() >= 3:
            rmin[i, j] = r[s].min(); rmax[i, j] = np.percentile(r[s], 99)


def fill_rows(arr):
    arr = arr.copy()
    for j in range(arr.shape[1]):
        col = arr[:, j]; ok = ~np.isnan(col)
        if ok.any():
            idx = np.arange(len(col)); arr[:, j] = np.interp(idx, idx[ok], col[ok])
    return arr


rmin, rmax = fill_rows(rmin), fill_rows(rmax)


def interp_prof(arr, h, ph):
    nh_ = arr.shape[0]
    i = np.clip(np.asarray(h, float) / HB, 0, nh_ - 1 - 1e-6); j = (((np.asarray(ph, float) + np.pi) / (2 * np.pi) * NPH).astype(int)) % NPH
    i0 = np.floor(i).astype(int); fr_ = i - i0
    out = arr[i0, j] * (1 - fr_) + arr[np.minimum(i0 + 1, nh_ - 1), j] * fr_
    return np.where(np.isnan(out), np.nanmedian(arr), out)


def in_blue(Jw, Pb, Rb):
    loc = (Jw - Pb) @ Rb
    h_, r_, ph_ = cyl(loc)
    rin = interp_prof(rmin, h_, ph_); rout = interp_prof(rmax, h_, ph_)
    inside_h = (h_ > 0) & (h_ < AX["H"])
    return inside_h & (r_ < rout + 0.008), inside_h & (r_ < rin)


fres = {}
for lab, t, P_, R_ in (("ctl", tc, Pc, Rc), ("seat", ts, Ps, Rs)):
    JR = np.array(t["mano_right_joints"], float)
    JL = np.array(t["mano_left_joints"], float)
    env_fr, cav_fr, cnt = [], [], {}
    for f in range(55, 125):
        env, cav = in_blue(JR[f], P_[f, iB], R_[f, iB])
        if env.any():
            env_fr.append(f)
        if cav.any():
            cav_fr.append(f)
        if f % 5 == 0:
            cnt[f] = [int(env.sum()), int(cav.sum())]
    lcav = [f for f in range(19, 124) if in_blue(JL[f], P_[f, iB], R_[f, iB])[1].any()]
    fres[lab] = {"right_in_blue_envelope_frames": [env_fr[0], env_fr[-1], len(env_fr)] if env_fr else [],
                 "right_in_blue_envelope_list": env_fr,
                 "right_in_blue_cavity_frames": [cav_fr[0], cav_fr[-1], len(cav_fr)] if cav_fr else [],
                 "right_counts_every5_env_cav": cnt,
                 "left_in_blue_cavity_frames": [lcav[0], lcav[-1], len(lcav)] if lcav else []}
RES["f"] = fres
RES["blue_axis_height_cm"] = round(AX["H"] * 100, 2)

# ---------------- labelled hand-object distances (nearest MANO joint to the posed mesh surface)
meshG = trimesh.load(sorted(glob.glob(os.path.join(SEAT, "assets/human_motion_data/ego_recon/processed/*beige_cup.obj")))[0],
                     force="mesh", process=False)
pq_ = {iB: trimesh.proximity.ProximityQuery(meshB), iG: trimesh.proximity.ProximityQuery(meshG)}
dres = {}
for lab, t, P_, R_ in (("ctl", tc, Pc, Rc), ("seat", ts, Ps, Rs)):
    for side, obj, nm in (("left", iB, "blue"), ("right", iG, "beige"), ("right", iB, "blue"), ("left", iG, "beige")):
        J = np.array(t[f"mano_{side}_joints"], float)
        loc = np.einsum("fji,fkj->fki", R_[:, obj], J - P_[:, obj][:, None, :])
        dmin = np.array([np.abs(pq_[obj].signed_distance(loc[f])).min() for f in range(N)]) * 100
        dres[f"{lab}_{side}_{nm}_nearest_joint_cm_every5"] = {int(f): round(float(dmin[f]), 2) for f in range(0, N, 5)}
        dres[f"{lab}_{side}_{nm}_median_f19_123_cm"] = round(float(np.median(dmin[19:124])), 2)
        dres[f"{lab}_{side}_{nm}_series_cm"] = [round(float(x), 2) for x in dmin]
RES["hand_object_distance"] = dres

# ---------------- attach shift: cleaned minus merged wrist, camera frame (levelling-free), per hand
ash = {}
for lab, sc in (("ctl", CTL), ("seat", SEAT)):
    zm = np.load(os.path.join(sc, "bundle_merged/result.npz"), allow_pickle=True)
    zc = np.load(os.path.join(sc, "bundle_cleaned/result.npz"), allow_pickle=True)
    for side in ("left", "right"):
        d = np.linalg.norm(zc[f"hand_{side}_wrist_trans_in_camera"].astype(float) -
                           zm[f"hand_{side}_wrist_trans_in_camera"].astype(float), axis=1) * 100
        ash[f"{lab}_{side}_max_cm"] = round(float(d.max()), 2)
        ash[f"{lab}_{side}_argmax"] = int(d.argmax())
        ash[f"{lab}_{side}_f45_125_max_cm"] = round(float(d[45:126].max()), 2)
        ash[f"{lab}_{side}_outside_f45_125_max_cm"] = round(float(np.concatenate([d[:45], d[126:]]).max()), 2)
        ash[f"{lab}_{side}_every10_cm"] = {int(f): round(float(d[f]), 2) for f in range(0, N, 10)}
RES["attach_shift_cleaned_minus_merged_camera"] = ash
print("attach_shift", json.dumps({k: v for k, v in ash.items() if "every" not in k}))
json.dump(RES, open(OUTJ, "w"), indent=1)
print(json.dumps({k: RES[k] for k in ("a", "b", "c", "f")}, indent=1)[:12000])
print("contact_windows", json.dumps(cw))
print("e_from_parquet", json.dumps(RES["e_from_parquet"]))
for k in [k for k in dres if k.endswith("median_f19_123_cm")]:
    print(k, dres[k])
print("CHECK_DONE")
