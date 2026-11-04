"""ep5 depth checks B (GT-free, read-only; diag ep5_objA):
 1) floor plane vs table plane from FoundationStereo depth (is the 5.5 deg table tilt a world-gravity error or a depth artefact?)
 2) point-to-plane ICP of the real depth points of each resting object onto our reference mesh (rigid correction)
 3) our pot up-axis / spoon long-axis vs the depth table
usage: python -I dt5b.py <scene_dir> <kenc_episode_dir> <out_dir>
Depth PNG: metres = 65535 / raw - 1 (raw 65535 / 0 = hole)."""
import sys, os, glob, json, time, importlib.util
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

t0 = time.time()


def log(*a):
    print(f"[dt5b {time.time() - t0:6.1f}s]", *a, flush=True)


spec = importlib.util.spec_from_file_location("rr", "/home/haisenk/v2d_ops/kenc/review/tool/render_review.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)
scene, E, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
P = rr.load_scene_parquet(rr.find_parquet(scene))
N, K = P["N"], P["pos"].shape[1]
proc = os.path.dirname(os.path.dirname(os.path.dirname(P["path"])))
b = rr.load_bundle(os.path.join(scene, "bundle_cleaned", "result.npz"))
short = []
for n in P["body_names"]:
    m = [bn for bn in b["names"] if n.endswith("_" + bn)]
    short.append(m[0])
objs = []
for k in range(K):
    mp = P["mesh_paths"][k]
    mp2 = os.path.join(proc, os.path.basename(mp))
    mp = mp2 if os.path.isfile(mp2) else mp
    V, F = rr.load_obj(mp)
    R = rr.wxyz_to_R(P["wxyz"][:, k]).reshape(N, 3, 3)
    T = rr.to_T(R, P["pos"][:, k])
    lo, hi = V.min(0), V.max(0)
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    objs.append(dict(name=short[k], V=V, F=F, T=T, C=C))
A, res = rr.estimate_A([o["T"] for o in objs], [b["objT"][k] for k in range(K)], [o["C"] for o in objs])
c2w = A[None] @ b["c2w"]
Kf = np.asarray(b["K"][0], float)
H, W, S = 1520, 2028, 2
H2, W2 = (H + S - 1) // S, (W + S - 1) // S
K2 = Kf / S
vv, uu = np.mgrid[0:H2, 0:W2]
xn = (uu - K2[2]) / K2[0]
yn = (vv - K2[3]) / K2[1]
usd = sorted(glob.glob(os.path.join(scene, "assets/human_motion_data/ego_recon/reconstructed_stage/*.usda")))[0]
prims = rr.parse_usda(usd)
sup = max(prims, key=lambda p: p["area"])
z_sup = float(sup["top_z"])
sup_c = sup["pts"][:, :2].mean(0)
rng = np.random.default_rng(0)


def sample_surface(V, F, n):
    a, bb, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(bb - a, c - a)
    ar = 0.5 * np.linalg.norm(cr, axis=1)
    idx = rng.choice(len(F), n, p=ar / ar.sum())
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    Pm = (1 - r1)[:, None] * a[idx] + (r1 * (1 - r2))[:, None] * bb[idx] + (r1 * r2)[:, None] * c[idx]
    Nm = cr[idx] / np.maximum(np.linalg.norm(cr[idx], axis=1, keepdims=True), 1e-12)
    return Pm, Nm


for o in objs:
    o["Ps"], o["Ns"] = sample_surface(o["V"], o["F"], 400000)


def load_depth(f):
    raw = np.asarray(Image.open(f"{E}/white_pot/depth/{f:06d}.png")).astype(np.float64)[::S, ::S]
    ok = (raw > 0) & (raw < 65535)
    Z = np.where(ok, 65535.0 / np.maximum(raw, 1) - 1.0, np.nan)
    Z[(Z < 0.1) | (Z > 2.5)] = np.nan
    return Z


def load_mask(p):
    if not os.path.isfile(p):
        return np.zeros((H2, W2), bool)
    return np.asarray(Image.open(p).convert("L"))[::S, ::S] > 127


def backproject(Z, sel, f):
    Pc = np.stack([xn[sel] * Z[sel], yn[sel] * Z[sel], Z[sel]], 1)
    return Pc @ c2w[f, :3, :3].T + c2w[f, :3, 3]


def fit_plane(Pts, thr=0.006, iters=400, seed=0, nzmin=0.85):
    n = len(Pts)
    if n < 100:
        return None
    r_ = np.random.default_rng(seed)
    Ps = Pts if n <= 80000 else Pts[r_.choice(n, 80000, replace=False)]
    best_cnt, best = -1, None
    for _ in range(iters):
        s = Ps[r_.choice(len(Ps), 3, replace=False)]
        nv = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(nv)
        if nn < 1e-12:
            continue
        nv /= nn
        if abs(nv[2]) < nzmin:
            continue
        d = -nv @ s[0]
        cnt = int((np.abs(Ps @ nv + d) < thr).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (nv, d)
    if best is None:
        return None
    nv, d = best
    inl = np.abs(Pts @ nv + d) < thr
    for _ in range(4):
        Q = Pts[inl]
        M = np.c_[Q[:, 0], Q[:, 1], np.ones(len(Q))]
        coef, *_ = np.linalg.lstsq(M, Q[:, 2], rcond=None)
        r = Pts[:, 2] - (Pts[:, 0] * coef[0] + Pts[:, 1] * coef[1] + coef[2])
        inl = np.abs(r) < thr
    a, bq, c = [float(x) for x in coef]
    nrm = np.array([-a, -bq, 1.0])
    nrm /= np.linalg.norm(nrm)
    return dict(a=a, b=bq, c=c, normal=nrm.tolist(), tilt_deg=float(np.degrees(np.arctan(np.hypot(a, bq)))),
                ascent_az_deg=float(np.degrees(np.arctan2(bq, a))), n=int(n), inlier_frac=float(inl.mean()),
                resid_std_mm=float(r[inl].std() * 1000))


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


def masks_for(f):
    Z = load_depth(f)
    fin = np.isfinite(Z)
    mx = ndi.maximum_filter(np.where(fin, Z, -1.0), size=5)
    mn = ndi.minimum_filter(np.where(fin, Z, 99.0), size=5)
    edge = (mx - mn) > 0.015
    mk = {o["name"]: load_mask(f"{E}/{o['name']}/masks/1/{f:06d}.png") for o in objs}
    hand = np.zeros((H2, W2), bool)
    for o in objs:
        for h in (2, 3):
            hand |= load_mask(f"{E}/{o['name']}/masks/{h}/{f:06d}.png")
    return Z, fin, edge, mk, hand


frames0 = list(range(0, 25, 2))
framesE = list(range(272, 288))
out = dict(scene=scene)
cam = {}
for f in (0, 12, 280):
    fwd = c2w[f, :3, :3] @ np.array([0, 0, 1.0])
    cam[f] = dict(pitch_deg=float(np.degrees(np.arcsin(fwd[2]))), heading_az_deg=float(np.degrees(np.arctan2(fwd[1], fwd[0]))),
                  right_axis_elev_deg=float(np.degrees(np.arcsin((c2w[f, :3, :3] @ np.array([1.0, 0, 0]))[2]))),
                  pos=c2w[f, :3, 3].round(4).tolist())
out["camera"] = cam
log("camera", cam)

# ---------------- 1) table vs floor planes
for tag, fl in (("start", frames0), ("end", framesE)):
    tab, flo = [], []
    for f in fl:
        Z, fin, edge, mk, hand = masks_for(f)
        objm = np.zeros((H2, W2), bool)
        for v in mk.values():
            objm |= v
        d_obj = ndi.distance_transform_edt(~objm) if objm.any() else np.full((H2, W2), 1e9)
        d_hand = ndi.distance_transform_edt(~hand) if hand.any() else np.full((H2, W2), 1e9)
        cand = fin & ~edge & (d_obj > 8) & (d_hand > 25)
        Pw = backproject(Z, cand, f)
        rr_ = np.linalg.norm(Pw[:, :2] - sup_c, axis=1)
        t_ = Pw[(np.abs(Pw[:, 2] - z_sup) < 0.10) & (rr_ < 0.45)]
        fl_ = Pw[(Pw[:, 2] < z_sup - 0.40) & (Pw[:, 2] > z_sup - 1.40)]
        g = np.random.default_rng(f)
        tab.append(t_ if len(t_) <= 20000 else t_[g.choice(len(t_), 20000, replace=False)])
        flo.append(fl_ if len(fl_) <= 20000 else fl_[g.choice(len(fl_), 20000, replace=False)])
    T_ = np.concatenate(tab)
    F_ = np.concatenate(flo)
    pt = fit_plane(T_, seed=1)
    pf = fit_plane(F_, thr=0.015, seed=2, nzmin=0.7)
    rec = dict(table=pt, floor=pf, n_floor_pts=int(len(F_)))
    if len(F_):
        rec["floor_z_p5_p95"] = [float(np.percentile(F_[:, 2], 5)), float(np.percentile(F_[:, 2], 95))]
    if pt and pf:
        rec["angle_table_vs_floor_deg"] = ang(np.array(pt["normal"]), np.array(pf["normal"]))
        rec["table_minus_floor_height_m"] = float(pt["c"] + pt["a"] * sup_c[0] + pt["b"] * sup_c[1]
                                                  - (pf["c"] + pf["a"] * sup_c[0] + pf["b"] * sup_c[1]))
        r = F_[:, 2] - (F_[:, 0] * pf["a"] + F_[:, 1] * pf["b"] + pf["c"])
        dist = np.linalg.norm(F_ - c2w[fl[0], :3, 3], axis=1)
        bins = np.percentile(dist, [0, 20, 40, 60, 80, 100])
        rec["floor_resid_by_dist_cm"] = [[round(float(bins[i]), 2), round(float(bins[i + 1]), 2),
                                          round(float(np.median(r[(dist >= bins[i]) & (dist < bins[i + 1])]) * 100), 2)] for i in range(5)]
        rt = T_[:, 2] - (T_[:, 0] * pt["a"] + T_[:, 1] * pt["b"] + pt["c"])
        distt = np.linalg.norm(T_ - c2w[fl[0], :3, 3], axis=1)
        bt = np.percentile(distt, [0, 20, 40, 60, 80, 100])
        rec["table_resid_by_dist_cm"] = [[round(float(bt[i]), 2), round(float(bt[i + 1]), 2),
                                          round(float(np.median(rt[(distt >= bt[i]) & (distt < bt[i + 1])]) * 100), 2)] for i in range(5)]
        # table points: height (vs fitted plane is ~0) -- also height vs SIM plane as a function of distance from camera
        rs = T_[:, 2] - z_sup
        rec["table_height_vs_sim_by_dist_cm"] = [[round(float(bt[i]), 2), round(float(bt[i + 1]), 2),
                                                  round(float(np.median(rs[(distt >= bt[i]) & (distt < bt[i + 1])]) * 100), 2)] for i in range(5)]
    out[f"planes_{tag}"] = rec
    log(tag, "table", None if not pt else (round(pt["tilt_deg"], 2), round(pt["ascent_az_deg"], 1), round(pt["resid_std_mm"], 1), round(pt["inlier_frac"], 2)),
        "floor", None if not pf else (round(pf["tilt_deg"], 2), round(pf["ascent_az_deg"], 1), round(pf["resid_std_mm"], 1), round(pf["inlier_frac"], 2), pf["n"]),
        "angle", rec.get("angle_table_vs_floor_deg"), "height", rec.get("table_minus_floor_height_m"), "floor z p5/p95", rec.get("floor_z_p5_p95"))
    log("   floor resid by dist", rec.get("floor_resid_by_dist_cm"))
    log("   table resid by dist", rec.get("table_resid_by_dist_cm"))
    log("   table height vs sim by dist", rec.get("table_height_vs_sim_by_dist_cm"))


# ---------------- 2) ICP of real depth points onto our mesh (object resting phases)
def icp(Preal, Pm, Nm, iters=40, trim=0.8):
    cen = Pm.mean(0)
    Rc, tc = np.eye(3), np.zeros(3)
    hist = []
    for it in range(iters):
        Pq = (Pm - cen) @ Rc.T + cen + tc
        Nq = Nm @ Rc.T
        d, ii = cKDTree(Pq).query(Preal)
        q, nq = Pq[ii], Nq[ii]
        r = np.einsum("ij,ij->i", Preal - q, nq)
        thr = np.quantile(np.abs(r), trim)
        keep = (np.abs(r) <= thr) & (d < 0.03)
        hist.append(float(np.median(np.abs(r))))
        c_ = cen + tc
        J = np.c_[np.cross(q[keep] - c_, nq[keep]), nq[keep]]
        x, *_ = np.linalg.lstsq(J, r[keep], rcond=None)
        dR = Rot.from_rotvec(x[:3]).as_matrix()
        Rc = dR @ Rc
        tc = (dR @ (cen + tc - c_) + c_ + x[3:]) - cen
        if np.linalg.norm(x[:3]) < 1e-5 and np.linalg.norm(x[3:]) < 1e-5:
            break
    Pq = (Pm - cen) @ Rc.T + cen + tc
    Nq = Nm @ Rc.T
    d, ii = cKDTree(Pq).query(Preal)
    r = np.einsum("ij,ij->i", Preal - Pq[ii], Nq[ii])
    return Rc, tc, cen, hist, float(np.median(np.abs(r)))


icp_out = {}
for tag, fl in (("start", frames0), ("end", framesE)):
    cache = {f: masks_for(f) for f in fl}
    for k, o in enumerate(objs):
        nm = o["name"]
        Tref = o["T"][fl[0]]
        drift = max(float(np.linalg.norm(o["T"][f][:3, 3] - Tref[:3, 3])) for f in fl)
        Preal_all = []
        for f in fl:
            Z, fin, edge, mk, hand = cache[f]
            other = [oo for oo in objs if oo["name"] != nm][0]
            d_in = ndi.distance_transform_edt(mk[nm]) if mk[nm].any() else np.zeros((H2, W2))
            d_hand = ndi.distance_transform_edt(~hand) if hand.any() else np.full((H2, W2), 1e9)
            d_oth = ndi.distance_transform_edt(~mk[other["name"]]) if mk[other["name"]].any() else np.full((H2, W2), 1e9)
            sel = (d_in > 3) & fin & ~edge & (d_hand > 10) & (d_oth > 4)
            Pr = backproject(Z, sel, f)
            if len(Pr) > 4000:
                Pr = Pr[np.random.default_rng(f).choice(len(Pr), 4000, replace=False)]
            Tf = o["T"][f]
            Pb = (Pr - Tf[:3, 3]) @ Tf[:3, :3]
            Preal_all.append(Pb @ Tref[:3, :3].T + Tref[:3, 3])
        Preal = np.concatenate(Preal_all)
        Pm = o["Ps"] @ Tref[:3, :3].T + Tref[:3, 3]
        Nm = o["Ns"] @ Tref[:3, :3].T
        Rc, tc, cen, hist, med_after = icp(Preal, Pm, Nm)
        rv = Rot.from_matrix(Rc).as_rotvec()
        angle = float(np.degrees(np.linalg.norm(rv)))
        axis = (rv / max(np.linalg.norm(rv), 1e-12)).tolist()
        Vw = o["V"] @ Tref[:3, :3].T + Tref[:3, 3]
        Vc = (Vw - cen) @ Rc.T + cen + tc
        rec = dict(frames=[fl[0], fl[-1]], ref_drift_m=drift, n_pts=int(len(Preal)), rot_deg=angle, rot_axis_world=axis,
                   tilt_change_deg=float(np.degrees(np.arccos(np.clip((Rc @ np.array([0, 0, 1.0]))[2], -1, 1)))),
                   trans_cm=(tc * 100).round(2).tolist(), resid_med_before_mm=hist[0] * 1000, resid_med_after_mm=med_after * 1000,
                   iters=len(hist), bottom_z_before=float(Vw[:, 2].min()), bottom_z_after=float(Vc[:, 2].min()),
                   centroid_shift_cm=((Vc.mean(0) - Vw.mean(0)) * 100).round(2).tolist())
        icp_out[f"{nm}_{tag}"] = rec
        log(f"ICP {nm} {tag}: n {len(Preal)} rot {angle:.2f} deg axis {np.round(axis, 2)} tilt change {rec['tilt_change_deg']:.2f} deg, "
            f"trans {rec['trans_cm']} cm, centroid shift {rec['centroid_shift_cm']} cm, resid med {rec['resid_med_before_mm']:.1f} -> "
            f"{rec['resid_med_after_mm']:.1f} mm, bottom z {rec['bottom_z_before']:.4f} -> {rec['bottom_z_after']:.4f}")
out["icp"] = icp_out

# ---------------- 3) object axes vs world z / depth table normal
ax = {}
for tag, fl in (("start", frames0), ("end", framesE)):
    pt = out[f"planes_{tag}"]["table"]
    n_t = np.array(pt["normal"])
    for o in objs:
        T0 = o["T"][fl[0]]
        Vw = o["V"] @ T0[:3, :3].T + T0[:3, 3]
        cen = Vw.mean(0)
        E_, U_ = np.linalg.eigh(np.cov((Vw - cen).T))
        long_, short_ = U_[:, -1], U_[:, 0]
        h = long_.copy()
        h[2] = 0
        h /= np.linalg.norm(h)
        slope_along = float(np.degrees(np.arctan(-(n_t[0] * h[0] + n_t[1] * h[1]) / n_t[2])))
        s = (Vw - cen) @ long_
        ends = {}
        for nmend, selx in (("end_neg", s < np.percentile(s, 8)), ("end_pos", s > np.percentile(s, 92))):
            Pend = Vw[selx]
            zt = pt["a"] * Pend[:, 0] + pt["b"] * Pend[:, 1] + pt["c"]
            ends[nmend] = dict(min_height_above_depth_table_cm=float((Pend[:, 2] - zt).min() * 100),
                               min_height_above_sim_cm=float((Pend[:, 2].min() - z_sup) * 100),
                               xy=Pend[:, :2].mean(0).round(3).tolist(), along_axis_cm=float(s[selx].mean() * 100))
        ax[f"{o['name']}_{tag}"] = dict(long_axis_elev_deg=float(np.degrees(np.arcsin(long_[2]))),
                                        long_axis_az_deg=float(np.degrees(np.arctan2(long_[1], long_[0]))),
                                        depth_table_slope_along_long_axis_deg=slope_along,
                                        short_axis_angle_to_world_z_deg=min(ang(short_, np.array([0, 0, 1.0])), ang(-short_, np.array([0, 0, 1.0]))),
                                        short_axis_angle_to_depth_table_normal_deg=min(ang(short_, n_t), ang(-short_, n_t)),
                                        ends=ends)
        log(f"axes {o['name']} {tag}:", json.dumps(ax[f"{o['name']}_{tag}"]))
out["axes"] = ax
with open(os.path.join(OUT, "ep5_depthB.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log("done")

