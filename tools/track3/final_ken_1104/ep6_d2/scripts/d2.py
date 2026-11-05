#!/usr/bin/env python3
"""refine1104 ep6 round 2 (D2): data-driven de-smoothing of bob1's in-pot spoon (rule + gates pre-registered in
../gates.json, written 12:22 EST before this script existed).  bob1's spoon mesh and its levelling are kept: only a
world-frame rigid correction is left-multiplied onto bob1's reference poses on f36-171.
Steps: Ken's raw-vs-smoothed FoundationPose residual -> flip rejection (> 30 deg) -> Gaussian sigma s -> per-frame choice
between the corrected pose and bob1's pose by the round-1 mask/depth cost -> choice smoothed (sigma s) -> 6-frame ramps.
Gates computed here: G1 (primary parity CV, mesh-relative), G1s (strict parity CV), G2 (depth), G3 (steps),
G4 (kit-registration roll consistency + SR margins), G5 (raw right hand).  Read-only on every input; writes only
into WD/out."""
import importlib.util
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

WD = "/data/user_data/haisenk/v2d/refine1104/ep6_d2/work"
OUT = os.path.join(WD, "out")
os.makedirs(OUT, exist_ok=True)
sys.argv = ["fit6.py", OUT, "none"]
sys.path.insert(0, WD)
import fit6 as F6  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter1d  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

T0 = time.time()
sp, pot = F6.objs[1], F6.objs[0]
K, w2c, S, N = F6.K, F6.w2c, F6.S, F6.N
c2w = np.linalg.inv(w2c)
W = np.arange(42, 166)
REST_FR = list(range(0, 25)) + list(range(220, 243))
RAMP_FR = list(range(36, 42)) + list(range(166, 172))
SIGMAS = (1.0, 1.5, 2.0)
FLIP_DEG = 30.0
IDX = np.arange(N)


def log(m):
    print(f"[d2 +{time.time() - T0:6.1f}s] {m}", flush=True)


# spoon long axis (body frame), bowl end
Vs = sp["V"]
MU = Vs.mean(0)
_, EVEC = np.linalg.eigh(np.cov((Vs - MU).T))
AX1, AX2 = EVEC[:, 2], EVEC[:, 1]
_s = (Vs - MU) @ AX1
BOWL = 1.0 if np.ptp(((Vs - MU) @ AX2)[_s > 0.5 * _s.max()]) > np.ptp(((Vs - MU) @ AX2)[_s < 0.5 * _s.min()]) else -1.0


def metrics(f, Tc):
    """round-1 cost_at (cost, IoU_occ, median FS-depth residual cm) + D3 depth-gradient slope + visible centroid."""
    Ms, Hm, D = F6.frame_data(f)
    M = Ms[1]
    O = Hm | Ms[0]
    sil, zb = F6.render(sp, Tc, K[f], need_depth=True)
    inter = (sil & M).sum()
    uo = (M | (sil & ~O)).sum()
    iou = inter / uo if uo else 0.0
    valid = (M & sil) & np.isfinite(D) & np.isfinite(zb)
    c = 1.0 - iou
    dm = float("nan")
    if valid.sum() > 30:
        dm = float(np.median(D[valid] - zb[valid]) * 100)
        c += 0.04 * min(abs(dm), 6.0)
    slope = float("nan")
    z_obj = F6.obj_depth(sp, Tc)
    pxcm = S * z_obj / K[f][0] * 100.0
    dh = distance_transform_edt(~Hm) * pxcm if Hm.any() else np.full(M.shape, 1e9)
    vv = binary_erosion(M & sil, iterations=1) & np.isfinite(D) & np.isfinite(zb) & (dh >= 1.5)
    ys, xs = np.nonzero(vv)
    if len(ys) >= 60:
        u = (xs + 0.5) * S - 0.5
        v = (ys + 0.5) * S - 0.5
        zz = zb[ys, xs]
        Kf = K[f]
        Pc = np.stack([(u - Kf[2]) / Kf[0] * zz, (v - Kf[3]) / Kf[1] * zz, zz], 1)
        body = (Pc - Tc[:3, 3]) @ Tc[:3, :3]
        s = ((body - MU) @ AX1) * BOWL * 100.0
        r = (D[ys, xs] - zz) * 100.0
        bins = np.arange(np.floor(s.min() / 2) * 2, s.max() + 2, 2.0)
        bc, bmed = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = (s >= a) & (s < b)
            if sel.sum() >= 15:
                bc.append(0.5 * (a + b))
                bmed.append(float(np.median(r[sel])))
        if len(bc) >= 3:
            slope = float(np.polyfit(bc, bmed, 1)[0])
    vis = sil & ~O
    yv, xv = np.nonzero(vis)
    vc = [float(xv.mean() * S), float(yv.mean() * S)] if len(xv) else [float("nan")] * 2
    return [float(c), float(iou), dm, slope, vc[0], vc[1]]


def frame_task(args):
    f, poses = args
    Ms, Hm, D = F6.frame_data(f)
    M = Ms[1] & ~Hm
    ys, xs = np.nonzero(M)
    out = {"_mask": [float(xs.mean() * S), float(ys.mean() * S), int(len(xs))] if len(xs) else None}
    for key, Tw in poses.items():
        out[key] = metrics(f, w2c[f] @ np.asarray(Tw))
    return f, out


def run_requests(req):
    jobs = [(f, {k: np.asarray(v).tolist() for k, v in d.items()}) for f, d in sorted(req.items()) if d]
    with Pool(8) as pool:
        return dict(pool.map(frame_task, jobs, chunksize=2))


# ---------------------------------------------------------------- smoothing residual from Ken's FP run
rows = json.load(open(os.path.join(WD, "d4_rawfp.json")))["rows"]
Tr = np.array([r["Tr"] for r in rows])
Ts = np.array([r["Ts"] for r in rows])
kV, kF = F6.rr.load_obj(os.path.join(F6.E, "wooden_spoon", "result", "mesh.obj"))
MUK = kV.mean(0)
RV = np.zeros((N, 3))
DL = np.zeros((N, 3))
PV = np.zeros((N, 3))
ANG = np.zeros(N)
for f in range(N):
    Rd = Tr[f][:3, :3] @ Ts[f][:3, :3].T
    cs = Ts[f][:3, :3] @ MUK + Ts[f][:3, 3]
    cr = Tr[f][:3, :3] @ MUK + Tr[f][:3, 3]
    Rc = c2w[f][:3, :3]
    RV[f] = Rot.from_matrix(Rc @ Rd @ Rc.T).as_rotvec()
    DL[f] = Rc @ (cr - cs)
    PV[f] = Rc @ cs + c2w[f][:3, 3]
    ANG[f] = np.degrees(np.linalg.norm(RV[f]))
GOOD = ANG <= FLIP_DEG
log(f"loaded; flip frames (> {FLIP_DEG} deg) in W: {IDX[W][~GOOD[W]].tolist()}; all: {int((~GOOD).sum())}")


def build_corr(sig, use=None):
    good = GOOD.copy() if use is None else (GOOD & use)
    r, d = RV.copy(), DL.copy()
    for arr in (r, d):
        for k in range(3):
            arr[~good, k] = np.interp(IDX[~good], IDX[good], arr[good, k])
    return gaussian_filter1d(r, sig, axis=0, mode="nearest"), gaussian_filter1d(d, sig, axis=0, mode="nearest")


def apply_corr(Tw, w, r, d, p):
    Rd = Rot.from_rotvec(w * r).as_matrix()
    T = Tw.copy()
    T[:3, :3] = Rd @ Tw[:3, :3]
    T[:3, 3] = Rd @ (Tw[:3, 3] - p) + p + w * d
    return T


def full_weight(wW):
    w = np.zeros(N)
    w[42:166] = wW
    for j in range(1, 7):
        w[42 - j] = w[42] * (1.0 - j / 7.0)
        w[165 + j] = w[165] * (1.0 - j / 7.0)
    return w


def track(w, r, d):
    Tw = sp["T"].copy()
    for f in range(N):
        if w[f] > 0:
            Tw[f] = apply_corr(sp["T"][f], w[f], r[f], d[f], PV[f])
    return Tw


def steps(Tw):
    rs = np.array([np.degrees(Rot.from_matrix(Tw[i][:3, :3].T @ Tw[i + 1][:3, :3]).magnitude()) for i in range(N - 1)])
    cen = np.einsum("nij,j->ni", Tw[:, :3, :3], MU) + Tw[:, :3, 3]
    ts = np.linalg.norm(np.diff(cen, axis=0), axis=1) * 100
    return rs, ts


def step_gate(Tw):
    rs, ts = steps(Tw)
    a, b = rs[35:172], ts[35:172]
    o = dict(rot_max=round(float(a.max()), 2), rot_argmax=int(35 + a.argmax()), rot_p90=round(float(np.percentile(a, 90)), 2),
             trans_max_cm=round(float(b.max()), 3), trans_argmax=int(35 + b.argmax()), trans_p90_cm=round(float(np.percentile(b, 90)), 3))
    o["pass"] = bool(a.max() <= 15.0 and np.percentile(a, 90) <= 10.6 and b.max() <= 3.0 and np.percentile(b, 90) <= 1.63)
    return o


def twist_deg(Rrel, axis):
    q = Rot.from_matrix(Rrel).as_quat()  # x y z w
    q = q * np.sign(q[..., 3:4] + 1e-15)
    proj = q[..., :3] @ axis
    return np.degrees(2.0 * np.arctan2(proj, q[..., 3]))


if __name__ == "__main__":
    # ------------------------------------------------ stage 1: reference + candidates
    CORR = {}
    req = {f: {} for f in range(N)}
    for f in list(W) + REST_FR + RAMP_FR:
        req[f]["ref"] = sp["T"][f]
    for s in SIGMAS:
        CORR[("full", s)] = build_corr(s)
        for P in (0, 1):
            CORR[("strict", s, P)] = build_corr(s, use=(IDX % 2 == P))
        r, d = CORR[("full", s)]
        for f in W:
            req[f][f"cand_full_{s}"] = apply_corr(sp["T"][f], 1.0, r[f], d[f], PV[f])
            rs_, ds_ = CORR[("strict", s, f % 2)]
            req[f][f"cand_strict_{s}"] = apply_corr(sp["T"][f], 1.0, rs_[f], ds_[f], PV[f])
    res = run_requests(req)
    log(f"stage 1 done ({sum(len(v) for v in req.values())} renders)")
    REST = float(np.mean([res[f]["ref"][1] for f in REST_FR]))
    rest_parts = [round(float(np.mean([res[f]["ref"][1] for f in range(0, 25)])), 4),
                  round(float(np.mean([res[f]["ref"][1] for f in range(220, 243)])), 4)]
    log(f"REST (bob1 mesh own rest-phase IoU_occ) = {REST:.4f} {rest_parts}; bar 0.85*REST = {0.85 * REST:.4f}")

    # ------------------------------------------------ stage 2: tracks (full, primary CV, strict CV)
    req2 = {f: {} for f in range(N)}
    TR = {}
    for s in SIGMAS:
        r, d = CORR[("full", s)]
        cf = np.array([1.0 if res[f][f"cand_full_{s}"][0] < res[f]["ref"][0] else 0.0 for f in W])
        wf = full_weight(gaussian_filter1d(cf, s, mode="nearest"))
        Tf = track(wf, r, d)
        TR[("full", s)] = (Tf, wf, cf)
        for f in range(36, 172):
            req2[f][f"full_{s}"] = Tf[f]
        for P in (0, 1):
            trm = (W % 2) == P
            cI = np.interp(W, W[trm], cf[trm])
            wP = full_weight(gaussian_filter1d(cI, s, mode="nearest"))
            TP = track(wP, r, d)
            for f in W[~trm]:
                req2[f][f"cv_{s}"] = TP[f]
            rs_, ds_ = CORR[("strict", s, P)]
            ct = np.array([1.0 if res[f][f"cand_strict_{s}"][0] < res[f]["ref"][0] else 0.0 for f in W[trm]])
            cS = np.interp(W, W[trm], ct)
            wS = full_weight(gaussian_filter1d(cS, s, mode="nearest"))
            TS = track(wS, rs_, ds_)
            for f in W[~trm]:
                req2[f][f"strict_{s}"] = TS[f]
    res2 = run_requests(req2)
    log(f"stage 2 done ({sum(len(v) for v in req2.values())} renders)")

    def g1(key, src=res2):
        io_n = np.array([src[f][key][1] for f in W])
        io_o = np.array([res[f]["ref"][1] for f in W])
        de_n = np.array([src[f][key][2] for f in W])
        de_o = np.array([res[f]["ref"][2] for f in W])
        sl_n = np.array([src[f][key][3] for f in W])
        o = dict(n=len(W), iou_occ_mean_bob1=round(float(io_o.mean()), 4), iou_occ_mean_new=round(float(io_n.mean()), 4),
                 gain=round(float(io_n.mean() - io_o.mean()), 4), frac_worse_gt_0p05=round(float(np.mean(io_n < io_o - 0.05)), 4),
                 absdep_median_bob1_cm=round(float(np.nanmedian(np.abs(de_o))), 3), absdep_median_new_cm=round(float(np.nanmedian(np.abs(de_n))), 3),
                 slope_median_bob1=round(float(np.nanmedian([res[f]["ref"][3] for f in W])), 4), slope_median_new=round(float(np.nanmedian(sl_n)), 4),
                 mean_cost=round(float(np.mean([src[f][key][0] for f in W])), 5),
                 mean_cost_bob1=round(float(np.mean([res[f]["ref"][0] for f in W])), 5),
                 ratio_to_rest=round(float(io_n.mean() / REST), 4),
                 per_block_iou={f"f{a}-{b - 1}": [round(float(io_o[a - 42:b - 42].mean()), 3), round(float(io_n[a - 42:b - 42].mean()), 3)]
                                for a, b in ((42, 70), (70, 100), (100, 130), (130, 166))})
        o["absdep_drop_cm"] = round(o["absdep_median_bob1_cm"] - o["absdep_median_new_cm"], 3)
        return o

    per_sigma = {}
    for s in SIGMAS:
        Tf, wf, cf = TR[("full", s)]
        cv = g1(f"cv_{s}")
        st = g1(f"strict_{s}")
        ins = g1(f"full_{s}")
        per_sigma[s] = dict(steps=step_gate(Tf), cv_primary=cv, cv_strict=st, insample_full=ins,
                            choice_frac=round(float(cf.mean()), 3), weight_mean_W=round(float(wf[42:166].mean()), 3),
                            n_changed=int((wf > 0).sum()))
        log(f"sigma {s}: steps {per_sigma[s]['steps']} | CV {cv['iou_occ_mean_new']} (bob1 {cv['iou_occ_mean_bob1']}, gain {cv['gain']}, worse {cv['frac_worse_gt_0p05']}, "
            f"dep {cv['absdep_median_bob1_cm']}->{cv['absdep_median_new_cm']}, cost {cv['mean_cost']}) | strict {st['iou_occ_mean_new']} gain {st['gain']} worse {st['frac_worse_gt_0p05']} | "
            f"in-sample {ins['iou_occ_mean_new']} | choice {cf.mean():.2f}")
    passing = [s for s in SIGMAS if per_sigma[s]["steps"]["pass"]]
    if passing:
        s_star = sorted(passing, key=lambda s: (per_sigma[s]["cv_primary"]["mean_cost"], -s))[0]
    else:
        s_star = sorted(SIGMAS, key=lambda s: (per_sigma[s]["cv_primary"]["mean_cost"], -s))[0]
    log(f"step-passing sigmas {passing}; chosen sigma {s_star}")
    P_ = per_sigma[s_star]
    gates = {}
    cv = P_["cv_primary"]
    gates["G1"] = dict(cv, REST=round(REST, 4), REST_parts=rest_parts, bar=round(0.85 * REST, 4),
                       **{"pass": bool(cv["iou_occ_mean_new"] >= 0.85 * REST and cv["gain"] >= 0.05 and cv["frac_worse_gt_0p05"] < 0.20)})
    st = P_["cv_strict"]
    gates["G1s"] = dict(st, **{"pass": bool(st["gain"] >= 0.03 and st["frac_worse_gt_0p05"] < 0.25)})
    gates["G2"] = dict(absdep_median_bob1_cm=cv["absdep_median_bob1_cm"], absdep_median_new_cm=cv["absdep_median_new_cm"],
                       drop_cm=cv["absdep_drop_cm"], **{"pass": bool(cv["absdep_drop_cm"] >= 0.5)})
    gates["G3"] = dict(P_["steps"], sigma_passing=passing, bob1=step_gate(sp["T"]))
    gates["G3"]["pass"] = bool(s_star in passing)

    # ------------------------------------------------ G4 kit registration + roll / SR margins
    Tf, wf, cf = TR[("full", s_star)]
    g4 = {}
    try:
        spec = importlib.util.spec_from_file_location("kitauc", os.path.join(WD, "kit_AUC.py"))
        kit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kit)
        mp = os.path.join(F6.SCENE, "score_input", "meshes", "episode_000006", "wooden_spoon.obj")
        sv, sf = F6.rr.load_obj(mp)
        seed = 1_000_003 * 6 + 101
        cc, cr = kit._surface_moments(sv, sf)
        anchor, Rreg = kit._register_object_body((sv, sf), (sv, sf), cc, seed)
        g4["a_kit_self_registration_deg"] = round(float(np.degrees(Rot.from_matrix(Rreg).magnitude())), 4)
        g4["a_anchor_shift_mm"] = round(float(np.linalg.norm(anchor - cc) * 1000), 4)
        # info only: two different surface samplings of the same mesh (is the registration roll-stable?)
        cp = (kit._sample_mesh_surface(sv, sf, 3000, seed) - cc)
        rp = kit._sample_mesh_surface(sv, sf, 3000, seed + 1) - cc
        R2, _ = kit._align_rigid(rp, cp)
        g4["info_resampled_registration_deg"] = round(float(np.degrees(Rot.from_matrix(R2).magnitude())), 3)
        g4["score_mesh"] = mp
        g4["score_mesh_nV"] = int(len(sv))
    except Exception as e:  # noqa: BLE001
        g4["a_error"] = repr(e)
    Rrel = np.einsum("nji,njk->nik", sp["T"][:, :3, :3], Tf[:, :3, :3])
    angs = np.degrees(Rot.from_matrix(Rrel).magnitude())
    roll = np.abs(twist_deg(Rrel, AX1))
    cen_o = np.einsum("nij,j->ni", sp["T"][:, :3, :3], MU) + sp["T"][:, :3, 3]
    cen_n = np.einsum("nij,j->ni", Tf[:, :3, :3], MU) + Tf[:, :3, 3]
    dc = np.linalg.norm(cen_n - cen_o, axis=1) * 100
    g4.update(b_rot_max_deg=round(float(angs.max()), 2), b_rot_argmax=int(angs.argmax()), b_roll_max_deg=round(float(roll.max()), 2),
              b_roll_argmax=int(roll.argmax()), c_rot_mean_deg_all243=round(float(angs.mean()), 3),
              c_centroid_mean_cm_all243=round(float(dc.mean()), 3), info_rot_median_in_W=round(float(np.median(angs[42:166])), 2),
              info_centroid_median_in_W_cm=round(float(np.median(dc[42:166])), 3), info_centroid_max_cm=round(float(dc.max()), 3))
    g4["pass"] = bool(g4.get("a_kit_self_registration_deg", 99) <= 1.0 and angs.max() <= 30.0 and roll.max() <= 30.0
                      and angs.mean() <= 10.0 and dc.mean() <= 2.0)
    gates["G4"] = g4

    # ------------------------------------------------ G5 raw right hand
    z = np.load(os.path.join(WD, "f1d_joints_cam.npz"))
    bc = np.load(os.path.join(F6.SCENE, "bundle_cleaned", "result.npz"))
    Cw = F6.A[None] @ bc["camera_to_world_transform"].astype(np.float64)
    Jm = np.einsum("nij,nkj->nki", Cw[:, :3, :3], z["bundle_merged_joints_cam"]) + Cw[:, None, :3, 3]
    vm = z["bundle_merged_valid"]
    smp = sp["samples"][::4]
    dist = {}
    for f in range(N):
        if not (vm[f] and wf[f] > 0):
            continue
        r_old = cKDTree(smp @ sp["T"][f][:3, :3].T + sp["T"][f][:3, 3]).query(Jm[f])[0] * 100
        r_new = cKDTree(smp @ Tf[f][:3, :3].T + Tf[f][:3, 3]).query(Jm[f])[0] * 100
        dist[f] = (float(r_old.min()), float(r_new.min()))

    def bst(fs):
        fs = [f for f in fs if f in dist]
        if not fs:
            return None
        a = np.array([dist[f] for f in fs])
        return dict(n=len(fs), min_old=round(float(np.median(a[:, 0])), 3), min_new=round(float(np.median(a[:, 1])), 3),
                    frac_closer=round(float(np.mean(a[:, 1] < a[:, 0])), 3))
    g5 = dict(changed=bst(sorted(dist)), windows={f"f{a}-{b - 1}": bst(range(a, b)) for a, b in ((36, 70), (70, 100), (100, 130), (130, 172))})
    g5["pass"] = bool(g5["changed"] is not None and g5["changed"]["min_new"] <= g5["changed"]["min_old"] + 0.05)
    gates["G5"] = g5
    all_pass = all(gates[k]["pass"] for k in ("G1", "G1s", "G2", "G3", "G4", "G5"))

    # ------------------------------------------------ info: video motion agreement (mask centroid vs visible silhouette centroid)
    def motion_corr(key_of):
        a, b = [], []
        for f in range(42, 165):
            m0, m1 = res[f]["_mask"], res[f + 1]["_mask"]
            p0, p1 = key_of(f), key_of(f + 1)
            if m0 is None or m1 is None or p0 is None or p1 is None:
                continue
            if not all(np.isfinite(p0[4:6])) or not all(np.isfinite(p1[4:6])):
                continue
            a.append([m1[0] - m0[0], m1[1] - m0[1]])
            b.append([p1[4] - p0[4], p1[5] - p0[5]])
        a, b = np.array(a), np.array(b)
        cx = float(np.corrcoef(a[:, 0], b[:, 0])[0, 1])
        cy = float(np.corrcoef(a[:, 1], b[:, 1])[0, 1])
        return dict(n=len(a), corr_dx=round(cx, 3), corr_dy=round(cy, 3), mask_step_px_median=round(float(np.median(np.linalg.norm(a, axis=1))), 2),
                    pose_step_px_median=round(float(np.median(np.linalg.norm(b, axis=1))), 2))
    info_motion = dict(bob1=motion_corr(lambda f: res[f]["ref"]),
                       d2_insample=motion_corr(lambda f: res2[f].get(f"full_{s_star}")))

    # ------------------------------------------------ targets in the bundle_cleaned frame (for the H5_PATCH build)
    A = F6.A
    Ai = np.linalg.inv(A)
    Tb_ref = bc["object_1_to_world_transform"].astype(np.float64)
    map_res = float(np.abs(A[None] @ Tb_ref - sp["T"]).max())
    Tb_new = Tb_ref.copy()
    mask = wf > 0
    for f in np.flatnonzero(mask):
        Csc = Tf[f] @ np.linalg.inv(sp["T"][f])
        Tb_new[f] = (Ai @ Csc @ A) @ Tb_ref[f]
    chk = float(np.abs(A[None] @ Tb_new - Tf)[mask].max()) if mask.any() else 0.0
    np.savez(os.path.join(OUT, "d2_targets.npz"), object_1_to_world_transform=Tb_new, object_1_mask=mask,
             sigma=np.float64(s_star), weight=wf, choice_W=cf, scene_world_track=Tf, bob1_scene_world_track=sp["T"], A=A)
    # judge pseudo-truth (round-1 V1 construction: flip rejection + sigma 1, no choice, full weight on W + ramps)
    r1, d1 = CORR[("full", 1.0)]
    np.save(os.path.join(OUT, "pseudo_truth_V1_scene_world.npy"), track(full_weight(np.ones(len(W))), r1, d1))
    result = dict(rule="see ../gates.json fix_under_test", sigma_chosen=s_star, gates=gates, all_pass_G1_G5=all_pass,
                  per_sigma={str(k): v for k, v in per_sigma.items()}, info_video_motion=info_motion,
                  flip_frames_W=IDX[W][~GOOD[W]].tolist(), flip_frames_all=IDX[~GOOD].tolist(),
                  per_frame=dict(weight=np.round(wf, 4).tolist(), choice_W=cf.astype(int).tolist(),
                                 ref=[res[f]["ref"][:4] for f in W], cand=[res[f][f"cand_full_{s_star}"][:4] for f in W],
                                 new_full=[res2[f][f"full_{s_star}"][:4] for f in range(36, 172)],
                                 cv=[res2[f][f"cv_{s_star}"][:4] for f in W],
                                 rot_vs_bob1_deg=np.round(angs, 2).tolist(), roll_vs_bob1_deg=np.round(roll, 2).tolist(),
                                 centroid_vs_bob1_cm=np.round(dc, 3).tolist()),
                  bundle_map=dict(scene_eq_A_bundle_residual_m=map_res, targets_roundtrip_max=chk, n_target_frames=int(mask.sum()),
                                  target_frames=[int(np.flatnonzero(mask).min()), int(np.flatnonzero(mask).max())] if mask.any() else None),
                  sec=round(time.time() - T0, 1))
    json.dump(result, open(os.path.join(OUT, "d2_result.json"), "w"), indent=1)
    short = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict)) or kk in ("REST_parts", "sigma_passing")} for k, v in gates.items()}
    print(json.dumps(dict(sigma=s_star, all_pass_G1_G5=all_pass, gates=short, info_video_motion=info_motion,
                          bundle_map=result["bundle_map"]), indent=1), flush=True)
    log("D2 DONE")
