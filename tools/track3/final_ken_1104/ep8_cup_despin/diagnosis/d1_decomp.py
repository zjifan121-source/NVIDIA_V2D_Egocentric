"""d1_decomp.py (refine1104 ep8): per-frame decomposition of the adopted T6 rollout (h_ep8_sc_K_s42 @100M) errors.
GT-free: compares the rollout's achieved cup/pot poses with OUR reference (the scene parquet), plus the per-frame scorer
ADD from the verbatim-AUC.py emulator (emu8.py, REF judge, T6 cup mesh, centroid anchor).
Cup: centroid position error, symmetric-axis tilt error, spin (twist) about the cup axis; reference twist rate (FP
twist noise per ep8-B2) vs reference swing rate (real pour tilt) vs reference speed; right-wrist reference rotation rate.
Pot: per-axis position error and rotation error, f73-127 (open item #5), vs bob1's submitted rollout on the same reference.
Run: python -B -I d1_decomp.py   (reads only; writes d1_out.json next to this script)"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\v2d_demo\diag\ep_status_1104\ep8_work")
os.chdir(r"C:\v2d_demo\diag\ep_status_1104\ep8_work")  # emu8 resolves AUC.py / OBJ relative to its own file; nothing is written there (-B)
import emu8 as E
from scipy.spatial.transform import Rotation as Rot
import pyarrow.parquet as pq

NF = 219
SC = os.path.join(HERE, "sc_ep8", "assets", "human_motion_data", "ego_recon", "processed",
                  "sequence_id=ep8_ken_c_scored_attach", "robot_name=sharpa_wave", "4774016f0a65458c9245084fdc462018-0.parquet")
row = pq.read_table(SC).to_pylist()[0]
V, F = E.MESH["t6cup"]
c_body, _ = E.M._surface_moments(V, F)
a_body = E.AXIS["t6cup"]
REF = E.REF6  # (219, 2, 7) xyz + xyzw
ACH = E.ACH6
SUB_B1 = E.SUBS["bob1"][E.M._POSE_COLUMNS].to_numpy().reshape(NF, 2, 7)


def rot(q):
    return Rot.from_quat(q)


def twist_swing(rrel, axis):
    """rrel: Rotation (body-frame relative). returns (twist_deg about axis, swing_deg)."""
    q = rrel.as_quat()  # xyzw
    p = np.dot(q[..., :3], axis)
    tw = np.concatenate([np.outer(p, axis) if q.ndim > 1 else p * axis, q[..., 3:4]], -1)
    n = np.linalg.norm(tw, axis=-1, keepdims=True)
    tw = tw / np.where(n < 1e-12, 1, n)
    ang = 2 * np.degrees(np.arctan2(np.dot(tw[..., :3], axis), tw[..., 3]))
    ang = (ang + 180) % 360 - 180
    sw = rrel * Rot.from_quat(tw).inv()
    return ang, np.degrees(sw.magnitude())


def body(arr, i):
    return arr[:, i, :3], rot(arr[:, i, 3:])


# ---- per-frame scorer ADD (REF judge, T6 mesh, centroid anchor), T6 and bob1
add = {}
for cand in ("T6", "bob1"):
    E.M._METRIC = "add_auc"
    sol = E.solution_df(E.judge("REF"), "t6cup", "centroid")
    s = float(E.M.score(sol.copy(), E.SUBS[cand].copy(), "row_id"))
    add[cand] = [np.asarray(E._LAST["spe"][i]).reshape(NF, -1).mean(1) if np.asarray(E._LAST["spe"][i]).ndim > 1 else np.asarray(E._LAST["spe"][i]) for i in range(2)]
    print(cand, "episode add_auc", round(s, 4), flush=True)

# ---- cup decomposition
pr, Rr = body(REF, 0)
pa, Ra = body(ACH, 0)
cr = pr + Rr.apply(c_body)
ca = pa + Ra.apply(c_body)
axr = Rr.apply(a_body)
axa = Ra.apply(a_body)
e_pos = 100 * np.linalg.norm(ca - cr, axis=1)
e_tilt = np.degrees(np.arccos(np.clip(np.abs(np.sum(axr * axa, 1)), -1, 1)))
spin_err, _ = twist_swing(Rr.inv() * Ra, a_body)
ref_tw = np.zeros(NF); ref_sw = np.zeros(NF); ach_tw = np.zeros(NF); ach_sw = np.zeros(NF)
for t in range(1, NF):
    ref_tw[t], ref_sw[t] = twist_swing(Rr[t - 1].inv() * Rr[t], a_body)
    ach_tw[t], ach_sw[t] = twist_swing(Ra[t - 1].inv() * Ra[t], a_body)
ref_v = np.r_[0, 100 * np.linalg.norm(np.diff(cr, axis=0), axis=1)]
ach_v = np.r_[0, 100 * np.linalg.norm(np.diff(ca, axis=0), axis=1)]
# lag: achieved centroid vs reference centroid k frames earlier (best k)
lag_best = []
for t in range(NF):
    ks = [k for k in range(0, 6) if t - k >= 0]
    d = [np.linalg.norm(ca[t] - cr[t - k]) for k in ks]
    lag_best.append(ks[int(np.argmin(d))])
lag_best = np.array(lag_best)
wr = np.array(row["robot_right_wrist_wxyz"], float)
wrR = Rot.from_quat(wr[:, [1, 2, 3, 0]])
wr_rate = np.r_[0, np.degrees((wrR[:-1].inv() * wrR[1:]).magnitude())]
wr_p = np.array(row["robot_right_wrist_position"], float)
wr_v = np.r_[0, 100 * np.linalg.norm(np.diff(wr_p, axis=0), axis=1)]
# hand-to-cup offset in the cup frame (how rigid the reference grasp is)
off = Rr.inv().apply(wr_p - cr)
off_step = np.r_[0, 100 * np.linalg.norm(np.diff(off, axis=0), axis=1)]

WIN = {"pre f32-44": (32, 45), "blend-in f46-54": (46, 55), "spin1+pour f55-72": (55, 73), "carry f73-127": (73, 128),
       "pour2+spin2 f128-155": (128, 156), "carry f156-175": (156, 176), "blend-out f176-185": (176, 186), "set-down f186-199": (186, 200)}


def credit(e_cm):
    t = np.arange(1, 10)
    x = np.linspace(0, 1, 9)
    return np.array([np.trapezoid((e < t).astype(float), x=x) for e in e_cm])


out = {"cup": {}, "pot": {}, "frames": {}}
addT6c = 100 * add["T6"][0]; addB1c = 100 * add["bob1"][0]
addT6p = 100 * add["T6"][1]; addB1p = 100 * add["bob1"][1]
for w, (a, b) in WIN.items():
    sl = slice(a, b)
    out["cup"][w] = {
        "scorer_ADD_med_cm": round(float(np.median(addT6c[sl])), 2), "scorer_ADD_max_cm": round(float(addT6c[sl].max()), 2),
        "lost_ep_auc": round(float((1 - credit(addT6c[sl])).sum() / NF / 2), 4),
        "centroid_err_med_cm": round(float(np.median(e_pos[sl])), 2), "centroid_err_max_cm": round(float(e_pos[sl].max()), 2),
        "axis_tilt_err_med_deg": round(float(np.median(e_tilt[sl])), 1), "axis_tilt_err_max_deg": round(float(e_tilt[sl].max()), 1),
        "ref_twist_sum_abs_deg": round(float(np.abs(ref_tw[sl]).sum()), 1), "ref_twist_net_deg": round(float(ref_tw[sl].sum()), 1),
        "ach_twist_net_deg": round(float(ach_tw[sl].sum()), 1),
        "ref_swing_sum_deg": round(float(ref_sw[sl].sum()), 1), "ref_speed_med_cm_f": round(float(np.median(ref_v[sl])), 2),
        "ref_speed_max_cm_f": round(float(ref_v[sl].max()), 2),
        "wrist_rot_rate_max_deg_f": round(float(wr_rate[sl].max()), 1), "wrist_speed_max_cm_f": round(float(wr_v[sl].max()), 2),
        "hand_in_cupframe_step_max_cm": round(float(off_step[sl].max()), 2),
        "best_lag_frames_med": float(np.median(lag_best[sl])),
    }
# correlations over the lifted window f46-185 (frames where the cup is held)
sl = slice(46, 186)
def corr(x, y):
    return round(float(np.corrcoef(x[sl], y[sl])[0, 1]), 3)
out["cup_corr_f46_185"] = {
    "ADD~|ref_twist_rate|": corr(addT6c, np.abs(ref_tw)), "ADD~ref_swing_rate": corr(addT6c, ref_sw),
    "ADD~ref_speed": corr(addT6c, ref_v), "ADD~wrist_rot_rate": corr(addT6c, wr_rate),
    "ADD~centroid_err": corr(addT6c, e_pos), "ADD~axis_tilt_err": corr(addT6c, e_tilt),
    "centroid_err~|ref_twist_rate|": corr(e_pos, np.abs(ref_tw)), "centroid_err~ref_speed": corr(e_pos, ref_v),
    "centroid_err~ref_swing_rate": corr(e_pos, ref_sw),
    # lagged versions: error at t vs twist rate at t-2..t (sum)
    "ADD~|ref_twist|sum(t-3..t)": corr(addT6c, np.convolve(np.abs(ref_tw), np.ones(4), mode="full")[:NF]),
    "ADD~ref_swing sum(t-3..t)": corr(addT6c, np.convolve(ref_sw, np.ones(4), mode="full")[:NF]),
    "ADD~ref_speed sum(t-3..t)": corr(addT6c, np.convolve(ref_v, np.ones(4), mode="full")[:NF]),
}
# per-frame table for the two windows
for f in list(range(50, 75)) + list(range(125, 158)):
    out["frames"][f] = {"ADD_T6": round(float(addT6c[f]), 2), "ADD_bob1": round(float(addB1c[f]), 2), "cen_err": round(float(e_pos[f]), 2),
                        "tilt_err": round(float(e_tilt[f]), 1), "spin_err": round(float(spin_err[f]), 0),
                        "ref_tw": round(float(ref_tw[f]), 1), "ach_tw": round(float(ach_tw[f]), 1), "ref_sw": round(float(ref_sw[f]), 1),
                        "ref_v": round(float(ref_v[f]), 2), "ach_v": round(float(ach_v[f]), 2), "wr_rate": round(float(wr_rate[f]), 1),
                        "lag": int(lag_best[f])}

# ---- pot (open item #5)
pr1, Rr1 = body(REF, 1)
pa1, Ra1 = body(ACH, 1)
pb1, Rb1 = body(SUB_B1, 1)
Vp, Fp = E.MESH["pot"]
cp_body, _ = E.M._surface_moments(Vp, Fp)
crp = pr1 + Rr1.apply(cp_body); cap = pa1 + Ra1.apply(cp_body)
dp = 100 * (cap - crp)
rot_err = np.degrees((Rr1.inv() * Ra1).magnitude())
rot_err_b1 = np.degrees((Rr1.inv() * Rb1).magnitude())
ref_vp = np.r_[0, 100 * np.linalg.norm(np.diff(crp, axis=0), axis=1)]
for w, (a, b) in {"pre f0-13": (0, 14), "held-on-table f14-94": (14, 95), "carry f73-94": (73, 95), "lift f95-127": (95, 128),
                  "lifted f128-164": (128, 165), "setdown f165-197": (165, 198), "rest_end f198-218": (198, 219)}.items():
    sl = slice(a, b)
    out["pot"][w] = {"scorer_ADD_med_T6": round(float(np.median(addT6p[sl])), 2), "scorer_ADD_max_T6": round(float(addT6p[sl].max()), 2),
                     "scorer_ADD_med_bob1": round(float(np.median(addB1p[sl])), 2),
                     "lost_ep_auc_T6": round(float((1 - credit(addT6p[sl])).sum() / NF / 2), 4),
                     "lost_ep_auc_bob1": round(float((1 - credit(addB1p[sl])).sum() / NF / 2), 4),
                     "dpos_med_cm_xyz": np.round(np.median(dp[sl], 0), 2).tolist(), "dpos_max_abs_cm_xyz": np.round(np.abs(dp[sl]).max(0), 2).tolist(),
                     "rot_err_med_deg": round(float(np.median(rot_err[sl])), 2), "rot_err_max_deg": round(float(rot_err[sl].max()), 2),
                     "rot_err_med_deg_bob1": round(float(np.median(rot_err_b1[sl])), 2),
                     "ref_speed_max_cm_f": round(float(ref_vp[sl].max()), 2)}
pf = {}
for f in range(70, 130, 2):
    pf[f] = {"ADD_T6": round(float(addT6p[f]), 2), "ADD_b1": round(float(addB1p[f]), 2), "dpos": np.round(dp[f], 2).tolist(),
             "rot_err": round(float(rot_err[f]), 2), "rot_err_b1": round(float(rot_err_b1[f]), 2), "ref_v": round(float(ref_vp[f]), 2)}
out["pot_frames"] = pf
json.dump(out, open(os.path.join(HERE, "d1_out.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("cup", "cup_corr_f46_185")}, indent=1))
print(json.dumps(out["pot"], indent=1))
