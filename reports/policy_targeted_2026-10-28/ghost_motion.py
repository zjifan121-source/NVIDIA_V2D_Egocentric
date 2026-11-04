"""Frames where an object moves (smoothed speed > 1 cm/frame or > 3 deg/frame) with no reference contact near it from
either hand: motion the policy cannot cause without being punished for touching. usage: ghost_motion.py PLAN"""
import sys, glob, os, json
import numpy as np, pyarrow.parquet as pq
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

def A(t, n): return np.array(t.column(n).combine_chunks().to_pylist()[0], dtype=float)
def runs(idx):
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1: out[-1][1] = i
        else: out.append([i, i])
    return [(int(a), int(b)) for a, b in out if b - a >= 3]

plan = json.load(open(sys.argv[1]))
for ep, e in sorted(plan.items(), key=lambda x: int(x[0])):
    fs = sorted(glob.glob(os.path.join(e['scene'], "assets/human_motion_data/ego_recon/processed/*/robot_name=sharpa_wave/*.parquet")))
    if not fs: print(f"ep{ep}: no parquet"); continue
    t = pq.read_table(fs[0]); names = t.column_names
    bodies = t.column("object_body_names").combine_chunks().to_pylist()[0]
    pk = [n for n in names if n in ("object_body_position", "object_positions", "object_body_positions", "object_position")][0]
    P = A(t, pk); radius = A(t, "object_mesh_radius").ravel()
    if P.shape[0] == len(bodies) and P.shape[1] != len(bodies): P = P.transpose(1, 0, 2)
    qk = [n for n in names if "object" in n and ("quat" in n or "orient" in n or "rotation" in n)]
    Q = A(t, qk[0]) if qk else None
    if Q is not None and Q.shape[0] == len(bodies) and Q.shape[1] != len(bodies): Q = Q.transpose(1, 0, 2)
    side = [(A(t, f"mano_{s}_object_contact_part_ids"), A(t, f"mano_{s}_object_contact_positions")) for s in ("left", "right")]
    out = []
    for j, b in enumerate(bodies):
        v = np.r_[0, np.linalg.norm(np.diff(P[:, j], axis=0), axis=1)]
        v = uniform_filter1d(v, 5)
        w = np.zeros_like(v)
        if Q is not None and Q.shape[-1] == 4:
            r = Rotation.from_quat(Q[:, j][:, [1, 2, 3, 0]] if qk[0].endswith('wxyz') else Q[:, j])
            w = uniform_filter1d(np.r_[0, np.degrees((r[1:] * r[:-1].inv()).magnitude())], 5)
        moving = np.where((v > 0.0025) | (w > 3))[0]
        miss = []
        for i in moving:
            hit = False
            for pid, cp in side:
                m = pid[i].ravel() > 0; pts = cp[i].reshape(-1, 3)[m]
                hit |= bool(len(pts) and np.min(np.linalg.norm(pts - P[i, j], axis=1)) < radius[j] + 0.03)
            if not hit: miss.append(i)
        if miss:
            out.append(f"{b.split('_')[-2] + '_' + b.split('_')[-1]}: moving {len(moving)} no-contact {len(miss)} runs {runs(miss)[:6]}")
    print(f"ep{ep:>3} {e.get('source','?'):6} {plan[ep]['tracking']:.3f}  " + (" | ".join(out) if out else "clean") + (f"  [rot col: {qk[0]}]" if qk and ep == '1' else ""))
