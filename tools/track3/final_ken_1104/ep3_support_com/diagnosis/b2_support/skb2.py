"""skb2.py -- skeptic check of ep3-B2 (seating gap / Kaggle anchor shift). GT-free, read-only.
Pseudo-GT = the bob1 (h5_ep3_A1) scene reference. Kaggle proxy = AUC.py procedure (anchor-point poses,
rotation from slot-0 pot f0 orientation, translation from slot-0 f0 position, ADD AUC 1..9 cm).
python skb2.py OUT.json"""
import json, os, sys, glob
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rot

OUT = sys.argv[1]
N = 313
SUP = 0.9878071156409292
A1 = '/data/user_data/haisenk/v2d/hand5/scenes/h5_ep3_A1'
PH = [('P0', 0, 35), ('P1', 36, 55), ('P2', 56, 258), ('P3', 259, 288), ('P4', 289, 312)]
RB = '/data/user_data/haisenk/v2d/hand5_backup_20261103/runs'
RUNS = {
    'A1_s42_submitted': '/data/user_data/haisenk/v2d/edmond_bob1/experiments/edmond/bob1_submission_20261103/episode_000003/policy/results/episode_000003.parquet',
    'A1_s43': RB + '/ep3_h5A1_s43/results/episode_000003.parquet',
    'A1_s44': RB + '/ep3_h5A1_s44/results/episode_000003.parquet',
    'seat_s42': RB + '/ep3_h5A1seat_s42/results/episode_000003.parquet',
    'seat_s43': RB + '/ep3_h5A1seat_s43/results/episode_000003.parquet',
}


def load_obj(p):
    V, F = [], []
    for line in open(p):
        if line.startswith('v '):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) for t in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]])
    return np.array(V, float), np.array(F, int)


def sample_surface(V, F, n, seed):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    ch = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)[:, None]
    return (1 - s) * a[ch] + s * (1 - r2[:, None]) * b[ch] + s * r2[:, None] * c[ch]


pqp = glob.glob(A1 + '/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet')[0]
t = pq.read_table(pqp, columns=['object_body_names', 'object_mesh_paths', 'object_body_position', 'object_body_wxyz'])
d = {c: t.column(c)[0].as_py() for c in t.column_names}
names = list(d['object_body_names'])
proc = pqp.split('/sequence_id=')[0]
REFP = np.array(d['object_body_position'], float)          # N,B,3
REFQ = np.array(d['object_body_wxyz'], float)[..., [1, 2, 3, 0]]  # xyzw
MESH, PTS, ANCH = [], [], []
for k in range(len(names)):
    mp = d['object_mesh_paths'][k]
    mp = next(c for c in [mp, os.path.join(proc, os.path.basename(mp))] if os.path.isfile(c))
    V, F = load_obj(mp)
    MESH.append((V, F))
    P = sample_surface(V, F, 3000, 11 + k)
    PTS.append(P); ANCH.append(P.mean(0))
print('bodies', names, flush=True)


def rollout(path):
    tr = pq.read_table(path).to_pydict()
    env = np.array(tr['environment_id']); sel = env == env.min()
    fr = np.array(tr['reference_frame'])[sel].astype(int)
    bn = np.array([str(x) for x in tr['object_body_name']])[sel]
    ap = np.stack([np.array(tr[f'achieved_position_{a}'], float)[sel] for a in 'xyz'], 1)
    aq = np.stack([np.array(tr[f'achieved_quaternion_{a}'], float)[sel] for a in 'xyzw'], 1)
    rp = np.stack([np.array(tr[f'reference_position_{a}'], float)[sel] for a in 'xyz'], 1)
    rq = np.stack([np.array(tr[f'reference_quaternion_{a}'], float)[sel] for a in 'xyzw'], 1)
    P = np.full((N, 2, 3), np.nan); Q = np.full((N, 2, 4), np.nan)
    RP = np.full((N, 2, 3), np.nan); RQ = np.full((N, 2, 4), np.nan)
    for k, key in enumerate(('white_pot', 'wooden_spoon')):
        rows = np.where(np.char.endswith(bn.astype(str), key) & (fr >= 0) & (fr < N))[0]
        P[fr[rows], k] = ap[rows]; Q[fr[rows], k] = aq[rows]; RP[fr[rows], k] = rp[rows]; RQ[fr[rows], k] = rq[rows]
    return P, Q, RP, RQ


def auc(add_m):
    ths = np.arange(0.01, 0.10, 0.01)
    tz = getattr(np, 'trapezoid', None) or np.trapz
    return float(tz(np.array([np.mean(add_m < th) for th in ths]), x=np.linspace(0, 1, 9)))


def anchor_pose(P, Q):
    R = Rot.from_quat(Q.reshape(-1, 4)).as_matrix().reshape(N, 2, 3, 3)
    X = P + np.stack([R[:, k] @ ANCH[k] for k in range(2)], 1)
    return X, R


def kaggle(P, Q, GP, GQ, align=True, extra_shift=None):
    Xa, Ra = anchor_pose(P, Q)
    Xg, Rg = anchor_pose(GP, GQ)
    if align:
        Rq = Rg[0, 0] @ Ra[0, 0].T
        off = Xg[0, 0] - Rq @ Xa[0, 0]
    else:
        Rq = np.eye(3); off = np.zeros(3)
    if extra_shift is not None:
        off = off + extra_shift
    out = {}
    for k in range(2):
        pts = PTS[k] - ANCH[k]
        Pa = np.einsum('ij,njk,vk->nvi', Rq, Ra[:, k], pts) + (Xa[:, k] @ Rq.T + off)[:, None]
        Pg = np.einsum('njk,vk->nvj', Rg[:, k], pts) + Xg[:, k][:, None]
        out[k] = np.linalg.norm(Pa - Pg, axis=2).mean(1)
    return out, float(np.degrees(Rot.from_matrix(Rq).magnitude())), off


def bottoms(P, Q):
    R = Rot.from_quat(Q.reshape(-1, 4)).as_matrix().reshape(N, 2, 3, 3)
    out = []
    for k in range(2):
        V = MESH[k][0]
        W = np.einsum('nij,vj->nvi', R[:, k], V) + P[:, k][:, None]
        out.append(W[..., 2].min(1))
    return np.stack(out, 1)


res = {'support_top': SUP, 'bodies': names, 'runs': {}}
refB = bottoms(REFP, REFQ)
res['ref_bottom_vs_support_cm'] = {nm: {ph: round(float(np.mean(refB[a:b + 1, k] - SUP) * 100), 2) for ph, a, b in PH}
                                   for k, nm in enumerate(('pot', 'spoon'))}
for lab, path in RUNS.items():
    if not os.path.isfile(path):
        print('missing', lab, path); continue
    P, Q, RP, RQ = rollout(path)
    same_ref = float(np.nanmax(np.abs(RP - REFP))) * 1000
    r = {'ref_vs_A1_ref_max_mm': round(same_ref, 3)}
    # unaligned own-ref (rollout vs its own reference columns) and aligned Kaggle proxy vs A1 reference
    u, _, _ = kaggle(P, Q, RP, RQ, align=False)
    a, rdeg, off = kaggle(P, Q, REFP, REFQ, align=True)
    r['ownref_unaligned_auc'] = {nm: round(auc(u[k]), 4) for k, nm in enumerate(('pot', 'spoon'))}
    r['ownref_unaligned_auc']['mean'] = round(np.mean(list(r['ownref_unaligned_auc'].values())), 4)
    r['kaggle_proxy_auc'] = {nm: round(auc(a[k]), 4) for k, nm in enumerate(('pot', 'spoon'))}
    r['kaggle_proxy_auc']['mean'] = round(np.mean(list(r['kaggle_proxy_auc'].values())), 4)
    r['align_rot_deg'] = round(rdeg, 3); r['align_offset_vs_identity_cm'] = np.round((off) * 100, 3).tolist()
    r['f0_drop_mm'] = np.round((RP[0, :, 2] - P[0, :, 2]) * 1000, 2).tolist()
    r['phase_add_cm_unaligned'] = {ph: [round(float(u[k][a_:b_ + 1].mean() * 100), 2) for k in range(2)] for ph, a_, b_ in PH}
    r['phase_add_cm_kaggle'] = {ph: [round(float(a[k][a_:b_ + 1].mean() * 100), 2) for k in range(2)] for ph, a_, b_ in PH}
    B = bottoms(P, Q)
    r['rollout_bottom_vs_support_cm'] = {ph: [round(float(np.mean(B[a_:b_ + 1, k] - SUP) * 100), 2) for k in range(2)] for ph, a_, b_ in PH}
    dz = (P[..., 2] - RP[..., 2]) * 100
    r['phase_dz_cm'] = {ph: [round(float(dz[a_:b_ + 1, k].mean()), 2) for k in range(2)] for ph, a_, b_ in PH}
    # spoon lift timing: first frame z rises > +0.5 / +3 cm above own f0-20 rest median
    for k, nm in ((1, 'spoon'), (0, 'pot')):
        zr = REFP[:, k, 2]; za = P[:, k, 2]
        b0r, b0a = np.median(zr[:20]), np.median(za[2:20])
        r[f'{nm}_lift_frames_ref_vs_roll'] = {thr: [int(np.argmax(zr - b0r > thr / 100)), int(np.argmax(za - b0a > thr / 100))] for thr in (0.5, 1.0, 3.0)}
    # what Kaggle proxy would be if the anchor were NOT shifted by the step-0 drop (counterfactual: same rollout,
    # alignment offset computed as if the rollout pot f0 sat at the reference height)
    drop = np.array([0, 0, RP[0, 0, 2] - P[0, 0, 2]])
    c, _, _ = kaggle(P, Q, REFP, REFQ, align=True, extra_shift=-drop)
    r['kaggle_proxy_auc_no_anchor_shift'] = {nm: round(auc(c[k]), 4) for k, nm in enumerate(('pot', 'spoon'))}
    r['kaggle_proxy_auc_no_anchor_shift']['mean'] = round(np.mean(list(r['kaggle_proxy_auc_no_anchor_shift'].values())), 4)
    res['runs'][lab] = r
    print(lab, json.dumps({k: v for k, v in r.items() if k not in ('phase_add_cm_unaligned',)}), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
print('REF bottoms vs support', json.dumps(res['ref_bottom_vs_support_cm']))
print('DONE')
