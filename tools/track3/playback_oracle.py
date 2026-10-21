"""Oracle study on the 20 public episodes: replace one part of our reconstruction with ground truth
at a time and score with the official metric code (no simulation, poses played back).

Input: the cleaned reconstructions dryrun_public.py wrote (e<ep>_plain: merged for two objects,
rest hold and hand clean as stage 2 does), their meshes, and the public ground truth.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_load_retarget as L  # noqa: E402
import score_local as S  # noqa: E402

V2D_DATA = Path(os.environ.get('V2D_DATA', f"/data/user_data/{os.environ.get('USER', '')}/v2d"))
D = V2D_DATA / 'dryrun_all'
RUNS = V2D_DATA / 'flash_runs/diag'
FPS = L.TRACK3_FPS
METRICS = ('AUC', 'SP-SR', 'MP-SR', 'RPE', 'MPPE')


def T_of(P):
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3, :3] = R.from_quat(P[:, 3:]).as_matrix()
    T[:, :3, 3] = P[:, :3]
    return T


def P_of(T):
    return np.concatenate([T[:, :3, 3], R.from_matrix(T[:, :3, :3]).as_quat()], -1)


def fit_len(T, n):
    return T[:n] if len(T) >= n else np.concatenate([T, np.repeat(T[-1:], n - len(T), 0)])


def ours(e, n):
    ep = e['episode']
    data = dict(np.load(D / f'e{ep}_plain' / 'result.npz'))
    names = [o['name'] for o in e['objects']]
    P = np.stack([P_of(fit_len(data[pk].astype(float), n)) for pk, _, _ in L.object_slots(data)], 1)
    meshes = {(ep, names[i]): D / f'e{ep}_plain_{i}.obj' for i in range(len(names))}
    return P, meshes, data


def registration(e, meshes):
    """Per object, the metric's own registration of our mesh onto the scan: (anchor in our mesh frame,
    rotation into the scan frame), plus the scan anchor. Same seeds as AUC.py."""
    auc, out = S.metric_module('AUC'), []
    for slot, o in enumerate(e['objects']):
        _, v, f = S.mesh_payload(meshes[(e['episode'], o['name'])])
        ref = S.reference_object(o['name'])
        cand_anchor, rot = auc._register_object_body((v, f.astype(np.int64)), (ref['vertices'], ref['faces']),
                                                     ref['anchor'], 1_000_003 * int(e['episode']) + 101 * slot)
        out.append((cand_anchor, rot, ref['anchor']))
    return out


def to_anchor(P, reg, ours_frame):
    """Poses of the scan's anchor body: ours via the registration, ground truth via the scan anchor."""
    auc, out = S.metric_module('AUC'), P.copy()
    for k, (cand_anchor, rot, anchor) in enumerate(reg):
        if ours_frame:
            out[:, k] = auc._convert_object_body(P[:, k], cand_anchor, rot)
        else:
            out[:, k, :3] = P[:, k, :3] + R.from_quat(P[:, k, 3:]).apply(anchor)
    return out


def to_native(Pc, reg):
    """Anchor-body poses back to the scan's own frame (what a submission with the scan mesh holds)."""
    out = Pc.copy()
    for k, (_, _, anchor) in enumerate(reg):
        out[:, k, :3] = Pc[:, k, :3] - R.from_quat(Pc[:, k, 3:]).apply(anchor)
    return out


def move_to(P, Q):
    """Per object, P's motion left-multiplied so its frame 0 equals Q's (as the metric does for its anchor)."""
    out = P.copy()
    for k in range(P.shape[1]):
        A = T_of(Q[:1, k]) @ np.linalg.inv(T_of(P[:1, k]))
        out[:, k] = P_of(A @ T_of(P[:, k]))
    return out


def scene_move_to(P, Q):
    """Whole scene moved by object 0's frame-0 transform (relative placement kept)."""
    A = T_of(Q[:1, 0]) @ np.linalg.inv(T_of(P[:1, 0]))
    return np.stack([P_of(A @ T_of(P[:, k])) for k in range(P.shape[1])], 1)


def scale_fit(Pa, G):
    """Pa already matches G at frame 0; scale each object's displacement by the least-squares factor."""
    out, s_all = Pa.copy(), []
    for k in range(Pa.shape[1]):
        d, g = Pa[:, k, :3] - G[0, k, :3], G[:, k, :3] - G[0, k, :3]
        ds = smooth(Pa[:, k], 4.0)[:, :3] - G[0, k, :3]  # fit on the smoothed path so jitter does not bias it
        s = float(np.clip((ds * g).sum() / max((ds * ds).sum(), 1e-12), 0.25, 4.0))
        out[:, k, :3] = G[0, k, :3] + s * d
        s_all.append(s)
    return out, s_all


def moving_mask(P, v=0.02, w=np.deg2rad(10), win=5):
    """True where the object moves (displacement over `win` frames above v m/s or w rad/s)."""
    dt = win / FPS
    dp = np.linalg.norm(P[win:, :3] - P[:-win, :3], axis=1) / dt
    dr = (R.from_quat(P[win:, 3:]) * R.from_quat(P[:-win, 3:]).inv()).magnitude() / dt
    m = np.zeros(len(P), bool)
    for i in np.nonzero((dp > v) | (dr > w))[0]:
        m[i:i + win + 1] = True
    return m


def smooth(P, sigma=4.0):
    """Gaussian smoothing over time: positions, and quaternions made sign-continuous first."""
    from scipy.ndimage import gaussian_filter1d
    out, q = P.copy(), P[:, 3:].copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    out[:, :3] = gaussian_filter1d(P[:, :3], sigma, axis=0, mode='nearest')
    q = gaussian_filter1d(q, sigma, axis=0, mode='nearest')
    out[:, 3:] = q / np.linalg.norm(q, axis=1, keepdims=True)
    return out


def runs_of(mask):
    out, start = [], None
    for i, x in enumerate(list(mask) + [False]):
        if x and start is None:
            start = i
        if not x and start is not None:
            out.append((start, i))
            start = None
    return out


def hold(P, rest):
    """Each rest run held at the median position / mean rotation of P over that run."""
    out = P.copy()
    for a, b in runs_of(rest):
        if b - a < 3:
            continue
        out[a:b, :3] = np.median(P[a:b, :3], 0)
        out[a:b, 3:] = R.from_quat(P[a:b, 3:]).mean().as_quat()
    return out


def hand_near(data, P, verts, cm=10.0, dilate=5):
    tree, near = cKDTree(verts), np.zeros(len(P), bool)
    T = T_of(P)
    for side in ('left', 'right'):
        wrist = fit_len(data[f'hand_{side}_wrist_to_world_transform'].astype(float), len(P))[:, :3, 3]
        valid = fit_len(data[f'hand_{side}_is_valid'].astype(bool)[:, None, None], len(P))[:, 0, 0]
        local = np.einsum('tji,tj->ti', T[:, :3, :3], wrist - T[:, :3, 3])
        near |= (tree.query(local)[0] < cm / 100) & valid
    out = near.copy()
    for i in np.nonzero(near)[0]:
        out[max(0, i - dilate):i + dilate + 1] = True
    return out


def per_object(fn, P, *masks):
    return np.stack([fn(P[:, k], *[m[k] for m in masks]) for k in range(P.shape[1])], 1)


def variants(e):
    ep = e['episode']
    G = S.ground_truth(ep)
    n = len(G)
    P, meshes, data = ours(e, n)
    import trimesh
    verts = [np.asarray(trimesh.load(meshes[(ep, o['name'])], force='mesh', process=False).vertices)
             for o in e['objects']]
    K = P.shape[1]
    gt_move = [moving_mask(G[:, k]) for k in range(K)]  # frame-independent, fine before conversion
    gt_rest = [~m for m in gt_move]
    own_rest = [~moving_mask(P[:, k], v=0.05, w=np.deg2rad(25), win=7) for k in range(K)]
    hand_rest = [~hand_near(data, P[:, k], verts[k]) for k in range(K)]
    reg = registration(e, meshes)
    P0, G0 = P, G
    P, G = to_anchor(P0, reg, True), to_anchor(G0, reg, False)
    Pa = move_to(P, G)
    Ps, scales = scale_fit(Pa, G)
    v = {
        'still': np.repeat(P[:1], n, 0),
        'still_gt0': np.repeat(G[:1], n, 0),
        'recon': P,
        'recon_hold_hand': per_object(hold, P, hand_rest),
        'recon_hold_speed': per_object(hold, P, own_rest),
        'recon_hold_gt': per_object(hold, P, gt_rest),
        'recon_smooth': per_object(smooth, P),
        'recon_smooth_hold_hand': per_object(hold, per_object(smooth, P), hand_rest),
        'place_scale_smooth_hold_gt': per_object(hold, per_object(smooth, Ps), gt_rest),
        'place': Pa,
        'place_scale': Ps,
        'place_hold_gt': per_object(hold, Pa, gt_rest),
        'place_scale_hold_gt': per_object(hold, Ps, gt_rest),
        'place_rot_gt': np.concatenate([Pa[..., :3], G[..., 3:]], -1),
        'place_pos_gt': np.concatenate([G[..., :3], Pa[..., 3:]], -1),
        'place_gt_when_resting': np.where(np.stack(gt_rest, 1)[..., None], G, Pa),
        'place_gt_when_moving': np.where(np.stack(gt_move, 1)[..., None], G, Pa),
        'gt_motion_our_place': move_to(G, P),
        'gt_motion_our_scene': scene_move_to(G, P) if K > 1 else move_to(G, P),
        'gt': G,
    }
    v = {k: to_native(x, reg) for k, x in v.items()}
    v['recon_ourmesh'], v['gt_ourmesh'] = P0, G0
    # per-object diagnostics
    diag = []
    for k in range(K):
        pe = np.linalg.norm(Pa[:, k, :3] - G[:, k, :3], axis=1) * 100
        re = np.rad2deg((R.from_quat(Pa[:, k, 3:]) * R.from_quat(G[:, k, 3:]).inv()).magnitude())
        gpath = float(np.linalg.norm(np.diff(G[:, k, :3], axis=0), axis=1).sum() * 100)
        ppath = float(np.linalg.norm(np.diff(Pa[:, k, :3], axis=0), axis=1).sum() * 100)
        diag.append({
            'name': e['objects'][k]['name'], 'gt_moving_frac': float(gt_move[k].mean()),
            'gt_path_cm': gpath, 'our_path_cm': ppath,
            'gt_max_disp_cm': float(np.linalg.norm(G[:, k, :3] - G[0, k, :3], axis=1).max() * 100),
            'pos_err_cm': float(pe.mean()), 'rot_err_deg': float(re.mean()),
            'pos_err_rest_cm': float(pe[gt_rest[k]].mean()) if gt_rest[k].any() else None,
            'pos_err_moving_cm': float(pe[gt_move[k]].mean()) if gt_move[k].any() else None,
            'rot_err_rest_deg': float(re[gt_rest[k]].mean()) if gt_rest[k].any() else None,
            'rot_err_moving_deg': float(re[gt_move[k]].mean()) if gt_move[k].any() else None,
            'scale_fit': scales[k],
            'hand_rest_agree': float((hand_rest[k] == gt_rest[k]).mean()),
            'speed_rest_agree': float((own_rest[k] == gt_rest[k]).mean()),
        })
    extra = {'diag': diag}
    if K == 2:
        # placement at frame 0, in object 0's frame (what RPE sees)
        Tg, Tp = T_of(G[:1, 0])[0], T_of(P[:1, 0])[0]
        lg = Tg[:3, :3].T @ (G[0, 1, :3] - G[0, 0, :3])
        lp = Tp[:3, :3].T @ (P[0, 1, :3] - P[0, 0, :3])
        extra['gap_gt_cm'] = float(np.linalg.norm(lg) * 100)
        extra['gap_ours_cm'] = float(np.linalg.norm(lp) * 100)
        extra['gap_err_cm'] = float(np.linalg.norm(lg - lp) * 100)
        # common-mode (camera) vs differential (per-object) error while both objects rest
        both = gt_rest[0] & gt_rest[1]
        if both.sum() > 20:
            Pc = scene_move_to(P, G)
            dev = [(Pc[both, k, :3] - G[both, k, :3]) for k in range(2)]
            dev = [d - d.mean(0) for d in dev]
            common, diff = (dev[0] + dev[1]) / 2, (dev[0] - dev[1]) / 2
            extra['rest_frames_both'] = int(both.sum())
            extra['rest_common_rms_cm'] = float(np.sqrt((common ** 2).sum(1).mean()) * 100)
            extra['rest_diff_rms_cm'] = float(np.sqrt((diff ** 2).sum(1).mean()) * 100)
    return v, meshes, extra


def job(args):
    e, name = args
    ep = e['episode']
    v, meshes, extra = variants(e)
    poses = v['gt'] if name == 'gt_scan' else v[name]
    if name not in ('recon_ourmesh', 'gt_ourmesh'):
        meshes = {(ep, o['name']): S.scan_path(o['name']) for o in e['objects']}
    sub, _ = S.build_submission([e], {ep: poses}, meshes)
    sol = S.pose_solution([e])
    out = {'ep': ep, 'v': name, 'n': len(e['objects'])}
    out.update({m: S.metric_module(m).score(sol, sub, 'row_id') for m in METRICS})
    if name == 'recon_ourmesh':
        out['extra'] = extra
    print(json.dumps(out), flush=True)
    return out


def rollout_job(run):
    """Simulated tracking rollouts on public ep41 against the ground truth."""
    e = [x for x in S.public_episodes() if x['episode'] == 41][0]
    path = RUNS / run / 'results' / 'episode_000041.parquet'
    try:
        poses = S.evaluator_poses(path, e)
    except Exception as err:  # older runs used another frame numbering
        return {'run': run, 'error': str(err)[:200]}
    meshes = {(41, 'white_pot'): D / 'e41_plain_0.obj'}
    sub, _ = S.build_submission([e], {41: poses}, meshes)
    sol = S.pose_solution([e])
    out = {'run': run}
    out.update({m: S.metric_module(m).score(sol, sub, 'row_id') for m in METRICS})
    print(json.dumps(out), flush=True)
    return out


NAMES = ['still', 'still_gt0', 'recon', 'recon_hold_hand', 'recon_hold_speed', 'recon_hold_gt', 'place',
         'place_scale', 'place_hold_gt', 'place_scale_hold_gt', 'place_rot_gt', 'place_pos_gt',
         'place_gt_when_resting', 'place_gt_when_moving', 'gt_motion_our_place', 'gt_motion_our_scene',
         'gt_scan', 'recon_ourmesh', 'recon_smooth', 'recon_smooth_hold_hand', 'place_scale_smooth_hold_gt']

if __name__ == '__main__':
    out_dir, workers = Path(sys.argv[1]), int(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = S.public_episodes()
    jobs = [(e, nm) for e in eps for nm in NAMES]
    rollouts = [r.name for r in sorted(RUNS.glob('ep41_*')) if (r / 'results/episode_000041.parquet').is_file()]
    with ProcessPoolExecutor(workers) as ex:
        rf = [ex.submit(rollout_job, r) for r in rollouts]
        res = list(ex.map(job, jobs))
        rres = [f.result() for f in rf]
    json.dump({'variants': res, 'rollouts': rres}, open(out_dir / 'oracle.json', 'w'), indent=1)
