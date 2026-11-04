#!/usr/bin/env python3
"""sk10a2.py -- skeptic re-check of finding ep10_A2_blue_phantom_lift (GT-free).

Sources compared for the blue cup (bottom-centre height above the table, cm):
  raw FoundationPose (ken_c_scored/blue_cup/poses), smoothed FP (poses_smoothed),
  bundle_merged, bundle_cleaned (both via the merged-bundle camera / loader transform A),
  scene parquet (reference), bob1 rollout (achieved).
Observed: FS depth (beige_cup/depth) inside our SAM 3 blue / beige masks minus dilated hand masks,
  every frame: p2/p50/p95/p98 heights of the visible blue, p98 of the visible beige.
Also: left-wrist shift parquet vs bundle_merged (the attach), Kaggle-style AUC loss of the rollout
  blue lift, frame crops with projected rims (reference vs reference dropped onto the table).
Run inside a CPU step of job 11035728:  /home/haisenk/anaconda3/bin/python sk10a2.py
"""
import json
import os
import sys
import time

sys.path.insert(0, '/home/haisenk/v2d_ops/kenc/review/tool')
import numpy as np  # noqa: E402
import render_review as rr  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.ndimage import binary_dilation, binary_erosion  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

T0 = time.time()


def log(m):
    print(f'[sk10a2 +{time.time() - T0:6.1f}s] {m}', flush=True)


OUT = os.path.expanduser('~/v2d_ops/kenc/diag/skeptic_ep10A2')
SC = ('/scratch/haisenk-v2d/hand6/bob1_scenes/experiments/haisenk/hand_fixes_v1_20261102/'
      'episode_000010/scene')
KENC = ('/data/user_data/haisenk/v2d/v2d_recon/baseline/reconstruction/data/outputs/ken_c_scored/'
        'episode_000010')
ROLL = ('/data/user_data/haisenk/v2d/edmond_bob1/experiments/edmond/bob1_submission_20261103/'
        'episode_000010/policy/results/episode_000010.parquet')
ZT = 1.0
os.makedirs(OUT, exist_ok=True)
RES = {}

pqp = rr.find_parquet(SC)
P = rr.load_scene_parquet(pqp)
N = P['N']
proc_dir = os.path.dirname(os.path.dirname(os.path.dirname(pqp)))
Bc = rr.load_bundle(os.path.join(SC, 'bundle_cleaned', 'result.npz'))
Bm = rr.load_bundle(os.path.join(SC, 'bundle_merged', 'result.npz'))
zm = np.load(os.path.join(SC, 'bundle_merged', 'result.npz'), allow_pickle=True)
zc = np.load(os.path.join(SC, 'bundle_cleaned', 'result.npz'), allow_pickle=True)

objs = []
for k, bn in enumerate(P['body_names']):
    nm = 'blue_cup' if bn.endswith('blue_cup') else ('beige_cup' if bn.endswith('beige_cup') else bn)
    mp = P['mesh_paths'][k]
    mpath = next(c for c in [mp, os.path.join(proc_dir, os.path.basename(mp))] if c and os.path.isfile(c))
    V, F = rr.load_obj(mpath)
    R = rr.wxyz_to_R(P['wxyz'][:, k]).reshape(N, 3, 3)
    T = rr.to_T(R, P['pos'][:, k])
    lo, hi = V.min(0), V.max(0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    objs.append(dict(name=nm, V=V, F=np.asarray(F, np.int64), T=T, corners=corners, mesh=mpath))
ob = {o['name']: o for o in objs}
BL, BE = ob['blue_cup'], ob['beige_cup']
bnames = [str(x) for x in zm['object_names']]
iBLb, iBEb = bnames.index('blue_cup'), bnames.index('beige_cup')

# bundle meshes vs processed meshes
for o, fn in ((BL, 'mesh.obj' if iBLb == 0 else 'mesh_1.obj'), (BE, 'mesh_1.obj' if iBEb == 1 else 'mesh.obj')):
    Vb, _ = rr.load_obj(os.path.join(SC, 'bundle_merged', fn))
    o['Vb'] = Vb
    same = (Vb.shape == o['V'].shape) and float(np.abs(Vb - o['V']).max()) < 1e-6
    RES.setdefault('mesh_check', {})[o['name']] = dict(bundle_mesh=fn, n_bundle=len(Vb), n_proc=len(o['V']),
                                                       identical=bool(same),
                                                       max_abs_diff_m=(float(np.abs(Vb - o['V']).max())
                                                                       if Vb.shape == o['V'].shape else None))
log(f"mesh check {RES['mesh_check']}")

A, resA = rr.estimate_A([o['T'] for o in objs], [Bc['objT'][iBLb if o is BL else iBEb] for o in objs],
                        [o['corners'] for o in objs])
RES['loader_A_residual_mm'] = float(np.nanmax(resA) * 1000)
log(f'loader A (parquet vs bundle_cleaned) residual max {RES["loader_A_residual_mm"]:.3f} mm')
c2w = A[None] @ Bc['c2w']
w2c = np.linalg.inv(c2w)
Kc = Bc['K']
cam_diff = float(np.abs(Bc['c2w'] - Bm['c2w']).max())
RES['camera_cleaned_vs_merged_maxabs'] = cam_diff
log(f'camera cleaned vs merged max abs diff {cam_diff:.2e}')

# symmetry axis / bottom centre per object (as fit10.py)
rng = np.random.default_rng(0)


def sample_surface(V, F, n):
    tri = V[F]
    ar = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(F), n, p=ar / ar.sum())
    u, v = rng.random(n), rng.random(n)
    sw = u + v > 1
    u[sw], v[sw] = 1 - u[sw], 1 - v[sw]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


for o in objs:
    S = sample_surface(o['V'], o['F'], 60000)
    C = S - S.mean(0)
    w, E = np.linalg.eigh(C.T @ C / len(C))
    zb = o['T'][0][:3, :3].T @ np.array([0, 0, 1.0])
    i = int(np.argmax(np.abs(E.T @ zb)))
    a = E[:, i] * np.sign(E[:, i] @ zb)
    h = S @ a
    low = h < h.min() + 0.8 * (h.max() - h.min())
    ctr = (S[low] - np.outer(S[low] @ a, a)).mean(0)
    o.update(axis=a, ctr=ctr, hmin=float(h.min()), hmax=float(h.max()), height=float(h.max() - h.min()))
    hv = (o['V'] - ctr) @ a
    o['top_ring'] = o['V'][hv > o['hmax'] - 0.004]
    o['bot_ring'] = o['V'][hv < o['hmin'] + 0.004]
    o['bc'] = ctr + o['hmin'] * a   # bottom centre (mesh frame)
    log(f"{o['name']}: height {o['height'] * 100:.2f} cm; top ring {len(o['top_ring'])} bottom ring {len(o['bot_ring'])}")


def xf(T, p):
    return T[:3, :3] @ p + T[:3, 3]


def pose_json(path):
    d = json.load(open(path))
    return np.array(d['rotation'], float), np.array(d['translation'], float)


# FP raw / smoothed: find the quaternion convention that reproduces bundle_merged object_to_camera (blue)
Tcam_m = zm['object_to_camera_transform'].astype(float)   # object 0 = blue
assert iBLb == 0
q_s, t_s = zip(*[pose_json(os.path.join(KENC, 'blue_cup', 'poses_smoothed', f'{f:06d}.json')) for f in range(N)])
q_r, t_r = zip(*[pose_json(os.path.join(KENC, 'blue_cup', 'poses', f'{f:06d}.json')) for f in range(N)])
q_s, t_s, q_r, t_r = map(np.array, (q_s, t_s, q_r, t_r))
best = None
for conv in ('wxyz', 'xyzw'):
    qq = q_s[:, [1, 2, 3, 0]] if conv == 'wxyz' else q_s
    Rs = Rot.from_quat(qq).as_matrix()
    e = float(np.abs(Rs - Tcam_m[:, :3, :3]).max())
    et = float(np.abs(t_s - Tcam_m[:, :3, 3]).max())
    log(f'smoothed poses as {conv}: max |R - bundle_merged R| {e:.2e}, max |t - t_b| {et:.2e} m')
    if best is None or e < best[1]:
        best = (conv, e, et)
conv = best[0]
RES['fp_pose_convention'] = dict(conv=conv, smoothed_vs_bundle_merged_maxR=best[1], max_t_m=best[2])


def to_T_list(q, t):
    qq = q[:, [1, 2, 3, 0]] if conv == 'wxyz' else q
    return rr.to_T(Rot.from_quat(qq).as_matrix(), t)


Tc_raw = to_T_list(q_r, t_r)
Tc_smo = to_T_list(q_s, t_s)
# into the loader world: c2w (loader) @ T_cam_obj ; bundle objT into loader world via A
srcs = {
    'fp_raw': np.array([c2w[f] @ Tc_raw[f] for f in range(N)]),
    'fp_smoothed': np.array([c2w[f] @ Tc_smo[f] for f in range(N)]),
    'bundle_merged': A[None] @ Bm['objT'][iBLb],
    'bundle_cleaned': A[None] @ Bc['objT'][iBLb],
    'parquet_ref': BL['T'],
}
notes = []
roll = rr.load_rollout(ROLL, P, objs, [o['name'] for o in objs], N, notes)
srcs['rollout'] = roll['ach'][objs.index(BL)]
RES['rollout_mapping'] = roll['mapping']['how']


def heights(T, o, V=None):
    V = o['V'] if V is None else V
    bc = np.array([xf(T[f], o['bc']) for f in range(N)])
    zmin = np.array([(V @ T[f][:3, :3].T + T[f][:3, 3])[:, 2].min() for f in range(N)])
    zmax = np.array([(V @ T[f][:3, :3].T + T[f][:3, 3])[:, 2].max() for f in range(N)])
    ax = np.array([T[f][:3, :3] @ o['axis'] for f in range(N)])
    tilt = np.degrees(np.arccos(np.clip(ax[:, 2], -1, 1)))
    return dict(bc=bc, zmin=zmin, zmax=zmax, tilt=tilt)


H = {k: heights(v, BL, BL['V'] if k in ('parquet_ref', 'rollout') else BL['Vb']) for k, v in srcs.items()}
HB = {'parquet_ref': heights(BE['T'], BE), 'rollout': heights(roll['ach'][objs.index(BE)], BE)}
grid = [0, 10, 13, 16, 20, 30, 40, 46, 52, 58, 64, 70, 76, 80, 82, 88, 94, 100, 106, 112, 118, 124, 130, 140, 150, 160]
tab = {}
for f in grid:
    tab[f] = {k: dict(bottom_centre_cm=round((h['bc'][f, 2] - ZT) * 100, 2), lowest_cm=round((h['zmin'][f] - ZT) * 100, 2),
                      top_cm=round((h['zmax'][f] - ZT) * 100, 2), tilt_deg=round(float(h['tilt'][f]), 1),
                      xy_from_f0_cm=round(float(np.linalg.norm(h['bc'][f, :2] - h['bc'][0, :2]) * 100), 2))
              for k, h in H.items()}
RES['blue_heights_by_source'] = tab
for f in grid:
    log(f'f{f:3d} ' + ' | '.join(f"{k} bc {v['bottom_centre_cm']:6.2f} lo {v['lowest_cm']:6.2f} tilt {v['tilt_deg']:4.1f} xy {v['xy_from_f0_cm']:5.2f}"
                                 for k, v in tab[f].items()))
# raw vs smoothed difference along the camera ray / along the cup axis
d_rs = np.array([np.linalg.norm(Tc_raw[f][:3, 3] - Tc_smo[f][:3, 3]) for f in range(N)])
RES['raw_vs_smoothed_translation_cm'] = dict(max=round(float(d_rs.max() * 100), 2), at=int(d_rs.argmax()),
                                             median=round(float(np.median(d_rs) * 100), 2))
RES['bc_height_series_cm'] = {k: [round(float((h['bc'][f, 2] - ZT) * 100), 2) for f in range(N)] for k, h in H.items()}
RES['beige_bottom_series_cm'] = {k: [round(float((h['zmin'][f] - ZT) * 100), 2) for f in range(N)] for k, h in HB.items()}

# ------------------------------------------------------------------ observed depth heights, every frame
W, Hh = 2028, 1520


def load_mask(path):
    try:
        a = np.asarray(Image.open(path).convert('L'))
    except Exception:
        return None
    if a.shape != (Hh, W):
        a = np.asarray(Image.fromarray(a).resize((W, Hh), Image.NEAREST))
    return a > 127 if a.max() > 1 else a > 0


def hand_mask(f):
    m = np.zeros((Hh, W), bool)
    for nm in ('blue_cup', 'beige_cup'):
        for hid in ('2', '3'):
            x = load_mask(os.path.join(KENC, nm, 'masks', hid, f'{f:06d}.png'))
            if x is not None:
                m |= x
    return m


def depth_m(f):
    raw = np.asarray(Image.open(os.path.join(KENC, 'beige_cup', 'depth', f'{f:06d}.png'))).astype(np.float64)
    ok = (raw > 0) & (raw < 65535)
    return np.where(ok, 65535.0 / np.maximum(raw, 1) - 1.0, np.nan)


def bp(Z, f, sel):
    v, u = np.nonzero(sel)
    z = Z[v, u]
    fx, fy, cx, cy = Kc[f]
    Pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
    return Pc @ c2w[f][:3, :3].T + c2w[f][:3, 3]


obs = {}
for f in range(N):
    Z = depth_m(f)
    hm = hand_mask(f)
    hmd = binary_dilation(hm, iterations=8)
    row = dict(hand_px=int(hm.sum()))
    for nm in ('blue_cup', 'beige_cup'):
        m = load_mask(os.path.join(KENC, nm, 'masks', '1', f'{f:06d}.png'))
        if m is None:
            continue
        row[nm + '_mask_px'] = int(m.sum())
        mm = binary_erosion(m & ~hmd, iterations=4)
        sel = mm & np.isfinite(Z) & (Z > 0.15) & (Z < 2.0)
        Pw = bp(Z, f, sel)
        if len(Pw) < 100:
            row[nm] = None
            continue
        z_ = (Pw[:, 2] - ZT) * 100
        row[nm] = dict(n=int(len(Pw)), p2=round(float(np.percentile(z_, 2)), 2), p50=round(float(np.percentile(z_, 50)), 2),
                       p95=round(float(np.percentile(z_, 95)), 2), p98=round(float(np.percentile(z_, 98)), 2))
    obs[f] = row
    if f % 10 == 0:
        log(f'obs f{f}: {json.dumps(row)}')
RES['observed'] = {str(f): v for f, v in obs.items()}

# ------------------------------------------------------------------ left wrist: parquet vs bundle_merged (the attach)
try:
    jl = P['joints']['left']   # N x J x 3, loader world
    wm = A[None] @ zm['hand_left_wrist_to_world_transform'].astype(float)
    wc = A[None] @ zc['hand_left_wrist_to_world_transform'].astype(float)
    dl = jl[:, 0, :] - wm[:, :3, 3]
    dlc = wc[:, :3, 3] - wm[:, :3, 3]
    RES['left_wrist_parquet_minus_merged_cm'] = {str(f): [round(float(x) * 100, 2) for x in dl[f]] for f in grid}
    RES['left_wrist_cleaned_minus_merged_cm'] = {str(f): [round(float(x) * 100, 2) for x in dlc[f]] for f in grid}
    log('left wrist parquet-merged (x,y,z cm): ' + ' '.join(f'f{f}:{np.round(dl[f] * 100, 1).tolist()}' for f in grid))
    log('left wrist cleaned-merged (x,y,z cm): ' + ' '.join(f'f{f}:{np.round(dlc[f] * 100, 1).tolist()}' for f in grid))
except Exception as e:  # noqa: BLE001
    RES['left_wrist_error'] = repr(e)
    log(f'left wrist check failed: {e!r}')

# ------------------------------------------------------------------ Kaggle-style AUC loss of the rollout blue lift
thr = np.arange(0.01, 0.10, 0.01)
xs = np.linspace(0, 1, thr.size)


_trap = getattr(np, 'trapezoid', None) or np.trapz


def auc_of(d):
    return float(_trap(np.array([np.mean(d < t) for t in thr]), x=xs))


lift_roll = (H['rollout']['bc'][:, 2] - H['rollout']['bc'][0, 2])
lift_ref = (H['parquet_ref']['bc'][:, 2] - H['parquet_ref']['bc'][0, 2])
imp = {}
for base in (0.0, 0.01, 0.015, 0.02):
    d_with = np.sqrt(base ** 2 + np.maximum(lift_roll, 0) ** 2)
    d_without = np.full(N, base)
    d_refpath = np.sqrt(base ** 2 + np.maximum(lift_ref, 0) ** 2)
    imp[str(base)] = dict(blue_auc_with_lift=round(auc_of(d_with), 4), blue_auc_no_lift=round(auc_of(d_without), 4),
                          blue_auc_if_rollout_tracked_ref_exactly=round(auc_of(d_refpath), 4),
                          episode_gain_if_lift_removed=round((auc_of(d_without) - auc_of(d_with)) / 2, 4))
RES['kaggle_style_lift_loss'] = dict(note='blue ADD = sqrt(base^2 + rollout bottom-centre rise since f0); truth assumed '
                                          'on the table at its f0 height; 2 objects per episode',
                                     by_base_error_m=imp,
                                     rollout_lift_frames_over_1cm=[int(f) for f in np.where(lift_roll > 0.01)[0]],
                                     rollout_lift_max_cm=round(float(lift_roll.max() * 100), 2),
                                     rollout_lift_argmax=int(lift_roll.argmax()))
log(f"kaggle-style lift loss {json.dumps(RES['kaggle_style_lift_loss'])}")

# ------------------------------------------------------------------ crops with projected rims
VID = os.path.join(KENC, 'blue_cup', 'anycalib', 'undistorted.mp4')
CF = [30, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94, 100, 112, 124, 140, 160]
frames, nfr, (Wv, Hv), fnote = rr.load_frames(dict(kind='video', path=VID), CF)
RES['frames_note'] = fnote


def proj(Pw, f):
    uv, ok, z = rr.project(Pw, w2c[f], Kc[f])
    return uv[ok]


tiles = []
for f in CF:
    img = Image.fromarray(frames[f])
    dr = ImageDraw.Draw(img)
    Tref = BL['T'][f]
    drop = Tref.copy()
    drop[2, 3] -= (H['parquet_ref']['zmin'][f] - ZT)   # reference dropped so its lowest point is on the table
    Tro = srcs['rollout'][f]
    for T_, col in ((Tref, (255, 40, 40)), (drop, (40, 255, 40)), (Tro, (0, 230, 255))):
        for ring in ('top_ring', 'bot_ring'):
            Pw = BL[ring] @ T_[:3, :3].T + T_[:3, 3]
            for (u, v) in proj(Pw, f)[::3]:
                dr.ellipse([u - 2, v - 2, u + 2, v + 2], fill=col)
    m = load_mask(os.path.join(KENC, 'blue_cup', 'masks', '1', f'{f:06d}.png'))
    me = load_mask(os.path.join(KENC, 'beige_cup', 'masks', '1', f'{f:06d}.png'))
    mm = (m if m is not None else np.zeros((Hh, W), bool)) | (me if me is not None else np.zeros((Hh, W), bool))
    ys, xs_ = np.nonzero(mm)
    cx, cy = (int(np.median(xs_)), int(np.median(ys))) if len(ys) else (W // 2, Hh // 2)
    box = (max(cx - 380, 0), max(cy - 420, 0), min(cx + 380, W), min(cy + 460, Hh))
    cr = img.crop(box).resize((380, 440))
    d2 = ImageDraw.Draw(cr)
    d2.rectangle([0, 0, 380, 18], fill=(0, 0, 0))
    o_ = obs[f].get('blue_cup') or {}
    d2.text((4, 3), f"f{f} obs blue top {o_.get('p98', 'na')} ref bc {RES['bc_height_series_cm']['parquet_ref'][f]} "
                    f"roll {RES['bc_height_series_cm']['rollout'][f]}", fill=(255, 255, 255))
    tiles.append(cr)
cols = 4
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new('RGB', (380 * cols, 440 * rows + 24), (30, 30, 30))
ImageDraw.Draw(sheet).text((6, 6), 'ep10 blue cup rims: RED = reference, GREEN = reference dropped onto table, CYAN = bob1 rollout '
                                   '(top + bottom rings)', fill=(255, 255, 255))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 380, 24 + (i // cols) * 440))
pth = os.path.join(OUT, 'sk10a2_rims.jpg')
sheet.save(pth, quality=88)
RES['rims_sheet'] = pth
with open(os.path.join(OUT, 'sk10a2.json'), 'w') as fh:
    json.dump(RES, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
log('done')
