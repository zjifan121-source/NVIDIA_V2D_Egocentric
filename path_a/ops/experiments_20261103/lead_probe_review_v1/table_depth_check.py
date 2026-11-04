"""Measure the real table height from stereo depth and compare it with our seated support and mesh bottoms.

For each episode and chosen frame: unproject depth, drop pixels inside dilated object and hand masks and pixels more than
40 cm (XY) from the objects' frame-0 footprints, then RANSAC near-horizontal planes in the bundle's Z-up world. The top two
planes are reported, because a shelf or rack can out-vote the table. The same table patch is measured at frame 0 and at a
late frame: their disagreement bounds how far camera poses and depth can be trusted. Inputs: depth, masks, camera poses
and our meshes. No GT, no scans. Depth encoding is the pipeline's inverse uint16 metres.
Usage: table_depth_check.py FRAMES_ROOT OUT.json EP=SCENE_DIR [EP=SCENE_DIR ...]   (SCENE_DIR holds bundle/ and support_v1/)"""
import json, sys, os, glob
from pathlib import Path
import numpy as np, cv2
from scipy.spatial import cKDTree
DILATE, FAR_M, TH_M, MAX_TILT_DEG, N_PTS, N_ITER = 15, 0.40, 0.01, 20.0, 80000, 4000

def read_depth(p):
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED); d = np.full(raw.shape, np.nan, np.float64); v = raw > 0
    d[v] = 65535.0 / raw[v].astype(np.float64) - 1.0; return d

def read_v(p): return np.array([l.split()[1:4] for l in open(p) if l.startswith('v ')], float)

def plane_fit(P):
    c = P.mean(0); n = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    return (n if n[2] > 0 else -n), c

def ransac(P, rng):
    best = None; cos_t = np.cos(np.radians(MAX_TILT_DEG))
    for _ in range(N_ITER):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]; n = np.cross(b - a, c - a); L = np.linalg.norm(n)
        if L < 1e-9: continue
        n = n / L; n = n if n[2] > 0 else -n
        if n[2] < cos_t: continue
        k = int((np.abs((P - a) @ n) < TH_M).sum())
        if best is None or k > best[0]: best = (k, n, a)
    if best is None: return None
    inl = np.abs((P - best[2]) @ best[1]) < TH_M
    for _ in range(3):
        n, c = plane_fit(P[inl]); inl = np.abs((P - c) @ n) < TH_M
    r = (P[inl] - c) @ n
    return dict(n=n, c=c, inliers=int(inl.sum()), frac=float(inl.mean()), rms_cm=float(np.sqrt((r ** 2).mean()) * 100), tilt_deg=float(np.degrees(np.arccos(n[2]))), mask=inl)

def z_at(pl, xy):
    n, c = pl['n'], pl['c']; return float(c[2] - (n[0] * (xy[0] - c[0]) + n[1] * (xy[1] - c[1])) / n[2])

def points(root, ep, t, c2w, K, obj_names, foot_tree):
    d = read_depth(f'{root}/ep{ep:02d}/depth/{t:06d}.png'); h, w = d.shape; m = np.zeros((h, w), np.uint8)
    for n in obj_names:
        f = f'{root}/ep{ep:02d}/masks/{n}/{t:06d}.png'
        if os.path.isfile(f): m |= (cv2.imread(f, cv2.IMREAD_UNCHANGED) > 0).astype(np.uint8) if cv2.imread(f, cv2.IMREAD_UNCHANGED).ndim == 2 else (cv2.imread(f, cv2.IMREAD_UNCHANGED).max(-1) > 0).astype(np.uint8)
    f = f'{root}/ep{ep:02d}/hands/{t:06d}.png'
    if os.path.isfile(f):
        hm = cv2.imread(f, cv2.IMREAD_UNCHANGED); m |= ((hm if hm.ndim == 2 else hm.max(-1)) > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1)); m = cv2.dilate(m, k) > 0
    v, u = np.nonzero(np.isfinite(d) & (d > 0.1) & (d < 4.0) & ~m); z = d[v, u]
    X = np.stack([(u - K['cx']) * z / K['fx'], (v - K['cy']) * z / K['fy'], z], 1)
    W = X @ c2w[:3, :3].T + c2w[:3, 3]
    near = foot_tree.query(W[:, :2], k=1)[0] <= FAR_M
    return W[near], dict(valid_pixels=int(len(z)), near_pixels=int(near.sum()), masked_frac=round(float(m.mean()), 3))

root, out_path = sys.argv[1], sys.argv[2]; rows, meta = [], {}
for arg in sys.argv[3:]:
    ep, scene = arg.split('=', 1); ep = int(ep); scene = Path(scene); b = scene / 'bundle'; d = np.load(b / 'result.npz', allow_pickle=False)
    K = json.load(open(f'{root}/ep{ep:02d}/K.json')); names = [str(n) for n in d['object_names']]; spec = json.load(open(scene / 'support_v1' / 'support_in_bundle_world.json'))
    c2w = d['camera_to_world_transform'].astype(float); valid = d['camera_is_valid'].astype(bool); T = len(c2w); world = {}
    for i, n in enumerate(names):
        P = d['object_to_world_transform' if i == 0 else f'object_{i}_to_world_transform'].astype(float); s = float(d['object_scale' if i == 0 else f'object_{i}_scale']); v = read_v(b / ('mesh.obj' if i == 0 else f'mesh_{i}.obj')) * s
        world[n] = v @ P[0, :3, :3].T + P[0, :3, 3]
    tree = cKDTree(np.vstack([w[:, :2] for w in world.values()])[::7])
    late = max(t for t in range(T) if valid[t] and os.path.isfile(f'{root}/ep{ep:02d}/depth/{t:06d}.png'))
    meta[ep] = dict(scene=str(scene), late_frame=late, K_matches_bundle=bool(np.allclose([K['fx'], K['fy'], K['cx'], K['cy']], np.asarray(d['camera_intrinsics']).reshape(-1)[[0, 4, 2, 5]] if np.asarray(d['camera_intrinsics']).size == 9 else np.asarray(d['camera_intrinsics']).reshape(-1)[:4], atol=1.0)) if 'camera_intrinsics' in d.files else None, cylinders=spec['cylinders'], frames={})
    fits = {}
    for t in (0, 10, late):
        if not valid[t]: continue
        rng = np.random.default_rng(1000 * ep + t); P, info = points(root, ep, t, c2w[t], K, names, tree)
        if len(P) > N_PTS: P = P[rng.choice(len(P), N_PTS, replace=False)]
        a = ransac(P, rng); rest = P[~a['mask']] if a else P; b2 = ransac(rest, rng) if a and len(rest) > 2000 else None
        fits[t] = (a, b2); meta[ep]['frames'][t] = dict(info, first=None if a is None else {k: round(a[k], 3) for k in ('frac', 'rms_cm', 'tilt_deg')}, first_z_mean=None if a is None else round(float(a['c'][2]), 4), second=None if b2 is None else {k: round(b2[k], 3) for k in ('frac', 'rms_cm', 'tilt_deg')}, second_z_mean=None if b2 is None else round(float(b2['c'][2]), 4))
    for n, w in world.items():
        xy = w[:, :2].mean(0); bottom = float(w[:, 2].min()); under = [c for c in spec['cylinders'] if np.linalg.norm(np.array(c['center_bundle_world_m'][:2]) - xy) <= c['radius_m']]
        top = lambda c: c['center_bundle_world_m'][2] + c['height_m'] / 2
        table = [top(c) for c in spec['cylinders'] if c.get('kind') == 'table']; here = sorted(((top(c), c.get('kind')) for c in under), reverse=True)
        r = dict(episode=ep, object=n, mesh_bottom_z=round(bottom, 4), support_table_top_z=round(max(table), 4) if table else None, support_under_object=[[round(a_, 4), k_] for a_, k_ in here])
        for t, (a, b2) in fits.items():
            r[f'z{t}'] = None if a is None else round(z_at(a, xy), 4); r[f'z2_{t}'] = None if b2 is None else round(z_at(b2, xy), 4)
        rows.append(r); print(json.dumps(r), flush=True)
    print(json.dumps({k: v for k, v in meta[ep].items() if k != 'cylinders'}), flush=True)
json.dump(dict(rows=rows, meta=meta), open(out_path, 'w'), indent=1, default=float)
