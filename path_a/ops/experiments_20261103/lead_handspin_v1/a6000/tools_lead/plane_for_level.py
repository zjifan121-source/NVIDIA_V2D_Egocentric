"""Tilt of the depth-measured table relative to world Z, per episode, at frame 0. Depth, masks and camera poses only: no GT,
no scans. Reuses the plane fit of table_depth_check.py. Tilt means the angle between the plane normal and world Z.
Usage: table_tilt_all.py FRAMES_ROOT ROSTER.json STAGE_SCENES OUT.json"""
import json, os, sys
import numpy as np, cv2
from scipy.spatial import cKDTree
sys.argv, args = sys.argv[:1] + ['x', 'y'], sys.argv[1:]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'table_depth_check.py')).read().split("root, out_path = sys.argv[1]")[0]
exec(src)
root, roster, scenes, out = args; RO = {int(k): v for k, v in json.load(open(roster))['episodes'].items()}; res = []
for ep in ([int(x) for x in os.environ['EPS'].split(',')] if os.environ.get('EPS') else sorted(RO)):
    d0 = f'{root}/ep{ep:02d}'; b = f'{scenes}/episode_{ep:06d}/bundle'
    if not (os.path.isdir(d0) and os.path.isfile(b + '/result.npz')):
        print(f'ep{ep}: no frames or bundle here', flush=True); continue
    d = np.load(b + '/result.npz', allow_pickle=False); K = json.load(open(d0 + '/K.json')); names = [str(n) for n in d['object_names']] if 'object_names' in d.files else RO[ep]
    c2w = d['camera_to_world_transform'].astype(float); world = []
    for i in range(len(names)):
        P = d['object_to_world_transform' if i == 0 else f'object_{i}_to_world_transform'].astype(float); s = float(d['object_scale' if i == 0 else f'object_{i}_scale']); v = read_v(f'{b}/' + ('mesh.obj' if i == 0 else f'mesh_{i}.obj')) * s
        world.append(v @ P[0, :3, :3].T + P[0, :3, 3])
    tree = cKDTree(np.vstack([w[:, :2] for w in world])[::7]); rng = np.random.default_rng(ep)
    P_, info = points(root, ep, 0, c2w[0], K, names, tree)
    if len(P_) > N_PTS: P_ = P_[rng.choice(len(P_), N_PTS, replace=False)]
    a = ransac(P_, rng)
    if a is None: print(f'ep{ep}: no plane', flush=True); continue
    az = float(np.degrees(np.arctan2(a['n'][1], a['n'][0]))); r = dict(n=[float(x) for x in a['n']], c=[float(x) for x in a['c']], obj_centroid_f0=[float(x) for x in np.vstack(world).mean(0)], episode=ep, tilt_deg=round(a['tilt_deg'], 2), toward_deg=round(az, 0), z=round(float(a['c'][2]), 4), rms_cm=round(a['rms_cm'], 3), frac=round(a['frac'], 3), near_px=info['near_pixels'])
    res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open(out, 'w'), indent=1)
