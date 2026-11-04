"""Re-level a cleaned bundle so the depth-fitted table normal becomes +Z, and write the matching seated support spec.
Rotation about the fitted plane point c, mapping the unit normal n to +Z (smallest rotation). Camera-to-world, every object-to-world and the
stored hand wrist-to-world transforms get T = [R | c - R c]; camera-relative arrays (hands, object-to-camera, MANO) are unchanged, meshes unchanged.
The support spec: cylinder centres get T, then every support is re-seated at frame 0 in the new world (T_top = min object bottom - 2 mm; a disk sits
2 mm under its own object). No GT, no scans: n and c come from the depth plane fit (review session, table_depth_check.py).
usage: relevel.py <ep> <src bundle dir> <nx> <ny> <nz> <cx> <cy> <cz> <out root>"""
import sys, json, shutil, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ep = int(sys.argv[1]); src = Path(sys.argv[2]); n = np.array([float(x) for x in sys.argv[3:6]]); c = np.array([float(x) for x in sys.argv[6:9]]); out = Path(sys.argv[9])
SEAT = Path('/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_support_seating_v1/out')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
n = n / np.linalg.norm(n)
if n[2] < 0: n = -n
z = np.array([0.0, 0.0, 1.0]); ax = np.cross(n, z); s = np.linalg.norm(ax); ang = float(np.arctan2(s, float(n @ z)))
R = np.eye(3) if s < 1e-12 else Rotation.from_rotvec(ax / s * ang).as_matrix()
T = np.eye(4); T[:3, :3] = R; T[:3, 3] = c - R @ c
assert np.allclose(R @ n, z, atol=1e-9)
B = out / 'stage/scenes' / f'episode_{ep:06d}/bundle'
if B.exists(): shutil.rmtree(B)
shutil.copytree(src, B, symlinks=False)
d = dict(np.load(src / 'result.npz', allow_pickle=False)); changed = []
for k in list(d):
    if k.endswith('_to_world_transform'):
        X = d[k].astype(np.float64); d[k] = (T[None] @ X).astype(d[k].dtype); changed.append(k)
np.savez(B / 'result.npz', **d)
names = [str(x) for x in d['object_names']] if 'object_names' in d else None
def verts(p): return np.array([l.split()[1:4] for l in open(p) if l.startswith('v ')], float)
def slot(k): return 0 if k == 'object_to_world_transform' else int(k.split('_')[1])
bottoms = {}
for i, k in enumerate(sorted([k for k in d if k.endswith('_to_world_transform') and k.startswith('object')], key=slot)):
    j = slot(k); mesh = B / ('mesh.obj' if j == 0 else f'mesh_{j}.obj'); sc = float(d.get('object_scale' if j == 0 else f'object_{j}_scale', 1.0))
    v = verts(mesh) * sc; P = d[k][0].astype(float); w = v @ P[:3, :3].T + P[:3, 3]; bottoms[names[i] if names else k] = float(w[:, 2].min())
T_top = min(bottoms.values()) - 0.002
spec = json.loads((SEAT / f'episode_{ep:06d}/support_v1/support_in_bundle_world.json').read_text())
new = []
for cy in spec['cylinders']:
    ctr = (T @ np.r_[cy['center_bundle_world_m'], 1.0])[:3]; h = cy['height_m']
    if cy.get('kind', 'table') == 'table': ctr[2] = T_top - h / 2
    else:
        ob = cy.get('object') or cy.get('for_object'); gap = bottoms.get(ob, T_top + 0.002) - (T_top + 0.002)
        if gap < 0.005: continue  # the object now sits within 5 mm of the table: no disk
        ctr[2] = (bottoms.get(ob, T_top + 0.002) - 0.002) - h / 2
    new.append(dict(cy, center_bundle_world_m=[float(x) for x in ctr]))
spec['cylinders'] = new
spec['bundle_sha256'] = {k: sha(B / k) for k in spec['bundle_sha256']}
spec['relevel'] = dict(n=[float(x) for x in n], c=[float(x) for x in c], rotation_deg=round(float(np.degrees(ang)), 3), transform=T.tolist(), arrays_changed=changed, source_bundle=str(src), table_top_new=T_top, object_bottoms_f0=bottoms, note='camera, object and wrist world poses rotated about the depth-plane point so the table normal is +Z; supports re-seated in the new world')
spec['seating'] = dict(spec['seating'], table_top_after_relevel=T_top)
o = out / 'seat' / f'episode_{ep:06d}/support_v1'; o.mkdir(parents=True, exist_ok=True); (o / 'support_in_bundle_world.json').write_text(json.dumps(spec, indent=2) + '\n')
(B / 'RELEVEL.json').write_text(json.dumps(spec['relevel'], indent=1) + '\n')
print('RELEVEL', ep, 'rotation_deg', round(float(np.degrees(ang)), 3), 'arrays', changed, 'bottoms', {k: round(v, 4) for k, v in bottoms.items()}, 'T_top', round(T_top, 4), 'cylinders', [(x.get('kind'), [round(v, 3) for v in x['center_bundle_world_m']], round(x['radius_m'], 3)) for x in new])
