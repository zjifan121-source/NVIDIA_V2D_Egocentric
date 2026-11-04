"""Rest hold for a spinroll bundle: every object keeps its frame-0 pose until its first attach frame (from ATTACH_RESPIN.json), blending into the
tracked pose over the three frames before it. Removes the early drift of an unheld object (spoon starting to move before the hand reaches it).
Hands untouched; frame 0 untouched; meshes untouched. usage: rest_hold_patch.py <src bundle> <out bundle> [--blend 3]"""
import sys, json, shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
src = Path(sys.argv[1]); out = Path(sys.argv[2]); blend = int(sys.argv[sys.argv.index('--blend') + 1]) if '--blend' in sys.argv else 3
if out.exists(): shutil.rmtree(out)
shutil.copytree(src, out, symlinks=False)
d = dict(np.load(out / 'result.npz', allow_pickle=False)); att = json.loads((src / 'ATTACH_RESPIN.json').read_text())
names = [str(x) for x in d['object_names']]; c2w = d['camera_to_world_transform'].astype(np.float64); w2c = np.linalg.inv(c2w)
rec = []
for i, name in enumerate(names):
    key = 'object_to_world_transform' if i == 0 else f'object_{i}_to_world_transform'; cam = key.replace('to_world', 'to_camera')
    holds = att['attached'].get(f'object_{i}', [])
    if not holds: rec.append(dict(object=name, first_attach=None, changed=False)); continue
    fa = min(int(h[1]) for h in holds); T = d[key].astype(np.float64); T0 = T[0].copy(); n = len(T)
    start = max(1, fa - blend); moved_before = float(np.linalg.norm(T[1:fa, :3, 3] - T0[:3, 3], axis=1).max()) if fa > 1 else 0.0
    newT = T.copy()
    for t in range(1, fa):
        if t < start: newT[t] = T0
        else:
            w = (t - start + 1) / (fa - start + 1)
            R = Slerp([0, 1], Rotation.from_matrix(np.stack([T0[:3, :3], T[t, :3, :3]])))(w).as_matrix()
            newT[t, :3, :3] = R; newT[t, :3, 3] = (1 - w) * T0[:3, 3] + w * T[t, :3, 3]
    d[key] = newT.astype(d[key].dtype)
    if cam in d: d[cam] = (w2c @ newT).astype(d[cam].dtype)
    jump = float(np.linalg.norm(newT[fa, :3, 3] - newT[fa - 1, :3, 3])) if fa < n else 0.0
    rec.append(dict(object=name, first_attach=fa, changed=True, held_until=start - 1, blend_frames=[start, fa - 1], tracked_drift_before_attach_cm=round(moved_before * 100, 2), step_at_attach_cm=round(jump * 100, 2)))
np.savez(out / 'result.npz', **d)
(out / 'REST_HOLD.json').write_text(json.dumps(dict(rule='object pose = frame-0 pose until its first attach frame, blended over the previous %d frames; hands, frame 0 and meshes unchanged' % blend, objects=rec), indent=1) + '\n')
print('REST_HOLD', json.dumps(rec))
