"""ep3: put Ken's correct-handedness left hand (camera-frame arrays from the left-label HaMeR re-run) into Wentao's bundle.

The hand arrays the loader reads are camera-relative (wrist_orient/trans_in_camera, finger_pose); world poses are
recomputed from Wentao's own camera. Checks first that Wentao's camera-frame right hand matches Ken's (same image
evidence, same metric scale), so the left hand lands in the same place relative to the pot.
python ep3_lefthand_wentao.py KEN_NPZ WENTAO_BUNDLE_DIR OUT_BUNDLE_DIR"""
import os, sys, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
kp, wb, ob = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
k, w = np.load(kp), dict(np.load(wb / 'result.npz'))
v = k['hand_right_is_valid'].astype(bool) & w['hand_right_is_valid'].astype(bool)
tk, tw = k['hand_right_wrist_trans_in_camera'][v].astype(float), w['hand_right_wrist_trans_in_camera'][v].astype(float)
d = np.linalg.norm(tk - tw, axis=1) * 100
ratio = np.median(np.linalg.norm(tw, axis=1) / np.linalg.norm(tk, axis=1))
rk = Rotation.from_rotvec(k['hand_right_wrist_orient_in_camera'][v]); rw = Rotation.from_rotvec(w['hand_right_wrist_orient_in_camera'][v])
ang = np.degrees((rk * rw.inv()).magnitude())
print(f'right hand in camera, Ken vs Wentao: median {np.median(d):.2f} cm, p90 {np.percentile(d, 90):.2f} cm, '
      f'depth ratio {ratio:.3f}, rotation median {np.median(ang):.1f} deg p90 {np.percentile(ang, 90):.1f}')
if not (np.median(d) < 2 and abs(ratio - 1) < 0.05):
    sys.exit('camera-frame right hands disagree: not transplanting')
for key in ('hand_left_betas', 'hand_left_wrist_orient_in_camera', 'hand_left_wrist_trans_in_camera',
            'hand_left_finger_pose', 'hand_left_scale', 'hand_left_is_valid'):
    w[key] = k[key].astype(w[key].dtype) if key in w else k[key]
w2c = np.tile(np.eye(4), (len(w['hand_left_is_valid']), 1, 1))
w2c[:, :3, :3] = Rotation.from_rotvec(w['hand_left_wrist_orient_in_camera'].astype(float)).as_matrix()
w2c[:, :3, 3] = w['hand_left_wrist_trans_in_camera']
w['hand_left_wrist_to_camera_transform'] = w2c.astype(np.float32)
w['hand_left_wrist_to_world_transform'] = (w['camera_to_world_transform'].astype(float) @ w2c).astype(np.float32)
ob.mkdir(parents=True, exist_ok=True)
for f in wb.iterdir():
    if f.name != 'result.npz' and not (ob / f.name).exists():
        os.link(f, ob / f.name)
np.savez(ob / 'result.npz', **w)
print('wrote', ob / 'result.npz')
