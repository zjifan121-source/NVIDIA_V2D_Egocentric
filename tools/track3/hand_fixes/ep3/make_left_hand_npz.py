"""Build the ep3 merged bundle with the correct-handedness left hand (hand fixes, 2 Nov 2026).

Copies the baseline merged result.npz (ken_c_scored_attach/bundle_merged) and swaps in only the seven left-hand
arrays from the left-label HaMeR re-run of the physical left-hand track (track 3); right hand, objects and camera stay
bit-identical. Use the output as H5_MERGED_NPZ when building the scene.
    python make_left_hand_npz.py <baseline result.npz> <left re-run result.npz> <out .npz>
"""
import hashlib
import sys

import numpy as np

base, cand, out = sys.argv[1:4]
KEYS = ["hand_left_betas", "hand_left_finger_pose", "hand_left_scale", "hand_left_wrist_orient_in_camera",
        "hand_left_wrist_to_camera_transform", "hand_left_wrist_to_world_transform", "hand_left_wrist_trans_in_camera"]
a, c = dict(np.load(base)), dict(np.load(cand))
merged = dict(a)
for k in KEYS:
    assert c[k].shape == a[k].shape and c[k].dtype == a[k].dtype, k
    merged[k] = c[k]
np.savez(out, **merged)
b = dict(np.load(out))
assert set(b) == set(a)
for k in a:
    assert np.array_equal(b[k], c[k] if k in KEYS else a[k]), k
print("ok", hashlib.sha256(open(out, "rb").read()).hexdigest())
