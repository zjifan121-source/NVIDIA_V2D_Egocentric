"""pathb/fk.py: numpy FK reproduces the retargeter's robot_{side}_frames (ep41 task file, stages 2-3
on the VM) and agrees with FlashCHORD's sharpawave URDF on every shared frame."""
from pathlib import Path

import numpy as np

from pathb import fk

FIXTURE = Path(__file__).parent / "fixtures/ep41_sharpa_frames.npz"  # 6 frames of the ep41 s23 parquet


def test_frames_match_the_retargeter():
    z = np.load(FIXTURE)
    for side in ("right", "left"):
        hand = fk.sharpa(side)
        assert hand.names == list(z[f"{side}_frame_names"])  # the 67 names, in order
        q = dict(zip(z[f"{side}_finger_joint_names"], z[f"{side}_finger_joints"].T))
        F = hand.frames(z[f"{side}_wrist_position"], z[f"{side}_wrist_wxyz"], q)
        ref = z[f"{side}_frames"]
        assert np.abs(F[..., :3] - ref[..., :3]).max() < 1e-6  # float32 storage
        assert (1 - np.abs((F[..., 3:] * ref[..., 3:]).sum(-1))).max() < 1e-6


def test_urdf_and_mjcf_agree():
    rng = np.random.default_rng(0)
    for side in ("right", "left"):
        m, u = fk.sharpa(side, "mjcf"), fk.sharpa(side, "urdf")
        assert sorted(m.joints) == sorted(u.joints) and len(u.joints) == 22
        q = {j: rng.uniform(-0.3, 1.2, 20) for j in m.joints}
        wp, wq = rng.normal(0, 0.3, (20, 3)), rng.normal(size=(20, 4))
        wq /= np.linalg.norm(wq, axis=1, keepdims=True)
        pm, pu = m.poses(wp, wq, q), u.poses(wp, wq, q)
        shared = [n for n in m.names if n in pu and n != "universe"]
        assert len(shared) == 46  # root_joint, 22 joints, 23 links
        for n in shared:
            assert np.abs(pm[n][:, :3, 3] - pu[n][:, :3, 3]).max() < 1e-6
            assert np.abs(pm[n][:, :3, :3] - pu[n][:, :3, :3]).max() < 1e-5
        for digit in ("thumb", "index", "middle", "ring", "pinky"):  # URDF fingertip link = MJCF tip site
            assert np.abs(pm[f"{side}_{digit}_tip_site"][:, :3, 3] - pu[f"{side}_{digit}_fingertip"][:, :3, 3]).max() < 1e-6
