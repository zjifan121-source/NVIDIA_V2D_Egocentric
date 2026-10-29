"""S3e refine (pathb/stages/s3e_refine.py, §14.4 R1): which frames go to the refiner, from which pose; the Mac skips."""
import json

import cv2
import numpy as np

from pathb.runner import Context
from pathb.stages import s3e_refine as E


def test_frames_with_mask_and_depth_from_the_settled_pose(tmp_path, monkeypatch):
    root = tmp_path / "episode_000041"
    for sub in ("p1b_masks/obj_0", "p1d_hands/depth"):
        (root / sub).mkdir(parents=True)
    (root / "p1b_masks/prompts.json").write_text(json.dumps({"tracks": {"obj_0": {"text": "white pot"}}}))
    for t in range(6):
        if t != 2:  # no mask at frame 2
            cv2.imwrite(str(root / f"p1b_masks/obj_0/{t:06d}.png"), np.full((8, 8), 255, np.uint8))
        if t != 4:  # no depth at frame 4
            cv2.imwrite(str(root / f"p1d_hands/depth/{t:06d}.png"), np.ones((8, 8), np.uint16))
    ctx = Context({"episode": 41, "split": "public", "objects": [{"name": "white_pot"}], "num_frames": 6}, root, {})
    P = np.tile(np.eye(4), (6, 1, 1))
    P[:, 0, 3] = np.arange(6) * 0.1
    P[5] = np.nan  # an unsolved frame
    c2w = np.tile(np.eye(4), (6, 1, 1))
    c2w[:, 2, 3] = 1.0
    monkeypatch.setattr(E.B, "lost_frames", lambda root: {"obj_0": frozenset({1})})  # held by another object
    frames, poses = E.requested(ctx, 0, "white_pot", P, c2w)
    assert frames == [0, 3]
    assert np.allclose(np.asarray(poses["3"]), np.linalg.inv(c2w[3]) @ P[3])


def test_skipped_without_a_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(E.shutil, "which", lambda name: None)
    ctx = Context({"episode": 41, "split": "public", "objects": [{"name": "white_pot"}], "num_frames": 6}, tmp_path, {})
    assert E.run(ctx).skip
