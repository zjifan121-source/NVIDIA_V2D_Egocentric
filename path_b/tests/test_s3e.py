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


def record(frame, iou=(0.80, 0.82), res=(0.010, 0.008), step=(0.005, 2.0), logits=(100.0, 100.5), pose=None):
    return {"frame": frame, "start": {"iou": iou[0], "residual_m": res[0]}, "refined": {"iou": iou[1], "residual_m": res[1]},
            "step_m": step[0], "step_deg": step[1], "kept": True, "logits": list(logits),
            "refined_pose": (np.eye(4) if pose is None else pose).tolist()}


def test_the_keep_rule_needs_the_scorer_no_iou_drop_and_a_small_step(monkeypatch):
    assert E.keeps(record(0))
    assert E.keeps(record(0, res=(0.010, 0.012)))  # the depth residual is not asked by default
    assert not E.keeps(record(0, iou=(0.80, 0.79)))  # the mask IoU drops
    assert not E.keeps(record(0, step=(0.05, 2.0)))  # a 5 cm move is a re-registration
    assert not E.keeps(record(0, logits=(100.5, 100.0)))  # the scorer prefers the start
    assert not E.keeps(record(0, logits=(None, None)))  # not scored
    monkeypatch.setattr(E, "MIN_GAIN_M", 0.001)  # P4's first rule
    assert not E.keeps(record(0, res=(0.010, 0.0095)))  # 0.5 mm better: under the bar
    monkeypatch.setattr(E, "SCORE_MARGIN", None)
    assert E.keeps(record(0, logits=(100.5, 100.0)))


def test_refined_poses_replace_s3d_on_moving_frames_only():
    P = np.tile(np.eye(4), (6, 1, 1))
    c2w = np.tile(np.eye(4), (6, 1, 1))
    c2w[:, 2, 3] = 1.0
    moving = E.moving_frames([{"kind": "static", "start": 0, "end": 2}, {"kind": "grasp", "start": 3, "end": 5}], 6)
    lift = np.eye(4)
    lift[2, 3] = 0.004
    rows = [record(t, pose=lift) for t in range(6)]
    rows[4] = record(4, logits=(100.5, 100.0), pose=lift)  # the scorer prefers the start
    R, kept = E.regate(rows, P, c2w, moving)
    assert kept == [3, 5]
    assert np.allclose(R[3], c2w[3] @ lift) and np.allclose(R[1], np.eye(4)) and np.allclose(R[4], np.eye(4))


def offline_episode(tmp_path, P):
    """An episode folder with a GPU pass's records in s3e_refine/ and no GPU here."""
    from pathb.geometry import write_transform
    root = tmp_path / "episode_000005"
    for sub in ("p1b_masks/obj_0", "p4_export/camera_to_world", "s3d_settle/white_pot", "s3b_segments/white_pot",
                "s3e_refine/white_pot"):
        (root / sub).mkdir(parents=True)
    c2w = np.tile(np.eye(4), (6, 1, 1))
    c2w[:, 2, 3] = 1.0
    for t in range(6):
        write_transform(root / f"p4_export/camera_to_world/{t:06d}.json", c2w[t])
    np.savez(root / "s3d_settle/white_pot/object_to_world.npz", object_to_world=P)
    (root / "s3b_segments/white_pot/segments.json").write_text(json.dumps([{"kind": "grasp", "start": 0, "end": 5}]))
    start = {str(t): (np.linalg.inv(c2w[t]) @ P[t]).tolist() for t in range(6)}
    (root / "s3e_refine/white_pot/request.json").write_text(json.dumps({"frames": list(range(6)), "poses": start}))
    lift = np.linalg.inv(c2w[2]) @ P[2]
    lift[2, 3] += 0.004
    rows = [record(t, pose=np.linalg.inv(c2w[t]) @ P[t]) for t in range(6)]
    rows[2] = record(2, pose=lift)
    (root / "s3e_refine/white_pot/refine_meta.json").write_text(json.dumps({"frames": rows, "scorer_error": None}))
    return Context({"episode": 5, "split": "evaluation", "objects": [{"name": "white_pot"}], "num_frames": 6}, root, {})


def test_the_mac_reapplies_the_rule_to_a_gpu_pass_and_drops_a_stale_one(tmp_path, monkeypatch):
    monkeypatch.setattr(E.shutil, "which", lambda name: None)
    P = np.tile(np.eye(4), (6, 1, 1))
    P[:, 0, 3] = np.arange(6) * 0.01
    ctx = offline_episode(tmp_path, P)
    result = E.run(ctx)
    assert not result.skip
    R = np.load(ctx.root / "s3e_refine/white_pot/object_to_world.npz")["object_to_world"]
    assert np.isclose(R[2, 2, 3], P[2, 2, 3] + 0.004) and np.allclose(R[3], P[3])
    moved = P.copy()  # S3d changed since the GPU pass: its records are another track's
    moved[:, 1, 3] += 0.01
    np.savez(ctx.root / "s3d_settle/white_pot/object_to_world.npz", object_to_world=moved)
    E.run(ctx)
    assert not (ctx.root / "s3e_refine/white_pot/object_to_world.npz").exists()
