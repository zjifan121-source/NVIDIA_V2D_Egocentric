import numpy as np
import pytest

from pathb.stages.p3_scale_bridge import gt_trajectory_factor


def poses(positions):
    T = np.tile(np.eye(4), (len(positions), 1, 1))
    T[:, :3, 3] = positions
    return T.astype(np.float32)


def test_gt_factor_recovers_the_scale_of_the_motion_and_the_placement():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 120)
    true_a = np.stack([0.3 * t, 0.1 * np.sin(3 * t), 0.05 * t], axis=1)  # moves 32 cm
    true_b = np.tile([0.4, 0.2, 0.0], (120, 1))  # rests 45 cm away
    gt = np.zeros((120, 2, 7))
    gt[:, 0, :3], gt[:, 1, :3] = true_a + 1.0, true_b + 1.0  # another world origin
    gt[:, :, 6] = 1.0
    # the bundle's world is 1.25x too large and rotated 90 degrees about z
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    bundle_a, bundle_b = (true_a @ R.T) * 1.25, (true_b @ R.T) * 1.25
    data = {"camera_to_world_transform": np.zeros((120, 4, 4)),
            "object_to_world_transform": poses(bundle_a + rng.normal(0, 0.002, (120, 3))),
            "object_1_to_world_transform": poses(bundle_b)}
    slots = [("object_to_world_transform", "object_scale", "mesh.obj"),
             ("object_1_to_world_transform", "object_1_scale", "mesh_1.obj")]
    out = gt_trajectory_factor(gt, data, slots)
    assert out["factor"] == pytest.approx(0.8, abs=0.02)
    assert set(out["parts"]) == {"motion of object 0", "frame-0 distance between the objects"}


def test_gt_factor_is_none_when_nothing_moves():
    gt = np.zeros((50, 1, 7))
    gt[:, :, 6] = 1.0
    data = {"camera_to_world_transform": np.zeros((50, 4, 4)), "object_to_world_transform": poses(np.zeros((50, 3)))}
    assert gt_trajectory_factor(gt, data, [("object_to_world_transform", "s", "m")]) is None
