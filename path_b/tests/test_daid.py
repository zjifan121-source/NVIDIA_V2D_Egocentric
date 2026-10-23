"""pathb/daid.py: the reference at 50 Hz in DaiD's world and DaiD's result back at video frames
(DaiD itself runs on the GPU machine); pathb/sharpa_map.json: the two Sharpa models agree."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pathb import daid


def track(n, fps):
    t = np.arange(n) / fps
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, 3] = np.stack([0.3 + 0.1 * np.sin(t), 0.2 * t, 0.05 + 0.02 * t], axis=1)
    T[:, :3, :3] = Rotation.from_rotvec(np.stack([0.1 * t, 0.2 * np.sin(t), 0.3 * t], axis=1)).as_matrix()
    return T


def mano(n):
    rng = np.random.default_rng(0)
    out = {}
    for side in ("right", "left"):
        out[f"{side}_joints"] = rng.normal(0, 0.05, (n, 21, 3)) + [0.3, 0.0, 0.2]
        out[f"{side}_vertices"] = rng.normal(0, 0.05, (n, 778, 3))
        out[f"{side}_faces"] = np.zeros((1538, 3), int)
        out[f"{side}_global_orient"] = rng.normal(0, 0.2, (n, 3))
        out[f"{side}_finger_pose"] = rng.normal(0, 0.1, (n, 45))
        out[f"{side}_betas"] = np.zeros(10)
        out[f"{side}_valid"] = np.ones(n, bool)
    out["left_valid"][: n // 2] = False  # left hand on only half the frames: still bimanual
    return out


def test_prepare_and_read_back_round_trip(tmp_path):
    fps, n = 20.25411910142744, 60
    T = track(n, fps)
    meta = daid.prepare(T, mano(n), tmp_path, fps)
    z = np.load(tmp_path / "daid_input.npz")
    assert meta["embodiment"] == "bimanual" and len(z["qpos_obj"]) == meta["n_ref"]
    assert np.allclose(z["qpos_obj"][0, :2], 0.0)  # object frame-0 xy at DaiD's origin
    assert np.isclose(z["qpos_obj"][0, 2], T[0, 2, 3])  # the table stays the floor at z = 0
    assert meta["n_ref"] >= ((n - 1) / fps + daid.PAD_S) * daid.REF_HZ  # padded by the horizon past the last frame
    # DaiD's output: qpos per 5 ms sim step, robot joints then the object (pos + wxyz)
    t_sim = np.arange(0, (meta["n_ref"] - 1) / daid.REF_HZ, 0.005)
    t_sim = t_sim[: len(t_sim) // 10 * 10]
    obj = np.concatenate([daid.lerp(z["qpos_obj"][:, :3], z["t_ref"], t_sim),
                          daid.slerp(Rotation.from_quat(z["qpos_obj"][:, [4, 5, 6, 3]]), z["t_ref"], t_sim).as_quat()[:, [3, 0, 1, 2]]],
                         axis=1)
    qpos = np.concatenate([np.zeros((len(t_sim), 56)), obj], axis=1).reshape(-1, 10, 63)  # 10 steps per record
    np.savez(tmp_path / "trajectory_mjwp.npz", qpos=qpos, time=t_sim.reshape(-1, 10))
    back = daid.read_back(tmp_path / "trajectory_mjwp.npz", [f"j{i}" for i in range(63)], meta)
    assert back["covered_frames"] == n
    assert np.abs(back["object_to_world"][:, :3, 3] - T[:, :3, 3]).max() < 1e-3
    err = [Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude() for a, b in zip(back["object_to_world"], T)]
    assert max(err) < np.radians(0.5) and back["robot_qpos"].shape == (n, 56)  # two slerp resamplings


def test_sharpa_joints_match_flash_chord():
    m = json.loads((Path(__file__).parents[1] / "pathb/sharpa_map.json").read_text())
    assert m["unmatched"] == []
    for side in ("right", "left"):
        assert len(m[side]["finger_joints"]) == 22
        assert all(j["daid_limits"] == j["flash_chord_limits"] for j in m[side]["finger_joints"])


def test_two_objects_round_trip(tmp_path):
    """Phase 2: right_object (roster-first) and left_object, DaiD's last 14 qpos numbers."""
    fps, n = 20.25411910142744, 40
    A, B = track(n, fps), track(n, fps)
    B[:, :3, 3] += [0.2, -0.1, 0.0]
    meta = daid.prepare(A, mano(n), tmp_path, fps, second=B)
    z = np.load(tmp_path / "daid_input.npz")
    assert meta["objects"] == 2 and meta["embodiment"] == "bimanual"
    assert np.allclose(z["qpos_obj_left"][0, :2], [0.2, -0.1])  # same DaiD world as the first object
    t_sim = np.arange(0, (meta["n_ref"] - 1) / daid.REF_HZ, 0.005)
    t_sim = t_sim[: len(t_sim) // 10 * 10]

    def obj(q):
        return np.concatenate([daid.lerp(q[:, :3], z["t_ref"], t_sim),
                               daid.slerp(Rotation.from_quat(q[:, [4, 5, 6, 3]]), z["t_ref"], t_sim).as_quat()[:, [3, 0, 1, 2]]], axis=1)
    qpos = np.concatenate([np.zeros((len(t_sim), 56)), obj(z["qpos_obj"]), obj(z["qpos_obj_left"])], axis=1)
    np.savez(tmp_path / "trajectory_mjwp.npz", qpos=qpos.reshape(-1, 10, 70), time=t_sim.reshape(-1, 10))
    back = daid.read_back(tmp_path / "trajectory_mjwp.npz", [f"j{i}" for i in range(70)], meta)
    assert np.abs(back["object_to_world"][:, :3, 3] - A[:, :3, 3]).max() < 1e-3
    assert np.abs(back["second_object_to_world"][:, :3, 3] - B[:, :3, 3]).max() < 1e-3
    assert back["robot_qpos"].shape == (n, 56)
