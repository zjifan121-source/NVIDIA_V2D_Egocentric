import json

import numpy as np

from pathb import spider_scene as S

CUBE_V = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05)])
CUBE_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                   [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
TABLE = 0.70
T = 40


def hand(at: np.ndarray, frames: int = T) -> np.ndarray:
    """MANO-ordered joints: wrist at `at`, fingers along +x, every tip 10 cm from the wrist."""
    j = np.zeros((frames, 21, 3))
    j[:] = at
    for f, k in enumerate((1, 5, 9, 13, 17)):  # finger bases: thumb, index, middle, ring, pinky
        for s in range(4):
            j[:, k + s] = at + np.array([0.025 * (s + 1), 0.02 * (f - 2), 0.0])
    return j


def scene(b_on: str = "table", right_on: str = "a", left_on: str = "b") -> S.Reference:
    """Cube a at x=0, cube b at x=0.3 (or stacked on a, or raised); hands touch them in frames 10-30."""
    pos = np.zeros((T, 2, 3))
    pos[:, 0] = [0.0, 0.0, TABLE + 0.05]
    pos[:, 1] = {"table": [0.3, 0.0, TABLE + 0.05], "a": [0.0, 0.0, TABLE + 0.15],
                 "raised": [0.3, 0.0, TABLE + 0.15]}[b_on]
    wxyz = np.zeros((T, 2, 4))
    wxyz[..., 0] = 1.0
    far = np.array([0.0, 1.0, 1.5])
    joints = {}
    for side, target in (("right", right_on), ("left", left_on)):
        j = hand(far)
        b = "ab".index(target)
        # tips reach the cube's top face: wrist 10 cm left of the face centre, raised to the top
        j[10:30] = hand(pos[0, b] + np.array([-0.10, 0.0, 0.05]), 20)
        joints[side] = j
    return S.Reference(fps=20.0, names=["a", "b"], obj_pos=pos, obj_wxyz=wxyz, joints=joints,
                       vertices=[CUBE_V, CUBE_V], faces=[CUBE_F, CUBE_F])


DISKS = [(np.array([-1.0, -1.0, TABLE - 0.02]), np.array([1.0, 1.0, TABLE]))]


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.where(q[:, :1] < 0, -1, 1)
    back = S.matrix_to_quat_wxyz(S.quat_wxyz_to_matrix(q))
    assert np.allclose(back, q, atol=1e-9)


def test_hand_pose_is_a_proper_frame_and_left_is_flipped():
    j = hand(np.array([0.1, 0.2, 0.3]), 3)
    right = S.hand_pose(j, "right")
    left = S.hand_pose(j, "left")
    R = S.quat_wxyz_to_matrix(right[0, 3:])
    assert np.allclose(right[:, :3], [0.1, 0.2, 0.3])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(R[:, 2], [1, 0, 0], atol=1e-9)  # z: wrist -> middle knuckle
    L = S.quat_wxyz_to_matrix(left[0, 3:])
    assert np.allclose(L[:, :2], -R[:, :2], atol=1e-9) and np.allclose(L[:, 2], R[:, 2], atol=1e-9)


def test_slots_follow_the_touching_hand():
    straight = S.assign_slots(scene(right_on="a", left_on="b"))
    assert straight["slots"] == {"a": "right", "b": "left"} and straight["guidance_safe"]
    crossed = S.assign_slots(scene(right_on="b", left_on="a"))
    assert crossed["slots"] == {"a": "left", "b": "right"} and crossed["guidance_safe"]


def test_one_hand_on_both_objects_makes_guidance_unsafe():
    ref = scene(right_on="a", left_on="a")
    info = S.assign_slots(ref)
    assert not info["guidance_safe"]


def test_support_table_stack_and_raised():
    on_table = S.support_plan(scene("table"), DISKS)
    assert np.isclose(on_table["z0"], TABLE) and on_table["supported"]
    assert {s["rests_on"] for s in on_table["objects"].values()} == {"table"}
    stacked = S.support_plan(scene("a"), DISKS)
    assert stacked["objects"]["b"]["rests_on"] == "a" and stacked["supported"]
    raised = S.support_plan(scene("raised"), DISKS)
    assert raised["objects"]["b"]["rests_on"] == "nothing" and not raised["supported"]


def test_no_disk_falls_back_to_the_lowest_object():
    plan = S.support_plan(scene("table"), [])
    assert np.isclose(plan["z0"], TABLE) and "no support disk" in plan["source"]


def test_spider_layout_shifts_the_table_to_zero(tmp_path):
    ref = scene(right_on="b", left_on="a")
    info = S.write_spider_inputs(ref, tmp_path, 13, S.support_plan(ref, DISKS), {"note": "test"})
    kp_dir = S.processed_dir(tmp_path, "mano", "episode_000013")
    kp = np.load(kp_dir / "trajectory_keypoints.npz")
    assert set(kp.files) >= {f"qpos_{k}_{s}" for k in ("wrist", "finger", "obj") for s in ("right", "left")}
    assert kp["qpos_finger_right"].shape == (T, 5, 7) and kp["qpos_obj_left"].shape == (T, 7)
    assert np.isclose(kp["qpos_obj_right"][0, 2], 0.05)  # cube b (right slot) centre 5 cm above the table
    assert np.isclose(kp["qpos_obj_left"][0, 0], 0.0)  # cube a in the left slot
    saved = json.loads((kp_dir.parent / "task_info.json").read_text())
    assert saved["right_object_mesh_dir"].endswith("episode_000013_b") and saved["v2d"]["note"] == "test"
    obj = (tmp_path / saved["left_object_mesh_dir"] / "visual.obj").read_text().splitlines()
    assert sum(line.startswith("v ") for line in obj) == 8 and sum(line.startswith("f ") for line in obj) == 12
    assert np.isclose(info["v2d"]["table_z_world_m"], TABLE)


def test_one_object_takes_the_right_slot(tmp_path):
    ref = scene()
    ref = S.Reference(ref.fps, ["a"], ref.obj_pos[:, :1], ref.obj_wxyz[:, :1], ref.joints,
                      ref.vertices[:1], ref.faces[:1])
    info = S.write_spider_inputs(ref, tmp_path, 41, S.support_plan(ref, DISKS))
    assert info["v2d"]["slots"] == {"a": "right"} and info["left_object_mesh_dir"] is None
    kp = np.load(S.processed_dir(tmp_path, "mano", "episode_000041") / "trajectory_keypoints.npz")
    assert np.allclose(kp["qpos_obj_left"], [0, 0, 0, 1, 0, 0, 0])  # SPIDER's empty-slot placeholder


def test_partition_row_is_read_in_body_order():
    ref = scene()
    row = {"fps": 20.0, "object_body_position": ref.obj_pos.tolist(), "object_body_wxyz": ref.obj_wxyz.tolist(),
           "mano_right_joints": ref.joints["right"].tolist(), "mano_left_joints": ref.joints["left"].tolist()}
    back = S.reference_from_partition(row, [(CUBE_V, CUBE_F)] * 2, ["a", "b"])
    assert back.names == ["a", "b"] and back.obj_pos.shape == (T, 2, 3) and back.frames == T


def test_merge_runner_points_the_lookup_at_path_b(tmp_path):
    from pathb import merge_runner as M
    bundle = tmp_path / "episode_000013/blue_cup/result_slam_gravity_aligned"
    bundle.mkdir(parents=True)
    (bundle / "result.npz").write_bytes(b"")
    assert M.path_b_lookup(tmp_path)(13, "blue_cup") == bundle.resolve()
    try:
        M.path_b_lookup(tmp_path)(13, "plastic_dish_rack")
    except SystemExit as e:
        assert "no Path B bundle" in str(e)
    else:
        raise AssertionError("a missing bundle must stop the runner")
