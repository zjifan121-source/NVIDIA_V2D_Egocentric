"""S5 mode B pieces (stages/s5_task.py, pathb/daid.py contact slots). Synthetic tests always run. The
data test round-trips the ep41 task file from the VM (stages 2-3 on run 4c2eac31): its own IK robot
handed in as if DaiD had produced it must come back column for column, with every contact on the
object; it runs with PATHB_DATA_TESTS=1 when that parquet is present."""
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from pathb import daid, fk
from pathb.paths import OUTPUTS
from pathb.stages import s5_task as S5


def poses(n, seed=0):
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.5, (n, 3))).as_matrix()
    T[:, :3, 3] = rng.normal(0, 0.2, (n, 3))
    return T


def cells_from(T):
    return {"object_body_position": T[:, None, :3, 3].tolist(),
            "object_body_wxyz": Rotation.from_matrix(T[:, :3, :3]).as_quat()[:, None, [3, 0, 1, 2]].tolist()}


def test_world_shift_is_measured_and_checked():
    T = poses(30)
    shifted = T.copy()
    shifted[:, :3, 3] += [0.1, -0.2, 1.0]
    assert np.allclose(S5.world_shift(cells_from(shifted), T), [0.1, -0.2, 1.0])
    turned = shifted.copy()
    turned[:, :3, :3] = Rotation.from_rotvec([0, 0, 0.1]).as_matrix() @ turned[:, :3, :3]
    with pytest.raises(RuntimeError):
        S5.world_shift(cells_from(turned), T)


def test_contact_slots_keep_the_deepest_contact_per_link():
    names = np.array(["right_hand_C_MC", "right_index_DP", "left_thumb_DP", "right_pinky_MC"])
    realised = {"video_time": np.arange(3) / 20.0, "body_names": names,
                "contact_frame": np.array([0, 0, 1, 2, 2]), "contact_body": np.array([1, 1, 2, 3, 0]),
                "contact_dist": np.array([0.001, -0.002, 0.0, 0.0, -0.001]),
                "contact_object_point": np.arange(15, dtype=float).reshape(5, 3),
                "contact_normal": np.tile([0.0, 0, 1], (5, 1)), "contact_robot_local": np.zeros((5, 3))}
    right = daid.contact_slots(realised, np.array([1.0, 0, 0]), "right")
    index3 = daid.MANO_LINKS.index("link_index3")
    assert right["part_ids"][0, index3] == 1 and right["dist"][0, index3] == -0.002  # deepest of the two
    assert np.allclose(right["positions"][0, index3], [4.0, 4, 5])  # + offset
    assert right["part_ids"][1].sum() == 0  # frame 1's contact is the left thumb's
    assert right["part_ids"][2, 0] == 1 and right["dist"][2, 0] == -0.001  # palm and pinky metacarpal share link_palm
    left = daid.contact_slots(realised, np.zeros(3), "left")
    assert left["part_ids"][1, daid.MANO_LINKS.index("link_thumb3")] == 1


def test_write_partition_keeps_the_schema(tmp_path):
    vec3 = pa.list_(pa.list_(pa.float32(), 3))
    table = pa.table({"a": pa.array([[[1, 2, 3], [4, 5, 6]]], type=vec3),
                      "ids": pa.array([[1, 2]], type=pa.list_(pa.int32())), "name": pa.array(["x"])})
    S5.write_partition(table, {"a": np.array([[0.5, 0, 0], [0, 0.5, 0]]), "ids": np.array([7, 8], np.int32)},
                       tmp_path / "p/f.parquet")
    back = pq.read_table(tmp_path / "p/f.parquet")
    assert back.schema == table.schema
    assert back.column("a")[0].as_py() == [[0.5, 0, 0], [0, 0.5, 0]] and back.column("ids")[0].as_py() == [7, 8]
    assert back.column("name")[0].as_py() == "x"


def slot_body(side: str, slot: int) -> str:
    """A Sharpa body of a MANO link slot (the inverse of daid.mano_link)."""
    if slot == 0:
        return f"{side}_hand_C_MC"
    digit, seg = daid.DIGITS[(slot - 1) // 3], (slot - 1) % 3 + 1
    name = f"{side}_{digit}_" + ({1: "MC", 2: "PP", 3: "DP"} if digit == "thumb" else {1: "PP", 2: "MP", 3: "DP"})[seg]
    assert daid.mano_link(name) == slot
    return name


RUN = OUTPUTS / "4c2eac31/episode_000041/s23_load_retarget/run/assets/human_motion_data/ego_recon/processed"
PARQUETS = sorted(RUN.glob("sequence_id=ep41_white_pot/robot_name=sharpa_wave/*.parquet"))
DATA = os.environ.get("PATHB_DATA_TESTS") == "1" and bool(PARQUETS)


@pytest.mark.skipif(not DATA, reason="PATHB_DATA_TESTS=1 and the ep41 task file from run 4c2eac31")
def test_mode_b_round_trip_on_ep41(tmp_path):
    table, cells = S5.read_cells(PARQUETS[0])
    shift = np.array([0.03, -0.05, 1.0])
    T_o = S5.matrices(np.asarray(cells["object_body_position"])[:, 0], np.asarray(cells["object_body_wxyz"])[:, 0])
    v3 = T_o.copy()
    v3[:, :3, 3] -= shift  # the bundle's world
    traj, names, qpos, bodies = {}, [], [], []
    for side in S5.SIDES:
        fingers = [str(j) for j in cells[f"{side}_robot_finger_joint_names"]]
        names += fingers
        qpos.append(np.asarray(cells[f"robot_{side}_finger_joints"]))
        wp, wq = np.asarray(cells[f"robot_{side}_wrist_position"]), np.asarray(cells[f"robot_{side}_wrist_wxyz"])
        traj[f"{side}_wrist_position"], traj[f"{side}_wrist_wxyz"] = wp - shift, wq
        hand = fk.sharpa(side)
        P = hand.poses(wp, wq, dict(zip(fingers, np.asarray(cells[f"robot_{side}_finger_joints"]).T)))
        bodies += [(n, P[n]) for n in hand.names
                   if n.startswith(f"{side}_") and not n.endswith("_site") and n not in fingers]  # the 23 links
        # every MANO contact of the IK run handed in as a contact of the slot's distal or proximal body
        pid = np.asarray(cells[f"mano_{side}_object_contact_part_ids"])
        traj[f"{side}_contact_part_ids"] = pid
        traj[f"{side}_contact_positions"] = np.asarray(cells[f"mano_{side}_object_contact_positions"]) - shift
        traj[f"{side}_contact_normals"] = np.asarray(cells[f"mano_{side}_object_contact_normals"])
    body_names = [n for n, _ in bodies]
    traj["body_names"] = np.array(body_names)
    pose = np.stack([P[:, :3, 3] - shift for _, P in bodies], axis=1)
    quat = np.stack([Rotation.from_matrix(P[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]] for _, P in bodies], axis=1)
    traj["body_pose"] = np.concatenate([pose, quat], axis=-1)
    for side in S5.SIDES:
        pid = traj[f"{side}_contact_part_ids"]
        body = np.full(pid.shape, -1, np.int32)
        local = np.zeros(pid.shape + (3,))
        for f, s in zip(*np.nonzero(pid > 0)):
            b = body_names.index(slot_body(side, s))
            body[f, s] = b
            p = traj["body_pose"][f, b]
            R = Rotation.from_quat(p[[4, 5, 6, 3]])
            local[f, s] = R.inv().apply(traj[f"{side}_contact_positions"][f, s] - p[:3])
        traj[f"{side}_contact_body"], traj[f"{side}_contact_robot_local"] = body, local
    traj["joint_names"], traj["robot_qpos"] = np.array(names), np.concatenate(qpos, axis=1)

    assert np.allclose(S5.world_shift(cells, v3), shift, atol=1e-6)
    new, info = S5.mode_b_cells(cells, traj, shift)
    for side in S5.SIDES:
        assert np.abs(new[f"robot_{side}_frames"][..., :3] - np.asarray(cells[f"robot_{side}_frames"])[..., :3]).max() < 1e-5
        assert info[side]["task_frames_vs_ik_cm"]["max"] < 1e-3
        assert np.allclose(new[f"mano_{side}_object_contact_positions"],
                           np.asarray(cells[f"mano_{side}_object_contact_positions"]), atol=1e-6)
    dest = tmp_path / "sequence_id=x_daid/robot_name=sharpa_wave/t.parquet"
    S5.write_partition(table, new, dest)
    _, back = S5.read_cells(dest)
    over = S5.handover(back, traj, info, RUN / "ep41_white_pot.obj", shift)
    assert over["pass"] and over["within_1cm_share"] == 1.0  # MANO contacts are mesh vertices: distance ~0
