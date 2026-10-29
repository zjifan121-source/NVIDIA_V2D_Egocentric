"""Path B's stages 2-3 (pathb/stage23.py, pathb/nvidia.py): hand repair, still scene, the output check, and
the mirror of NVIDIA's code as imported. The driver itself needs $RETARGET_VENV (not run here)."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from pathb import nvidia, stage23
from pathb.paths import TRACK3_FPS


def bundle(n=40):
    """A camera moving along x, a right wrist sliding along y in the world, MANO params consistent with it."""
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, 0, 3] = np.linspace(0, 0.2, n)
    c2w[:, :3, :3] = Rotation.from_euler("z", np.linspace(0, 20, n)[:, None], degrees=True).as_matrix()
    W = np.tile(np.eye(4), (n, 1, 1))
    W[:, :3, 3] = np.stack([np.full(n, 0.3), np.linspace(0, 0.3, n), np.full(n, 0.1)], axis=1)
    W[:, :3, :3] = Rotation.from_euler("x", np.linspace(0, 30, n)[:, None], degrees=True).as_matrix()
    J0 = np.array([0.09, 0.006, 0.006])
    data = {"camera_to_world_transform": c2w.astype(np.float32), "object_to_world_transform": np.tile(np.eye(4), (n, 1, 1)),
            "object_to_camera_transform": np.linalg.inv(c2w), "object_is_valid": np.ones(n, bool)}
    for side in stage23.SIDES:
        R_cam = np.einsum("nji,njk->nik", c2w[:, :3, :3], W[:, :3, :3])
        data[f"hand_{side}_wrist_to_world_transform"] = W.astype(np.float32)
        data[f"hand_{side}_wrist_orient_in_camera"] = Rotation.from_matrix(R_cam).as_rotvec().astype(np.float32)
        data[f"hand_{side}_wrist_trans_in_camera"] = (np.einsum("nji,nj->ni", c2w[:, :3, :3], W[:, :3, 3] - c2w[:, :3, 3])
                                                      - J0).astype(np.float32)
        data[f"hand_{side}_finger_pose"] = np.tile(np.linspace(0, 1, n)[:, None, None], (1, 15, 3)).astype(np.float32)
        data[f"hand_{side}_betas"] = np.zeros(10, np.float32)
        data[f"hand_{side}_scale"] = np.full(n, 0.9, np.float32)
        data[f"hand_{side}_is_valid"] = np.ones(n, bool)
    return data, W.copy(), J0


def test_clean_hands_fills_a_wrist_spike_and_keeps_the_rest():
    data, W, J0 = bundle()
    spike = data["hand_right_wrist_to_world_transform"]
    spike[20, :3, 3] += [0.0, 0.0, 0.5]  # 0.5 m in one frame: ~10 m/s
    data["hand_right_finger_pose"][20] = 5.0
    report = stage23.clean_hands(data, TRACK3_FPS)
    assert report["right"]["replaced"] == 1 and report["right"]["frames"] == [20] and report["left"]["replaced"] == 0
    fixed = data["hand_right_wrist_to_world_transform"]
    assert np.allclose(fixed[20, :3, 3], W[20, :3, 3], atol=1e-6)  # linear in between, as the truth here
    assert np.allclose(fixed[19], W[19], atol=1e-6) and np.allclose(fixed[21], W[21], atol=1e-6)
    assert np.allclose(data["hand_right_finger_pose"][20], (data["hand_right_finger_pose"][19] + data["hand_right_finger_pose"][21]) / 2)
    c2w = data["camera_to_world_transform"].astype(np.float64)  # the camera-frame params follow the world pose
    back = np.einsum("nij,nj->ni", c2w[:, :3, :3], data["hand_right_wrist_trans_in_camera"] + J0) + c2w[:, :3, 3]
    assert np.allclose(back, fixed[:, :3, 3], atol=1e-5)
    R = np.einsum("nij,njk->nik", c2w[:, :3, :3], Rotation.from_rotvec(data["hand_right_wrist_orient_in_camera"]).as_matrix())
    assert np.allclose(R, fixed[:, :3, :3], atol=1e-5)


def test_still_scene_holds_object_hands_and_camera_at_frame_0():
    data, _, _ = bundle()
    data["object_to_world_transform"][:, 0, 3] = np.linspace(0, 1, 40)
    stage23.still_scene(data)
    for key in ("object_to_world_transform", "camera_to_world_transform", "hand_left_wrist_to_world_transform",
                "hand_right_finger_pose", "hand_right_wrist_trans_in_camera"):
        assert np.allclose(data[key], data[key][:1]), key
    assert data["hand_right_betas"].shape == (10,)


USDA = '''#usda 1.0
def Xform "support_surfaces"
{
    def Cylinder "ep1_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double height = 0.01
        double radius = 0.2
        double3 xformOp:translate = (0.0, 0.0, 0.995)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
'''


def test_check_counts_penetration_supports_and_seating(tmp_path):
    usda = tmp_path / "support.usda"
    usda.write_text(USDA)
    disks = stage23.supports(usda)
    assert len(disks) == 1 and disks[0]["top_z"] == pytest.approx(1.0) and disks[0]["radius"] == 0.2
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    mesh = tmp_path / "ep1_box.obj"
    box.export(mesh)
    n = 6
    pos = np.tile([[0.0, 0.0, 1.05]], (n, 1))[:, None]  # resting on the disk: bottom at z = 1.0
    wxyz = np.tile([[1.0, 0, 0, 0]], (n, 1))[:, None]
    frames = np.zeros((n, 3, 7))
    frames[:, :, :3] = [0.0, 0.0, 1.5]  # far above
    frames[2:4, 0, :3] = [0.0, 0.0, 1.05]  # two frames with a link at the box's centre: 5 cm inside
    row = {"object_body_position": [pos.tolist()], "object_body_wxyz": [wxyz.tolist()],
           "robot_right_frames": [frames.tolist()], "robot_left_frames": [frames[:, :, :].tolist()]}
    part = tmp_path / "part"
    part.mkdir()
    pq.write_table(pa.table(row), part / "x.parquet")
    out = stage23.check(part, [mesh], usda, n)
    assert out["frames"] == n and out["penetration_frames"]["over_2cm"] == 2 and out["penetration_frames"]["max_cm"] == pytest.approx(5, abs=0.1)
    assert out["support_disks"] == 1 and out["frame0_seating_cm"] == {"ep1_box": pytest.approx(0.0, abs=1e-6)}
    assert stage23.check(part, [mesh], usda, n + 1)["warnings"]
    assert stage23.worst_seating({"a": 0.5, "b": -2.0, "c": None}) == -2.0 and stage23.worst_seating([None]) is None


def test_pristine_mirror_restores_the_imported_files():
    import subprocess
    try:
        diff = nvidia.changed(nvidia.ROBOTIC)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("no git history of the import commit")
    root = nvidia.pristine(nvidia.ROBOTIC)
    for rel, status in diff.items():
        target = root / rel
        if status == "A":
            assert not target.exists()
        else:
            original = subprocess.run(["git", "-C", str(nvidia.REPO), "show", f"{nvidia.NVIDIA_IMPORT}:baseline/{nvidia.ROBOTIC}/{rel}"],
                                      check=True, capture_output=True).stdout
            assert not target.is_symlink() and target.read_bytes() == original
    script = root / "scripts/retarget/ego_recon_to_sharpa.py"  # unchanged files: links into baseline/
    assert script.resolve() == (nvidia.REPO / "baseline" / nvidia.ROBOTIC / "scripts/retarget/ego_recon_to_sharpa.py").resolve()


def test_check_seats_a_lid_on_its_pot_not_on_the_table(tmp_path):
    usda = tmp_path / "support.usda"
    usda.write_text(USDA)  # one disk, its top at z = 1.0
    pot, lid = trimesh.creation.cylinder(radius=0.1, height=0.12), trimesh.creation.cylinder(radius=0.1, height=0.02)
    meshes = [tmp_path / "ep2_white_pot.obj", tmp_path / "ep2_white_pot_lid.obj"]
    pot.export(meshes[0])
    lid.export(meshes[1])
    n = 3
    pos = np.tile([[0.0, 0.0, 1.06], [0.0, 0.0, 1.13]], (n, 1, 1))  # the pot on the disk, the lid on its rim
    wxyz = np.tile([1.0, 0, 0, 0], (n, 2, 1))
    frames = np.full((n, 3, 7), 0.0)
    frames[:, :, :3] = [1.0, 1.0, 1.5]
    row = {"object_body_position": [pos.tolist()], "object_body_wxyz": [wxyz.tolist()],
           "robot_right_frames": [frames.tolist()]}
    part = tmp_path / "part"
    part.mkdir()
    pq.write_table(pa.table(row), part / "x.parquet")
    seating = stage23.check(part, meshes, usda, n)["frame0_seating_cm"]
    assert seating["ep2_white_pot"] == pytest.approx(0.0, abs=1e-6) and seating["ep2_white_pot_lid"] == pytest.approx(0.0, abs=0.01)
