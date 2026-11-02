"""vm/refill_hands.py: a bundle set shipped with zeros on the frames without a hand, rebuilt from S4b's sources with
those frames filled; the objects and every other file kept."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pathb import stage23

spec = importlib.util.spec_from_file_location("refill_hands", Path(__file__).parents[1] / "vm/refill_hands.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

N = 40


def source_data() -> dict:
    """A camera moving along x, a right wrist sliding along y (world), MANO params consistent with it; the right hand
    parked on frames 15-25 as P4 stores frames without a hand (zero camera-frame parameters, the wrist at the camera)."""
    c2w = np.tile(np.eye(4), (N, 1, 1))
    c2w[:, 0, 3] = np.linspace(0, 0.2, N)
    W = np.tile(np.eye(4), (N, 1, 1))
    W[:, :3, 3] = np.stack([np.full(N, 0.3), np.linspace(0, 0.3, N), np.full(N, 0.1)], axis=1)
    J0 = np.array([0.09, 0.006, 0.006])
    data = {"camera_to_world_transform": c2w.astype(np.float32), "object_to_world_transform": np.tile(np.eye(4), (N, 1, 1)),
            "object_to_camera_transform": np.linalg.inv(c2w), "object_is_valid": np.ones(N, bool)}
    for side in stage23.SIDES:
        data[f"hand_{side}_wrist_to_world_transform"] = W.astype(np.float32)
        data[f"hand_{side}_wrist_orient_in_camera"] = Rotation.from_matrix(W[:, :3, :3]).as_rotvec().astype(np.float32)
        data[f"hand_{side}_wrist_trans_in_camera"] = (W[:, :3, 3] - c2w[:, :3, 3] - J0).astype(np.float32)
        data[f"hand_{side}_finger_pose"] = np.tile(np.linspace(0, 1, N)[:, None, None], (1, 15, 3)).astype(np.float32)
        data[f"hand_{side}_scale"] = np.full(N, 0.9, np.float32)
        data[f"hand_{side}_is_valid"] = np.ones(N, bool)
    for k in range(15, 26):
        data["hand_right_is_valid"][k] = False
        data["hand_right_wrist_trans_in_camera"][k] = data["hand_right_wrist_orient_in_camera"][k] = 0.0
        data["hand_right_finger_pose"][k] = 0.0
        data["hand_right_scale"][k] = 1.0
        data["hand_right_wrist_to_world_transform"][k] = data["camera_to_world_transform"][k]
    return data


def make_run(out: Path, episode: int, name: str, data: dict) -> Path:
    """An S3d-settled object as S4b reads it (s4b_reference.solved)."""
    e = out / "run" / f"episode_{episode:06d}"
    (e / f"s3d_settle/{name}/bundle").mkdir(parents=True)
    (e / f"s3b_segments/{name}").mkdir(parents=True)
    np.savez(e / f"s3d_settle/{name}/bundle/result.npz", **data)
    (e / f"s3d_settle/{name}/bundle/mesh.obj").write_text("v 0 0 0\n")
    np.savez(e / f"s3d_settle/{name}/object_to_world.npz", object_to_world=data["object_to_world_transform"])
    (e / f"s3d_settle/{name}/settle_report.json").write_text(json.dumps({"resolved": [{"kind": "rest", "start": 0, "end": N - 1}]}))
    (e / f"s3b_segments/{name}/segments.json").write_text("[]")
    return out / "run"


def ship(out: Path, run: Path, episode: int, name: str) -> Path:
    """The set as built while FILL_INVALID was off: S4b's candidate, moved objects, a mesh, a README and its sums."""
    track = np.tile(np.eye(4), (N, 1, 1))
    track[:, 2, 3] = np.linspace(0, 0.1, N)
    objects = {"object_to_world_transform": track, "object_to_camera_transform": np.linalg.inv(track)}
    shipped = F.candidate(F.source(run, episode, name), False, objects)
    b = out / "set" / f"episode_{episode:06d}" / name / "result_slam_gravity_aligned"
    b.mkdir(parents=True)
    np.savez(b / "result.npz", **shipped)
    (b / "mesh.obj").write_text("v 0 0 0\n")
    (out / "set/README.md").write_text("# set\n")
    return out / "set"


def sums(root: Path) -> None:
    paths = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (root / "SHA256SUMS.txt").write_text("".join(f"{hashlib.sha256((root / p).read_bytes()).hexdigest()}  {p}\n" for p in paths))


def test_refill_fills_the_frames_without_a_hand_and_keeps_the_rest(tmp_path):
    run = make_run(tmp_path, 7, "cup", source_data())
    full = source_data()
    full["hand_right_is_valid"][:] = True  # a second bundle with every hand frame: left as shipped
    make_run(tmp_path, 9, "box", full)
    s = ship(tmp_path, run, 7, "cup")
    ship(tmp_path, run, 9, "box")
    sums(s)
    before = {p.relative_to(s): p.read_bytes() for p in s.rglob("*") if p.is_file()}
    rows = F.refill(s, {}, default=run)
    assert [r["bundle"] for r in rows] == ["episode_000007/cup/result_slam_gravity_aligned"]
    assert rows[0]["right"]["frames_moved"] == 11 and rows[0]["right"]["restored_to_measured"] == []
    assert rows[0]["right"]["rotvecs_rewritten"] == 0
    assert rows[0]["right"]["max_gap_edge_step_m"]["after"] < 0.02 < rows[0]["right"]["max_gap_edge_step_m"]["before"]
    dest = tmp_path / "set_refill"
    old = dict(np.load(s / "episode_000007/cup/result_slam_gravity_aligned/result.npz"))
    new = dict(np.load(dest / "episode_000007/cup/result_slam_gravity_aligned/result.npz"))
    assert set(new) == set(old)
    for k in old:
        if not k.startswith("hand_right_") or k == "hand_right_is_valid":
            assert np.array_equal(new[k], old[k]), k  # objects, camera, the left hand, the validity flags
    W = new["hand_right_wrist_to_world_transform"][:, :3, 3]
    assert np.allclose(W[:, 1], np.linspace(0, 0.3, N), atol=1e-5)  # the gap now on the wrist's own line
    assert np.allclose(new["hand_right_scale"], 0.9)
    assert all((dest / rel).read_bytes() == data for rel, data in before.items()
               if rel.name not in ("README.md", "SHA256SUMS.txt") and "000007" not in rel.as_posix())
    assert "## Changes" in (dest / "README.md").read_text()
    for line in (dest / "SHA256SUMS.txt").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        assert digest == hashlib.sha256((dest / rel).read_bytes()).hexdigest()
    assert {p.relative_to(s) for p in s.rglob("*") if p.is_file()} == set(before)  # the shipped set untouched


def test_refill_puts_back_a_gap_edge_frame_the_spike_repair_pulled_to_the_camera(tmp_path):
    data = source_data()
    data["hand_right_wrist_to_world_transform"][14, 2, 3] += 0.1  # fast in from frame 13, then (gap at the camera) out to 15
    data["hand_right_wrist_trans_in_camera"][14, 2] += 0.1
    run = make_run(tmp_path, 7, "cup", data)
    s = ship(tmp_path, run, 7, "cup")
    shipped = np.load(s / "episode_000007/cup/result_slam_gravity_aligned/result.npz")["hand_right_wrist_to_world_transform"]
    assert abs(shipped[14, 2, 3] - 0.2) > 0.05  # the old repair took 14 for a spike and pulled it toward the camera
    rows = F.refill(s, {}, default=run)
    assert rows[0]["right"]["restored_to_measured"] == [14] and rows[0]["right"]["frames_moved"] == 12
    new = np.load(tmp_path / "set_refill/episode_000007/cup/result_slam_gravity_aligned/result.npz")
    assert np.allclose(new["hand_right_wrist_to_world_transform"][14], data["hand_right_wrist_to_world_transform"][14])
    assert "`episode_000007/cup` frame 14" in (tmp_path / "set_refill/README.md").read_text()


def test_refill_takes_float_noise_between_venvs(tmp_path):
    run = make_run(tmp_path, 7, "cup", source_data())
    s = ship(tmp_path, run, 7, "cup")
    f = s / "episode_000007/cup/result_slam_gravity_aligned/result.npz"
    data = dict(np.load(f))
    data["hand_right_wrist_orient_in_camera"] = data["hand_right_wrist_orient_in_camera"].astype(np.float64)
    data["hand_right_wrist_orient_in_camera"][20] += 5e-16  # evaluation ep20: E0's venv
    data["hand_right_wrist_orient_in_camera"] = data["hand_right_wrist_orient_in_camera"].astype(np.float32)
    np.savez(f, **data)
    rows = F.refill(s, {}, default=run)
    assert len(rows) == 1 and rows[0]["source_reproduces_within"] <= F.SOURCE_ATOL


def test_refill_stops_when_the_source_does_not_reproduce_the_set(tmp_path):
    run = make_run(tmp_path, 7, "cup", source_data())
    s = ship(tmp_path, run, 7, "cup")
    f = s / "episode_000007/cup/result_slam_gravity_aligned/result.npz"
    data = dict(np.load(f))
    data["hand_left_finger_pose"] = data["hand_left_finger_pose"] + 0.01  # shipped from another source
    np.savez(f, **data)
    with pytest.raises(SystemExit, match="hand_left_finger_pose"):
        F.refill(s, {}, default=run)
    assert not (tmp_path / "set_refill").exists()
