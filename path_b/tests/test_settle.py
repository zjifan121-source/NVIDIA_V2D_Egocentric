"""S3d settle (V4_BUILD_SPEC.md S3d, task 7): a floating cylinder settles onto the table; a cup above a
rack's floor settles in the rack, not through it (MuJoCo on the CPU, the Mac's sim venv)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from pathb import runner
from pathb.stages import s3d_settle

SIM = Path(os.environ.get("SETTLE_PY", Path.home() / "V2D/venvs/sim/bin/python"))
N = 20


def write_object(root: Path, name: str, parts: list, pose: np.ndarray, support: str, mass: float):
    d = root / f"s3a_mesh/{name}"
    (d / "coacd").mkdir(parents=True)
    for i, p in enumerate(parts):
        p.export(d / f"coacd/piece_{i:03d}.obj")
    (d / "coacd/pieces.json").write_text(json.dumps({"pieces": [{"file": f"piece_{i:03d}.obj"} for i in range(len(parts))]}))
    trimesh.util.concatenate(parts).export(d / "mesh.obj")
    (d / "mass.json").write_text(json.dumps({"mass_kg": mass}))
    (root / f"s3b_segments/{name}").mkdir(parents=True)
    (root / f"s3b_segments/{name}/segments.json").write_text(json.dumps(
        [{"kind": "static", "start": 0, "end": N - 1, "hand": None, "fingertips": [], "support": support}]))
    (root / f"s3c_invert/{name}").mkdir(parents=True)
    np.savez(root / f"s3c_invert/{name}/object_to_world.npz", object_to_world=np.repeat(pose[None], N, axis=0))
    b = root / f"p4_export/episode_000999/{name}/result_slam_gravity_aligned"
    b.mkdir(parents=True)
    np.savez(b / "result.npz", camera_to_world_transform=np.tile(np.eye(4, dtype=np.float32), (N, 1, 1)),
             object_to_world_transform=np.tile(np.eye(4, dtype=np.float32), (N, 1, 1)),
             object_to_camera_transform=np.tile(np.eye(4, dtype=np.float32), (N, 1, 1)), object_scale=np.float32(1))


def cup():
    c = trimesh.creation.cylinder(radius=0.04, height=0.10)
    c.apply_translation([0, 0, 0.05])  # canonical frame: bottom on z = 0
    return [c]


def rack():
    floor = trimesh.creation.box((0.3, 0.2, 0.02))
    floor.apply_translation([0, 0, 0.01])
    parts = [floor]
    for sx, sy, x, y in ((0.3, 0.01, 0, 0.095), (0.3, 0.01, 0, -0.095), (0.01, 0.2, 0.145, 0), (0.01, 0.2, -0.145, 0)):
        w = trimesh.creation.box((sx, sy, 0.12))
        w.apply_translation([x, y, 0.06])
        parts.append(w)
    return parts


@pytest.mark.skipif(not SIM.exists(), reason="needs a venv with mujoco (SETTLE_PY or ~/V2D/venvs/sim)")
def test_floating_cylinder_settles_and_cup_settles_in_the_rack(tmp_path, monkeypatch):
    monkeypatch.setenv("SETTLE_PY", str(SIM))
    root = tmp_path / "episode_000999"
    lifted = np.eye(4)
    lifted[:3, 3] = [0.5, 0.0, 0.01]  # 1 cm above the table
    write_object(root, "blue_cup", cup(), lifted, "table", 0.1)
    e = {"episode": 999, "split": "evaluation", "num_frames": N, "objects": [{"name": "blue_cup"}]}
    r = s3d_settle.run(runner.Context(e, root, {}))
    P = np.load(root / "s3d_settle/blue_cup/object_to_world.npz")["object_to_world"]
    assert abs(P[0, 2, 3]) < 0.002 and np.allclose(P[0, :2, 3], [0.5, 0.0], atol=0.002)
    status = {m.name: m.status for m in r.metrics}
    assert status["blue_cup: penetration into the support after settling (worst)"] == "PASS"

    root = tmp_path / "two" / "episode_000999"
    in_rack = np.eye(4)
    in_rack[:3, 3] = [0.0, 0.0, 0.03]  # 1 cm above the rack's floor (its top at z = 0.02)
    write_object(root, "blue_cup", cup(), in_rack, "plastic_dish_rack", 0.1)
    write_object(root, "plastic_dish_rack", rack(), np.eye(4), "table", 0.3)
    e = {"episode": 999, "split": "evaluation", "num_frames": N,
         "objects": [{"name": "blue_cup"}, {"name": "plastic_dish_rack"}]}
    r = s3d_settle.run(runner.Context(e, root, {}))
    P = np.load(root / "s3d_settle/blue_cup/object_to_world.npz")["object_to_world"]
    assert abs(P[0, 2, 3] - 0.02) < 0.002  # on the rack's floor, not through it to the table
    rep = json.loads((root / "s3d_settle/blue_cup/settle_report.json").read_text())
    assert rep["statics"][0]["settled"] and rep["statics"][0]["support"] == "plastic_dish_rack"
    status = {m.name: m.status for m in r.metrics}
    assert status["blue_cup: penetration into the support after settling (worst)"] == "PASS"


@pytest.mark.skipif(not SIM.exists(), reason="needs the sim venv (mujoco)")
def test_a_rest_fitted_inside_the_table_drops_onto_a_table_at_its_own_bottom(tmp_path, monkeypatch):
    monkeypatch.setenv("SETTLE_PY", str(SIM))
    monkeypatch.setattr(s3d_settle, "TABLE_AT_REST", True)
    root = tmp_path / "episode_000999"
    sunk = np.eye(4)
    sunk[:3, :3] = Rotation.from_euler("x", 2, degrees=True).as_matrix()
    sunk[:3, 3] = [0.5, 0.0, -0.012]  # S3c's fit 1.2 cm into the table (ep22's cup: 1.1 cm)
    write_object(root, "blue_cup", cup(), sunk, "table", 0.1)
    e = {"episode": 999, "split": "evaluation", "num_frames": N, "objects": [{"name": "blue_cup"}]}
    s3d_settle.run(runner.Context(e, root, {}))
    row = json.loads((root / "s3d_settle/blue_cup/settle_report.json").read_text())["statics"][0]
    P = np.load(root / "s3d_settle/blue_cup/object_to_world.npz")["object_to_world"]
    assert row["settled"] and -1.6 < row["table_cm"] < -1.2 and abs(P[0, 2, 3] - sunk[2, 3]) < 0.003
    assert np.allclose(P[0, :2, 3], [0.5, 0.0], atol=0.005)
    monkeypatch.setattr(s3d_settle, "TABLE_AT_REST", False)  # on z = 0 MuJoCo throws it out of the table
    root = tmp_path / "z0" / "episode_000999"
    write_object(root, "blue_cup", cup(), sunk, "table", 0.1)
    s3d_settle.run(runner.Context(e, root, {}))
    row = json.loads((root / "s3d_settle/blue_cup/settle_report.json").read_text())["statics"][0]
    assert "table_cm" not in row
