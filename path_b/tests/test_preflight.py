"""S7 preflight (pathb/preflight.py): mesh geometry across formats, selection and rollout presence,
the GT firewall over an episode's reports."""
import json

import numpy as np
import trimesh

from pathb import preflight


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_geometry_hash_ignores_format_and_order(tmp_path):
    m = trimesh.creation.box((0.1, 0.2, 0.3))
    m.export(tmp_path / "a.obj")
    m.export(tmp_path / "a.glb")
    shuffled = trimesh.Trimesh(m.vertices[::-1], (len(m.vertices) - 1 - m.faces), process=False)
    shuffled.export(tmp_path / "b.obj")
    moved = m.copy()
    moved.apply_translation([0.001, 0, 0])
    moved.export(tmp_path / "c.obj")
    h = {k: preflight.geometry_hash(tmp_path / k)[0] for k in ("a.obj", "a.glb", "b.obj", "c.obj")}
    assert h["a.obj"] == h["a.glb"] == h["b.obj"] != h["c.obj"]


def test_episode_checks(tmp_path):
    root = tmp_path / "episode_000035"
    _, problems = preflight.selected(root)
    assert problems == ["episode_000035: no s6_select/selection.json"]
    write(root / "s6_select/selection.json", json.dumps({"candidate": "keyframe"}))
    write(root / "s6_eval/keyframe/results/episode_000035.parquet", "x")
    cand, problems = preflight.selected(root)
    assert cand == "keyframe" and problems == ["episode_000035: no s6_eval/keyframe/results/episode_000035.json"]
    # meshes: S3a, the URDF's and the packer's must be one geometry
    processed = tmp_path / "proc/ego_recon/processed"
    write(root / "s5_task/keyframe/task.json", json.dumps({"sequence_id": "ep35_keyframe_wooden_spoon_daid",
                                                           "partition": str(processed / "sequence_id=x/robot_name=sharpa_wave")}))
    box = trimesh.creation.box((0.02, 0.2, 0.01))
    for f in (root / "s3a_mesh/wooden_spoon/mesh.obj", processed / "ep35_keyframe_wooden_spoon.obj",
              root / "s5_task/keyframe/run/score_input/meshes/episode_000035/wooden_spoon.obj"):
        f.parent.mkdir(parents=True, exist_ok=True)
        box.export(f)
    hashes, problems = preflight.mesh_checks(root, ["wooden_spoon"], "keyframe")
    assert problems == [] and len(set(hashes["wooden_spoon"].values())) == 1
    trimesh.creation.box((0.02, 0.21, 0.01)).export(processed / "ep35_keyframe_wooden_spoon.obj")
    _, problems = preflight.mesh_checks(root, ["wooden_spoon"], "keyframe")
    assert problems and "meshes differ" in problems[0]
    # GT firewall over the episode's reports
    write(root / "s3a_mesh/report.json", json.dumps({"files_read": ["/x/assets/data/track_3/public/mesh/blue_cup/blue_cup.glb"]}))
    write(root / "s3c_fuse/report.json", json.dumps({"files_read": ["/x/s3b_track/blue_cup/poses/000000.json"]}))
    assert preflight.firewall(root) == ["episode_000035/s3a_mesh: read /x/assets/data/track_3/public/mesh/blue_cup/blue_cup.glb"]
