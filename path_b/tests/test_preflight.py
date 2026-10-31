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


def test_mesh_checks_reroot_a_path_written_on_another_host(tmp_path):
    root = tmp_path / "v4_eval/episode_000005"
    processed = root / "s5_task/inverted/run/assets/human_motion_data/ego_recon/processed"
    vm = "/no_such_host_root/outputs/v4_eval/episode_000005/s5_task/inverted/run/assets/human_motion_data/ego_recon/processed"
    write(root / "s5_task/inverted/task.json", json.dumps({"sequence_id": "ep5_inverted_daid",
                                                           "partition": vm + "/sequence_id=ep5_inverted/robot_name=sharpa_wave"}))
    box = trimesh.creation.box((0.02, 0.2, 0.01))
    for name in ("white_pot", "wooden_spoon"):
        for f in (root / f"s3a_mesh/{name}/mesh.obj", processed / f"ep5_inverted_{name}.obj",
                  root / f"s5_task/inverted/run/score_input/meshes/episode_000005/{name}.obj"):
            f.parent.mkdir(parents=True, exist_ok=True)
            box.export(f)
    hashes, problems = preflight.mesh_checks(root, ["white_pot", "wooden_spoon"], "inverted")
    assert problems == [] and all(len(h) == 3 and len(set(h.values())) == 1 for h in hashes.values())
    # a path that exists, or one outside any episode folder, is left as it is
    assert preflight.here(str(processed), root) == processed
    assert str(preflight.here("/no_such_host_root/x.obj", root)) == "/no_such_host_root/x.obj"


def test_relabel_names_bodies_by_roster_slot(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.table({"object_id": [0, 1, 0, 1], "object_body_name": ["ep26_inverted_plastic_dish_rack", "ep26_inverted_white_pot"] * 2,
                  "x": [1.0, 2.0, 3.0, 4.0]})
    pq.write_table(t, tmp_path / "a.parquet")
    assert preflight.relabel(tmp_path / "a.parquet", tmp_path / "b.parquet", ["plastic_dish_rack", "white_pot"]) == []
    b = pq.read_table(tmp_path / "b.parquet")
    assert b.column("object_body_name").to_pylist() == ["plastic_dish_rack", "white_pot"] * 2
    assert b.column("x").to_pylist() == [1.0, 2.0, 3.0, 4.0]
    # a body in the wrong slot is refused, nothing written
    assert preflight.relabel(tmp_path / "a.parquet", tmp_path / "c.parquet", ["white_pot", "plastic_dish_rack"])
    assert not (tmp_path / "c.parquet").exists()


def test_dry_kit_rosters_the_public_episodes(tmp_path):
    import pandas as pd
    kit = preflight.dry_kit(tmp_path, [41, 26])
    cfg = json.loads((kit / "config/tracks.json").read_text())["tracks"]["track_3"]
    assert cfg["scored_episodes"] == [26, 41]
    roster = json.loads((kit / "data/track_3_evaluation_objects.json").read_text())["episodes"]
    assert roster["41"] == ["white_pot"] and len(roster["26"]) == 2
    rows = pd.read_parquet(kit / "data/track_3_sample_submission.parquet")["row_id"]
    m = preflight.manifest()
    assert len(rows) == m[41]["num_frames"] + 2 * m[26]["num_frames"]
    assert (kit / "tools/pack_submission.py").is_file() and not (kit / "tools/pack_submission.py").is_symlink()
    assert (kit / "v2dlb").is_symlink()
