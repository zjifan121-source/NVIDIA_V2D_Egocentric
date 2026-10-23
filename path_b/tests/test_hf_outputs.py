"""vm/hf_outputs.py uncovered(): which files of a run Hugging Face does not hold."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("hf_outputs", Path(__file__).parents[1] / "vm/hf_outputs.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def write(root: Path, rel: str, data: bytes = b"x") -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(data)


def test_uncovered(tmp_path):
    run = tmp_path / "run"
    files = {"waterfall.md": b"w", "episode_000040/p0_ingest/report.json": b"r0",
             "episode_000040/s4a_phases/report.json": b"new", "episode_000040/p2_depth/depth_rect/000000.png": b"d",
             "episode_000040/p3_scale_bridge/merged/result.npz": b"scratch",
             "episode_000026/p2_depth/depth_rect/000000.png": b"d"}
    for rel, data in files.items():
        write(run, rel, data)
    (run / "episode_000040/linked").symlink_to(tmp_path)
    remote = {"waterfall.md": 1, "episode_000040/p0_ingest/report.json": 99,  # changed since the upload
              "episode_000040/frames.tar": 10}
    assert H.uncovered(run, remote) == ["episode_000026/p2_depth/depth_rect/000000.png",
                                        "episode_000040/p0_ingest/report.json",
                                        "episode_000040/s4a_phases/report.json"]
    # never uploaded: every regular file, scratch included
    assert len(H.uncovered(run, {})) == len(files)


def test_per_frame_folders_go_into_the_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(H, "STAGING", tmp_path / "staging")
    run = tmp_path / "outputs/r"
    write(run, "episode_000040/p1b_masks/report.json", b"{}")
    for k in range(H.CROWDED + 1):  # one v3 per-frame folder (masks)
        write(run, f"episode_000040/p1b_masks/obj_0/{k:06d}.png")
    for k in range(3):
        write(run, f"episode_000040/s3a_mesh/blue_cup/views/{k:06d}.png")  # a few files stay results
    root, manifest = H.stage("r", frames=True)
    assert manifest["frames"]["episode_000040"]["files"] == H.CROWDED + 1
    assert manifest["results"]["files"] == 4
    with H.tarfile.open(root / H.PREFIX / "r/episode_000040/frames.tar") as tar:
        assert "p1b_masks/obj_0/000000.png" in tar.getnames()
    remote = {"episode_000040/frames.tar": 1, "episode_000040/p1b_masks/report.json": 2,
              **{f"episode_000040/s3a_mesh/blue_cup/views/{k:06d}.png": 1 for k in range(3)}}
    assert H.uncovered(run, remote) == []


def test_run_level_folders_are_uploaded(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(H, "STAGING", tmp_path / "staging")
    run = tmp_path / "outputs/r"
    write(run, "episode_000040/p0_ingest/report.json", b"{}")
    write(run, "waterfall.md", b"w")
    write(run, "canonical/white_pot/mesh.glb", b"g")
    root, manifest = H.stage("r", frames=False)
    assert (root / H.PREFIX / "r/canonical/white_pot/mesh.glb").is_file() and manifest["results"]["files"] == 3
