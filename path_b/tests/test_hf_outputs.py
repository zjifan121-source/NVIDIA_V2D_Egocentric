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
