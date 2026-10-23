"""Running the baseline's GPU modules from a stage: their host wrappers start their containers
(pathb/drivers/baseline_call.py in the baseline host venv); baseline/ is used as it is."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .paths import HOST_PY, PATH_B, RECON, TRACK3_FPS

WEIGHTS = RECON / "data/weights"


def call(function: str, kwargs: dict, log: Path) -> None:
    """One baseline wrapper call; its output goes to `log`. Raises with the log's tail on failure."""
    spec = log.with_suffix(".call.json")
    spec.write_text(json.dumps({"function": function, "kwargs": {k: str(v) if isinstance(v, Path) else v
                                                                  for k, v in kwargs.items()}}, indent=1) + "\n")
    with open(log, "a") as f:
        rc = subprocess.run([str(HOST_PY), str(PATH_B / "pathb/drivers/baseline_call.py"), str(spec)],
                            cwd=RECON, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        tail = "\n".join(log.read_text().splitlines()[-20:])
        raise RuntimeError(f"{function} failed (exit {rc}); last lines of {log}:\n{tail}")


def frames_to_mp4(frames: Path, path: Path, fps: float = TRACK3_FPS) -> Path:
    """A folder of %06d.jpg as an H.264 video at near-lossless quality (ffmpeg), else OpenCV mp4v."""
    if path.is_file():
        return path
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", repr(fps), "-i", str(frames / "%06d.jpg"),
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)], check=True)
        return path
    files = sorted(frames.glob("*.jpg"))
    first = cv2.imread(str(files[0]))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, first.shape[1::-1])
    for f in files:
        writer.write(cv2.imread(str(f)))
    writer.release()
    return path


def intrinsics_json(K: np.ndarray, size: tuple[int, int], path: Path) -> Path:
    """CameraIntrinsics JSON (fx, fy, cx, cy, width, height), as the baseline modules read it."""
    K = np.asarray(K, dtype=np.float64)
    path.write_text(json.dumps({"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                                "width": int(size[0]), "height": int(size[1])}) + "\n")
    return path


def mask_tracks(root: Path) -> dict[str, Path]:
    """P1b's mask folders: obj_<i>, hand_L, hand_R."""
    d = root / "p1b_masks"
    return {p.name: p for p in sorted(d.iterdir()) if p.is_dir() and (p.name.startswith("obj_") or p.name.startswith("hand_"))}


def read_mask(folder: Path, frame: int) -> np.ndarray | None:
    m = cv2.imread(str(folder / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0
