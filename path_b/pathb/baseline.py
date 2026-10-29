"""Running the baseline's GPU modules from a stage: their host wrappers start their containers
(pathb/drivers/baseline_call.py in the baseline host venv); baseline/ is used as it is."""
from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .paths import HOST_PY, PATH_B, RECON, TRACK3_FPS

WEIGHTS = RECON / "data/weights"


def shim() -> Path:
    """path_b/data/shim/v2d/common -> the baseline's v2d_common, for machines without the baseline
    host venv (the Mac): its CPU-only functions (result_bundle) then run in path_b's own Python."""
    root = PATH_B / "data/shim"
    common = root / "v2d/common"
    if not common.exists():
        common.parent.mkdir(parents=True, exist_ok=True)
        (common.parent / "__init__.py").write_text("")
        common.symlink_to(RECON / "modules/v2d_common")
    return root


def interpreter(function: str) -> tuple[str, dict]:
    if HOST_PY.exists():
        return str(HOST_PY), dict(os.environ)
    if function.startswith("v2d.common."):
        return sys.executable, dict(os.environ, PYTHONPATH=str(shim()))
    raise RuntimeError(f"{function} needs the baseline host venv ({HOST_PY}): run it on the GPU machine")


def call(function: str, kwargs: dict, log: Path) -> None:
    """One baseline wrapper call; its output goes to `log`. Raises with the log's tail on failure."""
    spec = log.with_suffix(".call.json")
    spec.write_text(json.dumps({"function": function, "kwargs": {k: str(v) if isinstance(v, Path) else v
                                                                  for k, v in kwargs.items()}}, indent=1) + "\n")
    py, env = interpreter(function)
    with open(log, "a") as f:
        rc = subprocess.run([py, str(PATH_B / "pathb/drivers/baseline_call.py"), str(spec)],
                            cwd=RECON, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
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


DUP_IOU = 0.9  # default: two objects' masks this alike on a frame show one object
DUP_FRAMES = 5  # default: the shortest run of such frames that counts
JUMP_GAP = 0.2  # default: the lost track's continuity into the run is at least this far below the other's
DUP_SCALE = 4  # masks compared at 1/4 resolution


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


@functools.lru_cache(maxsize=8)
def lost_frames(root: str) -> dict[str, frozenset[int]]:
    """Per object track (obj_<i>), the frames whose P1b mask is another object's: runs of at least DUP_FRAMES
    frames where two objects' masks overlap with IoU >= DUP_IOU; at the run's first frame, the track whose
    mask jumped there (IoU with its own previous mask at least JUMP_GAP below the other track's) lost its
    object for the run. ep0: the pot's track jumped onto the lifted lid at frame 75 (IoU with its previous
    mask 0.04, the lid's 0.57) and the two masks stayed one (IoU 0.98-1.00) to the clip's end. A run where
    neither jumped, or that starts at frame 0, is left as it is."""
    tracks = {k: v for k, v in mask_tracks(Path(root)).items() if k.startswith("obj_")}
    if len(tracks) < 2:
        return {k: frozenset() for k in tracks}
    frames = sorted({int(f.stem) for v in tracks.values() for f in v.glob("*.png")})
    small = {}
    for k, folder in tracks.items():
        for t in frames:
            m = read_mask(folder, t)
            small[k, t] = None if m is None else m[::DUP_SCALE, ::DUP_SCALE].copy()
    lost: dict[str, set[int]] = {k: set() for k in tracks}
    names = sorted(tracks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same = [t for t in frames if small[a, t] is not None and small[b, t] is not None
                    and iou(small[a, t], small[b, t]) >= DUP_IOU]
            runs, start = [], None
            for j, t in enumerate(same):
                if start is None:
                    start = j
                if j + 1 == len(same) or same[j + 1] != t + 1:
                    runs.append(same[start:j + 1])
                    start = None
            for run in runs:
                t0 = run[0]
                prev = max((t for t in frames if t < t0 and small[a, t] is not None and small[b, t] is not None), default=None)
                if len(run) < DUP_FRAMES or prev is None:
                    continue
                ca, cb = iou(small[a, t0], small[a, prev]), iou(small[b, t0], small[b, prev])
                if abs(ca - cb) >= JUMP_GAP:
                    lost[a if ca < cb else b].update(run)
    return {k: frozenset(v) for k, v in lost.items()}


def object_mask(root: Path, k: int, t: int) -> np.ndarray | None:
    """Object k's P1b mask at frame t, None where it is another object's (`lost_frames`)."""
    if t in lost_frames(str(root)).get(f"obj_{k}", frozenset()):
        return None
    return read_mask(mask_tracks(root)[f"obj_{k}"], t)
