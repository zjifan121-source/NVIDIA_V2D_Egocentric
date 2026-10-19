"""P0 ingest: the three head-camera videos decode, match the manifest frame for frame, and have the
resolutions the calibration was made for."""
from __future__ import annotations

import cv2
import numpy as np

from .. import calib as C
from ..paths import video
from ..report import Metric, StageResult


def probe(path) -> dict:
    """Decode every frame (container frame counts can be off) and look at the first one."""
    cap = cv2.VideoCapture(str(path))
    frames, first = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first is None:
            first = frame
        frames += 1
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if first is None:
        raise RuntimeError(f"no frames decoded from {path}")
    mono = bool(np.array_equal(first[..., 0], first[..., 1]) and np.array_equal(first[..., 1], first[..., 2]))
    return {"path": str(path), "frames": frames, "fps": fps, "size": [first.shape[1], first.shape[0]], "mono": mono}


def run(ctx) -> StageResult:
    calibration = C.load()
    expected = int(ctx.episode["num_frames"])
    info, metrics = {}, []
    for cam in ("a", "b", "c"):
        path = video(ctx.split, ctx.number, cam)
        if not path.is_file():
            raise FileNotFoundError(path)
        info[cam] = v = probe(path)
        metrics.append(Metric(f"ego_cam_{cam} frames minus manifest", v["frames"] - expected, "frames",
                              ok=(0, 0), note="every frame must be scored"))
        size_ok = v["size"] == calibration["cameras"][f"ego_cam_{cam}"]["resolution"]
        metrics.append(Metric(f"ego_cam_{cam} resolution matches calibration", int(size_ok), "", ok=(1, 1),
                              note=f"{v['size'][0]}x{v['size'][1]}"))
    metrics.append(Metric("ego_cam_b/c are mono", int(info["b"]["mono"] and info["c"]["mono"]), "", ok=(1, 1)))
    metrics.append(Metric("ego_cam_a fps", round(info["a"]["fps"], 3), "fps"))
    return StageResult(metrics, {"videos": info, "frames": expected})
