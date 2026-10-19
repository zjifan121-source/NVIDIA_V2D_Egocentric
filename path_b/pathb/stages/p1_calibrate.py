"""P1 calibrate: rectify the head stereo pair and undistort ego_cam_a with the factory calibration.

The calibration's reading (calib.READINGS) is chosen on this episode: the one whose rectified
pairs put matched features on the same row with positive disparity. All readings are reported.
Writes rect_left/, rect_right/ (PNG, the stereo module's input), cam_a/ (undistorted JPEG) and
calib.json.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import calib as C
from ..paths import video
from ..report import Metric, StageResult
from ..runner import write_json

CHECK_FRAMES = 5  # evenly spaced frames used to choose the reading


def frames(path, gray: bool):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if gray else frame
    cap.release()


def camera_video(ctx, camera: str):
    return video(ctx.split, ctx.number, camera.removeprefix("ego_cam_"))


def run(ctx) -> StageResult:
    calibration = C.load()
    n = int(ctx.episode["num_frames"])
    picks = set(np.linspace(0, n - 1, CHECK_FRAMES).round().astype(int).tolist())
    raw = {cam: [f for i, f in enumerate(frames(camera_video(ctx, cam), True)) if i in picks]
           for cam in ("ego_cam_b", "ego_cam_c")}

    checks = {}
    for reading in C.READINGS:
        rect = C.rectification(calibration, reading)
        checks[reading] = C.epipolar_check(rect, list(zip(raw[rect.left], raw[rect.right])))
    # A wrong left/right flips the disparity sign; a wrong intrinsics reading shifts rows.
    best = min(checks, key=lambda r: (checks[r]["positive_disparity_fraction"] < 0.8, checks[r]["median_dy_px"]))
    rect = C.rectification(calibration, best)
    rotate = C.READINGS[best][0]

    d = ctx.dir("p1_calibrate")
    for sub in ("rect_left", "rect_right", "cam_a"):
        (d / sub).mkdir(exist_ok=True)
    count = 0
    for i, (left, right) in enumerate(zip(frames(camera_video(ctx, rect.left), True),
                                          frames(camera_video(ctx, rect.right), True))):
        a, b = rect.rectify(left, right)
        cv2.imwrite(str(d / f"rect_left/{i:06d}.png"), a)
        cv2.imwrite(str(d / f"rect_right/{i:06d}.png"), b)
        count += 1
    und = C.undistorter(calibration, rotate)
    count_a = 0
    for i, frame in enumerate(frames(video(ctx.split, ctx.number, "a"), False)):
        cv2.imwrite(str(d / f"cam_a/{i:06d}.jpg"), und.apply(frame), [cv2.IMWRITE_JPEG_QUALITY, 95])
        count_a += 1

    write_json(d / "calib.json", {
        "reading": best, "rectification": rect.summary(),
        "cam_a": {"K": und.K.tolist(), "size": list(und.size), "intrinsics_turned_upright": rotate},
        "readings_checked": checks, "check_frames": sorted(picks),
    })
    chosen = checks[best]
    runner_up = min((c["median_dy_px"] for r, c in checks.items() if r != best), default=float("inf"))
    metrics = [
        Metric("rectified row offset of matched features (median)", chosen["median_dy_px"], "px", ok=(0, 1.0),
               warn=(0, 2.0), note=f"reading '{best}'; next best {runner_up:.2f} px"),
        Metric("matches with positive disparity", chosen["positive_disparity_fraction"], "", ok=(0.95, 1),
               warn=(0.85, 1), note="a wrong left/right gives ~0"),
        Metric("features matched", chosen["matches"], "", ok=(300, float("inf")), warn=(50, float("inf"))),
        Metric("rectified baseline", rect.baseline, "m", ok=(0.074, 0.076), note="calibration: 0.075"),
        Metric("rectified focal length", float(rect.K[0, 0]), "px"),
        Metric("rectified pairs written minus manifest", count - n, "frames", ok=(0, 0)),
        Metric("undistorted ego_cam_a frames minus manifest", count_a - n, "frames", ok=(0, 0)),
    ]
    return StageResult(metrics, {"reading": best, "rect_left": d / "rect_left", "rect_right": d / "rect_right",
                                 "cam_a": d / "cam_a", "calib": d / "calib.json"})
