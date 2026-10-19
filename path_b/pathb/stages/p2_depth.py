"""P2 stereo depth: FoundationStereo on P1's rectified pairs, the table plane in every frame, and
the depth reprojected into the undistorted ego_cam_a view.

The stereo model is the baseline's v2d_foundation_stereo module, called read-only through its own
launcher (TensorRT engine at 576x960; the module resizes and scales disparity back). Which reading
of ego_cam_a's extrinsics holds is decided here: stereo depth at SIFT features of the rectified left
image must land on the same features in ego_cam_a. Writes depth_rect/ (module output, rectified
left frame), planes.json, alignment.json and depth_a_half/ (ego_cam_a at half resolution).
"""
from __future__ import annotations

import json
import subprocess

import cv2
import numpy as np

from .. import calib as C
from ..geometry import fit_plane, pixel_rays, read_depth, table_points, write_depth
from ..paths import HOST_PY, RECON, STEREO_WEIGHTS
from ..report import Metric, StageResult
from ..runner import write_json

ALIGN_FRAMES = 3  # frames whose features choose the ego_cam_a extrinsics reading
NEAR, FAR = 0.1, 5.0  # metres of stereo depth used outside the table fit
REST_FRAMES = 30  # recordings open with both hands raised over the table (README §5.6)


def stereo_depth(p1, d, rect: dict) -> None:
    cal = d / "stereo_calibration.json"
    write_json(cal, {k: rect[k] for k in ("fx", "fy", "cx", "cy", "baseline")})
    cmd = [str(HOST_PY), "modules/v2d_foundation_stereo/docker/run_image_list_to_depth.py",
           "--left_dir", str(p1 / "rect_left"), "--right_dir", str(p1 / "rect_right"),
           "--depth_folder", str(d / "depth_rect"), "--intrinsics_folder", str(d / "intrinsics_rect"),
           "--model_dir", str(STEREO_WEIGHTS), "--calibration_file", str(cal)]
    with open(d / "stereo.log", "w") as log:
        result = subprocess.run(cmd, cwd=RECON, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join((d / "stereo.log").read_text().splitlines()[-15:])
        raise RuntimeError(f"stereo module failed (exit {result.returncode}):\n{tail}")


def feature_pairs(p1, d, Kr: np.ndarray, R1: np.ndarray, frames: list[int]):
    """3D points in the unrectified left camera (stereo depth at SIFT features of the rectified
    left image) and the ego_cam_a pixels the same features were found at."""
    sift = cv2.SIFT_create(4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    points, pixels = [], []
    for i in frames:
        left = cv2.imread(str(p1 / f"rect_left/{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        cam_a = cv2.imread(str(p1 / f"cam_a/{i:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        depth = read_depth(d / f"depth_rect/{i:06d}.png")
        kl, dl = sift.detectAndCompute(left, None)
        ka, da = sift.detectAndCompute(cam_a, None)
        if dl is None or da is None:
            continue
        for pair in matcher.knnMatch(dl, da, k=2):
            if len(pair) < 2 or pair[0].distance > 0.7 * pair[1].distance:
                continue
            (u, v), target = kl[pair[0].queryIdx].pt, ka[pair[0].trainIdx].pt
            z = depth[int(round(v)), int(round(u))]
            if NEAR < z < FAR:
                points.append(R1.T @ (z * np.linalg.solve(Kr, [u, v, 1.0])))
                pixels.append(target)
    return np.array(points).reshape(-1, 3), np.array(pixels).reshape(-1, 2)


def reprojection_errors(points: np.ndarray, pixels: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    X = points @ T[:3, :3].T + T[:3, 3]
    in_front = X[:, 2] > 0.05
    uv = (X[in_front] @ K.T)[:, :2] / X[in_front, 2:3]
    return np.linalg.norm(uv - pixels[in_front], axis=1)


def run(ctx) -> StageResult:
    p1 = ctx.root / "p1_calibrate"
    cal = json.loads((p1 / "calib.json").read_text())
    rect = cal["rectification"]
    Kr = np.array([[rect["fx"], 0, rect["cx"]], [0, rect["fy"], rect["cy"]], [0, 0, 1.0]])
    R1 = np.array(rect["R1"])
    n = int(ctx.episode["num_frames"])
    d = ctx.dir("p2_depth")

    stereo_depth(p1, d, rect)
    files = sorted((d / "depth_rect").glob("*.png"))

    planes, valid = [], []
    for i, f in enumerate(files):
        depth = read_depth(f)
        valid.append(float(np.mean((depth > NEAR) & (depth < FAR))))
        plane = fit_plane(table_points(depth, Kr), seed=i)
        planes.append({"frame": i, **plane} if plane else {"frame": i})
    write_json(d / "planes.json", {"frame": "rectified left camera", "planes": planes})

    # ego_cam_a extrinsics: the reading under which stereo points land on ego_cam_a's features.
    calibration = C.load()
    rectification = C.rectification(calibration, cal["reading"])
    K_a = np.array(cal["cam_a"]["K"])
    picks = np.linspace(0, len(files) - 1, ALIGN_FRAMES).round().astype(int).tolist()
    points, pixels = feature_pairs(p1, d, Kr, R1, picks)
    errors = {}
    for name, T in C.cam_a_candidates(calibration, rectification).items():
        e = reprojection_errors(points, pixels, K_a, T)
        errors[name] = {"median_px": float(np.median(e)) if len(e) else float("inf"),
                        "within_5px": float(np.mean(e < 5)) if len(e) else 0.0}
    best = min(errors, key=lambda k: errors[k]["median_px"])
    a_from_left = C.cam_a_candidates(calibration, rectification)[best]
    write_json(d / "alignment.json", {"reading": best, "a_from_left": a_from_left, "R1": R1,
                                      "features": int(len(points)), "candidates": errors})

    # Depth in ego_cam_a's undistorted view, at half resolution so every pixel gets a stereo point.
    w, h = cal["cam_a"]["size"]
    K_half = K_a.copy()
    K_half[:2] *= 0.5
    K_half[:2, 2] -= 0.25  # pixel-centre convention at half scale
    half = (w // 2, h // 2)
    rays = pixel_rays(Kr, tuple(rect["size"]))
    to_a = a_from_left[:3, :3] @ R1.T
    (d / "depth_a_half").mkdir(exist_ok=True)
    coverage = []
    for i, f in enumerate(files):
        z = read_depth(f).ravel()
        keep = (z > NEAR) & (z < FAR)
        X = (rays[keep] * z[keep, None]) @ to_a.T + a_from_left[:3, 3]
        X = X[X[:, 2] > NEAR]
        uv = X @ K_half.T
        u, v = np.round(uv[:, 0] / uv[:, 2]).astype(int), np.round(uv[:, 1] / uv[:, 2]).astype(int)
        inside = (u >= 0) & (u < half[0]) & (v >= 0) & (v < half[1])
        out = np.full(half[0] * half[1], np.inf)
        np.minimum.at(out, v[inside] * half[0] + u[inside], X[inside, 2])
        out[~np.isfinite(out)] = 0.0
        coverage.append(float(np.mean(out > 0)))
        write_depth(d / f"depth_a_half/{i:06d}.png", out.reshape(half[1], half[0]))
    write_json(d / "depth_a_half/intrinsics.json", {"K": K_half, "size": list(half)})

    fitted = [p for p in planes if "d" in p]
    rest = [p for p in fitted if p["frame"] < REST_FRAMES]
    table_dist = np.array([p["d"] for p in rest])
    chosen = errors[best]
    metrics = [
        Metric("depth frames minus manifest", len(files) - n, "frames", ok=(0, 0)),
        Metric("pixels with depth 0.1-5 m (median frame)", float(np.median(valid)) if valid else 0.0, "",
               ok=(0.8, 1), warn=(0.6, 1)),
        Metric("frames with a table plane", len(fitted) / max(len(planes), 1), "", ok=(0.9, 1), warn=(0.7, 1)),
        Metric("table plane inlier share (median)", float(np.median([p["inlier_fraction"] for p in fitted]))
               if fitted else 0.0, "", ok=(0.3, 1), warn=(0.15, 1), note="1 cm band, lower 60% of the image"),
        Metric("table plane RMS (median)", 100 * float(np.median([p["rms_m"] for p in fitted])) if fitted else
               float("inf"), "cm", ok=(0, 0.6), warn=(0, 1.0)),
        Metric("camera-to-table distance, first 30 frames (median)", float(np.median(table_dist)) if len(rest)
               else float("nan"), "m", ok=(0.2, 1.2), warn=(0.1, 1.5)),
        Metric("camera-to-table distance spread, first 30 frames", 100 * float(np.std(table_dist)) if len(rest)
               else float("inf"), "cm", ok=(0, 3), warn=(0, 6), note="head still, hands raised"),
        Metric("stereo-to-ego_cam_a reprojection error (median)", chosen["median_px"], "px", ok=(0, 4),
               warn=(0, 10), note=f"reading '{best}', {len(points)} SIFT features"),
        Metric("features within 5 px of ego_cam_a", chosen["within_5px"], "", ok=(0.5, 1), warn=(0.25, 1)),
        Metric("ego_cam_a pixels with stereo depth (median frame)", float(np.median(coverage)), "",
               ok=(0.6, 1), warn=(0.4, 1), note="half resolution"),
    ]
    return StageResult(metrics, {"depth_rect": d / "depth_rect", "planes": d / "planes.json",
                                 "alignment": d / "alignment.json", "depth_a_half": d / "depth_a_half"})
