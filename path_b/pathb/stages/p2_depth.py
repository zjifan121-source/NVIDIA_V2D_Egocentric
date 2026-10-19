"""P2 stereo depth: FoundationStereo on P1's rectified pairs, the large planes in every frame, and
the depth reprojected into the undistorted ego_cam_a view.

The stereo model is the baseline's v2d_foundation_stereo module, called read-only through its own
launcher (TensorRT engine at 576x960; the module resizes and scales disparity back).

ego_cam_a's pose relative to the stereo pair comes from the calibration (cam_a_candidates, all
readings) or from the data: stereo depth at SIFT features of the rectified left image, matched in
ego_cam_a, gives a PnP estimate. The candidate that puts the most features within 5 px is used;
how far the calibration's best reading is from the PnP estimate is reported.

Writes depth_rect/ (module output, rectified left frame), planes.json, alignment.json and
depth_a_half/ (ego_cam_a's view at half resolution).
"""
from __future__ import annotations

import json
import subprocess

import cv2
import numpy as np

from .. import calib as C
from ..geometry import fit_planes, pixel_rays, read_depth, table_points, write_depth
from ..paths import HOST_PY, RECON, STEREO_WEIGHTS
from ..report import Metric, StageResult
from ..runner import write_json

ALIGN_FRAMES = 6  # frames whose features locate ego_cam_a
NEAR, FAR = 0.1, 5.0  # metres of stereo depth used outside the plane fit


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
    left image) and the ego_cam_a pixels where the same features were found."""
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


def pnp(points: np.ndarray, pixels: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """T_a<-left from the features themselves (RANSAC, 4 px)."""
    if len(points) < 20:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(points, pixels, K, None, reprojectionError=4.0,
                                                 iterationsCount=2000, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(points[inliers[:, 0]], pixels[inliers[:, 0]], K, None, rvec, tvec)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return T


def pose_difference(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """(rotation in degrees, translation in mm) between two rigid transforms."""
    R = A[:3, :3] @ B[:3, :3].T
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return float(angle), float(1000 * np.linalg.norm(A[:3, 3] - B[:3, 3]))


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
        planes.append({"frame": i, "planes": fit_planes(table_points(depth, Kr), seed=i)})
    write_json(d / "planes.json", {"frame": "rectified left camera", "planes": planes})

    # Where ego_cam_a sits: the calibration's readings and the PnP estimate, judged on the features.
    calibration = C.load()
    K_a = np.array(cal["cam_a"]["K"])
    picks = np.linspace(0, len(files) - 1, ALIGN_FRAMES).round().astype(int).tolist()
    points, pixels = feature_pairs(p1, d, Kr, R1, picks)
    candidates = C.cam_a_candidates(calibration, C.rectification(calibration, cal["reading"]))
    calibrated = {}
    for name, T in candidates.items():
        e = reprojection_errors(points, pixels, K_a, T)
        calibrated[name] = {"median_px": float(np.median(e)) if len(e) else float("inf"),
                            "within_5px": float(np.mean(e < 5)) if len(e) else 0.0}
    best_calibrated = max(calibrated, key=lambda k: calibrated[k]["within_5px"])
    options = dict(calibrated)
    T_pnp = pnp(points, pixels, K_a)
    if T_pnp is not None:
        e = reprojection_errors(points, pixels, K_a, T_pnp)
        options["pnp"] = {"median_px": float(np.median(e)), "within_5px": float(np.mean(e < 5))}
        candidates["pnp"] = T_pnp
    best = max(options, key=lambda k: options[k]["within_5px"])
    a_from_left = candidates[best]
    gap = pose_difference(candidates[best_calibrated], T_pnp) if T_pnp is not None else (np.nan, np.nan)
    write_json(d / "alignment.json", {"reading": best, "a_from_left": a_from_left, "R1": R1,
                                      "features": int(len(points)), "candidates": options,
                                      "best_calibrated": best_calibrated,
                                      "calibration_vs_pnp": {"deg": gap[0], "mm": gap[1]}})

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

    first = [p["planes"][0] for p in planes if p["planes"]]
    chosen = options[best]
    metrics = [
        Metric("depth frames minus manifest", len(files) - n, "frames", ok=(0, 0)),
        Metric("pixels with depth 0.1-5 m (median frame)", float(np.median(valid)) if valid else 0.0, "",
               ok=(0.8, 1), warn=(0.6, 1)),
        Metric("frames with a plane", len(first) / max(len(planes), 1), "", ok=(0.9, 1), warn=(0.7, 1)),
        Metric("planes per frame (median)", float(np.median([len(p["planes"]) for p in planes])), ""),
        Metric("largest plane RMS (median)", 100 * float(np.median([p["rms_m"] for p in first])) if first else
               float("inf"), "cm", ok=(0, 0.6), warn=(0, 1.0), note="stereo noise on a flat surface"),
        Metric("stereo-to-ego_cam_a features within 5 px", chosen["within_5px"], "", ok=(0.6, 1), warn=(0.35, 1),
               note=f"pose '{best}', {len(points)} SIFT features over {len(picks)} frames"),
        Metric("stereo-to-ego_cam_a reprojection error (median)", chosen["median_px"], "px", ok=(0, 4),
               warn=(0, 10), note="includes mismatched features"),
        Metric("calibration vs data: ego_cam_a rotation", gap[0], "deg", ok=(0, 1), warn=(0, 3),
               note=f"best calibration reading '{best_calibrated}' vs PnP"),
        Metric("calibration vs data: ego_cam_a position", gap[1], "mm", ok=(0, 10), warn=(0, 25)),
        Metric("ego_cam_a pixels with stereo depth (median frame)", float(np.median(coverage)), "",
               ok=(0.6, 1), warn=(0.4, 1), note="half resolution"),
    ]
    return StageResult(metrics, {"depth_rect": d / "depth_rect", "planes": d / "planes.json",
                                 "alignment": d / "alignment.json", "depth_a_half": d / "depth_a_half"})
