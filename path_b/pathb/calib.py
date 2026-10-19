"""Factory calibration of the head rig, applied to the released (upright) Track 3 frames.

The head OAK is mounted upside down (rotation_deg 180) and the released videos are already turned
upright (README §5.6). The calibration states its stereo pair for upright frames (left ego_cam_c,
right ego_cam_b; see the note in its stereo_pairs), but it does not say whether the listed
intrinsics are native or upright, and the per-camera extrinsics_to_stereo_left name no reference
camera. So every plausible reading is built here (READINGS, cam_a_candidates), and P1 and P2 keep
the one that fits the episode's own frames best, recording all of them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .paths import CALIBRATION

# OpenCV coefficient order: k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tau_x tau_y. The tangential,
# thin-prism and sensor-tilt terms change sign when the image turns 180 degrees.
ODD_UNDER_TURN = [2, 3, 8, 9, 10, 11, 12, 13]
TURN = np.diag([-1.0, -1.0, 1.0])  # a 180 degree turn about the optical axis

# name: (turn the listed intrinsics upright, swap the stated left and right cameras)
READINGS = {"listed": (False, False), "listed_swapped": (False, True),
            "rotated": (True, False), "rotated_swapped": (True, True)}


def load(path: Path = CALIBRATION) -> dict:
    return json.loads(Path(path).read_text())


@dataclass
class Intrinsics:
    K: np.ndarray
    D: np.ndarray  # 14 coefficients, OpenCV order
    size: tuple[int, int]  # width, height


def intrinsics(calib: dict, camera: str, rotate: bool) -> Intrinsics:
    c = calib["cameras"][camera]
    K = np.array(c["intrinsic_matrix"], dtype=np.float64)
    D = np.array(c["distortion_coefficients"], dtype=np.float64)
    w, h = (int(v) for v in c["resolution"])
    if rotate:
        K[0, 2], K[1, 2] = w - 1 - K[0, 2], h - 1 - K[1, 2]
        D[ODD_UNDER_TURN] *= -1
    return Intrinsics(K, D, (w, h))


@dataclass
class Rectification:
    reading: str
    left: str  # camera names, e.g. ego_cam_c
    right: str
    R: np.ndarray  # X_right = R X_left + T, unrectified camera frames (OpenCV stereoCalibrate convention)
    T: np.ndarray
    R1: np.ndarray  # X_rectified_left = R1 X_left
    R2: np.ndarray  # X_rectified_right = R2 X_right
    P1: np.ndarray
    P2: np.ndarray
    size: tuple[int, int]
    maps: tuple

    @property
    def K(self) -> np.ndarray:
        return self.P1[:3, :3]

    @property
    def baseline(self) -> float:
        return float(-self.P2[0, 3] / self.P2[0, 0])

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        (lx, ly), (rx, ry) = self.maps
        return cv2.remap(left, lx, ly, cv2.INTER_LINEAR), cv2.remap(right, rx, ry, cv2.INTER_LINEAR)

    def summary(self) -> dict:
        K = self.K
        return {"reading": self.reading, "left": self.left, "right": self.right, "size": list(self.size),
                "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "baseline": self.baseline, "R1": self.R1.tolist(), "R": self.R.tolist(), "T": self.T.tolist()}


def rectify_rig(reading: str, left: str, right: str, L: Intrinsics, Rt: Intrinsics,
                R: np.ndarray, T: np.ndarray) -> Rectification:
    R1, R2, P1, P2, *_ = cv2.stereoRectify(L.K, L.D, Rt.K, Rt.D, L.size, np.asarray(R, float),
                                           np.asarray(T, float).reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    maps = tuple(cv2.initUndistortRectifyMap(I.K, I.D, Rr, P, I.size, cv2.CV_32FC1)
                 for I, Rr, P in ((L, R1, P1), (Rt, R2, P2)))
    return Rectification(reading, left, right, np.asarray(R, float), np.asarray(T, float).ravel(), R1, R2, P1, P2,
                         L.size, maps)


def rectification(calib: dict, reading: str) -> Rectification:
    rotate, swap = READINGS[reading]
    pair = next(p for p in calib["stereo_pairs"] if p["camera"] == "ego")
    left, right = pair["left"], pair["right"]
    M = np.array(pair["left_to_right"], dtype=np.float64)
    R, T = M[:3, :3], M[:3, 3]
    if swap:
        left, right, R, T = right, left, R.T, -R.T @ T
    return rectify_rig(reading, left, right, intrinsics(calib, left, rotate), intrinsics(calib, right, rotate), R, T)


def epipolar_check(rect: Rectification, pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Matched SIFT features (sub-pixel) of rectified pairs: a correct rectification puts each
    match on the same row, left feature right of its match (positive disparity). Also fits the
    row offset dy = y_left - y_right = a + b (x - w/2) + c (y - h/2) that refine_rows removes."""
    sift = cv2.SIFT_create(3000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    for left, right in pairs:
        a, b = rect.rectify(left, right)
        ka, da = sift.detectAndCompute(a, None)
        kb, db = sift.detectAndCompute(b, None)
        if da is None or db is None:
            continue
        for m in matcher.knnMatch(da, db, k=2):
            if len(m) == 2 and m[0].distance < 0.7 * m[1].distance:
                (xa, ya), (xb, yb) = ka[m[0].queryIdx].pt, kb[m[0].trainIdx].pt
                if abs(ya - yb) < 30:
                    rows.append((xa, ya, ya - yb, xa - xb))
    if len(rows) < 10:
        return {"reading": rect.reading, "matches": len(rows), "median_dy_px": float("inf"),
                "positive_disparity_fraction": 0.0, "row_fit": [0.0, 0.0, 0.0], "fit_residual_px": float("inf")}
    x, y, dy, disparity = np.array(rows).T
    w, h = rect.size
    A = np.column_stack([np.ones_like(x), x - w / 2, y - h / 2])
    fit = np.linalg.lstsq(A, dy, rcond=None)[0]
    on_row = np.abs(dy) < 2.0
    return {"reading": rect.reading, "matches": int(len(dy)), "median_dy_px": float(np.median(np.abs(dy))),
            "positive_disparity_fraction": float(np.mean(disparity[on_row] > 0.5)) if on_row.any() else 0.0,
            "row_fit": fit.tolist(), "fit_residual_px": float(np.median(np.abs(dy - A @ fit)))}


def refine_rows(rect: Rectification, fit: list[float]) -> Rectification:
    """Shift the rectified right image's rows by the fitted offset, so matches share a row: the
    factory rectification drifts by a pixel or two in some recordings. Disparity is untouched."""
    a, b, c = fit
    w, h = rect.size
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    source_rows = ys - (a + b * (xs - w / 2) + c * (ys - h / 2)).astype(np.float32)
    rx, ry = rect.maps[1]
    right = (cv2.remap(rx, xs, source_rows, cv2.INTER_LINEAR), cv2.remap(ry, xs, source_rows, cv2.INTER_LINEAR))
    return Rectification(rect.reading, rect.left, rect.right, rect.R, rect.T, rect.R1, rect.R2, rect.P1, rect.P2,
                         rect.size, (rect.maps[0], right))


@dataclass
class Undistorter:
    K: np.ndarray  # pinhole of the undistorted image (same as the factory K)
    size: tuple[int, int]
    maps: tuple

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, *self.maps, cv2.INTER_LINEAR)


def undistorter(calib: dict, rotate: bool, camera: str = "ego_cam_a") -> Undistorter:
    I = intrinsics(calib, camera, rotate)
    return Undistorter(I.K, I.size, cv2.initUndistortRectifyMap(I.K, I.D, None, I.K, I.size, cv2.CV_32FC1))


def cam_a_candidates(calib: dict, rect: Rectification) -> dict[str, np.ndarray]:
    """T_a<-left (4x4: X_a = T X_left, unrectified left camera) for each reading of ego_cam_a's
    extrinsics_to_stereo_left: either direction, relative to either stereo camera, as listed or
    turned upright. P2 keeps the one that maps stereo depth onto ego_cam_a's own features best."""
    E = np.array(calib["cameras"]["ego_cam_a"]["extrinsics_to_stereo_left"], dtype=np.float64)
    right_from_left = np.eye(4)
    right_from_left[:3, :3], right_from_left[:3, 3] = rect.R, rect.T
    turn = np.eye(4)
    turn[:3, :3] = TURN
    out = {}
    for mirrored in (False, True):
        M = turn @ E @ turn if mirrored else E
        for direction, a_from_ref in (("a_to_ref", np.linalg.inv(M)), ("ref_to_a", M)):
            for ref, ref_from_left in (("left", np.eye(4)), ("right", right_from_left)):
                out[f"{direction}_{ref}{'_turned' if mirrored else ''}"] = a_from_ref @ ref_from_left
    return out
