"""Depth files, back-projection and the table plane."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_depth(path: Path) -> np.ndarray:
    """Metres from a v2d_common inverse-depth PNG (pixel = 65535 / (depth_m + 1)); 0 = no depth."""
    pixel = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    with np.errstate(divide="ignore"):
        depth = 65535.0 / pixel - 1.0
    depth[pixel == 0] = 0.0  # infinity
    return depth


def write_depth(path: Path, depth: np.ndarray) -> None:
    """Same encoding as read_depth; pixels without depth (<= 0) are written as 0."""
    pixel = np.where(depth > 0, 65535.0 / (depth + 1.0), 0.0)
    cv2.imwrite(str(path), pixel.clip(0, 65535).astype(np.uint16))


def pixel_rays(K: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(H*W, 3) camera rays at unit depth for every pixel, row-major."""
    w, h = size
    v, u = np.mgrid[0:h, 0:w]
    return np.stack([(u.ravel() - K[0, 2]) / K[0, 0], (v.ravel() - K[1, 2]) / K[1, 1], np.ones(h * w)], axis=1)


def table_points(depth: np.ndarray, K: np.ndarray, from_row: float = 0.4, stride: int = 6,
                 near: float = 0.15, far: float = 1.5) -> np.ndarray:
    """Points where the table usually is in a head-camera frame: the lower 60% of the image, within
    arm's reach."""
    h, w = depth.shape
    v, u = np.mgrid[int(h * from_row):h:stride, 0:w:stride]
    z = depth[v, u]
    keep = (z > near) & (z < far)
    u, v, z = u[keep], v[keep], z[keep]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)


def fit_plane(points: np.ndarray, iterations: int = 256, threshold: float = 0.01, seed: int = 0) -> dict | None:
    """Largest plane n.X + d = 0 (RANSAC, then least squares on its inliers), |n| = 1, oriented
    so the camera at the origin is on its positive side: d is the camera's distance to it."""
    if len(points) < 100:
        return None
    rng = np.random.default_rng(seed)
    a, b, c = (points[i] for i in rng.integers(0, len(points), size=(3, iterations)))
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1)
    good = norm > 1e-9
    n, a = n[good] / norm[good, None], a[good]
    d = -np.einsum("ij,ij->i", n, a)
    inliers = np.abs(points @ n.T + d) < threshold
    inliers = inliers[:, inliers.sum(axis=0).argmax()]
    P = points[inliers]
    centroid = P.mean(axis=0)
    normal = np.linalg.svd(P - centroid)[2][-1]
    offset = -normal @ centroid
    if offset < 0:
        normal, offset = -normal, -offset
    residual = P @ normal + offset
    return {"normal": normal.tolist(), "d": float(offset), "inlier_fraction": float(inliers.mean()),
            "rms_m": float(np.sqrt(np.mean(residual ** 2))), "points": int(len(points))}


def plane_in(T: np.ndarray, normal: np.ndarray, d: float) -> tuple[np.ndarray, float]:
    """The plane n.X + d = 0 in another frame, where X_new = T X (4x4 rigid)."""
    n = T[:3, :3] @ np.asarray(normal, float)
    return n, float(d - n @ T[:3, 3])
