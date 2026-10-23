"""The v3 world frame (V3_BUILD_SPEC.md §3): table top z = 0, +z up, +y the mean camera forward
projected on the table; the origin, which the spec leaves open, is the point of the table under the
camera at the first rest frame. Built from P3's table plane in P1c's world."""
from __future__ import annotations

import numpy as np


def from_table(normal: np.ndarray, d: float, c2w: np.ndarray, frames: list[int]) -> np.ndarray:
    """4x4 W mapping P1c-world points to v3-world points (x' = W x)."""
    z = np.asarray(normal, dtype=np.float64)
    z /= np.linalg.norm(z)
    forward = np.mean([c2w[t, :3, 2] for t in frames], axis=0)  # camera +z (OpenCV) in P1c's world
    y = forward - (forward @ z) * z
    if np.linalg.norm(y) < 1e-6:
        raise ValueError("the camera looks straight along the table normal: +y is undefined")
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    c0 = c2w[frames[0], :3, 3]
    origin = c0 - (z @ c0 + d) * z  # on the plane z . p + d = 0
    R = np.stack([x, y, z])  # rows: the v3 axes in P1c's world
    W = np.eye(4)
    W[:3, :3] = R
    W[:3, 3] = -R @ origin
    return W
