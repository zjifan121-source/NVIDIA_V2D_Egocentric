"""Integer-pixel OpenCV triangle raster with perspective surface provenance.

Invalid pixels return depth=0, face_index=-1, barycentric=(0,0,0).
Equal-depth ties retain the first face. No segmentation, visibility confidence,
pose gate, or MANO wrist-cap exclusion is applied here: face indices let callers
make those decisions explicitly. Faces crossing z<=1e-5 are skipped, matching
tools.track3.r78.geometry_cpu.raster (no near-plane clipping).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=False)
def _raster_surface(vertices, faces, K, height, width):
    zbuf = np.full((height, width), np.inf)
    face_index = np.full((height, width), -1, dtype=np.int64)
    barycentric = np.zeros((height, width, 3), dtype=np.float64)
    projected = vertices @ K.T
    for i in range(len(vertices)):
        if projected[i, 2] != 0:
            projected[i, 0] /= projected[i, 2]
            projected[i, 1] /= projected[i, 2]
    for fi in range(len(faces)):
        i, j, k = faces[fi]
        z0, z1, z2 = vertices[i, 2], vertices[j, 2], vertices[k, 2]
        if min(z0, z1, z2) <= 1e-5:
            continue
        ax, ay = projected[i, 0], projected[i, 1]
        bx, by = projected[j, 0], projected[j, 1]
        cx, cy = projected[k, 0], projected[k, 1]
        low_x = max(0, int(np.ceil(min(ax, bx, cx))))
        high_x = min(width - 1, int(np.floor(max(ax, bx, cx))))
        low_y = max(0, int(np.ceil(min(ay, by, cy))))
        high_y = min(height - 1, int(np.floor(max(ay, by, cy))))
        if low_x > high_x or low_y > high_y:
            continue
        den = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(den) < 1e-12:
            continue
        for y in range(low_y, high_y + 1):
            for x in range(low_x, high_x + 1):
                a = ((by-cy)*(x-cx)+(cx-bx)*(y-cy))/den
                b = ((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/den
                c = 1-a-b
                if a >= -1e-10 and b >= -1e-10 and c >= -1e-10:
                    inv = a/z0+b/z1+c/z2
                    if inv > 0:
                        depth = 1/inv
                        if depth < zbuf[y, x]:
                            zbuf[y, x] = depth
                            face_index[y, x] = fi
                            barycentric[y, x, 0] = a/z0/inv
                            barycentric[y, x, 1] = b/z1/inv
                            barycentric[y, x, 2] = c/z2/inv
    for y in range(height):
        for x in range(width):
            if not np.isfinite(zbuf[y, x]):
                zbuf[y, x] = 0.0
    return zbuf, face_index, barycentric


def raster_surface(vertices, faces, K, shape):
    """Return camera-Z, face indices and 3-D (not screen-space) barycentrics."""
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    K = np.ascontiguousarray(K, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite [N,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F,3]")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("face vertex index out of bounds")
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.array_equal(K[2], [0, 0, 1]):
        raise ValueError("K must be finite OpenCV pinhole intrinsics")
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must be positive (height,width)")
    return _raster_surface(vertices, faces, K, int(shape[0]), int(shape[1]))
