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


def _run_contract_tests():
    import argparse
    from datetime import datetime, timezone
    import hashlib
    import importlib.util
    import json
    from pathlib import Path
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    assert not args.report.exists(), "Refuse to replace an existing test report"
    spec = importlib.util.spec_from_file_location("reference_raster", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    started = time.monotonic()
    K = np.array([[40., 0, 20], [0, 35, 18], [0, 0, 1]])
    v = np.array([[-.3, -.2, 1.], [.5, -.2, 1.8], [0, .4, .8],
                  [-.2, -.1, .7], [.25, -.1, .7], [0, .3, .7],
                  [-.8, -.7, 1.], [-.3, -.7, 1.], [-.6, -.2, 1.],
                  [-.2, 0, -.1], [.2, 0, 1.], [0, .2, 1.]])
    f = np.array([[0, 1, 2], [3, 4, 5], [3, 4, 5], [6, 7, 8],
                  [9, 10, 11], [0, 0, 0]], dtype=np.int64)
    tests = []

    def check(name, vv, ff, kk, shape):
        tick = time.monotonic()
        z, idx, bary = raster_surface(vv, ff, kk, shape)
        renderer_seconds = time.monotonic() - tick
        truth = reference.raster(vv, ff, kk, shape)
        assert np.array_equal(z > 0, truth > 0), name
        valid = z > 0
        assert np.array_equal(idx >= 0, valid)
        assert np.all(idx[~valid] == -1) and np.all(bary[~valid] == 0)
        xyz = (vv[ff[idx[valid]]] * bary[valid, :, None]).sum(1)
        yy, xx = np.nonzero(valid)
        q = xyz @ kk.T
        uv = q[:, :2] / q[:, 2, None]
        depth_error = float(np.max(abs(z - truth)))
        position_z_error = float(np.max(abs(xyz[:, 2] - z[valid])))
        projection_error = float(np.max(abs(uv - np.c_[xx, yy])))
        sum_error = float(np.max(abs(bary[valid].sum(1) - 1)))
        assert depth_error < 1e-10 and position_z_error < 1e-10
        assert projection_error < 1e-8 and sum_error < 1e-12
        assert bary[valid].min() >= -1e-8
        tests.append({"name": name, "covered_pixels": int(valid.sum()),
                      "reference_z_max_error_m": depth_error,
                      "barycentric_reconstructed_z_max_error_m": position_z_error,
                      "barycentric_reprojection_max_error_px": projection_error,
                      "barycentric_sum_max_error": sum_error,
                      "renderer_seconds_including_first_JIT_if_applicable": renderer_seconds})
        return z, idx, bary

    z, idx, bary = check("synthetic_overlapping_slanted_clipped_degenerate_nearplane", v, f, K, (40, 45))
    assert 1 in idx and 2 not in idx, "Equal-depth duplicate keeps first face"
    assert 4 not in idx and 5 not in idx, "Near-plane and degenerate faces skipped"
    with np.load(args.geometry, allow_pickle=False) as geom:
        vv = geom["C"][0].astype(float)
        ff = geom["faces"].astype(np.int64)
        assert np.array_equal(geom["frames"], np.arange(288))
    metadata = json.loads(args.intrinsics.read_text())
    check("saved_C_frame0", vv, ff, np.array(metadata["K"]), tuple(metadata["size"][::-1]))
    report = {"state": "PASS", "at": datetime.now(timezone.utc).isoformat(),
              "exit_code": 0, "seconds": time.monotonic() - started,
              "tests": tests, "tie_rule": "first face on exact equal depth",
              "near_plane": "skip entire face if any camera Z <=1e-5; same as reference",
              "pixel_coordinates": "integer OpenCV, no added half-pixel",
              "barycentrics": "perspective-correct weights on original triangle vertices",
              "wrist_cap_or_pose_filters": False, "GPU": 0, "SSH": 0,
              "prediction_changes": 0,
              "sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [args.reference, args.geometry, args.intrinsics, Path(__file__)]}}
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _run_contract_tests()
