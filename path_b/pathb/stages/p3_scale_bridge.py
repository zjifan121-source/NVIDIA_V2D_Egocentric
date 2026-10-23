"""P3 scale bridge (v3): a scale check of P1c's camera against the stereo depth, and the table.

v3 measures everything from the episode's own videos, so the bridge to the team's bundle is gone.
P1c is already metric (DROID with the stereo depth as its prior), so its camera height above the
table should match the stereo one: over the rest window (pathb.phases.rest_window_2d), per frame,
  stereo  the camera's distance to the support plane in ego_cam_a coordinates (P2's planes): the
          largest plane at most 35 cm under the objects' stereo surface points (P1b mask centres;
          the rule P3 has used since 4c2eac31), normals turned toward the camera
  P1c     the camera's height above that plane in P1c's world, the plane being the median of the
          per-frame planes moved into the world by P1c's poses
factor = median(stereo / P1c), expected 1.00 (pass 0.97-1.03, warn 0.9-1.1: a factor outside
says P1c is wrong). scale.json also carries the table plane (n, d) and gravity in P1c's world, from
which P4 builds the v3 world (table top z = 0, +z up). V3_BUILD_SPEC.md P3.
"""
from __future__ import annotations

import json

import numpy as np

from .. import baseline as B
from ..calib import stereo_frame
from ..geometry import plane_in, read_depth, read_transforms
from ..phases import rest_window_2d
from ..report import Metric, StageResult
from ..runner import write_json

REST_CAP = 30  # frames: the opening-frames rule
MIN_FRAMES = 5
MAX_TILT_DEG = 20.0  # a plane tilted further from "up" is not a support surface
SUPPORT_BELOW = 0.35  # the support surface lies at most this far (m) below the object's surface point
RAY_CONE_DEG = 1.5  # mesh vertices this close to the ray stand for the surface it hits
WINDOW = 3  # half-size of the depth window sampled around the ray (half-resolution pixels)


def table_distance(vertices: np.ndarray, object_to_camera: np.ndarray, up: np.ndarray) -> float:
    """Camera's height above the lowest point of an object resting on the table, along `up`."""
    X = vertices @ object_to_camera[:3, :3].T + object_to_camera[:3, 3]
    return float(-(X @ up).min())


def planes_in_cam_a(frame_planes: list[dict], alignment: dict) -> list[tuple[np.ndarray, float]]:
    """A frame's stereo planes (rectified left camera) in ego_cam_a coordinates."""
    R1, a_from_left = np.array(alignment["R1"]), np.array(alignment["a_from_left"])
    # the rectified and unrectified left cameras share their origin
    return [plane_in(a_from_left, R1.T @ np.array(p["normal"]), p["d"]) for p in frame_planes]


def support_plane(planes: list[tuple[np.ndarray, float, float]], up: np.ndarray,
                  point: np.ndarray) -> tuple[float, float] | None:
    """(distance from the camera, tilt from `up` in degrees) of the surface the object stands on,
    from (normal, distance, share of points) planes: the largest horizontal plane at most
    SUPPORT_BELOW under the object's surface point. Not the nearest plane below it: a dish rack's
    base or flat palms often lie between object and table (ep13: 12 of 18 rest frames)."""
    options = []
    for normal, distance, share in planes:
        angle = float(np.degrees(np.arccos(np.clip(normal @ up, -1, 1))))
        height = normal @ point + distance  # object's surface point above the plane
        if angle <= MAX_TILT_DEG and -0.03 <= height <= SUPPORT_BELOW:
            options.append((share, distance, angle))
    if not options:
        return None
    _, distance, angle = max(options)
    return float(distance), angle


def ray_surface(X: np.ndarray, direction: np.ndarray) -> float:
    """Distance to the nearest mesh vertex within RAY_CONE_DEG of the ray (nan if none)."""
    r = np.linalg.norm(X, axis=1)
    near = (X @ direction) / r > np.cos(np.radians(RAY_CONE_DEG))
    return float(r[near].min()) if near.any() else np.nan


def stereo_along(depth: np.ndarray, K: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    """Stereo surface point on the ray (median depth in a small window around its pixel)."""
    if direction[2] <= 0:
        return None
    u, v = (K @ direction)[:2] / (K @ direction)[2]
    u, v = int(round(u)), int(round(v))
    h, w = depth.shape
    if not (WINDOW <= u < w - WINDOW and WINDOW <= v < h - WINDOW):
        return None
    z = depth[v - WINDOW:v + WINDOW + 1, u - WINDOW:u + WINDOW + 1]
    z = z[z > 0]
    return None if z.size == 0 else float(np.median(z)) * direction / direction[2]


def object_points(ctx, t: int, K_full: np.ndarray, K_half: np.ndarray, depth: np.ndarray) -> list[np.ndarray]:
    """Each object's stereo surface point in ego_cam_a coordinates: the depth along the ray
    through its P1b mask's centre."""
    out = []
    for name, folder in B.mask_tracks(ctx.root).items():
        if not name.startswith("obj_"):
            continue
        m = B.read_mask(folder, t)
        if m is None or not m.any():
            continue
        ys, xs = np.nonzero(m)
        ray = np.linalg.inv(K_full) @ np.array([xs.mean(), ys.mean(), 1.0])
        point = stereo_along(depth, K_half, ray / np.linalg.norm(ray))
        if point is not None:
            out.append(point)
    return out


def table_in_camera(planes: list[tuple[np.ndarray, float, float]], points: list[np.ndarray]):
    """(normal toward the camera, distance, share) of the largest plane that every object point
    stands on: between 3 cm under and SUPPORT_BELOW above it."""
    best = None
    for n, d, share in planes:
        if d < 0:  # the camera (origin) on the positive side
            n, d = -n, -d
        heights = [float(n @ p + d) for p in points]
        if heights and all(-0.03 <= h <= SUPPORT_BELOW for h in heights) and (best is None or share > best[2]):
            best = (n, d, share)
    return best


def run(ctx) -> StageResult:
    d = ctx.dir("p3_scale_bridge")
    p2 = ctx.root / "p2_depth"
    planes = {p["frame"]: p["planes"] for p in json.loads((p2 / "planes.json").read_text())["planes"]}
    alignment = json.loads((p2 / "alignment.json").read_text())
    K_half = np.array(json.loads((p2 / "depth_a_half/intrinsics.json").read_text())["K"])
    K_full = np.array(json.loads((ctx.root / "p1_calibrate/calib.json").read_text())["cam_a"]["K"])
    n = int(ctx.episode["num_frames"])
    c2w, have = read_transforms(ctx.root / "p1c_camera/camera_to_world", n)
    tracks = B.mask_tracks(ctx.root)
    opening = range(min(n, REST_CAP))
    rest = rest_window_2d([[B.read_mask(p, k) for k in opening] for t, p in tracks.items() if t.startswith("obj_")],
                          [[B.read_mask(p, k) for k in opening] for t, p in tracks.items() if t.startswith("hand_")],
                          cap=REST_CAP)
    rows = []
    for t in range(rest[0], rest[1] + 1):
        if not have[t]:
            continue
        depth = read_depth(p2 / f"depth_a_half/{t:06d}.png")
        points = object_points(ctx, t, K_full, K_half, depth)
        frame_planes = planes.get(stereo_frame(t, alignment, len(planes)), [])
        in_a = planes_in_cam_a(frame_planes, alignment)
        table = table_in_camera([(nn, dd, p["inlier_fraction"]) for (nn, dd), p in zip(in_a, frame_planes)], points)
        if table is None:
            continue
        nc, dc, share = table
        R, c = c2w[t, :3, :3], c2w[t, :3, 3]
        nw = R @ nc
        rows.append({"frame": t, "stereo_height": float(dc), "share": share, "normal_world": nw,
                     "d_world": float(dc - nw @ c), "objects_seen": len(points)})
    if len(rows) < MIN_FRAMES:
        raise RuntimeError(f"only {len(rows)} rest frames {rest} have a support plane under the objects")
    normals = np.array([r["normal_world"] for r in rows])
    nw = normals.mean(axis=0)
    nw /= np.linalg.norm(nw)
    dw = float(np.median([r["d_world"] for r in rows]))
    ratios = []
    for r in rows:
        c = c2w[r["frame"], :3, 3]
        r["p1c_height"] = float(nw @ c + dw)
        ratios.append(r["stereo_height"] / r["p1c_height"])
    ratios = np.array(ratios)
    factor = float(np.median(ratios))
    tilt = np.degrees(np.arccos(np.clip(normals @ nw, -1, 1)))
    scale = {"factor": factor, "method": "stereo support plane vs P1c camera height (v3)", "rest_window": rest,
             "table_plane": {"normal": nw, "d": dw, "frame": "P1c world (camera_to_world)"},
             "gravity": -nw, "frames": rows}
    write_json(d / "scale.json", scale)
    metrics = [
        Metric("rest frames with a support plane", len(rows), "frames", ok=(10, np.inf), warn=(MIN_FRAMES, np.inf),
               note=f"rest window {rest} (2-D: before a hand touches an object)"),
        Metric("camera height above the table, stereo (median)", float(np.median([r["stereo_height"] for r in rows])), "m"),
        Metric("camera height above the table, P1c (median)", float(np.median([r["p1c_height"] for r in rows])), "m"),
        Metric("scale factor (stereo / P1c)", factor, "x", ok=(0.97, 1.03), warn=(0.9, 1.1),
               note="expected 1.00: P1c is metric; outside 0.9-1.1 P1c is wrong"),
        Metric("scale spread over rest frames (IQR / median)", 100 * float(np.subtract(*np.percentile(ratios, [75, 25])))
               / factor, "%", ok=(0, 5), warn=(0, 10)),
        Metric("support plane normal spread in the world (max)", float(tilt.max()), "deg", ok=(0, 5), warn=(0, 10)),
    ]
    return StageResult(metrics, {"scale": d / "scale.json", "factor": factor})


def gt_trajectory_factor(gt: np.ndarray, data: dict, slots) -> dict | None:
    """The factor that would make the bundle's object motion the size of the ground truth's.

    Per object, the median over frames of |GT displacement from frame 0| / |bundle displacement|
    where the GT object has moved more than 3 cm; for two objects also the ratio of their frame-0
    distances. Both are rotation-free, so the bundle's world need not be aligned to the mocap's.
    """
    # gt: (frames, objects, 7) from pathb.gt.ground_truth, per-frame slot order = episodes.json order
    n = min(len(gt), len(data["camera_to_world_transform"]))
    parts = {}
    for i, (pose, _, _) in enumerate(slots[: gt.shape[1]]):
        g, b = gt[:n, i, :3], data[pose][:n, :3, 3].astype(np.float64)
        dg, db = np.linalg.norm(g - g[0], axis=1), np.linalg.norm(b - b[0], axis=1)
        moving = (dg > 0.03) & (db > 1e-3)
        if moving.sum() >= 10:
            parts[f"motion of object {i}"] = float(np.median(dg[moving] / db[moving]))
    if gt.shape[1] >= 2 and len(slots) >= 2:
        g01 = float(np.linalg.norm(gt[0, 0, :3] - gt[0, 1, :3]))
        b01 = float(np.linalg.norm(data[slots[0][0]][0, :3, 3] - data[slots[1][0]][0, :3, 3]))
        if b01 > 1e-3:
            parts["frame-0 distance between the objects"] = g01 / b01
    return {"factor": float(np.median(list(parts.values()))), "parts": parts} if parts else None
