"""P3 scale bridge: one metric scale factor for the episode's existing reconstruction bundle.

Until Path B tracks objects itself (Stage 2), its output is the team's HF bundle rescaled to stereo
metric scale. Two rulers, both in the camera frame (the SLAM world's own scale problems stay out),
both over the rest frames before a hand reaches an object:
  support plane  the camera's height above the surface the object rests on: from stereo (the
                 largest horizontal plane at most 35 cm below the object's stereo surface point,
                 see support_plane) and from the bundle (its lowest mesh point along the bundle's
                 gravity "up");
  object ray     the distance from the camera to the object's surface along the ray through the
                 object's centre: from stereo depth in ego_cam_a's view and from the bundle mesh.
factor = median of the support-plane ratios (stereo / bundle); the object-ray ratio is the
cross-check, and the fallback when no support plane is found. Stage 2's rescale_world multiplies
every length in the bundle by the factor.
"""
from __future__ import annotations

import json

import numpy as np

from ..geometry import plane_in, read_depth
from ..report import Metric, StageResult
from ..runner import write_json
from ..team import retarget_tools, scorer

REST_FRAMES = 30  # recordings open with both hands raised over the table (README §5.6)
REACH_CM = 10.0  # rest ends when a wrist comes this close (stage 2's rest hold uses the same)
MIN_FRAMES = 5
MAX_TILT_DEG = 20.0  # a plane tilted further from the bundle's gravity is not a support surface
SUPPORT_BELOW = 0.35  # the support surface lies at most this far (m) below the object's surface point
RAY_CONE_DEG = 1.5  # mesh vertices this close to the ray stand for the surface it hits
WINDOW = 3  # half-size of the depth window sampled around the ray (half-resolution pixels)


def source_bundle(episode: int, objects: list[str], work):
    """The bundle stage 2 would build: merged for several objects, else the object's own."""
    L = retarget_tools()
    if len(objects) > 1:
        bundle, _ = L.merged_bundle(episode, work / "merged")
        return bundle
    return L.bundle_dir_for(episode, objects[0])


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


def measure(data: dict, bundle, frames: list[int], planes: dict, alignment: dict, depth_dir, K_half) -> dict:
    """Per rest frame: both rulers' readings, stereo and bundle."""
    L = retarget_tools()
    slots = L.object_slots(data)
    vertices = [L.object_vertices(bundle, data, scale_key, mesh) for _, scale_key, mesh in slots]
    c2w = data["camera_to_world_transform"].astype(np.float64)
    offset, last = int(alignment.get("offset", 0)), max(planes) if planes else 0
    rows = []
    for t in frames:  # bundle frame t = ego_cam_a frame t; depth_a_half already applies the offset
        stereo_planes = planes.get(min(max(t + offset, 0), last), [])
        up = c2w[t, :3, :3].T @ np.array([0.0, 0.0, 1.0])
        w2c = np.linalg.inv(c2w[t])
        depth = read_depth(depth_dir / f"{t:06d}.png")
        objects = []
        for V, (pose, _, _) in zip(vertices, slots):
            T = w2c @ data[pose][t].astype(np.float64)
            X = V @ T[:3, :3].T + T[:3, 3]
            direction = X.mean(axis=0) / np.linalg.norm(X.mean(axis=0))
            point = stereo_along(depth, K_half, direction)
            objects.append({"table_bundle": float(-(X @ up).min()), "ray_bundle": ray_surface(X, direction),
                            "point": point})
        # the lowest object rests on the support surface (a lid on a pot does not)
        anchor = max(objects, key=lambda o: o["table_bundle"])
        support, tilt = None, np.nan
        if anchor["point"] is not None:
            in_a = planes_in_cam_a(stereo_planes, alignment)
            chosen = support_plane([(n, dist, p["inlier_fraction"]) for (n, dist), p in zip(in_a, stereo_planes)],
                                   up, anchor["point"])
            if chosen:
                support, tilt = chosen
        by_object = [np.linalg.norm(o["point"]) / o["ray_bundle"]
                     if o["point"] is not None and np.isfinite(o["ray_bundle"]) else None for o in objects]
        rows.append({"frame": t, "table_bundle": anchor["table_bundle"], "table_stereo": support,
                     "tilt_deg": tilt, "ray_ratios": [r for r in by_object if r is not None],
                     "ray_by_object": by_object})
    return rows


def run(ctx) -> StageResult:
    L = retarget_tools()
    d = ctx.dir("p3_scale_bridge")
    p2 = ctx.root / "p2_depth"
    planes = {p["frame"]: p["planes"] for p in json.loads((p2 / "planes.json").read_text())["planes"]}
    alignment = json.loads((p2 / "alignment.json").read_text())
    K_half = np.array(json.loads((p2 / "depth_a_half/intrinsics.json").read_text())["K"])

    bundle = source_bundle(ctx.number, ctx.objects, d)
    data = dict(np.load(bundle / "result.npz"))
    slots = L.object_slots(data)
    reach = [L.reach_frame(data, L.object_vertices(bundle, data, scale_key, mesh), REACH_CM, pose)
             for pose, scale_key, mesh in slots]
    first_reach = min([REST_FRAMES] + [r for r in reach if r is not None])
    # A reach in the first frames means the mesh is wrong (ep24: blobs 7-9x the scan put the resting
    # hands "on" the object), so the reach test is dropped and the opening frames serve as rest.
    reach_fallback = first_reach < MIN_FRAMES
    rest_end = REST_FRAMES if reach_fallback else first_reach
    valid = data["object_is_valid"].astype(bool) & data["camera_is_valid"].astype(bool)
    frames = [t for t in range(min(rest_end, len(valid))) if valid[t]]
    if not frames:
        raise RuntimeError(f"no frame before {rest_end} has a valid camera and object in {bundle}")
    scan_ratios = {name: L.scan_size_ratio(L.object_vertices(bundle, data, scale_key, mesh), name)
                   for name, (_, scale_key, mesh) in zip(ctx.objects, slots)} if ctx.public else {}

    rows = measure(data, bundle, frames, planes, alignment, p2 / "depth_a_half", K_half)
    supported = [r for r in rows if r["table_stereo"] is not None]
    plane_ratios = np.array([r["table_stereo"] / r["table_bundle"] for r in supported])
    ray_ratios = np.array([x for r in rows for x in r["ray_ratios"]])
    plane_factor = float(np.median(plane_ratios)) if len(plane_ratios) >= MIN_FRAMES else None
    ray_factor = float(np.median(ray_ratios)) if len(ray_ratios) >= MIN_FRAMES else None
    method = "support plane" if plane_factor else "object ray" if ray_factor else "none (unscaled)"
    factor = plane_factor or ray_factor or 1.0
    used = plane_ratios if plane_factor else ray_ratios
    spread = float(np.subtract(*np.percentile(used, [75, 25])) / factor) if len(used) >= MIN_FRAMES else np.inf
    support = np.array([r["table_stereo"] for r in supported])

    hand = L.hand_scale(data)
    scale = {"factor": factor, "method": method, "applied": method != "none (unscaled)", "bundle": str(bundle),
             "rest_end_frame": int(rest_end), "first_reach_frame": int(first_reach), "reach_fallback": reach_fallback,
             "plane_factor": plane_factor, "ray_factor": ray_factor, "frames": rows}
    metrics = []
    if scan_ratios:
        worst_name = max(scan_ratios, key=lambda k: abs(np.log(scan_ratios[k] or 1.0)))
        metrics.append(Metric("bundle mesh / scan size (worst object)", scan_ratios[worst_name], "x", kind="gt",
                              ok=(0.5, 2.0), warn=(0.33, 3.0),
                              note=f"{worst_name}; far off means a blob or the wrong object, and every "
                                   "object-based ruler below is suspect"))
    metrics += [
        Metric("first frame a wrist reaches an object", int(first_reach), "frame", ok=(MIN_FRAMES, np.inf),
               warn=(1, np.inf), note="rest window forced to the opening frames (hands raised)" if reach_fallback
               else f"rest window = frames 0-{rest_end - 1}"),
        Metric("rest frames with a support plane", len(supported), "frames", ok=(10, np.inf),
               warn=(MIN_FRAMES, np.inf), note=f"rest ends at frame {rest_end}"),
        Metric("rest frames with object depth along the ray", sum(bool(r["ray_ratios"]) for r in rows), "frames"),
        Metric("camera-to-support, stereo (median)", float(np.median(support)) if len(support) else np.nan, "m"),
        Metric("camera-to-support spread, stereo", 100 * float(np.std(support)) if len(support) else np.inf, "cm",
               ok=(0, 2), warn=(0, 5), note="the head barely moves before the reach"),
        Metric("camera-to-support, bundle (median)", float(np.median([r["table_bundle"] for r in supported]))
               if supported else np.nan, "bundle units"),
        Metric("scale factor", factor, "x", ok=(0.67, 1.5), warn=(0.5, 2.0),
               note=f"from the {method}; multiplies every length in the bundle"),
        Metric("scale spread over rest frames (IQR / median)", 100 * spread, "%", ok=(0, 5), warn=(0, 10)),
        Metric("object-ray factor vs support-plane factor", 100 * (ray_factor / plane_factor - 1)
               if plane_factor and ray_factor else None, "%", ok=(-10, 10), warn=(-20, 20),
               note=f"object ray: {ray_factor:.3f}" if ray_factor else "no object-ray reading"),
        Metric("support plane vs bundle gravity (median)", float(np.nanmedian([r["tilt_deg"] for r in supported]))
               if supported else None, "deg", ok=(0, 8), warn=(0, 15)),
        Metric("hand scale after rescale", hand * factor if hand else None, "", ok=(0.85, 1.3), warn=(0.7, 1.5),
               note=f"before: {hand:.2f}; MANO hands are the team's ruler (BUNDLES.md)" if hand else "no valid hand"),
    ]
    if ctx.public:
        gt = gt_trajectory_factor(scorer(), ctx.number, data, slots)
        scale["gt_trajectory_factor"] = gt
        if gt:
            parts = ", ".join(f"{k} {v:.3f}" for k, v in gt["parts"].items())
            metrics += [
                Metric("scale factor vs GT trajectory scale", 100 * (factor / gt["factor"] - 1), "%", kind="gt",
                       ok=(-15, 15), warn=(-30, 30),
                       note=f"GT factor {gt['factor']:.3f} from {parts}; mesh origins differ from the scan's, "
                            "so rotating objects add a few percent"),
            ]
        else:
            metrics.append(Metric("scale factor vs GT trajectory scale", None, "%", kind="gt",
                                  note="no object moves more than 3 cm in the ground truth"))
        oracle = oracle_factor(scan_ratios)
        scale["oracle_scan_factor"] = oracle
        if oracle:
            metrics.append(Metric("scale implied by the scans' size", oracle, "x", kind="gt",
                                  note="1 / mesh-to-scan size ratio: mesh size, not world scale (information only)"))
    write_json(d / "scale.json", scale)
    return StageResult(metrics, {"scale": d / "scale.json", "factor": factor, "method": method, "bundle": bundle})


def oracle_factor(scan_ratios: dict) -> float | None:
    ratios = [r for r in scan_ratios.values() if r]
    return float(np.exp(-np.mean(np.log(ratios)))) if ratios else None


def gt_trajectory_factor(S, episode: int, data: dict, slots) -> dict | None:
    """The factor that would make the bundle's object motion the size of the ground truth's.

    Per object, the median over frames of |GT displacement from frame 0| / |bundle displacement|
    where the GT object has moved more than 3 cm; for two objects also the ratio of their frame-0
    distances. Both are rotation-free, so the bundle's world need not be aligned to the mocap's.
    """
    gt = S.ground_truth(episode)  # (frames, objects, 7), per-frame slot order = episodes.json order
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
