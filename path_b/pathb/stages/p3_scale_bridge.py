"""P3 scale bridge: one metric scale factor for the episode's existing reconstruction bundle.

Until Path B tracks objects itself (Stage 2), its output is the team's HF bundle rescaled to stereo
metric scale. The ruler is the camera's height above the table while every object rests on it at
the start of the clip:
  stereo   distance from ego_cam_a to the table plane fitted in stereo depth (P2), in metres;
  bundle   the same distance in the bundle's own units: the lowest point of the object meshes,
           measured along the bundle's gravity "up" in its camera frame.
factor = median over the rest frames of stereo / bundle; stage 2's rescale_world multiplies every
length in the bundle by it. Only camera-frame quantities are compared, so the SLAM world's own
scale problems do not enter.
"""
from __future__ import annotations

import json

import numpy as np

from ..geometry import plane_in
from ..report import Metric, StageResult
from ..runner import write_json
from ..team import retarget_tools

REST_FRAMES = 30  # recordings open with both hands raised over the table (README §5.6)
REACH_CM = 10.0  # rest ends when a wrist comes this close (stage 2's rest hold uses the same)
MIN_FRAMES = 5
MAX_TILT_DEG = 20.0  # stereo plane vs bundle gravity beyond this: not the table


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


def bundle_distances(data: dict, bundle, frames) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: the bundle's camera-to-table distance (the lowest object rests on the table; a lid
    on a pot does not) and its gravity up in camera coordinates."""
    L = retarget_tools()
    slots = L.object_slots(data)
    vertices = [L.object_vertices(bundle, data, scale_key, mesh) for _, scale_key, mesh in slots]
    c2w = data["camera_to_world_transform"].astype(np.float64)
    distances, ups = [], []
    for t in frames:
        up = c2w[t, :3, :3].T @ np.array([0.0, 0.0, 1.0])
        w2c = np.linalg.inv(c2w[t])
        distances.append(max(table_distance(v, w2c @ data[pose][t].astype(np.float64), up)
                             for v, (pose, _, _) in zip(vertices, slots)))
        ups.append(up)
    return np.array(distances), np.array(ups)


def stereo_distances(planes: list[dict], alignment: dict, frames) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: ego_cam_a's distance to the stereo table plane, its normal in ego_cam_a, inlier share."""
    by_frame = {p["frame"]: p for p in planes}
    R1, a_from_left = np.array(alignment["R1"]), np.array(alignment["a_from_left"])
    distances, normals, inliers = [], [], []
    for t in frames:
        p = by_frame.get(t, {})
        if "d" not in p:
            distances.append(np.nan), normals.append([np.nan] * 3), inliers.append(0.0)
            continue
        n_left = R1.T @ np.array(p["normal"])  # rectified and unrectified left share the origin
        n_a, d_a = plane_in(a_from_left, n_left, p["d"])
        distances.append(d_a), normals.append(n_a), inliers.append(p["inlier_fraction"])
    return np.array(distances), np.array(normals), np.array(inliers)


def run(ctx) -> StageResult:
    L = retarget_tools()
    d = ctx.dir("p3_scale_bridge")
    p2 = ctx.root / "p2_depth"
    planes = json.loads((p2 / "planes.json").read_text())["planes"]
    alignment = json.loads((p2 / "alignment.json").read_text())

    bundle = source_bundle(ctx.number, ctx.objects, d)
    data = dict(np.load(bundle / "result.npz"))
    reach = [L.reach_frame(data, L.object_vertices(bundle, data, scale_key, mesh), REACH_CM, pose)
             for pose, scale_key, mesh in L.object_slots(data)]
    rest_end = min([REST_FRAMES] + [r for r in reach if r is not None])
    valid = data["object_is_valid"].astype(bool) & data["camera_is_valid"].astype(bool)
    frames = [t for t in range(min(rest_end, len(valid))) if valid[t]]
    if not frames:
        raise RuntimeError(f"no frame before {rest_end} has a valid camera and object in {bundle}")

    h_bundle, ups = bundle_distances(data, bundle, frames)
    h_stereo, normals, inliers = stereo_distances(planes, alignment, frames)
    tilt = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", np.nan_to_num(normals), ups), -1, 1)))
    usable = np.isfinite(h_stereo) & (h_bundle > 0) & (tilt <= MAX_TILT_DEG) & (inliers >= 0.15)
    ratios = h_stereo[usable] / h_bundle[usable]
    factor = float(np.median(ratios)) if usable.sum() >= MIN_FRAMES else 1.0
    spread = float(np.subtract(*np.percentile(ratios, [75, 25])) / factor) if usable.sum() >= MIN_FRAMES else np.inf

    hand = L.hand_scale(data)
    scale = {"factor": factor, "applied": usable.sum() >= MIN_FRAMES, "bundle": str(bundle),
             "rest_end_frame": int(rest_end), "frames": frames, "usable": usable,
             "stereo_m": h_stereo, "bundle_units": h_bundle, "tilt_deg": tilt}
    metrics = [
        Metric("rest frames used", int(usable.sum()), "frames", ok=(10, np.inf), warn=(MIN_FRAMES, np.inf),
               note=f"rest ends at frame {rest_end}; below {MIN_FRAMES} the bundle is left unscaled"),
        Metric("camera-to-table, stereo (median)", float(np.median(h_stereo[usable])) if usable.any() else np.nan, "m"),
        Metric("camera-to-table, bundle (median)", float(np.median(h_bundle[usable])) if usable.any() else np.nan,
               "bundle units"),
        Metric("scale factor (stereo / bundle)", factor, "x", ok=(0.67, 1.5), warn=(0.5, 2.0),
               note="multiplies every length in the bundle"),
        Metric("scale spread over rest frames (IQR / median)", 100 * spread, "%", ok=(0, 5), warn=(0, 10)),
        Metric("stereo table normal vs bundle gravity (median)", float(np.median(tilt)) if len(tilt) else np.nan,
               "deg", ok=(0, 8), warn=(0, 15)),
        Metric("hand scale after rescale", hand * factor if hand else None, "", ok=(0.85, 1.3), warn=(0.7, 1.5),
               note=f"before: {hand:.2f}; MANO hands are the team's ruler (BUNDLES.md)" if hand else "no valid hand"),
    ]
    if ctx.public:
        oracle = oracle_factor(L, bundle, data, ctx.objects)
        scale["oracle_scan_factor"] = oracle
        if oracle:
            metrics += [
                Metric("oracle scale from public scans", oracle, "x", kind="gt",
                       note="1 / mesh-to-scan size ratio (--rescale-by-scan), geometric mean over objects"),
                Metric("scale error vs oracle", 100 * (factor / oracle - 1), "%", kind="gt", ok=(-10, 10),
                       warn=(-20, 20)),
            ]
    write_json(d / "scale.json", scale)
    return StageResult(metrics, {"scale": d / "scale.json", "factor": factor, "bundle": bundle})


def oracle_factor(L, bundle, data: dict, names: list[str]) -> float | None:
    ratios = [L.scan_size_ratio(L.object_vertices(bundle, data, scale_key, mesh), name)
              for name, (_, scale_key, mesh) in zip(names, L.object_slots(data))]
    ratios = [r for r in ratios if r]
    return float(np.exp(-np.mean(np.log(ratios)))) if ratios else None
