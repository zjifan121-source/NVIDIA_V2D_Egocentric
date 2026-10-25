"""S3a fuse scan (V4_BUILD_SPEC.md S3a, "per-clip partial scan"; v3 A2.1): per object, the surface the
head camera saw at rest, fused from stereo depth alone (no generative model). Runs right after P4 (S1),
so its rest windows come from the 2-D masks, not from S3b's segments: rest-before = the frames before a
hand mask (dilated 5 px at half resolution, as P1c) first touches the object's P1b mask; rest-after =
the frames from SETTLE frames after it last touches it. Never touched: the whole clip is rest-before.
Frames used: inside a window, the object's mask covers >= 1 % of the image and no hand touches it. Each
frame's P2 depth (depth_a_half, ego_cam_a) is kept inside the object's mask only, the mask eroded by
MASK_ERODE_PX so the mixed depth at its edge stays out. Fused in the world (P4's camera_to_world) with
pathb/scan.py's TSDF (voxel 4 mm, sdf_trunc 16 mm). The object moved between the windows, so each is
its own partial: partial_scan.ply (rest-before: the object at its frame-0 placement, which
s3a2_register registers) and partial_scan_after.ply (rest-after); run.py canonical pools both, as
clips. Out per object: the two .ply (metres, world, visible surfaces only), fuse_report.json (per frame:
window, mask share, used or why not).

Bars: frames fused, both windows, >= 30 (PASS) / 10 (WARN); surface RMS against the stereo depth of 10
held-out frames <= 5 mm (PASS) / 10 mm (WARN), from a second fusion without them (the scan itself fuses
every usable frame), on partial_scan.ply (on the rest-after one when rest-before has no frame);
public only: one-sided Chamfer partial -> scan <= 0.3 cm, the scan registered onto the partial (scale
fixed, best of a global start and 24 orientation starts).
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from .. import baseline as B
from .. import gt
from .. import scan as S
from ..geometry import read_depth, read_transforms
from ..report import Metric, StageResult
from ..runner import write_json

MIN_AREA = 0.01  # share of the image (spec)
HAND_DILATE_PX = 5  # half resolution, as P1c
MASK_ERODE_PX = 2  # default: half resolution; stereo depth at mask edges mixes object and background
HOLD_OUT = 10  # spec
SETTLE = 5  # default: frames after the last touch before rest-after starts (S4a's reach dilation)
WINDOWS = {"rest_before": "partial_scan.ply", "rest_after": "partial_scan_after.ply"}
BARS = {"frames": ((30, np.inf), (10, np.inf)), "rms_mm": ((0, 5), (0, 10)), "chamfer_cm": (0, 0.3)}


def frame_masks(ctx, k: int, t: int, shape: tuple[int, int]) -> tuple[np.ndarray | None, bool, float]:
    """The object's mask at depth resolution (eroded), whether a hand touches it, and its area share
    before erosion (the spec's 1 % bar is on the mask itself)."""
    tracks = B.mask_tracks(ctx.root)
    m = B.read_mask(tracks[f"obj_{k}"], t)
    if m is None:
        return None, False, 0.0
    obj = cv2.resize(m.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    hand = np.zeros(shape, bool)
    for side in ("hand_L", "hand_R"):
        h = B.read_mask(tracks[side], t) if side in tracks else None
        if h is not None:
            hand |= cv2.resize(h.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_AREA) > 0
    kernel = np.ones((2 * HAND_DILATE_PX + 1,) * 2, np.uint8)
    touched = bool((cv2.dilate(hand.astype(np.uint8), kernel) > 0)[obj].any())
    eroded = cv2.erode(obj.astype(np.uint8), np.ones((2 * MASK_ERODE_PX + 1,) * 2, np.uint8)) > 0
    return eroded, touched, float(m.mean())


def touch_windows(touched: list[bool], moving: list[bool] | None = None) -> dict[str, list[int] | None]:
    """Inclusive rest windows: before the first frame a hand touches the object or it moves, and from
    SETTLE frames after the last such frame. `moving` (S3b's motion test on the object's own points)
    guards against hand masks P1b missed: ep13's right hand had no mask before frame 32, so the cup was
    fused while carried (an 18 cm smear 15-30 cm above the table)."""
    n = len(touched)
    busy = [bool(a) or bool(b) for a, b in zip(touched, moving or [False] * n)]
    hits = [t for t, x in enumerate(busy) if x]
    if not hits:
        return {"rest_before": [0, n - 1], "rest_after": None}
    a, b = hits[0], hits[-1] + 1 + SETTLE
    return {"rest_before": [0, a - 1] if a > 0 else None, "rest_after": [b, n - 1] if b < n else None}


def object_moving(ctx, k: int, n: int) -> list[bool]:
    """Per frame: the object's points moved >= segments.STATIC_M over segments.WINDOW frames (S3b's
    test); a frame without points counts as not moving (the touch test still applies there)."""
    from .. import segments as G
    from ..observe import View
    view, rng = View(ctx), np.random.default_rng(k)
    m = G.motion([view.points(k, t, 3000, rng) for t in range(n)])
    return [bool(np.isfinite(x) and x >= G.STATIC_M) for x in m]


def rest_frames(ctx, k: int, K: np.ndarray, shape, c2w: np.ndarray, n: int) -> tuple[dict, dict, list]:
    """The windows, the usable frames per window and the per-frame log."""
    half = ctx.root / "p2_depth/depth_a_half"
    masks = [frame_masks(ctx, k, t, shape) for t in range(n)]
    moving = object_moving(ctx, k, n)
    windows = touch_windows([touched for _, touched, _ in masks], moving)
    usable, log = {key: [] for key in WINDOWS}, []
    for key, w in windows.items():
        if not w:
            continue
        for t in range(w[0], w[1] + 1):
            row = {"frame": t, "window": key}
            mask, touched, share = masks[t]
            path = half / f"{t:06d}.png"
            if mask is None or not path.is_file():
                log.append(row | {"used": False, "why": "no mask" if mask is None else "no depth"})
                continue
            row["mask_share"] = round(share, 4)
            if share < MIN_AREA or touched:
                log.append(row | {"used": False, "why": "hand touches the object" if touched else "mask < 1 %"})
                continue
            depth = read_depth(path)
            depth[~mask] = 0.0
            row["depth_share_in_mask"] = round(float((depth[mask] > 0).mean()), 3)
            usable[key].append(S.Frame(t, depth, K, c2w[t]))
            log.append(row | {"used": True})
    return windows, usable, log


def held_out_rms(frames: list) -> tuple[list, dict]:
    """RMS of a fusion without HOLD_OUT spread frames against those frames' stereo depth."""
    step = max(1, len(frames) // HOLD_OUT)
    held = frames[::step][:HOLD_OUT] if len(frames) >= 2 * HOLD_OUT else frames[::4]
    kept = [f for f in frames if f not in held]
    return held, (S.depth_rms(S.largest_part(S.integrate(kept)), held) if kept and held else {"rms_m": None})


def chamfer_to_scan(partial: o3d.geometry.TriangleMesh, vertices: np.ndarray, faces: np.ndarray) -> dict:
    """One-sided partial -> scan (cm) with the scan registered onto the partial (scale fixed)."""
    scan_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    scan_mesh.compute_vertex_normals()
    target, source = S.mesh_points(partial, 5000), S.mesh_points(scan_mesh, 5000)  # the search; refined below
    centre = np.asarray(target.points).mean(axis=0) - np.asarray(source.points).mean(axis=0)
    starts = [S.global_start(source, target, seed=s) for s in range(2)]
    for R in Rotation.create_group("O").as_matrix():  # 24 orientations about the clouds' centres
        T = np.eye(4)
        c = np.asarray(source.points).mean(axis=0)
        T[:3, :3], T[:3, 3] = R, centre + c - R @ c
        starts.append(T)
    pts = np.asarray(target.points)
    best = None
    for T0 in starts:
        T = S.icp(source, target, T0, max_dist=0.02, iterations=30)["T"]
        cm = S.one_sided_cm(pts, vertices @ T[:3, :3].T + T[:3, 3], faces)
        if best is None or cm < best["cm"]:
            best = {"cm": cm, "T": T}
    target, source = S.mesh_points(partial, 20000), S.mesh_points(scan_mesh, 20000)
    T = S.icp(source, target, S.icp(source, target, best["T"], max_dist=0.02)["T"], max_dist=0.005)["T"]
    return {"cm": S.one_sided_cm(np.asarray(target.points), vertices @ T[:3, :3].T + T[:3, 3], faces), "T": T}


def run(ctx) -> StageResult:
    d = ctx.dir("s3a2_fuse_scan")
    n = int(ctx.episode["num_frames"])
    intr = json.loads((ctx.root / "p2_depth/depth_a_half/intrinsics.json").read_text())
    K, shape = np.asarray(intr["K"], dtype=np.float64), (intr["size"][1], intr["size"][0])
    c2w, have = read_transforms(ctx.root / "p4_export/camera_to_world", n)
    if not have.all():
        raise RuntimeError(f"P4 camera_to_world has {int(have.sum())} of {n} frames")
    metrics, outputs = [], {}
    for k, name in enumerate(ctx.objects):
        out = d / name
        out.mkdir(parents=True, exist_ok=True)
        windows, usable, log = rest_frames(ctx, k, K, shape, c2w, n)
        report = {"object": name, "windows": windows, "settle_frames": SETTLE, "frames": log, "partials": {},
              "window_rule": "before the first touch or motion; from SETTLE frames after the last"}
        main = "rest_before" if usable["rest_before"] else "rest_after"
        fused = len(usable["rest_before"]) + len(usable["rest_after"])
        metrics.append(Metric(f"{name}: frames fused", fused, "", ok=BARS["frames"][0], warn=BARS["frames"][1],
                              note=f"rest before {len(usable['rest_before'])} (window {windows['rest_before']}), "
                                   f"after {len(usable['rest_after'])} (window {windows['rest_after']})"))
        for key, file in WINDOWS.items():
            frames = usable[key]
            partial = S.largest_part(S.integrate(frames)) if frames else o3d.geometry.TriangleMesh()
            o3d.io.write_triangle_mesh(str(out / file), partial)
            held, rms = held_out_rms(frames)
            v = np.asarray(partial.vertices)
            row = {"file": file, "fused": [f.index for f in frames], "held_out": [f.index for f in held],
                   "held_out_rms": rms, "vertices": len(v),
                   "extents_cm": (np.ptp(v, axis=0) * 100).round(1).tolist() if len(v) else None}
            bar = key == main
            metrics.append(Metric(f"{name}: {key} surface vs held-out stereo depth (RMS)",
                                  rms["trimmed_rms_m"] * 1000 if rms.get("trimmed_rms_m") is not None else None, "mm",
                                  ok=BARS["rms_mm"][0] if bar else None, warn=BARS["rms_mm"][1] if bar else None,
                                  note=f"closest {rms.get('trimmed_share', 1):.0%} of the pixels; median "
                                       f"{(rms.get('median_abs_m') or 0) * 1000:.1f} mm, plain RMS "
                                       f"{(rms.get('rms_m') or 0) * 1000:.0f} mm; {len(held)} held-out frames; {file}"))
            if ctx.public and len(v):
                ref = gt.scan_mesh(ctx.episode, name, ctx.files_read)
                if ref is not None:
                    row["chamfer_to_scan_cm"] = chamfer_to_scan(partial, *ref)["cm"]
                    metrics.append(Metric(f"{name}: {key} one-sided Chamfer partial -> scan", row["chamfer_to_scan_cm"],
                                          "cm", kind="gt", ok=BARS["chamfer_cm"] if bar else None,
                                          note="the partial should lie on the true surface"))
            report["partials"][key] = row
            outputs[f"{name}/{key}"] = out / file
        write_json(out / "fuse_report.json", report)
    return StageResult(metrics, outputs)
