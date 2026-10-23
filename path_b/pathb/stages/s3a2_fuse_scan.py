"""S3a2 fuse scan (V3_BUILD_SPEC.md §11 A2.1): per object, the surface the head camera saw at rest, fused
from stereo depth alone (no generative model). Phase A2 variant: not in the default stage chain.

Frames: S4a's rest_before and rest_after windows, where the object's P1b mask covers >= 1 % of the
image and no hand mask (dilated 5 px at half resolution, as P1c) touches it. Each frame's P2 depth
(depth_a_half, ego_cam_a) is kept inside the object's mask only, the mask eroded by MASK_ERODE_PX so
the mixed depth at its edge stays out. Fused in the v3 world (P4's camera_to_world: P1c's camera in
the world every later stage uses) with pathb/scan.py's TSDF (voxel 4 mm, sdf_trunc 16 mm). The object
moved between the two windows, so the rest-after frames are fused where the object rested before: S3c's
anchors give the move (A_before A_after^-1), point-to-plane ICP of the rest-after patch onto the
rest-before one refines it. Out per object: partial_scan.ply (metres, world, the object at its frame-0
placement, visible surfaces only), fuse_report.json (per frame: window, mask share, used or why not).

Bars: frames fused >= 30 (PASS) / 10 (WARN); surface RMS against the stereo depth of 10 held-out rest
frames <= 5 mm (PASS) / 10 mm (WARN), from a second fusion without them (the scan itself fuses every
usable frame); public only: one-sided Chamfer partial -> scan <= 0.3 cm, the scan registered onto the
partial (scale fixed, best of a global start and 24 orientation starts).
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
ICP_MIN_FITNESS = 0.3  # rest-after onto rest-before: below this the anchors' move is kept
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


def rest_frames(ctx, k: int, ph: dict, K: np.ndarray, shape, c2w: np.ndarray) -> tuple[dict, list]:
    """Usable frames per window and the per-frame log."""
    half = ctx.root / "p2_depth/depth_a_half"
    usable, log = {"rest_before": [], "rest_after": []}, []
    for key in usable:
        if not ph.get(key):
            continue
        a, b = ph[key]
        for t in range(a, b + 1):
            row = {"frame": t, "window": key}
            mask, touched, share = frame_masks(ctx, k, t, shape)
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
    return usable, log


def rest_after_move(ctx, name: str, before: list, after: list) -> tuple[np.ndarray | None, dict]:
    """World -> world move taking the rest-after placement to the rest-before one."""
    f = ctx.root / f"s3c_fuse/{name}/anchors.json"
    anchors = json.loads(f.read_text()) if f.is_file() else {}
    if not (before and after and anchors.get("rest_before") and anchors.get("rest_after")):
        return None, {"used": False, "why": "no S3c anchor for both windows" if before and after else "one window"}
    M0 = np.asarray(anchors["rest_before"]["pose"]) @ np.linalg.inv(np.asarray(anchors["rest_after"]["pose"]))
    patch_b = S.largest_part(S.integrate(before))
    patch_a = S.largest_part(S.integrate([S.Frame(f.index, f.depth, f.K, M0 @ f.camera_to_world) for f in after]))
    if len(patch_a.triangles) == 0 or len(patch_b.triangles) == 0:
        return M0, {"used": True, "icp": None, "why": "a window fused nothing: anchors' move kept"}
    r = S.icp(S.mesh_points(patch_a, 10000), S.mesh_points(patch_b, 10000), max_dist=0.01)
    if r["fitness"] < ICP_MIN_FITNESS:
        return M0, {"used": True, "icp": {"fitness": r["fitness"], "rmse_m": r["rmse_m"]},
                    "why": "ICP overlap too small: anchors' move kept"}
    return r["T"] @ M0, {"used": True, "icp": {"fitness": r["fitness"], "rmse_m": r["rmse_m"],
                                                "correction_mm": float(np.linalg.norm(r["T"][:3, 3]) * 1000)}}


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
        ph = json.loads((ctx.root / f"s4a_phases/{name}/phases.json").read_text())
        usable, log = rest_frames(ctx, k, ph, K, shape, c2w)
        M, move = rest_after_move(ctx, name, usable["rest_before"], usable["rest_after"])
        frames = usable["rest_before"] + ([S.Frame(f.index, f.depth, f.K, M @ f.camera_to_world)
                                           for f in usable["rest_after"]] if M is not None else [])
        if not usable["rest_before"] and M is None:
            frames = usable["rest_after"]  # never at rest before: the rest-after placement
        partial = S.largest_part(S.integrate(frames))
        o3d.io.write_triangle_mesh(str(out / "partial_scan.ply"), partial)
        step = max(1, len(frames) // HOLD_OUT)
        held = frames[::step][:HOLD_OUT] if len(frames) >= 2 * HOLD_OUT else frames[::4]
        kept = [f for f in frames if f not in held]
        rms = S.depth_rms(S.largest_part(S.integrate(kept)), held) if kept and held else {"rms_m": None}
        v = np.asarray(partial.vertices)
        report = {"object": name, "frames": log, "fused": [f.index for f in frames], "rest_after_move": move,
                  "held_out": [f.index for f in held], "held_out_rms": rms, "vertices": len(v),
                  "extents_cm": (np.ptp(v, axis=0) * 100).round(1).tolist() if len(v) else None}
        metrics += [
            Metric(f"{name}: frames fused", len(frames), "", ok=BARS["frames"][0], warn=BARS["frames"][1],
                   note=f"rest before {len(usable['rest_before'])}, after {len(usable['rest_after'])}"
                        + ("" if M is not None or not usable["rest_after"] else " (rest after left out: "
                           + move["why"] + ")")),
            Metric(f"{name}: surface vs held-out stereo depth (RMS)",
                   rms["rms_m"] * 1000 if rms.get("rms_m") is not None else None, "mm",
                   ok=BARS["rms_mm"][0], warn=BARS["rms_mm"][1], note=f"{len(held)} held-out frames"),
            Metric(f"{name}: rest-after patch onto rest-before (ICP fitness)",
                   (move.get("icp") or {}).get("fitness"), "", note=move.get("why", "")),
        ]
        if ctx.public and len(v):
            ref = gt.scan_mesh(ctx.episode, name, ctx.files_read)
            if ref is not None:
                best = chamfer_to_scan(partial, *ref)
                report["chamfer_to_scan_cm"] = best["cm"]
                metrics.append(Metric(f"{name}: one-sided Chamfer partial -> scan", best["cm"], "cm", kind="gt",
                                      ok=BARS["chamfer_cm"], note="the partial should lie on the true surface"))
        write_json(out / "fuse_report.json", report)
        outputs[name] = out / "partial_scan.ply"
    return StageResult(metrics, outputs)
