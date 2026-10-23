"""S3c fuse: the object's world track, smoothed and pinned at rest (V3_BUILD_SPEC.md S3c).

S3b's object_to_camera poses go into the v3 world through P4's camera. The phases come from
pathb/phases.py (the S4a function) with P1d's wrists and S3a's mesh. Rest anchors: per rest window,
the median translation and chordal-mean rotation (pathb/fuse.rest_anchor, symmetric equivalents
aligned first) of the frames whose stereo depth residual is <= 2 cm. Fusion: pathb/fuse.smooth
(the baseline ESKF/RTS on world poses; IoU-scaled noise along each camera's axis; anchor frames
2 mm / 0.005 rad; recovered frames 3x; a Point2Pose stream if s3b_track/<object>/poses_p2p/ exists;
the object's symmetry group). The P4 bundle is then completed: object track and S3a's mesh, by the
same baseline writer.
Out per object: object_to_world.npz, anchors.json, fuse_report.json, object_to_camera/ (the
writer's input). Bars: rest-before spread < 5 mm and 1 deg (PASS); object speed < 1.5 m/s on every
frame; public: the playback score is the waterfall's `fused` column.
"""
from __future__ import annotations

import json

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .. import fuse, symmetry
from .. import phases as P
from ..bundle import p4_bundle, write_v3_bundle
from ..geometry import read_transforms, write_transform
from ..paths import TRACK3_FPS
from ..report import Metric, StageResult
from ..runner import write_json
from .p4_export import world_extra

ANCHOR_RESIDUAL = 0.02  # m: frames with a larger stereo depth residual do not shape an anchor
MAX_SPEED = 1.5  # m/s


def window_frames(window, have, residual) -> list[int]:
    if not window:
        return []
    return [t for t in range(window[0], window[1] + 1) if have[t] and residual.get(t, 1.0) <= ANCHOR_RESIDUAL]


def spread(T: np.ndarray, anchor: np.ndarray) -> tuple[float, float]:
    """Largest distance (m) and rotation (deg) of poses T from the anchor."""
    d = np.linalg.norm(T[:, :3, 3] - anchor[:3, 3], axis=1).max()
    r = max(np.degrees(Rotation.from_matrix(anchor[:3, :3].T @ t[:3, :3]).magnitude()) for t in T)
    return float(d), float(r)


def run(ctx) -> StageResult:
    d = ctx.dir("s3c_fuse")
    n = int(ctx.episode["num_frames"])
    c2w, cam_ok = read_transforms(ctx.root / "p4_export/camera_to_world", n)
    W = np.array(json.loads((ctx.root / "p4_export/world.json").read_text())["world_from_p1c"])
    metrics, outputs = [], {}
    for name in ctx.objects:
        out = d / name
        out.mkdir(parents=True, exist_ok=True)
        T_cam, have = read_transforms(ctx.root / f"s3b_track/{name}/poses", n)
        have &= cam_ok
        meta = {r["frame"]: r for r in json.loads((ctx.root / f"s3b_track/{name}/track_meta.json").read_text())["frames"]}
        T = c2w @ T_cam
        mesh = trimesh.load(ctx.root / f"s3a_mesh/{name}/mesh.obj", force="mesh", process=False)
        vertices = np.asarray(mesh.vertices)
        G = symmetry.group(name, vertices)
        bundle = np.load(p4_bundle(ctx, name) / "result.npz")
        wrists = [(bundle[f"hand_{s}_wrist_to_world_transform"][:, :3, 3], bundle[f"hand_{s}_is_valid"] & have)
                  for s in ("left", "right")]
        ph = P.phases(P.near_frames(T, vertices, wrists), T)
        residual = {k: r["depth_residual_m"] for k, r in meta.items() if r.get("depth_residual_m") is not None}
        anchors, report_anchors = [], {}
        for key in ("rest_before", "rest_after"):
            frames = window_frames(ph[key], have, residual)
            if frames:
                A = fuse.rest_anchor(T[frames], G)
                anchors.append((ph[key], A))
                report_anchors[key] = {"window": ph[key], "frames_used": len(frames), "pose": A}
            else:
                report_anchors[key] = None
        iou = np.array([meta[k]["iou"] if k in meta and meta[k]["iou"] is not None else 0.0 for k in range(n)])
        recovered = np.array([bool(meta.get(k, {}).get("recovered")) for k in range(n)])
        p2p_dir = ctx.root / f"s3b_track/{name}/poses_p2p"
        second = second_ok = None
        if p2p_dir.is_dir():
            S_cam, second_ok = read_transforms(p2p_dir, n)
            second, second_ok = c2w @ S_cam, second_ok & cam_ok
        fused, info = fuse.smooth(T, valid=have & (iou > 0), iou=iou, recovered=recovered, cam_rot=c2w[:, :3, :3],
                                  anchors=anchors, second=second, second_valid=second_ok, G=G)
        np.savez(out / "object_to_world.npz", object_to_world=fused)
        write_json(out / "anchors.json", report_anchors)
        cams = out / "object_to_camera"
        cams.mkdir(exist_ok=True)
        obj_cam = np.linalg.inv(c2w) @ fused
        for t in range(n):
            if cam_ok[t]:
                write_transform(cams / f"{t:06d}.json", obj_cam[t])
        write_v3_bundle(ctx, p4_bundle(ctx, name), out / "writer.log", mesh_path=ctx.root / f"s3a_mesh/{name}/mesh.obj",
                        object_poses_dir=cams, extra=world_extra(W))
        speed = np.linalg.norm(np.diff(fused[:, :3, 3], axis=0), axis=1) * TRACK3_FPS
        raw_spread = fused_spread = (np.nan, np.nan)
        if report_anchors["rest_before"]:
            rb = report_anchors["rest_before"]["window"]
            A = report_anchors["rest_before"]["pose"]
            fused_spread = spread(fused[rb[0]:rb[1] + 1], A)
            raw = [t for t in range(rb[0], rb[1] + 1) if have[t]]
            raw_spread = spread(T[raw], A) if raw else (np.nan, np.nan)
        write_json(out / "fuse_report.json", {"phases": ph, "fusion": info, "anchors": report_anchors,
                                              "rest_before_spread_fused": fused_spread, "rest_before_spread_raw": raw_spread,
                                              "max_speed_m_s": float(speed.max())})
        metrics += [
            Metric(f"{name}: rest-before spread (fused)", fused_spread[0] * 1000, "mm", ok=(0, 5),
                   note=f"raw S3b track: {raw_spread[0] * 1000:.1f} mm, {raw_spread[1]:.1f} deg"),
            Metric(f"{name}: rest-before rotation spread (fused)", fused_spread[1], "deg", ok=(0, 1)),
            Metric(f"{name}: object speed (max)", float(speed.max()), "m/s", ok=(0, MAX_SPEED)),
            Metric(f"{name}: anchor frames", info["anchor_frames"], "frames",
                   note=", ".join(f"{k} {v['frames_used']} of {v['window']}" if v else f"{k} none"
                                  for k, v in report_anchors.items())),
            Metric(f"{name}: symmetric switches resolved", info["symmetric_switches"], ""),
        ]
        outputs[name] = out / "object_to_world.npz"
    return StageResult(metrics, outputs)
