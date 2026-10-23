"""P1c camera: a metric camera path from DROID-SLAM with the stereo depth as its prior.

The baseline's v2d_droid_slam (run_video_to_slam, through its container; no change inside it) on
ego_cam_a's undistorted frames with input_depth_folder (depth into bundle adjustment) and
align_to_depth_folder (post-hoc scale), both pointing at depth_prior/: P2's depth_a_half with every
hand and object pixel (P1b masks, dilated) set to 0, which DROID reads as no prior, so moving hands
and objects do not pull the camera (V3_BUILD_SPEC.md P1c). The prior stays at half resolution:
DROID resizes it to its 384x512 input anyway.

Out: camera_to_world/%06d.json (Transform3d), trajectory.txt (TUM), camera_report.json.
Bars: median camera speed < 1.5 m/s; the P2 table plane's height in the world over the rest window
spreads < 1 cm (PASS) / 2 cm (WARN); DROID's own scale correction against stereo within 3 %.
The rest window here is the opening frames before a hand mask touches an object mask (capped at
30, pathb.phases.rest_window_2d): S4a's windows need the object track, which comes later.
"""
from __future__ import annotations

import json
import re

import cv2
import numpy as np

from .. import baseline as B
from ..calib import stereo_frame
from ..geometry import read_depth, read_transforms, write_depth
from ..phases import rest_window_2d
from ..paths import TRACK3_FPS
from ..report import Metric, StageResult
from ..runner import write_json
from .p3_scale_bridge import planes_in_cam_a

MASK_DILATE_PX = 5  # default: at half resolution
MAX_SPEED = 1.5  # m/s
REST_CAP = 30  # frames: P3's opening-frames rule


def depth_prior(ctx, d) -> tuple[int, float]:
    """depth_prior/ = depth_a_half without hand and object pixels; (frames, share of pixels masked)."""
    half = ctx.root / "p2_depth/depth_a_half"
    out = d / "depth_prior"
    out.mkdir(exist_ok=True)
    tracks = B.mask_tracks(ctx.root)
    kernel = np.ones((2 * MASK_DILATE_PX + 1, 2 * MASK_DILATE_PX + 1), np.uint8)
    files = sorted(half.glob("[0-9]*.png"))
    masked = []
    for f in files:
        k = int(f.stem)
        depth = read_depth(f)
        drop = np.zeros(depth.shape, np.uint8)
        for folder in tracks.values():
            m = B.read_mask(folder, k)
            if m is not None:
                drop |= cv2.resize(m.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_AREA) > 0
        drop = cv2.dilate(drop.astype(np.uint8), kernel) > 0
        masked.append(drop.mean())
        depth[drop] = 0.0
        write_depth(out / f.name, depth)
    return len(files), float(np.mean(masked)) if masked else 0.0


def plane_heights(ctx, c2w: np.ndarray, frames: list[int]) -> tuple[np.ndarray, float]:
    """Per frame, the world offset of the frame's largest stereo plane (sign-aligned normals), and
    the normals' spread in degrees."""
    p2 = ctx.root / "p2_depth"
    alignment = json.loads((p2 / "alignment.json").read_text())
    planes = {p["frame"]: p["planes"] for p in json.loads((p2 / "planes.json").read_text())["planes"]}
    heights, normals = [], []
    for t in frames:
        frame_planes = planes.get(stereo_frame(t, alignment, len(planes)), [])
        if not frame_planes:
            continue
        in_a = planes_in_cam_a(frame_planes, alignment)
        i = int(np.argmax([p["inlier_fraction"] for p in frame_planes]))
        n, dist = in_a[i]
        R, tc = c2w[t, :3, :3], c2w[t, :3, 3]
        nw = R @ n
        dw = dist - nw @ tc
        if normals and nw @ normals[0] < 0:
            nw, dw = -nw, -dw
        normals.append(nw)
        heights.append(dw)
    if not normals:
        return np.array([]), np.nan
    mean = np.mean(normals, axis=0)
    mean /= np.linalg.norm(mean)
    spread = float(np.degrees(np.max(np.arccos(np.clip(np.array(normals) @ mean, -1, 1)))))
    return np.array(heights), spread


def run(ctx) -> StageResult:
    d = ctx.dir("p1c_camera")
    calib = json.loads((ctx.root / "p1_calibrate/calib.json").read_text())
    frames = ctx.root / "p1_calibrate/cam_a"
    n = len(list(frames.glob("*.jpg")))
    prior_frames, masked_share = depth_prior(ctx, d)
    video = B.frames_to_mp4(frames, d / "cam_a.mp4")
    intrinsics = B.intrinsics_json(calib["cam_a"]["K"], calib["cam_a"]["size"], d / "intrinsics.json")
    log = d / "droid.log"
    log.write_text("")
    B.call("v2d.droid_slam.docker.run_video_to_slam:run_video_to_slam",
           {"video_path": video, "poses_folder": d / "camera_to_world", "weights_path": B.WEIGHTS / "droid_slam",
            "input_intrinsics_path": intrinsics, "input_depth_folder": d / "depth_prior",
            "align_to_depth_folder": d / "depth_prior", "trajectory_path": d / "trajectory.txt"}, log)
    c2w, have = read_transforms(d / "camera_to_world", n)
    scale = re.findall(r"applying global scale ([0-9.eE+-]+)", log.read_text())
    droid_scale = float(scale[-1]) if scale else 1.0

    tracks = B.mask_tracks(ctx.root)
    opening = range(min(n, REST_CAP))  # the rest window lies in the opening frames
    objects = [[B.read_mask(p, k) for k in opening] for t, p in tracks.items() if t.startswith("obj_")]
    hands = [[B.read_mask(p, k) for k in opening] for t, p in tracks.items() if t.startswith("hand_")]
    rest = rest_window_2d(objects, hands, cap=REST_CAP)
    heights, normal_spread = plane_heights(ctx, c2w, [t for t in range(rest[0], rest[1] + 1) if have[t]])
    speed = np.linalg.norm(np.diff(c2w[have][:, :3, 3], axis=0), axis=1) * TRACK3_FPS
    path_m = float(np.linalg.norm(np.diff(c2w[have][:, :3, 3], axis=0), axis=1).sum())
    write_json(d / "camera_report.json", {"rest_window": rest, "droid_scale_correction": droid_scale,
                                          "plane_heights_m": heights, "path_m": path_m,
                                          "masked_prior_share": masked_share})
    metrics = [
        Metric("frames with a camera pose minus manifest", int(have.sum()) - int(ctx.episode["num_frames"]), "frames",
               ok=(0, 0)),
        Metric("depth prior frames", prior_frames, "frames", note=f"{masked_share:.1%} of pixels masked (hands, objects)"),
        Metric("camera speed (median)", float(np.median(speed)) if len(speed) else np.nan, "m/s", ok=(0, MAX_SPEED)),
        Metric("camera speed (99th percentile)", float(np.percentile(speed, 99)) if len(speed) else np.nan, "m/s"),
        Metric("camera path length", path_m, "m"),
        Metric("table plane height spread over the rest window", 100 * float(np.std(heights)) if len(heights) >= 3
               else np.inf, "cm", ok=(0, 1), warn=(0, 2), note=f"rest window {rest} (2-D: before a hand touches an object)"),
        Metric("table plane normal spread over the rest window", normal_spread, "deg"),
        Metric("DROID scale correction against stereo", droid_scale, "x", ok=(0.97, 1.03),
               note="1.0 = the metric prior already made the path metric"),
    ]
    if ctx.public:  # the team bundle's camera path, for comparison only (numeric_qa)
        try:
            from ..team import retarget_tools
            team = np.load(retarget_tools().bundle_dir_for(ctx.number, ctx.objects[0]) / "result.npz")
            team_path = float(np.linalg.norm(np.diff(team["camera_to_world_transform"][:, :3, 3], axis=0), axis=1).sum())
            metrics.append(Metric("camera path vs the team bundle's", path_m / team_path if team_path else None, "x",
                                  note=f"team bundle {team_path:.2f} m (its own scale)"))
        except SystemExit:
            pass
    return StageResult(metrics, {"camera_to_world": d / "camera_to_world", "trajectory": d / "trajectory.txt",
                                 "report": d / "camera_report.json"})
