"""S3b track: FoundationPose on S3a's mesh with the symmetry-aware recovery rule (V3_BUILD_SPEC.md S3b).

pathb/drivers/fp_track.py runs inside the baseline's v2d_foundation_pose image (started by the
baseline's own run_in_container, path_b mounted read-only), on ego_cam_a's frames, P1d's
full-resolution stereo depth, the object's P1b masks and P4's intrinsics; S3a's mesh is used at its
metric size (no scale search). Registration frame: the S3a view with the largest object mask.
Recovery rule and particle frames: see the driver; defaults below.
Out per object: poses/%06d.json (object_to_camera), track_meta.json (per frame: iou, recovered,
particles, depth residual). Bars: frames with IoU >= 0.5: >= 80 % (PASS) / 60 % (WARN); the largest
rotation step between frames outside recovered ones < 30 deg (PASS) / 90 deg (WARN); no step
> 60 deg anywhere (FAIL otherwise).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from .. import baseline as B
from .. import symmetry
from ..geometry import read_transforms
from ..paths import PATH_B, RECON
from ..report import Metric, StageResult
from ..runner import write_json

# spec S3b: the driver runs in the FoundationPose container with --dev (the host's baseline modules at
# /workspace: the GHCR image predates the TensorRT backend), keeping the image's compiled extensions
# (v2d_foundation_pose/docker/_config.py DEV_PRESERVE_VOLUMES)
FP_DEV_PRESERVE = ["/workspace/v2d_foundation_pose/lib/FoundationPose/mycpp/build",
                   "/workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda"]
RECOVERY = {"iou_thresh": 0.3,  # the baseline's re-registration threshold
            "trusted_iou": 0.5,  # default: a pose this good becomes the last good pose
            "lambda": 2.0,  # default: per rad of rotation away from the last good pose
            "mu": 50.0,  # default: per m of median depth residual
            "iou_weight": 1.0,
            "particles": 20,  # default
            "particle_overlap": 0.4}  # default: hand share of the object mask that switches to particles


def hand_overlap(ctx, obj_dir, n: int) -> dict[str, float]:
    tracks = B.mask_tracks(ctx.root)
    hands = [p for t, p in tracks.items() if t.startswith("hand_")]
    out = {}
    for k in range(n):
        m = B.read_mask(obj_dir, k)
        if m is None or not m.any():
            continue
        h = np.zeros_like(m)
        for d in hands:
            hm = B.read_mask(d, k)
            if hm is not None:
                h |= hm
        out[str(k)] = float((m & h).sum() / m.sum())
    return out


def steps_deg(T: np.ndarray, have: np.ndarray) -> np.ndarray:
    """Rotation change between consecutive tracked frames, in degrees (nan where a frame is missing)."""
    out = np.full(len(T), np.nan)
    for k in range(1, len(T)):
        if have[k] and have[k - 1]:
            out[k] = np.degrees(Rotation.from_matrix(T[k - 1, :3, :3].T @ T[k, :3, :3]).magnitude())
    return out


def run(ctx) -> StageResult:
    d = ctx.dir("s3b_track")
    n = int(ctx.episode["num_frames"])
    tracks = B.mask_tracks(ctx.root)
    metrics, outputs = [], {}
    for i, name in enumerate(ctx.objects):
        obj_dir = tracks[f"obj_{i}"]
        mesh = ctx.root / f"s3a_mesh/{name}/mesh.obj"
        views = json.loads((ctx.root / f"s3a_mesh/{name}/mesh_views.json").read_text())["views"]
        areas = {k: float(m.mean()) for k in views if (m := B.read_mask(obj_dir, k)) is not None}
        ref = max(areas, key=areas.get)
        import trimesh
        G = symmetry.group(name, np.asarray(trimesh.load(mesh, force="mesh", process=False).vertices))
        out = d / name
        out.mkdir(parents=True, exist_ok=True)
        frames = [k for k in range(n) if (ctx.root / f"p1d_hands/depth/{k:06d}.png").is_file()
                  and (obj_dir / f"{k:06d}.png").is_file()]
        overlap = hand_overlap(ctx, obj_dir, n)
        write_json(out / "request.json", {"reference_frame": ref, "frames": frames, "hand_overlap": overlap,
                                          "symmetry": G, "recovery": RECOVERY})
        log = out / "fp.log"
        log.write_text("")
        B.call("v2d.docker.container:run_in_container", {
            "image": "v2d_foundation_pose", "module": "pathb.drivers.fp_track", "gpus": True,
            "inputs": {"request": str(out / "request.json"), "frames_dir": str(ctx.root / "p1_calibrate/cam_a"),
                       "depth_dir": str(ctx.root / "p1d_hands/depth"), "masks_dir": str(obj_dir),
                       "mesh_path": str(mesh), "intrinsics_path": str(ctx.root / "p1c_camera/intrinsics.json"),
                       "weights_dir": str(B.WEIGHTS / "foundation_pose")},
            "outputs": {"output_dir": str(out)},
            "dev": True, "modules_dir": str(RECON / "modules"),
            "env": {"PYTHONPATH": "/pathb_src"}, "extra_volumes": FP_DEV_PRESERVE + [f"{PATH_B}:/pathb_src:ro"]}, log)
        meta = json.loads((out / "track_meta.json").read_text())["frames"]
        T, have = read_transforms(out / "poses", n)
        steps = steps_deg(T, have)
        recovered = np.zeros(n, bool)
        seen = [r for r in meta if r["iou"] is not None]
        for r in meta:
            recovered[r["frame"]] = r["recovered"]
        calm = steps[~recovered & ~np.r_[False, recovered[:-1]]]
        metrics += [
            Metric(f"{name}: frames with mask IoU >= 0.5", float(np.mean([r["iou"] >= 0.5 for r in seen])) if seen else 0.0,
                   "share", ok=(0.8, 1), warn=(0.6, 1), note=f"{len(seen)} frames with a mask; reference frame {ref}"),
            Metric(f"{name}: largest rotation step outside recovered frames", float(np.nanmax(calm)) if np.isfinite(calm).any()
                   else None, "deg", ok=(0, 30), warn=(0, 90)),
            Metric(f"{name}: frames with a rotation step > 60 deg", int(np.nansum(steps > 60)), "frames", ok=(0, 0)),
            Metric(f"{name}: recovered frames", int(recovered.sum()), "frames",
                   note=f"symmetry group of {len(G)}; particles on {sum(r['particles'] for r in meta)} frames"),
            Metric(f"{name}: depth residual (median, frames with a mask)",
                   float(np.median([r["depth_residual_m"] for r in seen])) * 100 if seen else None, "cm"),
        ]
        outputs[name] = out / "poses"
    return StageResult(metrics, outputs)
