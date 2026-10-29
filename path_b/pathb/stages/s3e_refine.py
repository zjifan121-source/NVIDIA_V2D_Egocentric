"""S3e refine (V4_BUILD_SPEC.md §14.4 R1): FoundationPose's refiner from S3d's settled track, kept frame by frame
only where it explains the frame better (GPU; the baseline's v2d_foundation_pose image, no new licence).

pathb/drivers/fp_refine.py runs in that image as S3b's tracker does (the baseline's run_in_container, path_b mounted
read-only), on ego_cam_a's frames, P1d's full-resolution stereo depth, the object's P1b mask, P1c's intrinsics and
S3a's canonical mesh at its metric size. Frames: those with the object's mask (not a frame B.lost_frames gives to
another object) and its depth. Start: S3d's pose (object_to_camera = camera_to_world^-1 @ object_to_world). The
driver keeps the refined pose where both its mask IoU and its median depth residual beat the start's and it moved
at most MAX_STEP_M / MAX_STEP_DEG; every other frame keeps S3d's pose. S4b reads this track when it exists.
Out per object: object_to_world.npz (N, 4, 4), refine_meta.json, poses/. Without nvidia-smi (the Mac) the stage
is skipped and S4b reads S3d's track as before. Public, one object: the refined track's playback AUC.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from .. import baseline as B
from .. import gt
from ..geometry import read_transforms
from ..paths import PATH_B, RECON
from ..report import Metric, StageResult
from ..runner import write_json
from .s3b_track import FP_DEV_PRESERVE

ITERATIONS = 5  # default: the baseline tracker's refiner passes per frame (track_one)
MAX_STEP_M, MAX_STEP_DEG = 0.03, 15.0  # default: the refiner polishes; a larger move is a re-registration, not kept


def requested(ctx, k: int, name: str, P: np.ndarray, c2w: np.ndarray) -> tuple[list[int], dict]:
    """Frames to refine (the object's own mask and depth there) and their start poses, object_to_camera."""
    tracks = B.mask_tracks(ctx.root)
    lost = B.lost_frames(str(ctx.root)).get(f"obj_{k}", frozenset())
    frames = [t for t in range(len(P)) if t not in lost and (tracks[f"obj_{k}"] / f"{t:06d}.png").is_file()
              and (ctx.root / f"p1d_hands/depth/{t:06d}.png").is_file() and np.isfinite(P[t]).all()]
    return frames, {str(t): (np.linalg.inv(c2w[t]) @ P[t]).tolist() for t in frames}


def run(ctx) -> StageResult:
    if not shutil.which("nvidia-smi"):
        return StageResult([], skip="no GPU here: S4b reads S3d's track")
    d = ctx.dir("s3e_refine")
    n = int(ctx.episode["num_frames"])
    c2w, have = read_transforms(ctx.root / "p4_export/camera_to_world", n)
    tracks = B.mask_tracks(ctx.root)
    metrics, outputs = [], {}
    for k, name in enumerate(ctx.objects):
        P = np.load(ctx.root / f"s3d_settle/{name}/object_to_world.npz")["object_to_world"].astype(np.float64)
        frames, poses = requested(ctx, k, name, P, c2w)
        frames = [t for t in frames if have[t]]
        out = d / name
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "request.json", {"frames": frames, "poses": poses, "iterations": ITERATIONS,
                                          "bounds": {"max_step_m": MAX_STEP_M, "max_step_deg": MAX_STEP_DEG},
                                          "symmetry": [np.eye(4).tolist()]})
        log = out / "fp.log"
        log.write_text("")
        B.call("v2d.docker.container:run_in_container", {
            "image": "v2d_foundation_pose", "module": "pathb.drivers.fp_refine", "gpus": True,
            "inputs": {"request": str(out / "request.json"), "frames_dir": str(ctx.root / "p1_calibrate/cam_a"),
                       "depth_dir": str(ctx.root / "p1d_hands/depth"), "masks_dir": str(tracks[f"obj_{k}"]),
                       "mesh_path": str(ctx.root / f"s3a_mesh/{name}/mesh.obj"),
                       "intrinsics_path": str(ctx.root / "p1c_camera/intrinsics.json"),
                       "weights_dir": str(B.WEIGHTS / "foundation_pose")},
            "outputs": {"output_dir": str(out)},
            "dev": True, "modules_dir": str(RECON / "modules"),
            "env": {"PYTHONPATH": "/pathb_src"}, "extra_volumes": FP_DEV_PRESERVE + [f"{PATH_B}:/pathb_src:ro"]}, log)
        meta = json.loads((out / "refine_meta.json").read_text())["frames"]
        Tc, got = read_transforms(out / "poses", n)
        R = P.copy()
        for t in np.nonzero(got)[0]:
            R[t] = c2w[t] @ Tc[t]
        np.savez(out / "object_to_world.npz", object_to_world=R)
        kept = [r for r in meta if r["kept"]]
        metrics += [
            Metric(f"{name}: frames refined (kept / asked)", f"{len(kept)} / {len(meta)}", "",
                   note=f"kept where mask IoU and depth residual both improve, moved <= {MAX_STEP_M * 100:.0f} cm / {MAX_STEP_DEG:.0f} deg"),
            Metric(f"{name}: mask IoU, median over asked frames (start -> kept)",
                   f"{np.median([r['start']['iou'] for r in meta]):.3f} -> "
                   f"{np.median([(r['refined'] if r['kept'] else r['start'])['iou'] for r in meta]):.3f}" if meta else None, ""),
            Metric(f"{name}: depth residual, median over asked frames (start -> kept)",
                   f"{np.median([r['start']['residual_m'] for r in meta]) * 100:.2f} -> "
                   f"{np.median([(r['refined'] if r['kept'] else r['start'])['residual_m'] for r in meta]) * 100:.2f}" if meta else None, "cm"),
        ]
        if ctx.public and len(ctx.objects) == 1:
            poses7 = np.concatenate([R[:, :3, 3], Rotation.from_matrix(R[:, :3, :3]).as_quat()], axis=1)[:, None]
            scores = gt.score(ctx.episode, poses7, {name: ctx.root / f"s3a_mesh/{name}/mesh.obj"}, ctx.files_read)
            metrics.append(Metric(f"{name}: playback AUC of the refined track", scores["AUC"], "", kind="gt",
                                  note=f"MPPE {scores['MPPE cm']:.2f} cm"))
        outputs[name] = out / "object_to_world.npz"
    return StageResult(metrics, outputs)
