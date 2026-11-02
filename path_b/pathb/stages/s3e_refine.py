"""S3e refine (V4_BUILD_SPEC.md §14.4 R1): FoundationPose's refiner from S3d's settled track, kept frame by frame
only where it explains the frame better (GPU; the baseline's v2d_foundation_pose image, no new licence).

pathb/drivers/fp_refine.py runs in that image as S3b's tracker does (the baseline's run_in_container, path_b mounted
read-only), on ego_cam_a's frames, P1d's full-resolution stereo depth, the object's P1b mask, P1c's intrinsics and
S3a's canonical mesh at its metric size. Frames: those with the object's mask (not a frame B.lost_frames gives to
another object) and its depth. Start: S3d's pose (object_to_camera = camera_to_world^-1 @ object_to_world). The
driver refines every such frame once and records both poses, their mask IoU and median depth residual, and the
scorer's logits of the two.
Keep rule (Recon round P4, applied here to the driver's records, `keeps`): a refined pose replaces S3d's only on a
frame of a moving segment (S3b: grasp or unstable; S4b holds each static segment at its first frame's pose, so one
frame's polish would move the whole rest), where the mask IoU does not drop by more than IOU_DROP, the scorer
prefers it by SCORE_MARGIN, the depth residual falls by MIN_GAIN_M (when set), and it moved at most
MAX_STEP_M / MAX_STEP_DEG. The scorer, not the depth residual, since 2026-11-02: on moving frames the refiner raises
the mask IoU but not the depth residual (ep40: median +0.049 IoU, residual 0.77 mm worse), and with the scorer in
the depth bar's place public rose 0.8202 -> 0.8239 (BUILD_LOG). S4b reads this track when it exists.
Out per object: object_to_world.npz (N, 4, 4), refine_meta.json, poses/ (the driver's own pick), request.json.
Without nvidia-smi (the Mac) the stage re-applies the keep rule to a GPU pass's records already in s3e_refine/
(copied from the GPU machine), so the rule can be tuned without the GPU; a pass whose start poses are not this
S3d's track is stale and not used. With no records it is skipped and S4b reads S3d's track as before.
Public, one object: the refined track's playback AUC.
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
MOVING_ONLY = True  # default (P4): refined poses only on S3b's grasp / unstable frames
MIN_GAIN_M = None  # P4's first rule: 0.001 (the median masked-depth residual falls by 1 mm); None = not used (default)
IOU_DROP = 0.0  # default (P4): the mask IoU does not drop
SCORE_MARGIN = 0.0  # default (P4, 2026-11-02): the scorer's logit gain (refined - start) a kept pose needs; None = not used
SMOOTH_SIGMA = None  # frames: the kept corrections averaged over each moving segment (`smoothed`); None = off (default)
STALE_M = 1e-6  # a record's start pose this far from this S3d's (object_to_camera, any entry) is another track's


def requested(ctx, k: int, name: str, P: np.ndarray, c2w: np.ndarray) -> tuple[list[int], dict]:
    """Frames to refine (the object's own mask and depth there) and their start poses, object_to_camera."""
    tracks = B.mask_tracks(ctx.root)
    lost = B.lost_frames(str(ctx.root)).get(f"obj_{k}", frozenset())
    frames = [t for t in range(len(P)) if t not in lost and (tracks[f"obj_{k}"] / f"{t:06d}.png").is_file()
              and (ctx.root / f"p1d_hands/depth/{t:06d}.png").is_file() and np.isfinite(P[t]).all()]
    return frames, {str(t): (np.linalg.inv(c2w[t]) @ P[t]).tolist() for t in frames}


def keeps(row: dict) -> bool:
    """P4's rule on one driver record (module docstring)."""
    s, r = row["start"], row["refined"]
    if r["iou"] < s["iou"] - IOU_DROP:
        return False
    if MIN_GAIN_M is not None and s["residual_m"] - r["residual_m"] < MIN_GAIN_M:
        return False
    if row["step_m"] > MAX_STEP_M or row["step_deg"] > MAX_STEP_DEG:
        return False
    if SCORE_MARGIN is not None:
        lg = row.get("logits")
        if not lg or lg[0] is None or lg[1] is None or lg[1] - lg[0] < SCORE_MARGIN:
            return False
    return True


def moving_frames(segments: list[dict], n: int) -> np.ndarray:
    """Frames inside S3b's grasp / unstable segments."""
    moving = np.zeros(n, bool)
    for s in segments:
        if s["kind"] != "static":
            moving[s["start"]:s["end"] + 1] = True
    return moving


def regate(rows: list[dict], P: np.ndarray, c2w: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """S3d's track (object_to_world) with the refined pose on every frame the keep rule takes; those frames."""
    R, kept = P.copy(), []
    for row in rows:
        t = row["frame"]
        if (MOVING_ONLY and not moving[t]) or "refined_pose" not in row or not keeps(row):
            continue
        R[t] = c2w[t] @ np.asarray(row["refined_pose"], dtype=np.float64)
        kept.append(t)
    return R, kept


def smoothed(R: np.ndarray, P: np.ndarray, kept: list[int], segments: list[dict]) -> np.ndarray:
    """The kept frames' corrections in the object's own frame (P^-1 R; a grasp's offset error stays fixed there)
    averaged with Gaussian weights of SMOOTH_SIGMA frames inside each grasp / unstable segment, and applied to every
    frame of the segment within 2 sigma of a kept one, the kept frames included (rotation vectors and translations
    averaged: the corrections are small, within MAX_STEP). Other frames keep R."""
    out, took = R.copy(), set(kept)
    for s in segments:
        if s["kind"] == "static":
            continue
        a, b = s["start"], min(s["end"], len(P) - 1)
        ts = np.array([t for t in range(a, b + 1) if t in took], dtype=np.float64)
        if not len(ts):
            continue
        E = np.stack([np.linalg.inv(P[int(t)]) @ R[int(t)] for t in ts])
        rotvec, shift = Rotation.from_matrix(E[:, :3, :3]).as_rotvec(), E[:, :3, 3]
        for u in range(a, b + 1):
            if np.abs(ts - u).min() > 2 * SMOOTH_SIGMA or not np.isfinite(P[u]).all():
                continue
            w = np.exp(-0.5 * ((ts - u) / SMOOTH_SIGMA) ** 2)
            w /= w.sum()
            D = np.eye(4)
            D[:3, :3], D[:3, 3] = Rotation.from_rotvec(w @ rotvec).as_matrix(), w @ shift
            out[u] = P[u] @ D
    return out


def stale(request: dict, P: np.ndarray, c2w: np.ndarray) -> bool:
    """Whether a GPU pass started from another track than this S3d's."""
    for t, pose in request["poses"].items():
        t = int(t)
        if t >= len(P) or not np.isfinite(P[t]).all() or \
                np.abs(np.asarray(pose) - np.linalg.inv(c2w[t]) @ P[t]).max() > STALE_M:
            return True
    return False


def recorded(ctx) -> bool:
    """Whether s3e_refine/ holds a GPU pass's records (refined poses) for any object."""
    for name in ctx.objects:
        f = ctx.root / f"s3e_refine/{name}/refine_meta.json"
        if f.is_file() and any("refined_pose" in r for r in json.loads(f.read_text())["frames"]):
            return True
    return False


def run(ctx) -> StageResult:
    gpu = shutil.which("nvidia-smi") is not None
    if not gpu and not recorded(ctx):
        return StageResult([], skip="no GPU here: S4b reads S3d's track")
    d = ctx.dir("s3e_refine")
    n = int(ctx.episode["num_frames"])
    c2w, have = read_transforms(ctx.root / "p4_export/camera_to_world", n)
    tracks = B.mask_tracks(ctx.root)
    metrics, outputs = [], {}
    for k, name in enumerate(ctx.objects):
        P = np.load(ctx.root / f"s3d_settle/{name}/object_to_world.npz")["object_to_world"].astype(np.float64)
        out = d / name
        out.mkdir(parents=True, exist_ok=True)
        if gpu:
            frames, poses = requested(ctx, k, name, P, c2w)
            frames = [t for t in frames if have[t]]
            write_json(out / "request.json", {"frames": frames, "poses": {str(t): poses[str(t)] for t in frames},
                                              "iterations": ITERATIONS,
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
        meta_file, request_file = out / "refine_meta.json", out / "request.json"
        if not gpu and (not meta_file.is_file() or not request_file.is_file()
                        or stale(json.loads(request_file.read_text()), P, c2w)):
            (out / "object_to_world.npz").unlink(missing_ok=True)  # S4b then reads S3d's track for this object
            metrics.append(Metric(f"{name}: GPU pass recorded for this S3d track", 0, "", ok=(1, 1),
                                  note="no records, or records of another track (stale): S3d's track kept"))
            continue
        meta = json.loads(meta_file.read_text())
        rows = meta["frames"]
        segments = json.loads((ctx.root / f"s3b_segments/{name}/segments.json").read_text())
        moving = moving_frames(segments, n)
        R, kept = regate(rows, P, c2w, moving)
        if SMOOTH_SIGMA:
            R = smoothed(R, P, kept, segments)
        np.savez(out / "object_to_world.npz", object_to_world=R)
        took = set(kept)
        pick = [r["refined"] if r["frame"] in took else r["start"] for r in rows]
        asked_moving = sum(1 for r in rows if moving[r["frame"]])
        scored = [r["logits"][1] - r["logits"][0] for r in rows if r.get("logits") and None not in r["logits"]]
        metrics += [
            Metric(f"{name}: frames refined (kept / asked, asked on moving segments)",
                   f"{len(kept)} / {len(rows)} ({asked_moving} moving)", "",
                   note=f"P4 rule: {'moving segments only; ' if MOVING_ONLY else ''}IoU drop <= {IOU_DROP}, scorer "
                        f"margin {SCORE_MARGIN}, depth residual "
                        f"{'not used' if MIN_GAIN_M is None else f'-{MIN_GAIN_M * 1000:.0f} mm or more'}, "
                        f"moved <= {MAX_STEP_M * 100:.0f} cm / {MAX_STEP_DEG:.0f} deg"),
            Metric(f"{name}: mask IoU, median over asked frames (start -> chosen)",
                   f"{np.median([r['start']['iou'] for r in rows]):.3f} -> "
                   f"{np.median([p['iou'] for p in pick]):.3f}" if rows else None, ""),
            Metric(f"{name}: depth residual, median over asked frames (start -> chosen)",
                   f"{np.median([r['start']['residual_m'] for r in rows]) * 100:.2f} -> "
                   f"{np.median([p['residual_m'] for p in pick]) * 100:.2f}" if rows else None, "cm"),
            Metric(f"{name}: scorer logit gain (refined - start), median over scored frames",
                   round(float(np.median(scored)), 3) if scored else None, "",
                   note=f"{len(scored)} of {len(rows)} frames scored" + (f"; scorer error: {meta['scorer_error']}"
                                                                         if meta.get("scorer_error") else "")),
        ]
        if ctx.public and len(ctx.objects) == 1:
            poses7 = np.concatenate([R[:, :3, 3], Rotation.from_matrix(R[:, :3, :3]).as_quat()], axis=1)[:, None]
            scores = gt.score(ctx.episode, poses7, {name: ctx.root / f"s3a_mesh/{name}/mesh.obj"}, ctx.files_read)
            metrics.append(Metric(f"{name}: playback AUC of the refined track", scores["AUC"], "", kind="gt",
                                  note=f"MPPE {scores['MPPE cm']:.2f} cm"))
        outputs[name] = out / "object_to_world.npz"
    return StageResult(metrics, outputs)
