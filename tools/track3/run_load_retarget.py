"""Stages 2-3 for one reconstruction bundle: check it, load it, retarget it to the Sharpa hands.

Mirrors NVIDIA's retarget stages (baseline/tools/e2e/stages.py) and their onboarding recipe
(baseline/.claude/skills/onboard_ego_clip/), run natively instead of in containers:

  check bundle -> load (MANO FK, ManoSharpaData) -> retarget (IK) -> object URDF
  -> support surface -> QA video -> check output -> report.json

Track 3 settings: clip fps 20.254 (V2D_EGO_FPS), table height 1.0 m (--ground_z_offset), and
--no_ground_align because bundles are gravity-aligned upstream. Before load, the bundle is cleaned
(cleaned_bundle: hand and object jumps, rest hold; optional rescale or still scene). --merge puts
every object of a multi-object episode into one scene (merged_bundle). Steps whose outputs exist
are skipped (use --force to redo).

Run with the load+retarget environment (README "Stages 2-3"), from anywhere:
    $RETARGET_VENV/bin/python tools/track3/run_load_retarget.py --episode 41 --object white_pot
    $RETARGET_VENV/bin/python tools/track3/run_load_retarget.py --bundle <dir with result.npz> \
        --sequence-id tissue_box_test --out <dir>
    $RETARGET_VENV/bin/python tools/track3/run_load_retarget.py --episode 33 --merge --rescale-by-scan
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
RECON = REPO / "baseline/reconstruction"
RG = REPO / "baseline/robotic_grounding"
OUTPUTS = RECON / "data/outputs"
MANO_DIR = RECON / "data/weights/hand"  # holds models/MANO_{LEFT,RIGHT}.pkl
SCANS = REPO / "assets/data/track_3/public/mesh"
MANIFEST = Path(__file__).with_name("episodes.json")
TRACK3_FPS = 20.25411910142744  # assets/data/track_3/public/meta/info.json
ROBOT = "sharpa_wave"


def bundle_dir_for(episode: int, obj: str) -> Path:
    """Bundle location under data/outputs/hub (bundles.py layout); also accepts the raw run dir."""
    run = OUTPUTS / "hub" / f"episode_{episode:06d}" / obj
    for candidate in (run / "result_slam_gravity_aligned", run):
        if (candidate / "result.npz").is_file():
            return candidate
    raise SystemExit(f"no result.npz under {run} (download it first, or pass --bundle)")


# ---------------------------------------------------------------- bundle checks
def oriented_extents(vertices: np.ndarray) -> list[float]:
    """Extents of the bbox-volume-minimizing box, longest first (raw bbox lies for tilted meshes)."""
    _, extents = trimesh.bounds.oriented_bounds(vertices)
    return sorted((float(e) for e in extents), reverse=True)


def footprint(world_vertices: np.ndarray, band: float = 0.005) -> list[float]:
    """Size of the patch within `band` of the lowest point: broad = resting on a face, thin = an edge."""
    low = world_vertices[world_vertices[:, 2] < world_vertices[:, 2].min() + band][:, :2]
    if len(low) < 3:
        return [0.0, 0.0]
    centred = low - low.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    return sorted((float(np.ptp(centred @ vectors[:, i])) for i in range(2)), reverse=True)


def first_still_run(T: np.ndarray, min_frames: int = 5) -> int | None:
    """Start of the first run of >= min_frames frames moving < 1 mm and < 0.01 rad per frame.

    Same test as NVIDIA's support detector (support_recon._frames_where_object_still): with no
    such run near frame 0, no table is built under the object and it falls at reset.
    """
    from scipy.spatial.transform import Rotation
    step = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)
    turn = (Rotation.from_matrix(T[1:, :3, :3]) * Rotation.from_matrix(T[:-1, :3, :3]).inv()).magnitude()
    still = (step < 0.001) & (turn < 0.01)  # still[i]: frame i -> i+1
    run = 0
    for i, s in enumerate(still):
        run = run + 1 if s else 0
        if run == min_frames:
            return i - min_frames + 1
    return None


def check_bundle(bundle: Path, object_name: str | None) -> dict:
    data = dict(np.load(bundle / "result.npz"))
    frames = data["camera_to_world_transform"].shape[0]
    nan_keys = [k for k, v in data.items() if v.dtype.kind == "f" and not np.isfinite(v).all()]
    valid = {role: float(np.mean(data[f"{role}_is_valid"])) for role in
             ("camera", "object", "hand_left", "hand_right") if f"{role}_is_valid" in data}
    hand_scale = {side: float(np.mean(data[f"hand_{side}_scale"][data[f"hand_{side}_is_valid"].astype(bool)]))
                  for side in ("left", "right") if f"hand_{side}_scale" in data}
    mesh = trimesh.load(bundle / "mesh.obj", force="mesh", process=False)
    vertices = np.asarray(mesh.vertices) * float(data.get("object_scale", 1.0))
    obj_T = data["object_to_world_transform"]
    world0 = vertices @ obj_T[0, :3, :3].T + obj_T[0, :3, 3]
    drift = np.linalg.norm(obj_T[: min(30, frames), :3, 3] - obj_T[0, :3, 3], axis=1)
    step_cm = np.linalg.norm(np.diff(obj_T[:, :3, 3], axis=0), axis=1) * 100
    report = {
        "frames": int(frames),
        "nan_arrays": nan_keys,
        "valid_fraction": valid,
        "hand_scale_mean": hand_scale,
        "gravity_alignment_applied": bool(data.get("gravity_alignment_applied", False)),
        "object_drift_first_30_frames_cm": round(float(drift.max()) * 100, 2),
        "object_step_median_cm": round(float(np.median(step_cm)), 2),
        "first_still_frame": first_still_run(obj_T),
        "mesh_vertices": int(len(vertices)),
        "mesh_extents_cm": [round(e * 100, 1) for e in oriented_extents(vertices)],
        "rest_footprint_cm": [round(e * 100, 1) for e in footprint(world0)],
    }
    # Same preference as the kit's v2dlb.vertices.mesh_path_for.
    scan = next((p for p in (SCANS / f"{object_name}/{object_name}_visual.glb", SCANS / f"{object_name}/{object_name}.glb")
                 if p.is_file()), None) if object_name else None
    if scan is not None:
        scan_mesh = trimesh.load(scan, force="mesh", process=False)
        scan_extents = oriented_extents(np.asarray(scan_mesh.vertices))
        report["scan_extents_cm"] = [round(e * 100, 1) for e in scan_extents]
        report["mesh_to_scan_size_ratio"] = round(float(np.mean(np.array(oriented_extents(vertices)) /
                                                                np.array(scan_extents))), 3)
    problems = []
    if nan_keys:
        problems.append(f"NaN in {nan_keys}: rebuild the bundle upstream")
    problems += [f"{role} valid on only {v:.0%} of frames" for role, v in valid.items() if v < 0.95]
    problems += [f"{side} hand scale {s:.2f}: world may be off by that factor" for side, s in hand_scale.items()
                 if not 0.85 <= s <= 1.3]
    if not report["gravity_alignment_applied"]:
        problems.append("bundle is not gravity-aligned; --no_ground_align keeps its tilt")
    if report["object_drift_first_30_frames_cm"] > 3:
        problems.append("object moves in the first 30 frames (rest contract expects mm-cm)")
    if report["first_still_frame"] is None or report["first_still_frame"] > 30:
        problems.append("object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 "
                        "frames: no table is built under it, so it falls in simulation")
    long, short = report["rest_footprint_cm"]
    if long > 3 and short < 0.25 * long:
        problems.append(f"rest footprint {long}x{short} cm is a strip: object may sit on an edge (level it)")
    if "mesh_to_scan_size_ratio" in report and not 0.8 <= report["mesh_to_scan_size_ratio"] <= 1.25:
        problems.append(f"mesh is {report['mesh_to_scan_size_ratio']}x the scanned object's size")
    report["warnings"] = problems
    return report


# ---------------------------------------------------------------- objects in a bundle
def object_slots(data: dict) -> list[tuple[str, str, str]]:
    """(pose key, scale key, mesh file) per object. A merged bundle (merged_bundle) lists its
    objects in object_names and stores object i >= 1 as object_<i>_*, mesh_<i>.obj; the loader
    reads the same layout (ego_recon_loader.py)."""
    n = len(data["object_names"]) if "object_names" in data else 1
    return [("object_to_world_transform", "object_scale", "mesh.obj")] + [
        (f"object_{i}_to_world_transform", f"object_{i}_scale", f"mesh_{i}.obj") for i in range(1, n)]


def object_vertices(bundle: Path, data: dict, scale_key: str, mesh_file: str) -> np.ndarray:
    mesh = trimesh.load(bundle / mesh_file, force="mesh", process=False)
    return np.asarray(mesh.vertices) * float(data.get(scale_key, 1.0))


def merged_bundle(episode: int, dest: Path) -> tuple[Path, dict]:
    """One bundle holding every object of a multi-object episode, in the scorer's object order.

    Each object is a separate reconstruction run with its own SLAM world; on ep33 the two runs'
    camera paths differ 4x in length, so worlds cannot be matched through the cameras. Hands and
    depth do agree in camera coordinates across runs (wrists within ~2 cm, depth ratio 1.000), so
    the run with the shortest (steadiest) camera path supplies the world, camera and hands, and
    every other object enters through its camera-frame pose: world_A(t) = camera_A(t) @ obj_cam(t).
    """
    episode_info = {e["episode"]: e for e in json.loads(MANIFEST.read_text())["episodes"]}[episode]
    names = [o["name"] for o in episode_info["objects"]]
    runs = {name: bundle_dir_for(episode, name) for name in names}
    datas = {name: dict(np.load(runs[name] / "result.npz")) for name in names}
    path_len = {name: float(np.linalg.norm(np.diff(d["camera_to_world_transform"][:, :3, 3], axis=0), axis=1).sum())
                for name, d in datas.items()}
    world = min(path_len, key=path_len.get)
    data = dict(datas[world])
    c2w = data["camera_to_world_transform"].astype(np.float64)
    dest.mkdir(parents=True, exist_ok=True)
    for i, (name, (pose_key, scale_key, mesh_file)) in enumerate(zip(names, object_slots({"object_names": names}))):
        src = datas[name]
        T = (src["object_to_world_transform"].astype(np.float64) if name == world
             else c2w @ src["object_to_camera_transform"].astype(np.float64))
        data[pose_key] = T.astype(np.float32)
        data[scale_key] = np.asarray(float(src.get("object_scale", 1.0)), dtype=np.float32)
        if i == 0:
            data["object_to_camera_transform"] = (np.linalg.inv(c2w) @ T).astype(np.float32)
            data["object_is_valid"] = src["object_is_valid"]
        # each object's material and texture get their own names in the shared directory
        stem = Path(mesh_file).stem
        mtl = (runs[name] / "material.mtl").read_text()
        for line in mtl.splitlines():
            if line.startswith("map_"):
                tex = line.split()[-1]
                shutil.copy(runs[name] / tex, dest / f"{stem}_{tex}")
                mtl = mtl.replace(tex, f"{stem}_{tex}")
        (dest / f"{stem}.mtl").write_text(mtl)
        (dest / mesh_file).write_text((runs[name] / "mesh.obj").read_text().replace("mtllib material.mtl", f"mtllib {stem}.mtl"))
    data["object_names"] = np.array(names)
    np.savez(dest / "result.npz", **data)
    shutil.copy(runs[world] / "manifest.json", dest / "manifest.json")
    return dest, {"objects": names, "world_from": world, "camera_path_m": {k: round(v, 2) for k, v in path_len.items()}}


# ---------------------------------------------------------------- rest hold
def reach_frame(data: dict, vertices: np.ndarray, cm: float = 10.0,
                pose_key: str = "object_to_world_transform") -> int | None:
    """First frame a valid wrist is within `cm` of the object surface (object frame), or None."""
    from scipy.spatial import cKDTree
    T = data[pose_key].astype(float)
    tree = cKDTree(vertices)
    near = np.zeros(len(T), dtype=bool)
    for side in ("left", "right"):
        wrist = data[f"hand_{side}_wrist_to_world_transform"][:, :3, 3].astype(float)
        local = np.einsum("tji,tj->ti", T[:, :3, :3], wrist - T[:, :3, 3])
        near |= (tree.query(local)[0] < cm / 100) & data[f"hand_{side}_is_valid"].astype(bool)
    hits = np.nonzero(near)[0]
    return int(hits[0]) if len(hits) else None


def hold_until(T: np.ndarray, frame: int, blend: int = 5) -> np.ndarray:
    """Object at rest before `frame` (median position, mean rotation), easing into the track after it."""
    from scipy.spatial.transform import Rotation
    out = T.copy()
    rest_pos = np.median(T[:frame, :3, 3], axis=0)
    rest_rot = Rotation.from_matrix(T[:frame, :3, :3]).mean()
    out[:frame, :3, 3] = rest_pos
    out[:frame, :3, :3] = rest_rot.as_matrix()
    for j, f in enumerate(range(frame, min(frame + blend, len(T)))):
        w = (j + 1) / (blend + 1)
        delta = (rest_rot.inv() * Rotation.from_matrix(T[f, :3, :3])).as_rotvec()
        out[f, :3, 3] = (1 - w) * rest_pos + w * T[f, :3, 3]
        out[f, :3, :3] = (rest_rot * Rotation.from_rotvec(w * delta)).as_matrix()
    return out


def bad_hand_frames(wrist_world: np.ndarray, cam_dist: np.ndarray, valid: np.ndarray, fps: float,
                    max_reach: float, max_speed: float = 2.0) -> np.ndarray:
    """Frames where a hand track is unusable: flagged invalid, farther than an arm's reach from
    the head camera, or moving faster than `max_speed` m/s since the last good frame."""
    bad = ~valid.astype(bool) | (cam_dist > max_reach)
    last = None
    for i in range(len(bad)):
        if bad[i]:
            continue
        if last is not None and np.linalg.norm(wrist_world[i] - wrist_world[last]) * fps / (i - last) > max_speed:
            bad[i] = True
        else:
            last = i
    return bad


def fill_hand(data: dict, side: str, bad: np.ndarray) -> None:
    """Replace bad frames of one hand: interpolate the world wrist pose and finger pose between
    good frames (hold the nearest good frame at the ends), then write it back in camera frame,
    the frame the loader reads (wrist_orient/trans_in_camera, finger_pose)."""
    from scipy.spatial.transform import Rotation, Slerp
    good = np.nonzero(~bad)[0]
    if len(good) == 0 or not bad.any():
        return
    c2w = data["camera_to_world_transform"].astype(np.float64)
    Rc, tc = c2w[:, :3, :3], c2w[:, :3, 3]
    orient = data[f"hand_{side}_wrist_orient_in_camera"].astype(np.float64)
    trans = data[f"hand_{side}_wrist_trans_in_camera"].astype(np.float64)
    fingers = data[f"hand_{side}_finger_pose"].astype(np.float64).reshape(len(bad), -1)
    rot_w = Rotation.from_matrix(Rc @ Rotation.from_rotvec(orient).as_matrix())
    pos_w = np.einsum("nij,nj->ni", Rc, trans) + tc
    frames = np.clip(np.arange(len(bad)), good[0], good[-1])  # hold at the ends
    if len(good) > 1:
        new_rot = Slerp(good, rot_w[good])(frames)
    else:
        new_rot = Rotation.from_quat(np.repeat(rot_w[good].as_quat(), len(bad), axis=0))
    new_pos = np.stack([np.interp(frames, good, pos_w[good, k]) for k in range(3)], axis=1)
    new_fingers = np.stack([np.interp(frames, good, fingers[good, k]) for k in range(fingers.shape[1])], axis=1)
    cam_rot = Rotation.from_matrix(np.einsum("nji,njk->nik", Rc, new_rot.as_matrix()))  # Rc^T @ R_w
    cam_pos = np.einsum("nji,nj->ni", Rc, new_pos - tc)
    for key, value in ((f"hand_{side}_wrist_orient_in_camera", cam_rot.as_rotvec()),
                       (f"hand_{side}_wrist_trans_in_camera", cam_pos),
                       (f"hand_{side}_finger_pose", new_fingers)):
        data[key] = np.where(bad.reshape(-1, *[1] * (data[key].ndim - 1)),
                             value.reshape(data[key].shape), data[key]).astype(data[key].dtype)
    w2c = np.tile(np.eye(4), (len(bad), 1, 1))
    w2c[:, :3, :3] = Rotation.from_rotvec(data[f"hand_{side}_wrist_orient_in_camera"].astype(np.float64)).as_matrix()
    w2c[:, :3, 3] = data[f"hand_{side}_wrist_trans_in_camera"]
    data[f"hand_{side}_wrist_to_camera_transform"] = w2c.astype(np.float32)
    data[f"hand_{side}_wrist_to_world_transform"] = (c2w @ w2c).astype(np.float32)


def bad_object_frames(T: np.ndarray, fps: float, max_speed: float = 1.5, max_turn: float = 10.0) -> np.ndarray:
    """Frames where the object track jumps: faster than `max_speed` m/s or `max_turn` rad/s since the
    last good frame. Public ground truth never exceeds 1.31 m/s and 8.6 rad/s."""
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(T[:, :3, :3])
    bad = np.zeros(len(T), dtype=bool)
    last = 0
    for i in range(1, len(T)):
        dt = (i - last) / fps
        if (np.linalg.norm(T[i, :3, 3] - T[last, :3, 3]) / dt > max_speed
                or (rot[i] * rot[last].inv()).magnitude() / dt > max_turn):
            bad[i] = True
        else:
            last = i
    return bad


def fill_object(T: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Interpolate bad object frames between good ones (hold the last good pose at the end)."""
    from scipy.spatial.transform import Rotation, Slerp
    good = np.nonzero(~bad)[0]
    frames = np.clip(np.arange(len(T)), good[0], good[-1])
    out = T.copy()
    out[:, :3, 3] = np.stack([np.interp(frames, good, T[good, :3, 3][:, k]) for k in range(3)], axis=1)
    out[:, :3, :3] = Slerp(good, Rotation.from_matrix(T[good, :3, :3]))(frames).as_matrix()
    return np.where(bad[:, None, None], out, T)


def hand_scale(data: dict) -> float | None:
    """Mean over both hands of the median fitted MANO scale on valid frames (1.0 = true size)."""
    scales = [float(np.median(data[f"hand_{side}_scale"][data[f"hand_{side}_is_valid"].astype(bool)]))
              for side in ("left", "right") if data[f"hand_{side}_is_valid"].any()]
    return float(np.mean(scales)) if scales else None


def scan_size_ratio(bundle_mesh_vertices: np.ndarray, object_name: str) -> float | None:
    """Reconstructed mesh size / public scan size (mean of sorted oriented extents), or None."""
    scan = next((p for p in (SCANS / f"{object_name}/{object_name}_visual.glb", SCANS / f"{object_name}/{object_name}.glb")
                 if p.is_file()), None)
    if scan is None:
        return None
    scan_v = np.asarray(trimesh.load(scan, force="mesh", process=False).vertices)
    return float(np.mean(np.array(oriented_extents(bundle_mesh_vertices)) / np.array(oriented_extents(scan_v))))


def rescale_world(data: dict, factor: float) -> None:
    """Multiply every length in the bundle by `factor` (about the world origin): camera, hand and
    object translations and the object mesh scale. Rotations and finger poses are unchanged."""
    slots = object_slots(data)
    for key in ("camera_to_world_transform", "object_to_camera_transform", *[k for k, _, _ in slots],
                "hand_left_wrist_to_world_transform", "hand_right_wrist_to_world_transform",
                "hand_left_wrist_to_camera_transform", "hand_right_wrist_to_camera_transform"):
        data[key] = data[key].copy()
        data[key][..., :3, 3] *= factor
    for side in ("left", "right"):
        data[f"hand_{side}_wrist_trans_in_camera"] = data[f"hand_{side}_wrist_trans_in_camera"] * factor
        data[f"hand_{side}_scale"] = data[f"hand_{side}_scale"] * factor
    for _, scale_key, _ in slots:
        data[scale_key] = np.asarray(float(data.get(scale_key, 1.0)) * factor, dtype=np.float32)


def still_scene(data: dict) -> None:
    """Safety-net scene: the object rests at its frame-0 pose and both hands stay where they were at
    frame 0 (raised over the table at the start of every recording) for the whole clip.

    Hands are held in world by counter-rotating their camera-frame parameters (what the loader
    reads) with the camera motion. Translations ignore the MANO rest-wrist offset, so a held hand
    can drift a few cm as the head turns; it stays far from the object either way."""
    from scipy.spatial.transform import Rotation
    n = len(data["object_to_world_transform"])
    for key in ("object_to_camera_transform", *[k for k, _, _ in object_slots(data)]):
        data[key] = np.repeat(data[key][:1], n, axis=0)
    c2w = data["camera_to_world_transform"].astype(np.float64)
    rel = np.einsum("nji,jk->nik", c2w[:, :3, :3], c2w[0, :3, :3])  # R_c2w(t)^T R_c2w(0)
    mirror = np.diag([1.0, -1.0, -1.0])  # HaMeR stores left hands in right-hand parameter space
    for side in ("left", "right"):
        a = np.einsum("ij,njk,kl->nil", mirror, rel, mirror) if side == "left" else rel
        r0 = Rotation.from_rotvec(data[f"hand_{side}_wrist_orient_in_camera"][0].astype(np.float64)).as_matrix()
        data[f"hand_{side}_wrist_orient_in_camera"] = Rotation.from_matrix(a @ r0).as_rotvec().astype(np.float32)
        p0 = c2w[0, :3, :3] @ data[f"hand_{side}_wrist_trans_in_camera"][0] + c2w[0, :3, 3]
        cam_t = np.einsum("nji,nj->ni", c2w[:, :3, :3], p0 - c2w[:, :3, 3])
        data[f"hand_{side}_wrist_trans_in_camera"] = cam_t.astype(np.float32)
        data[f"hand_{side}_finger_pose"] = np.repeat(data[f"hand_{side}_finger_pose"][:1], n, axis=0)
        w2c = np.tile(np.eye(4), (n, 1, 1))
        w2c[:, :3, :3] = Rotation.from_rotvec(data[f"hand_{side}_wrist_orient_in_camera"].astype(np.float64)).as_matrix()
        w2c[:, :3, 3] = cam_t
        data[f"hand_{side}_wrist_to_camera_transform"] = w2c.astype(np.float32)
        data[f"hand_{side}_wrist_to_world_transform"] = (c2w @ w2c).astype(np.float32)
        data[f"hand_{side}_is_valid"] = np.ones(n, dtype=bool)


def stable_tilt(vertices: np.ndarray, R0: np.ndarray) -> np.ndarray | None:
    """World rotation (3x3) that tips an object with orientation R0 onto the nearest face it can
    rest on (trimesh stable poses of the convex hull), or None if it already rests (< 1 deg)."""
    import trimesh
    from scipy.spatial.transform import Rotation
    stable, _ = trimesh.poses.compute_stable_poses(trimesh.Trimesh(vertices).convex_hull, n_samples=1)
    down = np.array([0.0, 0.0, -1.0])
    rest_downs = [R0 @ (S[:3, :3].T @ down) for S in stable]  # each resting face's down, in world
    target = max(rest_downs, key=lambda d: d @ down)
    axis = np.cross(target, down)
    angle = np.arctan2(np.linalg.norm(axis), target @ down)
    if angle < np.radians(1.0):
        return None
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()


def level_scene(data: dict, bundle: Path, world_tilt: float = 20.0, max_tilt: float = 30.0) -> dict:
    """Make every object start resting on one of its faces.

    Gravity alignment leaves objects tilted 5-27 deg off any face they could rest on (ep35-38, 41),
    so in simulation the object tips over in the first 5 frames and sits 8-24 deg off its
    reference for the rest of the clip (NVIDIA's tissue sample: < 1 deg). The first object is
    levelled by one rotation of the whole world about its frame-0 position, as NVIDIA's
    level_result_bundle.py does, so hands keep their grip; the scorer registers trajectories, so a
    global rotation costs nothing. Further objects, and a first object tilted more than
    `world_tilt`, get the tilt as a fixed rotation in their own frame about the mesh centre: a
    large tilt is an object-pose error, and rotating the world by it moves the table (ep36's
    27 deg left the spoon 10 cm above its support). Tilts above `max_tilt` mean another face and
    are left alone."""
    from scipy.spatial.transform import Rotation
    info = {}
    for i, (pose_key, scale_key, mesh_file) in enumerate(object_slots(data)):
        v = object_vertices(bundle, data, scale_key, mesh_file)
        T0 = data[pose_key][0].astype(np.float64)
        R = stable_tilt(v, T0[:3, :3])
        tilt = 0.0 if R is None else float(np.degrees(Rotation.from_matrix(R).magnitude()))
        info[f"tilt_deg_{i}"] = round(tilt, 1)
        if R is None or tilt > max_tilt:
            continue
        if i == 0 and tilt <= world_tilt:
            A = np.eye(4)
            A[:3, :3] = R
            A[:3, 3] = T0[:3, 3] - R @ T0[:3, 3]
            keys = ("camera_to_world_transform", "hand_left_wrist_to_world_transform",
                    "hand_right_wrist_to_world_transform", *[k for k, _, _ in object_slots(data)])
            for key in keys:
                data[key] = (A @ data[key].astype(np.float64)).astype(data[key].dtype)
        else:
            local = T0[:3, :3].T @ R @ T0[:3, :3]
            c = v.mean(axis=0)
            B = np.eye(4)
            B[:3, :3] = local
            B[:3, 3] = c - local @ c
            data[pose_key] = (data[pose_key].astype(np.float64) @ B).astype(data[pose_key].dtype)
        info[f"levelled_{i}"] = "world" if i == 0 and tilt <= world_tilt else "object"
    return info


def cleaned_bundle(bundle: Path, dest: Path, fps: float, rest_hold: bool = True, hand_clean: bool = True,
                   object_clean: bool = True, rescale_by_hands: bool = False, rescale_by_scan: str | None = None,
                   still: bool = False, level: bool = False, reach_cm: float = 10.0) -> tuple[Path, dict]:
    """Copy of the bundle with the repairs reconstructions need before load:

    hand_clean: the loader ignores hand_*_is_valid, and hand tracks often fly off (a wrist 2 m
      from the head camera, or jumping at up to 200 m/s on ep41). A reference wrist that jumps
      ends every training episode (FlashCHORD terminates at 15 cm wrist error), so bad frames
      are refilled from the good ones (bad_hand_frames, fill_hand).
    object_clean: the same for object frames that jump faster than any real object moves
      (ep41's track breaks at frames 232-261 at up to 7 m/s; termination is at 10 cm).
    rest_hold: every public ground-truth object is at rest until touched, while reconstructions
      jitter ~0.6 cm per frame from frame 0, so NVIDIA's support detector builds no table under
      them. The object is held still until a wrist comes within `reach_cm`. On the 14 public
      reconstructions this never cuts into real motion (the reach comes 10-20 frames before the
      true object moves) and raises reconstruction-only AUC 0.216 -> 0.240.
    rescale_by_hands: the loader rebuilds hands at true MANO size and ignores hand_*_scale, so a
      scene reconstructed 1.4x too large gets true-size hands around a 1.4x object. Dividing all
      lengths by the hand scale makes the hands and the object agree (and shrinks the path).
    rescale_by_scan: object name; divide all lengths by the reconstructed-mesh / public-scan size
      ratio, so the object comes out its real size. On 14 public reconstructions this raises
      reconstruction-only AUC 0.240 -> 0.291 (ep41 0.287 -> 0.430); the hand scale as ruler
      does worse (0.229). No public scan exists for wooden_spoon.
    level: start every object resting on a face (level_scene), after the object repairs.
    still: safety-net scene (still_scene), applied last.
    """
    data = dict(np.load(bundle / "result.npz"))
    info: dict = {}
    slots = object_slots(data)
    names = [str(n) for n in data["object_names"]] if "object_names" in data else [rescale_by_scan]
    if rescale_by_scan:
        ratios = [scan_size_ratio(object_vertices(bundle, data, sk, mf), name)
                  for name, (_, sk, mf) in zip(names, slots)]
        ratios = [r for r in ratios if r is not None]
        if not ratios:
            raise SystemExit(f"no public scan for {names}: cannot --rescale-by-scan")
        ratio = float(np.exp(np.mean(np.log(ratios))))  # one scene, one scale: geometric mean over objects
        info["scan_ratios"] = [round(r, 3) for r in ratios]
        rescale_world(data, 1.0 / ratio)
        info["rescaled_by"] = round(1.0 / ratio, 4)
    elif rescale_by_hands:
        scale = hand_scale(data)
        if scale:
            rescale_world(data, 1.0 / scale)
            info["rescaled_by"] = round(1.0 / scale, 4)
    if hand_clean:
        scale = np.nanmedian([np.median(data[f"hand_{s}_scale"]) for s in ("left", "right")])
        max_reach = 1.0 * max(1.0, float(scale))  # head camera to wrist; scenes come out too large
        for side in ("left", "right"):
            w2w = data[f"hand_{side}_wrist_to_world_transform"].astype(np.float64)
            cam_dist = np.linalg.norm(data[f"hand_{side}_wrist_to_camera_transform"][:, :3, 3], axis=1)
            bad = bad_hand_frames(w2w[:, :3, 3], cam_dist, data[f"hand_{side}_is_valid"], fps, max_reach)
            info[f"hand_{side}_refilled_frames"] = int(bad.sum())
            fill_hand(data, side, bad)
    for i, (pose_key, scale_key, mesh_file) in enumerate(slots):
        tag = "" if len(slots) == 1 else f"_{names[i]}"
        T = data[pose_key].astype(np.float64)
        if object_clean:
            bad = bad_object_frames(T, fps)
            info[f"object_refilled_frames{tag}"] = int(bad.sum())
            if bad.any():
                T = fill_object(T, bad)
        if rest_hold:
            frame = reach_frame(data, object_vertices(bundle, data, scale_key, mesh_file), reach_cm, pose_key)
            info[f"rest_hold_until_frame{tag}"] = frame
            if frame:
                T = hold_until(T, frame)
        data[pose_key] = T.astype(data[pose_key].dtype)
    if level:
        info.update(level_scene(data, bundle))
    cam_inv = np.linalg.inv(data["camera_to_world_transform"].astype(np.float64))
    data["object_to_camera_transform"] = (cam_inv @ data["object_to_world_transform"].astype(np.float64)).astype(
        data["object_to_camera_transform"].dtype)
    if still:
        still_scene(data)
        info["still"] = True
    dest.mkdir(parents=True, exist_ok=True)
    for f in bundle.iterdir():
        if f.name != "result.npz" and not (dest / f.name).exists():
            (dest / f.name).symlink_to(f.resolve())
    np.savez(dest / "result.npz", **data)
    return dest, info


# ---------------------------------------------------------------- stage commands
def run(cmd: list[str], cwd: Path, env: dict, log: Path) -> None:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        result = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join(log.read_text().splitlines()[-25:])
        raise SystemExit(f"step failed (exit {result.returncode}); last lines of {log}:\n{tail}")


def partition(root: Path, seq: str) -> Path:
    return root / f"sequence_id={seq}" / f"robot_name={ROBOT}"


def has_parquet(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.parquet"))


# ---------------------------------------------------------------- output checks
def read_partition(path: Path) -> dict:
    import pyarrow.parquet as pq
    table = pq.read_table(next(path.glob("*.parquet")))
    return {name: table.column(name)[0].as_py() for name in table.column_names}


def support_tops(usda: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """World-aligned (min, max) boxes of the support disks."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usda))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    boxes = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Cylinder):
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            boxes.append((np.array(box.GetMin()), np.array(box.GetMax())))
    return boxes


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], axis=-1)).as_matrix()


def check_output(processed: Path, mesh_paths: list[Path], usda: Path, frames: int, fps: float) -> dict:
    """Checks on the retargeted partition; penetration and seating over every object body."""
    d = read_partition(processed)
    errors = {side: np.asarray(d[f"robot_{side}_frame_task_errors"], dtype=float) for side in ("right", "left")}
    links = np.concatenate([np.asarray(d[f"robot_{side}_frames"], dtype=float)[:, 2:, :3]
                            for side in ("right", "left")], axis=1)  # skip universe/root_joint
    disks = support_tops(usda) if usda.is_file() else []
    depths, seating = [], []
    for b, mesh_path in enumerate(mesh_paths):
        position = np.asarray(d["object_body_position"], dtype=float)[:, b]
        rotation = quat_wxyz_to_matrix(np.asarray(d["object_body_wxyz"], dtype=float)[:, b])
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        # Robot link frames in the object's frame; depth > 0 means inside the mesh.
        local = np.einsum("tji,tkj->tki", rotation, links - position[:, None, :])
        points = local.reshape(-1, 3)
        depths.append(np.concatenate([trimesh.proximity.signed_distance(mesh, points[i:i + 2000])  # batched: memory
                                      for i in range(0, len(points), 2000)]).reshape(local.shape[:2]).max(axis=1))
        world0 = np.asarray(mesh.vertices) @ rotation[0].T + position[0]
        centre = world0[:, :2].mean(axis=0)
        under = [hi[2] for lo, hi in disks if np.all(lo[:2] <= centre) and np.all(centre <= hi[:2])]
        seating.append(round((float(world0[:, 2].min()) - max(under)) * 100, 2) if under else None)
    depth = np.max(depths, axis=0)
    position = np.asarray(d["object_body_position"], dtype=float)[:, 0]
    report = {
        "frames": len(position),
        "fps": float(d["fps"]),
        "task_error_cm": {side: {"mean": round(float(e.mean()) * 100, 2), "max": round(float(e.max()) * 100, 2)}
                          for side, e in errors.items()},
        "penetration_frames": {"any": round(float(np.mean(depth > 0)), 3),
                               "over_1cm": round(float(np.mean(depth > 0.01)), 3),
                               "over_2cm": round(float(np.mean(depth > 0.02)), 3)},
        "support_disks": len(disks),
        "frame0_seating_cm": seating[0] if len(seating) == 1 else seating,
    }
    problems = []
    if report["frames"] != frames:
        problems.append(f"{report['frames']} frames, bundle has {frames}")
    if abs(report["fps"] - fps) > 1e-6:
        problems.append(f"fps {report['fps']} != {fps}")
    for side, e in report["task_error_cm"].items():
        if e["mean"] > 3 or e["max"] > 6:
            problems.append(f"{side} task error {e['mean']} cm mean / {e['max']} cm max (bar ~2.2 / 6)")
    if report["penetration_frames"]["over_2cm"] > 0:
        problems.append(f"{report['penetration_frames']['over_2cm']:.1%} of frames have a link >2 cm inside the object")
    if not disks:
        problems.append("no support disk kept (loaded at floor level?)")
    for b, seat in enumerate(seating):
        what = "the object" if len(seating) == 1 else f"object {b}"
        if seat is None:
            problems.append(f"no support disk under {what} at frame 0")
        elif abs(seat) > 2:
            problems.append(f"{what} bottom is {seat} cm from the support top at frame 0")
    report["warnings"] = problems
    return report


def write_score_input(bundle: Path, out: Path, episode: int, obj: str | None) -> None:
    """Bundle object trajectories in tools/track3/score_local.py's format (reconstruction-only
    score): poses (frames, objects, 7) in the bundle's object order, one mesh per object name."""
    from scipy.spatial.transform import Rotation
    data = dict(np.load(bundle / "result.npz"))
    names = [str(n) for n in data["object_names"]] if "object_names" in data else [obj]
    (out / "score_input/results").mkdir(parents=True, exist_ok=True)
    (out / f"score_input/meshes/episode_{episode:06d}").mkdir(parents=True, exist_ok=True)
    poses = []
    for name, (pose_key, scale_key, mesh_file) in zip(names, object_slots(data)):
        T = data[pose_key].astype(float)
        poses.append(np.concatenate([T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()], axis=-1))
        mesh = trimesh.load(bundle / mesh_file, force="mesh", process=False)
        mesh.apply_scale(float(data.get(scale_key, 1.0)))
        mesh.export(out / f"score_input/meshes/episode_{episode:06d}/{name}.obj")
    np.save(out / f"score_input/results/episode_{episode:06d}.npy", np.stack(poses, axis=1))


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--object", help="object name as in episodes.json, e.g. white_pot")
    parser.add_argument("--bundle", type=Path, help="dir holding result.npz + mesh.obj (default: from the hub)")
    parser.add_argument("--sequence-id", help="unique per reconstruction (default: ep<N>_<object>)")
    parser.add_argument("--out", type=Path, help="default: data/outputs/load_retarget/episode_XXXXXX/<object>")
    parser.add_argument("--fps", type=float, default=TRACK3_FPS, help="clip frame rate (Track 3: 20.254)")
    parser.add_argument("--ground-z-offset", type=float, default=1.0, help="table height in m")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--force", action="store_true", help="redo every step")
    parser.add_argument("--no-rest-hold", action="store_true",
                        help="keep the object track as reconstructed (default: hold it still until a hand reaches it)")
    parser.add_argument("--no-hand-clean", action="store_true",
                        help="keep hand tracks as reconstructed (default: refill frames that fly off or jump)")
    parser.add_argument("--no-object-clean", action="store_true",
                        help="keep object jumps (default: interpolate frames faster than any real object)")
    parser.add_argument("--rescale-by-hands", action="store_true",
                        help="divide every length by the fitted hand scale so hands come out true size")
    parser.add_argument("--rescale-by-scan", action="store_true",
                        help="divide every length by the mesh-to-public-scan size ratio (object comes out real size)")
    parser.add_argument("--still-object", action="store_true",
                        help="safety net: object at rest and hands at their first pose for the whole clip")
    parser.add_argument("--level", action="store_true",
                        help="rotate the scene so each object starts resting on a face (else it tips over in sim)")
    parser.add_argument("--merge", action="store_true",
                        help="multi-object episode: merge every object's bundle from the hub into one scene")
    args = parser.parse_args()

    if args.merge:
        if args.episode is None or args.object or args.bundle:
            parser.error("--merge takes --episode only (all of the episode's objects are merged)")
    elif args.bundle is None:
        if args.episode is None or not args.object:
            parser.error("give --episode and --object, or --bundle, or --episode with --merge")
        args.bundle = bundle_dir_for(args.episode, args.object)
    label = "merged" if args.merge else args.object
    seq = args.sequence_id or (f"ep{args.episode}_{label}" if args.episode is not None else None)
    if not seq:
        parser.error("--sequence-id is required with --bundle and no --episode")
    out = (args.out or OUTPUTS / "load_retarget" / f"episode_{args.episode:06d}" / label).resolve()
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    merge_info = None
    if args.merge:
        args.bundle, merge_info = merged_bundle(args.episode, out / "bundle_merged")
    bundle = args.bundle.resolve()
    log = out / "pipeline.log"

    # Paths contain /assets/human_motion_data/ so FlashCHORD can re-root them if the tree moves.
    hmd = out / "assets/human_motion_data"
    loaded = out / "intermediate/ego_recon/loaded"
    processed = hmd / "ego_recon/processed"
    usda = hmd / f"ego_recon/reconstructed_stage/{seq}_{ROBOT}_support.usda"
    env = dict(os.environ, HUMAN_MOTION_DATA_DIR=str(hmd), ROBOTIC_GROUNDING_INTERMEDIATE_DIR=str(out / "intermediate"),
               V2D_EGO_FPS=repr(args.fps), PYOPENGL_PLATFORM="egl", PYTHONUNBUFFERED="1")
    py = sys.executable
    report = {"sequence_id": seq, "episode": args.episode, "object": args.object, "bundle": str(bundle),
              "fps": args.fps, "ground_z_offset": args.ground_z_offset,
              "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if merge_info:
        report["merge"] = merge_info
        print(f"== merged bundle: {merge_info}", flush=True)

    print(f"== check bundle {bundle}", flush=True)
    report["bundle_check"] = check_bundle(bundle, merge_info["objects"][0] if merge_info else args.object)
    for w in report["bundle_check"]["warnings"]:
        print(f"  ! {w}")
    if not (args.no_rest_hold and args.no_hand_clean and args.no_object_clean) or args.rescale_by_hands \
            or args.rescale_by_scan or args.still_object or args.level:
        bundle, report["cleaning"] = cleaned_bundle(
            bundle, out / "bundle_cleaned", args.fps, rest_hold=not args.no_rest_hold,
            hand_clean=not args.no_hand_clean, object_clean=not args.no_object_clean,
            rescale_by_hands=args.rescale_by_hands, rescale_by_scan=label if args.rescale_by_scan else None,
            still=args.still_object, level=args.level)
        print(f"== cleaned bundle: {report['cleaning']}", flush=True)

    print("== load (MANO FK -> ManoSharpaData)", flush=True)
    if not has_parquet(partition(loaded, seq)):
        run([py, "-m", "v2d.task_library_loader.lib.run_loader", "--dataset", "ego_recon",
             "--dataset_root", bundle, "--result_subpath", ".", "--output_dir", loaded,
             "--mano_model_dir", MANO_DIR, "--no_ground_align", "--ground_z_offset", args.ground_z_offset,
             "--sequence_name", seq, "--object_name", seq, "--sequence_id", seq, "--device", args.device, "--save"],
            RECON, env, log)

    print("== retarget (IK onto the Sharpa hands)", flush=True)
    if not has_parquet(partition(processed, seq)):
        run([py, "scripts/retarget/ego_recon_to_sharpa.py", "--input_dir", loaded, "--output_dir", processed,
             "--sequence_id", seq, "--device", args.device, "--save"], RG, env, log)

    # Loader body names: the sequence id for one object, <seq>_<object> per object when merged.
    bodies = [f"{seq}_{name}" for name in merge_info["objects"]] if merge_info else [seq]
    print("== object URDF", flush=True)
    if not all((processed / f"{body}_rigid.urdf").is_file() for body in bodies):
        run([py, "scripts/generate_rigid_urdfs.py", "--dataset", "ego_recon"], RG, env, log)

    print("== support surface", flush=True)
    if not usda.is_file():
        run([py, "scripts/reconstruct_support_surfaces.py", "--dataset", "ego_recon", "--input_dir", processed,
             "--sequence_id", seq, "--robot_name", ROBOT, "--output", usda], RG, env, log)

    video = out / "qa" / f"{seq}.mp4"
    if not args.skip_video and not video.is_file():
        print("== QA video", flush=True)
        run([py, "scripts/retarget/vis_retargeted.py", "--input_dir", processed, "--robot", ROBOT,
             "--sequence_id", seq, "--support_usd", usda, "--save_mp4", "--mp4_dir", out / "qa"], RG, env, log)

    print("== check output", flush=True)
    report["output_check"] = check_output(partition(processed, seq), [processed / f"{body}.obj" for body in bodies],
                                          usda, report["bundle_check"]["frames"], args.fps)
    for w in report["output_check"]["warnings"]:
        print(f"  ! {w}")
    if args.episode is not None:
        write_score_input(bundle, out, args.episode, args.object)
    report.update(processed=str(partition(processed, seq)), support_usd=str(usda),
                  video=str(video) if video.is_file() else None,
                  finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("bundle_check", "output_check")}, indent=1))
    print(f"\nreport: {out / 'report.json'}\nvideo:  {report['video']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
