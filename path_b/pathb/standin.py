"""Stand-in GPU outputs, so the CPU stages run on the Mac before the dev cache exists (James,
2026-10-22: "stand-in inputs are allowed; label them").

    RETARGET_VENV=~/V2D/venvs/retarget MANO_DIR=~/V2D/mano_v1_2 \\
        uv run --project path_b python path_b/run.py standin --episodes 40 26 --out <run>

Source: run 4c2eac31 (on the Mac with frames): P0-P2 as they are (hard links), and its v2 P4 bundles,
i.e. the team's HF reconstructions (NVIDIA baseline: SAM2 + FoundationPose + DROID-SLAM + HaMeR) at
Path B v2's stereo scale. From those, in the v3 stages' own formats:
  p1c_camera  camera_to_world/ (the bundle's camera; two objects: the roster-first object's bundle),
              intrinsics.json (the bundle's camera model, fx ~990: its camera-frame poses use it)
  p1d_hands   hand_tracks.json, hamer_aligned_filled/{2,3}/%06d.json (the bundle's MANO, camera frame)
  p1b_masks   obj_<k>/, hand_L/, hand_R/ silhouettes of the bundle's mesh and the loader's MANO hands,
              rendered with that camera model (where the team's reconstruction sees them in the image)
  s3a_mesh    mesh.obj / mesh.glb (the bundle's mesh x object_scale), mass.json, sha256.txt; no CoACD
  s3b_track   poses/ (each object's own bundle: object_to_camera), track_meta.json with the S3b
              driver's own measures: mask IoU and median depth residual against P2's stereo depth,
              the mesh rendered with ego_cam_a's factory model; nothing is marked recovered
Every report.json says "stand_in" with its source; the loss report and the analysis page label them.
Only the dev episodes, only the Mac: the dev cache replaces all of it.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

from . import scan as S
from .baseline import intrinsics_json
from .density import density
from .geometry import read_depth, write_transform
from .paths import OUTPUTS, PATH_B, episode_info, git_commit
from .runner import write_json

SOURCE = "4c2eac31"
REAL = ("p0_ingest", "p1_calibrate", "p2_depth")
STOOD_IN = ("p1b_masks", "p1c_camera", "p1d_hands", "s3a_mesh", "s3b_track")
NOTE = "stand-in from run 4c2eac31's v2 P4 bundle (team HF reconstruction x stereo scale), not this stage's model"


def source_bundle(number: int, name: str) -> Path:
    return OUTPUTS / SOURCE / f"episode_{number:06d}/p4_export/episode_{number:06d}/{name}/result_slam_gravity_aligned"


def K_of(intr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(intr, dtype=np.float64) * scale
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def rigid(T: np.ndarray) -> np.ndarray:
    """Strip a scale carried in the rotation block (the baseline's object poses)."""
    R = T[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = U @ Vt, T[:3, 3]
    return out


def write_report(root: Path, stage: str, number: int, split: str, metrics: list[dict], outputs: dict) -> None:
    write_json(root / stage / "report.json", {
        "stage": stage, "status": "PASS", "episode": number, "split": split, "commit": git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seconds": 0.0, "gpu_peak_mib": None,
        "metrics": [{"name": "stand-in source", "value": SOURCE, "unit": "", "kind": "health", "bar": "",
                     "status": "INFO", "note": NOTE}, *metrics],
        "outputs": outputs, "files_read": [], "skip_reason": None, "error": None,
        "stand_in": {"source_run": SOURCE, "note": NOTE}})


def info(name: str, value, unit: str = "", note: str = "") -> dict:
    return {"name": name, "value": value, "unit": unit, "kind": "health", "bar": "", "status": "INFO", "note": note}


def mano_world(bundle: Path, out: Path) -> dict:
    """The loader's MANO hands of a bundle, in its world (drivers/mano_world.py in $RETARGET_VENV)."""
    venv = os.environ.get("RETARGET_VENV")
    mano = os.environ.get("MANO_DIR")
    if not (venv and mano):
        raise SystemExit("set RETARGET_VENV and MANO_DIR (the folder holding models/MANO_*.pkl)")
    if not out.is_file():
        subprocess.run([f"{venv}/bin/python", str(PATH_B / "pathb/drivers/mano_world.py"), str(bundle), str(out), mano],
                       check=True)
    return dict(np.load(out))


def silhouette(mesh: o3d.geometry.TriangleMesh, K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """(H, W) depth of a camera-frame mesh (0 = not seen)."""
    return S.rendered_depth(mesh, K, np.eye(4), shape)


def tri(vertices: np.ndarray, faces: np.ndarray) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
                                     o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)))


def episode(number: int, out: Path) -> None:
    e = episode_info(number)
    names = [o["name"] for o in e["objects"]]
    n = int(e["num_frames"])
    root, src = out / f"episode_{number:06d}", OUTPUTS / SOURCE / f"episode_{number:06d}"
    root.mkdir(parents=True, exist_ok=True)
    for stage in REAL:
        if not (root / stage).exists():
            subprocess.run(["cp", "-al", str(src / stage), str(root / stage)], check=True)
    cam_src = source_bundle(number, names[0])  # two objects: the roster-first object's camera and hands
    base = dict(np.load(cam_src / "result.npz"))
    c2w = base["camera_to_world_transform"].astype(np.float64)
    full = cv2.imread(str(root / "p1_calibrate/cam_a/000000.jpg")).shape[:2]
    half = (full[0] // 2, full[1] // 2)
    K_team = K_of(base["camera_intrinsics"], 0.5)
    calib = json.loads((root / "p1_calibrate/calib.json").read_text())["cam_a"]
    K_factory = np.asarray(calib["K"], dtype=np.float64) * [[0.5], [0.5], [1]]
    print(f"== stand-ins for episode {number}: {names}, camera from {names[0]}'s bundle", flush=True)

    # P1c: the bundle's camera
    d = root / "p1c_camera"
    (d / "camera_to_world").mkdir(parents=True, exist_ok=True)
    for t in range(n):
        write_transform(d / f"camera_to_world/{t:06d}.json", c2w[t])
    intrinsics_json(K_of(base["camera_intrinsics"]), (full[1], full[0]), d / "intrinsics.json")
    path = float(np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1).sum())
    write_report(root, "p1c_camera", number, e["split"], [info("camera path length", path, "m"),
                 info("focal length of the stand-in camera model", float(base["camera_intrinsics"][0]), "px",
                      note=f"ego_cam_a factory: {calib['K'][0][0]:.0f} px")], {"camera_to_world": d / "camera_to_world"})

    # P1d: the bundle's MANO hands in HaMeR's per-frame files
    d = root / "p1d_hands"
    tracks = {"tracks": [{"object_id": 2, "role": "hand", "is_right": False, "mask_dir": "hand_L"},
                         {"object_id": 3, "role": "hand", "is_right": True, "mask_dir": "hand_R"}],
              "objects": [{"name": nm, "mask_dir": f"obj_{k}"} for k, nm in enumerate(names)]}
    d.mkdir(exist_ok=True)
    write_json(d / "hand_tracks.json", tracks)
    for oid, side in ((2, "left"), (3, "right")):
        hd = d / f"hamer_aligned_filled/{oid}"
        hd.mkdir(parents=True, exist_ok=True)
        for t in np.flatnonzero(base[f"hand_{side}_is_valid"]):
            write_json(hd / f"{t:06d}.json", {
                "mano": {"betas": base[f"hand_{side}_betas"], "global_orient": base[f"hand_{side}_wrist_orient_in_camera"][t],
                         "hand_pose": base[f"hand_{side}_finger_pose"][t]},
                "camera": {"cam_t": base[f"hand_{side}_wrist_trans_in_camera"][t]},
                "hand_scale": float(base[f"hand_{side}_scale"][t]), "stand_in": SOURCE})
    write_report(root, "p1d_hands", number, e["split"],
                 [info(f"{side} hand valid", float(base[f"hand_{side}_is_valid"].mean()), "share") for side in ("left", "right")],
                 {"hand_tracks": d / "hand_tracks.json"})

    # S3a: the bundle's mesh, metric
    meshes = {}
    for name in names:
        b = source_bundle(number, name)
        data = np.load(b / "result.npz")
        m = trimesh.load(b / "mesh.obj", force="mesh", process=False)
        m.apply_scale(float(data["object_scale"]))
        d = root / f"s3a_mesh/{name}"
        d.mkdir(parents=True, exist_ok=True)
        m.export(d / "mesh.obj")
        m.export(d / "mesh.glb")
        volume = float(m.volume if m.is_watertight else m.convex_hull.volume)
        write_json(d / "mass.json", {"volume_m3": volume, "volume_from": "mesh" if m.is_watertight else "convex hull",
                                     "density_kg_m3": density(name), "mass_kg": volume * density(name)})
        (d / "sha256.txt").write_text(hashlib.sha256((d / "mesh.glb").read_bytes()).hexdigest() + "  mesh.glb\n")
        write_json(d / "mesh_views.json", {"stand_in": SOURCE, "fusion": "the team's SAM3D mesh, as the bundle has it"})
        meshes[name] = (m, data)
    write_report(root, "s3a_mesh", number, e["split"],
                 [info(f"{nm}: largest extent", float(np.ptp(m.vertices, axis=0).max()), "m") for nm, (m, _) in meshes.items()],
                 {nm: root / f"s3a_mesh/{nm}/mesh.obj" for nm in names})

    # P1b masks (team camera model) and S3b's measures (factory model), frame by frame
    hands = mano_world(cam_src, root / "p1d_hands/stand_in_mano_world.npz")
    d = root / "p1b_masks"
    for track in [f"obj_{k}" for k in range(len(names))] + ["hand_L", "hand_R"]:
        (d / track).mkdir(parents=True, exist_ok=True)
    write_json(d / "hand_tracks.json", tracks)
    meta = {name: [] for name in names}
    seen = {track: 0 for track in [f"obj_{k}" for k in range(len(names))] + ["hand_L", "hand_R"]}
    for name in names:
        (root / f"s3b_track/{name}/poses").mkdir(parents=True, exist_ok=True)
    w2c = np.linalg.inv(c2w)
    for t in range(n):
        stereo = read_depth(root / f"p2_depth/depth_a_half/{t:06d}.png") if (
            root / f"p2_depth/depth_a_half/{t:06d}.png").is_file() else None
        for k, name in enumerate(names):
            m, data = meshes[name]
            P = rigid(data["object_to_camera_transform"][t].astype(np.float64))
            write_transform(root / f"s3b_track/{name}/poses/{t:06d}.json", P)
            cam_mesh = tri(m.vertices @ P[:3, :3].T + P[:3, 3], m.faces)
            sil_team = silhouette(cam_mesh, K_team, half) > 0
            seen[f"obj_{k}"] += int(sil_team.any())
            mask_full = cv2.resize(sil_team.astype(np.uint8) * 255, (full[1], full[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(d / f"obj_{k}/{t:06d}.png"), mask_full)
            r = silhouette(cam_mesh, K_factory, half)
            sil = r > 0.001
            union = (sil | sil_team).sum()
            sel = sil & sil_team & (stereo > 0) if stereo is not None else np.zeros_like(sil)
            meta[name].append({"frame": t, "iou": float((sil & sil_team).sum() / union) if union else 0.0,
                               "recovered": False, "particles": False, "method": "stand-in",
                               "depth_residual_m": float(np.median(np.abs(r[sel] - stereo[sel]))) if sel.sum() >= 20 else 1.0})
        for side, track in (("left", "hand_L"), ("right", "hand_R")):
            mask = np.zeros(half, bool)
            if hands[f"{side}_valid"][t]:
                v = hands[f"{side}_vertices"][t] @ w2c[t, :3, :3].T + w2c[t, :3, 3]
                mask = silhouette(tri(v, hands[f"{side}_faces"]), K_team, half) > 0
                seen[track] += int(mask.any())
            cv2.imwrite(str(d / f"{track}/{t:06d}.png"),
                        cv2.resize(mask.astype(np.uint8) * 255, (full[1], full[0]), interpolation=cv2.INTER_NEAREST))
    write_report(root, "p1b_masks", number, e["split"],
                 [info(f"{track}: frames with a non-empty mask", count, "frames") for track, count in seen.items()],
                 {"masks": d})
    for name in names:
        write_json(root / f"s3b_track/{name}/track_meta.json", {"frames": meta[name], "stand_in": SOURCE})
    write_report(root, "s3b_track", number, e["split"],
                 [m for name in names for m in (
                     info(f"{name}: frames with mask IoU >= 0.5 (factory camera vs the track's own silhouette)",
                          float(np.mean([r["iou"] >= 0.5 for r in meta[name]])), "share"),
                     info(f"{name}: median depth residual", float(np.median([r["depth_residual_m"] for r in meta[name]])) * 100,
                          "cm"))],
                 {nm: root / f"s3b_track/{nm}/poses" for nm in names})


def main(numbers: list[int], out: Path) -> int:
    for number in numbers:
        if episode_info(number)["split"] != "public" or not (OUTPUTS / SOURCE / f"episode_{number:06d}").is_dir():
            print(f"episode {number}: no stand-in source (run {SOURCE} on the Mac): skipped", flush=True)
            continue
        episode(number, out)
    print(f"stand-ins written under {out}; next: run.py run --episodes {' '.join(map(str, numbers))} --out {out} "
          "--stages p3_scale_bridge p4_export s3c_fuse s4a_phases s4b_reference s4c_retarget s5_task", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main([int(a) for a in sys.argv[2:]], Path(sys.argv[1])))
