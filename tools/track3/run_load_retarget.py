"""Stages 2-3 for one reconstruction bundle: check it, load it, retarget it to the Sharpa hands.

Mirrors NVIDIA's retarget stages (baseline/tools/e2e/stages.py) and their onboarding recipe
(baseline/.claude/skills/onboard_ego_clip/), run natively instead of in containers:

  check bundle -> load (MANO FK, ManoSharpaData) -> retarget (IK) -> object URDF
  -> support surface -> QA video -> check output -> report.json

Track 3 settings: clip fps 20.254 (V2D_EGO_FPS), table height 1.0 m (--ground_z_offset), and
--no_ground_align because bundles are gravity-aligned upstream. One bundle = one object; the
two-object merge comes later. Steps whose outputs exist are skipped (use --force to redo).

Run with the load+retarget environment (README "Stages 2-3"), from anywhere:
    $RETARGET_VENV/bin/python tools/track3/run_load_retarget.py --episode 41 --object white_pot
    $RETARGET_VENV/bin/python tools/track3/run_load_retarget.py --bundle <dir with result.npz> \
        --sequence-id tissue_box_test --out <dir>
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
    report = {
        "frames": int(frames),
        "nan_arrays": nan_keys,
        "valid_fraction": valid,
        "hand_scale_mean": hand_scale,
        "gravity_alignment_applied": bool(data.get("gravity_alignment_applied", False)),
        "object_drift_first_30_frames_cm": round(float(drift.max()) * 100, 2),
        "mesh_vertices": int(len(vertices)),
        "mesh_extents_cm": [round(e * 100, 1) for e in oriented_extents(vertices)],
        "rest_footprint_cm": [round(e * 100, 1) for e in footprint(world0)],
    }
    scan = SCANS / (object_name or "") / f"{object_name}_visual.glb"
    if object_name and scan.is_file():
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
    long, short = report["rest_footprint_cm"]
    if long > 3 and short < 0.25 * long:
        problems.append(f"rest footprint {long}x{short} cm is a strip: object may sit on an edge (level it)")
    if "mesh_to_scan_size_ratio" in report and not 0.8 <= report["mesh_to_scan_size_ratio"] <= 1.25:
        problems.append(f"mesh is {report['mesh_to_scan_size_ratio']}x the scanned object's size")
    report["warnings"] = problems
    return report


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


def check_output(processed: Path, mesh_path: Path, usda: Path, frames: int, fps: float) -> dict:
    d = read_partition(processed)
    errors = {side: np.asarray(d[f"robot_{side}_frame_task_errors"], dtype=float) for side in ("right", "left")}
    position = np.asarray(d["object_body_position"], dtype=float)[:, 0]
    rotation = quat_wxyz_to_matrix(np.asarray(d["object_body_wxyz"], dtype=float)[:, 0])
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    # Robot link frames in the object's frame; depth > 0 means inside the mesh.
    links = np.concatenate([np.asarray(d[f"robot_{side}_frames"], dtype=float)[:, 2:, :3]
                            for side in ("right", "left")], axis=1)  # skip universe/root_joint
    local = np.einsum("tji,tkj->tki", rotation, links - position[:, None, :])
    points = local.reshape(-1, 3)
    depth = np.concatenate([trimesh.proximity.signed_distance(mesh, points[i:i + 2000])  # batched: memory
                            for i in range(0, len(points), 2000)]).reshape(local.shape[:2]).max(axis=1)
    world0 = np.asarray(mesh.vertices) @ rotation[0].T + position[0]
    disks = support_tops(usda) if usda.is_file() else []
    centre = world0[:, :2].mean(axis=0)
    under = [hi[2] for lo, hi in disks if np.all(lo[:2] <= centre) and np.all(centre <= hi[:2])]
    report = {
        "frames": len(position),
        "fps": float(d["fps"]),
        "task_error_cm": {side: {"mean": round(float(e.mean()) * 100, 2), "max": round(float(e.max()) * 100, 2)}
                          for side, e in errors.items()},
        "penetration_frames": {"any": round(float(np.mean(depth > 0)), 3),
                               "over_1cm": round(float(np.mean(depth > 0.01)), 3),
                               "over_2cm": round(float(np.mean(depth > 0.02)), 3)},
        "support_disks": len(disks),
        "frame0_seating_cm": round((float(world0[:, 2].min()) - max(under)) * 100, 2) if under else None,
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
    elif report["frame0_seating_cm"] is None:
        problems.append("no support disk under the object at frame 0")
    elif abs(report["frame0_seating_cm"]) > 2:
        problems.append(f"object bottom is {report['frame0_seating_cm']} cm from the support top at frame 0")
    report["warnings"] = problems
    return report


def write_score_input(bundle: Path, out: Path, episode: int, obj: str) -> None:
    """Bundle object trajectory in tools/track3/score_local.py's format (reconstruction-only score)."""
    from scipy.spatial.transform import Rotation
    data = np.load(bundle / "result.npz")
    T = data["object_to_world_transform"].astype(float)
    pose = np.concatenate([T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()], axis=-1)[:, None, :]
    (out / "score_input/results").mkdir(parents=True, exist_ok=True)
    (out / f"score_input/meshes/episode_{episode:06d}").mkdir(parents=True, exist_ok=True)
    np.save(out / f"score_input/results/episode_{episode:06d}.npy", pose)
    mesh = trimesh.load(bundle / "mesh.obj", force="mesh", process=False)
    mesh.apply_scale(float(data["object_scale"]) if "object_scale" in data else 1.0)
    mesh.export(out / f"score_input/meshes/episode_{episode:06d}/{obj}.obj")


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
    args = parser.parse_args()

    if args.bundle is None:
        if args.episode is None or not args.object:
            parser.error("give --episode and --object, or --bundle")
        args.bundle = bundle_dir_for(args.episode, args.object)
    seq = args.sequence_id or (f"ep{args.episode}_{args.object}" if args.episode is not None else None)
    if not seq:
        parser.error("--sequence-id is required with --bundle and no --episode")
    out = (args.out or OUTPUTS / "load_retarget" / f"episode_{args.episode:06d}" / args.object).resolve()
    bundle = args.bundle.resolve()
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
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

    print(f"== check bundle {bundle}", flush=True)
    report["bundle_check"] = check_bundle(bundle, args.object)
    for w in report["bundle_check"]["warnings"]:
        print(f"  ! {w}")

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

    print("== object URDF", flush=True)
    if not (processed / f"{seq}_rigid.urdf").is_file():
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
    report["output_check"] = check_output(partition(processed, seq), processed / f"{seq}.obj", usda,
                                          report["bundle_check"]["frames"], args.fps)
    for w in report["output_check"]["warnings"]:
        print(f"  ! {w}")
    if args.episode is not None and args.object:
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
