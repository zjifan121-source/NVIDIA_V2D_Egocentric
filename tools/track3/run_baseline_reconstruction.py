"""Run NVIDIA's ego reconstruction baseline on one Track 3 episode, once per object.

The baseline tracks a single object per run, so an episode with two objects is two runs that
share nothing (each recomputes camera, depth and SLAM). Outputs go to
baseline/reconstruction/data/outputs/<branch>/episode_XXXXXX/<object>/, next to a run.json
recording the command, git commit and settings. Steps whose outputs exist are skipped by the
pipeline itself, so re-running resumes.

Run with the reconstruction host environment (README §5.2), from anywhere:
    baseline/reconstruction/.venv/bin/python tools/track3/run_baseline_reconstruction.py \
        --episode 0 [--reference-frame 30] [--object white_pot] [--dry-run]
The reference frame defaults to reference_frames.json (suggest_reference_frames.py).
--seed <dir> pre-fills each run with stereo_seed.py's calibrated video and stereo-scaled MoGe
depth, so the pipeline skips AnyCalib and MoGe. --scan-mesh tracks the released scan instead of a SAM3D mesh:
public episodes only, as a development ceiling (the FAQ asks for our own meshes on scored ones).
On Apptainer hosts, export V2D_CONTAINER_RUNTIME / V2D_APPTAINER_IMAGE_DIR first (README §5.10).
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

REPO = Path(__file__).resolve().parents[2]
RECON = REPO / "baseline/reconstruction"
SCANS = REPO / "assets/data/track_3/public/mesh"
SEED_ITEMS = ("anycalib", "depth", "intrinsics", "intrinsics_stable.json", "stereo_seed.json", "depth_scale.json")
MANIFEST = Path(__file__).with_name("episodes.json")
REFERENCE_FRAMES = Path(__file__).with_name("reference_frames.json")

# Baseline settings from README §5.7 (HaMeR hands, SAM3D mesh, SLAM, gravity, gsplat).
BASELINE_FLAGS = ["--undistort", "--hand_tracking", "hamer", "--run_droid_slam", "--run_gravity_alignment",
                  "--run_gsplat_refinement", "--export_threejs_result"]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def scan_obj(name: str, out: Path) -> Path:
    """The released scan as a textured OBJ (the pipeline's mesh format), in metres and scan axes."""
    import trimesh
    dst = out / "scan_mesh" / f"{name}.obj"
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        trimesh.load(SCANS / name / f"{name}.glb", force="mesh").export(dst)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--reference-frame", type=int,
                        help="frame where both hands and the object are visible (the run fails otherwise); "
                             "default: the suggestion in reference_frames.json")
    parser.add_argument("--object", action="append", help="only these object(s); default: all in the episode")
    parser.add_argument("--branch", default=None, help="output namespace (default: current git branch)")
    parser.add_argument("--prompt", action="append", default=[], metavar="OBJECT=TEXT",
                        help="override an object's detection prompt for this run (e.g. white_pot='white saucepan with handle')")
    parser.add_argument("--seed", type=Path, help="stereo_seed.py output folder for this episode")
    parser.add_argument("--scan-mesh", action="store_true",
                        help="track the released scan (public episodes only; development ceiling)")
    parser.add_argument("--reregister-iou", type=float,
                        help="re-detect the object when its rendered mask overlaps SAM2's less than this (pipeline default 0.3)")
    parser.add_argument("--late-hands", action="store_true",
                        help="also track a second hand that enters after the reference frame")
    parser.add_argument("--dev", action="store_true", help="mount local module code over the images (as the Hyperstack runs do)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    episodes = {e["episode"]: e for e in json.loads(MANIFEST.read_text())["episodes"]}
    if args.episode not in episodes:
        parser.error(f"episode {args.episode} is not in {MANIFEST.name}")
    episode = episodes[args.episode]
    if args.reference_frame is None:
        suggested = json.loads(REFERENCE_FRAMES.read_text())["episodes"] if REFERENCE_FRAMES.is_file() else {}
        if str(args.episode) not in suggested:
            parser.error("no --reference-frame given and no suggestion in reference_frames.json")
        args.reference_frame = suggested[str(args.episode)]["suggested"]
    if not 0 <= args.reference_frame < episode["num_frames"]:
        parser.error(f"--reference-frame must be in [0, {episode['num_frames']})")
    overrides = dict(p.split("=", 1) for p in args.prompt)
    objects = [dict(o, prompt=overrides.get(o["name"], o["prompt"]))
               for o in episode["objects"] if not args.object or o["name"] in args.object]
    if not objects:
        parser.error(f"none of {args.object} are in episode {args.episode}: {[o['name'] for o in episode['objects']]}")

    if args.scan_mesh and episode["split"] != "public":
        parser.error("--scan-mesh is for public episodes only: the FAQ asks for our own meshes on scored ones")
    if args.seed and not (args.seed / "depth_scale.json").is_file():
        parser.error(f"{args.seed} is not a finished stereo_seed.py output (needs --rescale too)")
    branch = args.branch or git("rev-parse", "--abbrev-ref", "HEAD")
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    video = REPO / episode["video"]
    status = 0
    for obj in objects:
        out = RECON / "data/outputs" / branch / f"episode_{args.episode:06d}" / obj["name"]
        cmd = [sys.executable, "modules/v2d_pipelines/run_ego_reconstruction.py",
               "--video", str(video), "--object_prompt", obj["prompt"], "--output_dir", str(out),
               "--reference_frame", str(args.reference_frame), *BASELINE_FLAGS]
        if args.scan_mesh:
            cmd += ["--object_mesh", str(scan_obj(obj["name"], out)), "--skip_object_scale_estimation"]
        if args.reregister_iou is not None:
            cmd += ["--reregister_iou_thresh", str(args.reregister_iou)]
        if args.late_hands:
            cmd.append("--late_hand_prompts")
        if args.dev:
            cmd.append("--dev")
        print(f"\n== episode {args.episode} / {obj['name']} (prompt {obj['prompt']!r}) -> {out}", flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        if args.seed:
            for item in SEED_ITEMS:  # hard links: one copy of the depth on disk for every run
                src, dst = args.seed / item, out / item
                if dst.exists():
                    continue
                if src.is_dir():
                    shutil.copytree(src, dst, copy_function=os.link)
                else:
                    os.link(src, dst)
        record = {"episode": args.episode, "split": episode["split"], "object": obj["name"],
                  "prompt": obj["prompt"], "reference_frame": args.reference_frame, "command": cmd,
                  "seed": str(args.seed) if args.seed else None, "scan_mesh": args.scan_mesh,
                  "reregister_iou": args.reregister_iou,
                  "git_commit": commit, "git_dirty": dirty, "branch": branch,
                  "container_runtime": os.environ.get("V2D_CONTAINER_RUNTIME", "docker"),
                  "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (out / "run.json").write_text(json.dumps(record, indent=1) + "\n")
        result = subprocess.run(cmd, cwd=RECON)
        record.update(returncode=result.returncode,
                      finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        (out / "run.json").write_text(json.dumps(record, indent=1) + "\n")
        status = status or result.returncode
    return status


if __name__ == "__main__":
    sys.exit(main())
