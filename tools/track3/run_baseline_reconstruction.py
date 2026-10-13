"""Run NVIDIA's ego reconstruction baseline on one Track 3 episode, once per object.

The baseline tracks a single object per run, so an episode with two objects is two runs that
share nothing (each recomputes camera, depth and SLAM). Outputs go to
baseline/reconstruction/data/outputs/<branch>/episode_XXXXXX/<object>/, next to a run.json
recording the command, git commit and settings. Steps whose outputs exist are skipped by the
pipeline itself, so re-running resumes.

Run with the reconstruction host environment (README §5.2), from anywhere:
    baseline/reconstruction/.venv/bin/python tools/track3/run_baseline_reconstruction.py \
        --episode 0 --reference-frame 30 [--object white_pot] [--dry-run]
On Apptainer hosts, export V2D_CONTAINER_RUNTIME / V2D_APPTAINER_IMAGE_DIR first (README §5.10).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECON = REPO / "baseline/reconstruction"
MANIFEST = Path(__file__).with_name("episodes.json")

# Baseline settings from README §5.7 (HaMeR hands, SAM3D mesh, SLAM, gravity, gsplat).
BASELINE_FLAGS = ["--undistort", "--hand_tracking", "hamer", "--run_droid_slam", "--run_gravity_alignment",
                  "--run_gsplat_refinement", "--export_threejs_result"]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--reference-frame", type=int, required=True,
                        help="frame where both hands and the object are visible (the run fails otherwise)")
    parser.add_argument("--object", action="append", help="only these object(s); default: all in the episode")
    parser.add_argument("--branch", default=None, help="output namespace (default: current git branch)")
    parser.add_argument("--dev", action="store_true", help="mount local module code (not for FoundationPose on Apptainer)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    episodes = {e["episode"]: e for e in json.loads(MANIFEST.read_text())["episodes"]}
    if args.episode not in episodes:
        parser.error(f"episode {args.episode} is not in {MANIFEST.name}")
    episode = episodes[args.episode]
    if not 0 <= args.reference_frame < episode["num_frames"]:
        parser.error(f"--reference-frame must be in [0, {episode['num_frames']})")
    objects = [o for o in episode["objects"] if not args.object or o["name"] in args.object]
    if not objects:
        parser.error(f"none of {args.object} are in episode {args.episode}: {[o['name'] for o in episode['objects']]}")

    branch = args.branch or git("rev-parse", "--abbrev-ref", "HEAD")
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    video = REPO / episode["video"]
    status = 0
    for obj in objects:
        out = RECON / "data/outputs" / branch / f"episode_{args.episode:06d}" / obj["name"]
        cmd = [sys.executable, "modules/v2d_pipelines/run_ego_reconstruction.py",
               "--video", str(video), "--object_prompt", obj["prompt"], "--output_dir", str(out),
               "--reference_frame", str(args.reference_frame), *BASELINE_FLAGS]
        if args.dev:
            cmd.append("--dev")
        print(f"\n== episode {args.episode} / {obj['name']} (prompt {obj['prompt']!r}) -> {out}", flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        record = {"episode": args.episode, "split": episode["split"], "object": obj["name"],
                  "prompt": obj["prompt"], "reference_frame": args.reference_frame, "command": cmd,
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
