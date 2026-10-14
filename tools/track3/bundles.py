"""Share reconstruction bundles through one private Hugging Face dataset.

Reconstruction runs on several machines; load and retarget (stages 2-3) need the results in one
place. This uploads only what later stages and reviewers use, not the per-frame intermediates
(RGB, depth, SLAM, masks: GBs per video):

  episode_XXXXXX/<object>/
    result_slam_gravity_aligned/   result.npz, mesh.obj + material, manifest.json (no threejs_scene/)
    run.json, status.json, ...     run records found next to the bundle, if any
    *overlay*.mp4                  diagnostic overlays for visual review (skip with --no-videos)

Downloads land in baseline/reconstruction/data/outputs/hub/, where run_load_retarget.py looks.
Log in once with `hf auth login` (a token with write access to the dataset). The dataset id comes
from --repo or $V2D_BUNDLE_REPO.

    uv run --project tools python tools/track3/bundles.py upload --episode 41 --object white_pot \
        --src baseline/reconstruction/data/outputs/<branch>/episode_000041/white_pot
    uv run --project tools python tools/track3/bundles.py list
    uv run --project tools python tools/track3/bundles.py download --episode 41 [--object white_pot]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HUB = REPO / "baseline/reconstruction/data/outputs/hub"
MANIFEST = Path(__file__).with_name("episodes.json")
BUNDLE = "result_slam_gravity_aligned"
RECORDS = ("run.json", "status.json", "provenance.json", "numeric_qa.json")


def episode_objects(episode: int) -> list[str]:
    episodes = {e["episode"]: e for e in json.loads(MANIFEST.read_text())["episodes"]}
    if episode not in episodes:
        raise SystemExit(f"episode {episode} is not in {MANIFEST.name}")
    return [o["name"] for o in episodes[episode]["objects"]]


def files_to_upload(src: Path, videos: bool) -> list[tuple[Path, str]]:
    """(local file, path inside the object folder). `src` is the run dir or the bundle dir itself."""
    run_dir, bundle = (src.parent, src) if (src / "result.npz").is_file() else (src, src / BUNDLE)
    if not (bundle / "result.npz").is_file():
        raise SystemExit(f"no {BUNDLE}/result.npz under {src}")
    files = [(p, f"{BUNDLE}/{p.relative_to(bundle)}") for p in sorted(bundle.rglob("*"))
             if p.is_file() and "threejs_scene" not in p.relative_to(bundle).parts]
    files += [(run_dir / name, name) for name in RECORDS if (run_dir / name).is_file()]
    # Wentao's runner keeps each attempt's records under runs/<UTC>; keep the latest.
    attempts = sorted((run_dir / "runs").glob("*/provenance.json"))
    if attempts:
        files.append((attempts[-1], "provenance.json"))
    if videos:
        files += [(p, p.name) for p in sorted(run_dir.glob("*overlay*.mp4"))]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=os.environ.get("V2D_BUNDLE_REPO"),
                        help="HF dataset id, e.g. <user>/v2d-t3-recon (default: $V2D_BUNDLE_REPO)")
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upload")
    up.add_argument("--episode", type=int, required=True)
    up.add_argument("--object", required=True)
    up.add_argument("--src", type=Path, required=True, help="reconstruction output dir for this object")
    up.add_argument("--no-videos", action="store_true")
    up.add_argument("--dry-run", action="store_true")
    down = sub.add_parser("download")
    down.add_argument("--episode", type=int, required=True)
    down.add_argument("--object")
    down.add_argument("--dest", type=Path, default=HUB)
    sub.add_parser("list")
    args = parser.parse_args()
    if not args.repo:
        parser.error("give --repo or set V2D_BUNDLE_REPO")

    from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
    api = HfApi()
    if args.command == "upload":
        if args.object not in episode_objects(args.episode):
            raise SystemExit(f"{args.object} is not in episode {args.episode}: {episode_objects(args.episode)}")
        prefix = f"episode_{args.episode:06d}/{args.object}"
        files = files_to_upload(args.src.resolve(), not args.no_videos)
        total = sum(p.stat().st_size for p, _ in files)
        for p, name in files:
            print(f"  {prefix}/{name}  ({p.stat().st_size / 2**20:.1f} MB)")
        print(f"{len(files)} files, {total / 2**20:.1f} MB -> {args.repo}")
        if args.dry_run:
            return
        api.create_commit(repo_id=args.repo, repo_type="dataset",
                          operations=[CommitOperationAdd(f"{prefix}/{name}", str(p)) for p, name in files],
                          commit_message=f"Add {prefix}")
    elif args.command == "download":
        pattern = f"episode_{args.episode:06d}/{args.object or '*'}/**"
        path = snapshot_download(repo_id=args.repo, repo_type="dataset", allow_patterns=[pattern],
                                 local_dir=args.dest)
        print(f"downloaded {pattern} to {path}")
    else:
        present: dict[str, set[str]] = {}
        for name in api.list_repo_files(args.repo, repo_type="dataset"):
            parts = name.split("/")
            if len(parts) > 2 and parts[0].startswith("episode_"):
                present.setdefault(parts[0], set()).add(parts[1])
        for episode in sorted(present):
            print(f"{episode}: {', '.join(sorted(present[episode]))}")


if __name__ == "__main__":
    main()
