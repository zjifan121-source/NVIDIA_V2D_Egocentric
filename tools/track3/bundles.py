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
    uv run --project tools python tools/track3/bundles.py audit [--split evaluation] [--out report.md]

`audit` checks every object the episode list expects: is it uploaded, are the run records there,
and does it pass the bundle checks of run_load_retarget.py (see tools/track3/BUNDLES.md).
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


def audit_rows(api, repo: str, split: str) -> list[dict]:
    """One row per expected (episode, object): upload state plus run_load_retarget's bundle checks."""
    import sys
    from huggingface_hub import snapshot_download
    sys.path.insert(0, str(Path(__file__).parent))
    from run_load_retarget import check_bundle

    files: dict[str, set[str]] = {}
    for name in api.list_repo_files(repo, repo_type="dataset"):
        parts = name.split("/", 2)
        if len(parts) == 3 and parts[0].startswith("episode_"):
            files.setdefault(f"{parts[0]}/{parts[1]}", set()).add(parts[2])
    rows = []
    episodes = sorted(json.loads(MANIFEST.read_text())["episodes"],
                      key=lambda e: (e["split"] != "evaluation", e["episode"]))  # scored episodes first
    for e in episodes:
        if split != "all" and e["split"] != split:
            continue
        for o in e["objects"]:
            prefix = f"episode_{e['episode']:06d}/{o['name']}"
            have = files.get(prefix, set())
            row = {"episode": e["episode"], "split": e["split"], "object": o["name"],
                   "objects_in_episode": len(e["objects"]), "num_frames": e["num_frames"], "uploaded": bool(have),
                   "run_json": "run.json" in have, "overlay": any("overlay" in f and f.endswith(".mp4") for f in have)}
            core = [f"{BUNDLE}/{f}" for f in ("result.npz", "mesh.obj", "manifest.json")]
            row["missing_files"] = [f for f in core if f not in have] if have else []
            if {f"{BUNDLE}/result.npz", f"{BUNDLE}/mesh.obj"} <= have:
                snapshot_download(repo_id=repo, repo_type="dataset", local_dir=HUB,
                                  allow_patterns=[f"{prefix}/{BUNDLE}/*", f"{prefix}/run.json"])
                check = check_bundle(HUB / prefix / BUNDLE, o["name"])
                if check["frames"] != e["num_frames"]:
                    check["warnings"].insert(0, f"{check['frames']} frames, the video has {e['num_frames']}")
                run_json = HUB / prefix / "run.json"
                if run_json.is_file():
                    row["reference_frame"] = json.loads(run_json.read_text()).get("reference_frame")
                row["check"] = check
            rows.append(row)
    return rows


def verdict(row: dict) -> str:
    """BUNDLES.md section 1: MISSING, REDO (a Must row fails), SHOULD (a Should row fails), OK.
    Missing run.json or overlay is shown in its own column: upload them, no rerun needed."""
    if not row["uploaded"]:
        return "MISSING"
    c = row.get("check")
    if row["missing_files"] or not c:
        return "REDO"
    ratio = c.get("mesh_to_scan_size_ratio")
    scales = [s for s in c["hand_scale_mean"].values() if s == s]  # NaN: hand never seen, a Should row
    if (c["nan_arrays"] or c["frames"] != row["num_frames"] or c["valid_fraction"].get("object", 0) < 0.95
            or (ratio is not None and not 0.8 <= ratio <= 1.25)
            or any(not 0.7 <= s <= 1.5 for s in scales)):
        return "REDO"
    hands_valid = [v for k, v in c["valid_fraction"].items() if k.startswith("hand")]
    if (any(not 0.85 <= s <= 1.3 for s in scales) or any(v < 0.95 for v in hands_valid)
            or c["object_drift_first_30_frames_cm"] > 3):
        return "SHOULD"
    return "OK"


def audit_table(rows: list[dict]) -> str:
    def fmt(v, digits=2):
        return "" if v is None else f"{v:.{digits}f}"
    lines = ["| ep | split | object | state | run.json | overlay | hand scale L/R | mesh/scan | "
             "drift 30f cm | step cm | still from | issues |", "|" + "---|" * 12]
    for r in rows:
        c = r.get("check", {})
        state = verdict(r)
        hs = c.get("hand_scale_mean", {})
        issues = "; ".join(r["missing_files"] + c.get("warnings", []))
        still = "" if not c else ("never" if c["first_still_frame"] is None else c["first_still_frame"])
        lines.append(f"| {r['episode']} | {r['split']} | {r['object']} | {state} | {'yes' if r['run_json'] else 'no'} | "
                     f"{'yes' if r['overlay'] else 'no'} | {fmt(hs.get('left'))}/{fmt(hs.get('right'))} | "
                     f"{fmt(c.get('mesh_to_scan_size_ratio'))} | {fmt(c.get('object_drift_first_30_frames_cm'), 1)} | "
                     f"{fmt(c.get('object_step_median_cm'))} | {still} | {issues} |")
    return "\n".join(lines)


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
    aud = sub.add_parser("audit", help="check every expected object against the BUNDLES.md standards")
    aud.add_argument("--split", choices=("evaluation", "public", "all"), default="all")
    aud.add_argument("--out", type=Path, help="also write the table (.md) and the full checks (.json) here")
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
    elif args.command == "audit":
        rows = audit_rows(api, args.repo, args.split)
        table = audit_table(rows)
        print(table)
        if args.out:
            args.out.write_text(table + "\n")
            args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
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
