"""Path B outputs to and from the team's Hugging Face dataset, under path_b/.

Layout in rose-tea-cafe/v2d-t3-recon (tools/track3/bundles.py reads only top-level episode_*
folders, so this folder does not affect it):
  path_b/runs/<run>/MANIFEST.json                 what was uploaded, from which commit, sizes, checksums
  path_b/runs/<run>/episode_XXXXXX/...             results: stage reports, loss reports, scales, planes,
                                                   alignment, exported bundles, stages 2-3 outputs,
                                                   waterfall, HTML pages
  path_b/runs/<run>/episode_XXXXXX/frames.tar      only with --frames: the per-frame intermediates
                                                   (rectified pairs, undistorted ego_cam_a, stereo
                                                   depth, and any folder of more than 50 files: v3's
                                                   masks, depth priors, hand stages, poses), one
                                                   archive per episode instead of thousands of files
<run> is a folder name under path_b/data/outputs/ (the commit a run was made with). Restoring puts
everything back there, so `run.py run --out` keeps the stages whose reports exist. An upload with
--episodes sends those episodes (and the run-level files) only and merges MANIFEST.json with the one
on Hugging Face, so a run can go up one episode at a time as each finishes (vm/phase_b.sh).

Usage (repo root; needs `hf auth login`, with a Write token to upload):
    uv run --project tools python path_b/vm/hf_outputs.py upload --run 4c2eac31 --frames
    uv run --project tools python path_b/vm/hf_outputs.py upload --run 0fa525bc
    uv run --project tools python path_b/vm/hf_outputs.py upload --run v4 --frames --episodes 41  # one episode
    uv run --project tools python path_b/vm/hf_outputs.py list
    uv run --project tools python path_b/vm/hf_outputs.py download --run 4c2eac31 [--frames] [--episodes 40 26]
    uv run --project tools python path_b/vm/hf_outputs.py pending --run t1_f238fec   # not on HF yet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import tarfile
from datetime import datetime, timezone
from pathlib import Path

PATH_B = Path(__file__).resolve().parents[1]
OUTPUTS = PATH_B / "data/outputs"
STAGING = PATH_B / "data/hf_staging"
REPO_ID = os.environ.get("V2D_BUNDLE_REPO", "rose-tea-cafe/v2d-t3-recon")
PREFIX = "path_b/runs"

# Per-frame intermediates (go into frames.tar, or are left out) and scratch copies (left out):
FRAMES = {("p1_calibrate", "rect_left"), ("p1_calibrate", "rect_right"), ("p1_calibrate", "cam_a"),
          ("p2_depth", "depth_rect"), ("p2_depth", "intrinsics_rect"), ("p2_depth", "depth_a_half")}
SCRATCH = {("p3_scale_bridge", "merged"), ("episode", "waterfall"), ("run", "intermediate"),
           ("run", "bundle_cleaned")}
CROWDED = 50  # files in one folder: more is a per-frame folder (v3's masks, priors, poses) -> frames.tar


def kind(rel: Path) -> str:
    """frames, scratch or result, for a file path relative to its episode folder."""
    parts = ("episode",) + rel.parts
    for a, b in zip(parts, parts[1:-1]):
        if (a, b) in SCRATCH:
            return "scratch"
        if (a, b) in FRAMES:
            # the half-resolution depth's intrinsics are a result: Stage 2 needs them with the depth
            return "result" if rel.name == "intrinsics.json" and b == "depth_a_half" else "frames"
    return "result"


def crowded(root: Path) -> set[Path]:
    """Folders under root with more than CROWDED entries: per-frame folders, archived like FRAMES."""
    return {d for d in root.rglob("*") if d.is_dir() and sum(1 for _ in d.iterdir()) > CROWDED}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def stage(run: str, frames: bool, episodes: list[int] | None = None) -> tuple[Path, dict]:
    """Build STAGING/<run>/path_b/runs/<run>/ (hard links for results, one tar per episode); with
    `episodes`, those episodes only (and the run-level files)."""
    src = OUTPUTS / run
    if not src.is_dir():
        raise SystemExit(f"no run folder {src}")
    root = STAGING / run
    if root.exists():
        shutil.rmtree(root)
    dest = root / PREFIX / run
    manifest = {"run": run, "repo": REPO_ID, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "host": socket.gethostname(), "commits": set(), "episodes": [], "results": {"files": 0, "bytes": 0},
                "frames": {}, "left_out": ["symlinks", *sorted("/".join(p) for p in SCRATCH)],
                "frames_included": frames, "results_by_episode": {}}
    for episode in sorted(src.glob("episode_*")):
        if episodes is not None and int(episode.name[len("episode_"):]) not in episodes:
            continue
        manifest["episodes"].append(episode.name)
        before = dict(manifest["results"])
        frame_files = []
        per_frame = crowded(episode)
        for path in sorted(p for p in episode.rglob("*") if p.is_file() and not p.is_symlink()):
            rel = path.relative_to(episode)
            k = kind(rel)
            if k == "result" and path.parent in per_frame:
                k = "frames"
            if k == "scratch":
                continue
            if k == "frames":
                frame_files.append(path)
                continue
            target = dest / episode.name / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
            manifest["results"]["files"] += 1
            manifest["results"]["bytes"] += path.stat().st_size
            if path.name == "report.json":
                commit = json.loads(path.read_text()).get("commit", {})
                if commit.get("sha"):
                    manifest["commits"].add(commit["sha"][:8] + ("-dirty" if commit.get("dirty") else ""))
        if frames and frame_files:
            tar_path = dest / episode.name / "frames.tar"
            tar_path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "w") as tar:  # PNG and JPEG are compressed already
                for path in frame_files:
                    tar.add(path, arcname=str(path.relative_to(episode)))
            manifest["frames"][episode.name] = {"files": len(frame_files), "bytes": tar_path.stat().st_size,
                                                "sha256": sha256(tar_path)}
        manifest["results_by_episode"][episode.name] = {k: manifest["results"][k] - before[k] for k in before}
    # run-level files and folders, e.g. the waterfall summary, A2's canonical/<object>/ meshes
    for path in sorted(p for p in src.rglob("*") if p.is_file() and not p.is_symlink()
                       and not p.relative_to(src).parts[0].startswith("episode_")):
        (dest / path.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
        os.link(path, dest / path.relative_to(src))
        manifest["results"]["files"] += 1
        manifest["results"]["bytes"] += path.stat().st_size
    manifest["commits"] = sorted(manifest["commits"])
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return root, manifest


def merged_manifest(remote: dict | None, new: dict) -> dict:
    """The run's MANIFEST.json after uploading `new`'s episodes on top of `remote` (None: no run there
    yet): episodes, frame archives, per-episode results and commits combined, totals recomputed."""
    if not remote:
        return new
    by = {**remote.get("results_by_episode", {}), **new["results_by_episode"]}
    run_level = {k: new["results"][k] - sum(v[k] for v in new["results_by_episode"].values()) for k in ("files", "bytes")}
    if "results_by_episode" not in remote:  # an older whole-run manifest: its totals stand for its episodes
        by = {**{e: {"files": 0, "bytes": 0} for e in remote["episodes"]}, **by}
        run_level = {k: run_level[k] + remote["results"][k] for k in run_level}
    out = dict(new)
    out.update(episodes=sorted(set(remote["episodes"]) | set(new["episodes"])),
               frames={**remote.get("frames", {}), **new["frames"]},
               commits=sorted(set(remote["commits"]) | set(new["commits"])), results_by_episode=by,
               results={k: run_level[k] + sum(v[k] for v in by.values()) for k in ("files", "bytes")},
               frames_included=bool(remote.get("frames_included")) or new["frames_included"])
    return out


def remote_manifest(run: str) -> dict | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    try:
        return json.loads(Path(hf_hub_download(REPO_ID, f"{PREFIX}/{run}/MANIFEST.json", repo_type="dataset")).read_text())
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None


def upload(run: str, frames: bool, episodes: list[int] | None = None) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    root, manifest = stage(run, frames, episodes)
    size = manifest["results"]["bytes"] + sum(f["bytes"] for f in manifest["frames"].values())
    print(f"staged {run}: {len(manifest['episodes'])} episodes, {manifest['results']['files']} result files, "
          f"{len(manifest['frames'])} frame archives, {size / 1e9:.2f} GB", flush=True)
    sent = f"{len(manifest['episodes'])} episodes"
    if episodes is not None:  # one episode at a time: keep what the run already has there
        manifest = merged_manifest(remote_manifest(run), manifest)
        (root / PREFIX / run / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
        sent = f"episodes {', '.join(str(e) for e in episodes)}; {len(manifest['episodes'])} in the run"
    message = f"Path B outputs: run {run} ({sent}" + (", results + per-frame archives" if frames else ", results") + ")"
    # upload_folder skips files whose content is already in the repo, so a rerun resumes
    commit = api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=str(root),
                               commit_message=message, allow_patterns=[f"{PREFIX}/{run}/**"])
    print(f"uploaded: {commit.commit_url if hasattr(commit, 'commit_url') else commit}")
    shutil.rmtree(root)


def list_runs() -> None:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    manifests = [f for f in api.list_repo_files(REPO_ID, repo_type="dataset")
                 if f.startswith(PREFIX + "/") and f.endswith("/MANIFEST.json")]
    for name in sorted(manifests):
        m = json.loads(Path(hf_hub_download(REPO_ID, name, repo_type="dataset")).read_text())
        frames = sum(f["bytes"] for f in m["frames"].values()) / 1e9
        print(f"{m['run']}: {len(m['episodes'])} episodes, commits {', '.join(m['commits'])}, "
              f"results {m['results']['bytes'] / 1e6:.0f} MB" + (f", frames {frames:.1f} GB" if m["frames"] else "")
              + f", uploaded {m['created_utc']}")


def download(run: str, frames: bool, episodes: list[int] | None = None) -> None:
    from huggingface_hub import snapshot_download
    patterns = ([f"{PREFIX}/{run}/MANIFEST.json"] + [f"{PREFIX}/{run}/episode_{e:06d}/**" for e in episodes]
                if episodes else [f"{PREFIX}/{run}/**"])
    ignore = [] if frames else [f"{PREFIX}/{run}/**/frames.tar"]
    tmp = STAGING / f"download_{run}"
    snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=patterns, ignore_patterns=ignore,
                      local_dir=str(tmp))
    src = tmp / PREFIX / run
    manifest = json.loads((src / "MANIFEST.json").read_text())
    dest = OUTPUTS / run
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = path.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "frames.tar":
            expected = manifest["frames"][rel.parts[0]]["sha256"]
            if sha256(path) != expected:
                raise SystemExit(f"checksum mismatch: {rel}")
            with tarfile.open(path) as tar:
                tar.extractall(target.parent, filter="data")
        elif rel.name != "MANIFEST.json":
            shutil.copy2(path, target)
    shutil.copy2(src / "MANIFEST.json", dest / "MANIFEST.json")
    shutil.rmtree(tmp)
    print(f"restored {run} into {dest}: " + (f"episodes {episodes}" if episodes else f"{len(manifest['episodes'])} episodes")
          + (", with per-frame intermediates" if frames and manifest["frames"] else ", results only"))


def uncovered(src: Path, remote: dict[str, int]) -> list[str]:
    """Run-relative paths of src's regular files that Hugging Face does not hold, given remote
    {run-relative path: size}: results missing there or of another size, frames whose episode has no
    frames.tar there. Scratch copies are left out of uploads on purpose and not listed. With an empty
    remote (the run was never uploaded) every regular file is listed."""
    out = []
    per_frame = crowded(src)
    for path in sorted(p for p in src.rglob("*") if p.is_file() and not p.is_symlink()):
        rel = path.relative_to(src)
        if remote and rel.parts[0].startswith("episode_") and len(rel.parts) > 1:
            k = kind(Path(*rel.parts[1:]))
            if k == "result" and path.parent in per_frame:
                k = "frames"
            if k == "scratch" or (k == "frames" and f"{rel.parts[0]}/frames.tar" in remote):
                continue
        if remote.get(str(rel)) != path.stat().st_size:
            out.append(str(rel))
    return out


def pending(run: str) -> None:
    """Print the files of a local run that Hugging Face does not hold (uncovered)."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError
    from huggingface_hub.hf_api import RepoFile
    src = OUTPUTS / run
    if not src.is_dir():
        raise SystemExit(f"no run folder {src}")
    try:
        tree = HfApi().list_repo_tree(REPO_ID, path_in_repo=f"{PREFIX}/{run}", repo_type="dataset", recursive=True)
        remote = {f.path[len(PREFIX) + len(run) + 2:]: f.size for f in tree if isinstance(f, RepoFile)}
    except EntryNotFoundError:
        remote = {}
    for rel in uncovered(src, remote):
        print(rel)


def main() -> None:
    global OUTPUTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("upload", "download", "list", "pending"))
    parser.add_argument("--run", help="folder under path_b/data/outputs/, e.g. 4c2eac31")
    parser.add_argument("--frames", action="store_true", help="include the per-frame intermediates (frames.tar)")
    parser.add_argument("--episodes", type=int, nargs="+",
                        help="upload / download: only these episodes of the run (an upload merges MANIFEST.json)")
    parser.add_argument("--outputs", type=Path, help=f"the runs' folder (default {OUTPUTS}); lets "
                        "save_instance.sh run the Mac's copy of this script against an instance's checkout")
    args = parser.parse_args()
    if args.outputs:
        OUTPUTS = args.outputs.resolve()
    if args.command != "list" and not args.run:
        parser.error("--run is required")
    {"upload": lambda: upload(args.run, args.frames, args.episodes), "download": lambda: download(args.run, args.frames, args.episodes),
     "list": list_runs, "pending": lambda: pending(args.run)}[args.command]()


if __name__ == "__main__":
    main()
