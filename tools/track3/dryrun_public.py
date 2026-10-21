"""What would a submission score on the 20 public episodes, built from our own bundles? No simulation.

Per episode, the scene stage 2 would build: merged bundle for two objects (merged_bundle), cleaned
as stage 2 does (hand and object clean, rest hold). Every object of the episode is scored together
with the official metric code (score_local.score), so two-object episodes pay for our relative
placement: Kaggle aligns a scene by the anchor object's frame-0 pose only, and the other object's
offset from it is never corrected. Choices:
  recon        the cleaned reconstruction played back exactly (the ceiling for a tracking policy)
  still        every object held at its frame-0 pose (what a still-object policy delivers)
  recon_scan   recon after rescaling the scene by the mesh/public-scan size ratio (--rescale-by-scan)
  still_scan   still after the same rescale (changes object sizes and the distance between objects)

2026-10-15, 20 public episodes, mean AUC: still 0.309, still_scan 0.329, recon 0.293, recon_scan 0.347.

Usage (repo root; writes merged/cleaned bundles and meshes under --work):
    uv run --project tools python tools/track3/dryrun_public.py --work <dir> [--episodes 0 41]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
import run_load_retarget as L  # noqa: E402
import score_local as S  # noqa: E402

CHOICES = ("still", "still_scan", "recon", "recon_scan")


def scene(e: dict, work: Path, scan: bool) -> tuple[list[np.ndarray], dict]:
    """Cleaned world poses per object (scorer order) and the mesh file each one is scored with."""
    ep, names = e["episode"], [o["name"] for o in e["objects"]]
    src = L.merged_bundle(ep, work / f"e{ep}_merged")[0] if len(names) > 1 else L.bundle_dir_for(ep, names[0])
    tag = "scan" if scan else "plain"
    bundle, _ = L.cleaned_bundle(src, work / f"e{ep}_{tag}", L.TRACK3_FPS, rescale_by_scan=names[0] if scan else None)
    data = dict(np.load(bundle / "result.npz"))
    poses, meshes = [], {}
    for i, (pose_key, scale_key, mesh_file) in enumerate(L.object_slots(data)):
        poses.append(data[pose_key].astype(np.float64))
        mesh = trimesh.load(bundle / mesh_file, force="mesh", process=False)
        mesh.apply_scale(float(data.get(scale_key, 1.0)))
        path = work / f"e{ep}_{tag}_{i}.obj"
        mesh.export(path)
        meshes[(ep, names[i])] = path
    return poses, meshes


def score_episode(e: dict, work: Path) -> dict:
    ep = e["episode"]
    n = len(S.ground_truth(ep))
    out = {"episode": ep, "objects": [o["name"] for o in e["objects"]]}
    for scan in (False, True):
        suffix = "_scan" if scan else ""
        try:
            poses, meshes = scene(e, work, scan)
        except SystemExit as err:  # e.g. no public scan for the object
            out["error" + suffix] = str(err)
            continue
        poses = [T[:n] if len(T) >= n else np.concatenate([T, np.repeat(T[-1:], n - len(T), 0)]) for T in poses]
        recon = np.stack([np.concatenate([T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()], -1)
                          for T in poses], 1)
        out["recon" + suffix] = S.score([e], {ep: recon}, meshes)
        out["still" + suffix] = S.score([e], {ep: np.repeat(recon[:1], n, 0)}, meshes)
    print(json.dumps(out), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    episodes = [e for e in S.public_episodes() if not args.episodes or e["episode"] in args.episodes]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(score_episode, episodes, [args.work] * len(episodes)))
    (args.work / "dryrun.json").write_text(json.dumps(rows, indent=1))
    print("\n| episode | objects | " + " | ".join(CHOICES) + " |\n|---|---|" + "---|" * len(CHOICES))
    for r in rows:
        cells = [f"{r[c]['AUC']:.3f}" if c in r else "" for c in CHOICES]
        print(f"| {r['episode']} | {' + '.join(r['objects'])} | " + " | ".join(cells) + " |")
    means = [np.mean([r[c]["AUC"] for r in rows if c in r]) for c in CHOICES]
    print("| mean AUC | | " + " | ".join(f"{m:.3f}" for m in means) + " |")


if __name__ == "__main__":
    main()
