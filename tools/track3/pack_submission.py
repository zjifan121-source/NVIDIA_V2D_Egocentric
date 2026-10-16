"""Gather per-episode rollouts and meshes into the kit's layout and pack the Track 3 submission.

Each scored episode needs its evaluator output (episode_XXXXXX.parquet + .json, written by
train_episode.sbatch / train_diag.sbatch under <run>/results) and the meshes of the scene that
policy was trained on (<scene>/score_input/meshes/episode_XXXXXX/<object>.obj, written by
run_load_retarget.py). Those meshes are in the rollout's body coordinates, as the kit requires.

The plan is a JSON object {"<episode>": {"run": <run dir>, "scene": <load_retarget out dir>}}.
This script links everything into --work/results and --work/meshes, checks every object of
data/track_3_evaluation_objects.json is covered, then runs the kit's tools/pack_submission.py.

Usage (repo root, with the kit's requirements installed in --python's environment):
    python tools/track3/pack_submission.py --plan plan.json --work <dir> \
        --code-commit https://github.com/<owner>/<repo>/commit/<sha> [--python <kit venv python>]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "assets/data/v2d_submission_kit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--python", default=sys.executable, help="interpreter with the kit's requirements")
    parser.add_argument("-o", "--out", type=Path, help="default: <work>/track3_submission.parquet")
    args = parser.parse_args()

    plan = {int(k): v for k, v in json.loads(args.plan.read_text()).items()}
    roster = {int(k): v for k, v in json.loads((KIT / "data/track_3_evaluation_objects.json").read_text())["episodes"].items()}
    problems = [f"episode {ep}: not in the plan" for ep in sorted(set(roster) - set(plan))]
    problems += [f"episode {ep}: not a scored episode" for ep in sorted(set(plan) - set(roster))]

    results, meshes = args.work / "results", args.work / "meshes"
    for d in (results, meshes):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for ep in sorted(set(plan) & set(roster)):
        stem = f"episode_{ep:06d}"
        run, scene = Path(plan[ep]["run"]), Path(plan[ep]["scene"])
        for suffix in (".parquet", ".json"):
            src = run / "results" / (stem + suffix)
            if src.is_file():
                (results / src.name).symlink_to(src.resolve())
            else:
                problems.append(f"episode {ep}: no {src}")
        src_dir = scene / "score_input/meshes" / stem
        dst_dir = meshes / stem
        dst_dir.mkdir()
        for name in roster[ep]:
            found = [p for p in src_dir.glob(name + ".*") if p.suffix in (".glb", ".obj", ".ply", ".stl", ".off")]
            if not found:
                problems.append(f"episode {ep}: no mesh for {name} in {src_dir}")
        for f in src_dir.iterdir():  # obj + mtl + textures together
            (dst_dir / f.name).symlink_to(f.resolve())
    if problems:
        raise SystemExit("not packing:\n  " + "\n  ".join(problems))

    out = args.out or args.work / "track3_submission.parquet"
    cmd = [args.python, "tools/pack_submission.py", "--track", "track_3", "--results", str(results.resolve()),
           "--meshes", str(meshes.resolve()), "--code-commit", args.code_commit, "-o", str(out.resolve())]
    print("==", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, cwd=KIT).returncode)


if __name__ == "__main__":
    main()
