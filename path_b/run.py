"""Path B entrypoint: Track 3 episodes to kit-ready bundles, with a loss report per episode.

Run from the repo root (README in path_b/):
    uv run --project path_b python path_b/run.py run --episodes 41
    uv run --project path_b python path_b/run.py run --split public [--stages p0_ingest p1_calibrate] [--force]
    uv run --project path_b python path_b/run.py waterfall --split public
    uv run --project path_b python path_b/run.py visualize --episodes 13     # after run and waterfall
    uv run --project path_b python path_b/run.py oracle --episodes 40 26     # S0.1 (not built yet: task 5)
    uv run --project path_b python path_b/run.py pack --split evaluation     # S7 (not built yet: task 16)
Outputs: path_b/data/outputs/<run>/episode_XXXXXX/{<stage>/report.json, run.json, loss_report.md, ...};
<run> defaults to the commit, so code changes never reuse stale results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathb.paths import OUTPUTS, git_commit, manifest  # noqa: E402
from pathb.runner import STAGES, run_episode  # noqa: E402


# commands of V3_BUILD_SPEC.md §1 whose code comes with a later task (§5)
NOT_BUILT = {"oracle": "S0.1, task 5", "pack": "S7, task 16"}


def episodes(args) -> list[int]:
    if args.episodes:
        return args.episodes
    return [n for n, e in manifest().items() if args.split in ("all", e["split"])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "waterfall", "visualize", "oracle", "pack"))
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--episodes", type=int, nargs="+")
    which.add_argument("--split", choices=("public", "evaluation", "all"))
    parser.add_argument("--out", type=Path, help="default: path_b/data/outputs/<commit>")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--force", action="store_true", help="rerun stages whose reports exist")
    args = parser.parse_args()
    commit = git_commit()
    out = (args.out or OUTPUTS / (commit["sha"][:8] + ("-dirty" if commit["dirty"] else ""))).resolve()
    print(f"outputs: {out}", flush=True)

    if args.command == "run":
        failed = []
        for number in episodes(args):
            report = run_episode(number, out, tuple(args.stages), args.force)
            if report["first_failing_stage"]:
                failed.append(f"{number}: {report['first_failing_stage']}")
        print("first failing stage per episode: " + (", ".join(failed) or "none"))
        return 0

    if args.command in NOT_BUILT:
        print(f"run.py {args.command} is not built yet ({NOT_BUILT[args.command]})", file=sys.stderr)
        return 2

    if args.command == "visualize":
        from pathb import visualize
        for number in episodes(args):
            root = out / f"episode_{number:06d}"
            if not (root / "waterfall.json").is_file():
                print(f"episode {number}: run and waterfall first, skipped", flush=True)
                continue
            print(f"page: {visualize.page(root)}", flush=True)
        return 0

    from pathb import waterfall
    rows = []
    for number in episodes(args):
        root = out / f"episode_{number:06d}"
        if manifest()[number]["split"] != "public":
            print(f"episode {number}: evaluation split, no ground truth: skipped", flush=True)
            continue
        if not root.is_dir():
            print(f"episode {number}: not run yet, skipped", flush=True)
            continue
        print(f"== waterfall episode {number}", flush=True)
        rows.append(waterfall.episode(root))
    if rows:
        print(waterfall.summary(out, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
