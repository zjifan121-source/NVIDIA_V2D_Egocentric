"""Path B entrypoint: Track 3 episodes to kit-ready bundles, with a loss report per episode.

Run from the repo root (README in path_b/):
    uv run --project path_b python path_b/run.py run --episodes 41
    uv run --project path_b python path_b/run.py run --split public [--stages p0_ingest p1_calibrate] [--force]
    uv run --project path_b python path_b/run.py waterfall --split public
    uv run --project path_b python path_b/run.py visualize --episodes 13     # after run and waterfall
    uv run --project path_b python path_b/run.py oracle --episodes 41 13 --hands-from <run>  # S0.1 (GPU), our hands
    uv run --project path_b python path_b/run.py oracle --variants place_pos_rot_gt --split public  # S0.2 variants (CPU)
    uv run --project path_b python path_b/run.py wristcheck --out <run>      # S0.3 on the dev episodes (CPU)
    uv run --project path_b python path_b/run.py pack --split evaluation --out <run>  # S7: preflight + pack, no upload (public: dry pack)
    uv run --project path_b python path_b/run.py analysis --out <run> [--tag v2]  # after waterfall; dev episodes by default
    uv run --project path_b python path_b/run.py review --split all --out <run>  # review page (after an all-episode run)
    uv run --project path_b python path_b/run.py canonical --object white_pot --split all --out <run>
                                                    # S3a: after s3a2_fuse_scan, -> <run>/canonical/
Outputs: path_b/data/outputs/<run>/episode_XXXXXX/{<stage>/report.json, run.json, loss_report.md, ...};
<run> defaults to the commit, so code changes never reuse stale results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathb.paths import DEV_EPISODES, OUTPUTS, git_commit, manifest  # noqa: E402
from pathb.runner import STAGES, run_episode  # noqa: E402


# commands of V4_BUILD_SPEC.md §1 whose code comes with a later task (§5)
NOT_BUILT: dict[str, str] = {}
# commands that default to the dev episodes when neither --episodes nor --split is given
DEV_DEFAULT = ("analysis", "wristcheck")


def episodes(args) -> list[int]:
    if args.episodes:
        return args.episodes
    if not args.split:
        return list(DEV_EPISODES)
    return [n for n, e in manifest().items() if args.split in ("all", e["split"])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "waterfall", "visualize", "oracle", "pack", "canonical",
                                            "analysis", "review", "wristcheck"))
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--episodes", type=int, nargs="+")
    which.add_argument("--split", choices=("public", "evaluation", "all"))
    parser.add_argument("--out", type=Path, help="default: path_b/data/outputs/<commit>")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES),
                        help="default: every stage of the chain")
    parser.add_argument("--force", action="store_true", help="rerun stages whose reports exist")
    parser.add_argument("--object", help="canonical: the roster object to build")
    parser.add_argument("--tag", default="", help="analysis: suffix of the page's name, e.g. v2")
    parser.add_argument("--variants", nargs="+", help="oracle: S0.2 variants to score (public only) instead of S0.1")
    parser.add_argument("--hands-from", type=Path, help="oracle (S0.1): the run whose P4 bundles (our P1d hands) and "
                        "partial scans the ground-truth task uses (V4); without it, the team's hub bundle (v3)")
    args = parser.parse_args()
    if args.command == "canonical" and not args.object:
        parser.error("canonical needs --object")
    if not (args.episodes or args.split or args.command in DEV_DEFAULT):
        parser.error(f"{args.command} needs --episodes or --split")
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

    if args.command == "pack":
        from pathb import preflight
        path = preflight.main(out, episodes(args))
        report = json.loads(path.read_text())
        print(f"preflight: {len(report['problems'])} problem(s); packed: {report['packed']}; "
              f"{len(report['mesh_warnings'])} MESH WARNING line(s) -> {path.parent}")
        for problem in report["problems"]:
            print("  " + problem)
        return 0 if report["packed"] else 1

    if args.command == "review":
        from pathb import review
        md, page, ranking = review.main(out, episodes(args))
        print(f"review: {page}\n        {md}\nwhat to fix next, ranked (mean public AUC at stake):")
        for i, r in enumerate(ranking, 1):
            print(f"  {i}. {r['stage']}: {r['auc_at_stake']:.3f} ({len(r['episodes'])} episodes; {r['kind']}: {r['action']})")
        return 0

    if args.command == "analysis":
        from pathb import analysis
        md, page = analysis.main(out, episodes(args), args.tag)
        print(f"analysis: {page}\n          {md}")
        return 0

    if args.command == "wristcheck":
        from pathb import wristcheck
        path = wristcheck.main(out, episodes(args))
        report = json.loads(path.read_text())
        for e in report["episodes"]:
            print(f"   ep{e['episode']}: sigma {e['sigma_cm']} cm (contact {e['sigma_contact_cm']}, jitter {e['jitter_cm']}) "
                  f"over {e['window_frames']} frames; coverage_carry {e['coverage_carry']}")
        print(f"S0.3: {report['decision']}; hand-gap rule {report['hand_gap_rule']} -> {path}")
        return 0

    if args.command == "canonical":
        from pathb import canonical
        canonical.main(args.object, out, episodes(args))
        return 0

    if args.command == "oracle":
        from pathb import oracle
        if args.variants:
            print(f"S0.2 variants: {oracle.variants(episodes(args), args.variants)}")
            return 0
        return oracle.main(episodes(args), args.out, args.hands_from.resolve() if args.hands_from else None)

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
