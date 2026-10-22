"""Stages 2-3 for a two-object episode on Path B's own bundles, with the team's runner unchanged.

tools/track3/run_load_retarget.py --merge builds one scene from every object of an episode, but it
finds each object's bundle through bundle_dir_for(), which looks only in the shared hub. Here that
one lookup is pointed at Path B's P4 export, in this process, and the runner's main() runs as-is:
merge (steadiest camera path supplies the world), cleaning, MANO load, IK retarget, URDFs, support
surfaces (support_stacked.py with --seat), checks and score input.

Run with the stage 2-3 environment (README §5.12):
    $RETARGET_VENV/bin/python path_b/pathb/merge_runner.py --p4 <episode root>/p4_export \
        -- --episode 13 --merge --out <dir> [other run_load_retarget flags]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLE = "result_slam_gravity_aligned"


def path_b_lookup(p4: Path):
    """bundle_dir_for(episode, object) over Path B's P4 export layout."""
    p4 = Path(p4).resolve()

    def bundle_dir_for(episode: int, obj: str) -> Path:
        path = p4 / f"episode_{episode:06d}" / obj / BUNDLE
        if not (path / "result.npz").is_file():
            raise SystemExit(f"no Path B bundle at {path} (run p4_export first)")
        return path

    return bundle_dir_for


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p4", type=Path, required=True, help="<episode root>/p4_export")
    ap.add_argument("runner_args", nargs=argparse.REMAINDER, help="after --: run_load_retarget.py arguments")
    args = ap.parse_args(argv)
    rest = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    sys.path.insert(0, str(REPO / "tools/track3"))
    import run_load_retarget as R

    R.bundle_dir_for = path_b_lookup(args.p4)  # merged_bundle() reads this module global at call time
    sys.argv = [R.__file__, *rest]
    return R.main()


if __name__ == "__main__":
    sys.exit(main())
