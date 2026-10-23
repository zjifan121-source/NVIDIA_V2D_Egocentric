"""Compare the stage reports of two Path B runs: same metrics, same values (within a tolerance), same
verdicts. Used to check that a code change leaves already-built stages unchanged.

    uv run --project path_b python path_b/vm/compare_runs.py <run A> <run B> --episodes 41 \
        --stages p0_ingest p1_calibrate p2_depth
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def same(a, b, rel: float) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (math.isfinite(a) and math.isfinite(b)):
            return a == b or (math.isnan(a) and math.isnan(b))
        return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)
    return a == b


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--stages", nargs="+", required=True)
    p.add_argument("--rel", type=float, default=1e-3, help="relative tolerance on numeric values")
    args = p.parse_args()
    bad = 0
    for ep in args.episodes:
        for stage in args.stages:
            ra, rb = (json.loads((r / f"episode_{ep:06d}/{stage}/report.json").read_text()) for r in (args.a, args.b))
            ma, mb = ({m["name"]: m for m in r["metrics"]} for r in (ra, rb))
            diffs = [f"status {ra['status']} vs {rb['status']}"] if ra["status"] != rb["status"] else []
            diffs += [f"only in A: {n}" for n in ma.keys() - mb.keys()] + [f"only in B: {n}" for n in mb.keys() - ma.keys()]
            for n in ma.keys() & mb.keys():
                if not same(ma[n]["value"], mb[n]["value"], args.rel) or ma[n]["status"] != mb[n]["status"]:
                    diffs.append(f"{n}: {ma[n]['value']} ({ma[n]['status']}) vs {mb[n]['value']} ({mb[n]['status']})")
            print(f"ep{ep} {stage:16s} {ra['status']:5s} {len(ma)} metrics: " + ("same" if not diffs else "DIFFERENT"))
            for d in diffs:
                print(f"    {d}")
            bad += bool(diffs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
