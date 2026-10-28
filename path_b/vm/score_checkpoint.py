"""Public score checkpoints: freeze a run's waterfall as a reference, and compare a later run with it.

    uv run --project path_b python path_b/vm/score_checkpoint.py save v4_fix --name public_2026-10-28
    uv run --project path_b python path_b/vm/score_checkpoint.py compare public_2026-10-28 v4_c1 [--episodes 41 18 43]

`save` reads each public episode's waterfall.json of the run (`run.py waterfall`; public ground truth,
scored by the official metric code) and writes path_b/reports/checkpoints/<name>.json: per episode and
candidate the six metrics, the commit and the run. Two means: `ref_inverted` (one candidate for every
episode, chosen without ground truth) and `best of three` (inverted / keyframe / still picked per episode
by AUC against ground truth: an upper bound until F3's GT-free selection exists). The goal is a mean of GOAL
(0.8) over the 20 public episodes. `compare` prints the
same means and the per-episode AUC change of a run against a checkpoint; it reads only the episodes both
have, and says so when they differ. It ends with the consistency check run after every change: episodes up,
down and unchanged, and each one that fell, marked HIGH when it scored at least HIGH before.
Compare only repeatable runs: S3c runs Open3D on one thread since 2026-10-28 (pathb/invert.solve); before that the
same code gave one episode 0.637 and 0.482.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PATH_B = Path(__file__).resolve().parents[1]
OUTPUTS = PATH_B / "data/outputs"
CHECKPOINTS = PATH_B / "reports/checkpoints"
CANDIDATES = ("ref_inverted", "ref_keyframe", "ref_still")
GOAL = 0.8  # James, 2026-10-28: the public mean AUC to reach (the baselines are out of date and not reported)
DROP = 0.005  # default: an episode's best-of-three change smaller than this counts as unchanged
FALLBACK = "v4"  # the S1 run: baselines of an episode a run has no waterfall for
HIGH = 0.7  # default: an episode at or above this before the change is a high scorer (its drop is flagged HIGH)


def rows(run: Path) -> dict[int, dict]:
    """Every public episode of the manifest (20) -> {candidate: metrics}, from the run's
    per-episode waterfalls. An episode the run has no candidate for (ep2: its P1b masks failed, so S3a-S4b never
    ran) is kept with no scores and counts 0 in every mean; its baselines come from the S1 run (FALLBACK)."""
    sys.path.insert(0, str(PATH_B))
    from pathb.paths import manifest
    out = {}
    for n, e in sorted(manifest().items()):
        if e["split"] != "public":
            continue
        f = next((r / f"episode_{n:06d}/waterfall.json" for r in (run, OUTPUTS / FALLBACK)
                  if (r / f"episode_{n:06d}/waterfall.json").is_file()), None)
        w = json.loads(f.read_text()) if f else {"scores": {}, "baselines": {}, "objects": [o["name"] for o in e["objects"]]}
        scores = {c: w["scores"][c] for c in CANDIDATES if (w["scores"].get(c) or {}).get("AUC") is not None} \
            if f is not None and f.parent.parent == run else {}
        out[n] = {"objects": w["objects"], "scores": scores}
    return out


def best(ep: dict) -> tuple[str, float]:
    if not ep["scores"]:
        return "none", 0.0
    c = max(ep["scores"], key=lambda k: ep["scores"][k]["AUC"])
    return c, ep["scores"][c]["AUC"]


def means(eps: dict[int, dict]) -> dict:
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    out = {"episodes scored": f"{sum(1 for e in eps.values() if e['scores'])} of {len(eps)}"}
    out |= {f"{c} AUC": mean([e["scores"].get(c, {}).get("AUC", 0.0) for e in eps.values()]) for c in CANDIDATES}
    out["best of three AUC"] = mean([best(e)[1] for e in eps.values()])  # an episode without candidates: 0
    for k in ("MPPE cm", "MP-SR", "SP-SR"):
        out[f"ref_inverted {k}"] = mean([e["scores"]["ref_inverted"][k] for e in eps.values()
                                         if "ref_inverted" in e["scores"] and e["scores"]["ref_inverted"].get(k) is not None])
    single = {n: e for n, e in eps.items() if len(e["objects"]) == 1}
    out["best of three AUC, single-object"] = mean([best(e)[1] for e in single.values()])
    out["best of three AUC, two-object"] = mean([best(e)[1] for n, e in eps.items() if n not in single])
    return out


def save(run_name: str, name: str) -> None:
    run = OUTPUTS / run_name
    eps = rows(run)
    commit = subprocess.run(["git", "-C", str(PATH_B), "rev-parse", "--short=8", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    doc = {"name": name, "run": run_name, "commit_at_save": commit, "saved": datetime.now(timezone.utc).isoformat(),
           "episodes": sorted(eps), "means": means(eps),
           "per_episode": {str(n): {**e, "best": best(e)[0]} for n, e in sorted(eps.items())}}
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    (CHECKPOINTS / f"{name}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{name}: {len(eps)} episodes from {run_name}")
    for k, v in doc["means"].items():
        print(f"  {k:36s} {v}")


def compare(name: str, run_name: str, episodes: list[int] | None) -> None:
    ref = json.loads((CHECKPOINTS / f"{name}.json").read_text())
    old = {int(n): e for n, e in ref["per_episode"].items()}
    new = rows(OUTPUTS / run_name)
    both = sorted((set(old) & set(new)) if episodes is None else set(episodes) & set(old) & set(new))
    if set(old) != set(new) and episodes is None:
        print(f"episodes only in {name}: {sorted(set(old) - set(new))}; only in {run_name}: {sorted(set(new) - set(old))}")
    print(f"{'ep':>3} {'inv before':>10} {'inv now':>8} {'best before':>12} {'best now':>9} {'change':>7}  candidate")
    for n in both:
        ib, inow = (old[n]["scores"].get("ref_inverted", {}).get("AUC"), new[n]["scores"].get("ref_inverted", {}).get("AUC"))
        (cb, bb), (cn, bn) = best(old[n]), best(new[n])
        fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"{n:>3} {fmt(ib):>10} {fmt(inow):>8} {bb:>12.3f} {bn:>9.3f} {bn - bb:>+7.3f}  {cb} -> {cn}")
    mo, mn = means({n: old[n] for n in both}), means({n: new[n] for n in both})
    print(f"means over {len(both)} episodes ({name} -> {run_name}):")
    for k in mo:
        d = f" ({mn[k] - mo[k]:+.4f})" if isinstance(mo[k], float) and isinstance(mn[k], float) else ""
        print(f"  {k:36s} {mo[k]} -> {mn[k]}{d}")
    # consistency across episodes (James, 2026-10-28): every change is run on all public episodes, and a drop on
    # an episode that already scored high is named even when the mean rises
    change = {n: best(new[n])[1] - best(old[n])[1] for n in both}
    up = [n for n in both if change[n] > DROP]
    down = [n for n in both if change[n] < -DROP]
    print(f"consistency: {len(up)} up, {len(down)} down, {len(both) - len(up) - len(down)} within {DROP} "
          f"(best of three)")
    for n in down:
        high = " HIGH" if best(old[n])[1] >= HIGH else ""
        print(f"  regression{high}: ep{n} {best(old[n])[1]:.3f} -> {best(new[n])[1]:.3f} ({change[n]:+.3f})")
    if not down:
        print("  no episode fell")
    print(f"goal {GOAL}: best of three {mn['best of three AUC']} ({mn['best of three AUC'] - GOAL:+.3f}), "
          f"inverted {mn['ref_inverted AUC']} ({mn['ref_inverted AUC'] - GOAL:+.3f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("save", "compare"))
    p.add_argument("args", nargs="+", help="save: <run>; compare: <checkpoint> <run>")
    p.add_argument("--name", help="save: the checkpoint's name")
    p.add_argument("--episodes", type=int, nargs="+", help="compare: only these episodes")
    a = p.parse_args()
    if a.command == "save":
        save(a.args[0], a.name or a.args[0])
    else:
        compare(a.args[0], a.args[1], a.episodes)


if __name__ == "__main__":
    main()
