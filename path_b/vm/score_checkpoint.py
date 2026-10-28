"""Public score checkpoints: freeze a run's waterfall as a reference, and compare a later run with it.

    uv run --project path_b python path_b/vm/score_checkpoint.py save v4_fix --name public_2026-10-28
    uv run --project path_b python path_b/vm/score_checkpoint.py compare public_2026-10-28 v4_c1 [--episodes 41 18 43]

`save` reads each public episode's waterfall.json of the run (`run.py waterfall`; public ground truth,
scored by the official metric code) and writes path_b/reports/checkpoints/<name>.json: per episode and
candidate the six metrics, the commit and the run. Two means: `ref_inverted` (one candidate for every
episode, chosen without ground truth) and `best of three` (inverted / keyframe / still picked per episode
by AUC against ground truth: an upper bound until F3's GT-free selection exists). `compare` prints the
same means and the per-episode AUC change of a run against a checkpoint; it reads only the episodes both
have, and says so when they differ.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PATH_B = Path(__file__).resolve().parents[1]
OUTPUTS = PATH_B / "data/outputs"
CHECKPOINTS = PATH_B / "reports/checkpoints"
CANDIDATES = ("ref_inverted", "ref_keyframe", "ref_still")
EDMOND = "S0.2 recon / place"


def rows(run: Path) -> dict[int, dict]:
    """Episode -> {candidate: metrics} and Edmond's place, from the run's per-episode waterfalls."""
    out = {}
    for f in sorted(run.glob("episode_*/waterfall.json")):
        w = json.loads(f.read_text())
        scores = {c: w["scores"][c] for c in CANDIDATES if (w["scores"].get(c) or {}).get("AUC") is not None}
        if scores:
            out[int(w["episode"])] = {"objects": w["objects"], "scores": scores,
                                      "edmond_place": w["baselines"].get(EDMOND)}
    return out


def best(ep: dict) -> tuple[str, float]:
    c = max(ep["scores"], key=lambda k: ep["scores"][k]["AUC"])
    return c, ep["scores"][c]["AUC"]


def means(eps: dict[int, dict]) -> dict:
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    out = {f"{c} AUC": mean([e["scores"][c]["AUC"] for e in eps.values() if c in e["scores"]]) for c in CANDIDATES}
    out["best of three AUC"] = mean([best(e)[1] for e in eps.values()])
    out["edmond_place AUC"] = mean([e["edmond_place"] for e in eps.values() if e["edmond_place"] is not None])
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
        d = "" if mo[k] is None or mn[k] is None else f" ({mn[k] - mo[k]:+.4f})"
        print(f"  {k:36s} {mo[k]} -> {mn[k]}{d}")


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
