"""S6 eval: each trained policy through the kit's evaluator (V4_BUILD_SPEC.md S6;
examples/run_policy_evaluation.py by pathb/policy.py: evaluation.world_count=1, reset_mode=explicit,
object_trajectories_output), every finished s6_train run (the 60M candidates and the 250M runs). Out per run: s6_eval/<candidate>/results/episode_XXXXXX.parquet
(the rollout the packer takes) and .json (FlashCHORD's metrics against the candidate's own reference).
"""
from __future__ import annotations

import json

from .. import policy
from ..reference import CANDIDATES
from ..report import Metric, StageResult
from ..team import scorer


def trained_runs(ctx) -> list[str]:
    """s6_train's finished runs: the candidates in S4b's order, then the long runs."""
    have = sorted(p.parent.name for p in ctx.root.glob("s6_train/*/train.json"))
    return [c for c in CANDIDATES if c in have] + [c for c in have if c not in CANDIDATES]


def run(ctx) -> StageResult:
    d = ctx.dir("s6_eval")
    metrics, outputs = [], {}
    for cand in trained_runs(ctx):
        trained = ctx.root / f"s6_train/{cand}/train.json"
        results = policy.evaluate(json.loads(trained.read_text())["checkpoint"], ctx.number, d / cand)
        rollout = results / f"episode_{ctx.number:06d}.parquet"
        scorer().evaluator_poses(rollout, ctx.episode)  # raises unless every video frame has a pose
        m = json.loads((results / f"episode_{ctx.number:06d}.json").read_text()).get("metrics", {})
        outputs[cand] = rollout
        metrics += [
            Metric(f"{cand}: rollout covers every frame", "yes", ""),
            Metric(f"{cand}: object position error vs its reference (mean)",
                   m["object_position_error_m"] * 100 if m.get("object_position_error_m") is not None else None, "cm",
                   note="FlashCHORD's metric; the reference is the candidate's own task"),
            Metric(f"{cand}: object orientation error vs its reference (mean)", m.get("object_orientation_error_deg"),
                   "deg"),
            Metric(f"{cand}: FlashCHORD ADD-AUC vs its reference", m.get("add_auc"), ""),
        ]
    if not outputs:
        raise RuntimeError("no trained policy (s6_train/<candidate>/train.json)")
    return StageResult(metrics, outputs)
