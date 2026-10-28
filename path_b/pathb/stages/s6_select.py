"""S6 select: the rollout an episode submits (V4_BUILD_SPEC.md S6; rules in pathb/selection.py).

Public: official metrics per rollout against ground truth (pathb/gt.py), meshes as S5's runner wrote
them for scoring; best AUC, ties to the lower MPPE. Acceptance (public): the selected rollout beats
the still rollout's AUC, else FAIL (S0.1's decision rule applies). The evaluation rule also runs on
public episodes, so its choice can be checked against the score. Evaluation: the observation cost
(selection.py) only; never ground truth. Out: s6_select/selection.json {candidate, reason, rollout,
scores, costs, rule}. Two objects (V4 §8): RPE breaks public ties before MPPE, and the objects'
relative-position error joins the observation cost.
"""
from __future__ import annotations

import json

import numpy as np
import trimesh

from .. import gt, selection, symmetry
from ..reference import CANDIDATES
from ..report import Metric, StageResult
from ..runner import write_json
from ..team import scorer


PINNED_POSITION_M = 0.10  # V4 S6: failure frames at pinned thresholds, never the checkpoint's
PINNED_ROTATION_RAD = 0.5


def observation(ctx) -> tuple[np.ndarray, np.ndarray, list]:
    """Our own reference (n, k, 4, 4): S4b's `inverted` (keyframe in a v3 run), its observed frames
    (held or solved from the observations: selection.OBSERVED), symmetry groups per object."""
    refs, observed, groups = [], [], []
    for name in ctx.objects:
        report = json.loads((ctx.root / f"s4b_reference/{name}/reference_report.json").read_text())["candidates"]
        ref = "inverted" if "inverted" in report else "keyframe"
        refs.append(np.load(ctx.root / f"s4b_reference/{name}/{ref}/result.npz")["object_to_world_transform"]
                    .astype(np.float64))
        observed.append(np.isin(report[ref]["how"], selection.OBSERVED))
        mesh = trimesh.load(ctx.root / f"s3a_mesh/{name}/mesh.obj", force="mesh", process=False)
        groups.append(symmetry.group(name, np.asarray(mesh.vertices)))
    return np.stack(refs, axis=1), np.stack(observed, axis=1), groups


def task_of(run_name: str) -> str:
    """The S5 task a rollout's policy trained on (inverted_250m_s42 -> inverted)."""
    return run_name.split("_250m")[0]


def run(ctx) -> StageResult:
    d = ctx.dir("s6_select")
    from .s6_eval import trained_runs
    rollouts = {}
    for cand in trained_runs(ctx):
        f = ctx.root / f"s6_eval/{cand}/results/episode_{ctx.number:06d}.parquet"
        if f.is_file():
            rollouts[cand] = (f, scorer().evaluator_poses(f, ctx.episode))
    if not rollouts:
        raise RuntimeError("no rollout (s6_eval/<candidate>/results)")
    reference, observed, groups = observation(ctx)
    two = len(ctx.objects) > 1
    costs = {c: selection.observation_cost(selection.matrices(p), reference, observed, groups, relative=two,
                                           max_position=PINNED_POSITION_M, max_rotation=PINNED_ROTATION_RAD)
             for c, (_, p) in rollouts.items()}
    by_observation = selection.pick_evaluation(costs)
    out = {"costs": costs, "evaluation_rule_pick": by_observation[0]}
    metrics = [Metric(f"{c}: observation cost", v["cost_m"] * 100 if v["cost_m"] is not None else None, "cm",
                      note=f"{v['frames']} observed frames, {v['termination_frames']} termination-equivalent")
               for c, v in costs.items()]
    if ctx.public:
        scores = {}
        for cand, (_, poses) in rollouts.items():
            meshes = {name: ctx.root / f"s5_task/{task_of(cand)}/run/score_input/meshes/episode_{ctx.number:06d}/{name}.obj"
                      for name in ctx.objects}
            scores[cand] = gt.score(ctx.episode, poses, meshes, ctx.files_read)
        choice, reason = selection.pick_public(scores, rpe=two)
        out.update(scores=scores, rule="public: official AUC (ties: MPPE)")
        metrics += [Metric(f"{c}: rollout AUC", s["AUC"], "", kind="gt",
                           note=f"MPPE {s['MPPE cm']:.2f} cm, MP-SR {s['MP-SR']:.2f}, CD-O {s['CD-O cm']:.2f} cm")
                    for c, s in scores.items()]
        if "still" in scores:
            metrics.append(Metric("selected AUC minus the still rollout's", scores[choice]["AUC"] - scores["still"]["AUC"],
                                  "", kind="gt", ok=(1e-9, np.inf),
                                  note="spec S6 acceptance; FAIL -> S0.1's decision rule"))
        metrics.append(Metric("evaluation rule's pick", by_observation[0], "", kind="gt",
                              note="same as the score's" if by_observation[0] == choice else f"score picks {choice}"))
    else:
        choice, reason = by_observation
        trust = selection.reference_trust(ctx.root, ctx.objects)
        out["reference_trust"] = trust
        if not trust["trusted"] and "still" in rollouts:  # the cost is against a reference that lost the object
            choice, reason = "still", f"our inverted reference is untrusted ({trust['why']})"
        out["rule"] = ("evaluation: mean position error against our own reference (S4b inverted) on its observed "
                       f"frames, failure frames at the pinned {PINNED_POSITION_M} m / {PINNED_ROTATION_RAD} rad; "
                       "the still rollout when that reference is untrusted (selection.reference_trust)")
    out.update(candidate=choice, task=task_of(choice), reason=reason, rollout=rollouts[choice][0],
               pinned={"position_m": PINNED_POSITION_M, "rotation_rad": PINNED_ROTATION_RAD})
    write_json(d / "selection.json", out)
    metrics.insert(0, Metric("selected", choice, "", note=reason))
    return StageResult(metrics, {"selection": d / "selection.json", "rollout": rollouts[choice][0]})
