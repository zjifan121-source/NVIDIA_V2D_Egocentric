"""S6 select: the rollout an episode submits (V3_BUILD_SPEC.md S6; rules in pathb/selection.py).

Public: official metrics per rollout against ground truth (pathb/gt.py), meshes as S5's runner wrote
them for scoring; best AUC, ties to the lower MPPE. Acceptance (public): the selected rollout beats
the still rollout's AUC, else FAIL (S0.1's decision rule applies). The evaluation rule also runs on
public episodes, so its choice can be checked against the score. Evaluation: the observation cost
(selection.py) only; never ground truth. Out: s6_select/selection.json {candidate, reason, rollout,
scores, costs, rule}.
"""
from __future__ import annotations

import json

import numpy as np
import trimesh

from .. import gt, phase2, selection, symmetry
from ..report import Metric, StageResult
from ..runner import write_json
from ..team import scorer


def observation(ctx) -> tuple[np.ndarray, np.ndarray, list]:
    """S4b keyframe reference (n, k, 4, 4), its observed frames (n, k), symmetry groups per object."""
    refs, observed, groups = [], [], []
    for name in ctx.objects:
        refs.append(np.load(ctx.root / f"s4b_reference/{name}/keyframe/result.npz")["object_to_world_transform"]
                    .astype(np.float64))
        how = json.loads((ctx.root / f"s4b_reference/{name}/reference_report.json").read_text())["candidates"][
            "keyframe"]["how"]
        observed.append(np.isin(how, selection.OBSERVED))
        mesh = trimesh.load(ctx.root / f"s3a_mesh/{name}/mesh.obj", force="mesh", process=False)
        groups.append(symmetry.group(name, np.asarray(mesh.vertices)))
    return np.stack(refs, axis=1), np.stack(observed, axis=1), groups


def run(ctx) -> StageResult:
    d = ctx.dir("s6_select")
    rollouts = {}
    for cand in ("keyframe", "tracking", "still"):
        f = ctx.root / f"s6_eval/{cand}/results/episode_{ctx.number:06d}.parquet"
        if f.is_file():
            rollouts[cand] = (f, scorer().evaluator_poses(f, ctx.episode))
    if not rollouts:
        raise RuntimeError("no rollout (s6_eval/<candidate>/results)")
    reference, observed, groups = observation(ctx)
    two = phase2.enabled(ctx)
    costs = {c: selection.observation_cost(selection.matrices(p), reference, observed, groups, relative=two)
             for c, (_, p) in rollouts.items()}
    by_observation = selection.pick_evaluation(costs)
    out = {"costs": costs, "evaluation_rule_pick": by_observation[0]}
    metrics = [Metric(f"{c}: observation cost", v["cost_m"] * 100 if v["cost_m"] is not None else None, "cm",
                      note=f"{v['frames']} observed frames, {v['termination_frames']} termination-equivalent")
               for c, v in costs.items()]
    if ctx.public:
        scores = {}
        for cand, (_, poses) in rollouts.items():
            meshes = {name: ctx.root / f"s5_task/{cand}/run/score_input/meshes/episode_{ctx.number:06d}/{name}.obj"
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
        out["rule"] = "evaluation: observation cost (S4b keyframe reference on held and trusted frames)"
    out.update(candidate=choice, reason=reason, rollout=rollouts[choice][0])
    write_json(d / "selection.json", out)
    metrics.insert(0, Metric("selected", choice, "", note=reason))
    return StageResult(metrics, {"selection": d / "selection.json", "rollout": rollouts[choice][0]})
