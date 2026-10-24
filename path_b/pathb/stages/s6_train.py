"""S6 train: one FlashSAC policy per S5 task (V4_BUILD_SPEC.md S6), by pathb/policy.py: the still
submission's recipe (tools/track3/jobs/still_submit.sh STEPS and EXTRA: 60,006,400 steps, Edmond's
overrides) with sim.njmax / sim.nconmax 512 for the L40S. Candidates in S4b's order (inverted,
keyframe, still; S5 made tasks for its moving candidate and still); a finished candidate is kept
(train.json) and an interrupted one resumes from its newest checkpoint. Long pass ($PATHB_S6_LONG =
number of seeds, 1 or 2; V4 S6: "then 250M on the selected candidate"): after a first S6 pass, the
candidate s6_select chose is trained again on FlashCHORD's 250M schedule, as <candidate>_250m_s<seed>
(seeds 42, 43); s6_eval and s6_select then take every rollout. Out per run: s6_train/<name>/train/
(FlashCHORD's run), train.json {checkpoint, partition, steps, overrides, seed, long, groups, seconds,
nefc_overflow_lines}.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .. import policy
from ..report import Metric, StageResult
from ..reference import CANDIDATES


def tasks(ctx) -> dict[str, dict]:
    """S5's task per candidate, in candidate order."""
    out = {}
    for cand in CANDIDATES:
        f = ctx.root / f"s5_task/{cand}/task.json"
        if f.is_file():
            out[cand] = json.loads(f.read_text())
    return out


LONG_SEEDS = (42, 43)


def jobs(ctx) -> list[tuple[str, dict, bool, int]]:
    """(run name, S5 task, long, seed) to train: every task at 60M; with $PATHB_S6_LONG, also the
    selected candidate at 250M for that many seeds."""
    found = tasks(ctx)
    out = [(cand, task, False, 42) for cand, task in found.items()]
    seeds = int(os.environ.get("PATHB_S6_LONG") or 0)
    sel = ctx.root / "s6_select/selection.json"
    if seeds and sel.is_file():
        base = json.loads(sel.read_text()).get("task") or json.loads(sel.read_text())["candidate"]
        if base in found:
            out += [(f"{base}_250m_s{seed}", found[base], True, seed) for seed in LONG_SEEDS[:seeds]]
    return out


def run(ctx) -> StageResult:
    d = ctx.dir("s6_train")
    todo = jobs(ctx)
    if not todo:
        raise RuntimeError("no S5 task (s5_task/<candidate>/task.json)")
    metrics, outputs = [], {}
    for cand, task, long, seed in todo:
        info = policy.train(task["partition"], d / cand, f"ep{ctx.number}_{cand}", long=long, seed=seed)
        outputs[cand] = d / cand / "train.json"
        metrics += [
            Metric(f"{cand}: training time", info["seconds"] / 3600, "h",
                   note=f"{info['steps']:,} steps; mode {task['mode']}" + (", resumed" if info["resumed_from"] else "")),
            Metric(f"{cand}: contact-buffer overflow lines", info["nefc_overflow_lines"], "", ok=(0, 0), warn=(0, np.inf),
                   note="'nefc overflow' in train.log: contacts dropped at njmax 512"),
        ]
    return StageResult(metrics, outputs)
