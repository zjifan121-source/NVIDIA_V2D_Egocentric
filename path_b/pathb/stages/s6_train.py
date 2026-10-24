"""S6 train: one FlashSAC policy per S5 task (V4_BUILD_SPEC.md S6), by pathb/policy.py: the still
submission's recipe (tools/track3/jobs/still_submit.sh STEPS and EXTRA: 60,006,400 steps, Edmond's
overrides) with sim.njmax / sim.nconmax 512 for the L40S. Candidates in S4b's order (inverted,
keyframe, still); a finished candidate is kept (train.json) and an interrupted one resumes from its
newest checkpoint. Out per candidate: s6_train/<candidate>/train/ (FlashCHORD's run), train.json
{checkpoint, partition, steps, overrides, seconds, nefc_overflow_lines}.
"""
from __future__ import annotations

import json

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


def run(ctx) -> StageResult:
    d = ctx.dir("s6_train")
    found = tasks(ctx)
    if not found:
        raise RuntimeError("no S5 task (s5_task/<candidate>/task.json)")
    metrics, outputs = [], {}
    for cand, task in found.items():
        info = policy.train(task["partition"], d / cand, f"ep{ctx.number}_{cand}")
        outputs[cand] = d / cand / "train.json"
        metrics += [
            Metric(f"{cand}: training time", info["seconds"] / 3600, "h",
                   note=f"{info['steps']:,} steps; mode {task['mode']}" + (", resumed" if info["resumed_from"] else "")),
            Metric(f"{cand}: contact-buffer overflow lines", info["nefc_overflow_lines"], "", ok=(0, 0), warn=(0, np.inf),
                   note="'nefc overflow' in train.log: contacts dropped at njmax 512"),
        ]
    return StageResult(metrics, outputs)
