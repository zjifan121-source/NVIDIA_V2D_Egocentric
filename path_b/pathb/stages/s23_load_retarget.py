"""Stages 2-3 on Path B's bundle with the team's runner (tools/track3/run_load_retarget.py, default
repairs), then the reconstruction-only score of what it would hand FlashCHORD (public episodes).

Needs RETARGET_VENV (README §5.12). Two-object episodes are skipped for now: the runner merges
objects only from the shared hub (data/outputs/hub), not from Path B's own bundles.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np

from ..paths import REPO
from ..report import Metric, StageResult
from ..team import scorer


def run(ctx) -> StageResult:
    if len(ctx.objects) > 1:
        return StageResult([], skip="two-object scene: run_load_retarget --merge reads only the shared hub; "
                                    "it needs a way to merge Path B's rescaled bundles first")
    venv = os.environ.get("RETARGET_VENV")
    if not venv:
        raise RuntimeError("set RETARGET_VENV to the stages 2-3 environment (README §5.12)")
    name = ctx.objects[0]
    bundle = ctx.root / f"p4_export/episode_{ctx.number:06d}/{name}/result_slam_gravity_aligned"
    d = ctx.dir("s23_load_retarget")
    out = d / "run"
    if out.exists():
        shutil.rmtree(out)  # the bundle changed, or this stage would have been kept
    cmd = [f"{venv}/bin/python", str(REPO / "tools/track3/run_load_retarget.py"), "--bundle", str(bundle),
           "--episode", str(ctx.number), "--object", name, "--out", str(out)]
    with open(d / "stdout.log", "w") as log:
        result = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join((d / "stdout.log").read_text().splitlines()[-15:])
        raise RuntimeError(f"run_load_retarget failed (exit {result.returncode}):\n{tail}")
    report = json.loads((out / "report.json").read_text())
    check = report["output_check"]
    task = check.get("task_error_cm", {})
    metrics = [
        Metric("frames out minus manifest", check["frames"] - int(ctx.episode["num_frames"]), "frames", ok=(0, 0)),
        *[Metric(f"robot {side} fingertip error (mean)", task[side]["mean"], "cm", ok=(0, 2.5), warn=(0, 4),
                 note="runner's bar ~2.2") for side in ("left", "right") if side in task],
        *[Metric(f"robot {side} fingertip error (max)", task[side]["max"], "cm", ok=(0, 6), warn=(0, 10))
          for side in ("left", "right") if side in task],
        Metric("frames with a link > 2 cm inside the object", check["penetration_frames"]["over_2cm"], "",
               ok=(0, 0), warn=(0, 0.02)),
        Metric("support disks under the object", check["support_disks"], "", ok=(1, np.inf)),
        Metric("frame-0 seating gap", check["frame0_seating_cm"], "cm", ok=(0, 1), warn=(0, 2)),
        Metric("runner warnings", len(report["bundle_check"]["warnings"]) + len(check["warnings"]), "",
               note="; ".join(report["bundle_check"]["warnings"] + check["warnings"])),
    ]
    outputs = {"report": out / "report.json", "processed": report["processed"], "video": report["video"]}
    if ctx.public:
        S = scorer()
        poses = np.load(out / f"score_input/results/episode_{ctx.number:06d}.npy")
        mesh = out / f"score_input/meshes/episode_{ctx.number:06d}/{name}.obj"
        scores = S.score([ctx.episode], {ctx.number: poses}, {(ctx.number, name): mesh})
        outputs["scores"] = scores
        metrics += [Metric(f"reconstruction-only {k}", v, "", kind="gt",
                           note="the trajectory FlashCHORD is trained to track: its ceiling") for k, v in scores.items()]
    return StageResult(metrics, outputs)
