"""Stages 2-3 on Path B's bundles with the team's runner (tools/track3/run_load_retarget.py, default
repairs), then the reconstruction-only score of what it would hand FlashCHORD (public episodes).

Needs RETARGET_VENV (README §5.12). One object: the runner on that object's bundle. Two-object
episodes are skipped for now: v2's merge runner is parked in cache/v2/, and v3 assembles them with
run_load_retarget.py --merge in S5 mode A (V3_BUILD_SPEC.md §5, task 15).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np

from ..bundle import p4_bundle
from ..paths import PATH_B, REPO
from ..report import Metric, StageResult
from .. import gt


def command(ctx, venv: str, out) -> list[str]:
    """The runner; on evaluation episodes through drivers/run_lr.py, so its bundle check reads no scan."""
    py = f"{venv}/bin/python"
    name = ctx.objects[0]
    bundle = p4_bundle(ctx, name)
    script = REPO / "tools/track3/run_load_retarget.py" if ctx.public else PATH_B / "pathb/drivers/run_lr.py"
    return [py, str(script), "--bundle", str(bundle), "--episode", str(ctx.number), "--object", name, "--out", str(out)]


def worst_seating(value) -> float | None:
    """frame0_seating_cm is one number for one object and a list for a merged scene."""
    values = [v for v in (value if isinstance(value, list) else [value]) if v is not None]
    return max(values, key=abs) if values else None


def run(ctx) -> StageResult:
    if len(ctx.objects) > 1:
        return StageResult([], skip="two-object scene: assembled with run_load_retarget.py --merge in S5 mode A "
                                    "(task 15); v2's merge runner is parked in cache/v2/")
    venv = os.environ.get("RETARGET_VENV")
    if not venv:
        raise RuntimeError("set RETARGET_VENV to the stages 2-3 environment (README §5.12)")
    d = ctx.dir("s23_load_retarget")
    out = d / "run"
    if out.exists():
        shutil.rmtree(out)  # the bundle changed, or this stage would have been kept
    with open(d / "stdout.log", "w") as log:
        result = subprocess.run(command(ctx, venv, out), cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join((d / "stdout.log").read_text().splitlines()[-15:])
        raise RuntimeError(f"run_load_retarget failed (exit {result.returncode}):\n{tail}")
    if ctx.public:
        ctx.files_read.extend(p for p in (gt.scan(n) for n in ctx.objects) if p)  # its bundle check reads the scans
    report = json.loads((out / "report.json").read_text())
    check = report["output_check"]
    task = check.get("task_error_cm", {})
    seating = worst_seating(check["frame0_seating_cm"])
    metrics = [
        Metric("frames out minus manifest", check["frames"] - int(ctx.episode["num_frames"]), "frames", ok=(0, 0)),
        *[Metric(f"robot {side} fingertip error (mean)", task[side]["mean"], "cm", ok=(0, 2.5), warn=(0, 4),
                 note="runner's bar ~2.2") for side in ("left", "right") if side in task],
        *[Metric(f"robot {side} fingertip error (max)", task[side]["max"], "cm", ok=(0, 6), warn=(0, 10))
          for side in ("left", "right") if side in task],
        Metric("frames with a link > 2 cm inside an object", check["penetration_frames"]["over_2cm"], "",
               ok=(0, 0), warn=(0, 0.02)),
        Metric("support disks", check["support_disks"], "", ok=(1, np.inf)),
        Metric("frame-0 seating gap (worst object)", seating, "cm", ok=(-1, 1), warn=(-2, 2),
               note="object bottom minus the support top under it; None: no disk under an object"),
        Metric("runner warnings", len(report["bundle_check"]["warnings"]) + len(check["warnings"]), "",
               note="; ".join(report["bundle_check"]["warnings"] + check["warnings"])),
    ]
    if report.get("merge"):
        metrics.append(Metric("merged scene world from", report["merge"]["world_from"], "",
                              note=f"camera path lengths (m): {report['merge']['camera_path_m']}"))
    outputs = {"report": out / "report.json", "processed": report["processed"], "video": report["video"]}
    if ctx.public:
        poses = np.load(out / f"score_input/results/episode_{ctx.number:06d}.npy")
        meshes = {name: out / f"score_input/meshes/episode_{ctx.number:06d}/{name}.obj" for name in ctx.objects}
        scores = gt.score(ctx.episode, poses, meshes, ctx.files_read)
        outputs["scores"] = scores
        metrics += [Metric(f"reconstruction-only {k}", v, "", kind="gt",
                           note="the trajectory FlashCHORD is trained to track: its ceiling") for k, v in scores.items()]
    return StageResult(metrics, outputs)
