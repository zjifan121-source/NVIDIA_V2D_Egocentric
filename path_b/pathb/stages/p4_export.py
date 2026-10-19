"""P4 export: Path B's bundles in the team's layout (tools/track3/BUNDLES.md §4), one per object.

Each object's HF bundle is rescaled by P3's factor (one factor per episode: the objects share the
head camera). Next to result_slam_gravity_aligned/ go run.json (how it was made), numeric_qa.json
(every P0-P3 measurement) and status.json (verdict and first failing stage), so a bundle carries
its own diagnosis wherever it is copied.
"""
from __future__ import annotations

import json
import shutil
import sys

import numpy as np

from ..bundle import rescaled_copy
from ..paths import git_commit
from ..report import Metric, StageResult, worst
from ..runner import write_json
from ..team import retarget_tools

UPSTREAM = ("p0_ingest", "p1_calibrate", "p2_depth", "p3_scale_bridge")


def run(ctx) -> StageResult:
    L = retarget_tools()
    scale = json.loads((ctx.root / "p3_scale_bridge/scale.json").read_text())
    factor = scale["factor"]
    base = ctx.dir("p4_export") / f"episode_{ctx.number:06d}"
    reports = {name: ctx.reports[name] for name in UPSTREAM if name in ctx.reports}
    qa = {name: r["metrics"] for name, r in reports.items()}
    status = {"state": worst(r["status"] for r in reports.values()),
              "first_failing_stage": next((n for n, r in reports.items() if r["status"] in ("FAIL", "ERROR")), None),
              "stages": {n: r["status"] for n, r in reports.items()}}
    metrics, finite, frames_off = [], True, 0
    for name in ctx.objects:
        src = L.bundle_dir_for(ctx.number, name)
        folder = base / name
        dest = folder / "result_slam_gravity_aligned"
        if dest.exists():
            shutil.rmtree(dest)
        # threejs_scene/ shows the unscaled scene, so it is left behind
        rescaled_copy(src, dest, factor, link=False, skip=("threejs_scene",))
        data = np.load(dest / "result.npz")
        finite &= all(np.isfinite(data[k]).all() for k in data.files if data[k].dtype.kind == "f")
        frames_off = max(frames_off, abs(data["camera_to_world_transform"].shape[0] - int(ctx.episode["num_frames"])))
        vertices = L.object_vertices(dest, dict(data), "object_scale", "mesh.obj")
        metrics.append(Metric(f"{name} mesh extents after rescale",
                              " x ".join(f"{e * 100:.1f}" for e in L.oriented_extents(vertices)), "cm"))
        if ctx.public:
            ratio = L.scan_size_ratio(vertices, name)
            if ratio:
                metrics.append(Metric(f"{name} mesh / scan size after rescale", ratio, "x", kind="gt",
                                      ok=(0.8, 1.25), note="BUNDLES.md Must: 0.8-1.25"))
        write_json(folder / "run.json", {
            "producer": "path_b", "command": sys.argv, "commit": git_commit(), "source_bundle": str(src),
            "scale_factor": factor, "scale_applied": scale["applied"],
            "notes": "Stage 1 bridge: the team's bundle rescaled to stereo metric scale (Path B P3).",
        })
        write_json(folder / "numeric_qa.json", qa)
        write_json(folder / "status.json", status)
    metrics[:0] = [
        Metric("objects exported minus manifest", len(ctx.objects) - len(list(base.iterdir())), "", ok=(0, 0)),
        Metric("frames off the manifest (worst object)", frames_off, "frames", ok=(0, 0)),
        Metric("all arrays finite", int(finite), "", ok=(1, 1)),
        Metric("overlay video", "not made yet", "", note="BUNDLES.md asks for one before an HF upload"),
    ]
    return StageResult(metrics, {"bundles": base, "factor": factor})
