"""S4a phases: rest before the reach, the carry and rest after, per object (V4_BUILD_SPEC.md S4a).

A view of S3b's segments: s3b_segments/<object>/phases.json (rest-before = before the first grasp,
carry = first grasp start to last grasp end, rest-after = after it). Fallback when S3b FAILed or did not
run: the v3 rule (pathb/phases.py: a wrist within 10 cm of the object surface, dilated 5 frames) on
the object track of S3c's solution (s3c_invert/<object>/object_to_world.npz) when there is one, else
the bundle's. Writes s4a_phases/<object>/phases.json with its `source`. Acceptance: a carry of at
least 10 frames unless the object is static.
"""
from __future__ import annotations

import json

import numpy as np

from .. import phases as P
from ..bundle import p4_bundle
from ..report import Metric, StageResult
from ..runner import write_json
from ..team import retarget_tools


def fallback(ctx, name: str, read: list) -> dict:
    """The v3 rule on S3c's track (or the bundle's)."""
    L = retarget_tools()
    bundle = p4_bundle(ctx, name)
    data = dict(np.load(bundle / "result.npz"))
    read += [bundle / "result.npz", bundle / "mesh.obj"]
    vertices = L.object_vertices(bundle, data, "object_scale", "mesh.obj")
    solved = ctx.root / f"s3c_invert/{name}/object_to_world.npz"
    if solved.is_file():
        data["object_to_world_transform"] = np.load(solved)["object_to_world"]
        read.append(solved)
    source = "v3 wrist rule on " + ("S3c's track" if solved.is_file() else "the bundle's track")
    return P.from_bundle(data, vertices) | {"source": source, "reach_cm": P.REACH_CM}


def run(ctx) -> StageResult:
    metrics, outputs, read = [], {}, []
    s3b = ctx.reports.get("s3b_segments", {}).get("status")
    for name in ctx.objects:
        seg = ctx.root / f"s3b_segments/{name}/phases.json"
        if seg.is_file() and s3b not in (None, "FAIL", "ERROR", "SKIP"):
            r = json.loads(seg.read_text())
            read.append(seg)
        else:
            r = fallback(ctx, name, read) | {"why_fallback": f"s3b_segments {s3b or 'not run'}"}
        path = ctx.dir("s4a_phases") / name / "phases.json"
        path.parent.mkdir(exist_ok=True)
        write_json(path, r | {"object": name, "windows": "inclusive [first, last] frames"})
        outputs[name] = path
        carry = P.length(r["carry"])
        metrics += [
            Metric(f"{name}: first reach", r["first_reach"], "frame",
                   note="never reached: the whole clip is rest" if r["first_reach"] is None else ""),
            Metric(f"{name}: last release", r["last_release"], "frame"),
            Metric(f"{name}: rest-before frames", P.length(r["rest_before"]), "frames"),
            Metric(f"{name}: carry frames", carry, "frames", ok=None if r["static"] else (P.MIN_CARRY, np.inf),
                   note="static object: no bar" if r["static"] else f"window {r['carry']}"),
            Metric(f"{name}: rest-after frames", P.length(r["rest_after"]), "frames"),
            Metric(f"{name}: displacement", r["displacement_cm"], "cm",
                   note=r.get("displacement", f"largest from frame 0; static below {P.STATIC_CM} cm")),
            Metric(f"{name}: phases from", r["source"], "", note=r.get("why_fallback", "")),
        ]
    return StageResult(metrics, outputs, files_read=read)
