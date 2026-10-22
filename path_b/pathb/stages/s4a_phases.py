"""S4a phases: rest before the reach, the carry and rest after, per object (pathb/phases.py).

Reads each object's bundle as P4 exports it (and S3c completes it): wrists in the world, the object
track and its metric mesh. Writes s4a_phases/<object>/phases.json. Acceptance (V3_BUILD_SPEC.md S4a):
a carry of at least 10 frames, unless the object never leaves its frame-0 position by 2 cm (static).
"""
from __future__ import annotations

import numpy as np

from .. import phases as P
from ..bundle import p4_bundle
from ..report import Metric, StageResult
from ..runner import write_json
from ..team import retarget_tools


def run(ctx) -> StageResult:
    L = retarget_tools()
    metrics, outputs, read = [], {}, []
    for name in ctx.objects:
        bundle = p4_bundle(ctx, name)
        data = dict(np.load(bundle / "result.npz"))
        read += [bundle / "result.npz", bundle / "mesh.obj"]
        r = P.from_bundle(data, L.object_vertices(bundle, data, "object_scale", "mesh.obj"))
        path = ctx.dir("s4a_phases") / name / "phases.json"
        path.parent.mkdir(exist_ok=True)
        write_json(path, r | {"object": name, "reach_cm": P.REACH_CM, "windows": "inclusive [first, last] frames"})
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
            Metric(f"{name}: largest displacement from frame 0", r["displacement_cm"], "cm",
                   note=f"static below {P.STATIC_CM} cm"),
        ]
    return StageResult(metrics, outputs, files_read=read)
