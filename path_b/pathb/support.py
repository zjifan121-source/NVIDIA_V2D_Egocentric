"""What an object rests on: the team's support_stacked rule (V4_BUILD_SPEC.md S3b "Support", S3d).

Each object's support disk (centre and radius of its footprint, height of its lowest point) goes to
tools/track3/support_stacked.py drop_stacked_supports in $RETARGET_VENV (drivers/support_relation.py):
an object whose disk sits more than 1 cm above another's and inside its circle rests on it.
Kept from v3's phase 2 (cache/v3/phase2.py), where S4a used it.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .paths import PATH_B
from .runner import write_json


def disk(vertices_world: np.ndarray) -> list[float]:
    """(cx, cy, z, r): footprint centre and radius, lowest point (support_recon's disk layout)."""
    c = vertices_world[:, :2].mean(axis=0)
    return [float(c[0]), float(c[1]), float(vertices_world[:, 2].min()),
            float(np.linalg.norm(vertices_world[:, :2] - c, axis=1).max())]


def supports(disks: dict[str, list[float]], work: Path) -> dict[str, str]:
    """Object -> what it rests on ("table" or another object), by support_stacked.drop_stacked_supports."""
    venv = os.environ.get("RETARGET_VENV")
    if not venv:
        raise RuntimeError("set RETARGET_VENV (support_stacked.py imports robotic_grounding)")
    request = work / "support_request.json"
    write_json(request, {"disks": {n: [d] for n, d in disks.items()}, "out": work / "support_kept.json"})
    subprocess.run([f"{venv}/bin/python", str(PATH_B / "pathb/drivers/support_relation.py"), str(request)], check=True,
                   capture_output=True)
    kept = set(json.loads((work / "support_kept.json").read_text())["kept"])
    out = {}
    for name, (cx, cy, z, _) in disks.items():
        if name in kept:
            out[name] = "table"
        else:  # dropped: it rests on the object whose circle holds it (the rule's own test)
            under = [o for o, (ox, oy, oz, orad) in disks.items() if o != name and z > oz + 0.01 and np.hypot(cx - ox, cy - oy) < orad]
            out[name] = under[0] if under else "table"
    return out
