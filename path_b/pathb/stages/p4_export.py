"""P4 export (v3): the episode's bundles before the object is known (V3_BUILD_SPEC.md P4).

The v3 world (pathb/world.py: table top z = 0, +z up) from P3's table plane; P1c's camera moved into
it (camera_to_world/); then per object a bundle in the BUNDLES.md layout written by the baseline's
own result writer, with that camera and P1d's hands, plus table_plane, gravity and world_from_p1c.
No object track yet (S3c completes each bundle) and no mesh.obj until S3a. Next to each bundle:
run.json, numeric_qa.json (P0-P3 measurements) and status.json.
V4 (P1d's exports, pathb/hands.py): hands/{fingertips_world,hand_verts_world,wrist_world}.npz and
hand_gaps.json, the hands in this world with the hand-gap rule applied, from the first object's
bundle through the baseline loader's re-posing ($RETARGET_VENV, $MANO_DIR).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .. import hands as Hd
from .. import world as V
from ..bundle import hand_dirs, p4_bundle, write_v3_bundle
from ..geometry import read_transforms, write_transform
from ..paths import TRACK3_FPS, git_commit
from ..report import Metric, StageResult, worst
from ..runner import write_json

UPSTREAM = ("p0_ingest", "p1_calibrate", "p2_depth", "p1b_masks", "p1c_camera", "p1d_hands", "p3_scale_bridge")


def world_extra(W: np.ndarray) -> dict:
    return {"table_plane": np.array([0.0, 0.0, 1.0, 0.0]), "gravity": np.array([0.0, 0.0, -1.0]),
            "world_from_p1c": W, "gravity_alignment_applied": True, "fps": TRACK3_FPS}


def run(ctx) -> StageResult:
    d = ctx.dir("p4_export")
    n = int(ctx.episode["num_frames"])
    scale = json.loads((ctx.root / "p3_scale_bridge/scale.json").read_text())
    c2w_p1c, have = read_transforms(ctx.root / "p1c_camera/camera_to_world", n)
    rest = [t for t in range(scale["rest_window"][0], scale["rest_window"][1] + 1) if have[t]]
    W = V.from_table(np.array(scale["table_plane"]["normal"]), scale["table_plane"]["d"], c2w_p1c, rest)
    cams = d / "camera_to_world"
    cams.mkdir(exist_ok=True)
    c2w = W[None] @ c2w_p1c
    for t in range(n):
        if have[t]:
            write_transform(cams / f"{t:06d}.json", c2w[t])
    write_json(d / "world.json", {"world_from_p1c": W, "table": "z = 0", "up": "+z",
                                  "y": "mean camera forward over the rest window, on the table",
                                  "origin": f"the table under the camera at frame {rest[0]}"})
    reports = {name: ctx.reports[name] for name in UPSTREAM if name in ctx.reports}
    qa = {name: r["metrics"] for name, r in reports.items()}
    status = {"state": worst(r["status"] for r in reports.values()),
              "first_failing_stage": next((s for s, r in reports.items() if r["status"] in ("FAIL", "ERROR")), None),
              "stages": {s: r["status"] for s, r in reports.items()}}
    metrics = [Metric("camera height above the table (median, rest window)", float(np.median(c2w[rest, 2, 3])), "m",
                      note="v3 world: the table top is z = 0"),
               Metric("camera z axis below the horizon (median, rest window)",
                      float(np.degrees(np.median(np.arcsin(np.clip(-c2w[rest, 2, 2], -1, 1))))), "deg",
                      note="how far the head looks down at the table")]
    log = d / "writer.log"
    log.write_text("")
    for name in ctx.objects:
        bundle = write_v3_bundle(ctx, p4_bundle(ctx, name), log, extra=world_extra(W))
        data = np.load(bundle / "result.npz")
        finite = all(np.isfinite(data[k]).all() for k in data.files if data[k].dtype.kind == "f")
        metrics += [
            Metric(f"{name}: frames minus manifest", int(data["camera_to_world_transform"].shape[0]) - n, "frames", ok=(0, 0)),
            Metric(f"{name}: all arrays finite", int(finite), "", ok=(1, 1)),
            Metric(f"{name}: camera valid", float(data["camera_is_valid"].mean()), "share", ok=(0.99, 1)),
            *[Metric(f"{name}: {side} hand valid", float(data[f"hand_{side}_is_valid"].mean()), "share",
                     note="BUNDLES.md Should: >= 95 % of the frames the hand is visible") for side in ("left", "right")],
        ]
        folder = bundle.parent
        write_json(folder / "run.json", {"producer": "path_b v3", "command": sys.argv, "commit": git_commit(),
                                         "fps": TRACK3_FPS, "world": "table top z = 0, +z up (p4_export/world.json)",
                                         "notes": "camera P1c, hands P1d; object and mesh come from S3a-S3c"})
        write_json(folder / "numeric_qa.json", qa)
        write_json(folder / "status.json", status)
    hands = Hd.export(p4_bundle(ctx, ctx.objects[0]), hand_dirs(ctx.root), d / "hands", log)
    for h, share in zip(Hd.HANDS, hands["valid_share"]):
        g = hands["gaps"][h]
        filled = [x for x in g if x["kind"] == "filled"]
        long = [x for x in g if x["kind"] == "long"]
        metrics += [
            Metric(f"hand_{h}: frames with a wrist (world exports)", share, "share",
                   note="pathb/hands.py: fingertips, vertices, wrist in p4_export/hands/"),
            Metric(f"hand_{h}: gaps filled (<= {Hd.MAX_GAP} frames, Hermite/slerp in the world)", len(filled), "gaps",
                   note=f"{sum(x['frames'] for x in filled)} frames; largest wrist move by the re-interpolation "
                        f"{hands['hermite_shift_cm'][h]:.1f} cm"),
            Metric(f"hand_{h}: gaps left empty (longer, or at the clip's ends)", len(long), "gaps",
                   note=("longest " + str(max(x["frames"] for x in long)) + " frames; S3b splits a grasp there")
                        if long else "none"),
        ]
    return StageResult(metrics, {"bundles": d / f"episode_{ctx.number:06d}", "world": d / "world.json",
                                 "hands": d / "hands"})
