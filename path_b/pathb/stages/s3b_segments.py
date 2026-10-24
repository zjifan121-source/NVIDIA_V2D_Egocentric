"""S3b contact segmentation (V4_BUILD_SPEC.md S3b; rules in pathb/segments.py): per object, the clip as
static, grasp and unstable segments, and what each static segment rests on.

In: the object's observed points per frame (pathb/observe.py: P1b mask x P2 depth_a_half x P4
camera), P1d's world exports (p4_export/hands/: fingertips, wrists, valid flags, long gaps). Out per
object: segments.json, phases.json (S4a's layout: rest-before / carry / rest-after from the first and
last grasp), segments_report.json (thresholds used, votes, segment table, support evidence) and
coverage.json (hands.coverage: the share of grasp frames with the grasping hand's wrist; S0.3).
Support is the points rule of pathb/segments.py; the team's support_stacked rule (pathb/support.py)
checks it when $RETARGET_VENV is set (a disagreement is a WARN, a check not run a WARN).
Public only: grasp + unstable frames against the ground truth's moving frames (S0.2's moving_mask)
as an agreement share (kind gt). Bars: every frame in exactly one segment; a grasp unless the
object is static all clip (a WARN: nothing to carry).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .. import gt, hands, support as SU
from .. import segments as G
from ..observe import View
from ..report import Metric, StageResult
from ..runner import write_json
from ..team import playback_oracle

LIMIT = 5000  # points per frame kept (the motion test queries <= segments.QUERY of them)


def long_gaps(d) -> dict[str, list[tuple[int, int]]]:
    gaps = json.loads((d / "hand_gaps.json").read_text())["gaps"]
    return {h: [(g["start"], g["end"]) for g in rows if g["kind"] == "long" and g["frames"] > G.LONG_GAP]
            for h, rows in gaps.items()}


def gt_agreement(ctx, k: int, segs: list[dict], n: int) -> float | None:
    """Share of frames where (grasp or unstable) equals the ground truth's moving mask."""
    truth = gt.ground_truth(ctx.episode, ctx.files_read)[:, k]
    moving = playback_oracle().moving_mask(truth)[:n]
    ours = np.zeros(len(moving), bool)
    for s in segs:
        if s["kind"] != "static":
            ours[s["start"]:min(s["end"] + 1, len(ours))] = True
    return float((ours == moving).mean()) if len(moving) else None


def run(ctx) -> StageResult:
    d = ctx.dir("s3b_segments")
    n = int(ctx.episode["num_frames"])
    hd = ctx.root / "p4_export/hands"
    tips = np.load(hd / "fingertips_world.npz")
    wrist = np.load(hd / "wrist_world.npz")["wrist"]
    gaps = long_gaps(hd)
    view = View(ctx)
    rng = np.random.default_rng(0)
    clouds = {name: [view.points(k, t, LIMIT, rng) for t in range(n)] for k, name in enumerate(ctx.objects)}
    metrics, outputs, found = [], {}, {}
    for k, name in enumerate(ctx.objects):
        segs, report = G.segment(clouds[name], tips["tips"], wrist, tips["valid"], gaps)
        found[name] = (segs, report)
    for k, name in enumerate(ctx.objects):  # support needs every object's segments
        segs, report = found[name]
        evidence = []
        for s in segs:
            if s["kind"] != "static":
                continue
            pts = G.pooled(clouds[name], s["start"], s["end"])
            if pts is None:
                s["support"] = None
                evidence.append({"segment": [s["start"], s["end"]], "why": "no points"})
                continue
            others = {o: G.pooled(clouds[o], s["start"], s["end"]) for o in ctx.objects if o != name}
            s["support"], ev = G.support(pts, others)
            check = None
            if os.environ.get("RETARGET_VENV") and others:
                disks = {name: SU.disk(pts), **{o: SU.disk(q) for o, q in others.items() if q is not None}}
                work = d / name / f"support_{s['start']}"
                work.mkdir(parents=True, exist_ok=True)
                check = SU.supports(disks, work)[name]
            evidence.append({"segment": [s["start"], s["end"]], "support": s["support"], "support_stacked": check, **ev})
        report["support"] = evidence
        out = d / name
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "segments.json", segs)
        cent = G.centroids(clouds[name])
        write_json(out / "phases.json", G.phases(segs, n, cent) | {"object": name})
        cov = hands.coverage(tips["valid"], segs)
        write_json(out / "coverage.json", cov)
        write_json(out / "segments_report.json", report)
        outputs[name] = out / "segments.json"
        covered = sorted(t for s in segs for t in range(s["start"], s["end"] + 1))
        grasps = [s for s in segs if s["kind"] == "grasp"]
        th = report["thresholds"]
        kinds = {kd: sum(1 for s in segs if s["kind"] == kd) for kd in ("static", "grasp", "unstable")}
        metrics += [
            Metric(f"{name}: every frame in one segment", int(covered == list(range(n))), "", ok=(1, 1)),
            Metric(f"{name}: segments static / grasp / unstable", f"{kinds['static']} / {kinds['grasp']} / {kinds['unstable']}",
                   "", note="; ".join(f"{s['kind']} {s['start']}-{s['end']}" + (f" {s['hand']}" if s["hand"] else "")
                                      for s in segs)[:300]),
            Metric(f"{name}: grasp found", len(grasps), "segments", ok=(1, np.inf), warn=(0, np.inf),
                   note="static all clip" if report["static"] else ""),
            Metric(f"{name}: near / far", f"{th['near_m'] * 100:.2f} / {th['far_m'] * 100:.2f}", "cm", note=th["source"]),
            Metric(f"{name}: first grasp start / last grasp end",
                   f"{grasps[0]['start']} / {grasps[-1]['end']}" if grasps else "none", "frame"),
            Metric(f"{name}: frames without object points", report["frames_without_points"], "frames"),
            Metric(f"{name}: grasp frames with the grasping hand's wrist", cov["coverage_carry"], "share",
                   note="S0.3's coverage_carry; < 0.9 makes P1d's hand-gap rule mandatory"),
        ]
        for ev in evidence:
            if "support" not in ev:
                continue
            a, b = ev["segment"]
            ran = ev["support_stacked"] is not None
            agrees = not ran or ev["support_stacked"] == ev["support"]
            note = (f"support_stacked: {ev['support_stacked']}" if ran else "support_stacked not run (no RETARGET_VENV or one object)") \
                + ("; nothing under it: floating" if ev.get("floating") else "")
            metrics.append(Metric(f"{name}: support of static {a}-{b}", ev["support"], "", note=note))
            if len(ctx.objects) > 1 or ev.get("floating"):
                metrics.append(Metric(f"{name}: support of static {a}-{b} confirmed", int(ran and agrees and not ev.get("floating")),
                                      "", ok=(1, 1), warn=(0, 1), note="support_stacked agrees and something holds it"))
        if ctx.public:
            agree = gt_agreement(ctx, k, segs, n)
            metrics.append(Metric(f"{name}: grasp + unstable frames vs ground-truth moving frames", agree, "share",
                                  kind="gt", note="S0.2's moving_mask (2 cm/s or 10 deg/s over 5 frames)"))
    return StageResult(metrics, outputs)
