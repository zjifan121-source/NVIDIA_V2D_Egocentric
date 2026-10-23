"""S4b reference: the kinematic references keyframe, tracking and still, per object (pathb/reference.py).

Inputs per object: S4a's phases.json; the object track and its rest anchors from S3c
(s3c_fuse/<object>/object_to_world.npz, anchors.json) and S3b's track_meta.json (trusted frames).
Until S3c exists, the P4 bundle's track stands in, with anchors from fuse.rest_anchor over the
rest windows and no trusted frame (the carry is then one interpolation); the report says which.
Hands: the P4 bundle's, cleaned with run_load_retarget's hand repairs (cleaned_bundle, hands only).
Each candidate is a complete bundle (BUNDLES.md layout) in s4b_reference/<object>/<candidate>/; the
still candidate also holds the hands at frame 0 (run_load_retarget.still_scene, as --still-object).
Public, one object: each candidate's playback score (official metrics) is reported; scenes of
several objects are scored in the waterfall (ref_* columns).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .. import fuse, gt
from .. import reference as R
from ..bundle import p4_bundle
from ..paths import TRACK3_FPS
from ..report import Metric, StageResult
from ..runner import write_json
from ..team import retarget_tools


def inputs(ctx, name: str, data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    """(track, anchor before, anchor after, trusted frames, source)."""
    phases = json.loads((ctx.root / f"s4a_phases/{name}/phases.json").read_text())
    fused = ctx.root / f"s3c_fuse/{name}"
    if (fused / "object_to_world.npz").is_file():
        T = np.load(fused / "object_to_world.npz")["object_to_world"]
        anchors = json.loads((fused / "anchors.json").read_text())
        source = "s3c_fuse"
        # S3c leaves an anchor out when no rest frame passes its 2 cm depth-residual test; the
        # window's own rest anchor over the fused track stands in, and the source says so
        missing = [k for k in ("rest_before", "rest_after") if phases.get(k) and not anchors.get(k)]
        def anchor(key):
            if anchors.get(key):
                return np.array(anchors[key]["pose"])
            w = phases.get(key)
            return fuse.rest_anchor(T[w[0]:w[1] + 1]) if w else None
        before = anchor("rest_before")
        before = T[0] if before is None else before
        after = anchor("rest_after")
        if missing:
            source += f" ({' and '.join(missing)} anchor from the fused track: no frame within S3c's depth-residual bar)"
        trusted = None
        meta = ctx.root / f"s3b_track/{name}/track_meta.json"
        if meta.is_file():
            rows = {f["frame"]: f for f in json.loads(meta.read_text())["frames"]}
            trusted = np.array([(rows.get(t, {}).get("iou") or 0.0) >= R.TRUST_IOU and not rows.get(t, {}).get("recovered")
                                for t in range(len(T))])
        return T, before, after, trusted, source
    T = data["object_to_world_transform"].astype(np.float64)
    rb, ra = phases["rest_before"], phases["rest_after"]
    before = fuse.rest_anchor(T[rb[0]:rb[1] + 1]) if rb else T[0]
    after = fuse.rest_anchor(T[ra[0]:ra[1] + 1]) if ra else None
    return T, before, after, None, "p4 bundle (stand-in until S3c)"


def write_bundle(src: Path, dest: Path, data: dict) -> Path:
    """dest = src's files (linked) with this result.npz."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in src.iterdir():
        if f.name != "result.npz":
            (dest / f.name).symlink_to(f.resolve())
    np.savez(dest / "result.npz", **data)
    return dest


def run(ctx) -> StageResult:
    L = retarget_tools()
    d = ctx.dir("s4b_reference")
    metrics, outputs, read = [], {}, []
    for name in ctx.objects:
        src = p4_bundle(ctx, name)
        read += [src / "result.npz"]
        hands, cleaning = L.cleaned_bundle(src, d / name / "hands_cleaned", TRACK3_FPS, rest_hold=False,
                                           hand_clean=True, object_clean=False)
        base = dict(np.load(hands / "result.npz"))
        phases = json.loads((ctx.root / f"s4a_phases/{name}/phases.json").read_text())
        T, before, after, trusted, source = inputs(ctx, name, base)
        built = R.build(T, phases, before, after, trusted)
        report = {"object": name, "input_track": source, "hand_cleaning": cleaning, "candidates": {}}
        c2w = base["camera_to_world_transform"].astype(np.float64)
        for cand in R.CANDIDATES:
            track, how = built[cand]
            data = dict(base)
            data["object_to_world_transform"] = track.astype(base["object_to_world_transform"].dtype)
            data["object_to_camera_transform"] = (np.linalg.inv(c2w) @ track).astype(
                base["object_to_camera_transform"].dtype)
            if cand == "still":
                L.still_scene(data)  # hands held at frame 0, as --still-object
            bundle = write_bundle(hands, d / name / cand, data)
            counts = R.provenance(how)
            report["candidates"][cand] = {"bundle": bundle, "frames": counts, "how": how}  # how: per frame (S6 select)
            outputs[f"{name}/{cand}"] = bundle
            metrics.append(Metric(f"{name}: {cand} frames held / trusted / interpolated / tracked / blended",
                                  " / ".join(str(counts[k]) for k in ("held", "trusted", "interpolated", "tracked",
                                                                      "blended")), "frames"))
        metrics.append(Metric(f"{name}: input track", source, ""))
        write_json(d / name / "reference_report.json", report)
    if ctx.public and len(ctx.objects) == 1:
        name = ctx.objects[0]
        for cand in R.CANDIDATES:
            poses, meshes = playback(d / name / cand, [name], int(ctx.episode["num_frames"]), d / name, cand)
            scores = gt.score(ctx.episode, poses, meshes, ctx.files_read)
            metrics.append(Metric(f"{name}: {cand} playback AUC", scores["AUC"], "", kind="gt",
                                  note=f"MPPE {scores['MPPE cm']:.2f} cm, MP-SR {scores['MP-SR']:.2f}"))
            outputs[f"{name}/{cand}/scores"] = scores
    return StageResult(metrics, outputs, files_read=read)


def playback(bundle: Path, names: list[str], frames: int, work: Path, tag: str):
    """A bundle's object poses (frames, objects, 7) and scaled meshes, as score_local takes them."""
    from ..waterfall import scene
    return scene(bundle, names, frames, work, tag)
