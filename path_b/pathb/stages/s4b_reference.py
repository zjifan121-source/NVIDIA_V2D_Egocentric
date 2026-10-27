"""S4b reference: the kinematic references inverted, keyframe and still, per object (V4_BUILD_SPEC.md
S4b; pathb/reference.py).

Inputs per object (V4): S3d's settled track (s3d_settle/<object>/object_to_world.npz; S3c's when S3d
has not run) and its bundle (the canonical mesh), S3b's segments, the grasp segments S3c (or S3d's
re-solve) marked `failed`, S4a's phases.
  inverted  that track; static segments held exactly; a failed grasp segment replaced by the keyframe
            interpolation between the poses around it
  keyframe  v3's rule: the first static pose held until the first grasp, the last one from the last
            release, one interpolation between them (no tracker, so no trusted frame): the baseline the
            go/no-go compares `inverted` with, and the fallback
  still     the first static pose on every frame
The moving candidate an episode trains (s4b_reference/moving.json, S5 reads it): `inverted`, or
`keyframe` when any object has a failed grasp segment or no solved track; $PATHB_MOVING (inverted or
keyframe) sets it for a whole run (the go/no-go's verdict).
v3 runs (s3c_fuse/, cached) keep their old inputs: the fused track, its anchors, S3b's trusted frames.
Hands: the bundle's, cleaned with run_load_retarget's hand repairs (cleaned_bundle, hands only).
Each candidate is a complete bundle (BUNDLES.md layout) in s4b_reference/<object>/<candidate>/; the
still candidate also holds the hands at frame 0 (run_load_retarget.still_scene, as --still-object).
Public, one object: each candidate's playback score (official metrics) is reported; scenes of
several objects are scored in the waterfall (ref_* columns).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

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


def solved(ctx, name: str) -> dict | None:
    """V4's solved track: S3d's (else S3c's), its bundle, S3b's segments and the failed grasps."""
    for stage in ("s3d_settle", "s3c_invert"):
        f = ctx.root / f"{stage}/{name}/object_to_world.npz"
        if not f.is_file():
            continue
        rows = []
        settle = ctx.root / f"s3d_settle/{name}/settle_report.json"
        if stage == "s3d_settle" and settle.is_file():
            rows = json.loads(settle.read_text()).get("resolved") or []
        if not rows:
            rows = json.loads((ctx.root / f"s3c_invert/{name}/invert_report.json").read_text())["segments"]
        return {"track": np.load(f)["object_to_world"].astype(np.float64), "bundle": ctx.root / f"{stage}/{name}/bundle",
                "segments": json.loads((ctx.root / f"s3b_segments/{name}/segments.json").read_text()),
                "failed": [(r["start"], r["end"]) for r in rows if r["kind"] == "grasp" and r.get("failed")],
                "source": stage}
    return None


def inverted(v4: dict) -> tuple[np.ndarray, list[str]]:
    """The solved track, statics held; failed grasps interpolated between the poses around them."""
    T = v4["track"].copy()
    n = len(T)
    how = ["inverted"] * n
    for s in v4["segments"]:
        if s["kind"] == "static":
            T[s["start"]:s["end"] + 1] = T[s["start"]]
            how[s["start"]:s["end"] + 1] = ["held"] * (s["end"] - s["start"] + 1)
    for a, b in v4["failed"]:
        lo, hi = max(a - 1, 0), min(b + 1, n - 1)
        if hi - lo < 2:
            continue
        s = np.arange(1, hi - lo) / (hi - lo)
        T[lo + 1:hi, :3, 3] = R.hermite(T[lo, :3, 3], T[hi, :3, 3], s)
        T[lo + 1:hi, :3, :3] = Slerp([0, 1], Rotation.from_matrix([T[lo, :3, :3], T[hi, :3, :3]]))(s).as_matrix()
        how[lo + 1:hi] = ["interpolated"] * (hi - lo - 1)
    return T, how


def rest_anchors(v4: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """The first static segment's pose; the last one's when it reaches the clip's end."""
    T, statics = v4["track"], [s for s in v4["segments"] if s["kind"] == "static"]
    if not statics:
        return T[0], None
    last = statics[-1]
    return T[statics[0]["start"]], (T[last["start"]] if last["end"] == len(T) - 1 else None)


def moving_candidate(ctx, found: dict) -> dict:
    """Which moving candidate S5/S6 train (V4 S6: inverted, or keyframe if it fell back)."""
    forced = os.environ.get("PATHB_MOVING")
    if forced in ("inverted", "keyframe"):
        return {"candidate": forced, "why": "PATHB_MOVING (run-wide: the go/no-go's verdict)"}
    missing = [n for n, v in found.items() if v is None]
    failed = {n: v["failed"] for n, v in found.items() if v and v["failed"]}
    if missing:
        return {"candidate": "keyframe", "why": f"no solved track for {missing}"}
    if failed:
        return {"candidate": "keyframe", "why": f"failed grasp segments: {failed}"}
    return {"candidate": "inverted", "why": "every grasp segment solved"}


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


B3_BARS = {"dt_mm": (0, 5), "dtheta_deg": (0, 1)}  # spec §13.2 B3


def b3_objects(names: list[str]) -> list[str]:
    """The objects the scorer aligns an episode by (V4 §13.2 B3): object 0 (translation) and the first
    one without a symmetry in the kit (rotation)."""
    from ..team import scorer
    sym = scorer().SYMMETRY_ORDER
    return list(dict.fromkeys([names[0]] + [n for n in names if n not in sym][:1]))


def frame0_check(ctx, name: str, track: np.ndarray) -> dict | None:
    """B3: the candidate's frame-0 pose against the settled pose of the object's first static segment
    (S3d's settled_<start>.json); None without one."""
    segs = json.loads((ctx.root / f"s3b_segments/{name}/segments.json").read_text())
    first = next((s for s in segs if s["kind"] == "static"), None)
    f = ctx.root / f"s3d_settle/{name}/settled_{first['start']:06d}.json" if first else None
    if f is None or not f.is_file():
        return None
    P = np.asarray(json.loads(f.read_text())["pose"], dtype=np.float64)
    T0 = np.asarray(track[0], dtype=np.float64)
    turn = Rotation.from_matrix(T0[:3, :3].T @ P[:3, :3]).magnitude()
    return {"segment_start": int(first["start"]), "dt_mm": float(np.linalg.norm(T0[:3, 3] - P[:3, 3]) * 1000),
            "dtheta_deg": float(np.degrees(turn))}


def run(ctx) -> StageResult:
    L = retarget_tools()
    d = ctx.dir("s4b_reference")
    metrics, outputs, read = [], {}, []
    prepared, found = {}, {}
    for name in ctx.objects:
        found[name] = v4 = solved(ctx, name)
        src = v4["bundle"] if v4 else p4_bundle(ctx, name)
        read += [src / "result.npz"]
        hands, cleaning = L.cleaned_bundle(src, d / name / "hands_cleaned", TRACK3_FPS, rest_hold=False,
                                           hand_clean=True, object_clean=False)
        base = dict(np.load(hands / "result.npz"))
        if v4:
            before, after = rest_anchors(v4)
            prepared[name] = (hands, cleaning, base, (v4["track"], before, after, None, v4["source"]))
        else:
            prepared[name] = (hands, cleaning, base, inputs(ctx, name, base))
    moving = moving_candidate(ctx, found)
    write_json(d / "moving.json", moving)
    for name in ctx.objects:
        hands, cleaning, base, (T, before, after, trusted, source) = prepared[name]
        phases = json.loads((ctx.root / f"s4a_phases/{name}/phases.json").read_text())
        built = R.build(T, phases, before, after, trusted, inverted=inverted(found[name]) if found[name] else None)
        report = {"object": name, "input_track": source, "hand_cleaning": cleaning, "candidates": {}}
        c2w = base["camera_to_world_transform"].astype(np.float64)
        for cand, (track, how) in built.items():
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
            metrics.append(Metric(f"{name}: {cand} frames " + " / ".join(R.KINDS),
                                  " / ".join(str(counts[k]) for k in R.KINDS), "frames"))
            if name in b3_objects(ctx.objects) and found[name]:
                b3 = frame0_check(ctx, name, track)
                report["candidates"][cand]["b3"] = b3
                for k, unit in (("dt_mm", "mm"), ("dtheta_deg", "deg")):
                    metrics.append(Metric(f"{name}: B3 {cand} frame 0 vs settled static ({k.split('_')[0]})",
                                          None if b3 is None else b3[k], unit, ok=B3_BARS[k],
                                          note="no settled static segment" if b3 is None else
                                          f"first static segment from frame {b3['segment_start']}"))
        metrics.append(Metric(f"{name}: input track", source, ""))
        write_json(d / name / "reference_report.json", report)
    metrics.append(Metric("moving candidate S5 and S6 train", moving["candidate"], "", note=moving["why"]))
    if ctx.public and len(ctx.objects) == 1:
        name = ctx.objects[0]
        for cand in [c for c in R.CANDIDATES if (d / name / c / "result.npz").is_file()]:
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
