"""S4b reference: the kinematic references inverted, keyframe and still, per object (V4_BUILD_SPEC.md
S4b; pathb/reference.py).

Inputs per object (V4): S3e's refined track when it ran (GPU), else S3d's settled track (s3d_settle/<object>/object_to_world.npz; S3c's when S3d
has not run) and its bundle (the canonical mesh), S3b's segments, the grasp segments S3c (or S3d's
re-solve) marked `failed`, S4a's phases.
  inverted  that track; static segments held exactly; a failed grasp segment replaced by the keyframe
            interpolation between the poses around it
  keyframe  v3's rule: the first static pose held until the first grasp, the last one from the last
            release, one interpolation between them (no tracker, so no trusted frame): the baseline the
            go/no-go compares `inverted` with, and the fallback
  still     the first static pose on every frame
The moving candidate an episode trains (s4b_reference/moving.json, S5 reads it): `inverted`, or
`keyframe` when an object has no solved track; $PATHB_MOVING (inverted or keyframe) sets it for a whole run
(the go/no-go's verdict). A failed grasp no longer falls back (2026-10-29): S3c bridges its frames inside
`inverted` (invert.BRIDGE_FAILED), and on the 9 public episodes with one inverted plays back at or above
keyframe (mean 0.535 against 0.411; ep41 0.831 / 0.406, ep43 0.435 / 0.132, ep0 0.353 / 0.354).
v3 runs (s3c_fuse/, cached) keep their old inputs: the fused track, its anchors, S3b's trusted frames.
Hands: the bundle's, with Path B's own repair of wrist spikes (pathb/stage23.py clean_hands).
Each candidate is a complete bundle (BUNDLES.md layout) in s4b_reference/<object>/<candidate>/; the
still candidate also holds the hands and the camera at frame 0 (pathb/stage23.py still_scene).
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
from ..report import Metric, StageResult
from ..runner import write_json
from ..stage23 import clean_hands, still_scene


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
    """V4's solved track: S3e's refined one (GPU, §14.4 R1), else S3d's, else S3c's; its bundle (S3e writes none:
    S3d's), S3b's segments and the failed grasps."""
    for stage in ("s3e_refine", "s3d_settle", "s3c_invert"):
        f = ctx.root / f"{stage}/{name}/object_to_world.npz"
        if not f.is_file():
            continue
        rows = []
        settle = ctx.root / f"s3d_settle/{name}/settle_report.json"
        if stage in ("s3e_refine", "s3d_settle") and settle.is_file():
            rows = json.loads(settle.read_text()).get("resolved") or []
        if not rows:
            rows = json.loads((ctx.root / f"s3c_invert/{name}/invert_report.json").read_text())["segments"]
        bundle = ctx.root / f"{'s3d_settle' if stage == 's3e_refine' else stage}/{name}/bundle"
        return {"track": np.load(f)["object_to_world"].astype(np.float64), "bundle": bundle,
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
    return {"candidate": "inverted", "why": f"failed grasp segments bridged inside it: {failed}" if failed
            else "every grasp segment solved"}


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
    from ..symmetry import CUPS, HALF_TURN
    sym = CUPS | HALF_TURN  # the objects the kit treats as symmetric
    return list(dict.fromkeys([names[0]] + [n for n in names if n not in sym][:1]))


def frame0_check(ctx, name: str) -> dict | None:
    """B3: how far physics moves the object's frame-0 pose (V4 §13.2): S3d's settle of the static segment
    that starts at frame 0, S3c's pose against the settled one. The reference holds the settled pose when
    S3d accepted it and S3c's otherwise, so the track's own frame 0 says nothing. None when frame 0 is not
    static or the settle did not run."""
    f = ctx.root / f"s3d_settle/{name}/settle_report.json"
    row = next((r for r in json.loads(f.read_text())["statics"] if r["segment"][0] == 0), None) if f.is_file() else None
    if row is None or "moved_cm" not in row:
        return None
    return {"dt_mm": row["moved_cm"] * 10, "dtheta_deg": row["turned_deg"], "settled": row["settled"],
            "contacts": row["contacts"], "support": row.get("support")}


def run(ctx) -> StageResult:
    d = ctx.dir("s4b_reference")
    metrics, outputs, read = [], {}, []
    prepared, found = {}, {}
    for name in ctx.objects:
        found[name] = v4 = solved(ctx, name)
        src = v4["bundle"] if v4 else p4_bundle(ctx, name)
        read += [src / "result.npz"]
        hands = d / name / "hands_cleaned"
        if hands.exists():
            shutil.rmtree(hands)
        shutil.copytree(src, hands)
        base = dict(np.load(hands / "result.npz"))
        cleaning = clean_hands(base)  # Path B's own repair (pathb/stage23.py)
        np.savez(hands / "result.npz", **base)
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
                still_scene(data)  # object, hands and camera held at frame 0
            bundle = write_bundle(hands, d / name / cand, data)
            counts = R.provenance(how)
            report["candidates"][cand] = {"bundle": bundle, "frames": counts, "how": how}  # how: per frame (S6 select)
            outputs[f"{name}/{cand}"] = bundle
            metrics.append(Metric(f"{name}: {cand} frames " + " / ".join(R.KINDS),
                                  " / ".join(str(counts[k]) for k in R.KINDS), "frames"))
        if name in b3_objects(ctx.objects) and found[name]:
            report["b3"] = b3 = frame0_check(ctx, name)
            note = ("frame 0 not in a settled static segment" if b3 is None else
                    f"settle {'accepted' if b3['settled'] else 'rejected'}, {b3['contacts']} contacts"
                    f" on {b3['support'] or 'the table'}")
            for k, unit in (("dt_mm", "mm"), ("dtheta_deg", "deg")):
                metrics.append(Metric(f"{name}: B3 frame 0 vs settled ({k.split('_')[0]})",
                                      None if b3 is None else b3[k], unit, ok=B3_BARS[k], note=note))
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
