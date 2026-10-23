"""Score waterfall on public episodes (V3_BUILD_SPEC.md §1): what each step of the v3 chain scores,
next to the numbers it replaces.

Columns, each scored with the official metric code against public ground truth (pathb/gt.py), when
the stage that makes it has run:
  raw                S3b's camera-frame poses through P4's camera (P1c in the v3 world), S3a's mesh:
                     the track before S3c's anchors and smoothing
  fused              the object track after S3c (the P4 bundle S3c completes)
  stage2_clean       Edmond's stage-2 repairs on that bundle (run_load_retarget.cleaned_bundle defaults:
                     rest hold, hand and object cleaning, levelling; a scene merged first, as his runner
                     does), so his historical numbers line up with ours
  ref_<candidate>    S4b's kinematic references: still, keyframe, tracking (playback, no simulation)
  realised_<cand.>   S4c's physically realised trajectories (Do as I Do)
  rollout_<cand.>    S6's evaluator rollouts of the trained policies
Phase A2's variants (pathb/a2.py, run folders <run>_a2_mesh_fp and <run>_a2_mesh_points beside this
one) add a column each: that variant's best candidate (rollout, else realised, else reference). Phase 2
(pathb/phase2.py, <run>_p2) adds p2_<column> for its references, realised and rollouts, and p2_best.
Every bundle column is scored twice: as submitted, and with the public scan as mesh, placed in our
mesh's frame as the official metric registers it (gt.scan_in_frame): `<column>@scan`, "scan mesh -
ceiling, not submittable", the score our poses would get with a perfect mesh.
Baselines (not recomputed): stage2_stereo from HF run 0fa525bc (oracles/waterfall_0fa525bc.json),
and the S0.2 oracle file's reconstruction (one object: recon; a scene: place) and still values
(oracles/oracle_2026-10-17.json). A scene of several objects is assembled from the per-object
bundles with run_load_retarget.merged_bundle, the merge --merge runs (mode A).
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from . import gt, loss
from .bundle import bundle_at
from .paths import PATH_B, TRACK3_FPS, episode_info
from .reference import CANDIDATES
from .runner import write_json
from .team import retarget_tools, scorer

BUNDLE_VARIANTS = ("fused", "stage2_clean", *[f"ref_{c}" for c in CANDIDATES],
                   *[f"realised_{c}" for c in CANDIDATES[:2]])
SCAN = "@scan"  # suffix: the same poses with the scan as mesh (ceiling, not submittable)
ROLLOUTS = tuple(f"rollout_{c}" for c in CANDIDATES)
SIBLINGS = ("a2_mesh_fp", "a2_mesh_points")  # phase A2 variant runs beside this one
P2 = ("p2_ref_keyframe", "p2_ref_tracking", "p2_ref_still", "p2_realised_keyframe", "p2_realised_tracking",
      "p2_rollout_keyframe", "p2_rollout_tracking", "p2_rollout_still", "p2_best")  # phase 2 run beside it
VARIANTS = ("raw", *BUNDLE_VARIANTS, *ROLLOUTS, *SIBLINGS, *P2)
ORACLES = PATH_B / "oracles"
BASELINES = ("0fa525bc stage2_stereo", "S0.2 recon / place", "S0.2 still")


def scene(bundle: Path, names: list[str], frames: int, work: Path, tag: str):
    """(frames, objects, 7) xyzw poses and one scaled mesh per object, as the scorer takes them."""
    L = retarget_tools()
    data = dict(np.load(bundle / "result.npz"))
    poses, meshes = [], {}
    for name, (pose_key, scale_key, mesh_file) in zip(names, L.object_slots(data)):
        T = data[pose_key].astype(np.float64)
        T = T[:frames] if len(T) >= frames else np.concatenate([T, np.repeat(T[-1:], frames - len(T), 0)])
        poses.append(np.concatenate([T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()], axis=-1))
        mesh = trimesh.load(bundle / mesh_file, force="mesh", process=False)
        mesh.apply_scale(float(data.get(scale_key, 1.0)))
        meshes[name] = work / f"{tag}_{name}.obj"
        mesh.export(meshes[name])
    return np.stack(poses, axis=1), meshes


@contextmanager
def bundle_namespace(path: Path):
    """run_load_retarget.bundle_dir_for reads $V2D_BUNDLE_NS; an absolute path replaces the hub."""
    old = os.environ.get("V2D_BUNDLE_NS")
    os.environ["V2D_BUNDLE_NS"] = str(path)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("V2D_BUNDLE_NS")
        else:
            os.environ["V2D_BUNDLE_NS"] = old


def merged(number: int, bundles: dict[str, Path], work: Path) -> Path:
    """One scene bundle from per-object bundles, by the merge --merge uses (mode A)."""
    ns = work / "ns" / f"episode_{number:06d}"
    ns.mkdir(parents=True, exist_ok=True)
    for name, bundle in bundles.items():
        link = ns / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(bundle.resolve())
    with bundle_namespace(ns.parent):
        dest, _ = retarget_tools().merged_bundle(number, work / "merged")
    return dest


def variant_bundles(root: Path, names: list[str]) -> dict[str, dict[str, Path]]:
    """Per variant, the bundle of every object, for the variants whose stages have run."""
    out = {}
    fused = root / "s3c_fuse/report.json"
    if fused.is_file() and json.loads(fused.read_text())["status"] not in ("SKIP", "ERROR"):
        out["fused"] = {n: bundle_at(root, int(root.name.removeprefix("episode_")), n) for n in names}
    for c in CANDIDATES:
        out[f"ref_{c}"] = {n: root / f"s4b_reference/{n}/{c}" for n in names}
        out[f"realised_{c}"] = {n: root / f"s4c_retarget/{n}/{c}/bundle" for n in names}
    return {v: b for v, b in out.items() if v in VARIANTS and all((p / "result.npz").is_file() for p in b.values())}


def baselines(number: int, objects: int) -> dict[str, float | None]:
    out = dict.fromkeys(BASELINES)
    wf = ORACLES / "waterfall_0fa525bc.json"
    if wf.is_file():
        row = next((r for r in json.loads(wf.read_text()) if r["episode"] == number), None)
        if row and "stage2_stereo" in row["scores"]:
            out[BASELINES[0]] = row["scores"]["stage2_stereo"]["AUC"]
    oracle = ORACLES / "oracle_2026-10-17.json"
    if oracle.is_file():
        rows = {r["v"]: r["AUC"] for r in json.loads(oracle.read_text())["variants"] if r["ep"] == number}
        out[BASELINES[1]] = rows.get("recon" if objects == 1 else "place")
        out[BASELINES[2]] = rows.get("still")
    return out


def raw_scene(root: Path, names: list[str], frames: int):
    """S3b's poses through P4's camera, S3a's meshes (None until both stages have run). A frame
    without a pose holds the previous one (the first held back to the first pose)."""
    from .geometry import read_transforms
    c2w, cam_ok = read_transforms(root / "p4_export/camera_to_world", frames)
    poses, meshes = [], {}
    for name in names:
        folder, mesh = root / f"s3b_track/{name}/poses", root / f"s3a_mesh/{name}/mesh.obj"
        if not (folder.is_dir() and mesh.is_file() and cam_ok.any()):
            return None
        T_cam, have = read_transforms(folder, frames)
        have &= cam_ok
        if not have.any():
            return None
        T = c2w @ T_cam
        last = int(np.argmax(have))
        for t in range(frames):
            last = t if have[t] else last
            T[t] = T[last]
        poses.append(np.concatenate([T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()], axis=-1))
        meshes[name] = mesh
    return np.stack(poses, axis=1), meshes


def scan_meshes(e: dict, meshes: dict[str, Path], work: Path, tag: str, read: list) -> dict[str, Path] | None:
    """The scan in each of our meshes' frames (gt.scan_in_frame), as files the scorer takes."""
    out = {}
    for name, path in meshes.items():
        m = trimesh.load(path, force="mesh", process=False)
        placed = gt.scan_in_frame(e, name, np.asarray(m.vertices), np.asarray(m.faces), read)
        if placed is None:
            return None
        out[name] = work / f"{tag}{SCAN}_{name}.obj"
        trimesh.Trimesh(*placed, process=False).export(out[name])
    return out


def episode(root: Path) -> dict:
    number = int(root.name.removeprefix("episode_"))
    e = episode_info(number)
    names = [o["name"] for o in e["objects"]]
    frames = int(e["num_frames"])
    work = root / "waterfall"
    work.mkdir(exist_ok=True)
    read: list = []
    scores = {}
    raw = raw_scene(root, names, frames)
    if raw is not None:
        scores["raw"] = gt.score(e, *raw, read)
        scanned = scan_meshes(e, raw[1], work, "raw", read)
        if scanned:
            scores["raw" + SCAN] = gt.score(e, raw[0], scanned, read)
    variants = variant_bundles(root, names)
    if "fused" in variants:  # Edmond's repairs on the fused bundle (merged first for a scene)
        L = retarget_tools()
        src = variants["fused"][names[0]] if len(names) == 1 else merged(number, variants["fused"], work / "stage2_merge")
        clean, _ = L.cleaned_bundle(src, work / "stage2_clean_bundle", TRACK3_FPS, level=True)  # the runner's defaults
        variants = {**variants, "stage2_clean": {names[0]: clean}} if len(names) == 1 else {**variants, "stage2_clean": clean}
    for variant, bundles in variants.items():
        if isinstance(bundles, Path):  # already one scene
            bundle = bundles
        else:
            bundle = bundles[names[0]] if len(names) == 1 else merged(number, bundles, work / variant)
        poses, meshes = scene(bundle, names, frames, work, variant)
        scores[variant] = gt.score(e, poses, meshes, read)
        scanned = scan_meshes(e, meshes, work, variant, read)
        if scanned:
            scores[variant + SCAN] = gt.score(e, poses, scanned, read)
    for c in CANDIDATES:
        rollout = root / f"s6_eval/{c}/results/episode_{number:06d}.parquet"
        mesh_dir = root / f"s5_task/{c}/run/score_input/meshes/episode_{number:06d}"  # S5's runner's meshes
        if rollout.is_file() and mesh_dir.is_dir():
            poses = scorer().evaluator_poses(rollout, e)  # parses the rollout only; scoring goes through gt
            scores[f"rollout_{c}"] = gt.score(e, poses, {n: next(mesh_dir.glob(f"{n}.*")) for n in names}, read)
    for sibling in SIBLINGS:  # the A2 variant's own waterfall, its best candidate
        other = root.parent.parent / f"{root.parent.name}_{sibling}" / root.name / "waterfall.json"
        if other.is_file():
            from .analysis import best_candidate
            theirs = json.loads(other.read_text())["scores"]
            best = best_candidate(theirs)
            if best:
                scores[sibling] = dict(theirs[best], candidate=best)
    p2 = root.parent.parent / f"{root.parent.name}_p2" / root.name / "waterfall.json"
    if p2.is_file():  # phase 2's run of this episode
        from .analysis import best_candidate
        theirs = json.loads(p2.read_text())["scores"]
        for col in [c for c in theirs if c.startswith(("ref_", "realised_", "rollout_")) and SCAN not in c]:
            scores[f"p2_{col}"] = theirs[col]
        best = best_candidate(theirs)
        if best:
            scores["p2_best"] = dict(theirs[best], candidate=best)
    base = baselines(number, len(names))
    reference = base[BASELINES[0]]
    attribution = {f"{v} vs 0fa525bc stage2_stereo": s["AUC"] - reference
                   for v, s in scores.items() if reference is not None}
    row = {"episode": number, "objects": names,
           "variants": [v for v in VARIANTS if v in scores] + [v + SCAN for v in VARIANTS if v + SCAN in scores],
           "scan_columns": "scan mesh - ceiling, not submittable",
           "scores": scores, "baselines": base, "auc_attribution": attribution,
           "files_read": [str(p) for p in read]}
    write_json(root / "waterfall.json", row)
    reports = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/report.json"))}
    loss.write(root, dict(sorted(reports.items(), key=lambda kv: stage_order(kv[0]))))
    return row


def stage_order(name: str) -> int:
    from .runner import STAGES
    return STAGES.index(name) if name in STAGES else len(STAGES)


def summary(out: Path, rows: list[dict]) -> str:
    """AUC per episode: baselines first, then every v3 column."""
    cols = [*BASELINES, *VARIANTS, *[v + SCAN for v in ("raw", *BUNDLE_VARIANTS)]]
    lines = ["| Episode | Objects | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in rows:
        values = [r["baselines"].get(c) for c in BASELINES] + [r["scores"].get(v, {}).get("AUC") for v in cols[len(BASELINES):]]
        lines.append(f"| {r['episode']} | {' + '.join(r['objects'])} | "
                     + " | ".join("" if v is None else f"{v:.3f}" for v in values) + " |")
    means = []
    for i, c in enumerate(cols):
        vals = [(r["baselines"].get(c) if i < len(BASELINES) else r["scores"].get(c, {}).get("AUC")) for r in rows]
        vals = [v for v in vals if v is not None]
        means.append(f"**{np.mean(vals):.3f}**" if vals else "")
    lines.append("| **mean AUC** | | " + " | ".join(means) + " |")
    text = "\n".join(lines) + "\n"
    write_json(out / "waterfall.json", rows)
    (out / "waterfall.md").write_text("# Path B v3 score waterfall (public episodes, AUC)\n\n" + text
                                      + f"\nColumns ending {SCAN}: the same poses with the public scan as mesh "
                                        "(scan mesh - ceiling, not submittable).\n")
    return text
