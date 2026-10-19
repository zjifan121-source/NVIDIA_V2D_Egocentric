"""Score waterfall on public episodes: how many points each step of the chain gains or loses.

Each variant is scored with the official metric code against public ground truth, as
tools/track3/dryrun_public.py does (no simulation; two-object episodes pay for our placement):
  raw            the team's bundle as uploaded (merged for two objects)
  stage2         after stage 2's default repairs (hand and object cleaning, rest hold, levelling)
  stage2_stereo  the same after Path B's stereo scale (P3)
  stage2_scan    the same after the oracle scale from the public scans (--rescale-by-scan)
AUC attribution: stage 2 repairs = stage2 - raw; Path B stereo scale = stage2_stereo - stage2;
scale still missing = stage2_scan - stage2_stereo; tracking, mesh and placement = 1 - stage2_scan.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from . import loss
from .bundle import rescaled_copy
from .paths import TRACK3_FPS, episode_info
from .runner import write_json
from .stages.p3_scale_bridge import source_bundle
from .team import retarget_tools, scorer

VARIANTS = ("raw", "stage2", "stage2_stereo", "stage2_scan")


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


def episode(root: Path) -> dict:
    L, S = retarget_tools(), scorer()
    number = int(root.name.removeprefix("episode_"))
    e = episode_info(number)
    names = [o["name"] for o in e["objects"]]
    frames = len(S.ground_truth(number))
    factor = json.loads((root / "p3_scale_bridge/scale.json").read_text())["factor"]
    work = root / "waterfall"
    work.mkdir(exist_ok=True)
    src = source_bundle(number, names, work)
    bundles = {
        "raw": src,
        "stage2": L.cleaned_bundle(src, work / "stage2", TRACK3_FPS, level=True)[0],
        "stage2_stereo": L.cleaned_bundle(rescaled_copy(src, work / "stereo_src", factor), work / "stage2_stereo",
                                          TRACK3_FPS, level=True)[0],
    }
    try:
        bundles["stage2_scan"] = L.cleaned_bundle(src, work / "stage2_scan", TRACK3_FPS, rescale_by_scan=names[0],
                                                  level=True)[0]
    except SystemExit:  # no public scan (wooden_spoon)
        pass
    scores = {}
    for tag, bundle in bundles.items():
        poses, meshes = scene(bundle, names, frames, work, tag)
        scores[tag] = S.score([e], {number: poses}, {(number, n): p for n, p in meshes.items()})
    auc = {k: v["AUC"] for k, v in scores.items()}
    attribution = {"stage 2 repairs": auc["stage2"] - auc["raw"],
                   "Path B stereo scale": auc["stage2_stereo"] - auc["stage2"]}
    if "stage2_scan" in auc:
        attribution["scale still missing (oracle - stereo)"] = auc["stage2_scan"] - auc["stage2_stereo"]
        attribution["tracking, mesh and placement (1 - oracle)"] = 1.0 - auc["stage2_scan"]
    row = {"episode": number, "objects": names, "stereo_factor": factor, "variants": list(VARIANTS),
           "scores": scores, "auc_attribution": attribution}
    write_json(root / "waterfall.json", row)
    reports = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/report.json"))}
    loss.write(root, dict(sorted(reports.items(), key=lambda kv: stage_order(kv[0]))))
    return row


def stage_order(name: str) -> int:
    from .runner import STAGES
    return STAGES.index(name) if name in STAGES else len(STAGES)


def summary(out: Path, rows: list[dict]) -> str:
    lines = ["| Episode | Objects | Stereo factor | " + " | ".join(VARIANTS) + " |",
             "|---|---|---|" + "---|" * len(VARIANTS)]
    for r in rows:
        cells = [f"{r['scores'][v]['AUC']:.3f}" if v in r["scores"] else "" for v in VARIANTS]
        lines.append(f"| {r['episode']} | {' + '.join(r['objects'])} | {r['stereo_factor']:.3f} | " + " | ".join(cells) + " |")
    means = [np.mean([r["scores"][v]["AUC"] for r in rows if v in r["scores"]]) for v in VARIANTS]
    lines.append("| **mean AUC** | | | " + " | ".join(f"**{m:.3f}**" for m in means) + " |")
    text = "\n".join(lines) + "\n"
    write_json(out / "waterfall.json", rows)
    (out / "waterfall.md").write_text("# Path B score waterfall (public episodes)\n\n" + text)
    return text
