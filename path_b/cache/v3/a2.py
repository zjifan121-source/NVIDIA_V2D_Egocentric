"""Phase A2.5 (V3_BUILD_SPEC.md §11): build the S3 variants, score them, apply the decision rule.

    uv run --project path_b python path_b/run.py a2 --episodes 40 26 --out <run>

<run> holds phase A's chain plus s3a2_fuse_scan, the canonical meshes (<run>/canonical/) and
s3a2_register (and s3b2_track_points once A2.4 ran on a GPU). Each variant is a run folder beside it:
  <run>_a2_mesh_fp      canonical mesh + phase A's FoundationPose track re-expressed for it (A2.3's D)
  <run>_a2_mesh_points  canonical mesh + A2.4's mesh-free track (only where s3b2_track_points exists)
P0-P4 and the A2 stages are hard links to <run>'s (never rewritten); s3a_mesh and s3b_track are new
folders (the canonical mesh, the variant's poses, reports saying so); S3c, S4a, S4b, S4c, S5 and the
waterfall then rerun there (S6 on the VM). Scored like phase A: the best candidate's AUC (rollout, else
realised, else reference) and every object's CD-O (gt.cdo_cm, eval_mesh.py's metric).
Decision rule (spec A2.5): per dev episode, rank the variants by AUC (higher better) and by mean CD-O
(lower better), rank points n..1; the variant with the highest mean of the two ranks over the episodes
becomes S3; ties go to the deterministic variant (an A2 variant: no generative sampling); if every A2
variant loses to phase A on both AUC and CD-O, phase A stays. Written to <run>/a2_decision.json.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh

from . import gt
from .analysis import best_candidate
from .density import density
from .geometry import read_transforms, write_transform
from .paths import episode_info, git_commit
from .runner import run_episode, write_json

LINKED = ("p0_ingest", "p1_calibrate", "p2_depth", "p1b_masks", "p1c_camera", "p1d_hands", "p3_scale_bridge",
          "p4_export", "s3a2_fuse_scan", "s3a2_register", "s3b2_track_points")
RERUN = ("s3c_fuse", "s4a_phases", "s4b_reference", "s4c_retarget", "s5_task")
VARIANTS = ("a2_mesh_fp", "a2_mesh_points")


def report(root: Path, stage: str, e: dict, metrics: list[dict], outputs: dict, note: str) -> None:
    write_json(root / stage / "report.json", {
        "stage": stage, "status": "PASS", "episode": e["episode"], "split": e["split"], "commit": git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seconds": 0.0, "gpu_peak_mib": None,
        "metrics": [{"name": "phase A2 variant", "value": note, "unit": "", "kind": "health", "bar": "", "status": "INFO",
                     "note": ""}, *metrics],
        "outputs": outputs, "files_read": [], "skip_reason": None, "error": None, "a2_variant": note})


def build(src: Path, variant: str, number: int) -> Path | None:
    """The variant's episode folder (None when its inputs are missing)."""
    e = episode_info(number)
    names, n = [o["name"] for o in e["objects"]], int(e["num_frames"])
    s_root = src / f"episode_{number:06d}"
    canon = src / "canonical"
    if not all((canon / nm / "mesh.obj").is_file() and (s_root / f"s3a2_register/{nm}/registration.json").is_file()
               for nm in names):
        return None
    if variant == "a2_mesh_points" and not all((s_root / f"s3b2_track_points/{nm}/poses").is_dir() for nm in names):
        return None
    root = src.parent / f"{src.name}_{variant}" / f"episode_{number:06d}"
    root.mkdir(parents=True, exist_ok=True)
    for stage in LINKED:
        if (s_root / stage).is_dir() and not (root / stage).exists():
            subprocess.run(["cp", "-al", str(s_root / stage), str(root / stage)], check=True)
    for name in names:
        # S3a: the canonical mesh (its frame is the object frame from here on)
        m = trimesh.load(canon / name / "mesh.obj", force="mesh", process=False)
        d = root / f"s3a_mesh/{name}"
        d.mkdir(parents=True, exist_ok=True)
        # a plain material: the bundle writer copies the mtl an OBJ names, and the runner's merge reads it
        (d / "material.mtl").write_text("newmtl material_0\nKd 0.8 0.8 0.8\n")
        (d / "mesh.obj").write_text("mtllib material.mtl\nusemtl material_0\n"
                                    + trimesh.exchange.obj.export_obj(m, include_normals=False, include_texture=False))
        m.export(d / "mesh.glb")
        volume = float(m.volume if m.is_watertight else m.convex_hull.volume)
        write_json(d / "mass.json", {"volume_m3": volume, "volume_from": "mesh" if m.is_watertight else "convex hull",
                                     "density_kg_m3": density(name), "mass_kg": volume * density(name)})
        (d / "sha256.txt").write_text(hashlib.sha256((d / "mesh.glb").read_bytes()).hexdigest() + "  mesh.glb\n")
        # S3b: phase A's track re-expressed (T D), or A2.4's track (already for the canonical mesh)
        t = root / f"s3b_track/{name}"
        (t / "poses").mkdir(parents=True, exist_ok=True)
        if variant == "a2_mesh_fp":
            D = np.asarray(json.loads((s_root / f"s3a2_register/{name}/registration.json").read_text())["canonical_to_phase_a"])
            T, have = read_transforms(s_root / f"s3b_track/{name}/poses", n)
            for k in np.flatnonzero(have):
                write_transform(t / f"poses/{k:06d}.json", T[k] @ D)
            meta = json.loads((s_root / f"s3b_track/{name}/track_meta.json").read_text())
        else:
            T, have = read_transforms(s_root / f"s3b2_track_points/{name}/poses", n)
            for k in np.flatnonzero(have):
                write_transform(t / f"poses/{k:06d}.json", T[k])
            meta = json.loads((s_root / f"s3b2_track_points/{name}/track_meta.json").read_text())
        write_json(t / "track_meta.json", meta | {"a2_variant": variant})
    note = {"a2_mesh_fp": "canonical mesh + phase A's track re-expressed for it",
            "a2_mesh_points": "canonical mesh + A2.4's mesh-free track"}[variant]
    report(root, "s3a_mesh", e, [], {nm: root / f"s3a_mesh/{nm}/mesh.obj" for nm in names}, note)
    report(root, "s3b_track", e, [], {nm: root / f"s3b_track/{nm}/poses" for nm in names}, note)
    return root


def scores(root: Path, read: list) -> dict:
    """Best candidate's scores and every object's CD-O for one episode folder."""
    wf = json.loads((root / "waterfall.json").read_text())
    best = best_candidate(wf["scores"])
    e = episode_info(int(root.name[8:]))
    cdo = {}
    for o in e["objects"]:
        m = trimesh.load(root / f"s3a_mesh/{o['name']}/mesh.obj", force="mesh", process=False)
        cdo[o["name"]] = gt.cdo_cm(e, o["name"], np.asarray(m.vertices), np.asarray(m.faces), read)
    return {"best": best, "AUC": wf["scores"][best]["AUC"] if best else None,
            "MPPE cm": wf["scores"][best]["MPPE cm"] if best else None, "CD-O cm": cdo,
            "mean CD-O cm": float(np.mean(list(cdo.values())))}


def decide(table: dict[int, dict[str, dict]]) -> dict:
    """The spec's rank rule over the dev episodes."""
    names = sorted({v for row in table.values() for v in row})
    points = {v: [] for v in names}
    for row in table.values():
        present = [v for v in names if v in row and row[v]["AUC"] is not None]
        for key, better in (("AUC", max), ("mean CD-O cm", min)):
            order = sorted(present, key=lambda v: row[v][key], reverse=better is max)
            for i, v in enumerate(order):
                points[v].append(len(order) - i)
    mean = {v: float(np.mean(p)) for v, p in points.items() if p}
    top = max(mean.values())
    tied = [v for v, m in mean.items() if m == top]
    choice = next((v for v in tied if v != "phase_a"), tied[0])  # ties: the deterministic (A2) variant
    lost_both = all(
        all(row[v]["AUC"] < row["phase_a"]["AUC"] and row[v]["mean CD-O cm"] > row["phase_a"]["mean CD-O cm"]
            for row in table.values() if v in row and "phase_a" in row)
        for v in names if v != "phase_a")
    if lost_both:
        choice = "phase_a"
    return {"rank_points": points, "mean_rank": mean, "choice": choice, "tied": tied,
            "a2_lost_on_both": lost_both}


def main(src: Path, numbers: list[int]) -> Path:
    read: list = []
    table: dict[int, dict[str, dict]] = {}
    for number in numbers:
        row = {"phase_a": scores(src / f"episode_{number:06d}", read)}
        for variant in VARIANTS:
            root = build(src, variant, number)
            if root is None:
                print(f"episode {number}: {variant} not built (its inputs are missing)", flush=True)
                continue
            run_episode(number, root.parent, tuple(RERUN), force=True)  # its S3a/S3b were just rebuilt
            from .waterfall import episode as waterfall_episode
            waterfall_episode(root)
            row[variant] = scores(root, read)
        table[number] = row
    decision = decide(table)
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": git_commit(),
           "source_run": src.name, "episodes": table, "decision": decision,
           "stand_ins": "GPU-stage inputs of the source run are stand-ins" if any(
               json.loads(p.read_text()).get("stand_in") for p in src.glob("episode_*/s3b_track/report.json")) else None,
           "files_read": sorted({str(p) for p in read})}
    write_json(src / "a2_decision.json", out)
    return src / "a2_decision.json"
