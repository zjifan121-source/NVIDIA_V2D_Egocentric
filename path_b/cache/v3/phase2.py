"""Phase 2 (V3_BUILD_SPEC.md §8): the two-object scope, on ep26 first (James, 2026-10-22: build it now,
after A2). On only for two-object episodes of a run made with PATHB_PHASE2=1 (run.py p2 sets it), so
phase 1's mode-A merge stays the comparison:
- S4a: the support relation per rest window (support.json): each object's support disk (centre and
  radius of its footprint, height of its lowest point) at the window's frame, and the team's own rule,
  tools/track3/support_stacked.py drop_stacked_supports (in $RETARGET_VENV, drivers/support_relation.py):
  an object whose disk sits more than 1 cm above another's and inside its circle rests on it.
- S4b: the roster-first object is the scene anchor (Kaggle aligns a scene by its frame-0 pose); every
  other object's rest anchor becomes the anchor object's times their relative pose, the median over the
  frames both rest of S3b's camera-frame relative pose (the camera cancels, so camera drift does not move
  one object against the other).
- S4c: Do as I Do's own bimanual scene with two free objects (right_object = the roster-first object,
  left_object = the other; generate_scene builds both from task_info), one run per candidate.
- S5: mode A over the merged scene, with --seat when an object rests on another; mode B with both
  bodies when S4c's gate takes the realised scene.
- S6: two-object selection adds RPE (public ties: RPE before MPPE; evaluation: the relative-position
  error between the objects joins the observation cost).
run.py p2 --episodes 26 --out <run>: <run>_p2 beside <run> (P0-S3c linked), S4a-S5 rerun with the flag,
its waterfall; <run>'s waterfall gains p2_* columns.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .paths import PATH_B, episode_info
from .runner import run_episode, write_json

FLAG = "PATHB_PHASE2"
LINKED = ("p0_ingest", "p1_calibrate", "p2_depth", "p1b_masks", "p1c_camera", "p1d_hands", "p3_scale_bridge",
          "p4_export", "s3a_mesh", "s3b_track", "s3c_fuse")
RERUN = ("s4a_phases", "s4b_reference", "s4c_retarget", "s5_task")


def enabled(ctx) -> bool:
    return os.environ.get(FLAG) == "1" and len(ctx.objects) > 1


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


def relative_anchor(T_cam_a: np.ndarray, T_cam_b: np.ndarray, frames: list[int]) -> np.ndarray | None:
    """Median translation and chordal-mean rotation of inv(A_t) B_t over the frames (camera frame)."""
    if not frames:
        return None
    rel = np.linalg.inv(T_cam_a[frames]) @ T_cam_b[frames]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_matrix(rel[:, :3, :3]).mean().as_matrix()
    out[:3, 3] = np.median(rel[:, :3, 3], axis=0)
    return out


def common_rest(phases: dict[str, dict], a: str, b: str, key: str, have: np.ndarray) -> list[int]:
    """Frames of b's rest window `key` where a rests too (any of a's rest windows) and both have poses."""
    wb = phases[b].get(key)
    if not wb:
        return []
    rest_a = set()
    for k in ("rest_before", "rest_after"):
        w = phases[a].get(k)
        if w:
            rest_a.update(range(w[0], w[1] + 1))
    return [t for t in range(wb[0], wb[1] + 1) if t in rest_a and have[t]]


def build(src: Path, number: int) -> Path:
    import subprocess as sp
    root = src.parent / f"{src.name}_p2" / f"episode_{number:06d}"
    s_root = src / f"episode_{number:06d}"
    root.mkdir(parents=True, exist_ok=True)
    for stage in LINKED:
        if (s_root / stage).is_dir() and not (root / stage).exists():
            sp.run(["cp", "-al", str(s_root / stage), str(root / stage)], check=True)
    return root


def main(src: Path, numbers: list[int]) -> list[Path]:
    """Phase 2 on the two-object episodes among numbers: <src>_p2, S4a-S5 rerun with the flag, waterfall."""
    from .waterfall import episode as waterfall_episode
    os.environ[FLAG] = "1"
    done = []
    for number in numbers:
        if len(episode_info(number)["objects"]) < 2:
            print(f"episode {number}: one object, no phase 2", flush=True)
            continue
        root = build(src, number)
        run_episode(number, root.parent, RERUN)
        waterfall_episode(root)
        done.append(root)
    return done
