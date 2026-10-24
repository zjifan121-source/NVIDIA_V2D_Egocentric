"""S0.1: a FlashCHORD policy trained on the ground-truth reference (V4_BUILD_SPEC.md S0.1), and
`run.py oracle --variants` (S0.2's new variants, CPU, public only; variants() below).

Per episode (public only, through pathb/gt.py), on a GPU machine:
  1. bundle  the team bundle (hub; merged for several objects, as run_load_retarget --merge does)
             moved into the ground-truth world by one similarity (scale, yaw, translation: both
             worlds have gravity up) fitted from its object tracks to the ground truth; the hands and
             camera move with it, and each object is replaced by its ground-truth track with the
             public scan as mesh (this test only). One bundle per object, all sharing that camera and
             those hands, so the merge rebuilds the scene exactly
  2. task    run_load_retarget.py (with --merge for two objects; --no-rest-hold --no-object-clean
             --no-level keep the ground truth as it is; hand cleaning stays)
  3. train   FlashSAC with the still submission's settings (jobs/still_submit.sh: STEPS and EXTRA),
             contact buffers as spec S6 sets them for the L40S
  4. eval    the kit's run_policy_evaluation.py (official evaluator, one world)
  5. score   the rollout with the official metrics against public ground truth (scan meshes)
Decision rule (spec): rollout(GT) below the episode's still score -> fix the S6 recipe first; >= 0.7
-> the reference is the whole problem. Each step leaves a marker, so a rerun resumes.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh

from . import gt, policy
from .paths import PATH_B, REPO, TRACK3_FPS, episode_info
from .policy import run_logged, still_settings  # noqa: F401 (still_settings: tests, summary)
from .team import retarget_tools

STILL_SCORES = {40: 0.40, 26: 0.50}  # spec S0.1: the episode's still score (S0.2 oracle file)
LR_FLAGS = ["--no-rest-hold", "--no-object-clean", "--no-level", "--skip-video"]


def T_of(P: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(P[:, 3:]).as_matrix()
    T[:, :3, 3] = P[:, :3]
    return T


def yaw_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    """(s, A): dst ~ s * Rz src + t, as a scale and a 4x4 rigid A (rotation about z, translation),
    least squares over corresponding points (n, 3)."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - cs, dst - cd
    theta = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a[:, :2] * b[:, :2]).sum())
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    s = float((b * (a @ Rz.T)).sum() / (a * a).sum())
    A = np.eye(4)
    A[:3, :3] = Rz
    A[:3, 3] = cd - s * Rz @ cs
    return s, A


def to_world(data: dict, s: float, A: np.ndarray) -> dict:
    """Every length times s (run_load_retarget.rescale_world), then every *_to_world pose moved by A."""
    L = retarget_tools()
    data = dict(data)
    L.rescale_world(data, s)
    for key in ("camera_to_world_transform", "hand_left_wrist_to_world_transform",
                "hand_right_wrist_to_world_transform", *[k for k, _, _ in L.object_slots(data)]):
        data[key] = (A @ data[key].astype(np.float64)).astype(data[key].dtype)
    return data


def gt_bundles(e: dict, work: Path, read: list) -> tuple[Path, dict]:
    """The per-object ground-truth bundles in work/ns/episode_XXXXXX/<object>/ and the fit."""
    L = retarget_tools()
    number, names = e["episode"], [o["name"] for o in e["objects"]]
    if len(names) > 1:
        src, merge = L.merged_bundle(number, work / "hub_merged")
    else:
        src, merge = L.bundle_dir_for(number, names[0]), None
    data = dict(np.load(src / "result.npz"))
    truth = gt.ground_truth(e, read)  # (frames, objects, 7)
    n = min(len(truth), len(data["camera_to_world_transform"]))
    slots = L.object_slots(data)
    ours = np.concatenate([data[k][:n, :3, 3].astype(np.float64) for k, _, _ in slots])
    theirs = np.concatenate([truth[:n, i, :3] for i in range(len(slots))])
    spread = float(np.linalg.norm(theirs - theirs.mean(axis=0), axis=1).max())
    if spread < 0.05:
        raise RuntimeError(f"episode {number}: ground-truth positions span {spread * 100:.1f} cm; yaw is undetermined")
    s, A = yaw_similarity(ours, theirs)
    world = to_world(data, s, A)
    resid = np.linalg.norm(np.concatenate([world[k][:n, :3, 3] for k, _, _ in slots]) - theirs, axis=1)
    c2w = world["camera_to_world_transform"].astype(np.float64)
    ns = work / "ns" / f"episode_{number:06d}"
    keep = [k for k in world if not (k.startswith("object_") and k[7:8].isdigit()) and k != "object_names"]
    for i, name in enumerate(names):
        dest = ns / name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        T = T_of(truth[:, i])
        if len(T) < len(c2w):
            T = np.concatenate([T, np.repeat(T[-1:], len(c2w) - len(T), axis=0)])
        one = {k: world[k] for k in keep}
        one["object_to_world_transform"] = T[:len(c2w)].astype(np.float32)
        one["object_to_camera_transform"] = (np.linalg.inv(c2w) @ T[:len(c2w)]).astype(np.float32)
        one["object_scale"] = np.asarray(1.0, dtype=np.float32)
        one["object_is_valid"] = np.ones(len(c2w), dtype=bool)
        np.savez(dest / "result.npz", **one)
        scan = trimesh.load(gt.scan_obj(e, name, work / "scans" / name, read), force="mesh", process=False)
        scan.export(dest / "mesh.obj")  # writes material.mtl and its texture beside it
        if not (dest / "material.mtl").is_file():
            (dest / "material.mtl").write_text("newmtl material_0\n")
        shutil.copy(src / "manifest.json", dest / "manifest.json")
    fit = {"scale": s, "yaw_deg": float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), "translation": A[:3, 3],
           "residual_cm_median": float(np.median(resid) * 100), "residual_cm_p90": float(np.percentile(resid, 90) * 100),
           "source_bundle": str(src), "merge": merge}
    return ns, fit


def episode(number: int, out: Path) -> dict:
    e = episode_info(number)
    gt.allowed(e)
    work = out / f"episode_{number:06d}"
    work.mkdir(parents=True, exist_ok=True)
    state_file = work / "state.json"
    state = json.loads(state_file.read_text()) if state_file.is_file() else {"episode": number, "files_read": []}
    read: list = []
    names = [o["name"] for o in e["objects"]]
    seq = f"s01_ep{number}"
    lr = work / "lr"
    retarget = os.environ.get("RETARGET_VENV") or "/root/v2d/venvs/retarget"

    def save():
        state["files_read"] = sorted(set(state["files_read"]) | {str(p) for p in read})
        state_file.write_text(json.dumps(state, indent=1, default=str) + "\n")

    if "fit" not in state:
        ns, state["fit"] = gt_bundles(e, work, read)
        state["ns"] = str(ns)
        save()
    if "partition" not in state:
        cmd = [f"{retarget}/bin/python", REPO / "tools/track3/run_load_retarget.py", "--episode", number,
               "--sequence-id", seq, "--out", lr, *LR_FLAGS]
        env = dict(os.environ)
        if len(names) > 1:
            cmd += ["--merge"]
            env["V2D_BUNDLE_NS"] = str(Path(state["ns"]).parent)
        else:
            cmd += ["--object", names[0], "--bundle", Path(state["ns"]) / names[0]]
        run_logged(cmd, work / "lr.log", REPO, env)
        read.extend(p for p in (gt.scan(n) for n in names) if p)  # its bundle check reads the scans
        report = json.loads((lr / "report.json").read_text())
        state["partition"], state["lr_check"] = report["processed"], report["output_check"]
        save()
    run = work / "flash"
    if "checkpoint" not in state:
        info = policy.train(state["partition"], run, seq)
        state.update(checkpoint=info["checkpoint"], train_seconds=info["seconds"],
                     nefc_overflow_lines=info["nefc_overflow_lines"])
        save()
    policy.evaluate(state["checkpoint"], number, run)
    from .team import scorer
    poses = scorer().evaluator_poses(run / "results" / f"episode_{number:06d}.parquet", e)  # parses the rollout
    meshes = {name: lr / f"score_input/meshes/episode_{number:06d}/{name}.obj" for name in names}
    state["scores"] = gt.score(e, poses, meshes, read)
    state["still"] = STILL_SCORES.get(number)
    auc = state["scores"]["AUC"]
    state["decision"] = ("S6 recipe first (rollout(GT) < still)" if state["still"] is not None and auc < state["still"]
                         else "reference is the whole problem (>= 0.7)" if auc >= 0.7 else "proceed as planned")
    save()
    return state


S02_FILE = PATH_B / "oracles/oracle_2026-10-17.json"  # Edmond's S0.2 regime split (tools/track3/playback_oracle.py)
VARIANTS = ("place_pos_rot_gt",)


def place_pos_rot_gt(e: dict, read: list) -> dict:
    """True positions and rotations (V4 S0.2, James 2026-10-24), the public scans as meshes, through the
    official metric code. S0.2 scores every variant but *_ourmesh with the scan mesh and converts poses
    through the scan's anchor and back (playback_oracle.to_anchor / to_native), so this is its `gt`
    variant, scored there as `gt_scan`; computed here, not copied."""
    meshes = {o["name"]: gt.scan(o["name"]) for o in e["objects"]}
    return gt.score(e, gt.ground_truth(e, read), meshes, read)


def variants(numbers: list[int], names: list[str]) -> Path:
    """`run.py oracle --variants`: each named variant per public episode, next to S0.2's place_pos_gt
    (true positions, our rotations); the gap is the rotation loss once position is fixed. Writes
    oracles/oracle_variants_<date>.json (per episode, and means)."""
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; known: {VARIANTS}")
    s02 = {}
    for r in json.loads(S02_FILE.read_text())["variants"]:
        s02.setdefault(r["ep"], {})[r["v"]] = r
    rows = []
    for number in numbers:
        e = episode_info(number)
        gt.allowed(e)  # public only: raises on an evaluation episode (GT firewall)
        read: list = []
        row = {"episode": number, "objects": [o["name"] for o in e["objects"]],
               "place_pos_gt": s02.get(number, {}).get("place_pos_gt", {}).get("AUC"),
               "gt_scan_s02": s02.get(number, {}).get("gt_scan", {}).get("AUC")}
        for name in names:
            row[name] = {"place_pos_rot_gt": place_pos_rot_gt}[name](e, read)
        auc = row["place_pos_rot_gt"]["AUC"] if "place_pos_rot_gt" in row else None
        row["rotation_loss_auc"] = auc - row["place_pos_gt"] if auc is not None and row["place_pos_gt"] is not None else None
        row["files_read"] = sorted({str(p) for p in read})
        rows.append(row)
        print(f"   ep{number}: " + ", ".join(f"{n} AUC {row[n]['AUC']:.3f}" for n in names)
              + f"; place_pos_gt {row['place_pos_gt']}; rotation loss {row['rotation_loss_auc']}", flush=True)

    def mean(key):
        vals = [r[key] if not isinstance(r[key], dict) else r[key]["AUC"] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source_s02": S02_FILE.name,
               "variants": list(names), "episodes": rows,
               "mean": {k: mean(k) for k in (*names, "place_pos_gt", "rotation_loss_auc")}}
    path = PATH_B / f"oracles/oracle_variants_{datetime.now(timezone.utc):%Y-%m-%d}.json"
    path.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return path


def main(numbers: list[int], out: Path | None = None) -> int:
    out = out or PATH_B / "data/oracles_s01"
    rows = []
    for number in numbers:
        print(f"== S0.1 episode {number}", flush=True)
        rows.append(episode(number, out))
        r = rows[-1]
        print(f"   AUC {r['scores']['AUC']:.3f} (still {r['still']}), MPPE {r['scores']['MPPE cm']:.2f} cm: "
              f"{r['decision']}", flush=True)
    summary = PATH_B / f"oracles/s01_{datetime.now(timezone.utc):%Y-%m-%d}.json"
    summary.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "settings": dict(zip(("steps", "overrides"), still_settings())),
                                   "episodes": rows}, indent=1, default=str) + "\n")
    print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main([int(a) for a in sys.argv[1:]]))
