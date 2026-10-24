"""S5 task file: FlashCHORD's ManoSharpa parquet per candidate (V4_BUILD_SPEC.md S5).

Each candidate's scene comes from the team's runner (tools/track3/run_load_retarget.py in $RETARGET_VENV,
through pathb/drivers/run_lr.py: no public scan on evaluation episodes, $RETARGET_DEVICE, $MANO_DIR):
object URDF and mesh, support USD, the MANO fields and an IK robot.
- Mode B (a candidate S4c ran on, when its gate takes the realised trajectory): the runner on S4c's
  realised bundle with --no-rest-hold --no-object-clean --no-level (the object stays where the physics
  left it, next to DaiD's hands); then the parquet's robot and contacts become DaiD's: wrists, the 22
  finger joints per hand, robot_{side}_frames by pathb/fk.py, contacts in the 16 MANO-link slots
  (object-side point, normal into the object, as contact_utils writes them; link_contact_* = the
  robot-side point). A hand DaiD did not retarget (one-hand embodiment) keeps the IK robot. The runner's
  world is the bundle's shifted (the loader with --no_ground_align centres the object in xy and puts
  its bottom at --ground-z-offset); the shift is measured from the object and checked constant.
  Written as a sibling partition sequence_id=<seq>_daid with the support USD linked under that name
  (FlashCHORD finds it by the sequence id); robot_{side}_frame_task_errors = distance of the 11 task
  frames to the IK robot's (how far physics moved the hands from the kinematic retarget).
- Mode A (fallback): the runner on S4b's kinematic bundle with --no-rest-hold (v3 holds already);
  still: S4b's still bundle with --still-object.
- Two-object episodes: mode A on the merged scene (--merge over each object's bundle: S4c's realised
  one where its gate says so, else S4b's); each object's DaiD run has hands of its own, so one task
  cannot take both (v3's phase-2 joint retarget is cached, V4 §8).
Checks: the runner's on the partition S6 trains on (frames, no robot link > 2 cm inside an object,
frame-0 seating gap < 1 cm, a support disk); mode B also the hand-over: every DaiD contact's robot-side
point, carried by the task file's robot (fk.py), within 1 cm of the object surface on >= 90 % of
contact frames. FlashCHORD's kinematic replay (scripts/debug/replay.py) is the GPU check (VM).
Candidates: S4b's moving one (s4b_reference/moving.json: inverted, or keyframe when it fell back) and
still (V4 S6: two policies); every candidate with a bundle when S4b wrote no choice (v3 runs).
Out: s5_task/<candidate>/run/ (runner), task.json {mode, partition, sequence_id, support_usd, ...}.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

from .. import fk, gt
from ..reference import CANDIDATES
from ..paths import PATH_B, REPO, TRACK3_FPS
from ..team import lr_command
from ..report import Metric, StageResult
from ..runner import write_json
from .s23_load_retarget import worst_seating

FLAGS = {"A": ["--no-rest-hold"], "B": ["--no-rest-hold", "--no-object-clean", "--no-level"],
         "still": ["--no-rest-hold", "--still-object"]}
CONTACT_M = 0.01  # hand-over bar: robot-side contact point to the object surface
CONTACT_SHARE = 0.9
SIDES = ("right", "left")


def retarget_venv() -> str:
    venv = os.environ.get("RETARGET_VENV")
    if not venv:
        raise RuntimeError("set RETARGET_VENV to the stages 2-3 environment (README §5.12)")
    return venv


def run_logged(cmd: list, log: Path, env: dict | None = None) -> None:
    with open(log, "a") as f:
        rc = subprocess.run([str(c) for c in cmd], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{Path(str(cmd[1])).name} failed (exit {rc}): " + "\n".join(log.read_text().splitlines()[-15:]))


def choose(ctx, cand: str) -> dict:
    """Mode and bundle per object for one candidate."""
    bundles, realised, missing_gate = {}, [], []
    for name in ctx.objects:
        bundles[name] = ctx.root / f"s4b_reference/{name}/{cand}"
        gate = ctx.root / f"s4c_retarget/{name}/{cand}/gate.json"
        if cand == "still":
            continue
        if not gate.is_file():
            missing_gate.append(name)
        elif json.loads(gate.read_text())["use_realised"]:
            bundles[name] = ctx.root / f"s4c_retarget/{name}/{cand}/bundle"
            realised.append(name)
    if cand == "still":
        return {"mode": "still", "bundles": bundles, "reason": "S4b still bundle with --still-object"}
    if len(ctx.objects) == 1 and realised:
        return {"mode": "B", "bundles": bundles, "reason": "S4c gate: realised trajectory (Do as I Do)"}
    if len(ctx.objects) > 1:
        return {"mode": "A", "bundles": bundles,
                "reason": f"two objects: merged scene, realised for {realised or 'none'}"}
    elif missing_gate:
        why = "S4c did not run on this candidate (off, or Do as I Do not installed): kinematic reference"
    else:
        why = "S4c gate kept the kinematic reference"
    return {"mode": "A", "bundles": bundles, "reason": why}


def runner(ctx, bundles: dict, flags: list, seq: str, out: Path) -> dict:
    """run_load_retarget.py on the candidate's bundle(s) (team.lr_command); its report."""
    args = ["--episode", ctx.number, "--sequence-id", seq, "--out", out, "--skip-video", *flags]
    if len(bundles) == 1:
        name, bundle = next(iter(bundles.items()))
        args += ["--object", name, "--bundle", bundle]
    else:
        args += ["--merge"]
    cmd, env = lr_command(ctx.public, args)
    if len(bundles) > 1:
        ns = out.parent / "ns" / f"episode_{ctx.number:06d}"
        ns.mkdir(parents=True, exist_ok=True)
        for name, bundle in bundles.items():
            link = ns / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path(bundle).resolve())
        env["V2D_BUNDLE_NS"] = str(ns.parent)  # run_load_retarget.bundle_dir_for: an absolute path replaces the hub
    if out.exists():
        shutil.rmtree(out)  # inputs changed, or this stage would have been kept
    log = out.parent / "runner.log"
    log.write_text("")
    run_logged(cmd, log, env)
    if ctx.public:
        ctx.files_read.extend(p for p in (gt.scan(n) for n in bundles) if p)  # its bundle check reads the scans
    return json.loads((out / "report.json").read_text())


def parquet_file(partition) -> Path:
    return next(Path(partition).glob("*.parquet"))


def read_cells(path: Path) -> tuple[pa.Table, dict]:
    table = pq.read_table(path)
    return table, {c: table.column(c)[0].as_py() for c in table.column_names}


def matrices(pos: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(np.asarray(wxyz)[:, [1, 2, 3, 0]]).as_matrix()
    T[:, :3, 3] = pos
    return T


def world_shift(cells: dict, object_to_world: np.ndarray) -> np.ndarray:
    """The runner's world minus the bundle's (a shift; checked constant over the clip)."""
    T_p = matrices(np.asarray(cells["object_body_position"])[:, 0], np.asarray(cells["object_body_wxyz"])[:, 0])
    W = T_p @ np.linalg.inv(np.asarray(object_to_world, dtype=np.float64))
    shift = np.median(W[:, :3, 3], axis=0)
    rot, drift = np.abs(W[:, :3, :3] - np.eye(3)).max(), np.abs(W[:, :3, 3] - shift).max()
    if rot > 1e-4 or drift > 1e-3:
        raise RuntimeError(f"the runner's world is not the bundle's shifted (rotation {rot:.2g}, drift {drift * 1000:.2f} mm): "
                           "mode B cannot place DaiD's hands")
    return shift


def robot_points(traj: dict, side: str, poses: dict[str, np.ndarray] | None, shift: np.ndarray) -> np.ndarray:
    """(n, 16, 3) robot-side contact points: in DaiD's replay (poses None) or carried by FK poses."""
    names = [str(b) for b in traj["body_names"]]
    body, local = traj[f"{side}_contact_body"], traj[f"{side}_contact_robot_local"]
    out = np.zeros(local.shape)
    for f, s in zip(*np.nonzero(body >= 0)):
        name = names[body[f, s]]
        if poses is None:
            p = traj["body_pose"][f, body[f, s]]
            out[f, s] = Rotation.from_quat(p[[4, 5, 6, 3]]).apply(local[f, s]) + p[:3] + shift
        else:
            T = poses[name][f]
            out[f, s] = T[:3, :3] @ local[f, s] + T[:3, 3]
    return out


def mode_b_cells(cells: dict, traj: dict, shift: np.ndarray) -> tuple[dict, dict]:
    """The parquet columns that become DaiD's; and per side what changed (for the report)."""
    n = len(cells["object_body_position"])
    joint_names = [str(j) for j in traj["joint_names"]]
    new, info = {}, {}
    for side in SIDES:
        if f"{side}_wrist_position" not in traj:
            info[side] = {"robot": "IK (not in DaiD's scene)"}
            continue
        fingers = [str(j) for j in cells[f"{side}_robot_finger_joint_names"]]
        q = np.stack([traj["robot_qpos"][:, joint_names.index(j)] for j in fingers], axis=1)
        wp, wq = traj[f"{side}_wrist_position"] + shift, traj[f"{side}_wrist_wxyz"]
        hand = fk.sharpa(side)
        if hand.names != [str(x) for x in cells[f"{side}_robot_frame_names"]]:
            raise RuntimeError(f"{side} frame names differ from fk.py's (the retargeter's MJCF changed?)")
        joints = dict(zip(fingers, q.T))
        frames = hand.frames(wp, wq, joints)
        tasks = [hand.names.index(str(t)) for t in cells[f"{side}_robot_frame_task_names"]]
        moved = np.linalg.norm(frames[:, tasks, :3] - np.asarray(cells[f"robot_{side}_frames"])[:, tasks, :3], axis=-1)
        active = traj[f"{side}_contact_part_ids"] > 0
        new.update({
            f"robot_{side}_wrist_position": wp, f"robot_{side}_wrist_wxyz": wq, f"robot_{side}_finger_joints": q,
            f"robot_{side}_frames": frames, f"robot_{side}_frame_task_errors": moved,
            f"robot_{side}_num_optimization_iterations": np.zeros(n, np.int32),  # not IK
            f"mano_{side}_object_contact_positions": np.where(active[..., None], traj[f"{side}_contact_positions"] + shift, 0.0),
            f"mano_{side}_object_contact_normals": np.where(active[..., None], traj[f"{side}_contact_normals"], 0.0),
            f"mano_{side}_object_contact_part_ids": traj[f"{side}_contact_part_ids"].astype(np.int32),
            f"mano_{side}_link_contact_positions": robot_points(traj, side, None, shift),
            f"mano_{side}_link_contact_normals": np.where(active[..., None], traj[f"{side}_contact_normals"], 0.0),
        })
        info[side] = {"robot": "Do as I Do", "contact_frames": int(active.any(axis=1).sum()),
                      "task_frames_vs_ik_cm": {"mean": round(float(moved.mean()) * 100, 2),
                                               "max": round(float(moved.max()) * 100, 2)},
                      "_poses": hand.poses(wp, wq, joints)}
    return new, info


def write_partition(table: pa.Table, new: dict, dest: Path) -> None:
    """The source row with some columns replaced, types as the source's."""
    cols = []
    for field in table.schema:
        if field.name in new:
            value = new[field.name]
            cols.append(pa.array([value.tolist() if isinstance(value, np.ndarray) else value], type=field.type))
        else:
            cols.append(table.column(field.name))
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(cols, schema=table.schema), dest)


def handover(cells: dict, traj: dict, info: dict, mesh_paths: list[Path], shift: np.ndarray) -> dict:
    """DaiD's robot-side contact points carried by the task file's robot: distance to the surface of the
    object each touched (part id 1, 2: the task's object bodies in order)."""
    meshes = [trimesh.load(m, force="mesh", process=False) for m in mesh_paths]
    pos, wxyz = np.asarray(cells["object_body_position"]), np.asarray(cells["object_body_wxyz"])
    T_o = [matrices(pos[:, b], wxyz[:, b]) for b in range(len(meshes))]
    worst = np.full(len(pos), -1.0)
    for side in SIDES:
        if "_poses" not in info[side]:
            continue
        pts = robot_points(traj, side, info[side]["_poses"], shift)
        part = traj[f"{side}_contact_part_ids"]
        for b, (mesh, T) in enumerate(zip(meshes, T_o)):
            f, s = np.nonzero((traj[f"{side}_contact_body"] >= 0) & (part == b + 1))
            if not len(f):
                continue
            local = np.einsum("nji,nj->ni", T[f, :3, :3], pts[f, s] - T[f, :3, 3])
            np.maximum.at(worst, f, mesh.nearest.on_surface(local)[1])
    frames = worst >= 0
    share = float(np.mean(worst[frames] <= CONTACT_M)) if frames.any() else None
    return {"contact_frames": int(frames.sum()), "within_1cm_share": share,
            "median_cm": round(float(np.median(worst[frames])) * 100, 2) if frames.any() else None,
            "pass": share is not None and share >= CONTACT_SHARE}


def lr_check(partition: Path, meshes: list, usda: Path, frames: int, out: Path) -> dict:
    request = out.with_suffix(".request.json")
    write_json(request, {"partition": partition, "meshes": meshes, "support_usd": usda, "frames": frames,
                         "fps": TRACK3_FPS, "out": out})
    run_logged([f"{retarget_venv()}/bin/python", PATH_B / "pathb/drivers/lr_check.py", request], out.with_suffix(".log"))
    return json.loads(out.read_text())


def mode_b(ctx, d: Path, choice: dict, report: dict, seq: str) -> dict:
    """One object, or (phase 2) the two-object scene: S4c's trajectory sits in the first object's folder
    (a link to the scene's), and the merged task's bodies follow the roster."""
    name = ctx.objects[0]
    traj = dict(np.load(ctx.root / f"s4c_retarget/{name}/{choice['cand']}/trajectory_mjwp.npz"))
    src = parquet_file(report["processed"])
    table, cells = read_cells(src)
    bundle = np.load(Path(choice["bundles"][name]) / "result.npz")
    shift = world_shift(cells, bundle["object_to_world_transform"])
    new, info = mode_b_cells(cells, traj, shift)
    seq_b = f"{seq}_daid"
    processed = Path(report["processed"]).parent.parent  # .../processed/sequence_id=<seq>/robot_name=...
    dest = processed / f"sequence_id={seq_b}" / Path(report["processed"]).name / src.name
    write_partition(table, new, dest)
    usda = Path(report["support_usd"])
    link = usda.with_name(usda.name.replace(seq, seq_b, 1))
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(usda.name)
    _, cells_b = read_cells(dest)
    meshes = [processed / f"{seq}.obj"] if len(ctx.objects) == 1 else [processed / f"{seq}_{o}.obj" for o in ctx.objects]
    over = handover(cells_b, traj, info, meshes, shift)
    check = lr_check(dest.parent, meshes, link, int(ctx.episode["num_frames"]), d / "check_b.json")
    for side in SIDES:
        info[side].pop("_poses", None)
    return {"partition": dest.parent, "sequence_id": seq_b, "support_usd": link, "world_shift_m": shift,
            "sides": info, "handover": over, "output_check": check, "ik_partition": report["processed"]}


def run(ctx) -> StageResult:
    d = ctx.dir("s5_task")
    metrics, outputs = [], {}
    chosen = ctx.root / "s4b_reference/moving.json"
    train = ([json.loads(chosen.read_text())["candidate"], "still"] if chosen.is_file() else list(CANDIDATES))
    for cand in [c for c in CANDIDATES if c in train]:  # V4 S6: the moving candidate S4b chose, and still
        if not all((ctx.root / f"s4b_reference/{o}/{cand}/result.npz").is_file() for o in ctx.objects):
            metrics.append(Metric(f"{cand}: task", "none", "", note="no S4b reference for this candidate"))
            continue
        choice = dict(choose(ctx, cand), cand=cand)
        cd = d / cand
        cd.mkdir(exist_ok=True)
        # body names end with the object's name (a merged scene's do: <seq>_<object>), which the team
        # packer's relabel needs (tools/track3/pack_submission.py)
        seq = f"ep{ctx.number}_{cand}" + (f"_{ctx.objects[0]}" if len(ctx.objects) == 1 else "")
        report = runner(ctx, choice["bundles"], FLAGS[choice["mode"]] + choice.get("extra_flags", []), seq, cd / "run")
        task = {"candidate": cand, "mode": choice["mode"], "reason": choice["reason"], "bundles": choice["bundles"],
                "flags": FLAGS[choice["mode"]], "partition": report["processed"], "sequence_id": seq,
                "support_usd": report["support_usd"], "output_check": report["output_check"],
                "bundle_warnings": report["bundle_check"]["warnings"]}
        if choice["mode"] == "B":
            task.update(mode_b(ctx, cd, choice, report, seq))
        write_json(cd / "task.json", task)
        outputs[cand] = cd / "task.json"
        check = task["output_check"]
        metrics += [
            Metric(f"{cand}: mode", choice["mode"], "", note=choice["reason"]),
            Metric(f"{cand}: frames out minus manifest", check["frames"] - int(ctx.episode["num_frames"]), "frames",
                   ok=(0, 0)),
            Metric(f"{cand}: frames with a robot link > 2 cm inside an object", check["penetration_frames"]["over_2cm"],
                   "", ok=(0, 0), warn=(0, 0.02)),
            Metric(f"{cand}: support disks", check["support_disks"], "", ok=(1, np.inf)),
            Metric(f"{cand}: frame-0 seating gap (worst object)", worst_seating(check["frame0_seating_cm"]), "cm",
                   ok=(-1, 1), warn=(-2, 2), note="object bottom minus the support top under it"),
        ]
        if choice["mode"] == "B":
            over = task["handover"]
            metrics.append(Metric(f"{cand}: DaiD contacts within 1 cm of the object in the task file",
                                  over["within_1cm_share"], "", ok=(CONTACT_SHARE, 1),
                                  note=f"{over['contact_frames']} contact frames; median worst {over['median_cm']} cm"))
            for side, s in task["sides"].items():
                if "task_frames_vs_ik_cm" in s:
                    metrics.append(Metric(f"{cand}: {side} task frames vs the kinematic retarget (mean)",
                                          s["task_frames_vs_ik_cm"]["mean"], "cm",
                                          note=f"max {s['task_frames_vs_ik_cm']['max']} cm: how far physics moved the hand"))
    return StageResult(metrics, outputs)
