"""S4c retarget: Do as I Do physics retargeting of the kinematic references (V3_BUILD_SPEC.md S4c).

Per object, keyframe first, then tracking: the candidate's hands in the world (drivers/mano_world.py,
$RETARGET_VENV), the reference at 50 Hz in DaiD's world (pathb/daid.py), then DaiD's stages 2-5
with the Sharpa hands in MuJoCo Warp (drivers/daid_run.py, $DAID_PY, DaiD in $DAID_DIR), settings in
pathb/s4c.yaml; the warm-up weld lasts S4a's rest-before window. The Sharpa joints match FlashCHORD's
sharpawave URDF one to one (pathb/sharpa_map.json).
Out per object and candidate: trajectory_mjwp.npz (at the video frames, v3 world: robot joints, realised
object, wrists, contacts in ManoSharpa's 16 MANO-link slots), scene.xml (DaiD's), metrics.json,
bundle/ (the candidate's bundle with object_to_world = realised), gate.json.
Phase 2 (pathb/phase2.py, two objects): one DaiD scene per candidate with both objects (the roster-first
is DaiD's right_object, the other its left_object; one density, their mean), in s4c_retarget/scene/<cand>/;
each object's folder links its trajectory and holds its bundle and gate; the gate scores the scene. Gate: realised used if its playback AUC >= the kinematic one - 0.01 (public; evaluation
takes the public decision of its task type, oracles/s4c_gate.json) and it passes the checks:
penetration <= 5 mm, object moved <= 1 cm before the first reach, object speed <= 1.5 m/s.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml

from .. import daid, gt, phase2
from .. import phases as P
from ..paths import PATH_B, TRACK3_FPS
from ..report import Metric, StageResult
from ..runner import write_json
from ..waterfall import merged, scene
from .s4b_reference import write_bundle

CANDIDATES = ("keyframe", "tracking")  # keyframe first (spec S4b/S4c)
GATE_AUC = 0.01
MAX_PENETRATION = 0.005  # m
MAX_PRE_REACH = 0.01  # m
MAX_SPEED = 1.5  # m/s
GATE_FILE = PATH_B / "oracles/s4c_gate.json"  # public decisions per task type, for evaluation episodes


def env_paths() -> tuple[str, str, str]:
    retarget = os.environ.get("RETARGET_VENV", "/root/v2d/venvs/retarget")
    return (f"{retarget}/bin/python", os.environ.get("DAID_PY", "/root/v2d/venvs/daid/bin/python"),
            os.environ.get("DAID_DIR", "/root/v2d/third_party/do-as-i-do/retargeting"))


def run_logged(cmd: list, log, cwd=None) -> None:
    with open(log, "a") as f:
        rc = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{cmd[1]} failed (exit {rc}): " + "\n".join(open(log).read().splitlines()[-15:]))


def checks(T: np.ndarray, ph: dict, penetration: float | None) -> dict:
    before = ph["first_reach"] if ph["first_reach"] is not None else len(T)
    pre = float(np.linalg.norm(T[:before, :3, 3] - T[0, :3, 3], axis=1).max()) if before > 0 else 0.0
    speed = float((np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) * TRACK3_FPS).max())
    ok = (penetration is not None and penetration <= MAX_PENETRATION) and pre <= MAX_PRE_REACH and speed <= MAX_SPEED
    return {"penetration_m": penetration, "pre_reach_motion_m": pre, "max_speed_m_s": speed, "pass": bool(ok)}


def playback_auc(ctx, name: str, bundle, work, tag: str) -> float:
    """Scene AUC with this object's bundle (the other objects at their S4b candidate of the same name)."""
    frames = int(ctx.episode["num_frames"])
    if len(ctx.objects) == 1:
        poses, meshes = scene(bundle, ctx.objects, frames, work, tag)
    else:
        cand = tag.split("_")[-1]
        bundles = {o: (bundle if o == name else ctx.root / f"s4b_reference/{o}/{cand}") for o in ctx.objects}
        poses, meshes = scene(merged(ctx.number, bundles, work / tag), ctx.objects, frames, work, tag)
    return gt.score(ctx.episode, poses, meshes, ctx.files_read)["AUC"]


def job(ctx, out: Path, objs: list[str], cand: str) -> dict:
    """One Do as I Do run on candidate `cand` of one object, or (phase 2) of two objects in one scene
    (the first is DaiD's right_object, the second its left_object). Writes out/trajectory_mjwp.npz and
    every object's realised bundle at s4c_retarget/<object>/<cand>/bundle."""
    mano_py, daid_py, daid_dir = env_paths()
    first = objs[0]
    src = {o: ctx.root / f"s4b_reference/{o}/{cand}" for o in objs}
    ph = json.loads((ctx.root / f"s4a_phases/{first}/phases.json").read_text())
    mass = [json.loads((ctx.root / f"s3a_mesh/{o}/mass.json").read_text()) for o in objs]
    out.mkdir(parents=True, exist_ok=True)
    log = out / "daid.log"
    log.write_text("")
    run_logged([mano_py, PATH_B / "pathb/drivers/mano_world.py", src[first], out / "mano_world.npz",
                os.environ.get("MANO_DIR", PATH_B.parent / "baseline/reconstruction/data/weights/hand")], log)
    mano = dict(np.load(out / "mano_world.npz"))
    refs = [np.load(src[o] / "result.npz")["object_to_world_transform"].astype(np.float64) for o in objs]
    meta = daid.prepare(refs[0], mano, out / "input", TRACK3_FPS, second=refs[1] if len(objs) > 1 else None)
    write_json(out / "input/meta.json", meta)
    request = {"daid_dir": daid_dir, "input": str(out / "input/daid_input.npz"), "output_root": str(out / "daid"),
               "task": f"ep{ctx.number}_{'_'.join(objs)}_{cand}", "embodiment": meta["embodiment"],
               "mesh": str(ctx.root / f"s3a_mesh/{first}/mesh.obj"), "convex": str(ctx.root / f"s3a_mesh/{first}/coacd"),
               # one density per scene in DaiD's generate_scene: the mean of the objects' (phase 2)
               "object_density": float(np.mean([m["density_kg_m3"] for m in mass])),
               "warmup_duration": P.length(ph["rest_before"]) / TRACK3_FPS,
               "settings": str(PATH_B / "pathb/s4c.yaml"), "video_fps": TRACK3_FPS, "n_video": meta["n_video"]}
    if len(objs) > 1:
        request["second"] = {"mesh": str(ctx.root / f"s3a_mesh/{objs[1]}/mesh.obj"),
                             "convex": str(ctx.root / f"s3a_mesh/{objs[1]}/coacd")}
    write_json(out / "request.json", request)
    run_logged([daid_py, PATH_B / "pathb/drivers/daid_run.py", out / "request.json"], log, cwd=daid_dir)
    summary = json.loads((out / "daid/daid_summary.json").read_text())
    back = daid.read_back(summary["trajectory"], summary["qpos_joint_names"], meta)
    realised = dict(np.load(summary["realised_frames"]))
    offset = np.asarray(meta["offset"])
    tracks = [back["object_to_world"]] + ([back["second_object_to_world"]] if len(objs) > 1 else [])
    arrays = {"object_to_world": tracks[0], "robot_qpos": back["robot_qpos"], "objects": np.array(objs),
              "joint_names": np.array(back["joint_names"]), "fps": TRACK3_FPS,
              "body_names": realised["body_names"], "body_pose": realised["body_pose"]}
    if len(objs) > 1:
        arrays["second_object_to_world"] = tracks[1]
    arrays["body_pose"][..., :3] += offset
    contact_frames = {}
    for side in ("right", "left"):
        w = daid.wrist(realised, offset, side)
        if w is None:
            continue  # not in DaiD's scene (one-hand embodiment): S5 keeps the kinematic retarget
        arrays[f"{side}_wrist_position"], arrays[f"{side}_wrist_wxyz"] = w
        slots = daid.contact_slots(realised, offset, side)
        arrays.update({f"{side}_contact_{k}": v for k, v in slots.items()})
        contact_frames[side] = int((slots["part_ids"] > 0).any(axis=1).sum())
    np.savez(out / "trajectory_mjwp.npz", **arrays)
    poses = realised["object_pose"].reshape(len(realised["object_pose"]), -1, 7)
    replay_gap = max(float(np.linalg.norm(poses[:, i, :3] + offset - T[:, :3, 3], axis=1).max()) for i, T in enumerate(tracks))
    if not (out / "scene.xml").exists():
        (out / "scene.xml").symlink_to(summary["scene"])
    bundles = {}
    for o, T in zip(objs, tracks):
        base = dict(np.load(src[o] / "result.npz"))
        base["object_to_world_transform"] = T.astype(base["object_to_world_transform"].dtype)
        c2w = base["camera_to_world_transform"].astype(np.float64)
        base["object_to_camera_transform"] = (np.linalg.inv(c2w) @ T).astype(base["object_to_camera_transform"].dtype)
        bundles[o] = write_bundle(src[o], ctx.root / f"s4c_retarget/{o}/{cand}/bundle", base)
    return {"summary": summary, "tracks": dict(zip(objs, tracks)), "bundles": bundles, "src": src,
            "contact_frames": contact_frames, "replay_gap": replay_gap, "covered": back["covered_frames"],
            "trajectory": out / "trajectory_mjwp.npz"}


def scene_auc(ctx, bundles: dict[str, Path], work: Path, tag: str) -> float:
    """Public playback AUC of a two-object scene from per-object bundles (the waterfall's merge)."""
    frames = int(ctx.episode["num_frames"])
    poses, meshes = scene(merged(ctx.number, bundles, work / tag), ctx.objects, frames, work, tag)
    return gt.score(ctx.episode, poses, meshes, ctx.files_read)["AUC"]


def object_metrics(name: str, cand: str, chk: dict, info: dict, gate: dict, public: bool) -> list[Metric]:
    summary = info["summary"]
    return [
        Metric(f"{name}/{cand}: penetration (max)", (chk["penetration_m"] or np.nan) * 1000, "mm",
               ok=(0, MAX_PENETRATION * 1000)),
        Metric(f"{name}/{cand}: object motion before the first reach", chk["pre_reach_motion_m"] * 100, "cm",
               ok=(0, MAX_PRE_REACH * 100)),
        Metric(f"{name}/{cand}: object speed (max)", chk["max_speed_m_s"], "m/s", ok=(0, MAX_SPEED)),
        Metric(f"{name}/{cand}: DaiD object tracking error", summary["errors"].get("obj_pos_err"), "m",
               note=f"rotation {summary['errors'].get('obj_quat_err')}; {summary['optimize_seconds']} s"),
        Metric(f"{name}/{cand}: frames with a robot-object contact",
               ", ".join(f"{side} {k}" for side, k in info["contact_frames"].items()) or "none", "",
               note="MuJoCo replay at the video frames, per hand"),
        Metric(f"{name}/{cand}: replayed vs read-back object (max)", info["replay_gap"] * 1000, "mm", ok=(0, 5),
               note="nearest 5 ms step vs interpolation: the two read-outs of one trajectory"),
        Metric(f"{name}/{cand}: gate", "realised" if gate["use_realised"] else "kinematic", "",
               note=(f"AUC realised {gate['realised_auc']:.3f} vs kinematic {gate['kinematic_auc']:.3f}"
                     if public else gate["rule"])),
    ]


def run(ctx) -> StageResult:
    d = ctx.dir("s4c_retarget")
    mano_py, daid_py, daid_dir = env_paths()
    if not (os.path.exists(daid_py) and os.path.isdir(daid_dir)):
        return StageResult([], skip=f"Do as I Do not installed ({daid_py}, {daid_dir}): kinematic references go to S5")
    decisions = json.loads(GATE_FILE.read_text()) if GATE_FILE.is_file() else {}
    sim = yaml.safe_load((PATH_B / "pathb/s4c.yaml").read_text())["sim"]
    metrics, outputs = [Metric("MuJoCo Warp buffers per world (contacts / constraint rows)",
                               f"{sim['nconmax_per_env']} / {sim['njmax_per_env']}", "",
                               note="changed from DaiD's 120 / 500: ep40's dust pan (31 CoACD pieces on the table) "
                                    "needs 578 rows at frame 0 (smoke test 2026-10-23)")], {}
    phases = {o: json.loads((ctx.root / f"s4a_phases/{o}/phases.json").read_text()) for o in ctx.objects}
    two = phase2.enabled(ctx)
    groups = [list(ctx.objects)] if two else [[o] for o in ctx.objects]  # phase 2: one scene, both objects
    for objs in groups:
        for cand in CANDIDATES:
            out = d / ("scene" if two else objs[0]) / cand
            info = job(ctx, out, objs, cand)
            chks = {o: checks(info["tracks"][o], phases[o], info["summary"].get("penetration_max_m")) for o in objs}
            ok = all(c["pass"] for c in chks.values())
            gate = {"checks": chks, "daid_errors": info["summary"]["errors"], "covered_frames": info["covered"],
                    "contact_frames": info["contact_frames"], "replay_object_gap_m": info["replay_gap"],
                    "objects": objs}
            if ctx.public:
                if len(objs) == 1:
                    k_auc = playback_auc(ctx, objs[0], info["src"][objs[0]], out, f"kinematic_{cand}")
                    r_auc = playback_auc(ctx, objs[0], info["bundles"][objs[0]], out, f"realised_{cand}")
                else:
                    k_auc = scene_auc(ctx, info["src"], out, f"kinematic_{cand}")
                    r_auc = scene_auc(ctx, info["bundles"], out, f"realised_{cand}")
                gate.update(kinematic_auc=k_auc, realised_auc=r_auc, use_realised=bool(ok and r_auc >= k_auc - GATE_AUC),
                            rule="public score" + (" (two-object scene)" if two else ""))
            else:
                key = f"{ctx.episode['task']}/{cand}" + ("/scene" if two else "")
                gate.update(use_realised=bool(ok and decisions.get(key, False)),
                            rule=f"public decision for {key}: {decisions.get(key, 'none (kinematic)')}")
            for o in objs:  # each object's folder holds its gate (and, in a scene, links to the scene's files)
                od = d / o / cand
                od.mkdir(parents=True, exist_ok=True)
                if two:
                    for f in ("trajectory_mjwp.npz", "scene.xml"):
                        if not (od / f).exists():
                            (od / f).symlink_to((out / f).resolve())
                write_json(od / "metrics.json", {"daid": info["summary"], "checks": chks[o]})
                write_json(od / "gate.json", gate | {"checks": chks[o]})
                outputs[f"{o}/{cand}"] = info["bundles"][o]
                metrics += object_metrics(o, cand, chks[o], info, gate, ctx.public)
    return StageResult(metrics, outputs)
