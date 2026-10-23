"""S4c retarget: Do as I Do physics retargeting of the kinematic references (V3_BUILD_SPEC.md S4c).

Per object, keyframe first, then tracking: the candidate's hands in the world (drivers/mano_world.py,
$RETARGET_VENV), the reference at 50 Hz in DaiD's world (pathb/daid.py), then DaiD's stages 2-5
with the Sharpa hands in MuJoCo Warp (drivers/daid_run.py, $DAID_PY, DaiD in $DAID_DIR), settings in
pathb/s4c.yaml; the warm-up weld lasts S4a's rest-before window. The Sharpa joints match FlashCHORD's
sharpawave URDF one to one (pathb/sharpa_map.json).
Out per object and candidate: trajectory_mjwp.npz (at the video frames, v3 world: robot joints, realised
object, wrists, contacts in ManoSharpa's 16 MANO-link slots), scene.xml (DaiD's), metrics.json,
bundle/ (the candidate's bundle with object_to_world = realised), gate.json. Gate: realised used if its playback AUC >= the kinematic one - 0.01 (public; evaluation
takes the public decision of its task type, oracles/s4c_gate.json) and it passes the checks:
penetration <= 5 mm, object moved <= 1 cm before the first reach, object speed <= 1.5 m/s.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from .. import daid, gt
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


def run(ctx) -> StageResult:
    d = ctx.dir("s4c_retarget")
    mano_py, daid_py, daid_dir = env_paths()
    if not (os.path.exists(daid_py) and os.path.isdir(daid_dir)):
        return StageResult([], skip=f"Do as I Do not installed ({daid_py}, {daid_dir}): kinematic references go to S5")
    decisions = json.loads(GATE_FILE.read_text()) if GATE_FILE.is_file() else {}
    metrics, outputs = [], {}
    for name in ctx.objects:
        ph = json.loads((ctx.root / f"s4a_phases/{name}/phases.json").read_text())
        warmup = P.length(ph["rest_before"]) / TRACK3_FPS
        mass = json.loads((ctx.root / f"s3a_mesh/{name}/mass.json").read_text())
        for cand in CANDIDATES:
            src = ctx.root / f"s4b_reference/{name}/{cand}"
            out = d / name / cand
            out.mkdir(parents=True, exist_ok=True)
            log = out / "daid.log"
            log.write_text("")
            run_logged([mano_py, PATH_B / "pathb/drivers/mano_world.py", src, out / "mano_world.npz",
                        PATH_B.parent / "baseline/reconstruction/data/weights/hand"], log)
            mano = dict(np.load(out / "mano_world.npz"))
            ref = np.load(src / "result.npz")["object_to_world_transform"].astype(np.float64)
            meta = daid.prepare(ref, mano, out / "input", TRACK3_FPS)
            write_json(out / "input/meta.json", meta)
            write_json(out / "request.json", {
                "daid_dir": daid_dir, "input": str(out / "input/daid_input.npz"), "output_root": str(out / "daid"),
                "task": f"ep{ctx.number}_{name}_{cand}", "embodiment": meta["embodiment"],
                "mesh": str(ctx.root / f"s3a_mesh/{name}/mesh.obj"), "convex": str(ctx.root / f"s3a_mesh/{name}/coacd"),
                "object_density": mass["density_kg_m3"], "warmup_duration": warmup,
                "settings": str(PATH_B / "pathb/s4c.yaml"), "video_fps": TRACK3_FPS, "n_video": meta["n_video"]})
            run_logged([daid_py, PATH_B / "pathb/drivers/daid_run.py", out / "request.json"], log, cwd=daid_dir)
            summary = json.loads((out / "daid/daid_summary.json").read_text())
            back = daid.read_back(summary["trajectory"], summary["qpos_joint_names"], meta)
            realised = dict(np.load(summary["realised_frames"]))
            offset = np.asarray(meta["offset"])
            arrays = {"object_to_world": back["object_to_world"], "robot_qpos": back["robot_qpos"],
                      "joint_names": np.array(back["joint_names"]), "fps": TRACK3_FPS,
                      "body_names": realised["body_names"], "body_pose": realised["body_pose"]}
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
            replay_gap = float(np.linalg.norm(realised["object_pose"][:, :3] + offset
                                              - back["object_to_world"][:, :3, 3], axis=1).max())
            if not (out / "scene.xml").exists():
                (out / "scene.xml").symlink_to(summary["scene"])
            base = dict(np.load(src / "result.npz"))
            base["object_to_world_transform"] = back["object_to_world"].astype(base["object_to_world_transform"].dtype)
            c2w = base["camera_to_world_transform"].astype(np.float64)
            base["object_to_camera_transform"] = (np.linalg.inv(c2w) @ back["object_to_world"]).astype(
                base["object_to_camera_transform"].dtype)
            write_bundle(src, out / "bundle", base)
            chk = checks(back["object_to_world"], ph, summary.get("penetration_max_m"))
            gate = {"checks": chk, "daid_errors": summary["errors"], "covered_frames": back["covered_frames"],
                    "contact_frames": contact_frames, "replay_object_gap_m": replay_gap}
            if ctx.public:
                k_auc = playback_auc(ctx, name, src, out, f"kinematic_{cand}")
                r_auc = playback_auc(ctx, name, out / "bundle", out, f"realised_{cand}")
                gate.update(kinematic_auc=k_auc, realised_auc=r_auc,
                            use_realised=bool(chk["pass"] and r_auc >= k_auc - GATE_AUC), rule="public score")
            else:
                key = f"{ctx.episode['task']}/{cand}"
                gate.update(use_realised=bool(chk["pass"] and decisions.get(key, False)),
                            rule=f"public decision for {key}: {decisions.get(key, 'none (kinematic)')}")
            write_json(out / "metrics.json", {"daid": summary, "checks": chk})
            write_json(out / "gate.json", gate)
            outputs[f"{name}/{cand}"] = out / "bundle"
            metrics += [
                Metric(f"{name}/{cand}: penetration (max)", (chk["penetration_m"] or np.nan) * 1000, "mm",
                       ok=(0, MAX_PENETRATION * 1000)),
                Metric(f"{name}/{cand}: object motion before the first reach", chk["pre_reach_motion_m"] * 100, "cm",
                       ok=(0, MAX_PRE_REACH * 100)),
                Metric(f"{name}/{cand}: object speed (max)", chk["max_speed_m_s"], "m/s", ok=(0, MAX_SPEED)),
                Metric(f"{name}/{cand}: DaiD object tracking error", summary["errors"].get("obj_pos_err"), "m",
                       note=f"rotation {summary['errors'].get('obj_quat_err')}; {summary['optimize_seconds']} s"),
                Metric(f"{name}/{cand}: frames with a robot-object contact",
                       ", ".join(f"{side} {k}" for side, k in contact_frames.items()) or "none", "",
                       note="MuJoCo replay at the video frames, per hand"),
                Metric(f"{name}/{cand}: replayed vs read-back object (max)", replay_gap * 1000, "mm", ok=(0, 5),
                       note="nearest 5 ms step vs interpolation: the two read-outs of one trajectory"),
                Metric(f"{name}/{cand}: gate", "realised" if gate["use_realised"] else "kinematic", "",
                       note=(f"AUC realised {gate['realised_auc']:.3f} vs kinematic {gate['kinematic_auc']:.3f}"
                             if ctx.public else gate["rule"])),
            ]
    return StageResult(metrics, outputs)
