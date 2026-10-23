"""S4c driver, run in the daid venv: Do as I Do retargeting of one S4b candidate onto the Sharpa hands.

    $DAID_PY path_b/pathb/drivers/daid_run.py <request.json>

request.json: {"daid_dir": <do-as-i-do/retargeting>, "input": <daid_input.npz from pathb/daid.py>,
"output_root": <dir>, "task": <name>, "embodiment": bimanual|right|left, "mesh": <S3a mesh.obj>,
"convex": <S3a coacd/ pieces>, "object_density": kg/m^3, "warmup_duration": s, "settings": <s4c.yaml>,
"video_fps": Hz, "n_video": frames}.

Stage 1 is replaced: trajectory_keypoints.npz and task_info.json are written with DaiD's own
process_dataset helpers (_build_side_qpos, joint indices) from the v3 reference; the object mesh goes
to assets/objects/<task>/visual.obj with S3a's CoACD pieces as convex/<i>.obj (so DaiD's
decomposition keeps them). Then DaiD's stages 2-5 as launch.py runs them, with s4c.yaml's settings.
Writes <output_root>/daid_summary.json (errors, joint names in qpos order, timings) and
realised_frames.npz (hand body poses, object pose and robot-object contacts at the video frames,
from a MuJoCo replay of the optimized trajectory; read by S4c for S5's task file).
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml


CONTACT_DIST = 0.005  # m: robot-object pairs closer than this count as contacts


def realised_frames(model, data, obj_body: int, q_all: np.ndarray, t_sim: np.ndarray, t_video: np.ndarray) -> dict:
    """The realised trajectory at the video frames (nearest sim step, <= 2.5 ms away): pose of every
    hand body (wrists = <side>_hand_C_MC), the object, and the robot-object contacts (object-side point,
    normal into the object, robot-side point in its body's frame, distance; DaiD writes no contacts)."""
    import mujoco
    order = np.argsort(t_sim, kind="stable")
    ts = t_sim[order]
    i = np.clip(np.searchsorted(ts, t_video), 1, len(ts) - 1)
    i = i - ((t_video - ts[i - 1]) < (ts[i] - t_video))
    steps = order[i]
    bodies = [b for b in range(model.nbody)
              if model.body(b).name.startswith(("right_", "left_")) and "_base_" not in model.body(b).name]
    pose = np.zeros((len(t_video), len(bodies), 7))
    obj = np.zeros((len(t_video), 7))
    rows = {k: [] for k in ("contact_frame", "contact_body", "contact_object_point", "contact_normal",
                            "contact_robot_local", "contact_dist")}
    slot = {b: k for k, b in enumerate(bodies)}
    for f, s in enumerate(steps):
        data.qpos[:] = q_all[s]
        mujoco.mj_forward(model, data)
        pose[f] = np.concatenate([data.xpos[bodies], data.xquat[bodies]], axis=1)
        obj[f] = np.concatenate([data.xpos[obj_body], data.xquat[obj_body]])
        for c in data.contact[: data.ncon]:
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (b1 == obj_body) == (b2 == obj_body) or min(b1, b2) == 0 or c.dist > CONTACT_DIST:
                continue
            robot = b2 if b1 == obj_body else b1
            if robot not in slot:
                continue
            n = np.asarray(c.frame[:3]) * (-1.0 if b1 == obj_body else 1.0)  # MuJoCo: geom1 -> geom2; here into the object
            robot_point = c.pos - n * c.dist / 2
            rows["contact_frame"].append(f)
            rows["contact_body"].append(slot[robot])
            rows["contact_object_point"].append(c.pos + n * c.dist / 2)
            rows["contact_normal"].append(n)
            rows["contact_robot_local"].append(data.xmat[robot].reshape(3, 3).T @ (robot_point - data.xpos[robot]))
            rows["contact_dist"].append(float(c.dist))
    out = {k: np.asarray(v, dtype=np.int32 if k in ("contact_frame", "contact_body") else np.float64).reshape(
        (-1, 3) if k in ("contact_object_point", "contact_normal", "contact_robot_local") else (-1,))
        for k, v in rows.items()}
    return dict(out, body_names=np.array([model.body(b).name for b in bodies]), body_pose=pose, object_pose=obj,
                step_time=t_sim[steps], video_time=t_video)


def main(request_path: str) -> int:
    req = json.loads(Path(request_path).read_text())
    sys.path.insert(0, req["daid_dir"])  # launch.py and the retargeting package
    from launch import load_mjwp_config
    from retargeting.pipeline import process_dataset as P
    from retargeting.pipeline.decompose_mesh import main as decompose_mesh
    from retargeting.pipeline.generate_scene import main as generate_scene
    from retargeting.pipeline.optimize_physics import main as optimize_physics
    from retargeting.pipeline.resolve_pedestal import resolve_scene_pedestal
    from retargeting.pipeline.solve_ik import main as solve_ik
    from retargeting.utils.io import get_mesh_dir, get_processed_data_dir

    s = yaml.safe_load(Path(req["settings"]).read_text())
    opt, scene = s["optimizer"], s["scene"]
    root, task, emb = str(Path(req["output_root"]).resolve()), req["task"], req["embodiment"]
    dataset, data_id = "do_as_i_do", 0
    z = np.load(req["input"])
    N = len(z["qpos_obj"])
    t0 = time.time()

    # stage 1 replacement: DaiD's keypoint layout from our reference
    sides = {"right": emb in ("right", "bimanual"), "left": emb in ("left", "bimanual")}
    out = {}
    for side, use in sides.items():
        if use:
            joints = z[f"{side}_joints"]
            w, f = P._build_side_qpos(joints, joints[:, P.FINGERTIP_JOINT_IDX], z[f"{side}_rot"], np.zeros(3),
                                      is_right=side == "right")
            out.update({f"qpos_wrist_{side}": w, f"qpos_finger_{side}": f,
                        f"qpos_pip_{side}": joints[:, P.PIP_JOINT_IDX], f"qpos_dip_{side}": joints[:, P.DIP_JOINT_IDX],
                        f"qpos_obj_{side}": z["qpos_obj"].copy(), f"mano_verts_{side}": z[f"{side}_vertices"],
                        f"mano_faces_{side}": z[f"{side}_faces"], f"mano_finger_pose_{side}": z[f"{side}_hand_pose"],
                        f"mano_global_orient_{side}": z[f"{side}_rot"], f"mano_betas_{side}": z[f"{side}_betas"]})
        else:
            out.update({f"qpos_wrist_{side}": P._identity_qpos((N, 7)), f"qpos_finger_{side}": P._identity_qpos((N, 5, 7)),
                        f"qpos_pip_{side}": np.zeros((N, 5, 3)), f"qpos_dip_{side}": np.zeros((N, 5, 3)),
                        f"qpos_obj_{side}": P._identity_qpos((N, 7)), f"mano_verts_{side}": np.zeros((0, 0, 3), np.float32),
                        f"mano_faces_{side}": np.zeros((0, 3), np.int32), f"mano_finger_pose_{side}": np.zeros((0, 45), np.float32),
                        f"mano_global_orient_{side}": np.zeros((0, 3), np.float32), f"mano_betas_{side}": np.zeros((0, 10), np.float32)})
        out.update({f"contact_{side}": np.zeros((N, 10)), f"contact_pos_{side}": np.zeros((10, 3))})
    processed = Path(get_processed_data_dir(root, dataset, "mano", emb, task, data_id))
    processed.mkdir(parents=True, exist_ok=True)
    np.savez(processed / "trajectory_keypoints.npz", centering_offset=np.zeros(3), **out)
    mesh_dir = Path(get_mesh_dir(root, dataset, task))
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    (mesh_dir / "convex").mkdir(parents=True)
    shutil.copy(req["mesh"], mesh_dir / "visual.obj")
    P._copy_obj_texture(req["mesh"], str(mesh_dir))
    pieces = sorted(Path(req["convex"]).glob("piece_*.obj"))
    for i, p in enumerate(pieces):
        shutil.copy(p, mesh_dir / f"convex/{i}.obj")
    rel = str(Path(mesh_dir).relative_to(root))
    (processed.parent / "task_info.json").write_text(json.dumps({
        "task": task, "dataset_name": dataset, "robot_type": "mano", "embodiment_type": emb, "data_id": data_id,
        "right_object_mesh_dir": rel if sides["right"] else None,
        "left_object_mesh_dir": rel if sides["left"] and not sides["right"] else None}, indent=2))

    # stages 2-5, as launch.py, with s4c.yaml
    decompose_mesh(output_root_dir=root, dataset_name=dataset, task=task, data_id=data_id, embodiment_type=emb,
                   thicken=s["decompose"]["thicken"], dilate=s["decompose"]["dilate"], force=False)
    generate_scene(output_root_dir=root, dataset_name=dataset, task=task, data_id=data_id, embodiment_type=emb,
                   robot_type="sharpa", show_viewer=False, friction_scale=scene["friction_scale"],
                   object_floor_collision=scene["object_floor_collision"], hand_floor_collision=scene["hand_floor_collision"],
                   use_pedestal=scene["use_pedestal"], use_support=scene["use_support"],
                   object_density=req["object_density"], force=True, add_ur3_arm=scene["add_ur3_arm"])
    solve_ik(output_root_dir=root, dataset_name=dataset, task=task, data_id=data_id, embodiment_type=emb,
             robot_type="sharpa", show_viewer=False, force=True, smoothing=s["ik"]["smoothing"],
             sim_dt=opt["sim_dt"], ref_dt=opt["ref_dt"])
    interp = min(req["warmup_duration"], s["warmup"]["ref_interp_max"])
    config = load_mjwp_config(dataset_name=dataset, task=task, data_id=data_id, robot_type="sharpa",
                              embodiment_type=emb, seed=opt["seed"], wait_on_finish=False,
                              max_sim_steps=opt["max_sim_steps"], force=True, show_viewer=False, output_root_dir=root,
                              sim_dt=opt["sim_dt"], ref_dt=opt["ref_dt"], ctrl_dt=opt["ctrl_dt"], horizon=opt["horizon"],
                              num_samples=opt["num_samples"], terminal_rew_scale=opt["terminal_rew_scale"],
                              warmup_duration=req["warmup_duration"], warmup_ref_base_interp_duration=interp,
                              warmup_ref_finger_interp_duration=interp)
    resolve_scene_pedestal(output_root_dir=root, dataset_name=dataset, robot_type="sharpa", embodiment_type=emb,
                           task=task, data_id=data_id, use_pedestal=scene["use_pedestal"], use_support=scene["use_support"],
                           hand_object_distance_thresh=config.hand_object_distance_thresh, force=True)
    t1 = time.time()
    errors = optimize_physics(config)
    import mujoco
    model = mujoco.MjModel.from_xml_path(config.model_path)
    names = []
    for j in range(model.njnt):
        width = {0: 7, 1: 4, 2: 1, 3: 1}[int(model.jnt_type[j])]  # free, ball, slide, hinge
        names += [model.joint(j).name] * width if width > 1 else [model.joint(j).name]
    robot_dir = Path(get_processed_data_dir(root, dataset, "sharpa", emb, task, data_id))
    # penetration: the realised trajectory replayed in the scene; contacts between the object (the
    # body with the free joint) and any robot geom, depth = -dist (DaiD reports none of its own)
    data = mujoco.MjData(model)
    obj_body = next(int(model.jnt_bodyid[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == 0)
    traj = np.load(robot_dir / "trajectory_mjwp.npz")
    q_all = np.asarray(traj["qpos"]).reshape(-1, model.nq)
    depths = []
    for q in q_all[:: max(1, len(q_all) // 400)]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        d = 0.0
        for c in data.contact[: data.ncon]:
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (b1 == obj_body) != (b2 == obj_body) and min(b1, b2) != 0:  # object vs robot, not the floor
                d = max(d, -float(c.dist))
        depths.append(d)
    realised = realised_frames(model, data, obj_body, q_all, np.asarray(traj["time"]).reshape(-1),
                               np.arange(req["n_video"]) / req["video_fps"])
    np.savez(Path(root) / "realised_frames.npz", **realised)
    (Path(root) / "daid_summary.json").write_text(json.dumps({
        "penetration_max_m": float(max(depths)) if depths else None,
        "penetration_p95_m": float(np.percentile(depths, 95)) if depths else None,
        "errors": {k: float(v) for k, v in (errors or {}).items()}, "qpos_joint_names": names,
        "trajectory": str(robot_dir / "trajectory_mjwp.npz"), "scene": config.model_path,
        "realised_frames": str(Path(root) / "realised_frames.npz"),
        "contacts": int(len(realised["contact_frame"])),
        "kinematic": str(robot_dir / "trajectory_kinematic.npz"), "embodiment": emb, "convex_pieces": len(pieces),
        "warmup_duration": req["warmup_duration"], "setup_seconds": round(t1 - t0, 1),
        "optimize_seconds": round(time.time() - t1, 1)}, indent=1) + "\n")
    print(f"[daid_run] {task}: errors {errors}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
