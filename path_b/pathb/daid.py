"""Do as I Do adapter (V3_BUILD_SPEC.md S4c): a v3 bundle and its world MANO hands at DaiD's reference
rate, and DaiD's optimized trajectory back at the video's frames.

DaiD's stage 1 (retargeting/pipeline/process_dataset.py) reads a reconstruction folder; S4c instead
gives pathb/drivers/daid_run.py (daid venv) one daid_input.npz, from which the driver writes
trajectory_keypoints.npz with DaiD's own functions. Here: the object track (S4b candidate) and the
hands (pathb/drivers/mano_world.py: joints, vertices, MANO parameters) resampled from the video's
20.254 Hz to 50 Hz (DaiD's ref_dt 0.02 = 4 sim steps of 0.005; positions linear, rotations slerp),
in DaiD's world: the v3 world with the object's frame-0 xy at the origin and the table, z = 0, as
the floor (DaiD itself centres xy too). DaiD optimizes up to one planning horizon before the
reference ends, so the last frame is held PAD_S past the clip. Back: DaiD's qpos per sim step
(object = the last 7 numbers, free joint pos + wxyz) sampled at the video's frame times, and from
the driver's MuJoCo replay (realised_frames.npz) the wrists and the robot-object contacts in
ManoSharpa's 16 MANO-link slots, for S5's task file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REF_HZ = 50.0
PAD_S = 3.5  # s: horizon (3.0) + ctrl_dt (0.5), s4c.yaml
FINGERTIP, PIP, DIP = [4, 8, 12, 16, 20], [2, 6, 10, 14, 18], [3, 7, 11, 15, 19]  # OpenPose order, as DaiD


def lerp(x: np.ndarray, t_in: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    """Linear interpolation along axis 0, holding the ends."""
    flat = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    t = np.clip(t_out, t_in[0], t_in[-1])
    out = np.stack([np.interp(t, t_in, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return out.reshape((len(t_out),) + np.asarray(x).shape[1:])


def slerp(R: Rotation, t_in: np.ndarray, t_out: np.ndarray) -> Rotation:
    return Slerp(t_in, R)(np.clip(t_out, t_in[0], t_in[-1]))


def embodiment(mano: dict) -> str:
    """bimanual when both hands are valid on at least half the frames, else the better hand."""
    share = {s: float(np.mean(mano[f"{s}_valid"])) for s in ("right", "left")}
    if min(share.values()) >= 0.5:
        return "bimanual"
    return max(share, key=share.get)


def prepare(object_to_world: np.ndarray, mano: dict, out: Path, fps: float) -> dict:
    """daid_input.npz in DaiD's world at REF_HZ; the meta needed to read the result back."""
    T = np.asarray(object_to_world, dtype=np.float64)
    n = len(T)
    t_video = np.arange(n) / fps
    t_ref = np.arange(int(round((t_video[-1] + PAD_S) * REF_HZ)) + 1) / REF_HZ
    offset = np.array([T[0, 0, 3], T[0, 1, 3], 0.0])
    q = slerp(Rotation.from_matrix(T[:, :3, :3]), t_video, t_ref).as_quat()
    arrays = {"qpos_obj": np.concatenate([lerp(T[:, :3, 3], t_video, t_ref) - offset, q[:, [3, 0, 1, 2]]], axis=1),
              "t_ref": t_ref}
    for side in ("right", "left"):
        arrays[f"{side}_joints"] = lerp(mano[f"{side}_joints"], t_video, t_ref) - offset
        arrays[f"{side}_vertices"] = (lerp(mano[f"{side}_vertices"], t_video, t_ref) - offset).astype(np.float32)
        arrays[f"{side}_faces"] = np.asarray(mano[f"{side}_faces"], dtype=np.int32)
        arrays[f"{side}_rot"] = slerp(Rotation.from_rotvec(mano[f"{side}_global_orient"]), t_video, t_ref).as_rotvec()
        fp = np.asarray(mano[f"{side}_finger_pose"], dtype=np.float64).reshape(n, 15, 3)
        arrays[f"{side}_hand_pose"] = np.stack([slerp(Rotation.from_rotvec(fp[:, j]), t_video, t_ref).as_rotvec()
                                                for j in range(15)], axis=1).reshape(len(t_ref), 45)
        arrays[f"{side}_betas"] = np.repeat(np.asarray(mano[f"{side}_betas"], dtype=np.float64).reshape(1, 10),
                                            len(t_ref), axis=0)
        valid = np.asarray(mano[f"{side}_valid"], dtype=bool)
        arrays[f"{side}_valid"] = valid[np.clip(np.round(t_ref * fps).astype(int), 0, n - 1)]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "daid_input.npz", **arrays)
    return {"offset": offset, "fps": fps, "n_video": n, "ref_hz": REF_HZ, "n_ref": len(t_ref), "pad_s": PAD_S,
            "embodiment": embodiment(mano)}


def read_back(trajectory: Path, joint_names: list[str], meta: dict) -> dict:
    """DaiD's optimized trajectory at the video frames: object_to_world (v3 world) and robot joints."""
    z = np.load(trajectory)
    qpos = np.asarray(z["qpos"], dtype=np.float64)
    qpos = qpos.reshape(-1, qpos.shape[-1])
    time = np.asarray(z["time"], dtype=np.float64).reshape(-1)
    order = np.argsort(time, kind="stable")
    qpos, time = qpos[order], time[order]
    t_video = np.arange(meta["n_video"]) / meta["fps"]
    covered = t_video <= time[-1] + 1e-9
    obj = qpos[:, -7:]
    T = np.tile(np.eye(4), (meta["n_video"], 1, 1))
    T[:, :3, 3] = lerp(obj[:, :3], time, t_video) + np.asarray(meta["offset"])
    T[:, :3, :3] = slerp(Rotation.from_quat(obj[:, [4, 5, 6, 3]]), time, t_video).as_matrix()
    robot = lerp(qpos[:, :-7], time, t_video)
    return {"object_to_world": T, "robot_qpos": robot, "joint_names": joint_names[: robot.shape[1]],
            "covered_frames": int(covered.sum()), "sim_seconds": float(time[-1])}


# ManoSharpa's 16 contact slots (robotic_grounding retarget/params.py MANO_HAND_LINKS, the order
# v2d_task_library_loader's contact_utils writes them in)
MANO_LINKS = ("link_palm", "link_thumb1", "link_thumb2", "link_thumb3", "link_index1", "link_index2", "link_index3",
              "link_middle1", "link_middle2", "link_middle3", "link_ring1", "link_ring2", "link_ring3",
              "link_pinky1", "link_pinky2", "link_pinky3")
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
SEGMENT = {"thumb": {"CMC_VL": 1, "MC": 1, "MCP_VL": 2, "PP": 2, "DP": 3, "elastomer": 3, "fingertip": 3},
           "finger": {"MCP_VL": 1, "PP": 1, "MP": 2, "DP": 3, "elastomer": 3, "fingertip": 3}}


def mano_link(body: str) -> int | None:
    """Slot of a Sharpa body: the MANO link of the same digit and segment (MANO link 1 = the thumb's
    metacarpal, the fingers' proximal phalanx); the palm and the pinky metacarpal -> link_palm."""
    name = body.split("_", 1)[1]
    if name.startswith("hand_") or name == "pinky_MC":
        return 0
    digit, part = name.split("_", 1)
    if digit not in DIGITS:
        return None
    seg = SEGMENT["thumb" if digit == "thumb" else "finger"].get(part)
    return None if seg is None else 1 + 3 * DIGITS.index(digit) + seg - 1


def contact_slots(realised: dict, offset: np.ndarray, side: str) -> dict:
    """One side's DaiD contacts in the 16 slots, v3 world: per frame and slot the deepest contact
    (object-side point, normal into the object, part id 1 = the object's one body), and which
    Sharpa body touched it with the robot-side point in that body's frame (for S5's check)."""
    n = len(realised["video_time"])
    names = [str(b) for b in realised["body_names"]]
    out = {"positions": np.zeros((n, 16, 3)), "normals": np.zeros((n, 16, 3)), "part_ids": np.zeros((n, 16), np.int32),
           "body": np.full((n, 16), -1, np.int32), "robot_local": np.zeros((n, 16, 3)), "dist": np.full((n, 16), np.inf)}
    for k, f in enumerate(realised["contact_frame"]):
        b = int(realised["contact_body"][k])
        s = mano_link(names[b]) if names[b].startswith(side + "_") else None
        if s is None or realised["contact_dist"][k] >= out["dist"][f, s]:
            continue
        out["dist"][f, s] = realised["contact_dist"][k]
        out["positions"][f, s] = realised["contact_object_point"][k] + offset
        out["normals"][f, s] = realised["contact_normal"][k]
        out["part_ids"][f, s] = 1
        out["body"][f, s] = b
        out["robot_local"][f, s] = realised["contact_robot_local"][k]
    out["dist"][out["part_ids"] == 0] = 0.0
    return out


def wrist(realised: dict, offset: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray] | None:
    """<side>_hand_C_MC (FlashCHORD's wrist, the URDF root link) in the v3 world: position, wxyz."""
    names = [str(b) for b in realised["body_names"]]
    if f"{side}_hand_C_MC" not in names:
        return None
    p = realised["body_pose"][:, names.index(f"{side}_hand_C_MC")]
    return p[:, :3] + offset, p[:, 3:] / np.linalg.norm(p[:, 3:], axis=1, keepdims=True)
