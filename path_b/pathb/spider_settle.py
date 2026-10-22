"""S4 scene check, run with SPIDER's Python (it has mujoco): does SPIDER's compiled scene hold the
reference's frame-0 objects still, with the hands lifted out of the way?

Every z-axis slide joint (the hands' base joints) is raised by 1 m, each real object is placed at its
frame-0 pose from trajectory_keypoints.npz, and the scene runs for --seconds under gravity. An object
passes when it drifts at most 1 cm and turns at most 2 degrees: then the table (SPIDER's floor) and
the object-object contacts hold the scene up, and S4 starts from a scene at rest.

    $SPIDER_PY path_b/pathb/spider_settle.py --scene <scene.xml> --keypoints <trajectory_keypoints.npz> \
        --out settle.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

DRIFT_CM, TURN_DEG = 1.0, 2.0


def turn_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    dot = abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))))
    return float(np.degrees(2 * np.arccos(min(1.0, dot))))


def settle(scene: Path, keypoints: Path, seconds: float = 1.0) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    kp = np.load(keypoints)
    lifted = 0
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE and np.allclose(model.jnt_axis[j], [0, 0, 1]):
            data.qpos[model.jnt_qposadr[j]] = 1.0
            lifted += 1
    objects = {}
    for side in ("right", "left"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_object_joint")
        if jid < 0:
            continue
        q0 = kp[f"qpos_obj_{side}"][0]
        adr = int(model.jnt_qposadr[jid])
        data.qpos[adr:adr + 7] = q0
        if not (np.allclose(q0[:3], 0) and np.allclose(q0[3:], [1, 0, 0, 0])):  # an empty slot
            objects[side] = adr
    mujoco.mj_forward(model, data)
    depth = min((float(data.contact[i].dist) for i in range(data.ncon)), default=0.0)
    start = {side: data.qpos[adr:adr + 7].copy() for side, adr in objects.items()}
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
    result = {"scene": str(scene), "seconds": seconds, "timestep": float(model.opt.timestep),
              "hand_joints_lifted": lifted, "frame0_deepest_contact_cm": round(-min(depth, 0.0) * 100, 3),
              "objects": {}}
    for side, adr in objects.items():
        end = data.qpos[adr:adr + 7]
        finite = bool(np.isfinite(end).all())
        drift = float(np.linalg.norm(end[:3] - start[side][:3]) * 100) if finite else float("inf")
        turn = turn_deg(start[side][3:], end[3:]) if finite else float("inf")
        result["objects"][side] = {"drift_cm": round(drift, 3), "turn_deg": round(turn, 3),
                                   "passed": finite and drift <= DRIFT_CM and turn <= TURN_DEG}
    result["passed"] = bool(result["objects"]) and all(o["passed"] for o in result["objects"].values())
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--keypoints", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = settle(args.scene, args.keypoints, args.seconds)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
