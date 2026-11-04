"""Lock one hand to an object over a frame range (bundle level, no GT).

Where the reference hand lets go of an object that is still in the air and the object then floats down on its own,
the contact reward punishes the policy for holding it. This keeps the hand's grip relative to the object, averaged
over --grip-frames, from --start to --end, with slerp blends of --blend frames on both sides. Finger pose is held at
the grip frames' mean over the same window.

    python tools/track3/hold_hand.py --npz IN/result.npz --out OUT/result.npz --side right --object blue_cup \
        --start 81 --end 107 --grip-frames 74 78 [--blend 4]
"""
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def object_key(data, name):
    names = [str(x) for x in data["object_names"]] if "object_names" in data else [name]
    i = next(k for k, n in enumerate(names) if n == name or n.endswith("_" + name))
    return "object_to_world_transform" if i == 0 else f"object_{i}_to_world_transform"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--grip-frames", type=int, nargs=2, required=True)
    ap.add_argument("--blend", type=int, default=4)
    a = ap.parse_args()

    d = dict(np.load(a.npz))
    n = len(d["camera_to_world_transform"])
    T = d[object_key(d, a.object)].astype(float)
    key = f"hand_{a.side}_wrist_to_world_transform"
    H = d[key].astype(float)
    g0, g1 = a.grip_frames
    G = np.stack([np.linalg.inv(T[t]) @ H[t] for t in range(g0, g1 + 1)])
    grip = np.eye(4)
    grip[:3, :3] = Rotation.from_matrix(G[:, :3, :3]).mean().as_matrix()
    grip[:3, 3] = G[:, :3, 3].mean(0)
    spread = float(np.degrees(np.max((Rotation.from_matrix(grip[:3, :3]).inv() * Rotation.from_matrix(G[:, :3, :3])).magnitude())))

    # weight 1 inside [start, end], linear ramps of `blend` frames outside it
    w = np.zeros(n)
    for t in range(n):
        if a.start <= t <= a.end:
            w[t] = 1.0
        elif a.start - a.blend <= t < a.start:
            w[t] = (t - (a.start - a.blend) + 1) / (a.blend + 1)
        elif a.end < t <= a.end + a.blend:
            w[t] = 1 - (t - a.end) / (a.blend + 1)
    new = H.copy()
    for t in np.flatnonzero(w > 0):
        locked = T[t] @ grip
        r = Slerp([0, 1], Rotation.from_matrix([H[t, :3, :3], locked[:3, :3]]))(w[t])
        new[t, :3, :3] = r.as_matrix()
        new[t, :3, 3] = (1 - w[t]) * H[t, :3, 3] + w[t] * locked[:3, 3]
    moved = np.linalg.norm(new[:, :3, 3] - H[:, :3, 3], axis=1)

    c2w = d["camera_to_world_transform"].astype(float)
    w2c = np.linalg.inv(c2w) @ new
    d[key] = new.astype(d[key].dtype)
    for k, v in ((f"hand_{a.side}_wrist_to_camera_transform", w2c),
                 (f"hand_{a.side}_wrist_orient_in_camera", Rotation.from_matrix(w2c[:, :3, :3]).as_rotvec()),
                 (f"hand_{a.side}_wrist_trans_in_camera", w2c[:, :3, 3])):
        if k in d:
            d[k] = v.astype(d[k].dtype)
    fk = f"hand_{a.side}_finger_pose"
    if fk in d:
        f = d[fk].astype(float)
        hold = f[g0:g1 + 1].mean(0)
        wf = w.reshape((-1,) + (1,) * (f.ndim - 1))
        d[fk] = ((1 - wf) * f + wf * hold).astype(d[fk].dtype)
    vk = f"hand_{a.side}_is_valid"
    if vk in d:
        d[vk] = (d[vk].astype(bool) | (w > 0.5)).astype(d[vk].dtype)
    np.savez(a.out, **d)
    print(json.dumps(dict(object_key=object_key(d, a.object), grip_spread_deg=spread, grip_pos_cm=(grip[:3, 3] * 100).round(1).tolist(),
                          frames_changed=int((w > 0).sum()), max_move_cm=float(moved.max() * 100),
                          move_cm_at={int(t): round(float(moved[t] * 100), 1) for t in (a.start, (a.start + a.end) // 2, a.end)})))


if __name__ == "__main__":
    main()
