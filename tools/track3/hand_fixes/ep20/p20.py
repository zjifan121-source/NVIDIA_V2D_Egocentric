# hand5 ep20 patch (exec'd into rlr_h5.py globals right before cleaned_bundle()).
# Arm A (objrel): object-relative refill of the right hand's long gap while an object is lifted.
#   fill_hand(right) first runs the team fill unchanged, then, for every bad run >= H5_MIN_GAP frames during
#   which an object is lifted > 2 cm, re-fills the frames between two GT-free anchors in that OBJECT's frame:
#   grip(t) = inv(T_obj(t)) @ H_wrist(t) (real MANO wrist, j0 included). Anchors: the last frame before the run and
#   the first frame after it that starts 3 time-contiguous frames whose grip agrees (rot < 20 deg, pos < 3 cm) with
#   the robust mean grip of the observed frames within 15 frames of the run (so isolated observed outliers next to
#   the run, e.g. ep20 f31 flipped 154 deg and f101-106, are refilled too); anchor grip = mean grip over up to 6
#   consistent frames from the anchor outward.
#   Interpolation: Slerp of grip rotation, linear grip position and finger pose; hand = T_obj(t) @ grip(t).
# Arm B (objrel + push): H5_PUSH=1 also pushes the right hand out of the lifted object after attach_hands:
#   per frame, MANO vertices (pipeline MANO) inside the object mesh (signed distance from dense surface samples,
#   face-normal sign) -> translate the wrist along the mean outward normal by depth + H5_PUSH_MARGIN (m), 3 passes,
#   only on frames where the right hand is attached to / filled against that object.
import os as _os
import numpy as _np
from scipy.spatial.transform import Rotation as _Rot, Slerp as _Slerp

_h5_orig_fill_hand = fill_hand
_h5_orig_attach_hands = attach_hands
_h5_orig_cleaned_bundle = cleaned_bundle
H5_INFO = {}
H5_FILLED = {}
_H5 = dict(min_gap=int(_os.environ.get("H5_MIN_GAP", 10)), lift=0.02, win=15, rot_tol=20.0, pos_tol=0.03, anchor_win=6,
           push=_os.environ.get("H5_PUSH", "0") == "1", margin=float(_os.environ.get("H5_PUSH_MARGIN", 0.0015)))


def _h5_runs(mask):
    e = _np.flatnonzero(_np.diff(_np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(e[::2], e[1::2]))


def _h5_robust_mean(rots, pos, tol_deg, tol_pos):
    keep = _np.ones(len(rots), bool)
    for _ in range(3):
        m = _Rot.from_quat(rots.as_quat()[keep]).mean()
        p = _np.median(pos[keep], axis=0)
        dev_r = _np.degrees((rots * m.inv()).magnitude())
        dev_p = _np.linalg.norm(pos - p, axis=1)
        new = (dev_r < tol_deg) & (dev_p < tol_pos)
        if new.sum() < 2 or (new == keep).all():
            break
        keep = new
    return m, p


def fill_hand(data, side, bad):
    keys = (f"hand_{side}_wrist_orient_in_camera", f"hand_{side}_wrist_trans_in_camera", f"hand_{side}_finger_pose")
    raw = {k: data[k].astype(_np.float64).copy() for k in keys}
    _h5_orig_fill_hand(data, side, bad)
    if side != "right" or not bad.any():
        return
    n = len(bad)
    c2w = data["camera_to_world_transform"].astype(_np.float64)
    Rc, tc = c2w[:, :3, :3], c2w[:, :3, 3]
    j0 = mano_rest_wrist(side, data[f"hand_{side}_betas"].reshape(10))
    Rw = _Rot.from_matrix(Rc @ _Rot.from_rotvec(raw[keys[0]]).as_matrix())
    pw = _np.einsum("nij,nj->ni", Rc, raw[keys[1]] + j0) + tc
    fingers = raw[keys[2]].reshape(n, -1)
    good = ~bad.astype(bool)
    slots = object_slots(data)
    out = []
    for s, e in _h5_runs(bad):
        if e - s < _H5["min_gap"] or s == 0 or e >= n:
            continue
        lifts = [float((data[pk][s:e, 2, 3] - data[pk][0, 2, 3]).max()) for pk, _, _ in slots]
        o = int(_np.argmax(lifts))
        if lifts[o] <= _H5["lift"]:
            continue
        T = data[slots[o][0]].astype(_np.float64)
        Gr = _Rot.from_matrix(T[:, :3, :3]).inv() * Rw
        Gp = _np.einsum("nji,nj->ni", T[:, :3, :3], pw - T[:, :3, 3])
        anchors = []
        for side_frames in ([t for t in range(max(0, s - _H5["win"]), s) if good[t]][::-1],
                            [t for t in range(e, min(n, e + _H5["win"])) if good[t]]):
            if len(side_frames) < 3:
                break
            idx = _np.array(side_frames)
            m, p = _h5_robust_mean(Gr[idx], Gp[idx], _H5["rot_tol"], _H5["pos_tol"])
            ok = [t for t in side_frames if _np.degrees((Gr[t] * m.inv()).magnitude()) < _H5["rot_tol"]
                  and _np.linalg.norm(Gp[t] - p) < _H5["pos_tol"]]
            okset = set(ok)
            # anchor: nearest frame to the run that starts a run of 3 time-contiguous consistent frames
            # (side_frames is ordered outward from the run); every frame between it and the run is refilled
            step = -1 if side_frames[0] < s else 1
            starts = [t for t in side_frames if all(t + step * j in okset for j in range(3))]
            if not starts:
                break
            a = starts[0]
            win = [t for t in ok if (t - a) * step >= 0][:_H5["anchor_win"]]
            anchors.append((a, _Rot.from_quat(Gr[win].as_quat()).mean(), Gp[win].mean(axis=0), fingers[win].mean(axis=0), win))
        if len(anchors) < 2:
            continue
        (a, ra, pa, fa, wa), (b, rb, pb, fb, wb) = anchors
        frames = _np.arange(a + 1, b)
        w = (frames - a) / float(b - a)
        g_rot = _Slerp([0.0, 1.0], _Rot.from_quat(_np.stack([ra.as_quat(), rb.as_quat()])))(w)
        g_pos = (1 - w)[:, None] * pa + w[:, None] * pb
        new_R = T[frames, :3, :3] @ g_rot.as_matrix()
        new_p = _np.einsum("nij,nj->ni", T[frames, :3, :3], g_pos) + T[frames, :3, 3]
        new_f = (1 - w)[:, None] * fa + w[:, None] * fb
        cam_R = _np.einsum("nji,njk->nik", Rc[frames], new_R)
        cam_t = _np.einsum("nji,nj->ni", Rc[frames], new_p - tc[frames]) - j0
        o_ = data[keys[0]].copy(); o_[frames] = _Rot.from_matrix(cam_R).as_rotvec(); data[keys[0]] = o_.astype(data[keys[0]].dtype)
        t_ = data[keys[1]].copy(); t_[frames] = cam_t; data[keys[1]] = t_.astype(data[keys[1]].dtype)
        f_ = data[keys[2]].copy().reshape(n, -1); f_[frames] = new_f
        data[keys[2]] = f_.reshape(data[keys[2]].shape).astype(data[keys[2]].dtype)
        w2c = _np.tile(_np.eye(4), (n, 1, 1))
        w2c[:, :3, :3] = _Rot.from_rotvec(data[keys[0]].astype(_np.float64)).as_matrix()
        w2c[:, :3, 3] = data[keys[1]]
        data[f"hand_{side}_wrist_to_camera_transform"] = w2c.astype(_np.float32)
        data[f"hand_{side}_wrist_to_world_transform"] = (c2w @ w2c).astype(_np.float32)
        H5_FILLED.setdefault(side, []).append((o, int(a), int(b)))
        out.append({"object": o, "bad_run": [int(s), int(e)], "anchor_pre": int(a), "anchor_post": int(b),
                    "anchor_pre_frames": [int(x) for x in wa], "anchor_post_frames": [int(x) for x in wb],
                    "grip_rot_change_deg": round(float(_np.degrees((rb * ra.inv()).magnitude())), 1),
                    "grip_pos_change_cm": round(float(_np.linalg.norm(pb - pa)) * 100, 1)})
    H5_INFO[f"h5_objrel_{side}"] = out


def _h5_mano(side, betas, orient, trans, fingers):
    import torch
    from v2d.task_library_loader.lib.read_mano import MANO
    mano = MANO(mano_assets_root=str(MANO_DIR), gender="neutral", device=torch.device("cpu"),
                flat_hand_mean=True, center_idx=None)
    f = lambda x: torch.from_numpy(_np.asarray(x, dtype=_np.float32))
    with torch.no_grad():
        out = mano.forward(side=side, betas=f(betas).reshape(10), global_orient=f(orient), transl=f(trans),
                           finger_pose=f(fingers).reshape(len(orient), 45))
    return out["vertices"].cpu().numpy().astype(_np.float64)


def _h5_push(data, bundle, side="right"):
    from scipy.spatial import cKDTree
    import trimesh
    info = {}
    slots = object_slots(data)
    c2w = data["camera_to_world_transform"].astype(_np.float64)
    Rc, tc = c2w[:, :3, :3], c2w[:, :3, 3]
    frames_by_obj = {}
    for o, a, b in H5_FILLED.get(side, []):
        frames_by_obj.setdefault(o, set()).update(range(a, b + 1))
    for key, holds_ in H5_INFO.get("_attach", {}).items():
        o = int(key.split("_")[1])
        for sd, a, b, _ in holds_:
            if sd == side:
                frames_by_obj.setdefault(o, set()).update(range(a, b))
    for o, fr in frames_by_obj.items():
        pose_key, scale_key, mesh_file = slots[o]
        mesh = trimesh.load(bundle / mesh_file, force="mesh", process=False)
        mesh.apply_scale(float(data.get(scale_key, 1.0)))
        S, fi = trimesh.sample.sample_surface_even(mesh, 60000, seed=0)
        NS = mesh.face_normals[fi]
        tree = cKDTree(S)
        T = data[pose_key].astype(_np.float64)
        fr = _np.array(sorted(fr))
        before, after, moved = None, _np.zeros(len(fr)), []
        for it in range(4):
            V = _h5_mano(side, data[f"hand_{side}_betas"].reshape(10), data[f"hand_{side}_wrist_orient_in_camera"][fr],
                         data[f"hand_{side}_wrist_trans_in_camera"][fr], data[f"hand_{side}_finger_pose"][fr])
            Vw = _np.einsum("nij,nvj->nvi", Rc[fr], V) + tc[fr][:, None]
            shift_cam = _np.zeros((len(fr), 3))
            depth = _np.zeros(len(fr))
            for i, t in enumerate(fr):
                loc = (Vw[i] - T[t, :3, 3]) @ T[t, :3, :3]
                d, k = tree.query(loc)
                inside = ((loc - S[k]) * NS[k]).sum(1) < 0
                if not inside.any():
                    continue
                depth[i] = d[inside].max()
                nrm = (NS[k[inside]] * d[inside, None]).sum(0)
                nrm = nrm / max(_np.linalg.norm(nrm), 1e-9)
                shift_cam[i] = Rc[t].T @ (T[t, :3, :3] @ nrm * (depth[i] + _H5["margin"]))
            if before is None:
                before = depth.copy()
            after = depth
            if it == 3 or not depth.any():
                break
            tr = data[f"hand_{side}_wrist_trans_in_camera"].astype(_np.float64)
            tr[fr] += shift_cam
            data[f"hand_{side}_wrist_trans_in_camera"] = tr.astype(data[f"hand_{side}_wrist_trans_in_camera"].dtype)
            moved.append(int((depth > 0).sum()))
        w2c = data[f"hand_{side}_wrist_to_camera_transform"].astype(_np.float64)
        w2c[:, :3, 3] = data[f"hand_{side}_wrist_trans_in_camera"]
        data[f"hand_{side}_wrist_to_camera_transform"] = w2c.astype(_np.float32)
        data[f"hand_{side}_wrist_to_world_transform"] = (c2w @ w2c).astype(_np.float32)
        info[f"h5_push_{side}_{o}"] = {"frames": [int(fr[0]), int(fr[-1])], "n_frames": int(len(fr)),
                                       "pen_before_mm_max": round(float(before.max()) * 1000, 2),
                                       "pen_frames_before": int((before > 0).sum()), "moved_per_pass": moved,
                                       "pen_after_mm_max": round(float(after.max()) * 1000, 2),
                                       "pen_frames_after": int((after > 0).sum())}
    return info


def attach_hands(data, bundle, *args, **kwargs):
    info = _h5_orig_attach_hands(data, bundle, *args, **kwargs)
    H5_INFO["_attach"] = {k: v for k, v in info.items() if k.startswith("attached_")}
    if _H5["push"]:
        H5_INFO.update(_h5_push(data, bundle))
    return info


def cleaned_bundle(*args, **kwargs):
    dest, info = _h5_orig_cleaned_bundle(*args, **kwargs)
    info.update({k: v for k, v in H5_INFO.items() if not k.startswith("_")})
    info["h5_cfg"] = dict(_H5)
    return dest, info
