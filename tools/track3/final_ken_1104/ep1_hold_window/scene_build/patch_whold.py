# w_hold_ep1 (3 Nov, Wentao base) H5_PATCH: team attach_hands + evidence-backed forced hold windows for ep1 (from ep1n patch_hold.py,
# 2 Nov). Windows: left on white_pot [32,304) (raw grip stable in the pot frame over f30-305, contact labels f28-308);
# right on white_pot_lid [45,68) and [250,274) (lid speed > 0.5 cm/frame: carried down to the table / back onto the pot).
# Grip = median offset of the unmoved hand over the hold interior. Only hand_*_wrist arrays change.
# shortest-path rotation blend over `blend` frames), plus forced hold windows for episode 1 from GT-free evidence
# (MANO fingertip-to-object distances, object motion, overlays). Forced windows override the team's per-frame owner
# for that hand; everything else (rest hold, levelling, objects, the other frames) is untouched: attach_hands only
# rewrites hand_*_wrist arrays. Windows: env EP1N_FORCE json {"left": [[slot, a, b], ...], "right": [...]}, b exclusive,
# default below. slot 0 = white_pot, slot 1 = white_pot_lid.
import json as _ep1n_json, os as _ep1n_os
_EP1N_GRIP = _ep1n_os.environ.get("EP1N_GRIP", "median")  # team: offset at the closest-wrist frame; median: robust
# median offset of the unmoved hand over the hold's interior frames (rotation mean + median translation)
_EP1N_FORCE = _ep1n_json.loads(_ep1n_os.environ.get("EP1N_FORCE") or _ep1n_json.dumps(
    {"left": [[0, 32, 304]], "right": [[1, 45, 68], [1, 250, 274]]}))


def attach_hands(data: dict, bundle: Path, lift: float = 0.02, reach: float = 0.12, blend: int = 5,
                 min_contact: float = 10.0) -> dict:
    from scipy.spatial.transform import Rotation
    c2w = data["camera_to_world_transform"].astype(np.float64)
    Rc, tc = c2w[:, :3, :3], c2w[:, :3, 3]
    mirror = np.diag([1.0, -1.0, -1.0])
    hands = {}
    for side in ("left", "right"):
        j0 = mano_rest_wrist(side, data[f"hand_{side}_betas"].reshape(10))
        R = Rotation.from_rotvec(data[f"hand_{side}_wrist_orient_in_camera"].astype(np.float64)).as_matrix()
        if side == "left":
            R = mirror @ R @ mirror
        H = np.tile(np.eye(4), (len(c2w), 1, 1))
        H[:, :3, :3] = Rc @ R
        H[:, :3, 3] = np.einsum("nij,nj->ni", Rc, data[f"hand_{side}_wrist_trans_in_camera"] + j0) + tc
        hands[side] = (H, j0)
    n = len(c2w)
    objects = []
    for i, (pose_key, scale_key, mesh_file) in enumerate(object_slots(data)):
        T = data[pose_key].astype(np.float64)
        v = object_vertices(bundle, data, scale_key, mesh_file)
        v = v[:: max(1, len(v) // 500)]
        surface = np.einsum("tij,vj->tvi", T[:, :3, :3], v) + T[:, None, :3, 3]
        lifted = np.zeros(n, bool)
        for a, b in lifted_segments(T[:, 2, 3], lift):
            lifted[a:b] = True
        near, dist = {}, {}
        for side, (H, _) in hands.items():
            d = np.linalg.norm(surface - H[:, None, :3, 3], axis=-1).min(axis=1)
            near[side] = np.where(lifted & (d <= reach), d, np.inf)
            dist[side] = d
        objects.append((i, T, near, dist, lifted))
    contact = mask_contact(bundle, data, n)
    info, moved = {"attach_rule": "mask contact" if contact else "nearest within reach"}, {side: np.zeros(n, bool) for side in hands}
    info["ep1n_forced_windows"] = _EP1N_FORCE
    info["ep1n_grip"] = _EP1N_GRIP
    for side, (H, _) in hands.items():
        H0 = H.copy()
        if contact is not None:
            C = contact[side]
            D = np.stack([np.where(lifted & (C[o] >= min_contact), 1.0 / (1.0 + C[o]), np.inf)
                          for o, (_, _, _, _, lifted) in enumerate(objects)])
            owner = hold_owner(D, switch=0.0, gap=10, min_len=2 * blend + 1, ratio=2.0)
        else:
            D = np.stack([near[side] for _, _, near, _, _ in objects])
            owner = hold_owner(D, switch=0.02, gap=10, min_len=2 * blend + 1)
        info[f"ep1n_team_holds_{side}"] = [list(h) for h in holds(owner)]
        for o_, a_, b_ in _EP1N_FORCE.get(side, []):  # ep1n: evidence-backed windows override the team owner
            owner[int(a_):int(b_)] = int(o_)
        for o, a, b in holds(owner):
            i, T, _, dist, _ = objects[o]
            k = a + int(np.argmin(dist[side][a:b]))
            grip = np.linalg.inv(T[k]) @ H0[k]
            core = np.arange(a + blend, b - blend) if b - a > 2 * blend + 2 else np.arange(a, b)
            rel = np.linalg.inv(T[core]) @ H0[core]
            gmed = np.eye(4)
            gmed[:3, :3] = Rotation.from_matrix(rel[:, :3, :3]).mean().as_matrix()
            gmed[:3, 3] = np.median(rel[:, :3, 3], axis=0)
            def _dev(g):
                dt = np.linalg.norm(rel[:, :3, 3] - g[:3, 3], axis=1) * 1000
                da = np.degrees((Rotation.from_matrix(rel[:, :3, :3]).inv() * Rotation.from_matrix(g[:3, :3])).magnitude())
                return [round(float(np.median(dt)), 1), round(float(np.percentile(dt, 90)), 1),
                        round(float(np.median(da)), 1), round(float(np.percentile(da, 90)), 1)]
            info.setdefault("ep1n_grip_dev_mm_deg", []).append(
                [side, int(o), int(a), int(b), int(k), "team(med,p90 mm; med,p90 deg)", _dev(grip), "median", _dev(gmed)])
            if _EP1N_GRIP == "median" and any(int(a_) <= a < int(b_) for o_, a_, b_ in _EP1N_FORCE.get(side, [])):
                grip = gmed
            for t in range(a, b):
                w = min(1.0, (t - a + 1) / (blend + 1), (b - t) / (blend + 1))
                target = T[t] @ grip
                q = Rotation.from_matrix([H0[t, :3, :3], target[:3, :3]])
                H[t, :3, :3] = (q[0] * Rotation.from_rotvec(w * (q[0].inv() * q[1]).as_rotvec())).as_matrix()
                H[t, :3, 3] = (1 - w) * H0[t, :3, 3] + w * target[:3, 3]
            moved[side][a:b] = True
            info.setdefault(f"attached_{i}", []).append([side, int(a), int(b), round(float(dist[side][k]) * 100, 1)])
            info.setdefault("ep1n_grip_frames", []).append([side, int(o), int(a), int(b), int(k)])
    for side, (H, j0) in hands.items():
        if not moved[side].any():
            continue
        R = np.einsum("nji,njk->nik", Rc, H[:, :3, :3])
        if side == "left":
            R = mirror @ R @ mirror
        cam_t = np.einsum("nji,nj->ni", Rc, H[:, :3, 3] - tc) - j0
        m = moved[side]
        data[f"hand_{side}_wrist_orient_in_camera"][m] = Rotation.from_matrix(R[m]).as_rotvec()
        data[f"hand_{side}_wrist_trans_in_camera"][m] = cam_t[m]
        w2c = data[f"hand_{side}_wrist_to_camera_transform"].astype(np.float64)
        w2c[m, :3, :3] = Rotation.from_rotvec(data[f"hand_{side}_wrist_orient_in_camera"][m].astype(np.float64)).as_matrix()
        w2c[m, :3, 3] = data[f"hand_{side}_wrist_trans_in_camera"][m]
        data[f"hand_{side}_wrist_to_camera_transform"] = w2c.astype(np.float32)
        data[f"hand_{side}_wrist_to_world_transform"] = (c2w @ w2c).astype(np.float32)
        info[f"hand_{side}_attached_frames"] = int(m.sum())
    print(f"== ep1n attach: forced {_EP1N_FORCE} grips {info.get('ep1n_grip_frames')}", flush=True)
    return info


