# hand5 ep10 patch body (exec'd by an arm file that sets _H5_MODE first, inside rlr_h5.py globals).
# Redefines attach_hands only. For objects that are NOT axially symmetric everything is identical to the team
# attach_hands. For axially symmetric objects (kit tracks/track_3/README: beige_cup, blue_cup; axial spin is
# unscored) the mode decides:
#   spinfree : hand-side only. Per hold the grip is frozen at the closest frame k as before, but the hand rides
#              T[t] @ S(-theta_t) @ grip, S = rotation about the cup's mesh symmetry axis (through its surface
#              centroid) and theta_t = twist of R[k]^T R[t] about that axis: the hand follows the cup's
#              translation and tilt but not its (unscored, FoundationPose-noise) spin. Blend in/out with a
#              continuous-branch relative rotation. Object reference untouched.
#   despin   : the object reference itself is de-spun relative to frame 0 (T[t] @ S(-theta_t), theta of
#              R[0]^T R[t]); then attach as spinfree. Changes the object track only by axial symmetries.
#   noattach : holds on symmetric objects are not attached (observed hand kept).
# Diagnostics (pre/post wrist poses, j0, axes, holds) -> /scratch/haisenk-v2d/hand5/ep10/diag/<seq>.npz
import sys as _sys

_H5_SYM_NAMES = {"blue_cup", "beige_cup"}
_H5_RESPIN_FREEZE_BLEND = globals().get("_H5_RESPIN_FREEZE_BLEND", False)


def _h5_sym_axis(v_scaled, faces):
    """Mesh symmetry axis (unit, object frame) and a point on it (area-weighted surface centroid)."""
    m = trimesh.Trimesh(v_scaled, faces, process=False)
    pts, _ = trimesh.sample.sample_surface(m, 20000, seed=0)
    c = pts.mean(0)
    w, U = np.linalg.eigh(np.cov((pts - c).T))
    sep = [min(abs(w[j] - w[(j + 1) % 3]), abs(w[j] - w[(j + 2) % 3])) for j in range(3)]
    a = U[:, int(np.argmax(sep))]
    return a / np.linalg.norm(a), c


def _h5_twist(Rrel, a):
    """Angle (rad) of the twist of rotation Rrel about unit axis a (swing-twist, Rrel = swing @ twist)."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(Rrel).as_quat()  # x y z w
    th = 2.0 * np.arctan2(float(np.dot(q[:3], a)), float(q[3]))
    return (th + np.pi) % (2 * np.pi) - np.pi


def _h5_spin(a, c, phi):
    """4x4 rotation by phi about the axis line (a through c), object frame."""
    from scipy.spatial.transform import Rotation
    S = np.eye(4)
    S[:3, :3] = Rotation.from_rotvec(a * phi).as_matrix()
    S[:3, 3] = c - S[:3, :3] @ c
    return S


def _h5_despun(T, a, c, k):
    """T[t] @ S(-theta_t), theta_t = twist of R[k]^T R[t] about a: equal to T[k]'s spin at every frame."""
    out = T.copy()
    th = np.zeros(len(T))
    for t in range(len(T)):
        th[t] = _h5_twist(T[k, :3, :3].T @ T[t, :3, :3], a)
        out[t] = T[t] @ _h5_spin(a, c, -th[t])
    return out, th


def _h5_az(T, P, a, c):
    """Azimuth (rad, unwrapped over time) of world points P[t] about the axis line (a through c) in object frame T[t]."""
    e1 = np.cross(a, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(a, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    x = np.einsum("tji,tj->ti", T[:, :3, :3], P - T[:, :3, 3]) - c
    return np.unwrap(np.arctan2(x @ e2, x @ e1))


def _h5_smooth(x, sigma=2.0):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x, sigma, mode="nearest") if len(x) > 1 else x


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
    names = [str(x) for x in data["object_names"]] if "object_names" in data else [""]
    sym = {}
    info = {"h5_sym_mode": _H5_MODE}
    for i, (pose_key, scale_key, mesh_file) in enumerate(object_slots(data)):
        if names[i] in _H5_SYM_NAMES:
            mesh = trimesh.load(bundle / mesh_file, force="mesh", process=False)
            a, c = _h5_sym_axis(np.asarray(mesh.vertices) * float(data.get(scale_key, 1.0)), np.asarray(mesh.faces))
            sym[i] = (a, c)
            info[f"h5_sym_axis_{i}"] = [round(float(x), 4) for x in a]
            info[f"h5_sym_center_cm_{i}"] = [round(float(x) * 100, 2) for x in c]
            if _H5_MODE == "despin":
                T = data[pose_key].astype(np.float64)
                Tds, th = _h5_despun(T, a, c, 0)
                info[f"h5_despin_max_deg_{i}"] = round(float(np.degrees(np.abs(th)).max()), 1)
                data[pose_key] = Tds.astype(data[pose_key].dtype)
    contact = mask_contact(bundle, data, n)
    info["attach_rule"] = "mask contact" if contact else "nearest within reach"

    def detect():
        objects = []
        for i, (pose_key, scale_key, mesh_file) in enumerate(object_slots(data)):
            T = data[pose_key].astype(np.float64)
            v = object_vertices(bundle, data, scale_key, mesh_file)
            v = v[:: max(1, len(v) // 500)]
            surface = np.einsum("tij,vj->tvi", T[:, :3, :3], v) + T[:, None, :3, 3]
            lifted = np.zeros(n, bool)
            for a_, b_ in lifted_segments(T[:, 2, 3], lift):
                lifted[a_:b_] = True
            near, dist = {}, {}
            for side, (H, _) in hands.items():
                d = np.linalg.norm(surface - H[:, None, :3, 3], axis=-1).min(axis=1)
                near[side] = np.where(lifted & (d <= reach), d, np.inf)
                dist[side] = d
            objects.append((i, T, near, dist, lifted))
        owners = {}
        for side in hands:
            if contact is not None:
                C = contact[side]
                D = np.stack([np.where(lifted & (C[o] >= min_contact), 1.0 / (1.0 + C[o]), np.inf)
                              for o, (_, _, _, _, lifted) in enumerate(objects)])
                owners[side] = hold_owner(D, switch=0.0, gap=10, min_len=2 * blend + 1, ratio=2.0)
            else:
                D = np.stack([near[side] for _, _, near, _, _ in objects])
                owners[side] = hold_owner(D, switch=0.02, gap=10, min_len=2 * blend + 1)
        return objects, owners

    objects, owners = detect()
    if _H5_MODE == "respin" and sym:
        # object spin follows the observed hand: frame-0 spin, plus, during each hold, the change of the holding
        # wrist's azimuth about the cup axis (in the frame-0-spin cup frame); constant outside holds
        for i, (a_ax, c_ax) in sym.items():
            pose_key = object_slots(data)[i][0]
            T = data[pose_key].astype(np.float64)
            Tds, th0 = _h5_despun(T, a_ax, c_ax, 0)
            psi = np.zeros(n)
            hl = sorted([(a, b, side) for side in hands for o, a, b in holds(owners[side]) if o == i])
            last, cur = 0, 0.0
            for a, b, side in hl:
                if a < last:
                    continue  # overlapping hold of the other hand: the first one drives the spin
                psi[last:a] = cur
                az = _h5_az(Tds[a:b], hands[side][0][a:b, :3, 3], a_ax, c_ax)
                if _H5_RESPIN_FREEZE_BLEND and b - a > 2 * blend + 2:  # spin only while fully held, not as the hand comes/goes
                    idx = np.clip(np.arange(b - a), blend, b - a - blend - 1)
                    az = az[idx]
                psi[a:b] = cur + (az - az[0])
                cur, last = float(psi[b - 1]), b
            psi[last:] = cur
            psi = _h5_smooth(psi)
            Tn = np.stack([Tds[t] @ _h5_spin(a_ax, c_ax, psi[t]) for t in range(n)])
            info[f"h5_respin_psi_deg_range_{i}"] = [round(float(np.degrees(psi.min())), 1), round(float(np.degrees(psi.max())), 1)]
            info[f"h5_respin_removed_max_deg_{i}"] = round(float(np.degrees(np.abs(((psi - th0 + np.pi) % (2 * np.pi)) - np.pi)).max()), 1)
            data[pose_key] = Tn.astype(data[pose_key].dtype)
        objects, owners = detect()
    moved = {side: np.zeros(n, bool) for side in hands}
    H0s = {}
    hold_list = []
    for side, (H, _) in hands.items():
        H0 = H.copy()
        H0s[side] = H0
        for o, a, b in holds(owners[side]):
            i, T, _, dist, _ = objects[o]
            k = a + int(np.argmin(dist[side][a:b]))
            hold_list.append((0 if side == "left" else 1, i, a, b, k))
            if i in sym and _H5_MODE == "noattach":
                info.setdefault(f"h5_skipped_{i}", []).append([side, int(a), int(b), round(float(dist[side][k]) * 100, 1)])
                continue
            grip = np.linalg.inv(T[k]) @ H0[k]
            if i in sym:
                if _H5_MODE == "respin":
                    Tuse, th = T[a:b], np.zeros(b - a)
                elif _H5_MODE == "azfollow":  # re-spin so the grip's azimuth follows the observed wrist's azimuth
                    az = _h5_az(T[a:b], H0[a:b, :3, 3], *sym[i])
                    phi = _h5_smooth(az - az[k - a])
                    Tuse = np.stack([T[t] @ _h5_spin(*sym[i], phi[t - a]) for t in range(a, b)])
                    th = np.array([_h5_twist(T[k, :3, :3].T @ Tuse[t - a, :3, :3], sym[i][0]) for t in range(a, b)])
                else:
                    Tuse, th = _h5_despun(T[a:b], *sym[i], k - a)
                info.setdefault(f"h5_spin_removed_deg_{i}", []).append(
                    [side, round(float(np.degrees(np.abs(th)).max()), 1)])
                prev = None
                for t in range(a, b):
                    w = min(1.0, (t - a + 1) / (blend + 1), (b - t) / (blend + 1))
                    target = Tuse[t - a] @ grip
                    q0 = Rotation.from_matrix(H0[t, :3, :3])
                    rv = (q0.inv() * Rotation.from_matrix(target[:3, :3])).as_rotvec()
                    ang = np.linalg.norm(rv)
                    if prev is not None and ang > 1e-9:  # continuous branch: the equivalent rotvec closest to the last one
                        alt = rv - 2 * np.pi * rv / ang
                        if np.linalg.norm(alt - prev) < np.linalg.norm(rv - prev):
                            rv = alt
                    prev = rv
                    H[t, :3, :3] = (q0 * Rotation.from_rotvec(w * rv)).as_matrix()
                    H[t, :3, 3] = (1 - w) * H0[t, :3, 3] + w * target[:3, 3]
            else:  # team behaviour, unchanged
                for t in range(a, b):
                    w = min(1.0, (t - a + 1) / (blend + 1), (b - t) / (blend + 1))
                    target = T[t] @ grip
                    q = Rotation.from_matrix([H0[t, :3, :3], target[:3, :3]])
                    H[t, :3, :3] = (q[0] * Rotation.from_rotvec(w * (q[0].inv() * q[1]).as_rotvec())).as_matrix()
                    H[t, :3, 3] = (1 - w) * H0[t, :3, 3] + w * target[:3, 3]
            moved[side][a:b] = True
            info.setdefault(f"attached_{i}", []).append([side, int(a), int(b), round(float(dist[side][k]) * 100, 1)])
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
    try:
        seq = _sys.argv[_sys.argv.index("--sequence-id") + 1]
        dd = Path("/scratch/haisenk-v2d/hand5/ep10/diag")
        dd.mkdir(parents=True, exist_ok=True)
        np.savez(dd / f"{seq}.npz", H0_left=H0s["left"], H0_right=H0s["right"], H_left=hands["left"][0],
                 H_right=hands["right"][0], j0_left=hands["left"][1], j0_right=hands["right"][1],
                 holds=np.array(hold_list, dtype=np.int64).reshape(-1, 5),
                 axes=np.array([sym[i][0] if i in sym else np.zeros(3) for i in range(len(objects))]),
                 centers=np.array([sym[i][1] if i in sym else np.zeros(3) for i in range(len(objects))]),
                 T0=objects[0][1], T1=objects[1][1] if len(objects) > 1 else objects[0][1])
        info["h5_diag"] = str(dd / f"{seq}.npz")
    except Exception as e:  # diagnostics never break a build
        info["h5_diag_error"] = repr(e)
    return info
