"""Synthesis verifier (ep1, candidate r_ep1_rl_K_s43 @95M vs adopted o_ep1_hold_s44 final).
Shared loaders + a registration-aware emulation of AUC.py built ONLY from the unmodified kit scorer (in/AUC_kit.py,
md5 93a20d94). Reads only copies in ../in. Writes nothing by itself."""
import os, io, base64, json, hashlib, importlib.util
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rot

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(D, "in")
OUT = os.path.join(D, "out")
NAMES = ["white_pot", "white_pot_lid"]
PH = dict(A=(0, 26), B=(27, 99), C=(100, 235), D=(236, 307), E=(308, 334))
POSE = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]


def load_auc():
    spec = importlib.util.spec_from_file_location("auc_kit", os.path.join(IN, "AUC_kit.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = load_auc()


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def load_obj(path):
    V, F = [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) for t in line.split()[1:]]
            for i in range(1, len(idx) - 1):
                F.append([idx[0] - 1, idx[i] - 1, idx[i + 1] - 1])
    return np.array(V, float), np.array(F, np.int64)


def load_rollout(path):
    """-> achieved, reference (N, 2, 7) xyzw in NAMES order (env 0)."""
    d = pq.read_table(path).to_pydict()
    env = np.array(d["environment_id"]); sel = env == env.min()
    fr = np.array(d["reference_frame"])[sel].astype(int)
    bn = np.array([str(x) for x in d["object_body_name"]])[sel]
    oid = np.array(d["object_id"])[sel].astype(int)
    N = fr.max() + 1
    out = {}
    for pref in ("achieved", "reference"):
        X = np.full((N, 2, 7), np.nan)
        cols = [f"{pref}_position_{a}" for a in "xyz"] + [f"{pref}_quaternion_{a}" for a in "xyzw"]
        M = np.stack([np.array(d[c], float)[sel] for c in cols], 1)
        for k, nm in enumerate(NAMES):
            m = oid == k
            assert m.sum() == N and all(b.endswith(nm) for b in bn[m]), (path, nm, m.sum())
            X[fr[m], k] = M[m]
        out[pref] = X
    return out["achieved"], out["reference"]


def load_packed(path, episode=1):
    """packed submission (one episode or full file) -> poses (N, K, 7), meshes {slot: (V, F)} decoded with AUC.py."""
    t = pq.read_table(path).to_pandas()
    t = t[t.row_id.str.startswith(f"track_3/e{episode:06d}/")].reset_index(drop=True)
    rid = t.row_id.str.extract(r"e(\d+)/w(\d+)/f(\d+)/o(\d+)").astype(int)
    N, K = rid[2].max() + 1, rid[3].max() + 1
    P = np.full((N, K, 7), np.nan)
    P[rid[2].values, rid[3].values] = t[POSE].to_numpy(float)
    meshes, raw = {}, {}
    enc = t.object_mesh_b64.fillna("").astype(str)
    for i in np.where(enc.str.strip() != "")[0]:
        v, f = A._decode_object_mesh(enc.iloc[i])
        meshes[int(rid[3].iloc[i])] = (np.asarray(v, float), np.asarray(f, np.int64))
        raw[int(rid[3].iloc[i])] = hashlib.md5(enc.iloc[i].encode()).hexdigest()
    return P, meshes, raw, t


def T_to_X(T):
    T = np.asarray(T, float)
    return np.concatenate([T[..., :3, 3], Rot.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(T.shape[:-2] + (4,))], -1)


def X_to_T(X):
    T = np.zeros(X.shape[:-1] + (4, 4))
    T[..., :3, :3] = Rot.from_quat(X[..., 3:].reshape(-1, 4)).as_matrix().reshape(X.shape[:-1] + (3, 3))
    T[..., :3, 3] = X[..., :3]; T[..., 3, 3] = 1
    return T


def xf(T, P):
    return P @ T[..., :3, :3].swapaxes(-1, -2) + T[..., None, :3, 3]


# ---------------------------------------------------------------- scorer emulation (AUC._score_registered_track3 body)
_REG = {}


def register(candidate, gt, anchor, seed):
    h = hashlib.sha1()
    for x in (candidate[0], candidate[1], gt[0], gt[1], np.asarray(anchor, float)):
        h.update(np.ascontiguousarray(x).tobytes())
    h.update(str(seed).encode())
    k = h.hexdigest()
    if k not in _REG:
        _REG[k] = A._register_object_body(candidate, gt, anchor, seed)
    ca, R = _REG[k]
    return ca.copy(), R.copy()


class GT:
    """hidden-GT stand-in: meshes (pot, lid), anchors (surface centroid), ADD points (surface samples)."""
    def __init__(self, pot, lid, npts=3000, anchor="centroid"):
        self.meshes = [pot, lid]
        self.anchors, self.points = [], []
        for i, (v, f) in enumerate(self.meshes):
            c, _ = A._surface_moments(v, f)
            a = {"centroid": c, "bbox": 0.5 * (v.min(0) + v.max(0)), "origin": np.zeros(3)}[anchor]
            self.anchors.append(a)
            self.points.append(A._sample_mesh_surface(v, f, npts, 7 + i))


def add_series(ach, sub_meshes, truth, gt, identity=False, seed_override=None):
    """ach, truth (N, 2, 7) xyzw. Returns per-frame ADD (N, 2) in metres, after AUC.py registration + alignment.
    identity=True replaces the pot+lid registration by the true correspondence (stand-ins share our body frame)."""
    achieved = A._normalise_quaternions(ach[:, None].astype(float).copy(), "submission", ValueError).copy()
    reference = A._normalise_quaternions(truth[:, None].astype(float).copy(), "solution", ValueError).copy()
    verts, axes, orders, regs = [], [], [], []
    for slot in range(2):
        anchor = gt.anchors[slot]
        seed = 1_000_003 * 1 + 101 * slot if seed_override is None else seed_override + 101 * slot
        if identity:
            ca, R = anchor.copy(), np.eye(3)
        else:
            ca, R = register(sub_meshes[slot], gt.meshes[slot], anchor, seed)
        regs.append((ca, R))
        achieved[:, :, slot] = A._convert_object_body(achieved[:, :, slot], ca, R)
        reference[:, :, slot, :3] += np.einsum("...ij,j->...i", A._rotation_matrix(reference[:, :, slot, 3:]), anchor)
        verts.append(gt.points[slot] - anchor); axes.append(np.zeros(3)); orders.append(1)
    aligned = A._align_registered_scene(achieved, reference, axes)
    add = np.stack([A._symmetry_point_errors(aligned[:, :, i], reference[:, :, i], verts[i], axes[i], orders[i])
                    for i in range(2)], axis=-1)
    return add[:, 0], regs, aligned[:, 0], reference[:, 0]


def auc(add):
    """episode AUC (mean over bodies) from (N, 2) metres, AUC.py trapezoid."""
    return A._add_auc(add[:, None])


def auc_body(add, k, s=0, e=None):
    e = add.shape[0] - 1 if e is None else e
    return A._add_auc(add[s:e + 1, None, k:k + 1])


def phases(add):
    return {p: dict(pot=auc_body(add, 0, s, e), lid=auc_body(add, 1, s, e)) for p, (s, e) in PH.items()}


def other_metrics(aligned, reference):
    """MP-SR / SP-SR / RPE / MPPE on already registered + aligned poses (non-symmetric objects)."""
    a = aligned[:, None]; r = reference[:, None]; axes = [np.zeros(3), np.zeros(3)]; orders = [1, 1]
    out = {}
    for m in ("spider_sr", "maniptrans_sr", "rpe_cm", "mppe_cm"):
        out[m] = A._registered_pose_metric(m, a, r, [None, None], axes, orders)
    pos = np.linalg.norm(a[..., :3] - r[..., :3], axis=-1).mean(axis=0)[0]
    ang = np.degrees(A._symmetry_orientation_errors(a, r, axes, orders).mean(axis=0)[0])
    out["mean_pos_cm"] = [round(100 * float(x), 2) for x in pos]
    out["mean_rot_deg"] = [round(float(x), 2) for x in ang]
    return out
