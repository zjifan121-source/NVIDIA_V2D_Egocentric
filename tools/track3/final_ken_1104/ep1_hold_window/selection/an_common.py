"""common loaders for ep5 object diagnostics (local copies of the orchard scene/rollouts; read-only)."""
import os, glob, json
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rot

ROOT = r"C:\v2d_demo\diag\ep5_objA\data"
SCN = os.path.join(ROOT, "scenes", "w7_attcom_ep5", "assets", "human_motion_data", "ego_recon")
PROC = os.path.join(SCN, "processed")
PQ = glob.glob(os.path.join(PROC, "sequence_id=*", "robot_name=*", "*.parquet"))[0]
ROLL = {"s42@95M": os.path.join(ROOT, "runs", "o_ep5_attcom_s42", "early_95006720", "results", "episode_000005.parquet"),
        "s43@90M": os.path.join(ROOT, "runs", "o_ep5_attcom_s43", "early_90005504", "results", "episode_000005.parquet")}
NAMES = ["white_pot", "wooden_spoon"]
FPS = 20.254119873046875


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


def to_T(R, t):
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def wxyz_to_R(q):
    q = np.asarray(q, float)
    return Rot.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))


def load_scene(pqpath=PQ, proc=PROC, names=NAMES):
    cols = ["object_body_names", "object_mesh_paths", "object_body_position", "object_body_wxyz", "mano_left_joints",
            "mano_right_joints", "mano_left_object_contact_part_ids", "mano_right_object_contact_part_ids", "mano_link_names",
            "mano_left_global_orient", "mano_right_global_orient", "object_urdf_paths"]
    t = pq.read_table(pqpath, columns=[c for c in cols if c in pq.read_schema(pqpath).names])
    d = {c: t.column(c)[0].as_py() for c in t.column_names}
    pos = np.array(d["object_body_position"], float)
    wxyz = np.array(d["object_body_wxyz"], float)
    N = pos.shape[0]
    objs = []
    for k, nm in enumerate(names):
        mp = os.path.join(proc, os.path.basename(d["object_mesh_paths"][k]))
        V, F = load_obj(mp)
        R = wxyz_to_R(wxyz[:, k])
        objs.append(dict(name=nm, V=V, F=F, T=to_T(R, pos[:, k]), R=R, p=pos[:, k], mesh=mp))
    J = {s: np.array(d[f"mano_{s}_joints"], float) for s in ("left", "right")}
    C = {s: np.array(d[f"mano_{s}_object_contact_part_ids"], np.int64) for s in ("left", "right")}
    return dict(N=N, objs=objs, joints=J, contact=C, links=d.get("mano_link_names"), body_names=d["object_body_names"], raw=d)


def load_rollout(path, body_names, N):
    t = pq.read_table(path)
    d = t.to_pydict()
    env = np.array(d["environment_id"])
    sel = env == env.min()
    frame = np.array(d.get("reference_frame", d["step"]))[sel].astype(int)
    bname = np.array([str(x) for x in d["object_body_name"]])[sel]
    out = {}
    for k, nm in enumerate(body_names):
        rows = np.where(bname == nm)[0]
        fr = frame[rows]
        ach = np.full((N, 4, 4), np.nan)
        ref = np.full((N, 4, 4), np.nan)
        for pref, dst in (("achieved", ach), ("reference", ref)):
            p = np.stack([np.array(d[f"{pref}_position_{a}"], float)[sel][rows] for a in "xyz"], 1)
            q = np.stack([np.array(d[f"{pref}_quaternion_{a}"], float)[sel][rows] for a in "xyzw"], 1)
            ok = (fr >= 0) & (fr < N)
            dst[fr[ok]] = to_T(Rot.from_quat(q[ok]).as_matrix(), p[ok])
        out[k] = dict(ach=ach, ref=ref)
    return out


def xf(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def rot_angle_deg(Ra, Rb):
    c = (np.einsum("...ij,...ij->...", Ra, Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def sample_surface(V, F, n, seed=0):
    rng = np.random.default_rng(seed)
    tri = V[F]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cr, axis=1) / 2
    fi = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    fl = u + v > 1
    u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    P = tri[fi, 0] + u[:, None] * (tri[fi, 1] - tri[fi, 0]) + v[:, None] * (tri[fi, 2] - tri[fi, 0])
    return P, cr[fi] / np.maximum(np.linalg.norm(cr[fi], axis=1, keepdims=True), 1e-12)


THR = np.arange(0.01, 0.10, 0.01)


def auc_cm(add_cm):
    a = np.asarray(add_cm, float)
    a = a[np.isfinite(a)] / 100.0
    if a.size == 0:
        return None
    ys = np.array([np.mean(a < t) for t in THR])
    dx = 1.0 / (THR.size - 1)
    return float(((ys[1:] + ys[:-1]) / 2).sum() * dx)
