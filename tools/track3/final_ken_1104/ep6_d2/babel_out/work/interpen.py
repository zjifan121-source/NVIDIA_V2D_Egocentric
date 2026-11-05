#!/usr/bin/env python3
"""Object-object interpenetration in a scene parquet (reference) and optionally a rollout parquet.
GT-free. Uses the processed meshes the scene parquet points to (same resolution as render_review).
Inside test = generalized winding number (> 0.5) of one mesh's surface samples w.r.t. the other mesh;
depth = distance to the other mesh's surface (dense samples, ~1 mm)."""
import sys, os, json, glob, argparse
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--rollout", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--nsamp", type=int, default=6000)
ap.add_argument("--cand_cm", type=float, default=1.5)
a = ap.parse_args()

proc = os.path.join(a.scene, "assets/human_motion_data/ego_recon/processed")
pqs = sorted(glob.glob(os.path.join(proc, "sequence_id=*/robot_name=*/*.parquet")), key=os.path.getmtime)
P = pqs[-1]
t = pq.read_table(P, columns=["object_body_names", "object_mesh_paths", "object_body_position", "object_body_wxyz"])
d = {c: t.column(c)[0].as_py() for c in t.column_names}
names = list(d["object_body_names"]); mps = list(d["object_mesh_paths"])
pos = np.array(d["object_body_position"], float); wxyz = np.array(d["object_body_wxyz"], float)
N, K = pos.shape[0], pos.shape[1]
print("parquet", P, "frames", N, "bodies", names, flush=True)

def load_obj(p):
    V, F = [], []
    for ln in open(p, errors="ignore"):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            idx = [int(x.split("/")[0]) for x in ln.split()[1:]]
            idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            for j in range(1, len(idx) - 1):
                F.append([idx[0], idx[j], idx[j + 1]])
    return np.array(V, float), np.array(F, np.int64)

meshes = []
for k in range(K):
    mp = mps[k]
    cand = [mp, os.path.join(proc, os.path.basename(mp))]
    mpath = next(c for c in cand if c and os.path.isfile(c))
    V, F = load_obj(mpath)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(1234 + k)
    def samp(n):
        fi = rng.choice(len(F), n, p=area / area.sum())
        u, v = rng.random(n), rng.random(n)
        m = u + v > 1; u[m], v[m] = 1 - u[m], 1 - v[m]
        tt = tri[fi]
        return tt[:, 0] + u[:, None] * (tt[:, 1] - tt[:, 0]) + v[:, None] * (tt[:, 2] - tt[:, 0])
    S = samp(a.nsamp)
    D = samp(200000)
    # signed volume (closedness / orientation hint) + boundary edges
    vol = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    meshes.append(dict(name=names[k], path=mpath, V=V, F=F, tri=tri, S=S, tree=cKDTree(D),
                       vol_cm3=vol * 1e6, boundary_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
                       extents_cm=((V.max(0) - V.min(0)) * 100).round(2).tolist()))
    print(f"mesh {names[k]}: {mpath} V{len(V)} F{len(F)} vol {vol*1e6:.1f} cm3 boundary_edges {(cnt==1).sum()} nonmanifold {(cnt>2).sum()} extents_cm {meshes[-1]['extents_cm']}", flush=True)

def winding(Q, tri):
    """generalized winding number of points Q (M,3) w.r.t. triangles tri (F,3,3)."""
    out = np.zeros(len(Q))
    for s in range(0, len(Q), 64):
        q = Q[s:s + 64]
        A = tri[None, :, 0, :] - q[:, None, :]
        B = tri[None, :, 1, :] - q[:, None, :]
        C = tri[None, :, 2, :] - q[:, None, :]
        la, lb, lc = np.linalg.norm(A, axis=2), np.linalg.norm(B, axis=2), np.linalg.norm(C, axis=2)
        det = np.einsum("mfi,mfi->mf", A, np.cross(B, C))
        den = la * lb * lc + np.einsum("mfi,mfi->mf", A, B) * lc + np.einsum("mfi,mfi->mf", B, C) * la + np.einsum("mfi,mfi->mf", C, A) * lb
        out[s:s + 64] = (2 * np.arctan2(det, den)).sum(1) / (4 * np.pi)
    return out

def poses_from(pos_, wxyz_):
    R = Rot.from_quat(wxyz_[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(pos_.shape[:-1] + (3, 3))
    return R

def run(pos_, R_, tag):
    res = {}
    for i in range(K):
        for j in range(K):
            if i == j: continue
            mi, mj = meshes[i], meshes[j]
            n_in = np.zeros(N, int); dmax = np.zeros(N); dmin_surf = np.full(N, np.nan)
            for f in range(N):
                # points of mesh i in body frame of j
                Pw = mi["S"] @ R_[f, i].T + pos_[f, i]
                Pj = (Pw - pos_[f, j]) @ R_[f, j]
                dist, _ = mj["tree"].query(Pj, distance_upper_bound=0.05)
                dmin_surf[f] = np.min(dist) * 100 if np.isfinite(dist).any() else np.nan
                cand = np.where(dist < a.cand_cm / 100)[0]
                # also points far inside: check a coarse subset of points within the other mesh's bbox
                lo, hi = mj["V"].min(0), mj["V"].max(0)
                inb = np.where(np.all((Pj > lo) & (Pj < hi), axis=1))[0]
                inb = np.setdiff1d(inb, cand)[::4]
                cc = np.concatenate([cand, inb])
                if len(cc) == 0: continue
                w = winding(Pj[cc], mj["tri"])
                ins = cc[w > 0.5]
                n_in[f] = len(ins)
                if len(ins):
                    dd, _ = mj["tree"].query(Pj[ins])
                    dmax[f] = dd.max() * 100
            key = f"{mi['name']}_in_{mj['name']}"
            fr = np.where(n_in > 0)[0]
            res[key] = dict(frames_with_penetration=int(len(fr)), first=int(fr[0]) if len(fr) else None,
                            last=int(fr[-1]) if len(fr) else None,
                            max_depth_cm=float(dmax.max()), frame_of_max=int(dmax.argmax()),
                            frames_depth_over_0p5cm=int((dmax > 0.5).sum()), frames_depth_over_1cm=int((dmax > 1.0).sum()),
                            max_points_inside=int(n_in.max()), samples=int(len(mi["S"])),
                            per_frame_depth_cm=[round(float(x), 3) for x in dmax],
                            per_frame_points_inside=[int(x) for x in n_in],
                            per_frame_min_surface_dist_cm=[round(float(x), 3) if np.isfinite(x) else None for x in dmin_surf])
            segs = []
            if len(fr):
                s0 = fr[0]; p = fr[0]
                for x in fr[1:]:
                    if x != p + 1: segs.append([int(s0), int(p)]); s0 = x
                    p = x
                segs.append([int(s0), int(p)])
            res[key]["segments"] = segs
            print(f"[{tag}] {key}: frames {len(fr)} segs {segs[:12]} max depth {dmax.max():.2f} cm @f{dmax.argmax()} (>0.5cm {int((dmax>0.5).sum())} fr, >1cm {int((dmax>1).sum())} fr) max pts {n_in.max()}/{len(mi['S'])}", flush=True)
    return res

out = dict(scene=a.scene, parquet=P, meshes=[{k: m[k] for k in ("name", "path", "vol_cm3", "boundary_edges", "nonmanifold_edges", "extents_cm")} for m in meshes])
R = poses_from(pos, wxyz)
out["reference"] = run(pos, R, "reference")
if a.rollout:
    rt = pq.read_table(a.rollout)
    cols = rt.column_names
    print("rollout columns (first 60):", cols[:60], flush=True)
    out["rollout_columns"] = cols
json.dump(out, open(a.out, "w"))
print("wrote", a.out)
