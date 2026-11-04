"""Spin-invariant own-ref add_auc (hand5 ep10). The official Track-3 AUC ignores axial spin of beige_cup/blue_cup
(kit tracks/track_3/README "Registration and symmetry"; metric_code/track_3/AUC.py _symmetry_point_errors, 360
axial rotations), while flash_chord's own-ref add_auc (evaluation/metrics.py object_add) scores full orientation.
This recomputes own-ref from the evaluator's object-trajectory parquet both ways:
  plain    : same formula as flash_chord (should match results json add_auc up to vertex sampling)
  sym      : per frame min over 360 rotations about the cup's mesh symmetry axis (through its surface centroid)
usage: python sym_ownref.py <results/episode_000010.parquet> <scene processed dir (holds <seq>_<object>.obj)> [...more pairs]
GT-free: uses only the rollout, our own reference and our own meshes."""
import sys, glob, json
import numpy as np, trimesh
from scipy.spatial.transform import Rotation

THR = np.arange(0.01, 0.10, 0.01)
SYM = ("blue_cup", "beige_cup")


def auc(add):  # add (T, W)
    axis = np.linspace(0.0, 1.0, THR.size)
    return float(np.trapezoid(np.array([np.mean(add < t) for t in THR]), x=axis))


def axis_of(v, f):
    pts, _ = trimesh.sample.sample_surface(trimesh.Trimesh(v, f, process=False), 20000, seed=0)
    c = pts.mean(0); w, U = np.linalg.eigh(np.cov((pts - c).T))
    sep = [min(abs(w[j] - w[(j + 1) % 3]), abs(w[j] - w[(j + 2) % 3])) for j in range(3)]
    return U[:, int(np.argmax(sep))], c


def body_add(A, Rf, m, name):
    """A, Rf: (T, W, 7) achieved / reference xyz + xyzw; m: trimesh body mesh. Returns plain and sym ADD (T, W)."""
    steps, worlds = A.shape[:2]
    v, _ = trimesh.sample.sample_surface(m, 500, seed=0)
    Ra = Rotation.from_quat(A[..., 3:].reshape(-1, 4)).as_matrix().reshape(steps, worlds, 3, 3)
    Rr = Rotation.from_quat(Rf[..., 3:].reshape(-1, 4)).as_matrix().reshape(steps, worlds, 3, 3)
    pa = np.einsum("twij,vj->twvi", Ra, v) + A[..., None, :3]
    pr = np.einsum("twij,vj->twvi", Rr, v) + Rf[..., None, :3]
    plain = np.linalg.norm(pa - pr, axis=-1).mean(-1)
    sym = plain.copy()
    if name in SYM:
        ax, c = axis_of(np.asarray(m.vertices), np.asarray(m.faces))
        for ang in np.radians(np.arange(360)):
            S = Rotation.from_rotvec(ax * ang).as_matrix()
            pr_s = np.einsum("twij,vj->twvi", Rr, (v - c) @ S.T + c) + Rf[..., None, :3]
            sym = np.minimum(sym, np.linalg.norm(pa - pr_s, axis=-1).mean(-1))
    return plain, sym


def score(pq_path, proc_dir):
    import pyarrow.parquet as pq
    t = pq.read_table(pq_path).to_pandas()
    t = t[t.environment_valid]
    out = {}
    plain_all, sym_all = [], []
    for body, g in t.groupby("object_body_name", observed=True):
        body = str(body)
        g = g.sort_values(["step", "environment_id"])
        steps, worlds = g.step.nunique(), g.environment_id.nunique()
        A = g[[f"achieved_{k}" for k in ("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")]].to_numpy(np.float64).reshape(steps, worlds, 7)
        Rf = g[[f"reference_{k}" for k in ("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")]].to_numpy(np.float64).reshape(steps, worlds, 7)
        name = next(n for n in SYM + ("",) if n and n in body) if any(n in body for n in SYM) else body
        objs = glob.glob(f"{proc_dir}/*{name}.obj")
        m = trimesh.load(objs[0], force="mesh", process=False)
        plain, sym = body_add(A, Rf, m, name)
        out[name] = {"plain_auc": round(auc(plain), 4), "sym_auc": round(auc(sym), 4),
                     "plain_mean_add_cm": round(float(plain.mean() * 100), 2), "sym_mean_add_cm": round(float(sym.mean() * 100), 2)}
        plain_all.append(auc(plain)); sym_all.append(auc(sym))
    out["episode"] = {"plain_add_auc": round(float(np.mean(plain_all)), 4), "sym_add_auc": round(float(np.mean(sym_all)), 4)}
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    for p, d in zip(a[::2], a[1::2]):
        print(p, json.dumps(score(p, d)))
