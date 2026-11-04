"""anch_phase.py MESHDIR ANCHOR NSEG ROLLOUT_A ROLLOUT_B: anchored per-object AUC in NSEG equal frame segments, A vs B (B - A)."""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, r"C:\v2d_demo\diag\ep5_skepA1")
from an_common import load_obj, load_rollout, xf, auc_cm
import pyarrow.parquet as pq
def adds(roll, mdir, anchor):
    t = pq.read_table(roll, columns=["object_body_name", "reference_frame", "environment_id"])
    env = np.array(t.column("environment_id").to_pylist()); sel = env == env.min()
    bn = np.array([str(x) for x in t.column("object_body_name").to_pylist()])[sel]
    N = int(np.array(t.column("reference_frame").to_pylist())[sel].max()) + 1
    bodies = list(dict.fromkeys(bn)); R = load_rollout(roll, bodies, N)
    meshes = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(mdir, "*.obj"))}
    names = [sorted([k for k in meshes if k in b], key=len)[-1] for b in bodies]
    ai = names.index(anchor); G = R[ai]["ref"][0] @ np.linalg.inv(R[ai]["ach"][0])
    out = {}
    for k, n in enumerate(names):
        V = load_obj(meshes[n])[0]; a = np.full(N, np.nan)
        for f in range(N):
            Ta, Tr = R[k]["ach"][f], R[k]["ref"][f]
            if np.isfinite(Ta).all() and np.isfinite(Tr).all():
                a[f] = 100 * np.linalg.norm(xf(G @ Ta, V) - xf(Tr, V), axis=1).mean()
        out[n] = a
    return out, N
mdirA = sys.argv[1]; anchor = sys.argv[2]; nseg = int(sys.argv[3]); ra, rb = sys.argv[4], sys.argv[5]
mdirB = sys.argv[6] if len(sys.argv) > 6 else mdirA
A, N = adds(ra, mdirA, anchor); B, _ = adds(rb, mdirB, anchor)
edges = np.linspace(0, N, nseg + 1).astype(int)
for n in A:
    row = []
    for s in range(nseg):
        a = auc_cm(A[n][edges[s]:edges[s+1]]); b = auc_cm(B[n][edges[s]:edges[s+1]])
        row.append(f"f{edges[s]}-{edges[s+1]-1}: {a:.3f}->{b:.3f} ({b-a:+.3f})")
    print(f"{n:14s} mean ADD {np.nanmean(A[n]):.2f}->{np.nanmean(B[n]):.2f} cm | " + " | ".join(row))
