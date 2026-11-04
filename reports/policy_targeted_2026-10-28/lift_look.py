"""Per frame window: object z (rel. frame 0), z velocity, nearest wrist distance per hand, and bundle hand validity.
usage: lift_look.py SCENE OBJ_SUBSTR F0 F1"""
import sys, glob, os
import numpy as np, pyarrow.parquet as pq
scene, sub, f0, f1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
f = sorted(glob.glob(os.path.join(scene, "assets/human_motion_data/ego_recon/processed/*/robot_name=sharpa_wave/*.parquet")))[0]
t = pq.read_table(f); names = t.column_names
A = lambda n: np.array(t.column(n).combine_chunks().to_pylist()[0], dtype=float)
bodies = t.column("object_body_names").combine_chunks().to_pylist()[0]
j = next(i for i, b in enumerate(bodies) if sub in b)
pk = [n for n in names if n in ("object_body_position", "object_positions", "object_body_positions", "object_position")][0]
P = A(pk)
if P.shape[0] == len(bodies) and P.shape[1] != len(bodies): P = P.transpose(1, 0, 2)
wk = {s: [n for n in names if s in n and "wrist" in n and "pos" in n] for s in ("left", "right")}
print("wrist cols", wk)
W = {s: A(k[0]) for s, k in wk.items() if k}
bd = os.path.join(scene, "bundle_cleaned", "result.npz")
V = {}
if os.path.exists(bd):
    d = np.load(bd); V = {s: d[f"hand_{s}_is_valid"] for s in ("left", "right") if f"hand_{s}_is_valid" in d}
z = P[:, j, 2] - P[0, j, 2]
for i in range(f0, f1 + 1):
    dist = {s: np.linalg.norm(W[s][i].reshape(-1, 3)[0] - P[i, j]) for s in W}
    print(i, f"z={z[i]*100:6.1f}cm dz={(z[i]-z[i-1])*100:+5.2f}", " ".join(f"{s[0]}:{dist[s]*100:5.1f}cm" for s in dist),
          " ".join(f"{s[0]}v={int(V[s][i])}" for s in V))
