set -u
C=$HOME/v2d_ops/kenc/cdo; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
$PY - <<'PY'
import os, numpy as np, pandas as pd
C = os.path.expanduser("~/v2d_ops/kenc/cdo")
D = {"w33": pd.read_parquet(f"{C}/w33/57979229.raw").set_index("row_id"), "bob1": pd.read_parquet(f"{C}/ref/bob1.cd-o.parquet").set_index("row_id"),
     "t3": pd.read_parquet(f"{C}/ref/t3.cd-o.parquet").set_index("row_id")}
def arr(df, ep, name, a):
    sub = df[df.index.str.startswith(f"t3m_{ep:06d}_{name}_{a}_")]
    idx = sub.index.str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    return sub[["x","y","z"]].to_numpy()[np.argsort(idx)]
def same(a, b, name):
    return all(np.array_equal(arr(D[a[0]], a[1], name, k), arr(D[b[0]], b[1], name, k)) for k in (0, 1))
print("ep1 lid w33 == bob1:", same(("w33",1),("bob1",1),"white_pot_lid"))
print("ep33 blue w33 == bob1:", same(("w33",33),("bob1",33),"blue_cup"))
print("ep5 spoon t3 == w33:", same(("t3",5),("w33",5),"wooden_spoon"), "| ep3 spoon w33 == bob1:", same(("w33",3),("bob1",3),"wooden_spoon"), "| ep6 spoon w33 == bob1:", same(("w33",6),("bob1",6),"wooden_spoon"))
# pot dimensions along the principal axes (cm): sorted extents of the vertex cloud after PCA
for tag, (k, ep) in {"P0 donor (w33 ep3)": ("w33",3), "P1 ep1 (w33)": ("w33",1), "P2 ep5 T3": ("t3",5), "P2' ep5 T8 (w33)": ("w33",5)}.items():
    V = arr(D[k], ep, "white_pot", 0); X = V - V.mean(0)
    w, U = np.linalg.eigh(X.T @ X); P = X @ U
    print(f"{tag:22s} extents along PCA axes (cm): {np.round(100*(P.max(0)-P.min(0)),2)}")
PY
