set -u
C=$HOME/v2d_ops/kenc/cdo; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
$PY - <<'PY'
import pandas as pd, numpy as np, os
C = os.path.expanduser("~/v2d_ops/kenc/cdo")
w = pd.read_parquet(f"{C}/w33/57979229.raw"); f = pd.read_parquet(f"{C}/m10/full.cd-o.parquet")
print("w33 cols", w.columns.tolist(), len(w), "| ours", len(f))
print("w33 commit urls:", w.code_commit_url.unique()[:3]); print("our commit urls:", f.code_commit_url.unique()[:3])
w = w.set_index("row_id"); f = f.set_index("row_id")
print("same row_id set:", set(w.index) == set(f.index))
f = f.loc[w.index]
diff = (np.abs(w[["x","y","z"]].to_numpy() - f[["x","y","z"]].to_numpy()).max(axis=1) > 0)
key = pd.Series(w.index[diff]).str.extract(r"^t3m_(\d+)_(.+)_([01])_\d+$")
print("rows differing:", int(diff.sum())); print(key.groupby([0,1,2]).size().to_string())
def arr(df, ep, name, a):
    sub = df[df.index.str.startswith(f"t3m_{ep}_{name}_{a}_")]
    idx = sub.index.str.rsplit("_", n=1).str[1].astype(int)
    return sub[["x","y","z"]].to_numpy()[np.argsort(idx)]
for a in (0, 1):
    e8, e10 = arr(f, 8, "beige_cup", a), arr(f, 10, "beige_cup", a)
    print(f"ours: ep8 vs ep10 beige array {a}: shapes {e8.shape} {e10.shape}, max abs diff {np.abs(e8-e10).max() if e8.shape==e10.shape else 'shape mismatch'}")
    w10 = arr(w, 10, "beige_cup", a); w8 = arr(w, 8, "beige_cup", a)
    print(f"w33:  ep8 vs ours ep8 array {a}: {np.abs(w8-e8).max()}; w33 ep10 vs bob1-shape? shape {w10.shape}")
PY
