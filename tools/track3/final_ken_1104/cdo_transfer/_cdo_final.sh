set -u
C=$HOME/v2d_ops/kenc/cdo; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
mkdir -p $C/final
$PY - <<'PY'
import pandas as pd, numpy as np, os, hashlib
C = os.path.expanduser("~/v2d_ops/kenc/cdo")
w = pd.read_parquet(f"{C}/w33/57979229.raw"); f = pd.read_parquet(f"{C}/m10/full.cd-o.parquet")
wi = w.set_index("row_id"); fi = f.set_index("row_id")
def arr(df, ep, name, a):
    sub = df[df.index.str.startswith(f"t3m_{ep:06d}_{name}_{a}_")]
    idx = sub.index.str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    return sub[["x","y","z"]].to_numpy()[np.argsort(idx)]
for a in (0, 1):
    e8, e10 = arr(fi, 8, "beige_cup", a), arr(fi, 10, "beige_cup", a)
    print(f"ours ep8 vs ep10 beige array {a}: {e8.shape} {e10.shape} max abs diff {np.abs(e8-e10).max():.3g}")
    print(f"  w33 ep8 vs ours ep8 array {a}: {np.abs(arr(wi,8,'beige_cup',a)-e8).max():.3g}; w33 ep10 (old bob1 beige) vs ours ep10: {np.abs(arr(wi,10,'beige_cup',a)-e10).max():.3g}")
# final = w33 with only the ep10 beige rows taken from ours; keep w33's single commit URL
m = w["row_id"].str.startswith("t3m_000010_beige_cup_")
out = w.copy()
src = fi.loc[out.loc[m, "row_id"], ["x","y","z"]].to_numpy()
out.loc[m, ["x","y","z"]] = src
print("rows replaced:", int(m.sum()), "| commit urls in final:", out.code_commit_url.unique().tolist())
chg = (np.abs(out[["x","y","z"]].to_numpy() - w[["x","y","z"]].to_numpy()).max(axis=1) > 0)
print("rows changed vs w33:", int(chg.sum()), "all in ep10 beige:", bool(out.loc[chg, "row_id"].str.startswith("t3m_000010_beige_cup_").all()))
p = f"{C}/final/cdo_w33_ep10beigeT6.parquet"; out.to_parquet(p, index=False)
print("wrote", p, os.path.getsize(p), hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
PY
cd $K && $PY submit/v2d_submit.py $C/final/cdo_w33_ep10beigeT6.parquet --track track_3_mesh --dry-run -m "CD-O test: team best (w33) + ep10 beige_cup <- T6's ep8 beige_cup mesh" 2>&1 | tail -8
