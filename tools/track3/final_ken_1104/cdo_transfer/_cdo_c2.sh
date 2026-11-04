set -u
C=$HOME/v2d_ops/kenc/cdo; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
F=$C/final/cdo_c2_pots.parquet
$PY - <<'PY'
import os, numpy as np, pandas as pd, hashlib
C = os.path.expanduser("~/v2d_ops/kenc/cdo")
base = pd.read_parquet(f"{C}/final/cdo_w33_ep10beigeT6.parquet")   # C1 file, scored 0.38183
out = base.copy(); bi = base.set_index("row_id")
def rows(ep, name, a): return out["row_id"].str.startswith(f"t3m_{ep:06d}_{name}_{a}_")
def arr(ep, name, a):
    sub = bi[bi.index.str.startswith(f"t3m_{ep:06d}_{name}_{a}_")]
    idx = sub.index.str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    return sub[["x","y","z"]].to_numpy()[np.argsort(idx)]
for ep in (1, 3, 6, 8):
    for a in (0, 1):
        m = rows(ep, "white_pot", a)
        idx = out.loc[m, "row_id"].str.rsplit("_", n=1).str[1].astype(int).to_numpy()
        out.loc[m, ["x","y","z"]] = arr(5, "white_pot", a)[idx]
chg = (np.abs(out[["x","y","z"]].to_numpy() - base[["x","y","z"]].to_numpy()).max(axis=1) > 0)
ch = out.loc[chg, "row_id"].str.extract(r"^t3m_(\d+)_(.+)_([01])_\d+$").groupby([0,1,2]).size()
print("rows changed vs C1:", int(chg.sum())); print(ch.to_string())
for ep in (1, 3, 6, 8):
    for a in (0, 1):
        sub = out[rows(ep, "white_pot", a)].copy(); sub["i"] = sub.row_id.str.rsplit("_", n=1).str[1].astype(int)
        assert np.array_equal(sub.sort_values("i")[["x","y","z"]].to_numpy(), arr(5, "white_pot", a)), (ep, a)
print("ep1/3/6/8 white_pot arrays == ep5 (T8) white_pot arrays: OK; commit urls:", out.code_commit_url.unique().tolist())
p = f"{C}/final/cdo_c2_pots.parquet"; out.to_parquet(p, index=False)
print("wrote", p, hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
PY
MSG="Ken CD-O test C2: C1 (0.38183) + white_pot in ep1/3/6/8 <- ep5 T8 white_pot mesh (same object; T8 trim measured -0.047 cm)"
cd $K && $PY submit/v2d_submit.py $F --track track_3_mesh --dry-run -m "$MSG" > $C/final/dry_c2.log 2>&1; tail -3 $C/final/dry_c2.log
LEFT=$(grep -o "[0-9]* left today" $C/final/dry_c2.log | awk '{print $1}' | sort -n | head -1)
if [ -z "$LEFT" ] || [ "$LEFT" -le 3 ]; then echo "COUNTER ${LEFT:-unknown} <= 3: NOT SUBMITTING"; exit 3; fi
grep -qi "error\|invalid\|fail" $C/final/dry_c2.log && { echo "DRY-RUN PROBLEM - not submitting"; exit 1; }
$PY submit/v2d_submit.py $F --track track_3_mesh -m "$MSG" > $C/final/submit_c2.log 2>&1; tail -4 $C/final/submit_c2.log
R=$(ls -t $C/final/cdo_c2_pots.parquet.*.kaggle-receipt.json | head -1)
for i in $(seq 1 30); do out=$($PY submit/v2d_submit.py --track track_3_mesh --poll-receipt $R 2>&1 | tail -3); echo "$out" | grep -q "COMPLETE\|ERROR\|FAILED" && { echo "$out"; exit 0; }; sleep 30; done; echo "pending: $out"
