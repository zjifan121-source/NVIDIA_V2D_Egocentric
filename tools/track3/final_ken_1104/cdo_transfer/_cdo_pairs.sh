set -u
C=$HOME/v2d_ops/kenc/cdo; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
# derive CD-O rows of bob1 and of T3 (ep5 before the pot trim) for comparison (dry-run only, nothing submitted)
mkdir -p $C/ref; cp -n $HOME/v2d_ops/kenc/bob1/candidate_bob1.parquet $C/ref/bob1.parquet; cp -n $HOME/v2d_ops/kenc/tests/o_ep5_attcom_s42/track3_submission.parquet $C/ref/t3.parquet
cd $K
for f in bob1 t3; do [ -e $C/ref/$f.cd-o.parquet ] || $PY submit/v2d_submit.py $C/ref/$f.parquet --track track_3 --only cd_o_cm --dry-run -m x 2>&1 | grep -i "extracted\|error"; done
$PY - <<'PY'
import os, importlib.util, numpy as np, pandas as pd, itertools
C = os.path.expanduser("~/v2d_ops/kenc/cdo")
spec = importlib.util.spec_from_file_location("cdo", "/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit/metric_code/track_3/CD-O.py")
cdo = importlib.util.module_from_spec(spec); spec.loader.exec_module(cdo)
files = {"w33": f"{C}/w33/57979229.raw", "bob1": f"{C}/ref/bob1.cd-o.parquet", "t3": f"{C}/ref/t3.cd-o.parquet"}
D = {k: pd.read_parquet(v).set_index("row_id") for k, v in files.items()}
def mesh(df, ep, name):
    out = []
    for a in (0, 1):
        sub = df[df.index.str.startswith(f"t3m_{ep:06d}_{name}_{a}_")]
        idx = sub.index.str.rsplit("_", n=1).str[1].astype(int).to_numpy()
        out.append(sub[["x","y","z"]].to_numpy()[np.argsort(idx)])
    V, F = out; return V, np.rint(F).astype(np.int64)
def cd(A, B): return np.mean([cdo._object_chamfer_cm(A, B, s) for s in (5, 6)])
pots = [("w33",1),("w33",3),("w33",5),("w33",6),("w33",8),("w33",28),("bob1",1),("bob1",5),("t3",5)]
M = {p: mesh(D[p[0]], p[1], "white_pot") for p in pots}
print("white_pot pairwise CD (cm), budgeted meshes as scored:")
for a, b in itertools.combinations(pots, 2):
    d = cd(M[a], M[b])
    if d < 0.6 or a[0] != b[0]: print(f"  {a[0]}:ep{a[1]} vs {b[0]}:ep{b[1]}: {d:.3f}")
sp = [("w33",e) for e in (3,5,6,15,16,32,33,35,36,37)] + [("bob1",5),("bob1",6),("bob1",33),("t3",5)]
S = {p: mesh(D[p[0]], p[1], "wooden_spoon") for p in sp}
print("wooden_spoon: identical-array groups and CD to w33 ep5 spoon")
for p in sp:
    print(f"  {p[0]}:ep{p[1]}  CD vs w33 ep5 = {cd(S[p], S[('w33',5)]):.3f}  vs bob1 ep5 = {cd(S[p], S[('bob1',5)]):.3f}")
PY
