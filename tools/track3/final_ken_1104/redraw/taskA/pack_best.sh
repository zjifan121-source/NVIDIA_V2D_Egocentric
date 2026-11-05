# pack_best.sh EP DRAW : pack one re-evaluation draw of the bob1 checkpoint exactly like earlier candidates
# (pack_one.sh, bob1 commit URL as in swap_submit.sh) into ~/v2d_ops/kenc/push1104/<ep>_best/one.parquet; compare to bob1 rows
set -u
EP=$1 D=$2
W=/data/user_data/haisenk/v2d/push1104/taskA
URL=https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6
RES=$W/draws/ep$EP/$D/results
if [ "$EP" = 20 ]; then SC=$W/hf/experiments/edmond/bob1_submission_20261103/episode_000020/scene; else SC=$W/scenes/ep$EP; fi
MESH=$SC/score_input/meshes
T=$HOME/v2d_ops/kenc/push1104/${EP}_best; mkdir -p $T
[ -e $T/one.parquet ] && { echo "exists $T/one.parquet - not overwriting"; exit 1; }
ls $MESH $MESH/* | head -5
bash $HOME/v2d_ops/pack_one.sh $EP $RES $MESH $URL $T/one.parquet > $T/pack.log 2>&1; grep -a "roster\|wrote\|rows\|Error\|error" $T/pack.log | head -6
echo "ep$EP draw $D results $RES meshes $MESH url $URL" > $T/SOURCE.txt
/home/haisenk/v2d_ops/kaggle_env/bin/python - $T/one.parquet $EP <<'PY'
import sys, pandas as pd
one = pd.read_parquet(sys.argv[1]); ep = int(sys.argv[2])
b = pd.read_parquet("/home/haisenk/v2d_ops/kenc/bob1/candidate_bob1.parquet"); b = b[b.row_id.str.startswith(f"track_3/e{ep:06d}/")]
print("rows one", len(one), "bob1", len(b), "row_ids equal", sorted(one.row_id) == sorted(b.row_id))
m = one.merge(b, on="row_id", suffixes=("", "_b"))
print("  same-as-bob1 fraction:", {c: round(float((m[c].astype(str) == m[c + "_b"].astype(str)).mean()), 3) for c in one.columns if c != "row_id"})
PY
