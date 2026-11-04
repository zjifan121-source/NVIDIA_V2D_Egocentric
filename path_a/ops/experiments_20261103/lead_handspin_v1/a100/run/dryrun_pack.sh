set -u
# End-to-end packing rehearsal: bob1 with ep33 swapped to OUR smoke checkpoint. Local only, no Kaggle upload.
M=/root/v2d/NVIDIA_V2D_Egocentric; T=/root/edmond_handoff_de66fea/tools/track3; TL=/root/v2d/tools_lead
SC=$M/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes
W=/root/probes/rehearsal_ep33; mkdir -p $W
export V2D_ROOT=/root/v2d LEAD_ROSTER=$T/episodes.json
echo "== export score_input"
/root/venv-cpu/bin/python $TL/export_score_input.py $SC/ep33_rawexcl $SC/ep25_rawexcl $SC/ep30_rawexcl /root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl 2>&1 | cut -c1-230
echo "== roster vs parquet body order"
/root/venv-cpu/bin/python - <<PY
import json,glob,pyarrow.parquet as pq
K="$M/assets/data/v2d_submission_kit/data/track_3_evaluation_objects.json"
d=json.load(open(K)); ro=d.get('episodes',d)
print("rollout:",pq.read_table("/root/runs/ep33_def_s200/early_10002432/results/episode_000033.parquet",columns=['object_id','object_body_name']).to_pandas().drop_duplicates().values.tolist())
for ep,s in [(33,"$SC/ep33_rawexcl"),(25,"$SC/ep25_rawexcl"),(30,"$SC/ep30_rawexcl"),(28,"/root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl")]:
    f=glob.glob(s+"/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet")[0]
    print(ep,"roster",ro[str(ep)],"scene bodies",pq.read_table(f,columns=['object_names']).column(0)[0].as_py())
PY
echo "== build swap plan"
/root/venv-cpu/bin/python - <<PY
import json
p=json.load(open("$TL/bob1_local.plan.json"))
r=json.load(open("/root/runs/ep33_def_s200/early_10002432/results/episode_000033.json"))
p["33"]=dict(run="/root/runs/ep33_def_s200/early_10002432",scene="$SC/ep33_rawexcl",kind="policy",tracking=r["metrics"]["add_auc"],name="REHEARSAL_ep33_def_s200/early_10002432",source="wentao")
json.dump(p,open("$W/plan.json","w"),indent=1)
PY
echo "== check plan"
/root/venv-cpu/bin/python $TL/check_probe_plan.py $W/plan.json --ref $TL/bob1_local.plan.json --expect-changes 1 --roster $M/assets/data/v2d_submission_kit/data/track_3_evaluation_objects.json --out $W/plan.check.json 2>&1 | tail -6 | cut -c1-260
echo "exit=$?"
echo "== pack"
cp -n $T/pack_submission.py $M/tools/track3/pack_submission.py 2>/dev/null
cd $M
time /root/venv-cpu/bin/python tools/track3/pack_submission.py --plan $W/plan.json --work $W/work --code-commit https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/de66fea3 --python /root/venv-cpu/bin/python -o $W/rehearsal.parquet 2>&1 | tail -25 | cut -c1-240
ls -la $W/rehearsal.parquet 2>&1 | cut -c20-
