set -u
M=/root/v2d/NVIDIA_V2D_Egocentric; T=/root/edmond_handoff_de66fea/tools/track3; TL=/root/v2d/tools_lead
SC=$M/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes
W=/root/probes/rehearsal_ep33
echo "== export results (head of first log)"; grep -n '"status"\|"failed"\|^{' /root/dryrun_pack.log | head -8 | cut -c1-200
echo "== roster vs body order"
/root/venv-cpu/bin/python - <<PY
import json,glob,pyarrow.parquet as pq
K="$M/assets/data/v2d_submission_kit/data/track_3_evaluation_objects.json"
d=json.load(open(K)); ro=d.get('episodes',d)
t=pq.read_table("/root/runs/ep33_def_s200/early_10002432/results/episode_000033.parquet",columns=['object_id','object_body_name']).to_pandas().drop_duplicates()
print("ep33 rollout:",t.values.tolist())
for ep,s in [(33,"$SC/ep33_rawexcl"),(25,"$SC/ep25_rawexcl"),(30,"$SC/ep30_rawexcl"),(28,"/root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl")]:
    f=glob.glob(s+"/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet")[0]
    b=pq.read_table(f,columns=['object_body_names']).column(0)[0].as_py()
    r=ro[str(ep)]; ok=all(x.endswith('_'+y) for x,y in zip(b,r)) and len(b)==len(r)
    print(ep,"ORDER_OK" if ok else "ORDER_MISMATCH","roster",r,"bodies",b)
PY
echo "== handoff vs our packer"; diff <(sed -n 1,200p $M/tools/track3/pack_submission.py) <(sed -n 1,200p $T/pack_submission.py) | cut -c1-150 | head -14
cp $T/pack_submission.py $M/tools/track3/pack_submission_handoff.py
echo "== pack"
cd $M
rm -rf $W/work
time /root/venv-cpu/bin/python tools/track3/pack_submission_handoff.py --plan $W/plan.json --work $W/work --code-commit https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/de66fea3 --python /root/venv-cpu/bin/python -o $W/rehearsal.parquet 2>&1 | tail -25 | cut -c1-240
ls -la $W/rehearsal.parquet 2>&1 | cut -c20-
echo REHEARSAL_END
