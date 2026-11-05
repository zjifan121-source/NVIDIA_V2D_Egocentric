W=/data/user_data/haisenk/v2d/push1104/taskA; H=$W/hf/experiments/edmond/bob1_submission_20261103
P20=/data/user_data/haisenk/v2d/hand5/scenes/h5_ep20_objrel_a1/assets/human_motion_data/ego_recon/processed/sequence_id=ep20_h5_h5_ep20_objrel_a1/robot_name=sharpa_wave/aee98d507bfc445ab8454429be2e9e3d-0.parquet
ls -la $P20 || exit 1
FC=/data/user_data/haisenk/v2d/v2d_train/baseline/robotic_grounding/flash_chord; PY=$FC/.venv/bin/python
$PY - $P20 <<'PY'
import sys, os, pyarrow.parquet as pq
t = pq.read_table(sys.argv[1]); bad = []
for c in t.column_names:
    v = t.column(c)[0].as_py(); vv = v if isinstance(v, list) else [v]
    bad += [x for x in vv if isinstance(x, str) and x.startswith("/") and not os.path.exists(x)]
print("ep20 original scene missing:", bad)
PY
POL20=$H/episode_000020/policy/policy_100003840.safetensors
POL25=$W/pol/ep25/policy_140005376.safetensors
sha256sum $POL20 | cut -c1-16
: > $W/jobs_c.txt; : > $W/jobs_d.txt
for d in $(seq 1 10); do printf "20 %s d%02d\n25 %s d%02d\n" $POL20 $d $POL25 $d >> $W/jobs_c.txt; done
for d in $(seq 11 20); do printf "20 %s d%02d\n25 %s d%02d\n" $POL20 $d $POL25 $d >> $W/jobs_d.txt; done
wc -l $W/jobs_c.txt $W/jobs_d.txt
