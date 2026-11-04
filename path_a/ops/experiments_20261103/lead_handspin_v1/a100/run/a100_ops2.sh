set -u
M=/root/v2d/NVIDIA_V2D_Egocentric
SC=$M/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes
R=$M/reports/policy_targeted_2026-10-28
# every checkpoint from 70M (Edmond's guide), not the handoff script's fixed target list
sed -e 's/^    targets=\[.*$/    targets=[]  # lead: evaluate every checkpoint from 70M (TRAINING_GUIDE) instead of fixed targets/' \
    -e 's/^    chosen={max((s,p) for s,p in policies if s<=t) for t in targets if any(s<=t for s,p in policies)}$/    chosen={(s,p) for s,p in policies if s>=70_000_000}/' \
    $R/evaluate_checkpoints.py > $R/evaluate_all70m.py
diff $R/evaluate_checkpoints.py $R/evaluate_all70m.py
/root/venv-flash-lead/bin/python -c "import ast,sys;ast.parse(open('$R/evaluate_all70m.py').read());print('EVAL_SCRIPT_OK')"
chmod +x /root/watch_run2.sh /root/queue_a100.sh
# replace the first watchers (anchored pattern: they only sleep/evaluate, training is untouched)
pkill -f '^/bin/bash /root/watch_run.sh' ; sleep 1; echo "old watchers left: $(pgrep -fc '^/bin/bash /root/watch_run.sh')"
for a in "ep33_def_s200 33" "ep25_rawexcl_def_s200 25" "ep28_v8pot_def_s200 28"; do set -- $a; setsid nohup /root/watch_run2.sh $1 $2 > /root/runs/$1/watch.log 2>&1 < /dev/null & done
# ep30 (user priority) on the fourth slot; identical data to lead_v3 ep30
bash /root/launch_scene.sh ep30_rawexcl_def_s200 $SC/ep30_rawexcl 200
setsid nohup /root/watch_run2.sh ep30_rawexcl_def_s200 30 > /root/runs/ep30_rawexcl_def_s200/watch.log 2>&1 < /dev/null &
# next arms: WIDE recipe on the priority episodes
W="action.normalized_mapping=linear training.network.initial_normalized_std=0.3 training.network.log_std_max=-0.5108256237659907 training.exploration.target_sigma=0.3"
cat > /root/queue_a100.txt <<Q
ep28_v8pot_wide_s201 /root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl 201 28 $W
ep33_wide_s201 $SC/ep33_rawexcl 201 33 $W
ep25_rawexcl_def_s201 $SC/ep25_rawexcl 201 25
ep30_rawexcl_def_s201 $SC/ep30_rawexcl 201 30
Q
echo 4 > /root/MAX_RUNS_A100
setsid nohup /root/queue_a100.sh > /root/queue_a100_daemon.log 2>&1 < /dev/null &
sleep 2
echo "watchers: $(pgrep -fc '^/bin/bash /root/watch_run2.sh') queue daemons: $(pgrep -fc '^/bin/bash /root/queue_a100.sh') trainings: $(pgrep -fc 'python scripts/train_flash_sac.py')"
find /root -maxdepth 5 -name train.log -not -path "/root/runs/*" -not -path "/root/.cache/*" -not -path "*/site-packages/*" 2>/dev/null | head
