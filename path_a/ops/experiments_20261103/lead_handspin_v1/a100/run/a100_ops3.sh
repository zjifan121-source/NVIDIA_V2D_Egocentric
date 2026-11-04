set -u
SC=/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes
P=$(pgrep -f "logging.run_name=ep33_def_s200" | head -1)
tr "\0" " " < /proc/$P/cmdline | grep -q "scripts/train_flash_sac.py.*logging.run_name=ep33_def_s200" && kill $P && echo "STOPPED ep33_def_s200 pid=$P at $(grep 'training progress' /root/runs/ep33_def_s200/train.log | tail -1 | cut -c20-50)"
sleep 4; echo "still alive: $(pgrep -fc 'logging.run_name=ep33_def_s200')"
W="action.normalized_mapping=linear training.network.initial_normalized_std=0.3 training.network.log_std_max=-0.5108256237659907 training.exploration.target_sigma=0.3"
cat > /root/queue_a100.txt <<Q
ep25_rawexcl_wide_s201 $SC/ep25_rawexcl 201 25 $W
ep28_v8pot_wide_s201 /root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl 201 28 $W
ep30_rawexcl_wide_s201 $SC/ep30_rawexcl 201 30 $W
Q
echo "--- other queue"
for q in $(pgrep -f '^/bin/bash ./queue_v3.sh'); do echo "pid $q cwd=$(readlink /proc/$q/cwd) ppid=$(ps -o ppid= -p $q | tr -d ' ') etime=$(ps -o etime= -p $q | tr -d ' ')"; done
D=$(readlink /proc/$(pgrep -f '^/bin/bash ./queue_v3.sh' | head -1)/cwd); cd $D; pwd; ls; echo "MAX_RUNS=$(cat MAX_RUNS 2>/dev/null)"; echo "--- queue.txt"; cut -c1-170 queue.txt 2>/dev/null; echo "--- queue.log"; tail -8 queue.log 2>/dev/null | cut -c1-150
sed -n 1,20p queue_v3.sh | grep -n "MAX\|pgrep" | cut -c1-150
nvidia-smi --query-gpu=memory.used --format=csv,noheader; date -u +%T
