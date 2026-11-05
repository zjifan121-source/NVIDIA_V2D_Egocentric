set -u
cd ~/v2d_ops/kenc/push1104/taskA
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
while read -r EP STEP R; do echo "== ep$EP"; bash setup_ep.sh $EP $STEP 2>&1 | tail -3; done < eps.txt
W=/data/user_data/haisenk/v2d/push1104/taskA
ls $W/pol/*/ | head -40
echo "--- TEST ep14 on gpu0"
echo "14 $W/pol/ep14/policy_75005952.safetensors t00" > $W/test_jobs.txt
PER=1 timeout 900 bash drv.sh $W/test_jobs.txt test0 0
tail -25 $W/draws/ep14/t00/eval.log
python3 -c "
import json; d=json.load(open('$W/draws/ep14/t00/results/episode_000014.json')); print(list(d.keys())); m=d['metrics']; print({k:v for k,v in m.items() if not isinstance(v,(list,dict))}); print({k:v for k,v in m.items() if isinstance(v,(list,dict))})" 2>&1 | head -30
