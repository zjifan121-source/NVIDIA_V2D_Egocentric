W=/data/user_data/haisenk/v2d/push1104/taskA; S=~/v2d_ops/kenc/push1104/taskA
bash $S/gen_jobs2.sh $W/jobs_j.txt 41 56 35
bash $S/gen_jobs2.sh $W/jobs_k.txt 57 72 35
python3 -c "
import json; r=json.load(open('$W/agg3.json'))['35']['main']; print('ep35 main values', r['values'])"
