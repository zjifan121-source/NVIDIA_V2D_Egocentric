W=/data/user_data/haisenk/v2d/push1104/taskA; S=~/v2d_ops/kenc/push1104/taskA
EPS="14 15 16 19 20 25 28 30 32 33 35 36 37 38"
bash $S/gen_jobs2.sh $W/jobs_f.txt 25 32 $EPS
bash $S/gen_jobs2.sh $W/jobs_g.txt 33 40 $EPS
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
