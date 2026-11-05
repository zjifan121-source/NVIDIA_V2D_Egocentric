W=/data/user_data/haisenk/v2d/push1104/taskA; S=~/v2d_ops/kenc/push1104/taskA
EPS="14 15 16 19 28 30 32 33 35 36 37 38"
bash $S/gen_jobs.sh $W/jobs_a.txt 1 12 $EPS
bash $S/gen_jobs.sh $W/jobs_b.txt 13 24 $EPS
ls -la $W/pol/*/
