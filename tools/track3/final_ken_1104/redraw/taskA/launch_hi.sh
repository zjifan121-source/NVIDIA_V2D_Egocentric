W=/data/user_data/haisenk/v2d/push1104/taskA; S=~/v2d_ops/kenc/push1104/taskA
EPS="14 15 16 19 20 25 28 30 32 33 35 36 37 38"
bash $S/gen_jobs2.sh $W/jobs_h.txt 5 10 $EPS; sed -i 's/ d\([0-9][0-9]\)$/ n\1/' $W/jobs_h.txt
bash $S/gen_jobs2.sh $W/jobs_i.txt 11 16 $EPS; sed -i 's/ d\([0-9][0-9]\)$/ n\1/' $W/jobs_i.txt
head -2 $W/jobs_h.txt; tail -1 $W/jobs_i.txt
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
grep -n "eval_code\|eval_script" $K/tools/pack_submission.py | head -12
