#!/bin/bash
#SBATCH -p maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 0:40:00
cd /home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28
df -h /data/user_data/edmondwa | tail -1
for t in $(python3 make_recipe_trials.py WIDE 1,3,6,8,14,15,16,20,28,30,32,33,35,38 132,133); do
  echo "$t: $(python3 submit_trial.py ${t,,}_trial.json 2>&1 | tail -1 | cut -c1-200)"
done
for rs in "ep36_RWIDE_s130 90005760" "ep25_RWIDE_s130 95006976"; do
  set -- $rs
  ls /data/user_data/edmondwa/v2d/flash_runs/diag/$1/train/policy_$2.safetensors >/dev/null || { echo "no policy for $1 $2"; ls /data/user_data/edmondwa/v2d/flash_runs/diag/$1/train | grep policy | tail -3; continue; }
  tf=$(python3 whc_cont.py $1 $2 | tail -1); echo "$1@$2 -> $tf: $(python3 submit_trial.py $tf 2>&1 | tail -1 | cut -c1-200)"
done
# disk keeper: prune finished runs every 90 min for 12 h
for k in $(seq 1 8); do
  sbatch --parsable -J prune$k -p maxlab-cpu --qos=maxlab_cpu_qos -c 1 --mem=2G -t 0:30:00 --begin=now+$((k*90))minutes -o prune-%j.out \
    --wrap "cd $PWD && squeue -u edmondwa,sarveshp -h -o %j > .queued && python3 prune_keep.py .queued | tail -2; df -h /data/user_data/edmondwa | tail -1"
done
