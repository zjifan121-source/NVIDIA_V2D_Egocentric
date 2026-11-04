#!/bin/bash
#SBATCH -p maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 1:00:00
H=/home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28
D=/data/user_data/edmondwa/v2d/flash_runs/diag
df -h /data/user_data/edmondwa | tail -1
# my held sweep jobs whose run already finished: cancel the stale requeue
for id in 11020196 11020194 11020187 11020186 11020185 11020184 11020182 11020181 11020180 11020177; do
  n=$(squeue -h -j $id -o %j 2>/dev/null)
  [ -n "$n" ] && [ -f $D/$n/train.done ] && scancel $id && echo "cancelled $id $n (finished)"
done
squeue -u edmondwa,sarveshp -h -o %j > $H/.queued
python3 $H/prune_keep.py $H/.queued | tail -15
df -h /data/user_data/edmondwa | tail -1
cd /data/user_data/edmondwa && timeout 1500 du -xsh v2d/* * 2>/dev/null | sort -h | tail -12
