#!/bin/bash
#SBATCH --job-name=v2d-target-diagnostic
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|6000Ada|H100|H200
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --array=0-2%2
#SBATCH --output=/home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28/trace-%A_%a.log
set -euo pipefail
REPO=/home/edmondwa/NVIDIA_V2D_Egocentric
PY=/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
cd "$REPO/baseline/robotic_grounding/flash_chord"
EPISODES=(5 19 30)
EP=${EPISODES[$SLURM_ARRAY_TASK_ID]}
"$PY" "$REPO/reports/policy_targeted_2026-10-28/diagnose.py" trained --episode "$EP"
