#!/bin/bash
#SBATCH -J nominal-moving-reference
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|6000Ada|H100|H200
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-3%2
#SBATCH --open-mode=append
#SBATCH -o /data/user_data/%u/v2d/jobs/logs/nominal-reference-%A_%a.log
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python -u /home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/evaluate_nominal.py
