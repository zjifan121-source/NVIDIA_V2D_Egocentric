#!/bin/bash
#SBATCH -J tracking-checkpoints
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A6000|6000Ada|VRAM_80GB|VRAM_141GB
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --array=0-7%2
#SBATCH --open-mode=append
#SBATCH -o /data/user_data/%u/v2d/jobs/logs/tracking-checkpoints-%A_%a.log
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python /home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/evaluate_saved.py
