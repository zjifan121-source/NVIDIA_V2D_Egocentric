#!/bin/bash
#SBATCH -J ep25-pinch-transfer
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|6000Ada|H100|H200
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --open-mode=append
#SBATCH -o /data/user_data/%u/v2d/jobs/logs/ep25-pinch-transfer-%j.log
set -euo pipefail
REPO=/home/edmondwa/NVIDIA_V2D_Egocentric
RUN=/data/user_data/edmondwa/v2d/flash_runs/diag/ep25_pinch_transfer_s44
mkdir -p "$RUN/train"
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python "$REPO/reports/submission_candidate_2026-10-28/transfer_pinch_checkpoint.py"
/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python "$REPO/assets/data/v2d_submission_kit/examples/run_policy_evaluation.py" --checkpoints "$RUN/checkpoints.json" --results "$RUN/results" --runtime-root "$REPO/baseline/robotic_grounding/flash_chord" > "$RUN/eval.log" 2>&1
touch "$RUN/eval.done"
