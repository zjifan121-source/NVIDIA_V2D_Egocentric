#!/bin/bash
# Stages 2-3 (load + retarget) for one bundle on a GPU node, including the QA video (EGL render).
# Usage:
#   sbatch load_retarget.sbatch --episode 41 --object white_pot
#   sbatch load_retarget.sbatch --bundle <dir> --sequence-id <id> --fps 30 --out <dir>
# Arguments are passed straight to tools/track3/run_load_retarget.py; steps already done are skipped.
#SBATCH -J load-retarget
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /data/user_data/%u/v2d/jobs/logs/%x-%j.log
set -uo pipefail
REPO=/home/$USER/NVIDIA_V2D_Egocentric
PY=/data/user_data/$USER/v2d/venvs/retarget/bin/python
echo "== $(date) $(hostname) restart=${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY "$REPO/tools/track3/run_load_retarget.py" "$@"
