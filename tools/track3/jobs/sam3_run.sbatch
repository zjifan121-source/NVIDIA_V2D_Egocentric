#!/bin/bash
# Reconstruction with SAM 3 object masks (--object-segmenter sam3; hands still SAM2), FoundationStereo depth,
# no refinement: the current recipe with only the object segmenter changed. Compare with fs_noref.
# Needs the one-time setup in /data/user_data/$USER/v2d/sam3 (setup.sbatch: sam3.pt + venvs/sam3).
# Usage: sbatch sam3_run.sbatch <episode> <object> [<namespace>, default sam3_noref]
#SBATCH -J sam3-run
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A6000|L40|6000Ada|A100_80GB|A100_80G|H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH -o /data/user_data/%u/v2d/stereo_test/logs/%x-%j.log
set -uo pipefail
source /home/$USER/NVIDIA_V2D_Egocentric/tools/track3/jobs/disk_guard.sh  # hold the job when the quota is nearly full
EP=${1:?episode}; OBJ=${2:?object}; NS=${3:-sam3_noref}
REPO=/home/$USER/NVIDIA_V2D_Egocentric
echo "== $(date) $(hostname) restart=${SLURM_RESTART_COUNT:-0} ep$EP $OBJ -> $NS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export V2D_CONTAINER_RUNTIME=apptainer V2D_APPTAINER_IMAGE_DIR=/data/user_data/$USER/v2d/sif V2D_APPTAINER_FLAGS="--no-mount /opt"
export HF_HUB_CACHE=/data/hf_cache/hub
export V2D_SAM3_PYTHON=/data/user_data/$USER/v2d/venvs/sam3/bin/python
cd $REPO
baseline/reconstruction/.venv/bin/python tools/track3/run_baseline_reconstruction.py --episode "$EP" --object "$OBJ" \
  --branch "$NS" --seed /data/user_data/$USER/v2d/stereo_test/seed_fs/ep$EP --reference-frame 0 --dev --no-refinement \
  --object-segmenter sam3 --sam3-weights /data/user_data/$USER/v2d/sam3
rc=$?
echo "== $(date) done (exit $rc)"
exit $rc
