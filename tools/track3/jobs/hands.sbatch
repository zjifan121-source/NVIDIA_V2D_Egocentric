#!/bin/bash
# R4 fix: rerun one object with --late-hands (a second hand that enters after the reference frame
# gets its own SAM2 prompt). Reuses fs_sam3d's frames, depth, WiLoR detections, SAM3D mesh, object
# poses and camera path via hard links, so only the hand tracks, hand fits and refinement change.
# Usage: sbatch hands.sbatch <episode> <object>
#SBATCH -J hands
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A6000|L40|6000Ada|A100_80GB|A100_80G|H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:15:00
#SBATCH -o /data/user_data/%u/v2d/stereo_test/logs/%x-%j.log
set -uo pipefail
source /home/$USER/NVIDIA_V2D_Egocentric/tools/track3/jobs/disk_guard.sh  # hold the job when the quota is nearly full
EP=${1:?episode}; OBJ=${2:?object}
REPO=/home/$USER/NVIDIA_V2D_Egocentric
O=/data/user_data/$USER/v2d/recon_data/outputs
SRC=$O/fs_sam3d/episode_$(printf %06d $EP)/$OBJ; DST=$O/fs_hands/episode_$(printf %06d $EP)/$OBJ
echo "== $(date) $(hostname) restart=${SLURM_RESTART_COUNT:-0}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export V2D_CONTAINER_RUNTIME=apptainer V2D_APPTAINER_IMAGE_DIR=/data/user_data/$USER/v2d/sif V2D_APPTAINER_FLAGS="--no-mount /opt"
export HF_HUB_CACHE=/data/hf_cache/hub
mkdir -p $DST
for f in frames depth anycalib intrinsics intrinsics_stable.json depth_scale.json stereo_seed.json geocalib wilor_raw \
         dino_detections.json object_track.json mesh mesh_pretransformed.obj mesh_scaled.obj material.mtl material_0.png \
         scale.json poses poses_smoothed slam_poses slam_trajectory.txt; do
  [ -e $SRC/$f ] && [ ! -e $DST/$f ] && cp -al $SRC/$f $DST/$f
done
"$REPO/baseline/reconstruction/.venv/bin/python" "$REPO/tools/track3/run_baseline_reconstruction.py" \
  --episode "$EP" --object "$OBJ" --branch fs_hands --seed /data/user_data/$USER/v2d/stereo_test/seed_fs/ep$EP \
  --reference-frame 0 --dev --late-hands
