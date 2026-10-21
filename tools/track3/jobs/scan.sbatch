#!/bin/bash
# Check only (public episodes): FoundationStereo depth + the released scan as the mesh, to measure how much
# score perfect meshes would win back. Reuses fs_sam3d masks/hands (mesh-independent) via hard links.
# Usage: sbatch scan.sbatch <episode> <object>
#SBATCH -J scan
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
EP=${1:?episode}; OBJ=${2:?object}
REPO=/home/$USER/NVIDIA_V2D_Egocentric
O=/data/user_data/$USER/v2d/recon_data/outputs
SRC=$O/fs_sam3d/episode_$(printf %06d $EP)/$OBJ; DST=$O/fs_scan/episode_$(printf %06d $EP)/$OBJ
echo "== $(date) $(hostname) restart=${SLURM_RESTART_COUNT:-0}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export V2D_CONTAINER_RUNTIME=apptainer V2D_APPTAINER_IMAGE_DIR=/data/user_data/$USER/v2d/sif V2D_APPTAINER_FLAGS="--no-mount /opt"
export HF_HUB_CACHE=/data/hf_cache/hub
mkdir -p $DST
for f in frames depth anycalib intrinsics intrinsics_stable.json depth_scale.json stereo_seed.json masks masks_overlay.mp4 \
         sam2_prompts.json sam2_hand_prompt_masks dino_detections.json prompts_overlay.png hand_detections.json \
         hand_tracks.json wilor_raw hamer hamer_overlay.mp4 geocalib object_track.json; do
  [ -e $SRC/$f ] && [ ! -e $DST/$f ] && cp -al $SRC/$f $DST/$f
done
"$REPO/baseline/reconstruction/.venv/bin/python" "$REPO/tools/track3/run_baseline_reconstruction.py" \
  --episode "$EP" --object "$OBJ" --branch fs_scan --seed /data/user_data/$USER/v2d/stereo_test/seed_fs/ep$EP \
  --reference-frame 0 --dev --scan-mesh
