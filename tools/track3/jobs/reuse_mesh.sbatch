#!/bin/bash
# Track one object with a mesh we reconstructed in another episode (one mesh per object, reused).
# Reuses this episode's frames, depth, masks, hands and camera from <src_ns> via hard links, so only the
# mesh, its size estimate and the object track change; no refinement (the current recipe).
# Usage: sbatch reuse_mesh.sbatch <episode> <object> <src_ns> <mesh.obj> <dst_ns>
#   mesh.obj: a mesh_pretransformed.obj from the donor episode's run (size is re-estimated here), or with
#   KEEP_SIZE=1 a metric mesh (the donor's mesh_scaled.obj) tracked at its own size
#SBATCH -J reuse-mesh
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A6000|L40|6000Ada|A100_80GB|A100_80G|H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:15:00
#SBATCH --open-mode=append
#SBATCH -o /data/user_data/%u/v2d/stereo_test/logs/%x-%j.log
set -uo pipefail
source /home/$USER/NVIDIA_V2D_Egocentric/tools/track3/jobs/disk_guard.sh  # hold the job when the quota is nearly full
EP=${1:?episode}; OBJ=${2:?object}; SRCNS=${3:?source namespace}; MESH=${4:?mesh}; NS=${5:?destination namespace}
REPO=/home/$USER/NVIDIA_V2D_Egocentric
O=/data/user_data/$USER/v2d/recon_data/outputs
E6=$(printf %06d $EP)
SRC=$O/$SRCNS/episode_$E6/$OBJ; DST=$O/$NS/episode_$E6/$OBJ
echo "== $(date) $(hostname) restart=${SLURM_RESTART_COUNT:-0} ep$EP $OBJ from $SRCNS with $MESH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export V2D_CONTAINER_RUNTIME=apptainer V2D_APPTAINER_IMAGE_DIR=/data/user_data/$USER/v2d/sif V2D_APPTAINER_FLAGS="--no-mount /opt"
export HF_HUB_CACHE=/data/hf_cache/hub
mkdir -p $DST
echo "$MESH" > $DST/mesh_source.txt
for f in frames depth anycalib intrinsics intrinsics_stable.json depth_scale.json stereo_seed.json masks masks_overlay.mp4 \
         sam2_prompts.json sam2_hand_prompt_masks dino_detections.json prompts_overlay.png hand_detections.json \
         hand_tracks.json late_hand_wilor.json wilor_raw hamer hamer_overlay.mp4 hamer_aligned hamer_aligned_overlay.mp4 \
         hamer_aligned_filled hamer_aligned_filled_overlay.mp4 geocalib object_track.json slam_poses slam_trajectory.txt; do
  [ -e $SRC/$f ] && [ ! -e $DST/$f ] && cp -al $SRC/$f $DST/$f
done
cd $REPO
baseline/reconstruction/.venv/bin/python tools/track3/run_baseline_reconstruction.py --episode "$EP" --object "$OBJ" \
  --branch $NS --seed /data/user_data/$USER/v2d/stereo_test/seed_fs/ep$EP --reference-frame 0 --dev --no-refinement \
  --object-mesh "$MESH" $([ "${KEEP_SIZE:-0}" = 1 ] && echo --keep-mesh-size)
echo "== $(date) done"
