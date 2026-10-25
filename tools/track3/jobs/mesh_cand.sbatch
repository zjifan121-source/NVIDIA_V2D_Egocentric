#!/bin/bash
# R7: new candidate meshes for one object from clearer views (mesh_candidates.py), each tracked
# with the run's own masks, hands and camera (hard links; namespaces mc_a / mc_b for the first and
# second candidate), size estimated as usual, no refinement. Compare candidates with select_mesh.py.
# Usage: sbatch mesh_cand.sbatch <episode> <object> <src_ns>
#SBATCH -J mesh-cand
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
EP=${1:?episode}; OBJ=${2:?object}; SRCNS=${3:?source namespace}
REPO=/home/$USER/NVIDIA_V2D_Egocentric
O=/data/user_data/$USER/v2d/recon_data/outputs
E6=$(printf %06d $EP)
SRC=$O/$SRCNS/episode_$E6/$OBJ
echo "== $(date) $(hostname) restart=${SLURM_RESTART_COUNT:-0} ep$EP $OBJ from $SRCNS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export V2D_CONTAINER_RUNTIME=apptainer V2D_APPTAINER_IMAGE_DIR=/data/user_data/$USER/v2d/sif V2D_APPTAINER_FLAGS="--no-mount /opt"
export HF_HUB_CACHE=/data/hf_cache/hub
cd $REPO
PY=$REPO/baseline/reconstruction/.venv/bin/python
$PY tools/track3/mesh_candidates.py "$SRCNS" mesh_cand "$EP:$OBJ" || exit 1
FRAMES=$($PY -c "import json;print(' '.join(map(str,json.load(open('$O/mesh_cand/episode_$E6/$OBJ/candidate_frames.json'))['frames'])))")
i=0
for F in $FRAMES; do
  NS=$([ $i -eq 0 ] && echo mc_a || echo mc_b); i=$((i+1))
  DST=$O/$NS/episode_$E6/$OBJ
  mkdir -p $DST
  echo "$F" > $DST/candidate_frame.txt
  for f in frames depth anycalib intrinsics intrinsics_stable.json depth_scale.json stereo_seed.json masks masks_overlay.mp4 \
           sam2_prompts.json sam2_hand_prompt_masks dino_detections.json prompts_overlay.png hand_detections.json \
           hand_tracks.json late_hand_wilor.json wilor_raw hamer hamer_overlay.mp4 hamer_aligned hamer_aligned_overlay.mp4 \
           hamer_aligned_filled hamer_aligned_filled_overlay.mp4 geocalib object_track.json slam_poses slam_trajectory.txt; do
    [ -e $SRC/$f ] && [ ! -e $DST/$f ] && cp -al $SRC/$f $DST/$f
  done
  $PY tools/track3/run_baseline_reconstruction.py --episode "$EP" --object "$OBJ" --branch $NS \
    --seed /data/user_data/$USER/v2d/stereo_test/seed_fs/ep$EP --reference-frame 0 --dev --no-refinement \
    --object-mesh $O/mesh_cand/episode_$E6/$OBJ/frame_$(printf %06d $F)/mesh_pretransformed.obj || echo "candidate frame $F failed"
done
echo "== $(date) done"
