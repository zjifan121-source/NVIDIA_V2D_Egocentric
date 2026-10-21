#!/bin/bash
# Finish the episode's stereo seed (MoGe on the calibrated video, rescaled to stereo; once per episode)
# and run the baseline ego reconstruction on it for one object (~1 h).
# Usage: sbatch run.sbatch <episode> <object> <calib_sam3d|stereo_sam3d|stereo_scan|fs_sam3d>   (after seed.sbatch <episode>)
#   calib_sam3d   released calibration (replaces AnyCalib), MoGe depth as is
#   stereo_sam3d  + MoGe rescaled to stereo-metric per frame
#   stereo_scan   + the released scan as the mesh (public episodes only; development ceiling)
#   fs_sam3d      released calibration, FoundationStereo depth (fs.sbatch chunks) instead of MoGe
# Preemptible: every stage skips finished work, so a requeued job resumes.
#SBATCH -J t3-stereo
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A6000|L40|6000Ada|A100_80GB|A100_80G|H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH -o /data/user_data/%u/v2d/stereo_test/logs/%x-%j.log
set -uo pipefail
source /home/$USER/NVIDIA_V2D_Egocentric/tools/track3/jobs/disk_guard.sh  # hold the job when the quota is nearly full
EP=${1:?episode}; OBJ=${2:?object}; VAR=${3:?calib_sam3d|stereo_sam3d|stereo_scan|fs_sam3d}; PROMPT=${4:-}
REPO=/home/$USER/NVIDIA_V2D_Egocentric
SEED=/data/user_data/$USER/v2d/stereo_test/seed/ep$EP
echo "== $(date) $(hostname) restart=${SLURM_RESTART_COUNT:-0}"; nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
export V2D_CONTAINER_RUNTIME=apptainer V2D_APPTAINER_IMAGE_DIR=/data/user_data/$USER/v2d/sif V2D_APPTAINER_FLAGS="--no-mount /opt"
export HF_HUB_CACHE=/data/hf_cache/hub
SV=/data/user_data/$USER/v2d/venvs/stereo
FSMODEL=/data/user_data/$USER/v2d/recon_data/weights/foundation_stereo/deployable_foundationstereo_small_576x960_v2.0.onnx
# 0. seed stage 1 (calibrated video + SGBM; ~1 min CPU) and, for fs_sam3d, FoundationStereo depth on this GPU
#    (~1-2 min per episode) unless seed.sbatch / fs.sbatch already made them; the lock serialises an episode's jobs
(
  flock 9
  [ -f $SEED/stereo_seed.json ] || $SV/bin/python $REPO/tools/track3/stereo_seed.py --episode "$EP" --out $SEED
  if [ "$VAR" = fs_sam3d ]; then
    N=$(python3 -c "import json;print(json.load(open('$SEED/stereo_seed.json'))['frames'])")
    if [ "$(ls $SEED/fs_depth 2>/dev/null | wc -l)" -lt "$N" ]; then
      LD_LIBRARY_PATH=$(ls -d $SV/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':') STEREO_DEVICE=cuda \
        $SV/bin/python $REPO/tools/track3/stereo_seed.py --episode "$EP" --out $SEED --foundation-stereo $FSMODEL 2>&1 | grep -v affinity
    fi
  fi
) 9>$SEED.stage1.lock
[ -f $SEED/stereo_seed.json ] || { echo "seed stage 1 failed"; exit 1; }
# 1. MoGe (the baseline's depth model) on the calibrated video, then stereo-metric rescale; the lock
#    makes this episode's other jobs wait for the first instead of redoing it
[ "$VAR" = fs_sam3d ] || (
  flock 9
  if [ ! -f $SEED/depth_scale.json ]; then
    cd $REPO/baseline/reconstruction && .venv/bin/python -c "
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
run_video_to_depth(video_path='$SEED/anycalib/undistorted.mp4', depth_folder='$SEED/moge_depth',
                   intrinsics_folder='$SEED/moge_intrinsics', weights_path='data/weights/moge',
                   input_intrinsics_path='$SEED/anycalib/undistorted_intrinsics.json', dev=True)" &&
    /data/user_data/$USER/v2d/venvs/stereo/bin/python $REPO/tools/track3/stereo_seed.py --episode "$EP" --out $SEED --rescale
  fi
) 9>$SEED.lock
[ "$VAR" = fs_sam3d ] || [ -f $SEED/depth_scale.json ] || { echo "seed stage 2 failed"; exit 1; }
RUNSEED=$SEED
if [ "$VAR" = calib_sam3d ]; then
  # same calibrated video and intrinsics, but MoGe's own depth (scale 1): hard links, no copies
  RUNSEED=/data/user_data/$USER/v2d/stereo_test/seed_calib/ep$EP
  (
    flock 9
    if [ ! -f $RUNSEED/depth_scale.json ]; then
      mkdir -p $RUNSEED && cp -al $SEED/anycalib $SEED/intrinsics $RUNSEED/ && cp -al $SEED/moge_depth $RUNSEED/depth &&
      ln -f $SEED/intrinsics_stable.json $SEED/stereo_seed.json $RUNSEED/ &&
      echo '{"median_scale": 1.0, "note": "MoGe depth unscaled (calibration-only variant)"}' > $RUNSEED/depth_scale.json
    fi
  ) 9>$RUNSEED.lock
elif [ "$VAR" = fs_sam3d ]; then
  RUNSEED=/data/user_data/$USER/v2d/stereo_test/seed_fs/ep$EP
  N=$(python3 -c "import json;print(json.load(open('$SEED/stereo_seed.json'))['frames'])")
  [ "$(ls $SEED/fs_depth | wc -l)" -eq "$N" ] || { echo "fs_depth incomplete: $(ls $SEED/fs_depth | wc -l)/$N"; exit 1; }
  (
    flock 9
    if [ ! -f $RUNSEED/depth_scale.json ]; then
      mkdir -p $RUNSEED && cp -al $SEED/anycalib $SEED/intrinsics $RUNSEED/ && cp -al $SEED/fs_depth $RUNSEED/depth &&
      ln -f $SEED/intrinsics_stable.json $SEED/stereo_seed.json $RUNSEED/ &&
      echo '{"median_scale": 1.0, "note": "FoundationStereo metric depth (no MoGe)"}' > $RUNSEED/depth_scale.json
    fi
  ) 9>$RUNSEED.lock
fi
# 2. the baseline pipeline on top
EXTRA=(); [ "$VAR" = stereo_scan ] && EXTRA=(--scan-mesh)
[ -n "$PROMPT" ] && EXTRA+=(--prompt "$OBJ=$PROMPT")
"$REPO/baseline/reconstruction/.venv/bin/python" "$REPO/tools/track3/run_baseline_reconstruction.py" \
  --episode "$EP" --object "$OBJ" --branch "${NS:-$VAR}" --seed $RUNSEED --reference-frame "${REF:-0}" --dev "${EXTRA[@]}"   # REF 0 = what the hub (Hyperstack) runs used
