#!/bin/bash
# Film trained policies against their reference (headless GL -> MP4), to see how a rollout fails.
# Usage: sbatch render_policy.sbatch <run name under flash_runs/diag> [...]
#   writes <run>/rollout.mp4 (newest policy checkpoint, deterministic, one loop of the clip); the camera
#   looks at the table centre (0, 0, 1.1) from 0.7 m away, override with CAM="[x,y,z]" TARGET="[x,y,z]"
#SBATCH -J render-policy
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A6000|6000Ada|VRAM_80GB
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH -o /data/user_data/%u/v2d/jobs/logs/%x-%j.log
set -uo pipefail
REPO=/home/$USER/NVIDIA_V2D_Egocentric
FC=$REPO/baseline/robotic_grounding/flash_chord
PY=/data/user_data/$USER/v2d/venvs/flash_chord/bin/python
export PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=True  # pyglet would otherwise look for an X display
cd "$FC"
echo "== $(date) $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
for NAME in "$@"; do
  RUN=/data/user_data/$USER/v2d/flash_runs/diag/$NAME
  [ -f "$RUN/rollout.mp4" ] && { echo "$NAME: done already"; continue; }
  CKPT=$(ls -1 "$RUN"/train/policy_*.safetensors | sort -V | tail -1)
  echo "== $NAME $CKPT"
  $PY scripts/view_policy.py "evaluation.checkpoint=$CKPT" viewer.backend=mp4 \
    "viewer.output_path=$RUN/rollout_tmp.mp4" viewer.num_frames=${FRAMES:-450} viewer.width=1280 viewer.height=720 \
    viewer.realtime=false markers.enabled=false "viewer.camera_position=${CAM:-[0.45,-0.55,1.5]}" "viewer.camera_target=${TARGET:-[0,0,1.1]}" 2>&1 | grep -v 'nefc overflow' | tail -5
  [ -s "$RUN/rollout_tmp.mp4" ] && mv "$RUN/rollout_tmp.mp4" "$RUN/rollout.mp4" && echo "$NAME -> $RUN/rollout.mp4"
done
echo "== $(date) done"
