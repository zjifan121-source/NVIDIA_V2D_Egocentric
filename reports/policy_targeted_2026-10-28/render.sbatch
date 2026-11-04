#!/bin/bash
#SBATCH -J v2d-render
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A6000|6000Ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o /home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28/videos/render-%A_%a.log
# Render the selected policy for one episode to MP4. usage: EP=19 [POLICY=...] sbatch render.sbatch
set -euo pipefail
umask 002
REPO=/home/edmondwa/NVIDIA_V2D_Egocentric
OUT=$REPO/reports/policy_targeted_2026-10-28/videos
EP=${EP:-${SLURM_ARRAY_TASK_ID}}
if [ -z "${POLICY:-}" ]; then
  POLICY=$(python3 -c "
import json,os
e=json.load(open(\"$REPO/reports/submission_candidate_2026-10-28/plan.json\"))[\"$EP\"]
p=json.load(open(e[\"run\"]+\"/checkpoints.json\"))[\"$EP\"]
r=p.replace(\"/train/\",\"/train_relocated/\").replace(\"/scratch/haisenk-v2d/night/\",\"/data/user_data/edmondwa/v2d/ken_night/night_2026-10-24/\")
print(r if os.path.exists(r) else p)")
fi
FRAMES=$(python3 -c "import json;print([e for e in json.load(open(\"$REPO/tools/track3/episodes.json\"))[\"episodes\"] if e[\"episode\"]==$EP][0][\"num_frames\"])")
CAM=$(/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python -c "
import json,numpy as np,pyarrow.parquet as pq
e=json.load(open(\"$REPO/reports/submission_candidate_2026-10-28/plan.json\"))[\"$EP\"]
t=pq.read_table(e[\"run\"]+\"/results/episode_%06d.parquet\"%$EP,columns=[\"reference_frame\"]+[\"reference_position_\"+a for a in \"xyz\"]).to_pydict()
c=np.mean([[t[\"reference_position_\"+a][i] for a in \"xyz\"] for i in range(len(t[\"reference_frame\"]))],0)
o=np.array(${CAM_OFFSET:-[0.55,-0.55,0.45]})
f=lambda v:\"[\"+\",\".join(\"%.3f\"%x for x in v)+\"]\"
print(\"viewer.camera_position=\"+f(c+o)+\" viewer.camera_target=\"+f(c))")
echo "camera: $CAM"
cd $REPO/baseline/robotic_grounding/flash_chord
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYOPENGL_PLATFORM=egl MUJOCO_GL=egl PYGLET_HEADLESS=true
/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python scripts/view_policy.py evaluation.checkpoint=$POLICY \
  viewer.backend=mp4 viewer.headless=true viewer.realtime=false viewer.output_path=$OUT/ep${EP}.mp4 viewer.num_frames=$FRAMES \
  viewer.width=960 viewer.height=540 viewer.video_fps=20.254 markers.keypoints=false markers.contacts=false $CAM hydra.run.dir=/tmp/view_$EP
ls -la $OUT/ep${EP}.mp4
