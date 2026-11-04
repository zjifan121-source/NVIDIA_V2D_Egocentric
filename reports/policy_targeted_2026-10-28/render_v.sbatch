#!/bin/bash
#SBATCH -J v2d-render-wk
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A6000|6000Ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o /home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28/videos_wk/render-%A_%a.log
# Render the policy of one evaluated run dir (early_* or rescored) to MP4, camera framed on that run's own reference.
# usage: sbatch --array=0-39 render_v.sbatch   (rows of videos_wk/jobs.tsv: EP TAG RUNDIR)
set -euo pipefail
umask 002
REPO=/home/edmondwa/NVIDIA_V2D_Egocentric
OUT=$REPO/reports/policy_targeted_2026-10-28/videos_wk
PY=/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python
read EP TAG RUN < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $OUT/jobs.tsv)
POLICY=$(python3 -c "
import json,os
p=json.load(open(\"$RUN/checkpoints.json\"))[\"$EP\"]
r=p.replace(\"/train/\",\"/train_relocated/\").replace(\"/scratch/haisenk-v2d/night/\",\"/data/user_data/edmondwa/v2d/ken_night/night_2026-10-24/\")
print(r if os.path.exists(r) else p)")
FRAMES=$(python3 -c "import json;print([e for e in json.load(open(\"$REPO/tools/track3/episodes.json\"))[\"episodes\"] if e[\"episode\"]==$EP][0][\"num_frames\"])")
CAM=$($PY -c "
import numpy as np,pyarrow.parquet as pq
t=pq.read_table(\"$RUN/results/episode_%06d.parquet\"%$EP,columns=[\"reference_position_\"+a for a in \"xyz\"]).to_pydict()
c=np.array([np.mean(t[\"reference_position_\"+a]) for a in \"xyz\"])
o=np.array([0.55,-0.55,0.45])
f=lambda v:\"[\"+\",\".join(\"%.3f\"%x for x in v)+\"]\"
print(\"viewer.camera_position=\"+f(c+o)+\" viewer.camera_target=\"+f(c))")
echo "ep$EP $TAG run=$RUN policy=$POLICY $CAM"
cd $REPO/baseline/robotic_grounding/flash_chord
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYOPENGL_PLATFORM=egl MUJOCO_GL=egl PYGLET_HEADLESS=true
$PY scripts/view_policy.py evaluation.checkpoint=$POLICY \
  viewer.backend=mp4 viewer.headless=true viewer.realtime=false viewer.output_path=$OUT/ep${EP}_${TAG}.mp4 viewer.num_frames=$FRAMES \
  viewer.width=960 viewer.height=540 viewer.video_fps=20.254 markers.keypoints=false markers.contacts=false $CAM hydra.run.dir=/tmp/view_wk_${EP}_${TAG}
ls -la $OUT/ep${EP}_${TAG}.mp4
