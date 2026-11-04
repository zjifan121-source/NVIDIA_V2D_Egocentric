#!/bin/bash
# Wentao scored20 scene with object rotations smoothed (sigma 2 frames), positions untouched. EP env selects the episode.
#SBATCH -p maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3:00:00
set -uo pipefail
REPO=/home/edmondwa/NVIDIA_V2D_Egocentric
W=/data/user_data/edmondwa/v2d/wentao_mix
PY=/data/user_data/edmondwa/v2d/venvs/retarget/bin/python
S=$W/hub/experiments/wentacc/scored20_alignment_mix_v1_20261030/scenes
ep=$EP; E6=$(printf %06d $ep); out=$W/scenes_rs/ep$ep
mkdir -p $W/scenes_rs; mkdir $out.lock 2>/dev/null || { echo "ep$ep already claimed"; exit 0; }
mkdir -p $out
$PY $REPO/tools/track3/run_load_retarget.py --bundle $S/episode_$E6/bundle --episode $ep \
  --attach-hands --smooth-hands 2 --smooth-object-rotation 2 --skip-video --device cpu \
  --sequence-id ep${ep}_wentao_rs --out $out > $out.log 2>&1
echo "ep$ep exit $? $(date +%T)"
