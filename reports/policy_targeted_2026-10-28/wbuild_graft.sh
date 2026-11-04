#!/bin/bash
# Wentao scene with Ken grasps grafted onto Wentao objects (tools/track3/graft_grasps.py). EP, DONOR, [ROTOBJ] env.
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
ep=$EP; E6=$(printf %06d $ep); b=$W/graft_bundles/ep$ep; out=$W/scenes_graft/ep$ep
mkdir -p $W/scenes_graft
$PY $REPO/tools/track3/graft_grasps.py --target $S/episode_$E6/bundle --donor $DONOR --out $b ${ROTOBJ:+--object-rotation-from-donor $ROTOBJ} > $b.graft.log 2>&1 || { echo "ep$ep graft failed"; exit 1; }
rm -rf $out; mkdir -p $out
$PY $REPO/tools/track3/run_load_retarget.py --bundle $b --episode $ep \
  --attach-hands --smooth-hands 2 --smooth-object-rotation ${OBJSMOOTH:-2} --skip-video --device cpu \
  --sequence-id ep${ep}_wentao_graft --out $out > $out.log 2>&1
echo "ep$ep exit $? $(date +%T)"
