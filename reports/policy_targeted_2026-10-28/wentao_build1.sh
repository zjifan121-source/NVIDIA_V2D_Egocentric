#!/bin/bash
# Download Wentao's scored20 alignment mix and build one training scene per episode (Ken's recipe: attach hands, smooth 2, CPU).
set -uo pipefail
REPO=/home/edmondwa/NVIDIA_V2D_Egocentric
W=/data/user_data/edmondwa/v2d/wentao_mix
PY=/data/user_data/edmondwa/v2d/venvs/retarget/bin/python
mkdir -p $W
S=$W/hub/experiments/wentacc/scored20_alignment_mix_v1_20261030/scenes
build() {
  ep=$1; E6=$(printf %06d $ep); out=$W/scenes/ep$ep
  mkdir $out.lock 2>/dev/null || { echo "ep$ep already claimed"; return; }
  mkdir -p $out
  $PY $REPO/tools/track3/run_load_retarget.py --bundle $S/episode_$E6/bundle --episode $ep \
    --attach-hands --smooth-hands 2 --skip-video --device cpu \
    --sequence-id ep${ep}_wentao_mix_attach --out $out > $out.log 2>&1
  echo "ep$ep exit $? $(date +%T)"
}
export -f build; export W S PY REPO
build $EP
echo ALL_DONE
