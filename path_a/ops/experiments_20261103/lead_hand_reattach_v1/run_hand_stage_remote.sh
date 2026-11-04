#!/bin/bash
# Hand stage on a cloud host (A100: /root/v2d layout, /root/venv-cpu). Usage: run_hand_stage_remote.sh <ep> [<ep> ...]
# Env: V2D_ROOT (default /root/v2d), LEAD_BUNDLES_ROOT (objects source: v1 stage scenes dir or an HF download of
# experiments/wentacc/scored20_lead_sprint_v1_20261103/scenes), PY (default /root/venv-cpu/bin/python).
set -u
export V2D_ROOT=${V2D_ROOT:-/root/v2d}
export LEAD_BUNDLES_ROOT=${LEAD_BUNDLES_ROOT:-$V2D_ROOT/work/lead-sprint-release-20261103/stage/scenes}
PY=${PY:-/root/venv-cpu/bin/python}
H=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1
O=${LEAD_HAND_OUT:-$H}   # output root: bundles/ and scenes/ live here (build_scene/attach_exclusive read LEAD_HAND_OUT)
RT=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime
cd $H
for ep in "$@"; do
  e=$(printf %02d $ep)
  echo "=== ep$e prep"; $PY prep_rawhands.py $ep 2>&1 | grep -e PREP_OK -e Error -e assert
  rm -rf $O/bundles/ep${e}_rawclean; cp -r $O/bundles/ep${e}_rawhands $O/bundles/ep${e}_rawclean
  for m in control rawclean; do echo "=== ep$e $m"; $PY build_scene.py $ep $m 2>&1 | grep -v -e Warning -e warnings.warn | grep -e 'loader exit' -e Error -e assert; done
  echo "=== ep$e attach_exclusive"; PYTHONPATH=$RT $PY attach_exclusive.py $ep 2>&1 | grep -e EXCL -e Error -e assert
  echo "=== ep$e rawexcl"; $PY build_scene.py $ep rawexcl 2>&1 | grep -v -e Warning -e warnings.warn | grep -e 'loader exit' -e Error -e assert
  $PY hand_contact.py $O/scenes/ep${e}_control $O/scenes/ep${e}_rawclean $O/scenes/ep${e}_rawexcl 2>&1 | grep -v Warning | cut -c1-200
done
$PY compare_hands.py > $O/compare.log 2>&1; echo HAND_STAGE_DONE
