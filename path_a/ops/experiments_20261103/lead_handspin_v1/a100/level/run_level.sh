#!/bin/bash
# usage: run_level.sh <ep> <nx> <ny> <nz> <cx> <cy> <cz>; re-levels the rawclean bundle, re-attaches hands with the hold rule, builds the scene with the re-seated support, applies the COM patch, exports score_input. Output /root/v2d/lead_v3/level/scenes/epNN_hold
set -u
EP=$1; E=$(printf %02d $EP); shift
export V2D_ROOT=/root/v2d L=/root/v2d/lead_v3/level
export LEAD_BUNDLES_ROOT=$L/stage/scenes LEAD_HAND_OUT=$L LEAD_SEAT_ROOT=$L/seat LEAD_HOLD_SRC=$L/stage/scenes/episode_$(printf %06d $EP)/bundle
PY=/root/venv-cpu/bin/python; H=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1; RT=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime
$PY $L/relevel.py $EP $H/scenes/ep${E}_rawclean/bundle_cleaned "$@" $L 2>&1 | grep -v Warn | tail -2 | cut -c1-600; grep -q RELEVEL.json $L/stage/scenes/episode_$(printf %06d $EP)/bundle/RELEVEL.json 2>/dev/null || [ -f $L/stage/scenes/episode_$(printf %06d $EP)/bundle/RELEVEL.json ] || { echo RELEVEL_FAILED; exit 1; }
cd $H
PYTHONPATH=$RT $PY /root/v2d/lead_v3/handspin/attach_hold.py $EP 2>&1 | grep -v -e Warning -e cross | tail -2 | cut -c1-900
$PY build_scene_hold.py $EP hold 2>&1 | grep -v -e Warning -e warnings.warn | grep -e "loader exit" -e Error -e assert -e Traceback | cut -c1-300
$PY hand_contact.py $L/scenes/ep${E}_hold 2>&1 | grep -v -e Warning -e cross | cut -c1-500 | tail -3
/root/venv-flash-lead/bin/python /root/v2d/lead_v3/handspin/com_fix.py $L/scenes/ep${E}_hold --apply --threshold-cm 2.0 2>&1 | tail -1 | cut -c1-400
LEAD_ROSTER=/root/v2d/NVIDIA_V2D_Egocentric/tools/track3/episodes.json /root/venv-kit/bin/python /root/v2d/tools_lead/export_score_input.py $L/scenes/ep${E}_hold 2>&1 | tail -1 | cut -c1-200
echo LEVEL_DONE $EP
