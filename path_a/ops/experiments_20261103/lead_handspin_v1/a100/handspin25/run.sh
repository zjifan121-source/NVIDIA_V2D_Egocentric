#!/bin/bash
# ep25: per-frame hold capped at the release frame, then right-hand pinch repair on board 2 (hold and anchor from Edmond, visually confirmed)
set -u
export V2D_ROOT=/root/v2d LEAD_BUNDLES_ROOT=/root/v2d/work/lead-sprint-release-20261103/stage/scenes LEAD_HAND_OUT=/root/v2d/lead_v3/handspin25 LEAD_LIFT_END=333
PY=/root/venv-cpu/bin/python; H=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1; RT=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime
O=$LEAD_HAND_OUT; A=${1:-220}
cd $H
rm -rf $O/bundles/ep25_hold $O/scenes/ep25_hold
PYTHONPATH=$RT $PY /root/v2d/lead_v3/handspin/attach_hold.py 25 2>&1 | grep -v -e Warning -e cross | tail -1 | cut -c1-420
$PY build_scene_hold.py 25 hold 2>&1 | grep -v -e Warning -e warnings.warn | grep -e "loader exit" -e Error -e assert | cut -c1-200
S=$O/scenes/ep25_hold
LEAD_ROSTER=/root/v2d/NVIDIA_V2D_Egocentric/tools/track3/episodes.json $PY /root/v2d/tools_lead/export_score_input.py $S 2>&1 | tail -1 | cut -c1-120
echo "{\"episode\": 25}" > $S/report.json
rm -rf $O/grip_tmp
PYTHONPATH=$RT:/root/v2d/NVIDIA_V2D_Egocentric/tools/track3 $PY /root/v2d/lead_v3/handspin/repair_grip_align.py $S $O/grip_tmp --object wooden_piece_2 --side right --hold 150 333 --anchor $A --blend 12 --rigid-grip --align-pinch --base-bundle $O/bundles/ep25_hold 2>&1 | grep -v -e Warning -e cross | grep -e translation_cm -e rotation -e alignment -e after_cm -A2 -e Error | cut -c1-120 | head -16
[ -d $O/grip_tmp ] || { echo REPAIR_FAILED; exit 1; }
rm -rf $O/bundles/ep25_hold $O/scenes/ep25_hold; mv $O/grip_tmp $O/bundles/ep25_hold
$PY build_scene_hold.py 25 hold 2>&1 | grep -v -e Warning -e warnings.warn | grep -e "loader exit" -e Error -e assert | cut -c1-200
$PY hand_contact.py $O/scenes/ep25_hold 2>&1 | grep -v -e Warning -e cross | cut -c1-640 | tail -2
echo GRIP25_DONE
