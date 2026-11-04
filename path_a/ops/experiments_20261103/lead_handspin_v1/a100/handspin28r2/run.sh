#!/bin/bash
# ep28 new pot, per-frame holds with the release rule
M=/root/v2d/lead_winners_v8/mini_stage
export V2D_ROOT=/root/v2d LEAD_BUNDLES_ROOT=$M/scenes LEAD_SEAT_ROOT=$M/seated_out LEAD_HAND_OUT=/root/v2d/lead_v3/handspin28r2 LEAD_HOLD_SRC=$M/hand_stage/scenes/ep28_rawclean/bundle_cleaned
PY=/root/venv-cpu/bin/python; H=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1; RT=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime
mkdir -p $LEAD_HAND_OUT/bundles $LEAD_HAND_OUT/scenes; cd $H
PYTHONPATH=$RT $PY /root/v2d/lead_v3/handspin/attach_hold.py 28 2>&1 | grep -v -e Warning -e cross | tail -1 | cut -c1-700
$PY build_scene_hold.py 28 hold 2>&1 | grep -v -e Warning -e warnings.warn | grep -e "loader exit" -e Error -e assert | cut -c1-200
$PY hand_contact.py $LEAD_HAND_OUT/scenes/ep28_hold 2>&1 | grep -v -e Warning -e cross | cut -c1-420 | tail -1
LEAD_ROSTER=$V2D_ROOT/NVIDIA_V2D_Egocentric/tools/track3/episodes.json $PY /root/v2d/tools_lead/export_score_input.py $LEAD_HAND_OUT/scenes/ep28_hold 2>&1 | tail -2 | head -1 | cut -c1-140
echo HOLD28R2_DONE
