#!/bin/bash
# usage: run_roll.sh <ep>; builds the lead_v3 rawspinroll scene on the A100 CPU into /root/v2d/lead_v3/handspinroll
export V2D_ROOT=/root/v2d LEAD_BUNDLES_ROOT=/root/v2d/work/lead-sprint-release-20261103/stage/scenes LEAD_HAND_OUT=/root/v2d/lead_v3/handspinroll
PY=/root/venv-cpu/bin/python; H=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1; RT=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime
EP=$1; E=$(printf %02d $EP); cd $H
PYTHONPATH=$RT $PY $LEAD_HAND_OUT/attach_respinroll.py $EP 2>&1 | grep -v Warning | tail -3 | cut -c1-1500
$PY build_scene_spinroll.py $EP rawspinroll 2>&1 | grep -v -e Warning -e warnings.warn | grep -e "loader exit" -e Error -e assert | cut -c1-200
$PY hand_contact.py $LEAD_HAND_OUT/scenes/ep${E}_rawspinroll 2>&1 | grep -v Warning | cut -c1-300 | tail -6
echo ROLL_DONE $EP
