#!/bin/bash
# usage: run_blast.sh <ep> <nx> <ny> <nz> <cx> <cy> <cz> [--level]; Ken's corrected hands -> loader clean -> cups shifted to the depth table -> exclusive attach + cup respin (+ spoon roll) -> re-seated support -> hold-mode scene -> COM patch -> score_input. Output /root/v2d/lead_v3/blast/scenes/epNN_hold
set -u
EP=$1; E=$(printf %02d $EP); shift; NX=$1; NY=$2; NZ=$3; CX=$4; CY=$5; CZ=$6; LEVEL=${7:-}; ATTACH=${8:-hold}
export V2D_ROOT=/root/v2d O=/root/v2d/lead_v3/blast
PY=/root/venv-cpu/bin/python; H=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1; RT=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime
export LEAD_HAND_OUT=$O LEAD_BUNDLES_ROOT=/root/v2d/work/lead-sprint-release-20261103/stage/scenes
cd $H
SRC=$O/scenes/ep${E}_kenclean/bundle_cleaned
if [ ! -f $SRC/SHIFT_CUPS.json ]; then
$PY prep_kenhands.py $EP 2>&1 | tail -1 | cut -c1-200
$PY build_scene.py $EP kenclean 2>&1 | grep -v -e Warning -e warnings.warn | grep -e "loader exit" -e Error -e assert -e Traceback | cut -c1-200
[ -f $SRC/result.npz ] || { echo KENCLEAN_MISSING; exit 1; }
$PY $O/shift_cups_to_table.py $SRC $NX $NY $NZ $CX $CY $CZ 2>&1 | tail -1 | cut -c1-700
else echo "kenclean + shift already done, reusing $SRC"; fi
if [ "$ATTACH" = "excl" ]; then
LEAD_SPIN_SRC=$SRC PYTHONPATH=$RT $PY /root/v2d/lead_v3/handspinroll/attach_respinroll.py $EP --ken 2>&1 | grep -v -e Warning -e cross | tail -1 | cut -c1-900
rm -rf $O/bundles/ep${E}_hold; cp -a $O/bundles/ep${E}_rawspinroll $O/bundles/ep${E}_hold
else
rm -rf $O/bundles/ep${E}_hold; LEAD_HOLD_SRC=$SRC PYTHONPATH=$RT $PY /root/v2d/lead_v3/handspin/attach_hold.py $EP 2>&1 | grep -v -e Warning -e cross | tail -1 | cut -c1-900
[ -f $O/bundles/ep${E}_hold/result.npz ] || { echo HOLD_FAILED; exit 1; }
fi
if [ "$LEVEL" = "--level" ]; then $PY $O/relevel.py $EP $O/bundles/ep${E}_hold $NX $NY $NZ $CX $CY $CZ $O 2>&1 | grep -v Warn | tail -1 | cut -c1-500; else $PY $O/relevel.py $EP $O/bundles/ep${E}_hold 0 0 1 0 0 0 $O 2>&1 | grep -v Warn | tail -1 | cut -c1-500; fi
[ -f $O/seat/episode_$(printf %06d $EP)/support_v1/support_in_bundle_world.json ] || { echo RESEAT_FAILED; exit 1; }
rm -rf $O/bundles/ep${E}_hold; cp -a $O/stage/scenes/episode_$(printf %06d $EP)/bundle $O/bundles/ep${E}_hold
export LEAD_BUNDLES_ROOT=$O/stage/scenes LEAD_SEAT_ROOT=$O/seat
$PY build_scene_hold.py $EP hold 2>&1 | grep -v -e Warning -e warnings.warn | grep -e "loader exit" -e Error -e assert -e Traceback | cut -c1-300
$PY hand_contact.py $O/scenes/ep${E}_hold 2>&1 | grep -v -e Warning -e cross | cut -c1-600 | tail -2
/root/venv-flash-lead/bin/python /root/v2d/lead_v3/handspin/com_fix.py $O/scenes/ep${E}_hold --apply --threshold-cm 2.0 2>&1 | tail -1 | cut -c1-400
LEAD_ROSTER=/root/v2d/NVIDIA_V2D_Egocentric/tools/track3/episodes.json /root/venv-kit/bin/python /root/v2d/tools_lead/export_score_input.py $O/scenes/ep${E}_hold 2>&1 | tail -1 | cut -c1-200
$PY /root/spin_check.py $O/bundles/ep${E}_hold $SRC 2>&1 | grep -v -i warn | cut -c1-240
echo BLAST_DONE $EP
