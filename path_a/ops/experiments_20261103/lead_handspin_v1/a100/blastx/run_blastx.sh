#!/bin/bash
# usage: run_blastx.sh <ep> <nx> <ny> <nz> <cx> <cy> <cz>; like run_blast.sh but the floating cup is EXTENDED downward (skirt) instead of shifted; reuses the kenclean bundle from /root/v2d/lead_v3/blast with the original meshes. Output /root/v2d/lead_v3/blastx/scenes/epNN_hold
set -u
EP=$1; E=$(printf %02d $EP); shift; NX=$1; NY=$2; NZ=$3; CX=$4; CY=$5; CZ=$6
export V2D_ROOT=/root/v2d O=/root/v2d/lead_v3/blastx; BL=/root/v2d/lead_v3/blast
PY=/root/venv-cpu/bin/python; H=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1; RT=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime
mkdir -p $O/src $O/bundles $O/scenes $O/logs
SRC=$O/src/ep${E}; rm -rf $SRC; cp -a $BL/scenes/ep${E}_kenclean/bundle_cleaned $SRC
STAGE=/root/v2d/work/lead-sprint-release-20261103/stage/scenes/episode_$(printf %06d $EP)/bundle
for m in $STAGE/mesh*.obj; do rm -f $SRC/$(basename $m); cp -f $m $SRC/$(basename $m); done   # original, unshifted meshes as real files (bundle_cleaned links its meshes to the kenclean bundle, which the shift modified)
sha256sum $SRC/mesh*.obj | cut -c1-16
rm -f $SRC/SHIFT_CUPS.json
$PY $O/extend_cups_down.py $SRC $NX $NY $NZ $CX $CY $CZ 2>&1 | tail -1 | cut -c1-900
export LEAD_HAND_OUT=$O
cd $H
rm -rf $O/bundles/ep${E}_hold; LEAD_HOLD_SRC=$SRC PYTHONPATH=$RT $PY /root/v2d/lead_v3/handspin/attach_hold.py $EP 2>&1 | grep -v -e Warning -e cross | tail -1 | cut -c1-700
[ -f $O/bundles/ep${E}_hold/result.npz ] || { echo HOLD_FAILED; exit 1; }
$PY $BL/relevel.py $EP $O/bundles/ep${E}_hold 0 0 1 0 0 0 $O 2>&1 | grep -v Warn | tail -1 | cut -c1-400
[ -f $O/seat/episode_$(printf %06d $EP)/support_v1/support_in_bundle_world.json ] || { echo RESEAT_FAILED; exit 1; }
rm -rf $O/bundles/ep${E}_hold; cp -a $O/stage/scenes/episode_$(printf %06d $EP)/bundle $O/bundles/ep${E}_hold
export LEAD_BUNDLES_ROOT=$O/stage/scenes LEAD_SEAT_ROOT=$O/seat
$PY build_scene_hold.py $EP hold 2>&1 | grep -v -e Warning -e warnings.warn | grep -e "loader exit" -e Error -e assert -e Traceback | cut -c1-300
$PY hand_contact.py $O/scenes/ep${E}_hold 2>&1 | grep -v -e Warning -e cross | cut -c1-600 | tail -2
/root/venv-flash-lead/bin/python /root/v2d/lead_v3/handspin/com_fix.py $O/scenes/ep${E}_hold --apply --threshold-cm 2.0 2>&1 | tail -1 | cut -c1-400
LEAD_ROSTER=/root/v2d/NVIDIA_V2D_Egocentric/tools/track3/episodes.json /root/venv-kit/bin/python /root/v2d/tools_lead/export_score_input.py $O/scenes/ep${E}_hold 2>&1 | tail -1 | cut -c1-200
cp $SRC/EXTEND_CUPS.json $O/scenes/ep${E}_hold/; cp $O/bundles/ep${E}_hold/ATTACH_HOLD.json $O/scenes/ep${E}_hold/ 2>/dev/null
echo BLASTX_DONE $EP
