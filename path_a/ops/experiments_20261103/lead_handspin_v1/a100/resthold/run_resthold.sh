#!/bin/bash
# usage: run_resthold.sh <ep>; rest-hold patch of the spinroll bundle, then the rawspinroll scene build, hand contact, COM patch (1.5 cm), score_input, spin check. Output /root/v2d/lead_v3/resthold/scenes/epNN_rawspinroll
set -u
EP=$1; E=$(printf %02d $EP)
export V2D_ROOT=/root/v2d O=/root/v2d/lead_v3/resthold LEAD_BUNDLES_ROOT=/root/v2d/work/lead-sprint-release-20261103/stage/scenes
export LEAD_HAND_OUT=$O
PY=/root/venv-cpu/bin/python; H=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1
mkdir -p $O/bundles $O/scenes $O/logs
$PY $O/rest_hold_patch.py /root/v2d/lead_v3/handspinroll/bundles/ep${E}_rawspinroll $O/bundles/ep${E}_rawspinroll 2>&1 | grep -v -i warn | tail -1 | cut -c1-600
[ -f $O/bundles/ep${E}_rawspinroll/REST_HOLD.json ] || { echo PATCH_FAILED; exit 1; }
cd $H
$PY build_scene_spinroll.py $EP rawspinroll 2>&1 | grep -v -e Warning -e warnings.warn | grep -e "loader exit" -e Error -e assert -e Traceback | cut -c1-300
$PY hand_contact.py $O/scenes/ep${E}_rawspinroll 2>&1 | grep -v -e Warning -e cross | cut -c1-500 | tail -2
/root/venv-flash-lead/bin/python /root/v2d/lead_v3/handspin/com_fix.py $O/scenes/ep${E}_rawspinroll --apply --threshold-cm 1.5 2>&1 | tail -1 | cut -c1-300
LEAD_ROSTER=/root/v2d/NVIDIA_V2D_Egocentric/tools/track3/episodes.json /root/venv-kit/bin/python /root/v2d/tools_lead/export_score_input.py $O/scenes/ep${E}_rawspinroll 2>&1 | tail -1 | cut -c1-200
$PY /root/spin_check.py $O/bundles/ep${E}_rawspinroll $H/scenes/ep${E}_rawclean/bundle_cleaned 2>&1 | grep -v -i warn | cut -c1-220
cp $O/bundles/ep${E}_rawspinroll/REST_HOLD.json $O/bundles/ep${E}_rawspinroll/ATTACH_RESPIN.json $O/scenes/ep${E}_rawspinroll/ 2>/dev/null
echo RESTHOLD_DONE $EP
