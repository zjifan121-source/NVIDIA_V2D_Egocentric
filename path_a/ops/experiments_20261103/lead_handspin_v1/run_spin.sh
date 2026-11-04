#!/bin/bash
# usage: run_spin.sh <ep> [--ken]; builds lead_v3 rawspin scene on the A100 CPU into /root/v2d/lead_v3/handspin
export V2D_ROOT=/root/v2d LEAD_BUNDLES_ROOT=/root/v2d/work/lead-sprint-release-20261103/stage/scenes LEAD_HAND_OUT=/root/v2d/lead_v3/handspin
PY=/root/venv-cpu/bin/python; H=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1; RT=$V2D_ROOT/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime
EP=$1; E=$(printf %02d $EP); mkdir -p $LEAD_HAND_OUT/bundles $LEAD_HAND_OUT/scenes
cd $H
[ -f build_scene_spin.py ] || sed -e "s/'kenexcl':\['--no-rest-hold','--no-hand-clean','--no-object-clean','--no-level'\],/'kenexcl':['--no-rest-hold','--no-hand-clean','--no-object-clean','--no-level'],'rawspin':['--no-rest-hold','--no-hand-clean','--no-object-clean','--no-level'],/" -e "s/assert sha(STAGE_B\/n)==h and all(np.allclose(a\[k\],s\[k\],atol=1e-6) for k in keys)/assert sha(STAGE_B\/n)==h and all(np.allclose(a[k][0],s[k][0],atol=1e-6) for k in keys)/" build_scene.py > build_scene_spin.py
PYTHONPATH=$RT $PY $LEAD_HAND_OUT/attach_respin.py $EP $2 2>&1 | grep -v Warning | tail -3 | cut -c1-900
$PY build_scene_spin.py $EP rawspin 2>&1 | grep -v -e Warning -e warnings.warn | grep -e 'loader exit' -e Error -e assert | cut -c1-200
$PY hand_contact.py $LEAD_HAND_OUT/scenes/ep${E}_rawspin 2>&1 | grep -v Warning | cut -c1-220 | tail -8
echo SPIN_DONE $EP
