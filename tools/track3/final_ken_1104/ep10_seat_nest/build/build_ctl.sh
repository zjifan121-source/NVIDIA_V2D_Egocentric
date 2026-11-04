#!/bin/bash
# k10 F1 control build (bob1 respinf recipe, unchanged patch). Runs inside a CPU step of 11035728.
set -uo pipefail
S=/data/user_data/haisenk/v2d/hand5/scenes
for n in k10_ep10_ctl; do
  if [ -e "$S/$n" ] || [ -e "$S/$n.log" ]; then echo "EXISTS $S/$n - refusing"; exit 3; fi
done
echo "start $(date) on $(hostname)"
NS=ken_c_scored SEAT=0 H5_PATCH=$HOME/v2d_ops/kenc/hand5/ep10/patch_respinf.py bash $HOME/v2d_ops/kenc/hand5/h5_build.sh k10_ep10_ctl 10
echo "rc=$? end $(date)"
