#!/bin/bash
# k10 F1-b build (coordinator-approved 4 Nov ~03:07): NAME = k10_ep10_seat_b or k10_ep10_nest_b, patch p10seat_b.py.
# Right after SCENE_OK the scene is tarred into ep10_fix/archive/<TGZ> (pre-edit snapshot). CPU step of 11035728.
set -uo pipefail
NAME=${1:?name}; TGZ=${2:?tgz}
S=/data/user_data/haisenk/v2d/hand5/scenes
F=$HOME/v2d_ops/kenc/diag/ep10_fix
if [ -e "$S/$NAME" ] || [ -e "$S/$NAME.log" ]; then echo "EXISTS $S/$NAME - refusing"; exit 3; fi
if [ -e "$F/archive/$TGZ" ]; then echo "EXISTS archive $TGZ - refusing"; exit 3; fi
echo "start $NAME $(date) on $(hostname)"
NS=ken_c_scored SEAT=0 H5_PATCH=$F/p10seat_b.py bash $HOME/v2d_ops/kenc/hand5/h5_build.sh $NAME 10
rc=$?
echo "rc=$rc end $NAME $(date)"
if [ $rc = 0 ] && [ -e "$S/$NAME/report.json" ]; then
  cd $S && tar czf $F/archive/$TGZ $NAME && md5sum $F/archive/$TGZ && echo "entries $(tar tzf $F/archive/$TGZ | wc -l)" && echo "archived $NAME $(date)"
fi
