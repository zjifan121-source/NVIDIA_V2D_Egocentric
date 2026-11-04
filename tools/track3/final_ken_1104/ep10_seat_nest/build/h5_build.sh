#!/bin/bash
# hand5 bench (2 Nov): build one two-object scene from namespace $NS (default ken_c_scored) with the private
# rlr_h5.py (= team run_load_retarget.py + 2 env hooks: H5_MERGED_NPZ replaces the merged result.npz before cleanup,
# H5_PATCH exec's a python file that redefines cleanup functions), same flags as kenc/scene.sh; SEAT=1 then applies
# kenc/seat_usda.py --apply (as seat2.sh, the ken_c_scored_st recipe). Never overwrites an existing scene.
#   NS=ken_c_scored SEAT=0 [H5_MERGED_NPZ=..] [H5_PATCH=..] bash ~/v2d_ops/kenc/hand5/h5_build.sh <name> <ep>
set -uo pipefail
NAME=${1:?name} EP=${2:?ep} NS=${NS:-ken_c_scored} SEAT=${SEAT:-0}
T=/data/user_data/haisenk/v2d/v2d_train H=$HOME/v2d_ops/kenc/hand5
OUT=/data/user_data/haisenk/v2d/hand5/scenes/$NAME SEQ=ep${EP}_h5_$NAME
[ -e "$OUT/report.json" ] && { echo "exists: $OUT (not rebuilding)"; } || {
  [ -e "$OUT" ] && { echo "partial scene dir exists: $OUT - use a new name"; exit 1; }
  mkdir -p "$OUT"
  cd "$T" && V2D_REPO=$T V2D_BUNDLE_NS=$NS "$T/.venv_retarget/bin/python" $H/rlr_h5.py --episode "$EP" --merge \
    --attach-hands --smooth-hands 2 --skip-video --device cpu --sequence-id "$SEQ" --out "$OUT" > "$OUT.log" 2>&1 \
    || { tail -n 30 "$OUT.log"; exit 1; }
  if [ "$SEAT" = 1 ]; then
    "$T/.venv_retarget/bin/python" $HOME/v2d_ops/kenc/seat_usda.py "$OUT" --apply 2>&1 | grep -v Warning | tee "$OUT.seat.log"
  fi
}
PQ=$OUT/assets/human_motion_data/ego_recon/processed/sequence_id=$SEQ/robot_name=sharpa_wave
[ -d "$PQ" ] || { echo "no parquet dir $PQ"; tail -n 20 "$OUT.log"; exit 1; }
echo "SCENE_OK $NAME $EP $PQ"
