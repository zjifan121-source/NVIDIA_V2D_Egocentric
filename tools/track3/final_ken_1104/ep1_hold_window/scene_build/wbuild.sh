#!/bin/bash
# hand5w (3 Nov): build a scene on Wentao's scored20_contactshape_proxy_mix_v1_20261103 bundle.
#   MODE=control  : his bundle as delivered, no cleanup (what the team loads), + his support_v1 table if present
#   MODE=rawhands : our raw hands in his world, hand-only cleanup against HIS objects (fill/smooth/attach; no rest hold,
#                   no object clean, no levelling, so his objects stay as delivered unless a patch changes them)
#                   [+ H5_PATCH=<patch>] [+ LEFT_NPZ=<npz with corrected hand_left_*>], + his support_v1 table
#   bash wbuild.sh <name> <ep> <control|rawhands>
set -u
NAME=$1 EP=$2 MODE=$3
HW=$HOME/v2d_ops/kenc/hand5w; R=/scratch/haisenk-v2d/hand5w/repo; P=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python
W=/scratch/haisenk-v2d/wentao_mix_20261103/experiments/wentacc/scored20_contactshape_proxy_mix_v1_20261103
B=/scratch/haisenk-v2d/hand5w/bundles/$NAME OUT=/scratch/haisenk-v2d/hand5w/scenes/$NAME SEQ=ep${EP}_hw_$NAME
[ -e $OUT/report.json ] && { echo "exists $OUT"; } || {
  [ -e $OUT ] && { echo "partial $OUT exists - use a new name"; exit 1; }
  mkdir -p $(dirname $B) $(dirname $OUT)
  $P $HW/wprep.py $EP $MODE $B ${LEFT_NPZ:+--left-npz $LEFT_NPZ} || exit 1
  if [ $MODE = control ]; then FL="--no-rest-hold --no-hand-clean --no-object-clean --no-level"
  else FL="--no-rest-hold --no-object-clean --no-level --attach-hands --smooth-hands 2"; fi
  mkdir -p $OUT
  (cd $R && env ${H5_PATCH:+H5_PATCH=$H5_PATCH} ${H5_DIAG_DIR:+H5_DIAG_DIR=$H5_DIAG_DIR} $P tools/track3/run_load_retarget.py --bundle $B --episode $EP \
     $FL --skip-video --device cpu --sequence-id $SEQ --out $OUT > $OUT.log 2>&1) || { echo "BUILD FAILED $NAME"; tail -25 $OUT.log; exit 1; }
}
PQD=$OUT/assets/human_motion_data/ego_recon/processed/sequence_id=$SEQ/robot_name=sharpa_wave
PQF=$(ls $PQD/*.parquet | head -1)
SPEC=$W/scenes/episode_$(printf %06d $EP)/support_v1/support_in_bundle_world.json
SUP=$OUT/assets/human_motion_data/ego_recon/reconstructed_stage/${SEQ}_sharpa_wave_support.usda
if [ -f $SPEC ] && [ ! -f $OUT/SUPPORT_V1_INSTALLED ]; then
  BREF=$B; [ -f $OUT/bundle_cleaned/result.npz ] && BREF=$OUT/bundle_cleaned
  [ -f $SUP ] && mv $SUP $SUP.generated_by_loader
  $P $W/tools/install_support.py --bundle $BREF --parquet $PQF --spec $SPEC > $OUT.support.log 2>&1 \
    && touch $OUT/SUPPORT_V1_INSTALLED || { echo "SUPPORT FAILED $NAME"; cat $OUT.support.log | tail -8; exit 1; }
fi
echo "SCENE_OK $NAME $EP $PQD"
