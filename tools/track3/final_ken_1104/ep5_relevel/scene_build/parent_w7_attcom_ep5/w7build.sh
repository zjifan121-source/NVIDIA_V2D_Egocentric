#!/bin/bash
# hand7 ep5: build the two w7 arms on Wentao's scored20_lead_sprint_v2 ep5 bundle (new names only), edit, check, dump, copy.
#  w7_com_ep5    = v2 bundle as delivered (fb11 hands), loader control build (same flags as wv2_ctl_ep5) + v2 seated support
#                  + pot URDF COM -> shell centroid (w7edit.py --com). Single change vs wv2_ctl_ep5.
#  w7_attcom_ep5 = our raw hands in the v2 world (w7prep.py rawhands), loader hand cleanup + attach (same flags as w_raw_ep5:
#                  --no-rest-hold --no-object-clean --no-level --attach-hands --smooth-hands 2) against the v2 objects
#                  + v2 seated support + pot COM -> shell centroid. Objects unchanged.
# Runs inside a step of CPU job 11035881 (t9-20).
set -u
E=$HOME/v2d_ops/kenc/hand7/ep5; P=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python
R=/scratch/haisenk-v2d/hand5w/repo; S=/scratch/haisenk-v2d/hand5w/scenes; B=/scratch/haisenk-v2d/hand5w/bundles
V2=/scratch/haisenk-v2d/wentao_v2/experiments/wentacc/scored20_lead_sprint_v2_20261103; SPEC=$V2/scenes/episode_000005/support_v1/support_in_bundle_world.json
X=/data/user_data/haisenk/v2d/hand6/scenes; C=$S/wv2_ctl_ep5
echo "== $(date +%T) start"
build () {  # name mode
  local N=$1 MODE=$2; local SEQ=ep5_hw_$N; local OUT=$S/$N
  if [ -e $OUT ] || [ -e $B/$N ]; then echo "EXISTS $N - refusing (new names only)"; return 1; fi
  if [ $MODE = control ]; then cp -a $V2/scenes/episode_000005/bundle $B/$N; FL="--no-rest-hold --no-hand-clean --no-object-clean --no-level"
  else $P $E/w7prep.py 5 rawhands $B/$N || return 1; FL="--no-rest-hold --no-object-clean --no-level --attach-hands --smooth-hands 2"; fi
  mkdir -p $OUT
  (cd $R && $P tools/track3/run_load_retarget.py --bundle $B/$N --episode 5 $FL --skip-video --device cpu --sequence-id $SEQ --out $OUT > $OUT.log 2>&1) || { echo "BUILD_FAILED $N"; tail -25 $OUT.log; return 1; }
  local PQF=$(ls $OUT/assets/human_motion_data/ego_recon/processed/sequence_id=$SEQ/robot_name=sharpa_wave/*.parquet | head -1)
  local SUP=$OUT/assets/human_motion_data/ego_recon/reconstructed_stage/${SEQ}_sharpa_wave_support.usda; [ -f $SUP ] && mv $SUP $SUP.generated_by_loader
  local BREF=$B/$N; [ -f $OUT/bundle_cleaned/result.npz ] && BREF=$OUT/bundle_cleaned
  $P $V2/tools/install_support.py --bundle $BREF --parquet $PQF --spec $SPEC > $OUT.support.log 2>&1 || { echo "SUPPORT_FAILED $N"; tail -8 $OUT.support.log; return 1; }
  $P $E/w7edit.py $OUT --com || return 1
  echo "SCENE_OK $N $PQF"
}
build w7_com_ep5 control; echo "== $(date +%T) built w7_com_ep5"
build w7_attcom_ep5 rawhands; echo "== $(date +%T) built w7_attcom_ep5"
for n in w7_com_ep5 w7_attcom_ep5; do
  O=$S/$n; echo "######## $n"
  echo "## cmp_pq wv2_ctl_ep5 $n"; $P $HOME/v2d_ops/kenc/hand5/cmp_pq.py $C $O 2>&1 | grep -v "Warning\|m = float" | head -45
  diff <(grep -v "^#\|support_\|prim\|def \|{\|}" $C/assets/human_motion_data/ego_recon/reconstructed_stage/*support.usda) <(grep -v "^#\|support_\|prim\|def \|{\|}" $O/assets/human_motion_data/ego_recon/reconstructed_stage/*support.usda) && echo SUPPORT_NUMBERS_SAME_AS_WV2_CTL
  for o in white_pot wooden_spoon; do cmp $C/assets/human_motion_data/ego_recon/processed/ep5_hw_wv2_ctl_ep5_${o}_visual.stl $O/assets/human_motion_data/ego_recon/processed/ep5_hw_${n}_${o}_visual.stl && echo "${o}_STL_SAME"; done
  grep -A2 "<inertial>" $O/assets/human_motion_data/ego_recon/processed/*_rigid.urdf
  $P -c "import json;r=json.load(open('$O/report.json'));print('REPORT cleaning', json.dumps(r.get('cleaning'))[:900]);print('REPORT output_check', json.dumps(r.get('output_check'))[:600])"
  $P $E/g6dump.py $n $O
done
$P /data/user_data/haisenk/v2d/hand5_backup_20261103/ship_w1/tools/pen0.py $C $S/w7_com_ep5 $S/w7_attcom_ep5 2>&1 | grep -v Warning
$P /data/user_data/haisenk/v2d/hand5_backup_20261103/ship_w1/tools/seatcheck.py $C $S/w7_com_ep5 $S/w7_attcom_ep5 2>&1 | grep -v Warning
for n in w7_com_ep5 w7_attcom_ep5; do [ -e $X/$n ] && echo "DATA_COPY_EXISTS $n" || { cp -a $S/$n $X/ && echo "COPIED $n -> $X"; }; done
cd $E && tar czf x7.tgz x/scenes/w7_com_ep5 x/scenes/w7_attcom_ep5 && ls -la x7.tgz
echo "== $(date +%T) W7BUILD_DONE"
