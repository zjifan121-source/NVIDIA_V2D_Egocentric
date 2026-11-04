#!/bin/bash
# ep8 planner feasibility (own dirs only): control rebuild of the base scene from its exact merged bundle, and the
# despin variant (sym_common despin, diag path redirected to my dir); then compare with the base. CPU only.
set -u
D=/scratch/haisenk-v2d/diag/ep8_plan; K=$HOME/v2d_ops/kenc/diag/ep8_plan; T=/data/user_data/haisenk/v2d/v2d_train
BASE=/data/user_data/haisenk/v2d/NVIDIA_V2D_Egocentric/baseline/reconstruction/data/outputs/load_retarget/episode_000008/ken_c_scored_attach
EB=/data/user_data/haisenk/v2d/edmond_bob1/relocated/eb1_ep8
PY=$T/.venv_retarget/bin/python
mkdir -p $D/diag
[ -d $D/bundle_base ] || cp -aL $BASE/bundle_merged $D/bundle_base
ls -la $D/bundle_base | head -20; md5sum $D/bundle_base/result.npz $BASE/bundle_merged/result.npz
sed "s#/scratch/haisenk-v2d/hand5/ep10/diag#$D/diag#" $HOME/v2d_ops/kenc/hand5/ep10/sym_common.py > $D/sym_common_ep8.py
printf '_H5_MODE = "despin"\n_P = "%s/sym_common_ep8.py"\nexec(compile(open(_P).read(), _P, "exec"))\n' $D > $D/patch_despin_ep8.py
grep -n "hand5/ep10/diag\|$D/diag" $D/sym_common_ep8.py | head -3
for arm in ctl despin; do
  [ -e $D/$arm/report.json ] && { echo "exists $arm"; continue; }
  P=""; [ $arm = despin ] && P=$D/patch_despin_ep8.py
  echo "== build $arm $(date +%T)"
  (cd $T && env V2D_REPO=$T ${P:+H5_PATCH=$P} timeout 900 $PY $HOME/v2d_ops/kenc/hand5/rlr_h5.py --bundle $D/bundle_base --episode 8 \
     --attach-hands --smooth-hands 2 --skip-video --device cpu --sequence-id ep8_ken_c_scored_attach --out $D/$arm > $D/$arm.log 2>&1) \
     || { echo "BUILD FAILED $arm"; tail -n 25 $D/$arm.log; }
  echo "== done $arm $(date +%T)"
done
for arm in ctl despin; do
  echo "== report $arm"; $PY -c "
import json,sys; r=json.load(open(sys.argv[1])); c=r.get('cleaning',{}); o=r.get('output_check',{})
print({k:c.get(k) for k in ('rest_hold_until_frame_beige_cup','rest_hold_until_frame_white_pot','tilt_deg_0','levelled_0','tilt_deg_1','levelled_1','attach_rule','attached_0','attached_1','hand_right_refilled_frames','h5_sym_mode','h5_despin_max_deg_0','h5_despin_max_deg_1')})
print('seat', o.get('frame0_seating_cm'), 'disks', o.get('support_disks'), 'task_err', o.get('task_error_cm'), 'pen', o.get('penetration_frames'))" $D/$arm/report.json 2>&1 | tail -3
done
echo "== cmp_pq ctl vs eb1_ep8"; $PY $HOME/v2d_ops/kenc/hand5/cmp_pq.py $EB $D/ctl 2>&1 | grep -v Warning | tail -n 30
echo "== cmp_scene ctl vs base"; $PY $HOME/v2d_ops/kenc/hand5/cmp_scene.py $BASE $D/ctl 2>&1 | grep -v Warning | tail -n 30
echo "== support usda"; for s in $EB $D/ctl $D/despin; do grep -h "radius\|translate" $s/assets/human_motion_data/ego_recon/reconstructed_stage/*support.usda; done
echo "== md5 meshes"; md5sum $EB/assets/human_motion_data/ego_recon/processed/*.obj $D/ctl/assets/human_motion_data/ego_recon/processed/*.obj $D/despin/assets/human_motion_data/ego_recon/processed/*.obj 2>&1
echo "== chk_scene"; $PY -I $K/chk_scene.py $EB $D/ctl $D/despin 2>&1 | grep -v Warning
echo FEAS_DONE $(date +%T)
