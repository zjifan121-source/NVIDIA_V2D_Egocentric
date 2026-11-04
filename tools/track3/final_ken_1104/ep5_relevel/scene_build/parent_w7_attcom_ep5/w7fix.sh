#!/bin/bash
# hand7 ep5: complete w7_attcom_ep5 (install_support.py refused the rawhands bundle: "Bundle mismatch: result.npz", so the build
# stopped before the support and the COM edit, but the scene had already been copied to /data). Same support as the control
# via hand5w/wsupport.py (checks frame-0 object poses agree), then the pot COM edit, checks, re-dump, and refresh the /data copy.
set -u
E=$HOME/v2d_ops/kenc/hand7/ep5; P=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python
S=/scratch/haisenk-v2d/hand5w/scenes; X=/data/user_data/haisenk/v2d/hand6/scenes; N=w7_attcom_ep5; O=$S/$N; C=$S/wv2_ctl_ep5
echo "== $(date +%T) start"
$P $HOME/v2d_ops/kenc/hand5w/wsupport.py $C ep5_hw_wv2_ctl_ep5 $O ep5_hw_$N || exit 1
$P $E/w7edit.py $O --com || exit 1
diff <(grep -v "^#" $C/assets/human_motion_data/ego_recon/reconstructed_stage/*support.usda) <(grep -v "^#" $O/assets/human_motion_data/ego_recon/reconstructed_stage/ep5_hw_${N}_sharpa_wave_support.usda) && echo SUPPORT_SAME_AS_WV2_CTL
grep -A1 "<inertial>" $O/assets/human_motion_data/ego_recon/processed/*_rigid.urdf | grep origin
$P /data/user_data/haisenk/v2d/hand5_backup_20261103/ship_w1/tools/pen0.py $C $S/w7_com_ep5 $O 2>&1 | grep -v Warning
$P /data/user_data/haisenk/v2d/hand5_backup_20261103/ship_w1/tools/seatcheck.py $C $S/w7_com_ep5 $O 2>&1 | grep -v Warning
$P $E/g6dump.py $N $O
# refresh the /data copy of this (my own, new) scene with the files added/changed above, then verify the two copies match
cp -a $O/assets/human_motion_data/ego_recon/reconstructed_stage/ep5_hw_${N}_sharpa_wave_support.usda $X/$N/assets/human_motion_data/ego_recon/reconstructed_stage/
cp -a $O/SUPPORT_V1_COPIED_FROM_CONTROL $O/W7EDIT.json $X/$N/; cp -a $O/w7_backup $X/$N/
cp -a $O/assets/human_motion_data/ego_recon/processed/ep5_hw_${N}_white_pot_rigid.urdf $X/$N/assets/human_motion_data/ego_recon/processed/
diff -rq $O $X/$N && echo "DATA_COPY_IDENTICAL $N"
diff -rq $S/w7_com_ep5 $X/w7_com_ep5 && echo "DATA_COPY_IDENTICAL w7_com_ep5"
cd $E && tar czf x7b.tgz x/scenes/w7_com_ep5 x/scenes/w7_attcom_ep5 && ls -la x7b.tgz
echo "== $(date +%T) W7FIX_DONE"
