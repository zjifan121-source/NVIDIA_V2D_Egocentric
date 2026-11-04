#!/bin/bash
# k10 F1-b checks on k10_ep10_seat_b (vs k10_ep10_ctl), each piece in its own CPU step of 11035728 (no GPU).
# Reads k10_ep10_nest_b only (never writes it).
F=~/v2d_ops/kenc/diag/ep10_fix
S=/data/user_data/haisenk/v2d/hand5/scenes
T=/data/user_data/haisenk/v2d/v2d_train
J="--jobid=11035728 --overlap -n1"
export PYTHONDONTWRITEBYTECODE=1
for n in k10_ep10_seat_b k10_ep10_nest_b; do [ -e $F/archive/${n}*_scene.tgz ] || true; done
srun $J -c4 --mem=24G --job-name=k10b-chk bash -c "cd $T && V2D_REPO=$T $T/.venv_retarget/bin/python $F/check_f1.py $S/k10_ep10_ctl $S/k10_ep10_seat_b $F/dryrun_p10.json $F/blue_toprise.json $F/check_f1b.json" > $F/logs/check_f1b.out 2>&1 &
srun $J -c4 --mem=24G --job-name=k10b-rr /home/haisenk/anaconda3/bin/python $HOME/v2d_ops/kenc/review/tool/render_review.py --scene $S/k10_ep10_seat_b --episode 10 --label seat_b --out $F/review_seat_b > $F/logs/rr_seat_b.out 2>&1 &
srun $J -c4 --mem=24G --job-name=k10b-sk bash -c "SK_SC=$S/k10_ep10_seat_b SK_OUT=$F/sk10a2_seat_b /home/haisenk/anaconda3/bin/python $F/sk10a2_k10.py" > $F/logs/sk10a2_seat_b.out 2>&1 &
srun $J -c4 --mem=16G --job-name=k10b-ex bash -c "/home/haisenk/anaconda3/bin/python $F/extra_f1.py $S/k10_ep10_ctl $S/k10_ep10_seat_b $F/extra_f1b.json" > $F/logs/extra_f1b.out 2>&1 &
srun $J -c2 --mem=12G --job-name=k10b-pq bash -c "cd $T; for p in 'k10_ep10_ctl k10_ep10_seat_b seatb_vs_ctl' 'k10_ep10_seat k10_ep10_seat_b seatb_vs_seat' 'k10_ep10_seat_b k10_ep10_nest_b nestb_vs_seatb' 'h5_ep10_respinf_v4 k10_ep10_seat_b seatb_vs_bob1'; do set -- \$p; V2D_REPO=$T $T/.venv_retarget/bin/python $F/pqdiff.py $S/\$1 $S/\$2 $F/\$3.json 1e-5 > $F/logs/pq_\$3.out 2>&1; echo \$3 rc=\$?; done; for s in k10_ep10_seat_b k10_ep10_nest_b; do echo == \$s; (cd $S/\$s && md5sum score_input/meshes/*/*.obj assets/human_motion_data/ego_recon/processed/*_visual.stl assets/human_motion_data/ego_recon/processed/*.urdf score_input/results/*.npy assets/human_motion_data/ego_recon/processed/*/robot_name=sharpa_wave/*.parquet); done; $T/.venv_retarget/bin/python -c \"
import numpy as np
a=dict(np.load('$S/k10_ep10_seat_b/bundle_cleaned/result.npz',allow_pickle=True)); b=dict(np.load('$F/dry_b/bundle_cleaned/result.npz',allow_pickle=True))
d=[k for k in a if (a[k].dtype.kind in 'fc' and (a[k].shape!=b[k].shape or np.abs(a[k].astype(float)-b[k].astype(float)).max()>0)) or (a[k].dtype.kind not in 'fc' and not np.array_equal(a[k],b[k]))]
print('seat_b bundle_cleaned vs dry_b differing keys:', d)
\"; cp $S/k10_ep10_seat_b/report.json $F/report_seat_b.json; cp $S/k10_ep10_nest_b/report.json $F/report_nest_b.json" > $F/logs/pqdiff_b.out 2>&1 &
wait
echo CHECKS_B_DONE
