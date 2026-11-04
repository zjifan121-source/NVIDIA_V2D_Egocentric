# RELEASE NOTE: copy of final_b2b.sh with the base64 payload lines removed (heredoc left empty); see ../../README.md
set -e
base64 -d > $HOME/v2d_ops/kenc/diag/ep10_fix/nestcol_nest_b/out_final.tgz <<'B64EOF'
B64EOF
md5sum $HOME/v2d_ops/kenc/diag/ep10_fix/nestcol_nest_b/out_final.tgz
srun --jobid=11035728 --overlap -n1 -c4 --mem=32G bash -s <<'INNER'
set -e
PY=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python
S=/data/user_data/haisenk/v2d/hand5/scenes/k10_ep10_nest_b
SEAT=/data/user_data/haisenk/v2d/hand5/scenes/k10_ep10_seat_b
D=~/v2d_ops/kenc/diag/ep10_fix
W=$D/nestcol_nest_b
cd $W && mkdir -p orch_out && tar xzf $HOME/v2d_ops/kenc/diag/ep10_fix/nestcol_nest_b/out_final.tgz -C orch_out 
cp orch_out/out_beige.npz hulls_beige.npz && cp orch_out/out_blue.npz hulls_blue.npz
$PY -c "
import json
a=json.load(open('orch_out/out_beige.json')); b=json.load(open('orch_out/out_blue.json'))
json.dump(dict(beige={k:v for k,v in a.items() if k!='pieces'}, blue={k:v for k,v in b.items() if k!='pieces'}, pieces=a['pieces']+b['pieces']), open('orch_summary.json','w'), indent=1)
"
cd $D
$PY nestcol.py check --scene $S --work $W --bob1_beige_hulls bob1_hulls/hull_beige_bf2fae51eba73fef.npz > $W/check.log 2>&1; echo "CHECK_EXIT $?"
grep -E "^a\)|^b\)|^d\)" $W/check.log | cut -c1-2500
$PY nestcol.py settle --scene $S --work $W --bob1_blue_hulls bob1_hulls/hull_blue_46e5d1341ef7c706.npz --bob1_beige_hulls bob1_hulls/hull_beige_bf2fae51eba73fef.npz > $W/settle.log 2>&1; echo "SETTLE_EXIT $?"
grep -E "^c\)" $W/settle.log | cut -c1-3000
echo "== apply"
$PY nestcol.py apply --scene $S --work $W --target k10_ep10_nest_b > $W/apply.log 2>&1; echo "APPLY_EXIT $?"
cat $W/apply.log
$PY nestcol.py ecmp --scene $S --work $W --ref_scene $SEAT > $W/ecmp.log 2>&1; echo "ECMP_EXIT $?"
grep -E "all_identical|\"identical\"|identical_except" $W/ecmp.log
ls -la $S $S/col_backup | head -30
ls $S/assets/human_motion_data/ego_recon/processed/ | wc -l
md5sum $S/assets/human_motion_data/ego_recon/processed/*_rigid.urdf $S/col_backup/*
grep -c "<collision>" $S/assets/human_motion_data/ego_recon/processed/*_rigid.urdf
INNER
