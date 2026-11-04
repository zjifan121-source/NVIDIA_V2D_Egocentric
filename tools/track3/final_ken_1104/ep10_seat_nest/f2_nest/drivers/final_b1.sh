set -e
srun --jobid=11035728 --overlap -n1 -c4 --mem=32G bash -s <<'INNER'
set -e
PY=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python
S=/data/user_data/haisenk/v2d/hand5/scenes/k10_ep10_nest_b
D=~/v2d_ops/kenc/diag/ep10_fix
W=$D/nestcol_nest_b
echo "== $(hostname) $(date)"
ls -la $S
test ! -e $S/NESTCOL.json && test ! -e $S/col_backup && echo "scene not yet edited: OK"
ls -la $S/assets/human_motion_data/ego_recon/processed/
md5sum $S/assets/human_motion_data/ego_recon/processed/*_rigid.urdf $S/assets/human_motion_data/ego_recon/processed/*_visual.stl $S/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet
test ! -e $W
mkdir -p $W
cd $D
$PY nestcol.py prep --scene $S --work $W --reuse dry_bob1
$PY nestcol.py slabs --scene $S --work $W > $W/slabs.log 2>&1; echo "SLABS_EXIT $?"
head -60 $W/slabs.log
cd $W/orch && cp $D/fc_hulls.py . && tar czf $W/orch_final.tgz fc_hulls.py beige_test.urdf beige_col.stl blue_test.urdf blue_col_*.stl blue_floor.stl && ls -la $W/orch_final.tgz && md5sum $W/orch_final.tgz
INNER
