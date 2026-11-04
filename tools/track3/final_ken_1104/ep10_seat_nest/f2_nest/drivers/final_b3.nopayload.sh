# RELEASE NOTE: copy of final_b3.sh with the base64 payload lines removed (heredoc left empty); see ../../README.md
set -e
base64 -d > $HOME/v2d_ops/kenc/diag/ep10_fix/nestcol_nest_b/NESTCOL_final.json <<'B64EOF'
B64EOF
srun --jobid=11035728 --overlap -n1 -c2 --mem=4G bash -s <<'INNER'
set -e
S=/data/user_data/haisenk/v2d/hand5/scenes/k10_ep10_nest_b
W=$HOME/v2d_ops/kenc/diag/ep10_fix/nestcol_nest_b
cmp -s $S/NESTCOL.json $W/NESTCOL_scene_copy.json && echo "scene NESTCOL.json == apply-time copy"
cp $W/NESTCOL_final.json $S/NESTCOL.json.tmp && mv -f $S/NESTCOL.json.tmp $S/NESTCOL.json
python3 -c "import json;d=json.load(open('$S/NESTCOL.json'));print('train_worthy',d['train_worthy'],d['gt_free_checks']['failed_checks'])"
md5sum $S/NESTCOL.json
ls -la $S
find $S -type f | wc -l
INNER
