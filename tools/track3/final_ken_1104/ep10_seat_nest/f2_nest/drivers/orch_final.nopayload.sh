# RELEASE NOTE: copy of orch_final.sh with the base64 payload lines removed (heredoc left empty); see ../../README.md
set -e
W=/tmp/haisenk_v2d/nestcol_work
test ! -e $W/final_b; mkdir -p $W/final_b; cd $W/final_b
base64 -d > in.tgz <<'B64EOF'
B64EOF
md5sum in.tgz; tar xzf in.tgz && rm -f in.tgz
export PYTHONPATH=/tmp/haisenk_v2d/fc_src_de66
export WARP_CACHE_PATH=$W/warpcache XDG_CACHE_HOME=$W/xdg
unset FLASH_CHORD_CACHE_DIR
PY=/tmp/haisenk_v2d/v2d_train/baseline/robotic_grounding/flash_chord/.venv/bin/python
$PY fc_hulls.py out_beige beige_test.urdf > beige_run.log 2>&1 || echo "FAILED beige $?"
$PY fc_hulls.py out_blue --off_compare --info_log blue_col_180.stl blue_test.urdf > blue_run.log 2>&1 || echo "FAILED blue $?"
grep -h -i -E "manifold|FC_HULLS|Error|Traceback" beige_run.log blue_run.log | cut -c1-300 | head -20
$PY - <<'PYEOF'
import json
for nm in ('out_beige.json', 'out_blue.json'):
    d = json.load(open(nm)); P = d['pieces']
    print(nm, 'pieces', len(P), 'hits_before', sum(p['cache_hit_before'] for p in P), 'n_hulls', sorted(set(p['n_hulls'] for p in P)) if len(P) > 1 else P[0]['n_hulls'],
          'max bbox diff mm', max(p['bbox_diff_mm'] for p in P), 'auto==off', [p.get('auto_equals_off') for p in P][:1] if len(P) == 1 else all(p.get('auto_equals_off') for p in P),
          'keys', [p['key'] for p in P][:2], 'mass', d['body_mass'], 'cache', d['cache_dir'])
PYEOF
tar czf out_final.tgz out_beige.json out_beige.npz out_blue.json out_blue.npz beige_run.log blue_run.log
