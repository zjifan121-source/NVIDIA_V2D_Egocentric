# RELEASE NOTE: copy of orch_final_urdf.sh with the base64 payload lines removed (heredoc left empty); see ../../README.md
set -e
W=/tmp/haisenk_v2d/nestcol_work
test ! -e $W/final_b_urdf; mkdir -p $W/final_b_urdf; cd $W/final_b_urdf
base64 -d > in.tgz <<'B64EOF'
B64EOF
base64 -d > fc_hulls.py <<'B64EOF'
B64EOF
tar xzf in.tgz && rm -f in.tgz && ls | wc -l
export PYTHONPATH=/tmp/haisenk_v2d/fc_src_de66
export WARP_CACHE_PATH=$W/warpcache XDG_CACHE_HOME=$W/xdg
unset FLASH_CHORD_CACHE_DIR
PY=/tmp/haisenk_v2d/v2d_train/baseline/robotic_grounding/flash_chord/.venv/bin/python
$PY fc_hulls.py out_urdf --no_decompose_uncached ep10_h5_k10_ep10_nest_b_blue_cup_rigid.urdf ep10_h5_k10_ep10_nest_b_beige_cup_rigid.urdf > urdf_run.log 2>&1 || echo "FAILED $?"
grep -E "FC_HULLS|Error|Traceback" urdf_run.log | head
$PY - <<'PYEOF'
import json
from collections import Counter
d = json.load(open('out_urdf.json')); P = d['pieces']
print('pieces', len(P), dict(Counter(p['urdf'] for p in P)), 'cache_hits', sum(p['cache_hit_before'] for p in P), 'skipped', sum(bool(p.get('skipped')) for p in P),
      'n_hulls', dict(Counter(p.get('n_hulls') for p in P)), 'mass', d['body_mass'])
PYEOF
cp out_urdf.json /dev/stdout >/dev/null
