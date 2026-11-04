#!/bin/bash
H=/home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28
PY=/data/user_data/edmondwa/v2d/venvs/retarget/bin/python
S=/data/user_data/edmondwa/v2d/wentao_mix/scenes
echo "### best Wentao-scene run per episode (per object)"
python3 - <<'X'
import json, glob, os, re
H = '/home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28'
D = '/data/user_data/edmondwa/v2d/flash_runs/diag'
eps = {e['episode']: [o['name'] for o in e['objects']] for e in json.load(open('/home/edmondwa/NVIDIA_V2D_Egocentric/tools/track3/episodes.json'))['episodes']}
best = {}
for tf in glob.glob(f'{H}/*_trial.json'):
    try: t = json.load(open(tf))
    except Exception: continue
    if 'wentao' not in t.get('scene', '') or 'wentao_fix' in t.get('scene', ''): continue
    log = f"{D}/{t['name']}/checkpoint_evaluations.log"
    if not os.path.exists(log): continue
    rows = [json.loads(l) for l in open(log) if l.startswith('{')]
    if not rows: continue
    b = max(rows, key=lambda r: r['auc']); ep = t['episode']
    if b['auc'] > best.get(ep, {'auc': -1})['auc']: best[ep] = dict(b, name=t['name'], scene=t['scene'].split('/v2d/')[-1])
for ep in sorted(best):
    b = best[ep]; print(f"ep{ep:>3} {b['auc']:.3f}  " + '  '.join(f"{n} {v:.2f}" for n, v in zip(eps[ep], b['per_body'])) + f"   [{b['name']} | {b['scene']}]")
X
echo "### fingertip gap / tremor on Wentao's scenes"
mkdir -p /tmp/wq_$$ && $PY $H/scene_qa.py $S --out /tmp/wq_$$/qa.json > /dev/null 2>&1
$PY -c "
import json; d=json.load(open('/tmp/wq_$$/qa.json'))
for r in d['episodes']: print(r['episode'], r['status'], {k: (v.get('median_gap_cm'), v.get('tremor_deg_per_frame2')) for k,v in r.get('objects',{}).items()})"
echo "### contact labels while lifted (Ken's audit)"
for e in 1 5 6 8 14 15 16 25 32 33 35 36 37; do $PY /home/edmondwa/NVIDIA_V2D_Egocentric/tools/track3/hand_fixes/bench/contact_audit.py $S/ep$e 2>&1 | grep -E "=====|lifted" ; done
