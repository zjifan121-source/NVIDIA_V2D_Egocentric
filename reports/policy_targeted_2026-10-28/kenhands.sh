#!/bin/bash
P=/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/packed
Q=/tmp/kenqa_$$; mkdir -p $Q
python3 - $P $Q <<'X'
import json, os, sys
P, Q = sys.argv[1:3]
k = json.load(open(f'{P}/candidate_0880320.plan.json'))
for e in ['1','3','6','10','14','16','20','25','32','33']:
    os.symlink(k[e]['scene'], f'{Q}/ep{e}')
X
/data/user_data/edmondwa/v2d/venvs/retarget/bin/python /home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28/scene_qa.py $Q --out $Q/qa.json | cat
/data/user_data/edmondwa/v2d/venvs/retarget/bin/python -c "
import json; d=json.load(open('$Q/qa.json'))
for r in d['episodes']: print(r['episode'], {k:v.get('median_gap_cm') for k,v in r.get('objects',{}).items()})"
# which bundle each Ken scene came from, and whether its hands equal Wentao's
/data/user_data/edmondwa/v2d/venvs/retarget/bin/python - $Q <<'X'
import glob, json, sys, numpy as np
Q = sys.argv[1]
W = '/data/user_data/edmondwa/v2d/wentao_mix/hub/experiments/wentacc/scored20_alignment_mix_v1_20261030/scenes/episode_{:06d}/bundle/result.npz'
for e in [1, 3, 10, 20]:
    cands = glob.glob(f'{Q}/ep{e}/**/result.npz', recursive=True)
    print('ep', e, 'Ken scene bundles:', [c.split(f"ep{e}/")[-1] for c in cands][:3])
    if not cands: continue
    k = np.load(cands[0]); w = np.load(W.format(e))
    for s in ('left', 'right'):
        a, b = k[f'hand_{s}_finger_pose'], w[f'hand_{s}_finger_pose']
        same = a.shape == b.shape and np.allclose(a, b, atol=1e-4)
        ta, tb = k[f'hand_{s}_wrist_trans_in_camera'], w[f'hand_{s}_wrist_trans_in_camera']
        d = np.linalg.norm(ta - tb, axis=1) if ta.shape == tb.shape else None
        print('   ', s, 'fingers identical:', same, ' wrist-in-camera diff median cm:', None if d is None else round(float(np.median(d)) * 100, 2))
X
