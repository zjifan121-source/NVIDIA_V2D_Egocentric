"""Wentao hill-climb: more seeds on the weak Wentao episodes, keeping late learner states for best-checkpoint warm starts.

python make_whc_trials.py SEED[,SEED...] [EP,EP...]
"""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
RS = {1, 3, 8, 14}  # smoothed object rotation scored best on these
EPS = [int(e) for e in sys.argv[2].split(',')] if len(sys.argv) > 2 else [33, 32, 6, 10, 20, 1, 3, 16, 25, 5, 14, 36, 37]
for ep in EPS:
    src = H / (f'ep{ep}_wrs_s90g_trial.json' if ep in RS else f'ep{ep}_wmix_s80g_trial.json')
    if not src.exists():
        src = sorted(H.glob(f'ep{ep}_wmix_s8*_trial.json'))[0]
    base = json.loads(src.read_text())
    for seed in map(int, sys.argv[1].split(',')):
        t = dict(base, name=f'ep{ep}_WHC_s{seed}', seed=seed, launcher='fresh_keep.sbatch', save_replay=False,
                 account='edmondwa', partition='preempt', qos='preempt_qos',
                 hypothesis=f'Wentao hill-climb seed on the {src.name.split("_s")[0]} scene; late states kept for low-LR continuation.')
        t.pop('slurm_account', None)
        (H / f'ep{ep}_whc_s{seed}g_trial.json').write_text(json.dumps(t, indent=2))
        print(t['name'], '/'.join(t['scene'].split('/')[-2:]), t['memory'], t['launcher'])
