"""Step 3 again with the carrier-wrist terms: ranks against the tax, LOO fits of at most two features, picks, 3.4 bars.

    python validate_wrist.py facts02.json out.json
"""
import sys, json, itertools
from pathlib import Path
import numpy as np
from scipy import stats
OUT = Path('/Users/jzzzzz79/V2D/NVIDIA_V2D_Egocentric/path_b/data/outputs')
VM = OUT / 'v4_h1_vm'
TWELVE = [40, 26, 18, 41, 39, 44, 27, 22, 13, 2, 42, 45]
rows = [r for r in json.load(open(sys.argv[1])) if 'error' not in r]
D = []
for r in rows:
    suffix = '_pre_k7' if r['folder'].endswith('_pre_k7') else ''
    f = json.loads((VM / f"episode_{r['episode']:06d}/s4b2_trackability{suffix}/{r['candidate']}/trackability.json").read_text())
    rss = f['rss']
    D.append(dict(ep=r['episode'], cand=r['candidate'], play=r['playback']['AUC'], roll=r['rollout']['AUC'], tax=r['tax'],
                  ums=f['ums'], ums_run=f['ums_run'], rss_flag=int(rss['rss0_cm'] > 1.0 or rss['rss_max_deg'] > 10.0),
                  wrist_p95=f['wrist_p95'], wrist_max=f['wrist_max'], wrist_fast=f['wrist_fast'],
                  wrist_acc_p95=f['wrist_acc_p95'], wrist_ang_p95=f['wrist_ang_p95'],
                  wrist_fast_any=int(f['wrist_fast'] > 0)))
eps = np.array([d['ep'] for d in D]); tax = np.array([d['tax'] for d in D]); roll = np.array([d['roll'] for d in D])
play = np.array([d['play'] for d in D]); inv = np.array([d['cand'] == 'inverted' for d in D])
W = ['wrist_p95', 'wrist_max', 'wrist_fast', 'wrist_fast_any', 'wrist_acc_p95', 'wrist_ang_p95']
print('ranks vs tax (rho_s): all 39 | inverted 19 | inverted with playback >= 0.6')
hi = inv & (play >= 0.6)
for k in W + ['ums', 'rss_flag']:
    x = np.array([d[k] for d in D], float)
    print(f'  {k:14} {stats.spearmanr(x, tax)[0]:+.3f} | {stats.spearmanr(x[inv], tax[inv])[0]:+.3f} | {stats.spearmanr(x[hi], tax[hi])[0]:+.3f}')

def loo(names):
    X = np.column_stack([np.ones(len(D))] + [np.array([d[k] for d in D], float) for k in names])
    pred = np.empty(len(D))
    for e in np.unique(eps):
        c, *_ = np.linalg.lstsq(X[eps != e], tax[eps != e], rcond=None)
        pred[eps == e] = X[eps == e] @ c
    c, *_ = np.linalg.lstsq(X, tax, rcond=None)
    return pred, c

def picks(pr, subset=None):
    right = total = 0
    for e in np.unique(eps):
        if subset is not None and e not in subset:
            continue
        idx = np.nonzero(eps == e)[0]
        if len(idx) > 1:
            total += 1
            right += int(np.argmax(pr[idx]) == np.argmax(roll[idx]))
    return right, total

res = {}
models = [[k] for k in W] + [[a, b] for a in ('ums', 'rss_flag', 'ums_run') for b in W] + [[a, b] for a, b in itertools.combinations(W, 2)]
for names in models:
    p, c = loo(names)
    pr = np.clip(play - p, 0, 1)
    res[' + '.join(names)] = dict(coef=[round(float(x), 4) for x in c], mae=float(np.abs(pr - roll).mean()),
                                  rho=float(stats.spearmanr(pr, roll)[0]), pick12=picks(pr, TWELVE), pick_all=picks(pr))
print(f'\n{len(res)} models (at most two features), LOO by episode; the 8 best by MAE:')
print('  model                              coefficients                 LOO MAE  rho_s   pick12  all')
for k, v in sorted(res.items(), key=lambda kv: kv[1]['mae'])[:8]:
    print(f"  {k:34} {str(v['coef']):28} {v['mae']:.3f}   {v['rho']:+.3f}   {v['pick12'][0]}/{v['pick12'][1]}   {v['pick_all'][0]}/{v['pick_all'][1]}")
print('  best by rho_s:', max(res.items(), key=lambda kv: kv[1]['rho'])[0], round(max(v['rho'] for v in res.values()), 3),
      '| best pick12:', max(res.items(), key=lambda kv: kv[1]['pick12'][0])[0], max(v['pick12'][0] for v in res.values()))
passing = [k for k, v in res.items() if v['rho'] >= 0.8 and v['mae'] <= 0.05 and v['pick12'][0] >= 11]
print('  3.4 pass (rho_s >= 0.8, MAE <= 0.05, pick >= 11/12):', passing or 'none')
print('  reference: ums + rss_flag 0.133 / +0.694 / 9 of 12; constant 0.146 / +0.658 / 9 of 12')
json.dump(res, open(sys.argv[2], 'w'), indent=1)
