"""Step 1: bridged / failed / unstable share of moving frames in the trained inverted reference vs F2's tax."""
import sys, json
from pathlib import Path
import numpy as np, trimesh
from scipy import stats
OUT = Path('/Users/jzzzzz79/V2D/NVIDIA_V2D_Egocentric/path_b/data/outputs')
rows = json.load(open(sys.argv[1]))

def labels(run, ep, obj, n):
    """Per frame: in_grasp, failed, unstable, bridged from S3c's invert_report."""
    rep = json.loads((OUT / run / f'episode_{ep:06d}/s3c_invert/{obj}/invert_report.json').read_text())
    lab = {k: np.zeros(n, bool) for k in ('grasp', 'failed', 'unstable', 'bridged')}
    for s in rep['segments']:
        a, b = int(s['start']), int(s['end']) + 1
        if s['kind'] == 'grasp':
            lab['grasp'][a:b] = True
            if s.get('failed'):
                lab['failed'][a:b] = True
        elif s['kind'] == 'unstable':
            lab['unstable'][a:b] = True
        if s.get('bridge'):
            lab['bridged'][int(s['bridge'][0]):int(s['bridge'][1]) + 1] = True
    return lab

def centroid_speed(run, ep, obj, cand):
    d = OUT / run / f'episode_{ep:06d}/s4b_reference/{obj}/{cand}'
    with np.load(d / 'result.npz') as z:
        T = z['object_to_world_transform'].astype(float); s = float(z['object_scale']) if 'object_scale' in z else 1.0
    m = trimesh.load(d / 'mesh.obj', force='mesh', process=False)
    c = np.asarray(m.vertices).mean(axis=0) * s
    cw = np.einsum('tij,j->ti', T[:, :3, :3], c) + T[:, :3, 3]
    v = np.r_[0.0, np.linalg.norm(np.diff(cw, axis=0), axis=1)]
    return v

out = []
for r in rows:
    if 'error' in r or r['candidate'] != 'inverted' or not r.get('playback_run'):
        continue
    run, ep = r['playback_run'], r['episode']
    tot = {'moving': 0, 'bridged': 0, 'failed': 0, 'unstable': 0, 'any': 0}
    for obj in r['objects']:
        v = centroid_speed(run, ep, obj, 'inverted')
        lab = labels(run, ep, obj, len(v))
        moving = (v > 0.01) | lab['grasp'] | lab['bridged']
        flagged = lab['bridged'] | lab['failed'] | lab['unstable']
        tot['moving'] += int(moving.sum())
        for k in ('bridged', 'failed', 'unstable'):
            tot[k] += int((moving & lab[k]).sum())
        tot['any'] += int((moving & flagged).sum())
    m = max(tot['moving'], 1)
    out.append(dict(episode=ep, folder=r['folder'], run=run, k7=r['is_k7'], tax=r['tax'], play=r['playback']['AUC'],
                    roll=r['rollout']['AUC'], n_moving=tot['moving'],
                    **{f'{k}_share': tot[k] / m for k in ('bridged', 'failed', 'unstable', 'any')}))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(f"{'ep':>3} {'run':7} {'tax':>6} {'moving':>6} {'bridged':>7} {'failed':>6} {'unstab':>6} {'any':>5}")
for o in sorted(out, key=lambda o: o['tax']):
    print(f"{o['episode']:>3} {o['run']:7} {o['tax']:6.3f} {o['n_moving']:6d} {o['bridged_share']:7.2f} {o['failed_share']:6.2f} {o['unstable_share']:6.2f} {o['any_share']:5.2f}")
tax = np.array([o['tax'] for o in out])
for k in ('bridged', 'failed', 'unstable', 'any'):
    x = np.array([o[f'{k}_share'] for o in out])
    rs, ps = stats.spearmanr(x, tax); kt, pk = stats.kendalltau(x, tax)
    print(f'{k:9} n={len(x)} spearman {rs:+.3f} (p {ps:.3f})  kendall {kt:+.3f} (p {pk:.3f})')
