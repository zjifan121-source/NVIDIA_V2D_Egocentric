"""Step 3: features vs F2's rollouts on every pair with the exact trained reference; LOO tax fits; pick test; six metrics."""
import sys, json
from pathlib import Path
REPO = Path('/Users/jzzzzz79/V2D/NVIDIA_V2D_Egocentric')
sys.path.insert(0, str(REPO / 'path_b'))
import numpy as np
from scipy import stats
from pathb import trackability as T
OUT = REPO / 'path_b/data/outputs'
VM = OUT / 'v4_h1_vm'
rows = [r for r in json.load(open(sys.argv[1])) if 'error' not in r]
TWELVE = [40, 26, 18, 41, 39, 44, 27, 22, 13, 2, 42, 45]  # F2 rollouts 1-12 (BUILD_LOG 2026-10-29 G2 / F2 / G3 rows)

def bridged_share(r, f):
    """Step 1's share (bridged / failed / unstable of moving frames) on this candidate's own S5 motion."""
    if r['candidate'] == 'still':
        return 0.0
    suffix = '_pre_k7' if r['folder'].endswith('_pre_k7') else ''
    t = T.task(VM / f"episode_{r['episode']:06d}/s5_task{suffix}/{r['candidate']}")
    n = len(t['cells']['object_body_position'])
    mov = flag = 0
    for o in T.objects(t):
        rep = OUT / r['playback_run'] / f"episode_{r['episode']:06d}/s3c_invert/{o['name']}/invert_report.json"
        lab = T.labels(json.loads(rep.read_text()) if rep.is_file() else None, n)
        m = T.moving_frames(T.load_mesh(o['mesh']), o['poses'], lab)
        mov += int(m.sum()); flag += int((m & (lab['bridged'] | lab['failed'] | lab['unstable'])).sum())
    return flag / max(mov, 1)

data = []
for r in rows:
    suffix = '_pre_k7' if r['folder'].endswith('_pre_k7') else ''
    f = json.loads((VM / f"episode_{r['episode']:06d}/s4b2_trackability{suffix}/{r['candidate']}/trackability.json").read_text())
    rss = f['rss']
    data.append(dict(ep=r['episode'], cand=r['candidate'], play=r['playback'], roll=r['rollout'], tax=r['tax'],
                     ums=f['ums'], ums_opp=f['ums_opp'], ums_run=f['ums_run'], d_hand=f['d_hand_cm'] or 0.0,
                     rcc1=f['rcc1'] if f['rcc1'] is not None else 1.0, rcc2=f['rcc2'] if f['rcc2'] is not None else 1.0,
                     rss0_cm=rss['rss0_cm'], rss0_deg=rss['rss0_deg'], rss_max_cm=rss['rss_max_cm'], rss_max_deg=rss['rss_max_deg'],
                     unsettled=rss['unsettled_rests'], support=int(f['support_ok']), bridged=bridged_share(r, f),
                     rss_flag=int(rss['rss0_cm'] > 1.0 or rss['rss_max_deg'] > 10.0), rests=rss['rests'], objects=f['objects']))
json.dump(data, open(sys.argv[2], 'w'), indent=1, default=float)
tax = np.array([d['tax'] for d in data]); roll = np.array([d['roll']['AUC'] for d in data]); play = np.array([d['play']['AUC'] for d in data])
eps = np.array([d['ep'] for d in data])
print(f"pairs {len(data)} ({sum(d['cand']=='inverted' for d in data)} inverted, {sum(d['cand']=='keyframe' for d in data)} keyframe, {sum(d['cand']=='still' for d in data)} still)")
print('3.1 feature            vs rollout AUC (rho_s, tau)   vs tax (rho_s, tau)')
feats = ['ums', 'ums_opp', 'ums_run', 'd_hand', 'rcc1', 'rcc2', 'rss0_cm', 'rss0_deg', 'rss_max_cm', 'rss_max_deg', 'unsettled', 'rss_flag', 'support', 'bridged']
for k in feats:
    x = np.array([d[k] for d in data], float)
    a = stats.spearmanr(x, roll)[0], stats.kendalltau(x, roll)[0]
    b = stats.spearmanr(x, tax)[0], stats.kendalltau(x, tax)[0]
    print(f'    {k:12} {a[0]:+.3f} {a[1]:+.3f}            {b[0]:+.3f} {b[1]:+.3f}')
inv = {d['ep']: d for d in data if d['cand'] == 'inverted'}
print('    UMS of the inverted pairs:', {e: round(inv[e]['ums'], 2) for e in [42, 45, 22, 27, 2, 40, 41, 44, 26]})
print('    RCC1 of the inverted pairs:', {e: round(inv[e]['rcc1'], 2) for e in [42, 45, 22, 27, 2, 40, 41, 44, 26]})
hi, lo = [42, 45, 22, 27, 2], [40, 41, 44, 26]
print('    UMS ranks 42 45 22 27 2 above 40 41 44 26:', min(inv[e]['ums'] for e in hi) > max(inv[e]['ums'] for e in lo),
      '| RCC1 below:', max(inv[e]['rcc1'] for e in hi) < min(inv[e]['rcc1'] for e in lo))
for d in data:
    if d['ep'] in (18, 26, 40, 41, 44):
        worst = max(d['rests'], key=lambda r: (r['moved_cm'], r['turned_deg'])) if d['rests'] else {}
        print(f"    RSS ep{d['ep']} {d['cand']:8} flag={d['rss_flag']} rss0 {d['rss0_cm']:.2f} cm {d['rss0_deg']:.1f} deg, max {d['rss_max_cm']:.2f} cm {d['rss_max_deg']:.1f} deg, worst {worst}")

def design(names):
    return np.column_stack([np.ones(len(data))] + [np.array([d[k] for d in data], float) for k in names])

def loo(names):
    X = design(names); pred = np.empty(len(data))
    for e in np.unique(eps):
        tr, te = eps != e, eps == e
        c, *_ = np.linalg.lstsq(X[tr], tax[tr], rcond=None)
        pred[te] = X[te] @ c
    c_all, *_ = np.linalg.lstsq(X, tax, rcond=None)
    return pred, c_all

def picks(pred_roll, subset=None):
    right = total = 0
    for e in np.unique(eps):
        if subset is not None and e not in subset:
            continue
        idx = np.nonzero(eps == e)[0]
        if len(idx) < 2:
            continue
        total += 1
        right += int(np.argmax(pred_roll[idx]) == np.argmax(roll[idx]))
    return right, total

print('3.2 fit                          coefficients (all pairs)            LOO MAE   rho_s(pred, rollout)  pick 12   pick all')
res = {}
for label, names in [('ums + rss_flag', ['ums', 'rss_flag']), ('ums alone', ['ums']), ('bridged alone', ['bridged']),
                     ('rss_flag alone', ['rss_flag']), ('ums_opp + rss_flag', ['ums_opp', 'rss_flag']), ('rcc1 + rss_flag', ['rcc1', 'rss_flag']),
                     ('d_hand + rss_flag', ['d_hand', 'rss_flag']), ('constant', [])]:
    pred, c = loo(names)
    pr = np.clip(play - pred, 0, 1)
    mae = float(np.abs(pr - roll).mean()); rho = stats.spearmanr(pr, roll)[0]
    p12, p_all = picks(pr, TWELVE), picks(pr)
    res[label] = dict(coef=c.tolist(), loo_mae=mae, rho=rho, pick12=p12, pick_all=p_all)
    print(f'    {label:22} {np.round(c, 3).tolist()!s:38} {mae:.3f}     {rho:+.3f}               {p12[0]}/{p12[1]}     {p_all[0]}/{p_all[1]}')
pl = picks(play)
print(f'    reference: playback alone picks {picks(play, TWELVE)} on 12, {pl} on all; the GT-free observation rule (F2 rows): 11 of 12, 17 of 19')
json.dump(res, open(sys.argv[3], 'w'), indent=1)

# 3.1b RSS against the rollout's own early drift: each object's max position / rotation error against its reference
# over the first 2 s (40 frames) of F2's rollout, worst object per pair (rests the policy meets before any carry).
import pandas as pd
from scipy.spatial.transform import Rotation as Rot
drift = []
for r, d in zip(rows, data):
    suffix = '_pre_k7' if r['folder'].endswith('_pre_k7') else ''
    t = pd.read_parquet(VM / f"episode_{r['episode']:06d}/s6_eval{suffix}/{r['candidate']}/results/episode_{r['episode']:06d}.parquet")
    t = t[(t.environment_id == 0) & (t.reference_frame <= 40)]
    e = np.linalg.norm(t[[f'achieved_position_{a}' for a in 'xyz']].to_numpy() - t[[f'reference_position_{a}' for a in 'xyz']].to_numpy(), axis=1)
    ang = np.degrees((Rot.from_quat(t[[f'achieved_quaternion_{a}' for a in 'xyzw']].to_numpy())
                      * Rot.from_quat(t[[f'reference_quaternion_{a}' for a in 'xyzw']].to_numpy()).inv()).magnitude())
    drift.append((100 * e.max(), ang.max()))
dc, dd = np.array(drift).T
early = (dc > 1.0) | (dd > 10.0)
flag = np.array([d['rss_flag'] for d in data], bool)
r0 = np.array([d['rss0_cm'] for d in data]); r0d = np.array([d['rss0_deg'] for d in data])
print('3.1b RSS vs the rollout drift in its first 2 s (worst object):')
print(f'    rho_s(rss0_cm, drift cm) {stats.spearmanr(r0, dc)[0]:+.3f}   rho_s(rss0_deg, drift deg) {stats.spearmanr(r0d, dd)[0]:+.3f}')
f0 = np.array([d['rss0_cm'] > 1.0 or d['rss0_deg'] > 10.0 for d in data])
print(f'    frame-0 flag (rss0 > 1 cm or > 10 deg) vs drift > 1 cm or > 10 deg: agree {int((f0 == early).sum())}/{len(data)}; '
      f'flag & no drift {int((f0 & ~early).sum())}, drift & no flag {int((~f0 & early).sum())}')
for r, d, a, b in zip(rows, data, dc, dd):
    print(f"    ep{d['ep']:>2} {d['cand']:8} rss0 {d['rss0_cm']:6.1f} cm {d['rss0_deg']:5.1f} deg | rollout 0-2 s {a:5.1f} cm {b:5.1f} deg")

# 3.3 predicted six metrics from the main fit's LOO tax: AUC = playback - tax; MP-SR 0 if rss fails or ums > c_ums;
# SP-SR 0 if ums_run > 60 frames; MPPE and RPE grow as AUC falls (playback / (1 - tax / playback)); CD-O kept.
pred, _ = loo(['ums', 'rss_flag'])
ratio = np.clip(1 - pred / np.maximum(play, 1e-6), 0.05, None)
ums = np.array([d['ums'] for d in data]); run = np.array([d['ums_run'] for d in data]); rf = np.array([d['rss_flag'] for d in data])
mp_play = np.array([d['play']['MP-SR'] for d in data]); mp_roll = np.array([d['roll']['MP-SR'] for d in data])
best_c = max(np.arange(0.0, 0.61, 0.05), key=lambda c: (np.where(rf.astype(bool) | (ums > c), 0, mp_play) == mp_roll).mean())
P = {'AUC': np.clip(play - pred, 0, 1),
     'SP-SR': np.where(run > 60, 0, np.array([d['play']['SP-SR'] for d in data])),
     'MP-SR': np.where(rf.astype(bool) | (ums > best_c), 0, mp_play),
     'MPPE cm': np.array([d['play']['MPPE cm'] for d in data]) / ratio,
     'RPE cm': np.array([d['play']['RPE cm'] for d in data]) / ratio,
     'CD-O cm': np.array([d['play']['CD-O cm'] for d in data])}
print(f'3.3 predicted six metrics (main fit, LOO; c_ums = {best_c:.2f} chosen in-sample for MP-SR):')
for k, v in P.items():
    actual = np.array([d['roll'][k] for d in data]); base = np.array([d['play'][k] for d in data])
    if k in ('SP-SR', 'MP-SR'):
        print(f'    {k:8} agree {int((v == actual).sum())}/{len(v)} (playback as the prediction: {int((base == actual).sum())}/{len(v)})')
    else:
        print(f'    {k:8} MAE {np.abs(v - actual).mean():.3f} (playback as the prediction: {np.abs(base - actual).mean():.3f}); rho_s {stats.spearmanr(v, actual)[0]:+.3f}')

# 3.4 verdict
main = res['ums + rss_flag']
bars = {'rho_s >= 0.8': main['rho'] >= 0.8, 'LOO MAE <= 0.05': main['loo_mae'] <= 0.05, 'pick >= 11 of 12': main['pick12'][0] >= 11}
print('3.4 verdict (ums + rss_flag):', {k: bool(v) for k, v in bars.items()}, '=> ACCEPT' if all(bars.values()) else '=> REJECT')
singles = {k: v for k, v in res.items() if k in ('ums alone', 'bridged alone', 'rss_flag alone')}
for k in feats:
    if k in ('ums', 'bridged', 'rss_flag'):
        continue
    p1, _ = loo([k]); pr1 = np.clip(play - p1, 0, 1)
    singles[f'{k} alone'] = dict(loo_mae=float(np.abs(pr1 - roll).mean()), rho=stats.spearmanr(pr1, roll)[0], pick12=picks(pr1, TWELVE), pick_all=picks(pr1))
best = min(singles.items(), key=lambda kv: kv[1]['loo_mae'])
print(f"    best single feature by LOO MAE: {best[0]} (MAE {best[1]['loo_mae']:.3f}, rho_s {best[1]['rho']:+.3f}, pick {best[1]['pick12']} / {best[1]['pick_all']})")
for k, v in sorted(singles.items(), key=lambda kv: kv[1]['loo_mae']):
    print(f"      {k:20} MAE {v['loo_mae']:.3f} rho_s {v['rho']:+.3f} pick12 {v['pick12'][0]}/{v['pick12'][1]} all {v['pick_all'][0]}/{v['pick_all'][1]}")
