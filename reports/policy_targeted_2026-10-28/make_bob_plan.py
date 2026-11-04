"""Best-of-both pack plan: per episode, the policy with the highest expected Kaggle AUC = ratio(recon) x own tracking.

Ratios come from whole-submission Kaggle results (no per-episode leaderboard probing): Ken-recon plan 0.711/0.880 = 0.81,
Wentao-only 0.589/0.686 = 0.86. Candidates: the submitted Ken plan (candidate_0880320), Ken's 2 Nov hand-fix runs (Ken
recon), every finished run on a Wentao or Ken scene (Path B excluded: no Kaggle ratio yet), and rescored repeats.
python make_bob_plan.py OUT_PLAN [--exclude-sources ...]"""
import glob, json, os, sys
from pathlib import Path
H = Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28')
P = Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/packed')
D = '/data/user_data/edmondwa/v2d/flash_runs/diag'
KF = '/data/user_data/edmondwa/v2d/ken_handfix/experiments/haisenk/hand_fixes_v1_20261102'
RATIO = {'ken': 0.81, 'wentao': 0.86}
cands = {}
def add(ep, src, track, entry):
    cands.setdefault(ep, []).append(dict(src=src, tracking=track, expected=RATIO[src] * track, entry=entry))
for ep, e in json.load(open(P / 'candidate_0880320.plan.json')).items():
    src = 'wentao' if 'wentao' in e['scene'] else 'ken'
    add(int(ep), src, e['tracking'], e)
summary = json.load(open(f'{KF}/SUMMARY.json'))['episodes']
for key, s in summary.items():
    ep = int(key.split('_')[1]); run = s['recommended_run'].split('/')[-1]
    add(ep, 'ken', s['runs'][run], dict(run=f'{KF}/{key}/runs/{run}', scene=f'{KF}/{key}/scene', kind='policy',
                                         tracking=s['runs'][run], name=f'haisenk/hand_fixes_v1/{run}'))
for tf in glob.glob(f'{H}/*_trial.json'):
    try: t = json.load(open(tf))
    except Exception: continue
    if 'pathb' in t.get('scene', '') or not t.get('scene'): continue
    src = 'wentao' if ('wentao' in t['scene'] or '/campaigns/wfix/' in t['scene']) else 'ken'
    log = f"{D}/{t['name']}/checkpoint_evaluations.log"
    if not os.path.exists(log) or not os.path.exists(f"{D}/{t['name']}/train.done"): continue
    rows = [json.loads(l) for l in open(log) if l.startswith('{')]
    if not rows: continue
    b = max(rows, key=lambda r: r['auc']); early = b.get('run') or f"{D}/{t['name']}/early_{b['step']}"
    if not os.path.exists(f"{early}/results"): continue
    add(t['episode'], src, b['auc'], dict(run=early, scene=t['scene'], kind='policy', tracking=b['auc'],
                                         name=f"{t['name']}/{os.path.basename(early)}"))
# Ken 3 Nov hand fixes on Wentao fb11 bundle (policies shipped for ep3/10/20); scenes imported by import_scene.py
KW = '/data/user_data/edmondwa/v2d/wentao_sprint/dl_55846863/experiments/haisenk/hand_fixes_wentao_v1_20261103'
for res in glob.glob(f'{KW}/episode_*/runs/*/results/episode_*.json'):
    ep = int(res.split('/episode_')[1][:6]); run = os.path.dirname(os.path.dirname(res))
    scene = f'/data/user_data/edmondwa/v2d/campaigns/wentao3/scenes/ep{ep}_kw'
    if not glob.glob(f'{run}/results/episode_{ep:06d}.parquet') or not os.path.isdir(f'{scene}/score_input/meshes'): continue
    auc = json.load(open(res))['metrics']['add_auc']
    add(ep, 'wentao', auc, dict(run=run, scene=scene, kind='policy', tracking=auc, name=f'haisenk_w/{os.path.basename(run)}'))
# rescored repeats (rescore.py: diag/<parent>_RS/early_<step>_r<rep>) of every candidate, same scene and source
for ep, cs in list(cands.items()):
    for c in list(cs):
        parent = c['entry']['name'].split('/')[0].removesuffix('_RS')
        for res in glob.glob(f"{D}/{parent}_RS/early_*_r*/results/episode_{ep:06d}.json"):
            auc = json.load(open(res))['metrics']['add_auc']; run = os.path.dirname(os.path.dirname(res))
            if not any(x['entry']['run'] == run for x in cs):
                add(ep, c['src'], auc, dict(c['entry'], run=run, tracking=auc, name=f"{parent}_RS/{os.path.basename(run)}"))
plan, report = {}, []
for ep in sorted(cands):
    best = max(cands[ep], key=lambda c: c['expected'])
    plan[str(ep)] = dict(best['entry'], tracking=best['tracking'], source=best['src'])
    alt = {s: max((c['tracking'] for c in cands[ep] if c['src'] == s), default=None) for s in RATIO}
    report.append(f"ep{ep:>3} {best['src']:6} track {best['tracking']:.3f} exp {best['expected']:.3f}   best ken {alt['ken']} wentao {alt['wentao']}  [{best['entry']['name']}]")
Path(sys.argv[1]).write_text(json.dumps(plan, indent=1))
print('\n'.join(report))
n = len(plan); print(f"episodes {n}; mean tracking {sum(v['tracking'] for v in plan.values())/n:.4f}; "
                     f"expected Kaggle {sum(RATIO[v['source']]*v['tracking'] for v in plan.values())/n:.4f}")
