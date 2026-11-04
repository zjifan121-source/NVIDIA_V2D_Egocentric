"""Build, gate, pack and merge a single-episode probe on the A6000. bob1 plus one rollout, gated by check_probe_plan.py against bob1, then merged onto the current best base with merge_swaps.py.
usage: make_probe.py <name> <episode> <evaluated run dir holding results/> <scene dir>
exit 0 only if the gate passed and the file was packed; prints the gate summary and the expected Kaggle delta."""
import json, subprocess, sys
from pathlib import Path
HERE = Path('/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe'); KIT = '/mnt/v2d/outputs/experiments_20261021/independent_pipeline_v1/packet/submission_kit'; PY = str(HERE / 'venv-kit/bin/python')
COMMIT = 'https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/f377f3b235f4ba8188d94f0025090ea65ab85ffc'
name, ep, run, scene = sys.argv[1:5]
out = HERE / 'out' / name; out.mkdir(parents=True, exist_ok=True)
ref_path = HERE / 'bob1_a6000.plan.json'
if not ref_path.exists():
    raw = json.loads((HERE / 'candidate_bob1.plan.json').read_text())
    base = HERE / 'probe_inputs/experiments/edmond/bob1_submission_20261103'
    ref = {}
    for k, v in raw.items():
        d = base / f'episode_{int(k):06d}'
        ref[k] = dict(run=str(d / 'policy'), scene=str(d / 'scene'), kind=v.get('kind', 'policy'), tracking=json.loads((d / f'policy/results/episode_{int(k):06d}.json').read_text())['metrics']['add_auc'], name=v.get('name'), source=v.get('source'))
    ref_path.write_text(json.dumps(ref, indent=1))
ref = json.loads(ref_path.read_text())
res = json.loads((Path(run) / 'results' / f'episode_{int(ep):06d}.json').read_text())
plan = dict(ref); plan[str(int(ep))] = dict(run=run, scene=scene, kind='policy', tracking=res['metrics']['add_auc'], name=name, source='wentao')
(out / 'plan.json').write_text(json.dumps(plan, indent=1))
r = subprocess.run([PY, str(HERE / 'check_probe_plan.py'), str(out / 'plan.json'), '--ref', str(ref_path), '--expect-changes', '1', '--code-commit', COMMIT, '--roster', KIT + '/data/track_3_evaluation_objects.json', '--out', str(out / 'check.json')], capture_output=True, text=True)
print(r.stdout[-900:], r.stderr[-400:])
if r.returncode:
    print('GATE_FAILED'); raise SystemExit(1)
p = subprocess.run([PY, str(HERE / 'track3_pack.py'), '--plan', str(out / 'plan.json'), '--work', str(out / 'work'), '--code-commit', COMMIT, '--python', PY, '-o', str(out / f'{name}.parquet')], capture_output=True, text=True)
print(p.stdout[-300:], p.stderr[-300:])
if p.returncode: raise SystemExit(p.returncode)
merged = out / f'{name}_on_best.parquet'
if merged.exists(): merged.unlink()
swaps = {1: HERE / "base_t2.parquet", 3: HERE / "base_t5.parquet", 5: HERE / "base_t8.parquet", 8: HERE / "base_t6.parquet"}; swaps[int(ep)] = out / (name + '.parquet')  # incumbents in the current base; a probe on one of these episodes replaces it
m = subprocess.run([PY, str(HERE / 'merge_swaps.py'), '--base', str(HERE / 'base_bob1.parquet')] + sum([['--swap', f'{k}={v}'] for k, v in swaps.items()], []) + ['--out', str(merged), '--code-commit', COMMIT], capture_output=True, text=True)
print(m.stdout[-500:], m.stderr[-500:])
raise SystemExit(m.returncode)
