"""For finished runs that keep learner states (keep/ or base/): keep only the learner state at the run's best checkpoint (in keep/ or train/),
and drop base/ start-state copies. Runs still training or not yet evaluated are left alone. Prints what it frees."""
import json, os, re, sys
from pathlib import Path
D = Path(os.environ.get('V2D_DATA', '/data/user_data/edmondwa/v2d')) / 'flash_runs/diag'
queued = set(Path(sys.argv[1]).read_text().split())
freed = n = 0
for run in D.iterdir():
    # only runs that keep states for continuation: keep/ (fresh_keep, refine_x_keep) or a copied start state in base/
    kept = (run / 'keep').is_dir() or (run / 'base').is_dir()
    if not kept or not re.match(r'ep\d+_', run.name) or run.name in queued or not (run / 'train.done').exists():
        continue
    log = run / 'checkpoint_evaluations.log'
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.startswith('{')] if log.exists() else []
    if not rows:
        continue
    best = max(rows, key=lambda r: r['auc'])['step']
    doomed = [p for d in ('keep', 'train') for p in (run / d).glob('state_*.safetensors') if int(p.stem.split('_')[1]) != best]
    doomed += list((run / 'base').glob('state_*.safetensors'))
    for p in doomed:
        freed += p.stat().st_size if p.stat().st_nlink == 1 else 0
        p.unlink(); n += 1
print(f'deleted {n} files, freed about {freed / 1e9:.1f} GB')
