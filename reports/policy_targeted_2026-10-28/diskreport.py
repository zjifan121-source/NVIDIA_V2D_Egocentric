"""Disk report for /data/user_data/edmondwa/v2d/flash_runs/diag: what pruning state_/replay_ files would free."""
import json, subprocess
from pathlib import Path
DIAG = Path('/data/user_data/edmondwa/v2d/flash_runs/diag')
plan = json.load(open('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/plan.json'))
selected = {Path(v['run']).name if 'early_' not in v['run'] else Path(v['run']).parent.name for v in plan.values()}
running = set(subprocess.run("squeue -h -u edmondwa,sarveshp -o %j", shell=True, capture_output=True, text=True).stdout.split())
G = 1024 ** 3
rows, tot = [], dict(keep=0, prune=0, other=0)
for run in sorted(DIAG.iterdir()):
    if not run.is_dir():
        continue
    big = [f for f in (run / 'train').glob('*') if f.name.startswith(('state_', 'replay_'))] if (run / 'train').is_dir() else []
    size = sum(f.stat().st_size for f in big)
    total = sum(f.stat().st_size for f in run.rglob('*') if f.is_file())
    if run.name in selected or run.name in running:
        why = 'selected' if run.name in selected else 'running'; tot['keep'] += size
    elif (run / 'train.done').exists() or not (run / 'train').is_dir():
        why = 'prunable'; tot['prune'] += size
    else:
        why = 'unfinished'; tot['other'] += size
    tot.setdefault('all', 0); tot['all'] += total
    rows.append((size / G, total / G, why, run.name))
for s, t, why, name in sorted(rows, reverse=True)[:30]:
    print(f'{s:6.1f}G state+replay {t:6.1f}G total  {why:10s} {name}')
print({k: f'{v / G:.1f}G' for k, v in tot.items()}, 'runs', len(rows))
