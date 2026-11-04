"""Rescore tasks for chosen episodes of a pack plan (rescore.py run format).
python bob_rescore_tasks.py PLAN OUT_TASKS EP:REPS[,EP:REPS...] [FIRST_REP]"""
import json, re, sys
from pathlib import Path
DIAG = Path('/data/user_data/edmondwa/v2d/flash_runs/diag')
plan = json.loads(Path(sys.argv[1]).read_text()); want = dict(x.split(':') for x in sys.argv[3].split(','))
first = int(sys.argv[4]) if len(sys.argv) > 4 else 1
tasks = []
for ep, reps in want.items():
    e = plan[ep]; ck = Path(e['run']) / 'checkpoints.json'
    pol = Path(json.loads(ck.read_text())[ep]) if ck.exists() else None
    if pol is None or not pol.is_file():
        print(f'ep{ep}: no policy ({ck})'); continue
    parent = e['name'].split('/')[0].removesuffix('_RS'); step = int(re.search(r'policy_(\d+)', pol.name)[1])
    for rep in range(first, first + int(reps)):
        tasks.append(dict(episode=int(ep), policy=str(pol), baseline=e['tracking'], out=str(DIAG / f'{parent}_RS' / f'early_{step}_r{rep}')))
Path(sys.argv[2]).write_text(json.dumps(tasks, indent=1)); print(len(tasks), 'tasks ->', sys.argv[2])
