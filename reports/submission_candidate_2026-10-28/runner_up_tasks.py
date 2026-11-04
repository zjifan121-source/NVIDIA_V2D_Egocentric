"""Tasks for rescore.py: re-run the top runner-up checkpoints of weak episodes.

python runner_up_tasks.py OUT.json [TOP] [REPS] [MAX_AUC]
"""
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAG = Path('/data/user_data/edmondwa/v2d/flash_runs/diag')
BAD = ('still', 'half', 'oracle', 'scan', '_uw', '_rs')


def main(out, top=3, reps=3, max_auc=0.9):
    plan = json.loads((HERE / 'plan.json').read_text())
    selected = {json.loads((Path(v['run']) / 'checkpoints.json').read_text())[k] for k, v in plan.items()}
    weak = {int(k) for k, v in plan.items() if v['tracking'] < max_auc}
    found = collections.defaultdict(dict)
    for parent in DIAG.iterdir():
        match = re.match(r'(?:kv_)?ep(\d+)_', parent.name)
        if not match or int(match[1]) not in weak or any(s in parent.name.lower() for s in BAD):
            continue
        ep = int(match[1])
        for run in [parent, *parent.glob('early_*')]:
            result = run / 'results' / f'episode_{ep:06d}.json'
            if not (run / 'eval.done').is_file() or not result.is_file() or not (run / 'checkpoints.json').is_file():
                continue
            record = json.loads(result.read_text())
            if '/assets/' not in record.get('parquet', '') or any(s in record['parquet'].lower() for s in BAD):
                continue
            policy = json.loads((run / 'checkpoints.json').read_text())[str(ep)]
            if policy in selected or not Path(policy).is_file():
                continue
            auc = record['metrics']['add_auc']
            if auc > found[ep].get(policy, (0,))[0]:
                found[ep][policy] = (auc, parent.name)
    tasks = []
    for ep in sorted(found):
        best = sorted(found[ep].items(), key=lambda kv: -kv[1][0])[:top]
        for policy, (auc, parent) in best:
            step = int(re.search(r'policy_(\d+)', policy)[1])
            print(f'ep{ep} {parent} @{step}: {auc:.4f} (selected {plan[str(ep)]["tracking"]:.4f})')
            for rep in range(1, reps + 1):
                tasks.append(dict(episode=ep, policy=policy, baseline=auc, out=str(DIAG / f'{parent}_RS' / f'early_{step}_r{rep}')))
    Path(out).write_text(json.dumps(tasks, indent=1))
    print(f'{len(tasks)} tasks -> {out}')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:3]), *(float(x) for x in a[3:4]))
