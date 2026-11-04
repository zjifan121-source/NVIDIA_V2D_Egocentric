"""Per-body scores of the best checkpoint of runs, and an evaluator result json's metrics. usage: per_body.py RUN|JSON ..."""
import json, sys, os
D = '/data/user_data/edmondwa/v2d/flash_runs/diag'
for a in sys.argv[1:]:
    if a.endswith('.json'):
        m = json.load(open(a))['metrics']
        print('==', a.split('/diag/')[-1], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if not isinstance(v, (list, dict))})
        for k, v in m.items():
            if isinstance(v, dict): print('   ', k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in list(v.items())[:12]})
        continue
    rows = [json.loads(l) for l in open(f'{D}/{a}/checkpoint_evaluations.log') if l.startswith('{')]
    b = max(rows, key=lambda r: r['auc'])
    print('==', a, b['step'], round(b['auc'], 3), b.get("per_body"))
