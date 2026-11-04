"""Continue finished runs from their best step on Sarvesh's general allowance (writes trials; submit as sarveshp).
python cont_general.py RUN [RUN...]   -> prints trial paths, one per line, also into .cont_general.txt"""
import json, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent
D = Path('/data/user_data/edmondwa/v2d/flash_runs/diag')
out = []
for run in sys.argv[1:]:
    rows = [json.loads(l) for l in (D / run / 'checkpoint_evaluations.log').read_text().splitlines() if l.startswith('{')]
    step = max(rows, key=lambda r: r['auc'])['step']
    tf = subprocess.run(['python3', str(H / 'whc_cont.py'), run, str(step)], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    p = H / tf; t = json.loads(p.read_text())
    t.update(account='sarveshp', partition='general', qos='normal', slurm_account='msimchow')
    p.write_text(json.dumps(t, indent=2)); (D / run / '.continued').write_text(tf + '\n')
    out.append(str(p)); print(f'{run}@{step}: {p.name}', file=sys.stderr)
(H / '.cont_general.txt').write_text('\n'.join(out) + '\n')
