"""Graft trials (default + WIDE, given seeds) for episodes with a scenes_graft scene, on Sarvesh's general allowance.
python wg2_trials.py EP[,EP...] SEED[,SEED...]   -> trial paths in .wg2_general.txt"""
import json, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent
WIDE = (' action.normalized_mapping=linear training.network.initial_normalized_std=0.3'
        ' training.network.log_std_max=-0.5108256237659907 training.exploration.target_sigma=0.3')
r = subprocess.run(['python3', str(H / 'make_wg_trials.py'), sys.argv[1], sys.argv[2]], capture_output=True, text=True)
print(r.stdout, r.stderr)
out = []
for ep in sys.argv[1].split(','):
    for seed in sys.argv[2].split(','):
        p = H / f'ep{ep}_wg_s{seed}g_trial.json'
        if not p.exists():
            continue
        t = json.loads(p.read_text())
        t.update(account='sarveshp', partition='general', qos='normal', slurm_account='msimchow')
        p.write_text(json.dumps(t, indent=2)); out.append(str(p))
        w = dict(t, name=f'ep{ep}_WGWIDE_s{seed}', extra=t['extra'] + WIDE, hypothesis=t['hypothesis'] + ' WIDE recipe.')
        q = H / f'ep{ep}_wgwide_s{seed}g_trial.json'; q.write_text(json.dumps(w, indent=2)); out.append(str(q))
(H / '.wg2_general.txt').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
