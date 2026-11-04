"""Move trials to Edmond's preempt allocation: rewrite account/partition/qos, set aside the old job receipt.
python to_preempt.py TRIAL.json [...]  (prints the trial files ready to submit)"""
import json, sys
from pathlib import Path
for f in map(Path, sys.argv[1:]):
    t = json.loads(f.read_text())
    t.update(account='edmondwa', partition='preempt', qos='preempt_qos'); t.pop('slurm_account', None)
    f.write_text(json.dumps(t, indent=2))
    job = f.with_name(f.name.replace('_trial.json', '_trial.job.json'))
    if job.exists():
        n = 1
        while job.with_name(job.name.replace('.job.json', f'.job.general{n}.json')).exists(): n += 1
        job.rename(job.with_name(job.name.replace('.job.json', f'.job.general{n}.json')))
    print(f.name)
