"""Submit one explicitly recorded trial; no implicit sweep or changes to existing jobs."""
import argparse
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('trial',type=Path);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
trial=a.trial.resolve();t=json.loads(trial.read_text());receipt=trial.with_suffix('.job.json')
if receipt.exists():raise FileExistsError(f'Trial already submitted: {receipt}')
export=f'ALL,NAME={t["name"]},EP={t["episode"]},SEED={t["seed"]},STEPS={t["total_environment_steps"]},PARQUET={t["parquet"]},EXTRA={t["extra"]}'
for key,field in [('INIT_STATE','init_state'),('V2D_WARM_ACTOR','warm_actor'),('V2D_WARM_BASE_STATE','init_state'),('SAVE_REPLAY','save_replay'),('CURRICULUM','curriculum')]:
    if t.get(field):export+=f',{key}={t[field]}'
cmd=['sbatch','--parsable',f'--partition={t["partition"]}',f'--qos={t["qos"]}',f'--job-name={t["name"]}',f'--export={export}',f'--output={HERE}/train-%j.log']
if t.get('slurm_account'):cmd.append(f'--account={t["slurm_account"]}')
for key,flag in [('cpus','--cpus-per-task'),('memory','--mem'),('time','--time')]:
    if t.get(key):cmd.append(f'{flag}={t[key]}')
cmd.append(str(HERE/t.get('launcher','train.sbatch')))
if a.dry_run:print(json.dumps(cmd,indent=2))
else:
    identity=subprocess.check_output(['id','-un'],text=True).strip()
    if identity!=t['account']:raise ValueError(f'Expected {t["account"]}, got {identity}')
    job=subprocess.check_output(cmd,text=True).strip()
    receipt.write_text(json.dumps(dict(job=job,trial=t,command=cmd),indent=2)+'\n')
    print(job,flush=True)
