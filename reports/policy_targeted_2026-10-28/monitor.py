"""Read-only CPU monitor for these trials; never cancels jobs or uploads submissions."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
DATA=Path('/data/user_data/edmondwa/v2d/flash_runs/diag')
EXPECTED={'ep5_TARGET_WIDE_s42','ep19_TARGET_CONTACTFILL_s43','ep30_TARGET_WIDE_s42','ep5_TARGET_BEST_FC_s42'}


def snapshot():
    out=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),trials=[])
    seen=set()
    for receipt in sorted(HERE.glob('*_trial.job.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        r=json.loads(receipt.read_text());trial=r['trial']
        if trial['name'] not in EXPECTED:continue
        if trial['name'] in seen:continue
        seen.add(trial['name'])
        run=DATA/trial['name']
        policies=sorted(int(p.stem.split('_')[1]) for p in (run/'train').glob('policy_*.safetensors'))
        state=subprocess.check_output(['sacct','-n','-X','-j',r['job'],'--format=State,Elapsed,ExitCode'],text=True).strip()
        row=dict(name=trial['name'],episode=trial['episode'],job=r['job'],account=trial['account'],
                 scheduler=state,latest_policy_steps=policies[-1] if policies else 0,
                 train_done=(run/'train.done').exists(),run=str(run),baseline_auc=trial['baseline_episode_auc'])
        log=run/'train.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-3500));row['log_tail']=stream.read().decode(errors='replace')
        selection=run/'checkpoint_selection.json'
        if selection.exists():row['evaluated_checkpoints']=json.loads(selection.read_text())
        out['trials'].append(row)
    for ep in (5,19,30):
        f=HERE/f'ep{ep}_trace_summary.json'
        out.setdefault('diagnostic_summaries',{})[str(ep)]=str(f) if f.exists() else None
    tmp=HERE/'status.tmp.json';tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(HERE/'status.json')
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    with (HERE/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            result=snapshot()
            print(json.dumps({r['name']:{'steps':r['latest_policy_steps'],'scheduler':r['scheduler']} for r in result['trials']}),flush=True)
            finished=len(result['trials'])==len(EXPECTED) and all(
                r['train_done'] and r.get('evaluated_checkpoints') and r['scheduler'].split()[0]=='COMPLETED'
                for r in result['trials'])
            if args.once or finished:break
            time.sleep(60)
