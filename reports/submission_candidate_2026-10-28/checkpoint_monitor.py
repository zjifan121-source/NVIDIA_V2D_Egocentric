"""Evaluate available late checkpoints for authorized trials, at most two jobs at once.

This controller never occupies a GPU while waiting for a checkpoint. It resumes
from an atomic ledger, leaves training jobs untouched, and never uploads to Kaggle.
"""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,time
P=Path(__file__).resolve().parent
TRIALS=[(25,'ep25_assembly2_OT1M_s42'),(25,'ep25_assembly2_WIDE_s42'),(25,'ep25_assembly2_WIDE_EARLY_s42'),(33,'ep33_WIDE_s42'),(33,'ep33_WIDE_EARLY_s42'),(33,'ep33_R100_OT_s42'),(5,'ep5_R005_OT_s42'),(19,'ep19_ST_OT_s42'),(30,'ep30_table_OT1M_s42'),(25,'ep25_assembly2_OT1M_s43'),(33,'ep33_WIDE_EARLY_NOUNINT_s42')]
TARGETS=[50010000,75010000,90010000,95010000]
RUNS=Path('/data/user_data/edmondwa/v2d/flash_runs/diag')

def save(state):
 tmp=P/'checkpoint_monitor_state.tmp.json';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(P/'checkpoint_monitor_state.json')

def iterate(state,dry_run):
 active=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%A'],text=True).split();active={j.split('_')[0] for j in active};outstanding=[]
 for key,r in state.items():
  if r.get('status')!='queued':continue
  done=Path(r['run'])/f"early_{r['step']}"/'eval.done'
  if done.exists():r['status']='done';continue
  if r['job'] in active:outstanding.append(r);continue
  terminal=subprocess.check_output(['sacct','-n','-X','-j',r['job'],'--format=State'],text=True).strip().split()
  if terminal and terminal[0].split('+')[0] in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']:
   r['status']='terminal_without_result';r['scheduler_state']=terminal[0];print('No successful eval:',key,r['scheduler_state'],flush=True)
  else:outstanding.append(r)
 budget=2-len(outstanding)
 trial_file=P/'checkpoint_monitor_trials.json'
 trials=json.loads(trial_file.read_text()) if trial_file.exists() else TRIALS
 for ep,name in trials:
  run=RUNS/name;policies=sorted(int(p.stem.split('_')[1]) for p in (run/'train').glob('policy_*.safetensors'))
  for target in (TARGETS+[100010000,105010000,110010000,115010000,120010000,125010000] if 'LOWLR' in name else TARGETS):
   candidates=[s for s in policies if s<=target]
   if not candidates:continue
   step=max(candidates)
   # A missing target is not ready; do not evaluate an earlier checkpoint by accident.
   if target-step>10000:continue
   key=f'{name}/{step}';done=run/f'early_{step}'/'eval.done'
   if done.exists():state.setdefault(key,dict(run=str(run),step=step,status='done'));continue
   if key in state:continue
   if budget<=0:continue
   print('Ready:',key,flush=True)
   if dry_run:budget-=1;continue
   tasks=P/f'auto_eval_{name}_{step}.json';tasks.write_text(json.dumps([dict(episode=ep,run=str(run),steps=[step])])+'\n')
   job=subprocess.check_output(['sbatch','--parsable','--array=0',f'--job-name=checkpoint-ep{ep}-{step//1000000}M',f'--export=ALL,V2D_CHECKPOINT_TASKS={tasks}',str(P/'evaluate_saved.sbatch')],text=True).strip();state[key]=dict(run=str(run),episode=ep,step=step,status='queued',job=job);save(state);budget-=1;print('Submitted:',key,job,flush=True)
 if not dry_run:save(state)
 return all((RUNS/name/'eval.done').exists() for ep,name in trials) and not any(r.get('status')=='queued' for r in state.values())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');args=ap.parse_args();lock=(P/'checkpoint_monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ledger=P/'checkpoint_monitor_state.json';state=json.loads(ledger.read_text()) if ledger.exists() else {}
 while True:
  try:finished=iterate(state,args.dry_run)
  except (subprocess.SubprocessError,OSError) as e:print('Transient controller read failure:',type(e).__name__,str(e),flush=True);finished=False
  if args.dry_run or finished:return
  time.sleep(30)
if __name__=='__main__':main()
