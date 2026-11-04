"""Detached, bounded EP38 policy replay and EP32/33 baseline training/evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess, sys, time

BASE=Path('/data/v2d');REPO=BASE/'repo';FC=REPO/'baseline/robotic_grounding/flash_chord';KIT=REPO/'assets/data/v2d_submission_kit'
PY=BASE/'envs/flash/bin/python';RUNS=BASE/'runs';RUNS.mkdir(exist_ok=True)
DEADLINE=datetime(2026,11,4,13,35,32,tzinfo=timezone.utc).timestamp()
env=os.environ.copy();env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_MODE='disabled',MUJOCO_GL='egl',OMP_NUM_THREADS='4')
ledger=BASE/'receipts/baseline_jobs.json';events=[]
def save():ledger.write_text(json.dumps(events,indent=2))
def execute(label,argv,log,limit):
 remaining=int(DEADLINE-time.time())
 if remaining<300:raise RuntimeError('Budget wall-clock deadline reached; no additional job')
 command=['timeout','--signal=TERM','--kill-after=60s',str(min(remaining,limit))]+list(map(str,argv))
 event={'label':label,'start_utc':datetime.now(timezone.utc).isoformat(),'argv':list(map(str,argv)),'log':str(log),'max_seconds':min(remaining,limit)};events.append(event);save()
 with log.open('w') as stream:
  proc=subprocess.Popen(command,cwd=FC,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);event['pid']=proc.pid;save();rc=proc.wait()
 event['exit_code']=rc;event['end_utc']=datetime.now(timezone.utc).isoformat();save();return rc

assert (BASE/'receipts/runtime.json').is_file(),'Environment compatibility gate did not pass'
rc=execute('prepare_scenes',[PY,BASE/'prepare_scenes.py'],BASE/'logs/prepare_scenes.log',300)
assert rc==0,'Scene closure validation failed'
prepared=json.loads((BASE/'receipts/prepared_scenes.json').read_text())

dest=RUNS/'ep38_control_replay';dest.mkdir(exist_ok=True)
if not (dest/'replay.done').exists():
 rc=execute('ep38_control_replay',[PY,KIT/'examples/run_policy_evaluation.py','--checkpoints',BASE/'receipts/control38_checkpoints.json','--results',dest/'results','--runtime-root',FC],dest/'eval.log',1200)
 assert rc==0,'Official evaluation compatibility gate failed'
 (dest/'replay.done').write_text('complete\n')

for ep in (32,33):
 label=f'ep{ep}_went_v2_s42';run=RUNS/label;run.mkdir(exist_ok=True)
 parquet=prepared[f'ep{ep}_went_v2']['parquet']
 if not (run/'train.done').exists():
  args=[PY,FC/'scripts/train_flash_sac.py','experiment=sharpa_flash_sac',f"task.parquet='{parquet}'",'reset.seed=42','logging.mode=disabled','training.total_environment_steps=100003840','curriculum=tracking_100m_object2x','task.motion_speed=1.0','sim.fps=20.254119873046875','sim.njmax=1024','sim.nconmax=512','termination.wrist_position.threshold=0.3','training.checkpoint.save_replay=false',f'output_dir={run}/train']
  rc=execute(label,args,run/'train.log',9600)
  (run/'train.exit').write_text(str(rc))
  if rc==0:(run/'train.done').write_text('complete\n')
  elif rc not in (124,137):raise RuntimeError(f'{label} training failed: {rc}')
 policies=sorted((run/'train').glob('policy_*.safetensors'),key=lambda p:int(p.stem.split('_')[-1]))
 assert policies,f'{label}: no saved policy'
 # Full sequence, explicit initial reset and no helper; all retained checkpoints may be assessed.
 for checkpoint in policies:
  step=int(checkpoint.stem.split('_')[-1]);result=run/f'eval_{step}';result.mkdir(exist_ok=True)
  if (result/'eval.done').exists():continue
  (result/'checkpoints.json').write_text(json.dumps({str(ep):str(checkpoint)}))
  rc=execute(f'{label}_eval_{step}',[PY,KIT/'examples/run_policy_evaluation.py','--checkpoints',result/'checkpoints.json','--results',result/'results','--runtime-root',FC],result/'eval.log',600)
  if rc==0:(result/'eval.done').write_text('complete\n')
  else:raise RuntimeError(f'{label} checkpoint evaluation failed at {step}: {rc}')
(BASE/'receipts/baselines.complete').write_text(datetime.now(timezone.utc).isoformat())
