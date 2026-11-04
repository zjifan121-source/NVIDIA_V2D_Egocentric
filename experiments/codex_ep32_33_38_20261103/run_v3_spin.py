"""Remote-only bounded positive-control experiment after the existing GPU queue.

Uses the newly staged, validated Wentao v3-r1 EP32/33 respin scenes. No Kaggle uploads.
Never edits the active baseline queue or overlaps its GPU allocation.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess, time

BASE=Path('/data/v2d');FC=BASE/'repo/baseline/robotic_grounding/flash_chord'
KIT=BASE/'repo/assets/data/v2d_submission_kit';PY=BASE/'envs/flash/bin/python'
DEADLINE=datetime(2026,11,4,13,35,32,tzinfo=timezone.utc).timestamp()
ledger=BASE/'receipts/v3_spin_jobs.json';events=[]
env=os.environ.copy();env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_MODE='disabled',MUJOCO_GL='egl',OMP_NUM_THREADS='4')
def save():ledger.write_text(json.dumps(events,indent=2))
def run(label,args,log,limit):
 remaining=int(DEADLINE-time.time())
 if remaining<300:raise RuntimeError('Budget deadline reached; no further work')
 item={'label':label,'start_utc':datetime.now(timezone.utc).isoformat(),'argv':list(map(str,args)),'max_seconds':min(remaining,limit),'log':str(log)}
 events.append(item);save()
 with log.open('w') as output:
  p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=60s',str(item['max_seconds'])]+item['argv'],cwd=FC,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
  item['pid']=p.pid;save();rc=p.wait()
 item.update(exit_code=rc,end_utc=datetime.now(timezone.utc).isoformat());save()
 if rc:raise RuntimeError(f'{label} failed: {rc}')

events.append({'label':'wait_for_baseline_queue','state':'WAITING','deadline_utc':'2026-11-04T13:35:32Z'});save()
while not (BASE/'receipts/baselines.complete').exists():
 if (BASE/'receipts/baseline_queue_v2.exit').exists():
  raise RuntimeError('Baseline queue exited before completion; inspect before using GPU')
 if DEADLINE-time.time()<10800:raise RuntimeError('Insufficient bounded time for another 100M experiment')
 time.sleep(30)
events[0]['state']='COMPLETE';save()
for ep in (32,33):
 RUN=BASE/'runs'/f'ep{ep}_went_v3_r1_s42';RUN.mkdir(exist_ok=True)
 prepared=json.loads((BASE/'receipts/prepared_v3_scenes.json').read_text())[f'ep{ep}_went_v3_r1']
 args=[PY,FC/'scripts/train_flash_sac.py','experiment=sharpa_flash_sac',f"task.parquet='{prepared['parquet']}'",'reset.seed=42','logging.mode=disabled','training.total_environment_steps=100003840','curriculum=tracking_100m_object2x','task.motion_speed=1.0','sim.fps=20.254119873046875','sim.njmax=1024','sim.nconmax=512','termination.wrist_position.threshold=0.3','training.checkpoint.save_replay=false',f'output_dir={RUN}/train']
 run(f'ep{ep}_went_v3_r1_s42_train',args,RUN/'train.log',9600)
 (RUN/'train.done').write_text('complete\n')
 policies=sorted((RUN/'train').glob('policy_*.safetensors'),key=lambda x:int(x.stem.split('_')[-1]))
 assert policies
 for checkpoint in policies:
  step=int(checkpoint.stem.split('_')[-1]);result=RUN/f'eval_{step}';result.mkdir(exist_ok=True)
  (result/'checkpoints.json').write_text(json.dumps({str(ep):str(checkpoint)}))
  run(f'ep{ep}_went_v3_r1_s42_eval_{step}',[PY,KIT/'examples/run_policy_evaluation.py','--checkpoints',result/'checkpoints.json','--results',result/'results','--runtime-root',FC],result/'eval.log',600)
  (result/'eval.done').write_text('complete\n')
(BASE/'receipts/v3_spin.complete').write_text(datetime.now(timezone.utc).isoformat())
