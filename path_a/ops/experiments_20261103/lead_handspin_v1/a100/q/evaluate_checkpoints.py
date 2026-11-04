"""Official full rollouts for saved checkpoints (Edmond's evaluate_checkpoints.py with KIT/FC from the environment)."""
import argparse, fcntl, json, os, subprocess, sys
from pathlib import Path
KIT=Path(os.environ['KIT']); FC=Path(os.environ['FC'])
def main(ep,run):
    policies=sorted((int(p.stem.split('_')[1]),p) for p in (run/'train').glob('policy_*.safetensors'))
    assert policies,run
    chosen={(s,p) for s,p in policies if s>=70_000_000}  # recipe: evaluate every checkpoint from 70M; the best is usually not the last
    if not chosen: return
    scores=[]
    for step,policy in sorted(chosen):
        out=run/f'early_{step}'; out.mkdir(exist_ok=True)
        with (out/'.evaluation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (out/'eval.done').exists():
                checkpoint=out/'checkpoints.json'
                checkpoint.write_text(json.dumps({str(ep):str(policy)})+'\n')
                with (out/'eval.log').open('a') as log:
                    subprocess.run([sys.executable,str(KIT/'examples/run_policy_evaluation.py'),'--checkpoints',str(checkpoint),'--results',str(out/'results'),'--runtime-root',str(FC)],cwd=FC,stdout=log,stderr=subprocess.STDOUT,check=True)
                (out/'eval.done').touch()
        result=json.loads((out/'results'/f'episode_{ep:06d}.json').read_text())
        scores.append(dict(episode=ep,step=step,auc=result['metrics']['add_auc'],per_body=result['metrics']['add_auc_per_body'],mppe_cm=result['metrics'].get('mppe_cm'),run=str(out)))
        print(json.dumps(scores[-1]),flush=True)
        (run/'checkpoint_selection.json').write_text(json.dumps(sorted(scores,key=lambda x:x['auc'],reverse=True),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.episode,a.run)
