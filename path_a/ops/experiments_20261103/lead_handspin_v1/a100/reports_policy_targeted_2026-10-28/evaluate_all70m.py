"""Official full rollouts for saved checkpoints; preserve intermediates and choose the best."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
KIT=REPO/'assets/data/v2d_submission_kit'
FC=REPO/'baseline/robotic_grounding/flash_chord'


def main(ep,run):
    policies=sorted((int(p.stem.split('_')[1]),p) for p in (run/'train').glob('policy_*.safetensors'))
    assert policies,run
    targets=[]  # lead: evaluate every checkpoint from 70M (TRAINING_GUIDE) instead of fixed targets
    chosen={(s,p) for s,p in policies if s>=70_000_000}
    chosen.add(policies[-1])
    # Include every later refinement checkpoint, which may outperform the final policy.
    chosen.update((s,p) for s,p in policies if s>100_010_000)
    scores=[]
    for step,policy in sorted(chosen):
        out=run/f'early_{step}'
        out.mkdir(exist_ok=True)
        with (out/'.evaluation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (out/'eval.done').exists():
                checkpoint=out/'checkpoints.json'
                checkpoint.write_text(json.dumps({str(ep):str(policy)})+'\n')
                with (out/'eval.log').open('a') as log:
                    subprocess.run([sys.executable,str(KIT/'examples/run_policy_evaluation.py'),
                        '--checkpoints',str(checkpoint),'--results',str(out/'results'),
                        '--runtime-root',str(FC)],cwd=FC,stdout=log,stderr=subprocess.STDOUT,check=True)
                (out/'eval.done').touch()
        result=json.loads((out/'results'/f'episode_{ep:06d}.json').read_text())
        scores.append(dict(episode=ep,step=step,auc=result['metrics']['add_auc'],
                           per_body=result['metrics']['add_auc_per_body'],run=str(out)))
        print(json.dumps(scores[-1]),flush=True)
        (run/'checkpoint_selection.json').write_text(json.dumps(sorted(scores,key=lambda x:x['auc'],reverse=True),indent=2)+'\n')
    # No global plan is changed automatically. New run folders are discoverable by refresh_tracking.py.


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.episode,a.run)
