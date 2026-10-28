"""Evaluate saved post-helper checkpoints for weak episodes, one episode per array task."""
import json
import fcntl
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYTHON = '/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python'
RUNTIME = REPO / 'baseline/robotic_grounding/flash_chord'
SCRIPT = REPO / 'assets/data/v2d_submission_kit/examples/run_policy_evaluation.py'


def main():
    tasks = json.loads(Path(os.environ.get('V2D_CHECKPOINT_TASKS', str(HERE / 'saved_checkpoint_tasks.json'))).read_text())
    task = tasks[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    ep, run = task['episode'], Path(task['run'])
    policies = sorted((int(p.stem.split('_')[1]), p) for p in (run / 'train').glob('policy_*.safetensors'))
    targets = tuple(task.get('steps') or [int(s) for s in os.environ.get('V2D_CHECKPOINT_STEPS', '50010000,75010000,90010000').split(',')])
    for target in targets:
        step, policy = max((s, p) for s, p in policies if s <= target)
        out = run / f'early_{step}'
        if (out / 'eval.done').exists():
            continue
        out.mkdir(exist_ok=True)
        with (out / '.evaluation.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f'Another evaluator owns {out}; leaving it to that worker', flush=True)
                continue
            if (out / 'eval.done').exists():
                continue
            checkpoints = out / 'checkpoints.json'
            checkpoints.write_text(json.dumps({str(ep): str(policy)}))
            with (out / 'eval.log').open('a') as log:
                result = subprocess.run([PYTHON, str(SCRIPT), '--checkpoints', str(checkpoints), '--results', str(out / 'results'), '--runtime-root', str(RUNTIME)], cwd=RUNTIME, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'ep{ep} @{step} evaluation failed: {out}/eval.log')
            (out / 'eval.done').touch()
            score = json.loads((out / 'results' / f'episode_{ep:06d}.json').read_text())['metrics']['add_auc']
            print(f'ep{ep} {run.name} @{step}: {score:.6f}', flush=True)


if __name__ == '__main__':
    main()
