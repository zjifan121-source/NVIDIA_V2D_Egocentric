"""Re-run the official evaluation on each selected checkpoint (the evaluator is not deterministic).

Build tasks:   python rescore.py build [REPS] [MAX_AUC]   -> rescore_tasks.json
Array worker:  python rescore.py run                      (one (episode, rep) per SLURM_ARRAY_TASK_ID)
Outputs land in diag/<parent>_RS/early_<step>_r<rep>/ so refresh_tracking.py picks them up.
"""
import fcntl
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = Path('/data/user_data/edmondwa/v2d')
DIAG = DATA / 'flash_runs/diag'
NIGHT = DATA / 'ken_night/night_2026-10-24'
PYTHON = DATA / 'venvs/flash_chord/bin/python'
RUNTIME = REPO / 'baseline/robotic_grounding/flash_chord'
SCRIPT = REPO / 'assets/data/v2d_submission_kit/examples/run_policy_evaluation.py'
TASKS = Path(os.environ.get('V2D_RESCORE_TASKS', HERE / 'rescore_tasks.json'))
FIRST_REP = int(os.environ.get('V2D_RESCORE_FIRST_REP', '1'))
ONLY = {e for e in os.environ.get('V2D_RESCORE_EPISODES', '').split(',') if e}


def resolve(policy):
    path = Path(policy)
    if not path.is_file() and str(path).startswith('/scratch/haisenk-v2d/night/'):
        path = NIGHT / str(path).split('/scratch/haisenk-v2d/night/')[1]
    return path if path.is_file() else None


def build(reps=4, max_auc=0.95):
    plan = json.loads((HERE / 'plan.json').read_text())
    tasks = []
    for ep, entry in plan.items():
        if entry['tracking'] >= max_auc or (ONLY and ep not in ONLY):
            continue
        policy = resolve(json.loads((Path(entry['run']) / 'checkpoints.json').read_text())[ep])
        if policy is None:
            print(f'ep{ep}: policy not found for {entry["name"]}, skipped')
            continue
        parent = entry['name'].split('/')[0].removesuffix('_RS')
        step = int(re.search(r'policy_(\d+)', policy.name)[1])
        for rep in range(FIRST_REP, FIRST_REP + reps):
            tasks.append(dict(episode=int(ep), policy=str(policy), baseline=entry['tracking'],
                              out=str(DIAG / f'{parent}_RS' / f'early_{step}_r{rep}')))
    TASKS.write_text(json.dumps(tasks, indent=1))
    print(f'{len(tasks)} tasks over {len(tasks) // reps} episodes -> {TASKS}')


def run():
    task = json.loads(TASKS.read_text())[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    ep, out = task['episode'], Path(task['out'])
    if (out / 'eval.done').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'Another evaluator owns {out}', flush=True)
            return
        checkpoints = out / 'checkpoints.json'
        checkpoints.write_text(json.dumps({str(ep): task['policy']}))
        with (out / 'eval.log').open('a') as log:
            result = subprocess.run([str(PYTHON), str(SCRIPT), '--checkpoints', str(checkpoints), '--results', str(out / 'results'),
                                     '--runtime-root', str(RUNTIME)], cwd=RUNTIME, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'ep{ep} rescore failed: {out}/eval.log')
        (out / 'eval.done').touch()
        score = json.loads((out / 'results' / f'episode_{ep:06d}.json').read_text())['metrics']['add_auc']
        print(f'ep{ep} {out}: {score:.6f} (selected {task["baseline"]:.6f})', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(*(int(sys.argv[2]),) if len(sys.argv) > 2 else (), *(float(sys.argv[3]),) if len(sys.argv) > 3 else ())
    else:
        run()
