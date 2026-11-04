"""Overnight keeper for V2D training: prune, release my held jobs, refresh the plan, chain low-LR continuations.

Runs as edmondwa on maxlab-cpu. One pass every PERIOD seconds until the job's time limit.
Only touches jobs submitted by edmondwa whose names match this effort's patterns.
"""
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

H = Path(__file__).resolve().parent
REPO = H.parents[1]
SUB = REPO / 'reports/submission_candidate_2026-10-28'
D = Path('/data/user_data/edmondwa/v2d/flash_runs/diag')
LOG = H / 'night_keeper.log'
PERIOD = 1200
MINE = re.compile(r'^ep\d+_(DEF_s\d+|OBJ4X_250M_s\d+|FOCUS_s\d+|RELAX[AB]_s\d+|WMIX_s\d+|CAVITY_OBJ4X_250M_s\d+|.*LOWLR.*)$')
CHAIN = re.compile(r'^ep(\d+)_(DEF_s\d+|OBJ4X_250M_s\d+|FOCUS_s\d+|RELAX[AB]_s\d+|WMIX_s\d+|CAVITY_OBJ4X_250M_s\d+|LOWLRc\d+_.*)$')
EXTRA_STEPS = 25001984
MAX_CHAIN = 3


def log(msg):
    with LOG.open('a') as f:
        f.write(time.strftime('%m-%d %H:%M ') + msg + '\n')


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def free_gb():
    return shutil.disk_usage('/data/user_data/edmondwa').free / 1e9


def release_held():
    for line in sh('squeue -u edmondwa -h -o "%i|%j|%r"').splitlines():
        jid, name, reason = line.split('|', 2)
        if 'held' in reason and MINE.match(name):
            sh(f'scontrol release {jid}')
            log(f'released {jid} {name}')


def best_eval(run):
    f = run / 'checkpoint_evaluations.log'
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.startswith('{')]
    return max(rows, key=lambda r: r['auc']) if rows else None


def chain(plan, queued):
    """Low-LR continuation of a finished run from its newest matched state/replay pair."""
    for run in sorted(D.iterdir()):
        m = CHAIN.match(run.name)
        if not m or not (run / 'train.done').exists() or (run / '.chained').exists() or run.name in queued:
            continue
        ep = m[1]
        depth = run.name.count('LOWLRc')
        best = best_eval(run)
        if best is None:
            continue
        if depth >= MAX_CHAIN or best['auc'] < plan[ep]['tracking'] - 0.05:
            (run / '.chained').write_text('skipped\n')
            continue
        train = run / 'train'
        steps = sorted({int(p.stem.split('_')[1]) for p in train.glob('state_*.safetensors')} &
                       {int(p.stem.split('_')[1]) for p in train.glob('replay_*.safetensors')})
        if not steps:
            (run / '.chained').write_text('no matched state/replay pair\n')
            continue
        step = steps[-1]
        name = f'ep{ep}_LOWLRc{depth + 1}_{step // 1000000}M_{run.name.split("_", 1)[1]}'[:80]
        rec = json.loads(sh(f'python3 {H}/recipe_overrides.py {train}/policy_{step}.safetensors'))
        new = D / name / 'train'
        new.mkdir(parents=True, exist_ok=True)
        for kind in ('state', 'replay'):
            src, dst = train / f'{kind}_{step}.safetensors', new / f'{kind}_{step}.safetensors'
            if not dst.exists():
                try:
                    os.link(src, dst)
                except PermissionError:
                    shutil.copy2(src, dst)
        lr = 5e-5
        extra = rec['overrides'] + (f' training.checkpoint.save_interval_environment_steps=5000192 training.optimization.learning_rate_initial={lr}'
                                    f' training.optimization.learning_rate_peak={lr} training.optimization.learning_rate_end={lr}')
        base = json.loads((H / 'ep10_nest_train_trial.json').read_text())
        base.pop('slurm_account', None)
        trial = dict(base, name=name, episode=int(ep), seed=rec['seed'], account='edmondwa', partition='preempt', qos='preempt_qos',
                     scene=rec['parquet'].split('/assets/')[0], parquet=str(Path(rec['parquet']).parent), save_replay='true',
                     total_environment_steps=step + EXTRA_STEPS, baseline_episode_auc=plan[ep]['tracking'], memory='96G', time='01:30:00',
                     extra=extra, hypothesis=f'Night keeper: constant LR {lr} continuation of {run.name}@{step} (best {best["auc"]:.3f}) with its replay.')
        if run.name.split('_')[1] == 'CAVITY':
            trial['memory'] = '128G'
        path = H / f'{name.lower()}_trial.json'
        path.write_text(json.dumps(trial, indent=2))
        job = sh(f'python3 {H}/submit_trial.py {path}').strip()
        (run / '.chained').write_text(f'{name} {job}\n')
        log(f'chained {run.name} (best {best["auc"]:.3f}, plan {plan[ep]["tracking"]:.3f}) -> {name} job {job}')


def main():
    deadline = time.time() + 5.6 * 3600  # maxlab-cpu allows 6 h; hand over to a fresh keeper before then
    while time.time() < deadline:
        try:
            out = sh(f'python3 {H}/prune_today.py; python3 {H}/prune_live.py').strip().splitlines()
            log(f'prune: {out[-1] if out else "?"}; free {free_gb():.0f} GB')
            if free_gb() < 40:
                out = sh(f'python3 {H}/prune_runs.py').strip()
                log(f'low disk, pruned finished runs: {out}; free {free_gb():.0f} GB')
            if free_gb() > 35:
                release_held()
            out = sh(f'cd {REPO} && tools/.venv/bin/python {SUB}/refresh_tracking.py 2>&1 | tail -1').strip()
            log(f'refresh: {out}')
            plan = json.loads((SUB / 'plan.json').read_text())
            queued = set(sh('squeue -u edmondwa,sarveshp -h -o %j').split())
            if free_gb() > 50:
                chain(plan, queued)
        except Exception as e:  # keep the night going; the log says what broke
            log(f'error: {e!r}')
        time.sleep(PERIOD)
    job = sh(f'cd {H} && sbatch --parsable -p maxlab-cpu --qos=maxlab_cpu_qos -c1 --mem=4G -t 6:00:00 -J v2d-night-keeper -o {H}/night_keeper.out night_keeper.sh').strip()
    log(f'handed over to keeper job {job}')


if __name__ == '__main__':
    main()
