"""One training campaign on a reconstruction bundle set: build scenes, check them, train, continue, pack.

    python campaign.py init   NAME BUNDLES_ROOT [--tag T]   # BUNDLES_ROOT/episode_XXXXXX/bundle/result.npz
    python campaign.py init   NAME --ns NS [--tag T]        # hub layout under recon data/outputs/NS (one dir per object)
    python campaign.py build  NAME [--eps 1,3]              # raw and rotation-smoothed scene per episode (maxlab-cpu)
    python campaign.py check  NAME                          # scene_qa.py on both scene sets
    python campaign.py train  NAME [--seeds 2] [--recipe DEF,WIDE] [--variant raw|rs|both] [--eps ...] [--where preempt|general] [--dry-run]
    python campaign.py next   NAME                          # continue each episode's best finished run at low LR
    python campaign.py status NAME                          # best tracking per episode
    python campaign.py prune  NAME                          # keep only best-step learner states of finished runs
    python campaign.py pack   NAME                          # plan + packed parquet (never uploads)

Setup, recipe choice, seeds and what to expect: TRAINING_GUIDE.md next to this file. Paths and Slurm identity come from
V2D_DATA, V2D_USER, V2D_USERS, V2D_PARTITION, V2D_QOS, V2D_SLURM_ACCOUNT (defaults: Edmond's), see the guide.

Training goes to Edmond's preempt allocation and keeps learner states from 70M steps on (fresh_keep.sbatch), so the
best checkpoint of any finished run can be continued. State lives in campaigns/NAME.json next to this file.

Default portfolio (11-02 sweep on Wentao's scored20 scenes, 100M fresh runs): no single recipe wins everywhere and seed
spread is large (ep5 default 0.25-0.65), so each episode gets 2 default + 2 WIDE seeds on the raw scene and keeps the
best. WIDE beat default on ep5 (+0.13), ep25 (+0.09) and ep36 (+0.20) but lost on ep19 (-0.28) and ep10 (-0.08); RELAXA was mixed
and is left out; the rotation-smoothed scene did not beat raw on average.
"""
import argparse
import glob
import json
import os
import re
import subprocess
from pathlib import Path

H = Path(__file__).resolve().parent
REPO = H.parents[1]
DATA = Path(os.environ.get('V2D_DATA', '/data/user_data/edmondwa/v2d'))
USER = os.environ.get('V2D_USER', 'edmondwa')  # account that submits (submit_trial.py checks it)
USERS = os.environ.get('V2D_USERS', 'edmondwa,sarveshp')  # whose queue counts as 'still running'
D = DATA / 'flash_runs/diag'
EPISODES = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
VARIANTS = {'raw': 0, 'rs': 2}  # --smooth-object-rotation sigma in frames
DEFAULT = ('task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=1024 sim.nconmax=512 termination.wrist_position.threshold=0.3 '
           'termination.object_position.threshold=0.3 termination.object_orientation.threshold=1.5 '
           'training.checkpoint.save_interval_environment_steps=5000192 training.replay.capacity=1048576 '
           'training.replay.observation_storage_dtype=float16')
BIG = {10: '128G'}  # episodes whose contact buffers need more room
RECIPES = {
    'DEF': {},
    'WIDE': dict(extra=' action.normalized_mapping=linear training.network.initial_normalized_std=0.3'
                       ' training.network.log_std_max=-0.5108256237659907 training.exploration.target_sigma=0.3'),
    'RELAXA': dict(curriculum='tracking_100m_relaxhand_a'),
    # Ken's team recipe, read from the resolved config in his 3 Nov policies: tighter object termination, 4x replay in float32.
    # On ep10_kw it gave 0.709-0.713 vs 0.594 for DEF; use it on scenes Ken fixed and wherever DEF stalls on a hard object.
    'KEN': dict(subs=[('termination.object_position.threshold=0.3', 'termination.object_position.threshold=0.1'),
                      ('termination.object_orientation.threshold=1.5', 'termination.object_orientation.threshold=0.5'),
                      ('training.replay.capacity=1048576', 'training.replay.capacity=4194304'),
                      ('training.replay.observation_storage_dtype=float16', 'training.replay.observation_storage_dtype=float32'),
                      ('sim.njmax=2048 sim.nconmax=1024', 'sim.njmax=1024 sim.nconmax=512')]),
}
EPISODE_INFO = {e['episode']: e for e in json.loads((REPO / 'tools/track3/episodes.json').read_text())['episodes']}
MARGIN = 0.08  # continue a run only if its best is within this of the episode's best


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def state_path(name):
    return H / 'campaigns' / f'{name}.json'


def load(name):
    return json.loads(state_path(name).read_text())


def save(c):
    state_path(c['name']).parent.mkdir(exist_ok=True)
    state_path(c['name']).write_text(json.dumps(c, indent=2))


def scene_dir(c, variant, ep):
    return DATA / 'campaigns' / c['name'] / f'scenes_{variant}' / f'ep{ep}'


def queued():
    return set(sh(f'squeue -u {USERS} -h -o %j').stdout.split())


def runs(c):
    """Finished campaign runs with their best evaluated checkpoint."""
    out = []
    for run in D.glob('ep*'):
        m = re.match(rf"ep(\d+)_(?:WHCc_\d+M_)*{c['tag']}(raw|rs)_", run.name)  # continuations prefix WHCc_<step>M_
        log = run / 'checkpoint_evaluations.log'
        if not m or not log.exists():
            continue
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.startswith('{')]
        if rows:
            b = max(rows, key=lambda r: r['auc'])
            out.append(dict(run=run, ep=int(m[1]), variant=m[2], auc=b['auc'], step=b['step'], done=(run / 'train.done').exists()))
    return out


def cmd_init(a):
    c = dict(name=a.name, bundles=str(Path(a.bundles).resolve()) if a.bundles else None, tag=a.tag or re.sub(r'\W', '', a.name)[:6].upper())
    if a.ns:
        c['ns'] = a.ns
    save(c)
    print(json.dumps(c, indent=2))


def cmd_build(a):
    c = load(a.name)
    for ep in a.eps:
        if c.get('ns'):  # the loader finds (and merges) each object's bundle under data/outputs/NS
            names = [o['name'] for o in EPISODE_INFO[ep]['objects']]
            src = f"NS={c['ns']}" + (f",OBJ={names[0]}" if len(names) == 1 else '')
        else:
            bundle = Path(c['bundles']) / f'episode_{ep:06d}' / 'bundle'
            if not (bundle / 'result.npz').exists():
                print(f'ep{ep}: no bundle at {bundle}'); continue
            src = f'BUNDLE={bundle}'
        for v, sigma in VARIANTS.items():
            if a.variant != 'both' and v != a.variant:
                continue
            out = scene_dir(c, v, ep); out.parent.mkdir(parents=True, exist_ok=True)
            env = f"ALL,EP={ep},{src},OUT={out},SEQ=ep{ep}_{c['tag']}{v},OBJSMOOTH={sigma}"
            r = sh(f'sbatch --parsable -J build-{c["tag"]}-{v}{ep} -o {out}.build.out --export={env} {H}/campaign_build.sbatch')
            print(f'ep{ep} {v}: job {r.stdout.strip() or r.stderr.strip()}')


def cmd_check(a):
    c = load(a.name)
    for v in VARIANTS:
        root = scene_dir(c, v, 0).parent
        print(f'== {v}: {root}')
        r = sh(f'{DATA}/venvs/retarget/bin/python {H}/scene_qa.py {root} --out {root}/scene_qa.json')
        print(r.stdout, r.stderr[-2000:])


def trial(c, ep, v, recipe, seed, where='preempt'):
    pq = glob.glob(f'{scene_dir(c, v, ep)}/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave')
    if len(pq) != 1:
        return None
    extra = DEFAULT.replace('sim.njmax=1024 sim.nconmax=512', 'sim.njmax=2048 sim.nconmax=1024') if ep in BIG else DEFAULT
    for a_, b_ in RECIPES[recipe].get('subs', []):
        extra = extra.replace(a_, b_)
    name = f"ep{ep}_{c['tag']}{v}_{recipe}_s{seed}"
    t = dict(name=name, episode=ep, account=USER, partition=os.environ.get('V2D_PARTITION', 'preempt'),
             qos=os.environ.get('V2D_QOS', 'preempt_qos'), launcher='fresh_keep.sbatch',
             cpus=8, memory=BIG.get(ep, '64G'), time='02:00:00', scene=str(scene_dir(c, v, ep)), parquet=pq[0], seed=seed,
             total_environment_steps=100003840, save_replay=False, extra=extra + RECIPES[recipe].get('extra', ''),
             hypothesis=f"Campaign {c['name']}: {recipe} recipe on the {v} scene.")
    if os.environ.get('V2D_SLURM_ACCOUNT'):
        t['slurm_account'] = os.environ['V2D_SLURM_ACCOUNT']
    if 'curriculum' in RECIPES[recipe]:
        t['curriculum'] = RECIPES[recipe]['curriculum']
    if where == 'general':  # Sarvesh's general allowance: submit as sarveshp (submit_trial.py checks the identity)
        t.update(account='sarveshp', partition='general', qos='normal', slurm_account='msimchow')
    path = H / f'{name.lower()}_trial.json'
    path.write_text(json.dumps(t, indent=2))
    return path


def submit(path, dry=False):
    r = sh(f'python3 {H}/submit_trial.py {"--dry-run " if dry else ""}{path}')
    return (r.stdout.strip() or r.stderr.strip().splitlines()[-1]) if (r.stdout or r.stderr) else '?'


def cmd_train(a):
    c = load(a.name)
    variants = list(VARIANTS) if a.variant == 'both' else [a.variant]
    for ep in a.eps:
        for v in variants:
            for recipe in a.recipe:
                for i in range(a.seeds):
                    seed = a.seed0 + i
                    p = trial(c, ep, v, recipe, seed, a.where)
                    if a.where == 'general':  # written only; submit as sarveshp
                        print(p or f'ep{ep} {v}: no scene'); continue
                    print(f'ep{ep} {v} {recipe} s{seed}:', (submit(p, a.dry_run) if p else 'no scene')[-400:])


def cmd_next(a):
    c = load(a.name); q = queued(); rs = runs(c)
    top = {}
    for r in rs:
        top[r['ep']] = max(top.get(r['ep'], 0), r['auc'])
    best = {}
    for r in rs:
        st = [p for p in (r['run'] / 'keep' / f"state_{r['step']}.safetensors", r['run'] / 'train' / f"state_{r['step']}.safetensors") if p.exists()]
        if r['done'] and r['run'].name not in q and st and not (r['run'] / '.continued').exists() and r['auc'] >= top[r['ep']] - MARGIN:
            if r['auc'] > best.get(r['ep'], {'auc': -1})['auc']:
                best[r['ep']] = r
    for ep, r in sorted(best.items()):
        out = sh(f"python3 {H}/whc_cont.py {r['run'].name} {r['step']}")
        tfile = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ''
        if not tfile.endswith('_trial.json'):
            print(f'ep{ep}: could not write continuation: {out.stderr[-300:]}'); continue
        (r['run'] / '.continued').write_text(tfile + '\n')
        print(f"ep{ep} {r['auc']:.3f} -> {submit(H / tfile)} ({tfile})")


def cmd_status(a):
    c = load(a.name); best = {}
    for r in runs(c):
        if r['auc'] > best.get(r['ep'], {'auc': -1})['auc']:
            best[r['ep']] = r
    for ep in EPISODES:
        r = best.get(ep)
        print(f"ep{ep:>3} " + (f"{r['auc']:.3f}  {r['run'].name}@{r['step'] // 1000000}M" if r else '—'))
    if best:
        print(f'mean over {len(EPISODES)} episodes (missing count as 0): {sum(r["auc"] for r in best.values()) / len(EPISODES):.4f}')


def cmd_prune(a):
    print(sh(f"squeue -u {USERS} -h -o %j > {H}/.queued && python3 {H}/prune_keep.py {H}/.queued").stdout)


def cmd_pack(a):
    c = load(a.name); best = {}
    for r in runs(c):
        if r['auc'] > best.get(r['ep'], {'auc': -1})['auc']:
            best[r['ep']] = r
    plan = {}
    for ep, r in best.items():
        early = r['run'] / f"early_{r['step']}"
        plan[str(ep)] = dict(run=str(early), scene=str(scene_dir(c, r['variant'], ep)), kind='policy', tracking=r['auc'],
                             name=f"{r['run'].name}/{early.name}")
    missing = [e for e in EPISODES if str(e) not in plan]
    if missing:
        raise SystemExit(f'no finished run for episodes {missing}; not packing')
    out = REPO / 'reports/submission_candidate_2026-10-28/packed' / f"candidate_{c['name']}.parquet"
    if out.exists():
        raise SystemExit(f'{out} exists; packed candidates are never overwritten')
    plan_path = out.with_suffix('.plan.json'); plan_path.write_text(json.dumps(plan, indent=1))
    commit = sh(f'git -C {REPO} rev-parse HEAD').stdout.strip()
    py = DATA / 'venvs/kit/bin/python'
    r = sh(f'cd {REPO} && {py} tools/track3/pack_submission.py --plan {plan_path} --work {DATA}/pack_{c["name"]} '
           f'--code-commit https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/{commit} --python {py} -o {out}')
    print(r.stdout[-1500:], r.stderr[-1500:])
    print(f'tracking mean {sum(v["tracking"] for v in plan.values()) / len(plan):.4f}; upload only with the user\'s go-ahead')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    eps = lambda s: [int(x) for x in s.split(',')]
    p = sub.add_parser('init'); p.add_argument('name'); p.add_argument('bundles', nargs='?'); p.add_argument('--tag'); p.add_argument('--ns')
    p = sub.add_parser('build'); p.add_argument('name'); p.add_argument('--eps', type=eps, default=EPISODES)
    p.add_argument('--variant', choices=['raw', 'rs', 'both'], default='raw')
    p = sub.add_parser('check'); p.add_argument('name')
    p = sub.add_parser('train'); p.add_argument('name'); p.add_argument('--eps', type=eps, default=EPISODES)
    p.add_argument('--seeds', type=int, default=2, help='seeds per recipe'); p.add_argument('--seed0', type=int, default=200)
    p.add_argument('--recipe', help='comma list of DEF, WIDE, KEN, RELAXA', type=lambda s: [r for r in s.split(',') if r in RECIPES or ap.error(f'unknown recipe {r}')], default=['DEF', 'WIDE'])
    p.add_argument('--variant', choices=['raw', 'rs', 'both'], default='raw')
    p.add_argument('--where', choices=['preempt', 'general'], default='preempt')
    p.add_argument('--dry-run', action='store_true')
    for n in ('next', 'status', 'prune', 'pack'):
        sub.add_parser(n).add_argument('name')
    a = ap.parse_args()
    globals()[f'cmd_{a.cmd}'](a)


if __name__ == '__main__':
    main()
