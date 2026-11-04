"""Fresh runs (default + WIDE recipes) on the scene a pack plan uses for each episode.
python plan_scene_trials.py PLAN EPS SEEDS TAG [general]   -> trial paths in .plan_scene_trials.txt"""
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign import DEFAULT, BIG, RECIPES, H
plan = json.loads(Path(sys.argv[1]).read_text()); eps = sys.argv[2].split(','); seeds = [int(s) for s in sys.argv[3].split(',')]
tag = sys.argv[4]; general = len(sys.argv) > 5 and sys.argv[5] == 'general'
out = []
for ep in eps:
    scene = plan[ep]['scene']
    pq = glob.glob(f'{scene}/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave')
    if len(pq) != 1:
        print(f'ep{ep}: {len(pq)} parquets under {scene}'); continue
    extra = DEFAULT.replace('sim.njmax=1024 sim.nconmax=512', 'sim.njmax=2048 sim.nconmax=1024') if int(ep) in BIG else DEFAULT
    for recipe in ('DEF', 'WIDE'):
        for seed in seeds:
            name = f'ep{ep}_{tag}_{recipe}_s{seed}'
            t = dict(name=name, episode=int(ep), account='edmondwa', partition='preempt', qos='preempt_qos', launcher='fresh_keep.sbatch',
                     cpus=8, memory=BIG.get(int(ep), '64G'), time='02:00:00', scene=scene, parquet=pq[0], seed=seed,
                     total_environment_steps=100003840, save_replay=False, extra=extra + RECIPES[recipe].get('extra', ''),
                     hypothesis=f'Best-of-both: more seeds on the plan scene ({recipe}).')
            if general:
                t.update(account='sarveshp', partition='general', qos='normal', slurm_account='msimchow')
            p = H / f'{name.lower()}_trial.json'; p.write_text(json.dumps(t, indent=2)); out.append(str(p))
(H / '.plan_scene_trials.txt').write_text('\n'.join(out) + '\n'); print(len(out), 'trials')
