"""Recipe sweep on Wentao's original scenes: candidate training recipes against the default, to pick one that carries over
to a new reconstruction bundle without per-episode tuning. Runs go to Edmond's preempt allocation and keep late states.

python make_recipe_trials.py RECIPE[,RECIPE...] EP[,EP...] SEED[,SEED...]
  WIDE    wider, linear-mapped exploration (lifted ep30's sweeper .32 -> .85 on Ken's scene)
  RELAXA  curriculum that weakens hand/contact imitation, so the policy can find its own grip
"""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
RECIPES = {
    'WIDE': dict(extra=' action.normalized_mapping=linear training.network.initial_normalized_std=0.3'
                       ' training.network.log_std_max=-0.5108256237659907 training.exploration.target_sigma=0.3'),
    'RELAXA': dict(curriculum='tracking_100m_relaxhand_a'),
}
recipes, eps, seeds = (sys.argv[i].split(',') for i in (1, 2, 3))
for ep in map(int, eps):
    base = json.loads(sorted(H.glob(f'ep{ep}_wmix_s8*_trial.json'))[0].read_text())
    base.pop('slurm_account', None); base.pop('curriculum', None)
    for r in recipes:
        for seed in map(int, seeds):
            t = dict(base, name=f'ep{ep}_R{r}_s{seed}', seed=seed, launcher='fresh_keep.sbatch', save_replay=False,
                     account='edmondwa', partition='preempt', qos='preempt_qos', total_environment_steps=100003840,
                     hypothesis=f'Recipe sweep on Wentao scene: {r} vs the default recipe.')
            t['extra'] = base['extra'] + RECIPES[r].get('extra', '')
            if 'curriculum' in RECIPES[r]: t['curriculum'] = RECIPES[r]['curriculum']
            (H / f'ep{ep}_r{r.lower()}_s{seed}_trial.json').write_text(json.dumps(t, indent=2))
            print(t['name'])
