"""Default-recipe trials on Wentao's scored20 alignment-mix scenes, saving replay so the keeper can continue them.

python make_wentao_trials.py EP[,EP...] SEED[,SEED...] [general]
"""
import glob
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
W = Path('/data/user_data/edmondwa/v2d/wentao_mix/scenes_rs')
plan = json.load(open(H.parent / 'submission_candidate_2026-10-28/plan.json'))
base = json.load(open(H / 'ep10_nest_train_trial.json'))
DEFAULT = ('task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=1024 sim.nconmax=512 termination.wrist_position.threshold=0.3 '
           'termination.object_position.threshold=0.3 termination.object_orientation.threshold=1.5 training.checkpoint.save_interval_environment_steps=5000192 '
           'training.replay.capacity=1048576 training.replay.observation_storage_dtype=float16')
general = len(sys.argv) > 3 and sys.argv[3] == 'general'
for ep in map(int, sys.argv[1].split(',')):
    pq = glob.glob(f'{W}/ep{ep}/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave')
    if len(pq) != 1:
        print(f'ep{ep}: no scene, skipped'); continue
    for seed in map(int, sys.argv[2].split(',')):
        t = dict(base, name=f'ep{ep}_WRS_s{seed}', episode=ep, seed=seed, account='edmondwa', partition='preempt', qos='preempt_qos',
                 scene=str(W / f'ep{ep}'), parquet=pq[0], total_environment_steps=100003840, baseline_episode_auc=plan[str(ep)]['tracking'],
                 memory='64G', time='02:00:00', extra=DEFAULT, save_replay=False,
                 hypothesis="Wentao's scored20 alignment mix (better video alignment than Ken) with object rotations smoothed (sigma 2), default recipe.")
        t.pop('slurm_account', None)
        if general:
            t.update(account='sarveshp', partition='general', qos='normal', slurm_account='msimchow')
        if ep == 10:
            t['extra'] = DEFAULT.replace('sim.njmax=1024 sim.nconmax=512', 'sim.njmax=2048 sim.nconmax=1024'); t['memory'] = '128G'
        name = f'ep{ep}_wrs_s{seed}' + ('g' if general else '')
        json.dump(t, open(H / f'{name}_trial.json', 'w'), indent=2)
        print(name)
