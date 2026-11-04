"""Default-recipe trials on the grafted Wentao scenes (Ken grasps on Wentao objects), late states kept.

python make_wg_trials.py EP[,EP...] SEED[,SEED...]
"""
import glob
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
G = Path('/data/user_data/edmondwa/v2d/wentao_mix/scenes_graft')
for ep in map(int, sys.argv[1].split(',')):
    pq = glob.glob(f'{G}/ep{ep}/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave')
    if len(pq) != 1:
        print(f'ep{ep}: no scene, skipped'); continue
    base = json.loads(sorted(H.glob(f'ep{ep}_wmix_s8*_trial.json'))[0].read_text())
    for seed in map(int, sys.argv[2].split(',')):
        t = dict(base, name=f'ep{ep}_WG_s{seed}', seed=seed, scene=str(G / f'ep{ep}'), parquet=pq[0],
                 launcher='fresh_keep.sbatch', save_replay=False, total_environment_steps=100003840,
                 account='edmondwa', partition='preempt', qos='preempt_qos',
                 hypothesis="Ken's grasps grafted onto Wentao's object trajectories (graft_grasps.py).")
        t.pop('slurm_account', None)
        (H / f'ep{ep}_wg_s{seed}g_trial.json').write_text(json.dumps(t, indent=2))
        print(t['name'])
