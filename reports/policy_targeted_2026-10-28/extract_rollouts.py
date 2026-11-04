"""Dump reference vs achieved object trajectories of every selected rollout, for the status page."""
import json
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

R = '/home/edmondwa/NVIDIA_V2D_Egocentric'
plan = json.load(open(f'{R}/reports/submission_candidate_2026-10-28/plan.json'))
roster = {e['episode']: [o['name'] for o in e['objects']] for e in json.load(open(f'{R}/tools/track3/episodes.json'))['episodes']}
out = []
for ep, entry in sorted(plan.items(), key=lambda kv: int(kv[0])):
    res = json.load(open(f"{entry['run']}/results/episode_{int(ep):06d}.json"))['metrics']
    t = {k: np.asarray(v) for k, v in pq.read_table(f"{entry['run']}/results/episode_{int(ep):06d}.parquet").to_pydict().items()}
    bodies = []
    for slot, name in enumerate(roster[int(ep)]):
        m = t['object_id'] == slot
        order = np.argsort(t['reference_frame'][m])
        g = lambda k: t[k][m][order]
        ref = np.stack([g(f'reference_position_{a}') for a in 'xyz'], 1)
        ach = np.stack([g(f'achieved_position_{a}') for a in 'xyz'], 1)
        rq = Rotation.from_quat(np.stack([g(f'reference_quaternion_{a}') for a in 'xyzw'], 1))
        aq = Rotation.from_quat(np.stack([g(f'achieved_quaternion_{a}') for a in 'xyzw'], 1))
        rot = np.degrees((rq.inv() * aq).magnitude())
        bodies.append(dict(name=name, auc=round(float(res['add_auc_per_body'][slot]), 4),
                           ref=np.round(100 * (ref - ref[0]), 2).tolist(), ach=np.round(100 * (ach - ref[0]), 2).tolist(),
                           pos_err_cm=np.round(100 * np.linalg.norm(ach - ref, axis=1), 2).tolist(), rot_err_deg=np.round(rot, 1).tolist()))
    out.append(dict(episode=int(ep), auc=round(entry['tracking'], 4), run=entry['name'], frames=len(bodies[0]['ref']), bodies=bodies))
    print(ep, entry['name'], round(entry['tracking'], 4), flush=True)
json.dump(dict(mean=round(float(np.mean([e['auc'] for e in out])), 4), episodes=out), open(f'{R}/reports/policy_targeted_2026-10-28/status_rollouts.json', 'w'))
