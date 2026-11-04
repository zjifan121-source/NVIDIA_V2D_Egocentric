"""Does the reference nesting pose penetrate the meshes? python nestcheck.py EP FRAME..."""
import json, sys, glob
import numpy as np, pyarrow.parquet as pq, trimesh
from scipy.spatial.transform import Rotation
plan = json.load(open('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/plan.json'))
ep = sys.argv[1]; e = plan[ep]
t = pq.read_table(f"{e['run']}/results/episode_{int(ep):06d}.parquet").to_pydict()
t = {k: np.asarray(v) for k, v in t.items()}
def row(slot, f):
    i = np.flatnonzero((t['object_id'] == slot) & (t['reference_frame'] == f))[0]
    return {k: v[i] for k, v in t.items()}
meshdir = f"{e['scene']}/score_input/meshes/episode_{int(ep):06d}"
objs = json.load(open('/home/edmondwa/NVIDIA_V2D_Egocentric/tools/track3/episodes.json'))
roster = [o['name'] for o in next(x for x in objs['episodes'] if x['episode'] == int(ep))['objects']]
meshes = [trimesh.load(f'{meshdir}/{n}.obj', force='mesh', process=False) for n in roster]
print(roster, [m.is_watertight for m in meshes])
for f in map(int, sys.argv[2:]):
    for kind in ('reference', 'achieved'):
        world = []
        for slot, m in enumerate(meshes):
            r = row(slot, f)
            R = Rotation.from_quat([r[f'{kind}_quaternion_{a}'] for a in 'xyzw']).as_matrix()
            world.append(m.vertices @ R.T + [r[f'{kind}_position_{a}'] for a in 'xyz'])
        pts = world[1][::max(1, len(world[1]) // 3000)]
        r0 = row(0, f)
        R0 = Rotation.from_quat([r0[f'{kind}_quaternion_{a}'] for a in 'xyzw']).as_matrix()
        local = (pts - [r0[f'{kind}_position_{a}'] for a in 'xyz']) @ R0
        inside = meshes[0].contains(local)
        depth = np.abs(trimesh.proximity.signed_distance(meshes[0], local[inside])) if inside.any() else np.zeros(1)
        print(f'f{f} {kind}: {inside.mean()*100:.1f}% of {roster[1]} verts inside {roster[0]}, max depth {100*depth.max():.2f} cm')
