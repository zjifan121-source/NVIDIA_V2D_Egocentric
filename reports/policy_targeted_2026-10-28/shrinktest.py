import sys, glob, numpy as np, trimesh, pyarrow.parquet as pq
sys.path.insert(0, '/home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28')
from cavity_scene import penetration, PROCESSED
from pathlib import Path
S = Path(sys.argv[1]); frames = [104, 130, 156]
t = pq.read_table(next(S.glob(f'{PROCESSED}/sequence_id=*/robot_name=sharpa_wave/*.parquet')))
names = t['object_body_names'][0].as_py()
load = lambda n: [trimesh.load(p, force='mesh') for p in sorted(glob.glob(f'{S}/{PROCESSED}/{n}_collision/*.stl'))]
blue, beige = load('blue_cup'), load('beige_cup')
centre = np.concatenate([h.vertices for h in beige]).mean(0)
for s in (1.0, 0.95, 0.9, 0.85, 0.8):
    shr = [trimesh.Trimesh((h.vertices - centre) * s + centre, h.faces).convex_hull for h in beige]
    rows = penetration(t, names, {names[0]: blue, names[1]: shr}, frames)
    print(s, [r['max_depth_cm'] for r in rows if r['object'] == names[1]], [r['max_depth_cm'] for r in rows if r['object'] == names[0]])
