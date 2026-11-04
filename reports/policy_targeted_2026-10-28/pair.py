import json, sys
import numpy as np, pyarrow.parquet as pq
plan = json.load(open('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/plan.json'))
ep = sys.argv[1]
df = pq.read_table(f"{plan[ep]['run']}/results/episode_{int(ep):06d}.parquet").to_pandas()
b = {s: g.sort_values('reference_frame') for s, g in df.groupby('object_id')}
n = len(b[0])
for f in range(0, n, max(1, n // 12)):
    r0, r1 = b[0].iloc[f], b[1].iloc[f]
    rd = np.array([r1[f'reference_position_{a}'] - r0[f'reference_position_{a}'] for a in 'xyz']) * 100
    ad = np.array([r1[f'achieved_position_{a}'] - r0[f'achieved_position_{a}'] for a in 'xyz']) * 100
    print(f"f{f:3d} ref obj1-obj0 xy {np.hypot(*rd[:2]):5.1f} dz {rd[2]:5.1f} | ach xy {np.hypot(*ad[:2]):5.1f} dz {ad[2]:5.1f}")
