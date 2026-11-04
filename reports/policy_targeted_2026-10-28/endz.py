import json, sys
import pyarrow.parquet as pq
plan = json.load(open('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/plan.json'))
for ep in sys.argv[1:]:
    df = pq.read_table(f"{plan[ep]['run']}/results/episode_{int(ep):06d}.parquet").to_pandas()
    for slot, b in df.groupby('object_id'):
        b = b.sort_values('reference_frame')
        n = len(b)
        out = []
        for f in (0, n//4, n//2, 3*n//4, n-1):
            r = b.iloc[f]
            out.append(f"f{int(r.reference_frame)} ref {100*r.reference_position_z:.1f} ach {100*r.achieved_position_z:.1f}")
        print(f"ep{ep} obj{slot}: " + " | ".join(out))
