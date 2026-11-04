"""k10 F1 step 1 (GT-free): blue_toprise.json from the A2 skeptic's FS-depth measurements.
rise_cm[f] = p95[f] - median(p95[f0-35]) of the visible blue cup (SAM 3 blue mask minus 8-px-dilated hand
masks, eroded 4 px; FS depth; heights above the table plane of the loader world), plus the per-frame
number of valid depth points n. Source: ~/v2d_ops/kenc/diag/skeptic_ep10A2/sk10a2.json observed[f]['blue_cup'].
usage: python make_toprise.py <sk10a2.json> <out.json>"""
import json
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
obs = json.load(open(src))["observed"]
N = len(obs)
p95 = np.array([obs[str(f)]["blue_cup"]["p95"] if obs[str(f)].get("blue_cup") else np.nan for f in range(N)])
n = np.array([obs[str(f)]["blue_cup"]["n"] if obs[str(f)].get("blue_cup") else 0 for f in range(N)], int)
REST = (0, 35)  # plan F1 implementation step 1: median(p95[f0-35]) (A2 verdict: rest level 16.61 +- 0.12 on f0-35)
rest = float(np.nanmedian(p95[REST[0]:REST[1] + 1]))
res = {"source": src, "object": "blue_cup", "rest_frames": list(REST), "rest_top_cm": round(rest, 3),
       "rest_top_cm_f0_12_for_reference": round(float(np.nanmedian(p95[:13])), 3),
       "p95_cm": {str(f): (None if np.isnan(p95[f]) else float(p95[f])) for f in range(N)},
       "rise_cm": {str(f): (None if np.isnan(p95[f]) else round(float(p95[f] - rest), 3)) for f in range(N)},
       "n": {str(f): int(n[f]) for f in range(N)}}
json.dump(res, open(out, "w"), indent=1)
print("frames", N, "rest_top_cm", res["rest_top_cm"], "(f0-12:", res["rest_top_cm_f0_12_for_reference"], ")",
      "max rise", round(float(np.nanmax(p95 - rest)), 2), "min n", int(n.min()))
