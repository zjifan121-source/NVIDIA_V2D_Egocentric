"""compare_review_numbers.py BASE_numbers.json NEW_numbers.json
GT-free check used on 4 Nov 2026 (00:16 EST) after the support edit: per object, the reference mesh silhouette IoU against
our SAM 3 masks (whole clip and f0-20) and the reference height above the sim support at rest (median | max over frames
where |h| < 6 cm), base scene vs edited scene. Inputs are the *_numbers.json files written by render_review.py
(depth_table/review_tool/render_review.py --scene <scene> --episode 3 --label <label> --out <dir>).
The edit must leave the IoU unchanged (no pose/mesh change) and bring the rest heights towards 0.
Same computation as the inline check run in the session (copied here as a script)."""
import json
import sys

import numpy as np

a, b = sys.argv[1:3]
arr = lambda v: np.array([np.nan if x is None else x for x in v], float)  # noqa: E731
A, B = json.load(open(a))["per_frame"]["objects"], json.load(open(b))["per_frame"]["objects"]
for obj in A:
    ia, ib = arr(A[obj]["mask_iou"]), arr(B[obj]["mask_iou"])
    ha, hb = arr(A[obj]["h_support_cm"]), arr(B[obj]["h_support_cm"])
    rest = np.abs(ha) < 6
    print(f"{obj:14s} IoU all {np.nanmean(ia):.3f}->{np.nanmean(ib):.3f}  f0-20 {np.nanmean(ia[:21]):.3f}->{np.nanmean(ib[:21]):.3f}  "
          f"| h_support at rest (median|max) {np.nanmedian(ha[rest]):+.2f}|{np.nanmax(ha[rest]):+.2f} -> "
          f"{np.nanmedian(hb[rest]):+.2f}|{np.nanmax(hb[rest]):+.2f} cm; min {np.nanmin(hb):+.2f}")
