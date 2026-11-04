"""hand5: compare two built scenes (bundle_cleaned arrays, cleaning report, processed parquet numeric columns).
usage: python cmp_scene.py <sceneA dir> <sceneB dir>"""
import sys, json, glob, os
import numpy as np
A, B = sys.argv[1], sys.argv[2]
def npz(d): return dict(np.load(os.path.join(d, "bundle_cleaned/result.npz"), allow_pickle=True))
a, b = npz(A), npz(B)
diff = []
for k in sorted(set(a) | set(b)):
    if k not in a or k not in b: diff.append((k, "missing")); continue
    x, y = a[k], b[k]
    if x.shape != y.shape: diff.append((k, f"shape {x.shape} vs {y.shape}")); continue
    if x.dtype.kind in "fc":
        m = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
        if m > 0: diff.append((k, f"maxabs {m:.3g}"))
    elif not np.array_equal(x, y): diff.append((k, "differs"))
print("bundle_cleaned keys", len(a), "differing:", diff if diff else "none")
ra, rb = json.load(open(os.path.join(A, "report.json"))), json.load(open(os.path.join(B, "report.json")))
ca, cb = ra.get("cleaning", {}), rb.get("cleaning", {})
print("cleaning report equal:", ca == cb)
if ca != cb:
    for k in sorted(set(ca) | set(cb)):
        if ca.get(k) != cb.get(k): print("  cleaning", k, ca.get(k), "|", cb.get(k))
import pyarrow.parquet as pq
def pqfiles(d): return sorted(glob.glob(os.path.join(d, "assets/human_motion_data/ego_recon/processed/*/robot_name=sharpa_wave/*.parquet")))
fa, fb = pqfiles(A), pqfiles(B)
print("parquet files", len(fa), len(fb))
for x, y in zip(fa, fb):
    ta, tb = pq.read_table(x).to_pandas(), pq.read_table(y).to_pandas()
    print(" ", os.path.basename(x), ta.shape, tb.shape)
    for c in ta.columns:
        if c not in tb.columns: print("   missing col", c); continue
        va, vb = ta[c].values, tb[c].values
        try:
            xa = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in va]); xb = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vb])
            m = float(np.nanmax(np.abs(xa - xb))) if xa.size else 0.0
            if m > 1e-6: print(f"   col {c}: maxabs {m:.3g}")
        except Exception:
            if not all(str(p) == str(q) for p, q in zip(va, vb)): print(f"   col {c}: non-numeric differs (e.g. {str(va[0])[:60]} | {str(vb[0])[:60]})")
print("CMP_DONE")
