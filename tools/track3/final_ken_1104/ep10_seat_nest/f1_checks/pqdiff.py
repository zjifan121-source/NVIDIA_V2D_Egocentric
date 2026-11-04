"""k10: compare two built scenes' processed parquets column by column (frame-resolved), plus md5s.
String columns are compared verbatim and after swapping the scene/sequence names (absolute paths differ by name).
usage: pqdiff.py <sceneA> <sceneB> <out.json> [tol=1e-5]"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

A, B, OUTJ = sys.argv[1:4]
TOL = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5


def pqf(d, sub="processed"):
    root = "assets/human_motion_data/ego_recon/processed" if sub == "processed" else "intermediate/ego_recon/loaded"
    fs = sorted(glob.glob(os.path.join(d, root, "*/robot_name=sharpa_wave/*.parquet")))
    return fs[0] if fs else None


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


nameA, nameB = os.path.basename(os.path.normpath(A)), os.path.basename(os.path.normpath(B))
res = {"A": A, "B": B, "tol": TOL}
for sub in ("processed", "loaded"):
    fa, fb = pqf(A, sub), pqf(B, sub)
    r = {"A_file": fa, "B_file": fb, "A_md5": md5(fa), "B_md5": md5(fb)}
    r["md5_equal"] = r["A_md5"] == r["B_md5"]
    ta, tb = pq.read_table(fa), pq.read_table(fb)
    r["schema_equal"] = ta.schema.equals(tb.schema)
    ra, rb = ta.to_pylist(), tb.to_pylist()
    r["rows"] = [len(ra), len(rb)]
    cols = {}
    for c in ta.column_names:
        if c not in tb.column_names:
            cols[c] = "missing in B"
            continue
        va, vb = ra[0][c], rb[0][c]
        try:
            xa = np.array(va, dtype=np.float64)
            xb = np.array(vb, dtype=np.float64)
            if xa.shape != xb.shape:
                cols[c] = f"shape {xa.shape} vs {xb.shape}"
                continue
            d = np.abs(xa - xb)
            d = np.where(np.isnan(xa) & np.isnan(xb), 0.0, d)
            m = float(np.nanmax(d)) if d.size else 0.0
            if m > TOL:
                ent = {"maxabs": m, "shape": list(xa.shape)}
                if xa.ndim >= 1 and xa.shape[0] > 1:
                    per = d.reshape(xa.shape[0], -1).max(1)
                    fr = np.flatnonzero(per > TOL)
                    ent["frames_changed"] = [int(fr[0]), int(fr[-1]), int(len(fr))] if len(fr) else []
                cols[c] = ent
            elif m > 0:
                cols[c] = {"maxabs_below_tol": m}
        except (TypeError, ValueError):
            sa, sb = json.dumps(va, default=str), json.dumps(vb, default=str)
            if sa == sb:
                continue
            seqA, seqB = f"ep10_h5_{nameA}", f"ep10_h5_{nameB}"
            sa2 = sa.replace(seqA, "SEQ").replace(nameA, "NAME")
            sb2 = sb.replace(seqB, "SEQ").replace(nameB, "NAME")
            cols[c] = "differs only by scene name" if sa2 == sb2 else f"differs: {sa[:120]} | {sb[:120]}"
    r["columns_differing"] = cols
    r["n_columns"] = len(ta.column_names)
    res[sub] = r
    print(sub, "md5", r["A_md5"][:8], r["B_md5"][:8], "schema_equal", r["schema_equal"], "cols", len(ta.column_names))
    for c, v in cols.items():
        print("  ", c, v)
# bundle_cleaned arrays
na = dict(np.load(os.path.join(A, "bundle_cleaned/result.npz"), allow_pickle=True))
nb = dict(np.load(os.path.join(B, "bundle_cleaned/result.npz"), allow_pickle=True))
bd = {}
for k in sorted(set(na) | set(nb)):
    if k not in na or k not in nb:
        bd[k] = "missing"
        continue
    x, y = na[k], nb[k]
    if x.shape != y.shape:
        bd[k] = f"shape {x.shape} vs {y.shape}"
    elif x.dtype.kind in "fc":
        m = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
        if m > 0:
            per = np.abs(x.astype(np.float64) - y.astype(np.float64)).reshape(x.shape[0], -1).max(1) if x.ndim > 1 else None
            fr = np.flatnonzero(per > 0) if per is not None else []
            bd[k] = {"maxabs": m, "frames_changed": ([int(fr[0]), int(fr[-1]), int(len(fr))] if len(fr) else [])}
    elif not np.array_equal(x, y):
        bd[k] = "differs"
res["bundle_cleaned_diff"] = bd
print("bundle_cleaned differing keys:", json.dumps(bd))
# meshes: every mesh file under processed/ and score_input/
mm = {}
for d, lab in ((A, "A"), (B, "B")):
    for p in sorted(glob.glob(os.path.join(d, "assets/human_motion_data/ego_recon/processed/*.obj")) +
                    glob.glob(os.path.join(d, "assets/human_motion_data/ego_recon/processed/*.stl")) +
                    glob.glob(os.path.join(d, "assets/human_motion_data/ego_recon/processed/**/*.stl"), recursive=True) +
                    glob.glob(os.path.join(d, "score_input/meshes/*/*.obj"))):
        key = os.path.relpath(p, d).replace(f"ep10_h5_{os.path.basename(os.path.normpath(d))}", "SEQ")
        mm.setdefault(key, {})[lab] = md5(p)
res["mesh_md5"] = mm
res["mesh_all_equal"] = all(v.get("A") == v.get("B") for v in mm.values())
print("mesh md5 equal:", res["mesh_all_equal"], "files", len(mm))
for k, v in mm.items():
    print("  ", k, v.get("A", "-")[:8], v.get("B", "-")[:8])
# reports
for lab, d in (("A", A), ("B", B)):
    rep = json.load(open(os.path.join(d, "report.json")))
    res[f"report_{lab}"] = {"cleaning": {k: v for k, v in rep.get("cleaning", {}).items()
                                         if not (isinstance(v, list) and len(v) > 20)},
                            "output_check": rep.get("output_check")}
json.dump(res, open(OUTJ, "w"), indent=1)
print("PQDIFF_DONE")
