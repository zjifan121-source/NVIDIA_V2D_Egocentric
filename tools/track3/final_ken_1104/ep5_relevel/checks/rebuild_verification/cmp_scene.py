"""cmp_scene.py A B : compare two built V2D scene dirs (rebuild check). Numeric parquet columns -> max abs diff;
string columns -> equal after reducing absolute paths to their part below the scene dir; meshes -> byte-equal or max
vertex diff; other files -> sha256."""
import glob, hashlib, json, os, sys
import numpy as np, pyarrow.parquet as pq, trimesh

A, B = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])


def rel_files(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            if os.path.islink(p):
                continue
            out[os.path.relpath(p, root).replace("\\", "/")] = p
    return out


fa, fb = rel_files(A), rel_files(B)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
res = {"only_in_A": sorted(set(fa) - set(fb)), "only_in_B": sorted(set(fb) - set(fa)), "files": {}}


def strip(v):
    if isinstance(v, list):
        return [strip(x) for x in v]
    if isinstance(v, str) and ("/" in v or "\\" in v) and "assets" in v.replace("\\", "/"):
        return v.replace("\\", "/").split("/assets/", 1)[1]
    return v


for k in sorted(set(fa) & set(fb)):
    pa_, pb_ = fa[k], fb[k]
    if os.path.getsize(pa_) == 0 and os.path.getsize(pb_) == 0:
        res["files"][k] = "both empty"
        continue
    if sha(pa_) == sha(pb_):
        res["files"][k] = "identical"
        continue
    if k.endswith(".parquet"):
        ta, tb = pq.read_table(pa_), pq.read_table(pb_)
        rep = {"columns_equal": ta.column_names == tb.column_names, "numeric_max_abs_diff": {}, "string_cols_differ": []}
        for c in ta.column_names:
            va, vb = ta.column(c).to_pylist(), tb.column(c).to_pylist()
            try:
                xa, xb = np.array(va, float), np.array(vb, float)
                d = float(np.nanmax(np.abs(xa - xb))) if xa.size else 0.0
                if d != 0.0:
                    rep["numeric_max_abs_diff"][c] = d
            except (TypeError, ValueError):
                if strip(va) != strip(vb):
                    rep["string_cols_differ"].append(c)
        rep["max_numeric_diff_overall"] = max(rep["numeric_max_abs_diff"].values(), default=0.0)
        res["files"][k] = rep
    elif k.endswith((".obj", ".stl")):
        ma = trimesh.load(pa_, force="mesh", process=False)
        mb = trimesh.load(pb_, force="mesh", process=False)
        same_shape = ma.vertices.shape == mb.vertices.shape and ma.faces.shape == mb.faces.shape
        d = float(np.abs(np.asarray(ma.vertices) - np.asarray(mb.vertices)).max()) if same_shape else None
        fe = bool(np.array_equal(ma.faces, mb.faces)) if same_shape else False
        res["files"][k] = {"bytes_differ": True, "same_shape": same_shape, "vertex_max_abs_diff_m": d, "faces_equal": fe}
    else:
        ta_ = open(pa_, encoding="utf-8", errors="replace").read()
        tb_ = open(pb_, encoding="utf-8", errors="replace").read()
        res["files"][k] = {"bytes_differ": True, "text_equal_after_path_strip": strip(ta_) == strip(tb_),
                           "A_head": ta_[:400], "B_head": tb_[:400]}
print(json.dumps(res, indent=1, default=str))
