"""Pairwise CD-O-style distance between local meshes, using the kit's own CD-O functions (scale-normalised to the
first mesh's RMS radius, 25 PCA starts + trimmed ICP, symmetric chamfer in cm). Usage: cdo_pairs.py A.obj B.obj [...]
(pairs consecutive arguments: A-B, C-D, ...). Also prints height/extent of every mesh."""
import sys, importlib.util, numpy as np
KIT = r"C:\v2d_demo\diag\ep10_verify\scorer_2\kit\metric_code\track_3\CD-O.py"
spec = importlib.util.spec_from_file_location("cdo", KIT); cdo = importlib.util.module_from_spec(spec); spec.loader.exec_module(cdo)

def load_obj(p):
    V, F = [], []
    for line in open(p, encoding="utf-8", errors="ignore"):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) for t in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(V, float), np.asarray(F, np.int64)

args = sys.argv[1:]
meshes = {a: load_obj(a) for a in dict.fromkeys(args)}
for a, (V, F) in meshes.items():
    c, r = cdo._surface_moments(V, F)
    ext = V.max(0) - V.min(0)
    print(f"{a.split('meshes/')[-1] if 'meshes/' in a else a}: {len(V)} V {len(F)} F; extent cm {np.round(100*ext,2)}; RMS radius {100*r:.2f} cm")
for i in range(0, len(args) - 1, 2):
    A, B = meshes[args[i]], meshes[args[i + 1]]
    d = [cdo._object_chamfer_cm(A, B, s) for s in (11, 22, 33)]
    print(f"CD({args[i].split('/')[-3] if '/' in args[i] else args[i]} -> {args[i+1].split('/')[-3] if '/' in args[i+1] else args[i+1]}) = {np.mean(d):.3f} cm (seeds {np.round(d,3)})")
