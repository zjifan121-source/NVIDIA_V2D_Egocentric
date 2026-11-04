"""anch.py ROLLOUT.parquet MESHDIR ANCHOR [ANCHOR2...]: own-ref add_auc of a rollout, plain (evaluator-style, no alignment)
and anchored (AUC.py-style: whole rollout scene rigidly aligned so the anchor's frame-0 pose equals its reference f0 pose).
Mesh per body = MESHDIR/<name>.obj whose name is a substring of the body name. Uses env 0. Per-object AUC over 1-9 cm."""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, r"C:\v2d_demo\diag\ep5_skepA1")
from an_common import load_obj, load_rollout, xf, auc_cm, rot_angle_deg
import pyarrow.parquet as pq
roll, mdir, anchors = sys.argv[1], sys.argv[2], sys.argv[3:]
t = pq.read_table(roll, columns=["object_body_name", "reference_frame", "environment_id"])
env = np.array(t.column("environment_id").to_pylist()); sel = env == env.min()
bn = np.array([str(x) for x in t.column("object_body_name").to_pylist()])[sel]
fr = np.array(t.column("reference_frame").to_pylist())[sel]
bodies = list(dict.fromkeys(bn)); N = int(fr.max()) + 1
R = load_rollout(roll, bodies, N)
meshes = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(mdir, "*.obj"))}
def mesh_for(b):
    c = [k for k in meshes if k in b]; c.sort(key=len, reverse=True); return c[0]
names = [mesh_for(b) for b in bodies]
V = [load_obj(meshes[n])[0] for n in names]
ai = [names.index(a) for a in anchors if a in names][0]
A = R[ai]["ref"][0] @ np.linalg.inv(R[ai]["ach"][0])   # rollout -> reference alignment at f0
res = {}
for mode, G in (("plain", np.eye(4)), ("anchored", A)):
    per = []
    for k, n in enumerate(names):
        add = []
        for f in range(N):
            Ta, Tr = R[k]["ach"][f], R[k]["ref"][f]
            if not (np.isfinite(Ta).all() and np.isfinite(Tr).all()): continue
            add.append(100 * np.linalg.norm(xf(G @ Ta, V[k]) - xf(Tr, V[k]), axis=1).mean())
        per.append(auc_cm(add))
    res[mode] = dict(auc=round(float(np.mean(per)), 4), per_body=[round(p, 4) for p in per])
rot0 = float(rot_angle_deg(R[ai]["ach"][0][:3, :3], R[ai]["ref"][0][:3, :3]))
print(json.dumps(dict(rollout=roll.replace("\\", "/").split("/runs/")[-1][-90:], bodies=names, anchor=names[ai], N=N,
                      anchor_f0_rot_deg=round(rot0, 2), **res)))
