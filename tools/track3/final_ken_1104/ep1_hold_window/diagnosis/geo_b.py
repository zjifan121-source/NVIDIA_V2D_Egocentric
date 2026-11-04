"""ep1n arm-B feasibility (GT-free): pot handle cross-section near the grip and where the MANO left tips sit relative
to it, in the pot frame, over a hold window. usage: geo_b.py <scene> <a> <b>"""
import sys, glob, os
import numpy as np, trimesh, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
scene, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PQ = glob.glob(scene + "/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave")[0]
tab = pq.read_table(glob.glob(PQ + "/*.parquet")[0])
d = {n: tab.column(n)[0].as_py() for n in ("object_body_names", "object_body_position", "object_body_wxyz", "mano_left_joints", "mano_right_joints")}
proc = os.path.dirname(os.path.dirname(PQ))
names = d["object_body_names"]
opos = np.asarray(d["object_body_position"], float); ow = np.asarray(d["object_body_wxyz"], float)
T = len(opos)
rot = R.from_quat(ow[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(T, -1, 3, 3)
m = trimesh.load(f"{proc}/{names[0]}.obj", force="mesh", process=False)
V = np.asarray(m.vertices)
J = np.asarray(d["mano_left_joints"], float)
fr = np.arange(a, b)
loc = np.stack([(J[t] - opos[t, 0]) @ rot[t, 0] for t in fr])  # (n,21,3) pot frame
med = np.median(loc, axis=0)
up_obj = np.median(np.stack([rot[t, 0].T @ np.array([0, 0, 1.0]) for t in fr]), axis=0); up_obj /= np.linalg.norm(up_obj)
# handle axis: from pot centroid toward the median MANO middle MCP (joint 9), horizontal part
c0 = V.mean(axis=0)
ax = med[9] - c0; ax -= up_obj * (ax @ up_obj); ax /= np.linalg.norm(ax)
lat = np.cross(up_obj, ax)
print("pot bbox", V.min(0).round(3), V.max(0).round(3), "up_obj", up_obj.round(3), "handle axis", ax.round(3))
names_j = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky", 9: "middleMCP", 0: "wrist"}
for j, nm in names_j.items():
    p = med[j]; s = (p - c0) @ ax
    sec = V[np.abs((V - c0) @ ax - s) < 0.008]
    if len(sec) < 5:
        print(f"{nm:10s} s={s*100:5.1f} cm: no mesh section (beyond handle?)"); continue
    h = (sec - c0) @ up_obj; l = (sec - c0) @ lat
    ph, pl = (p - c0) @ up_obj, (p - c0) @ lat
    print(f"{nm:10s} along {s*100:5.1f} cm | section top {h.max()*100:5.1f} bottom {h.min()*100:5.1f} (thick {100*(h.max()-h.min()):.1f}) lat {l.min()*100:5.1f}..{l.max()*100:5.1f} (width {100*(l.max()-l.min()):.1f}) cm | tip height {ph*100:5.1f} lat {pl*100:5.1f} cm -> above top {100*(ph-h.max()):5.1f} cm, below bottom {100*(h.min()-ph):5.1f}")
print("handle extent along axis (vertices with lateral |l|<3cm and s>0):", np.percentile(((V - c0) @ ax)[np.abs((V - c0) @ lat) < 0.03], [50, 90, 99, 100]).round(3))
