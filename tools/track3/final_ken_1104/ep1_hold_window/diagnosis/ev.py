"""ep1n evidence: per-frame object motion + MANO tip/wrist signed distance to pot and lid, contact-label owners. GT-free.
usage: ev.py <scene dir> <out.npz>"""
import sys, glob, os
import numpy as np, trimesh, pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
scene, out = sys.argv[1], sys.argv[2]
PQ = glob.glob(scene + "/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave")[0]
tab = pq.read_table(glob.glob(PQ + "/*.parquet")[0])
d = {n: tab.column(n)[0].as_py() for n in tab.column_names}
proc = os.path.dirname(os.path.dirname(PQ))
names = d["object_body_names"]; print(names)
opos = np.asarray(d["object_body_position"], float); owxyz = np.asarray(d["object_body_wxyz"], float)
T = opos.shape[0]
orot = R.from_quat(owxyz[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(T, -1, 3, 3)
SAMP = []
for n in names:
    m = trimesh.load(f"{proc}/{n}.obj", force="mesh", process=False)
    np.random.seed(0); pts, fi = trimesh.sample.sample_surface(m, 80000)
    pts = np.vstack([pts, m.vertices]); nrm = np.vstack([m.face_normals[fi], m.vertex_normals])
    SAMP.append((cKDTree(pts), pts, nrm))
def sd(b, P, t):
    L = (P - opos[t, b]) @ orot[t, b]
    tree, pts, nrm = SAMP[b]; dist, i = tree.query(L)
    s = np.sign(np.einsum("ij,ij->i", L - pts[i], nrm[i])); s[s == 0] = 1
    return dist * s
TIPS = [4, 8, 12, 16, 20]
res = dict(opos=opos, names=np.array(names))
for side in ("left", "right"):
    J = np.asarray(d[f"mano_{side}_joints"], float)  # (T,21,3)
    D = np.zeros((T, len(names), 6))
    for t in range(T):
        for b in range(len(names)):
            D[t, b] = sd(b, J[t, [0] + TIPS], t)
    res[f"{side}_sd"] = D
    res[f"{side}_J"] = J
    pid = np.asarray(d[f"mano_{side}_object_contact_part_ids"], float)
    res[f"{side}_pid"] = pid
    res[f"{side}_tipd"] = np.asarray(d[f"mano_{side}_tips_distance"], float)
np.savez(out, **res)
# summary
z = opos[:, :, 2]
for t in range(0, T, 5):
    row = f"f{t:3d} pot dz {100*(z[t,0]-z[0,0]):5.1f} dxy {100*np.linalg.norm(opos[t,0,:2]-opos[0,0,:2]):5.1f} | lid dz {100*(z[t,1]-z[0,1]):5.1f} dxy {100*np.linalg.norm(opos[t,1,:2]-opos[0,1,:2]):5.1f}"
    for side in ("left", "right"):
        D = res[f"{side}_sd"][t]
        pid = res[f"{side}_pid"][t]
        row += f" | {side[0]} pot w{1000*D[0,0]:4.0f} tips " + " ".join(f"{1000*x:4.0f}" for x in D[0, 1:]) + \
               f" lid w{1000*D[1,0]:4.0f} tips " + " ".join(f"{1000*x:4.0f}" for x in D[1, 1:]) + \
               f" pid " + ",".join(f"{int(v)}:{int((pid==v).sum())}" for v in np.unique(pid) if v > 0)
    print(row)
