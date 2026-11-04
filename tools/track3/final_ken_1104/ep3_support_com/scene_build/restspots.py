"""restspots.py SCENE FRAMES(comma): per object, bottom point (lowest mesh vertex) xyz and footprint (xy hull radius of
vertices within 1 cm of the bottom) at the given frames, plus distances between object footprints."""
import glob, re, sys, numpy as np, pyarrow.parquet as pq, trimesh
from scipy.spatial.transform import Rotation as Rot
sc = sys.argv[1]; frames = [int(x) for x in sys.argv[2].split(",")]
pqf = sorted(glob.glob(f"{sc}/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet"))[-1]
seq = re.search(r"sequence_id=([^/]+)", pqf).group(1)
t = pq.read_table(pqf, columns=["object_body_names", "object_body_position", "object_body_wxyz"])
names = list(t.column("object_body_names")[0].as_py())
P = np.array(t.column("object_body_position")[0].as_py(), float); Q = np.array(t.column("object_body_wxyz")[0].as_py(), float)
if P.ndim == 2: P, Q = P[:, None], Q[:, None]
for bi, n in enumerate(names):
    short = n.replace(seq + "_", "")
    V = np.asarray(trimesh.load(f"{sc}/assets/human_motion_data/ego_recon/processed/{seq}_{short}.obj", force="mesh", process=False).vertices)
    for f in frames:
        W = V @ Rot.from_quat(Q[f, bi][[1, 2, 3, 0]]).as_matrix().T + P[f, bi]
        zb = W[:, 2].min(); low = W[W[:, 2] < zb + 0.01]; c = low[:, :2].mean(0)
        rr = np.linalg.norm(W[:, :2] - c, axis=1).max()
        print(f"{short:14s} f{f:4d} bottom z {zb:.4f} footprint center ({c[0]:+.4f}, {c[1]:+.4f}) r_base {np.linalg.norm(low[:, :2]-c, axis=1).max():.3f} r_all {rr:.3f}")
