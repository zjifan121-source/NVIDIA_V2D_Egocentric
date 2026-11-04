"""comfix.py check SCENE [SCENE...]        report per object: URDF inertial origin (sim COM) vs area-weighted shell centroid of
                                         the visual mesh, and whether each lies inside the base footprint at the first and
                                         last frame (static stability at rest)
   comfix.py fix SCENE OUT OBJ [OBJ...]   copy SCENE to OUT and set those objects' URDF inertial origin to the shell centroid
                                         (uniform-shell mass assumption; the mesh origin from reconstruction is arbitrary),
                                         then repoint paths (relocate_scene.py). Same rule as the adopted ep5 scene (w7_attcom).
"""
import glob, os, re, shutil, subprocess, sys
import numpy as np, pyarrow.parquet as pq, trimesh
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as Rot
RELOCATE = os.environ.get("RELOCATE", os.path.expanduser("~/v2d_orchard/relocate_scene.py"))


def scene_info(sc):
    pqf = sorted(glob.glob(f"{sc}/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet"))[-1]
    seq = re.search(r"sequence_id=([^/]+)", pqf).group(1)
    t = pq.read_table(pqf, columns=["object_body_names", "object_body_position", "object_body_wxyz"])
    names = list(t.column("object_body_names")[0].as_py())
    P = np.array(t.column("object_body_position")[0].as_py(), float); Q = np.array(t.column("object_body_wxyz")[0].as_py(), float)
    if P.ndim == 2: P, Q = P[:, None], Q[:, None]
    return seq, names, P, Q


def shell_centroid(m):
    tri = np.asarray(m.triangles); a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return (tri.mean(1) * a[:, None]).sum(0) / a.sum()


def urdf_origin(path):
    txt = open(path).read()
    m = re.search(r"<inertial>\s*<origin xyz=\"([^\"]+)\"", txt)
    return np.array([float(x) for x in m.group(1).split()]), txt


def inside_footprint(Vw, comw):
    zb = Vw[:, 2].min(); base = Vw[Vw[:, 2] < zb + 0.003][:, :2]
    if len(base) < 4: return None, None
    try:
        hull = ConvexHull(base)
    except Exception:
        return None, None
    inside = Delaunay(base[hull.vertices]).find_simplex(comw[None, :2])[0] >= 0
    # signed distance to the hull boundary (positive inside)
    eq = hull.equations; d = -(eq[:, :2] @ comw[:2] + eq[:, 2]).min()
    return bool(inside), float(d)


mode = sys.argv[1]
if mode == "check":
    for sc in sys.argv[2:]:
        seq, names, P, Q = scene_info(sc)
        print(f"== {os.path.basename(sc)}")
        for bi, n in enumerate(names):
            obj = n.replace(seq + "_", "")
            proc = f"{sc}/assets/human_motion_data/ego_recon/processed/{seq}_{obj}"
            m = trimesh.load(f"{proc}_visual.stl", force="mesh", process=False)
            org, _ = urdf_origin(f"{proc}_rigid.urdf"); sh = shell_centroid(m)
            out = [f"  {obj:14s} COM(urdf) {np.round(org*100,1)} cm, shell centroid {np.round(sh*100,1)} cm, |diff| {100*np.linalg.norm(sh-org):.1f} cm"]
            for f in (0, P.shape[0] - 1):
                R = Rot.from_quat(Q[f, bi][[1, 2, 3, 0]]).as_matrix(); Vw = np.asarray(m.vertices) @ R.T + P[f, bi]
                for lab, c in (("COM", org), ("shell", sh)):
                    ins, d = inside_footprint(Vw, R @ c + P[f, bi])
                    out.append(f"      f{f}: {lab:5s} inside base footprint: {ins} (margin {None if d is None else round(100*d,1)} cm)")
            print("\n".join(out))
elif mode == "fix":
    sc, OUT, objs = os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]), sys.argv[4:]
    if os.path.exists(OUT): sys.exit(f"{OUT} exists")
    shutil.copytree(sc, OUT, symlinks=True)
    seq, names, P, Q = scene_info(OUT)
    for obj in objs:
        proc = f"{OUT}/assets/human_motion_data/ego_recon/processed/{seq}_{obj}"
        m = trimesh.load(f"{proc}_visual.stl", force="mesh", process=False); sh = shell_centroid(m)
        org, txt = urdf_origin(f"{proc}_rigid.urdf")
        old = re.search(r"(<inertial>\s*<origin xyz=\")([^\"]+)(\")", txt)
        txt2 = txt[:old.start(2)] + f"{sh[0]:.6f} {sh[1]:.6f} {sh[2]:.6f}" + txt[old.end(2):]
        if os.path.islink(f"{proc}_rigid.urdf"): os.remove(f"{proc}_rigid.urdf")
        open(f"{proc}_rigid.urdf", "w").write(txt2)
        print(f"{obj}: inertial origin {np.round(org, 6)} -> {np.round(sh, 6)}")
    r = subprocess.run([sys.executable, RELOCATE, OUT], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip().splitlines()[:1])
