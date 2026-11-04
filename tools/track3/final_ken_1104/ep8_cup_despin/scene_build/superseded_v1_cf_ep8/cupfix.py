"""cupfix.py SCENE OUT OBJECT [OBJECT ...] [--min-gap 0.01]

Geometry-only reconstruction fix for UPRIGHT cup-like objects in a built V2D scene (writes a fixed copy to OUT; SCENE is
only read). Automatic rules, no per-episode numbers:
 1. BOTTOM COMPLETION: at frame 0 the object stands upright (mesh symmetry axis within 20 deg of vertical) on the support
    under it, but its mesh's lowest point is d > --min-gap above that support (the table plane, ignoring a disk that
    was built only for this object). The image silhouette matches the rim, so the mesh is missing its lower part
    (single-view SAM 3D cannot see a cup's base from above). The mesh is stretched along its axis with the rim fixed so
    the bottom reaches the support: v' = v + a * h(v) * d / H (a = axis pointing down, h = depth below the rim plane,
    H = rim-to-bottom height). Object poses, hands and contacts are unchanged; only the mesh files change.
    A disk built only for this object is moved down so its top is the table plane again.
 2. BOTTOM CAP: if a ray down the axis from the opening hits no surface (open tube), a closed 3 mm disk is added at the
    bottom ring so nothing falls through it in simulation (flash_chord collides objects via CoACD of the visual STL).
Files rewritten in OUT: processed/<seq>_<obj>.obj, processed/<seq>_<obj>_visual.stl, score_input/meshes/episode_X/<obj>.obj,
the matching bundle_cleaned/mesh*.obj, the support USD; absolute paths are then repointed to OUT (relocate_scene.py).
Writes OUT/cupfix_report.json.
"""
import glob, json, os, re, shutil, subprocess, sys
import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as Rot

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MIN_GAP = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--min-gap=")), 0.01))
SCENE, OUT, OBJS = os.path.abspath(args[0]), os.path.abspath(args[1]), args[2:]
RELOCATE = os.environ.get("RELOCATE", os.path.expanduser("~/v2d_orchard/relocate_scene.py"))


def ray_hits(tri, o, d):
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(d, e2); det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12; inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
    t0 = o - v0; u = (t0 * p).sum(1) * inv
    q = np.cross(t0, e1); v = (q * d).sum(1) * inv; t = (e2 * q).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return np.sort(t[hit])


def read_support(path):
    txt = open(path).read()
    prims = []
    for m in re.finditer(r'def Cylinder "([^"]+)"[^{]*\{(.*?)\n    \}', txt, re.S):
        name, body = m.group(1), m.group(2)
        h = float(re.search(r"double height = ([-\d.e]+)", body).group(1))
        r = float(re.search(r"double radius = ([-\d.e]+)", body).group(1))
        t = [float(x) for x in re.search(r"xformOp:translate = \(([^)]+)\)", body).group(1).split(",")]
        prims.append(dict(name=name, height=h, radius=r, t=t, top=t[2] + h / 2))
    return txt, prims


if os.path.exists(OUT):
    sys.exit(f"{OUT} exists; refusing to overwrite")
shutil.copytree(SCENE, OUT, symlinks=False)
pqf = sorted(glob.glob(f"{OUT}/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet"))[-1]
t = pq.read_table(pqf, columns=["object_body_names", "object_body_position", "object_body_wxyz", "raw_motion_file"])
names = list(t.column("object_body_names")[0].as_py())
pos0 = np.array(t.column("object_body_position")[0].as_py(), float)
wxyz0 = np.array(t.column("object_body_wxyz")[0].as_py(), float)
if pos0.ndim == 2: pos0, wxyz0 = pos0[:, None], wxyz0[:, None]
pos0, wxyz0 = pos0[0], wxyz0[0]
seq = re.search(r"sequence_id=([^/]+)", pqf).group(1)
episode = int(re.search(r"episode_0*(\d+)", " ".join(glob.glob(f"{OUT}/score_input/meshes/episode_*")) or "episode_0").group(1))
usd = glob.glob(f"{OUT}/assets/human_motion_data/ego_recon/reconstructed_stage/*_support.usda")[0]
usd_txt, prims = read_support(usd)
report = dict(scene=SCENE, out=OUT, sequence=seq, episode=episode, objects={})

for obj in OBJS:
    bi = next(i for i, n in enumerate(names) if n == obj or n.endswith(obj) or obj in n)
    R0 = Rot.from_quat(wxyz0[bi][[1, 2, 3, 0]]).as_matrix(); p0 = pos0[bi]
    proc_obj = f"{OUT}/assets/human_motion_data/ego_recon/processed/{seq}_{obj}.obj"
    m = trimesh.load(proc_obj, force="mesh", process=False)
    V = np.asarray(m.vertices, float)
    c = V.mean(0); ev, evec = np.linalg.eigh(np.cov((V - c).T))
    dis = [abs(ev[i] - ev[(i + 1) % 3]) + abs(ev[i] - ev[(i + 2) % 3]) for i in range(3)]
    ax = evec[:, int(np.argmax(dis))]
    down = R0.T @ np.array([0, 0, -1.0])
    if ax @ down < 0: ax = -ax
    tilt = float(np.degrees(np.arccos(np.clip(ax @ down, -1, 1))))
    W = V @ R0.T + p0
    zmin = float(W[:, 2].min())
    xy = W[np.argmin(W[:, 2]), :2]
    # support under the object: highest prim top below the object at its xy, excluding a disk built for this object
    own = [p for p in prims if p["name"].rsplit("_", 1)[0].endswith(obj)]
    others = [p for p in prims if p not in own and np.hypot(*(np.array(p["t"][:2]) - xy)) <= p["radius"] + 0.02 and p["top"] <= zmin + 1e-3]
    table = max([p["top"] for p in others], default=None)
    if table is None:  # no table under it: the table plane = top of the lowest-standing other object's own disk, else 1.0
        table = min([p["top"] for p in prims if p not in own], default=1.0)
    gap = zmin - table
    h = (V - c) @ ax; top_h = h.min(); H = h.max() - top_h
    info = dict(body_index=bi, tilt_deg=round(tilt, 1), bottom_z_f0=round(zmin, 4), support_top=round(table, 4),
                gap_cm=round(100 * gap, 2), height_cm=round(100 * H, 2), own_disk=[p["name"] for p in own])
    k_add = gap if (tilt <= 20 and gap > MIN_GAP) else 0.0
    # open-bottom test on the ORIGINAL mesh: ray from above the opening down the axis
    o = c + ax * (top_h - 0.05)
    hits = ray_hits(np.asarray(m.triangles), o, ax)
    inner_bottom = float(hits[0] - 0.05) if len(hits) else None
    open_bottom = inner_bottom is None or inner_bottom > H - 0.002
    info.update(inner_bottom_depth_cm=None if inner_bottom is None else round(100 * inner_bottom, 2), open_bottom=bool(open_bottom),
                extend_cm=round(100 * k_add, 2))

    def mapv(P):
        hh = (P - c) @ ax - top_h
        return P + np.outer(np.clip(hh, 0, None) * (k_add / H), ax)

    cap = None
    if open_bottom:
        Vn = mapv(V); hn = (Vn - c) @ ax - top_h; Hn = hn.max()
        ring = Vn[hn > Hn - 0.004]
        rad = np.linalg.norm((ring - c) - np.outer((ring - c) @ ax, ax), axis=1)
        r_cap = float(np.percentile(rad, 50))
        # 3 mm disk whose underside is flush with the bottom
        cyl = trimesh.creation.cylinder(radius=r_cap, height=0.003, sections=48)
        zax = np.array([0, 0, 1.0]); R = trimesh.geometry.align_vectors(zax, ax)
        cyl.apply_transform(R)
        cyl.apply_translation(c + ax * (top_h + Hn - 0.0015))
        cap = cyl
        info["cap_radius_cm"] = round(100 * r_cap, 2)

    def fix_obj_file(path):
        lines = open(path).read().split("\n"); out = []; nv = 0
        for ln in lines:
            if ln.startswith("v "):
                xyz = np.array([float(x) for x in ln.split()[1:4]])
                rest = ln.split()[4:]
                x2 = mapv(xyz[None])[0]
                out.append("v " + " ".join(f"{x:.7f}" for x in x2) + (" " + " ".join(rest) if rest else "")); nv += 1
            else:
                out.append(ln)
        if cap is not None:
            out.append("# cupfix bottom cap"); out.append("g cupfix_cap")
            out += [f"v {a:.7f} {b:.7f} {d:.7f}" for a, b, d in cap.vertices]
            out += [f"f {nv + 1 + a} {nv + 1 + b} {nv + 1 + d}" for a, b, d in cap.faces]
        open(path, "w").write("\n".join(out))
        return nv

    files = {}
    files["processed_obj"] = fix_obj_file(proc_obj)
    stl = f"{OUT}/assets/human_motion_data/ego_recon/processed/{seq}_{obj}_visual.stl"
    sm = trimesh.load(stl, force="mesh", process=False)
    sv = mapv(np.asarray(sm.vertices, float))
    sm2 = trimesh.Trimesh(sv, np.asarray(sm.faces), process=False)
    if cap is not None: sm2 = trimesh.util.concatenate([sm2, cap])
    sm2.export(stl); files["visual_stl"] = len(sv)
    for so in glob.glob(f"{OUT}/score_input/meshes/episode_*/{obj}.obj"):
        files["score_obj"] = fix_obj_file(so)
    # bundle_cleaned mesh file with the same vertex count as the original processed obj
    nproc = sum(1 for ln in open(f"{SCENE}/assets/human_motion_data/ego_recon/processed/{seq}_{obj}.obj") if ln.startswith("v "))
    for bm in glob.glob(f"{OUT}/bundle_cleaned/mesh*.obj"):
        if sum(1 for ln in open(bm) if ln.startswith("v ")) == nproc:
            bv = np.asarray(trimesh.load(bm, force="mesh", process=False).vertices)
            if np.abs(bv.min(0) - V.min(0)).max() < 1e-3 and np.abs(bv.max(0) - V.max(0)).max() < 1e-3:
                files["bundle_obj"] = (os.path.basename(bm), fix_obj_file(bm))
            else:
                files["bundle_obj_skipped_frame_differs"] = os.path.basename(bm)
    # own disk: move it down so its top is the support level again
    for p in own:
        if k_add > 0:
            newz = table - p["height"] / 2
            usd_txt = usd_txt.replace(f"xformOp:translate = ({p['t'][0]}, {p['t'][1]}, {p['t'][2]})",
                                      f"xformOp:translate = ({p['t'][0]}, {p['t'][1]}, {newz})")
            info.setdefault("disk_moved", []).append([p["name"], round(p["t"][2], 4), round(newz, 4)])
    # verify: new lowest point at frame 0
    m2 = trimesh.load(proc_obj, force="mesh", process=False)
    info["bottom_z_f0_after"] = round(float((np.asarray(m2.vertices) @ R0.T + p0)[:, 2].min()), 4)
    info["files"] = files
    report["objects"][obj] = info

open(usd, "w").write(usd_txt)
r = subprocess.run([sys.executable, RELOCATE, OUT], capture_output=True, text=True)
report["relocate"] = (r.stdout + r.stderr).strip().splitlines()[:3]
json.dump(report, open(f"{OUT}/cupfix_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
