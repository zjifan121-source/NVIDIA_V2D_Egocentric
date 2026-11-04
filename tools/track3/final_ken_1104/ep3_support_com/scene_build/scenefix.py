"""scenefix.py SCENE OUT [--obj NAME:DELTA_CM[:cap]]... [--support-top PRIM:TOPZ]... [--add-disk NAME:X:Y:R:TOPZ]...

Geometry/support-only reconstruction fixes on a COPY of a built V2D scene (SCENE is only read). Inputs come from GT-free
measurements (our stereo-depth table plane, our video masks); this tool only applies them:
  --obj NAME:DELTA_CM[:cap]  rim-anchored vertical height correction of an upright object's mesh: the mesh is stretched
        (DELTA>0, bottom moves down) or compressed (DELTA<0, bottom moves up) along the world vertical at frame 0 mapped
        into the object frame, top (rim) fixed: v' = v + down * h(v) * DELTA / H. Poses/hands unchanged. ':cap' also adds
        a 3 mm disk closing an open bottom.
  --support-top PRIM:TOPZ    move an existing support prim so its top is at TOPZ (sim world, m).
  --add-disk NAME:X:Y:R:TOPZ add a support cylinder (10 mm thick) with top at TOPZ.
Rewrites processed/<seq>_<obj>.obj + _visual.stl, score_input/meshes/episode_X/<obj>.obj, the matching bundle_cleaned
mesh (only if in the same frame), the support USD; repoints absolute paths (relocate_scene.py); writes
OUT/scenefix_report.json including each object's frame-0 bottom and the support prims.
"""
import glob, json, os, re, shutil, subprocess, sys
import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as Rot

SCENE, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
opts = sys.argv[3:]
OBJS, TOPS, DISKS = [], [], []
i = 0
while i < len(opts):
    k, v = opts[i], opts[i + 1]; i += 2
    if k == "--obj": OBJS.append(v)
    elif k == "--support-top": TOPS.append(v)
    elif k == "--add-disk": DISKS.append(v)
    else: sys.exit(f"unknown option {k}")
RELOCATE = os.environ.get("RELOCATE", os.path.expanduser("~/v2d_orchard/relocate_scene.py"))

if os.path.exists(OUT): sys.exit(f"{OUT} exists; refusing to overwrite")
shutil.copytree(SCENE, OUT, symlinks=False)
pqf = sorted(glob.glob(f"{OUT}/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet"))[-1]
t = pq.read_table(pqf, columns=["object_body_names", "object_body_position", "object_body_wxyz"])
names = list(t.column("object_body_names")[0].as_py())
pos = np.array(t.column("object_body_position")[0].as_py(), float); wxyz = np.array(t.column("object_body_wxyz")[0].as_py(), float)
if pos.ndim == 2: pos, wxyz = pos[:, None], wxyz[:, None]
seq = re.search(r"sequence_id=([^/]+)", pqf).group(1)
proc = f"{OUT}/assets/human_motion_data/ego_recon/processed"
usd = glob.glob(f"{OUT}/assets/human_motion_data/ego_recon/reconstructed_stage/*_support.usda")[0]
usd_txt = open(usd).read()
report = dict(scene=SCENE, out=OUT, sequence=seq, objects={}, support=[])


def cap_for(Vn, down, top_h):
    hn = Vn @ down - top_h; Hn = hn.max()
    ring = Vn[hn > Hn - 0.004]; c = ring.mean(0)
    rad = np.linalg.norm((ring - c) - np.outer((ring - c) @ down, down), axis=1)
    cyl = trimesh.creation.cylinder(radius=float(np.percentile(rad, 50)), height=0.003, sections=48)
    cyl.apply_transform(trimesh.geometry.align_vectors(np.array([0, 0, 1.0]), down))
    cyl.apply_translation(c - down * 0.0015 + down * (Hn - (c @ down - top_h)))
    return cyl


for spec in OBJS:
    parts = spec.split(":"); obj, delta = parts[0], float(parts[1]) / 100.0; want_cap = len(parts) > 2 and parts[2] == "cap"
    bi = next(j for j, n in enumerate(names) if n == obj or n.endswith(obj))
    R0 = Rot.from_quat(wxyz[0, bi][[1, 2, 3, 0]]).as_matrix(); p0 = pos[0, bi]
    down = R0.T @ np.array([0, 0, -1.0])
    pobj = f"{proc}/{seq}_{obj}.obj"
    V = np.asarray(trimesh.load(pobj, force="mesh", process=False).vertices, float)
    h = V @ down; top_h = h.min(); H = h.max() - top_h

    def mapv(P, delta=delta, down=down, top_h=top_h, H=H):
        hh = P @ down - top_h
        return P + np.outer(np.clip(hh, 0, None) * (delta / H), down)

    cap = cap_for(mapv(V), down, top_h) if want_cap else None

    def fix_obj_file(path):
        out, nv = [], 0
        for ln in open(path).read().split("\n"):
            if ln.startswith("v "):
                xyz = np.array([float(x) for x in ln.split()[1:4]]); rest = ln.split()[4:]
                x2 = mapv(xyz[None])[0]; nv += 1
                out.append("v " + " ".join(f"{x:.7f}" for x in x2) + (" " + " ".join(rest) if rest else ""))
            else:
                out.append(ln)
        if cap is not None:
            out += ["# scenefix bottom cap", "g scenefix_cap"] + [f"v {a:.7f} {b:.7f} {c:.7f}" for a, b, c in cap.vertices]
            out += [f"f {nv + 1 + a} {nv + 1 + b} {nv + 1 + c}" for a, b, c in cap.faces]
        open(path, "w").write("\n".join(out)); return nv

    files = {"processed_obj": fix_obj_file(pobj)}
    stl = f"{proc}/{seq}_{obj}_visual.stl"
    sm = trimesh.load(stl, force="mesh", process=False)
    sm2 = trimesh.Trimesh(mapv(np.asarray(sm.vertices, float)), np.asarray(sm.faces), process=False)
    if cap is not None: sm2 = trimesh.util.concatenate([sm2, cap])
    sm2.export(stl); files["visual_stl"] = len(sm.vertices)
    for so in glob.glob(f"{OUT}/score_input/meshes/episode_*/{obj}.obj"): files["score_obj"] = fix_obj_file(so)
    nproc = sum(1 for ln in open(f"{SCENE}/assets/human_motion_data/ego_recon/processed/{seq}_{obj}.obj") if ln.startswith("v "))
    for bm in glob.glob(f"{OUT}/bundle_cleaned/mesh*.obj"):
        if sum(1 for ln in open(bm) if ln.startswith("v ")) == nproc:
            bv = np.asarray(trimesh.load(bm, force="mesh", process=False).vertices)
            if np.abs(bv.min(0) - V.min(0)).max() < 1e-3 and np.abs(bv.max(0) - V.max(0)).max() < 1e-3:
                files["bundle_obj"] = (os.path.basename(bm), fix_obj_file(bm))
    V2 = np.asarray(trimesh.load(pobj, force="mesh", process=False).vertices, float)
    bottoms = [float((V2 @ Rot.from_quat(wxyz[f, bi][[1, 2, 3, 0]]).as_matrix().T + pos[f, bi])[:, 2].min()) for f in range(pos.shape[0])]
    report["objects"][obj] = dict(delta_cm=round(100 * delta, 2), height_before_cm=round(100 * H, 2), cap=want_cap,
                                  bottom_z_f0_before=round(float((V @ R0.T + p0)[:, 2].min()), 4),
                                  bottom_z_f0_after=round(bottoms[0], 4), bottom_z_last_after=round(bottoms[-1], 4), files=files)

for spec in TOPS:
    prim, topz = spec.split(":"); topz = float(topz)
    m = re.search(r'def Cylinder "%s"[^{]*\{(.*?)\n    \}' % re.escape(prim), usd_txt, re.S)
    body = m.group(1); hgt = float(re.search(r"double height = ([-\d.e]+)", body).group(1))
    tr = re.search(r"xformOp:translate = \(([^)]+)\)", body).group(1); x, y, z = [float(s) for s in tr.split(",")]
    body2 = body.replace(f"xformOp:translate = ({tr})", f"xformOp:translate = ({x}, {y}, {topz - hgt / 2})")
    usd_txt = usd_txt.replace(body, body2)
    report["support"].append(dict(prim=prim, top_before=round(z + hgt / 2, 4), top_after=round(topz, 4)))

for spec in DISKS:
    name, x, y, r, topz = spec.split(":"); x, y, r, topz = float(x), float(y), float(r), float(topz)
    disk = f'''    def Cylinder "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        uniform token axis = "Z"
        double height = 0.01
        color3f[] primvars:displayColor = [(0.8, 0.6, 0.4)]
        double radius = {r}
        double3 xformOp:translate = ({x}, {y}, {topz - 0.005})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
'''
    k = usd_txt.rstrip().rfind("}")
    usd_txt = usd_txt[:k] + disk + usd_txt[k:]
    report["support"].append(dict(prim=name, added=True, center=[x, y], radius=r, top=topz))

open(usd, "w").write(usd_txt)
r = subprocess.run([sys.executable, RELOCATE, OUT], capture_output=True, text=True)
report["relocate"] = (r.stdout + r.stderr).strip().splitlines()[:2]
json.dump(report, open(f"{OUT}/scenefix_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
