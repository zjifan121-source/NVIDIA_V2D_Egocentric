# ep3 skeptic (white_pot COM): verify the sim COM / footprint numbers on the SIM collision shapes (cached CoACD hulls)
# and run a no-hands MuJoCo settle of the bob1 ep3 reference poses with COM at origin / shell centroid / volume centroid.
# usage (Babel, flash_chord venv python): ep3com_phys.py <scene dir> <out.json>
import sys, glob, json, os, numpy as np, pyarrow.parquet as pq, trimesh, mujoco
SC = sys.argv[1].rstrip('/'); OUT = sys.argv[2]
FC = '/data/user_data/haisenk/v2d/v2d_train/baseline/robotic_grounding/flash_chord'
CACHE = FC + '/src/flash_chord/assets/.cache/convex'
proc = SC + '/assets/human_motion_data/ego_recon/processed/'
pqp = glob.glob(proc + 'sequence_id=*/robot_name=*/*.parquet')[0]
t = pq.read_table(pqp)
def col(k): return np.array(t.column(k)[0].as_py(), dtype=float)
P = col('object_body_position'); Q = col('object_body_wxyz')
names = t.column('object_body_names')[0].as_py()
print('parquet', pqp, P.shape, Q.shape, names, flush=True)
usd = glob.glob(SC + '/assets/human_motion_data/ego_recon/reconstructed_stage/*.usda')[0]
txt = open(usd).read()
import re
r_ = float(re.search(r'double radius = ([0-9.eE+-]+)', txt).group(1)); h_ = float(re.search(r'double height = ([0-9.eE+-]+)', txt).group(1))
c_ = [float(x) for x in re.search(r'xformOp:translate = \(([^)]*)\)', txt).group(1).split(',')]
SUP = dict(r=r_, h=h_, c=c_, top=c_[2] + h_ / 2)
print('support', SUP, flush=True)

def q2m(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def hull2d(pts):
    pts = np.unique(np.round(np.asarray(pts, float), 7), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def cr(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(tuple(p))
    for p in pts[::-1]:
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
        up.append(tuple(p))
    return np.array(lo[:-1] + up[:-1])

def margin(H, p):
    """signed distance point->polygon boundary: + inside, - outside (true distance)"""
    p = np.asarray(p, float)[:2]; inside = True; dmin = 1e9
    for i in range(len(H)):
        a = H[i]; b = H[(i + 1) % len(H)]; e = b - a
        if e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) < 0: inside = False
        tt = np.clip(np.dot(p - a, e) / np.dot(e, e), 0, 1); dmin = min(dmin, float(np.linalg.norm(p - (a + tt * e))))
    return dmin if inside else -dmin

res = {'scene': SC, 'parquet': pqp, 'support': SUP, 'objects': {}}
OBJ = {}
for k, nm in enumerate(names):
    short = 'white_pot' if nm.endswith('white_pot') else 'wooden_spoon'
    urdf = open(proc + nm + '_rigid.urdf').read()
    stlname = re.search(r'<collision>.*?filename="([^"]+)"', urdf, re.S).group(1)
    inert = re.search(r'<inertial>\s*<origin xyz="([^"]*)"', urdf).group(1)
    stl = os.path.normpath(os.path.join(proc, stlname))
    m = trimesh.load(stl, force='mesh', process=True)
    V = np.asarray(m.vertices); tri = np.asarray(m.triangles); ar = np.asarray(m.area_faces)
    shell = (tri.mean(1) * ar[:, None]).sum(0) / ar.sum()
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v6 = np.einsum('ij,ij->i', a, np.cross(b, c)); volc = (v6[:, None] * (a + b + c) / 4).sum(0) / v6.sum()
    # compare with the textured OBJ used by earlier analysis
    objp = proc + nm + '.obj'
    try:
        mo = trimesh.load(objp, force='mesh', process=True)
        objcmp = dict(obj_bbox_min=np.round(mo.bounds[0], 5).tolist(), obj_bbox_max=np.round(mo.bounds[1], 5).tolist(),
                      stl_bbox_min=np.round(m.bounds[0], 5).tolist(), stl_bbox_max=np.round(m.bounds[1], 5).tolist())
    except Exception as e:
        objcmp = {'err': str(e)}
    # matching CoACD hull set in the shared flash_chord cache
    cands = []
    for f in glob.glob(CACHE + '/*.npz'):
        try:
            d = np.load(f); n = int(d['n']); HV = [np.asarray(d[f'v{i}'], float) for i in range(n)]
        except Exception:
            continue
        allv = np.concatenate(HV)
        diff = float(np.abs(allv.min(0) - V.min(0)).max() + np.abs(allv.max(0) - V.max(0)).max())
        cands.append((diff, f, n, HV))
    cands.sort(key=lambda x: x[0])
    best = cands[0]
    HV = best[3]; HA = np.concatenate(HV)
    OBJ[short] = dict(k=k, V=V, HV=HV, shell=shell, volc=volc)
    o = {'urdf_inertial_origin': inert, 'collision_mesh': stl, 'n_vertices': int(len(V)), 'area_cm2': round(float(ar.sum()) * 1e4, 1),
         'signed_volume_cm3': round(float(v6.sum() / 6) * 1e6, 1), 'watertight': bool(m.is_watertight),
         'shell_centroid_local_m': np.round(shell, 5).tolist(), 'vol_centroid_local_m': np.round(volc, 5).tolist(),
         'origin_to_shell_cm': round(float(np.linalg.norm(shell)) * 100, 2), 'origin_to_volc_cm': round(float(np.linalg.norm(volc)) * 100, 2),
         'obj_vs_stl': objcmp,
         'hull_cache_best': [os.path.basename(best[1]), best[2], round(best[0] * 1000, 3)],
         'hull_cache_next': [[os.path.basename(c[1]), c[2], round(c[0] * 1000, 3)] for c in cands[1:4]]}
    # upright axis = local vector that is world +z at the f0 reference pose
    R0 = q2m(Q[0, k]); up = R0.T @ np.array([0, 0, 1.0])
    o['tilt_vs_f0_deg'] = {f: round(float(np.degrees(np.arccos(np.clip((q2m(Q[f, k]) @ up)[2], -1, 1)))), 2) for f in (0, 50, 100, 258, 266, 288, 300, 312)}
    # pose-invariant ("settled flat on its own base") footprint margins, in the plane normal to `up`
    e1 = np.cross(up, [1, 0, 0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-6 else np.cross(up, [0, 1, 0]) / np.linalg.norm(np.cross(up, [0, 1, 0]))
    e2 = np.cross(up, e1)
    for lab, pts in (('hulls', HA), ('stl', V)):
        hgt = pts @ up; base = pts[hgt < hgt.min() + 0.003]; B2 = np.c_[base @ e1, base @ e2]; H = hull2d(B2)
        ext = (B2.max(0) - B2.min(0)) * 100
        o[f'flatbase_{lab}'] = {'n_base_pts': int(len(base)), 'extent_cm': np.round(ext, 1).tolist(),
                                'margin_origin_cm': round(margin(H, np.array([0.0, 0.0])) * 100, 2),
                                'margin_shell_cm': round(margin(H, np.array([shell @ e1, shell @ e2])) * 100, 2),
                                'margin_volc_cm': round(margin(H, np.array([volc @ e1, volc @ e2])) * 100, 2),
                                'origin_height_above_base_cm': round(float(0 - hgt.min()) * 100, 2),
                                'shell_height_above_base_cm': round(float(shell @ up - hgt.min()) * 100, 2)}
    # world-frame footprint at given frames (vertices within 3 mm of the lowest point), like com_tip.py
    for f in (0, 312):
        R = q2m(Q[f, k]); W = HA @ R.T + P[f, k]; zmin = W[:, 2].min(); base = W[W[:, 2] < zmin + 0.003][:, :2]; H = hull2d(base)
        o[f'world_f{f}_hulls'] = {'zmin_minus_support_top_cm': round(float(zmin - SUP['top']) * 100, 2), 'extent_cm': np.round((base.max(0) - base.min(0)) * 100, 1).tolist(),
                                  'margin_origin_cm': round(margin(H, P[f, k]) * 100, 2), 'margin_shell_cm': round(margin(H, R @ shell + P[f, k]) * 100, 2),
                                  'margin_volc_cm': round(margin(H, R @ volc + P[f, k]) * 100, 2)}
    res['objects'][short] = o
    print(short, json.dumps(o), flush=True)

# ---------------- MuJoCo no-hands settle ----------------
def settle(frame, pot_com, with_spoon=True, dt=0.002, T=3.0, spoon_com='origin'):
    assets, bodies, info = [], [], []
    for short in (['white_pot', 'wooden_spoon'] if with_spoon else ['white_pot']):
        O = OBJ[short]; k = O['k']; pos = P[frame, k]; q = Q[frame, k]
        cm = pot_com if short == 'white_pot' else spoon_com
        com = np.zeros(3) if cm == 'origin' else (O['shell'] if cm == 'shell' else O['volc'])
        geoms = []
        for i, v in enumerate(O['HV']):
            assets.append(f'<mesh name="{short}{i}" vertex="{" ".join(f"{x:.6f}" for x in v.ravel())}"/>')
            geoms.append(f'<geom type="mesh" mesh="{short}{i}" friction="1 0.005 0.0001" condim="3"/>')
        bodies.append(f'<body name="{short}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}"><freejoint/>'
                      f'<inertial pos="{com[0]} {com[1]} {com[2]}" mass="0.3" diaginertia="0.001 0.001 0.001"/>{"".join(geoms)}</body>')
        info.append((short, k))
    c = SUP['c']
    sup = f'<geom type="cylinder" size="{SUP["r"]} {SUP["h"] / 2}" pos="{c[0]} {c[1]} {c[2]}" friction="1 0.005 0.0001"/>'
    floor = '<geom type="plane" size="5 5 0.1" pos="0 0 0"/>'
    xml = (f'<mujoco><compiler inertiafromgeom="false"/><option timestep="{dt}" gravity="0 0 -9.81"/><asset>{"".join(assets)}</asset>'
           f'<worldbody>{floor}{sup}{"".join(bodies)}</worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    rows = []; marks = {int(round(tt / dt)): tt for tt in (0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 3.0)}
    for step in range(int(T / dt) + 1):
        if step in marks:
            row = {'t': marks[step]}
            for bi, (short, k) in enumerate(info):
                O = OBJ[short]; R0 = q2m(Q[frame, k]); Rm = d.xmat[bi + 1].reshape(3, 3); pos = d.xpos[bi + 1]
                upl = q2m(Q[0, k]).T @ np.array([0, 0, 1.0])
                ang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ Rm) - 1) / 2, -1, 1)))
                tilt_world = np.degrees(np.arccos(np.clip((Rm @ upl)[2], -1, 1)))
                add = np.linalg.norm(O['V'] @ Rm.T + pos - (O['V'] @ R0.T + P[frame, k]), axis=1).mean()
                row[short] = dict(rot_from_start_deg=round(float(ang), 1), tilt_vs_upright_deg=round(float(tilt_world), 1),
                                  dz_cm=round(float(pos[2] - P[frame, k][2]) * 100, 2), ADD_vs_start_cm=round(float(add) * 100, 2))
            rows.append(row)
        mujoco.mj_step(m, d)
    return rows

res['settle'] = {}
for frame in (0, 289, 312):
    for pc in ('origin', 'volc', 'shell'):
        for ws in (False, True):
            for dt in (0.002, 1 / 101.27):
                lab = f'f{frame} | pot COM {pc} | {"pot+spoon" if ws else "pot alone"} | dt {dt:.4f}'
                try:
                    rows = settle(frame, pc, with_spoon=ws, dt=dt)
                except Exception as e:
                    rows = [{'err': str(e)[:300]}]
                res['settle'][lab] = rows
                last = rows[-1]
                print('##', lab, '->', json.dumps(rows[1] if len(rows) > 1 else rows[0]), '| t=0.25', json.dumps(rows[3]) if len(rows) > 3 else '', '| final', json.dumps(last), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
print('DONE', OUT)
