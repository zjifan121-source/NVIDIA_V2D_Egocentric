"""relevel.py SCENE OUT NX NY NZ A B C ANCHOR [--drop-other-supports] [--margin 0.001]
Re-level a built V2D scene so that the table plane MEASURED from our own stereo depth (z = A*x + B*y + C in the scene's sim
world, unit normal (NX,NY,NZ)) becomes horizontal: every world-frame quantity of the processed parquet is rotated by R
(normal -> +z) about the pivot p = the measured table point under ANCHOR's frame-0 footprint. The sim's gravity is then
perpendicular to the real table, so objects resting on the real table are not tipped by the simulator at reset (diagnosed:
the sim re-seats the anchor pot 3-6 deg in ep1/ep5 because GeoCalib gravity is 4.7-5.5 deg off the measured table).
The scorer aligns the whole scene by the anchor's frame-0 pose, so a global rotation by itself does not change the score.
Rotated: object body/root poses, MANO joints (+orientations), scale centres, contact positions (non-zero entries) and
normals, MANO trans/global_orient (root joint moved with the world, per-frame offset kept), robot wrist poses and the 67
robot frames (pos + wxyz). Support: support_0 re-centred (rotated) with its top at the (now horizontal) measured table height,
capped so no object resting at frame 0 is below it (minus --margin); other support prims dropped with --drop-other-supports.
Checks (printed): object-object relative poses and hand-object distances unchanged; measured normal -> +z.
"""
import glob, json, os, re, shutil, subprocess, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, trimesh
from scipy.spatial.transform import Rotation as Rot

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SCENE, OUT = os.path.abspath(args[0]), os.path.abspath(args[1])
n = np.array([float(v) for v in args[2:5]]); n /= np.linalg.norm(n)
A, B, C = [float(v) for v in args[5:8]]; ANCHOR = args[8]
DROP = "--drop-other-supports" in sys.argv
MARGIN = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--margin=")), 0.001))
RELOCATE = os.environ.get("RELOCATE", os.path.expanduser("~/v2d_orchard/relocate_scene.py"))
if os.path.exists(OUT): sys.exit(f"{OUT} exists")
shutil.copytree(SCENE, OUT, symlinks=True)
pqf = sorted(glob.glob(f"{OUT}/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet"))[-1]
seq = re.search(r"sequence_id=([^/]+)", pqf).group(1)
t = pq.read_table(pqf)
get = lambda k: np.array(t.column(k)[0].as_py(), float)
names = list(t.column("object_body_names")[0].as_py())
ai = next(i for i, nm in enumerate(names) if nm.endswith(ANCHOR))

# rotation normal -> +z
z = np.array([0, 0, 1.0]); ax = np.cross(n, z); s = np.linalg.norm(ax); ang = np.arctan2(s, n @ z)
Rm = Rot.from_rotvec(ax / s * ang).as_matrix() if s > 1e-12 else np.eye(3); qR = Rot.from_matrix(Rm)
# pivot: measured table point under the anchor's frame-0 footprint
P = get("object_body_position"); Q = get("object_body_wxyz")
proc = f"{OUT}/assets/human_motion_data/ego_recon/processed/{seq}"
meshes = {nm: np.asarray(trimesh.load(f"{proc}_{nm.replace(seq + '_', '')}.obj", force="mesh", process=False).vertices) for nm in names}
def world(i, f, PP=P, QQ=Q):
    return meshes[names[i]] @ Rot.from_quat(QQ[f, i][[1, 2, 3, 0]]).as_matrix().T + PP[f, i]
W0 = world(ai, 0); zb = W0[:, 2].min(); xy = W0[W0[:, 2] < zb + 0.01][:, :2].mean(0)
p = np.array([xy[0], xy[1], A * xy[0] + B * xy[1] + C])
rot_pts = lambda X: (X - p) @ Rm.T + p

newcols = {}
POS3 = ["object_body_position", "object_root_position", "mano_right_scale_center", "mano_left_scale_center", "mano_right_joints",
        "mano_left_joints", "robot_right_wrist_position", "robot_left_wrist_position"]
CONTACT_POS = ["mano_right_link_contact_positions", "mano_left_link_contact_positions", "mano_right_object_contact_positions",
               "mano_left_object_contact_positions"]
QUAT = ["object_body_wxyz", "mano_right_joints_wxyz", "mano_left_joints_wxyz", "robot_right_wrist_wxyz", "robot_left_wrist_wxyz"]
NORMALS = ["mano_right_link_contact_normals", "mano_left_link_contact_normals", "mano_right_object_contact_normals",
           "mano_left_object_contact_normals"]
AXANG = ["object_root_axis_angle", "mano_right_global_orient", "mano_left_global_orient"]


def rot_wxyz(Qw):
    sh = Qw.shape; q = Qw.reshape(-1, 4)
    out = (qR * Rot.from_quat(q[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    out *= np.sign((out * q).sum(1, keepdims=True) + 1e-12) * 0 + 1          # keep as computed (unit, valid)
    return out.reshape(sh)


for k in POS3:
    if k in t.column_names: newcols[k] = rot_pts(get(k))
for k in CONTACT_POS:
    if k in t.column_names:
        X = get(k); valid = np.abs(X).sum(-1) > 0; Y = X.copy(); Y[valid] = rot_pts(X[valid]); newcols[k] = Y
for k in NORMALS:
    if k in t.column_names: newcols[k] = get(k) @ Rm.T
for k in QUAT:
    if k in t.column_names: newcols[k] = rot_wxyz(get(k))
for k in AXANG:
    if k in t.column_names:
        X = get(k); newcols[k] = (qR * Rot.from_rotvec(X.reshape(-1, 3))).as_rotvec().reshape(X.shape)
for side in ("right", "left"):
    if f"mano_{side}_trans" in t.column_names:
        tr = get(f"mano_{side}_trans"); J0 = get(f"mano_{side}_joints")[:, 0]
        newcols[f"mano_{side}_trans"] = rot_pts(J0) - (J0 - tr)
for side in ("right", "left"):
    k = f"robot_{side}_frames"
    if k in t.column_names:
        Fm = get(k); G = Fm.copy(); G[..., :3] = rot_pts(Fm[..., :3]); G[..., 3:7] = rot_wxyz(Fm[..., 3:7]); newcols[k] = G

cols = {}
for k in t.column_names:
    if k in newcols:
        cols[k] = pa.array([newcols[k].tolist()], type=t.schema.field(k).type)
    else:
        cols[k] = t.column(k)
pq.write_table(pa.table(cols, schema=t.schema), pqf)

# ---- checks
t2 = pq.read_table(pqf); get2 = lambda k: np.array(t2.column(k)[0].as_py(), float)
P2, Q2 = get2("object_body_position"), get2("object_body_wxyz")
chk = {}
if len(names) > 1:
    rel = lambda PP, QQ: [np.linalg.inv(np.block([[Rot.from_quat(QQ[f, 0][[1, 2, 3, 0]]).as_matrix(), PP[f, 0][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
                           @ np.block([[Rot.from_quat(QQ[f, 1][[1, 2, 3, 0]]).as_matrix(), PP[f, 1][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]) for f in range(0, P.shape[0], 10)]
    d = [np.abs(a - b).max() for a, b in zip(rel(P, Q), rel(P2, Q2))]
    chk["object_relative_pose_max_change"] = float(max(d))
for side in ("right", "left"):
    J, J2 = get(f"mano_{side}_joints"), get2(f"mano_{side}_joints")
    da = np.linalg.norm(J[:, 9] - P[:, ai], axis=1); db = np.linalg.norm(J2[:, 9] - P2[:, ai], axis=1)
    chk[f"{side}_hand_to_anchor_dist_max_change_m"] = float(np.abs(da - db).max())
    W, W2 = get(f"robot_{side}_wrist_position"), get2(f"robot_{side}_wrist_position")
    chk[f"robot_{side}_wrist_to_anchor_dist_max_change_m"] = float(np.abs(np.linalg.norm(W - P[:, ai], axis=1) - np.linalg.norm(W2 - P2[:, ai], axis=1)).max())
chk["measured_normal_after"] = (Rm @ n).round(6).tolist()

# ---- support
usd = glob.glob(f"{OUT}/assets/human_motion_data/ego_recon/reconstructed_stage/*_support.usda")[0]
txt = open(usd).read()
bottoms0 = {names[i]: float(world(i, 0, P2, Q2)[:, 2].min()) for i in range(len(names))}
table_z = p[2]
resting = [v for k2, v in bottoms0.items() if v < table_z + 0.03]          # objects within 3 cm of the table at f0
top = min([table_z] + resting) - MARGIN
prims = re.findall(r'def Cylinder "([^"]+)"', txt)
for pr in prims:
    m = re.search(r'(def Cylinder "%s"[^{]*\{.*?double height = ([-\d.e]+).*?xformOp:translate = \(([^)]+)\))' % re.escape(pr), txt, re.S)
    h = float(m.group(2)); x, y, zc = [float(v) for v in m.group(3).split(",")]
    if pr == "support_0" or len(prims) == 1:
        c2 = rot_pts(np.array([[x, y, zc]]))[0]
        txt = txt.replace(f"xformOp:translate = ({m.group(3)})", f"xformOp:translate = ({c2[0]}, {c2[1]}, {top - h / 2})")
        chk["support_0"] = dict(old_center=[x, y, zc], new_center=[float(c2[0]), float(c2[1]), top - h / 2], top=top)
    elif DROP:
        blk = re.search(r'\n[ \t]*def Cylinder "%s".*?\n[ \t]*\}\n' % re.escape(pr), txt, re.S)
        if blk: txt = txt.replace(blk.group(0), "\n"); chk.setdefault("dropped", []).append(pr)
open(usd, "w").write(txt)
r = subprocess.run([sys.executable, RELOCATE, OUT], capture_output=True, text=True)
rep = dict(scene=SCENE, out=OUT, tilt_deg=float(np.degrees(ang)), pivot=p.tolist(), table_z_after=table_z, support_top=top,
           frame0_bottoms=bottoms0, checks=chk, relocate=(r.stdout + r.stderr).strip().splitlines()[:1])
json.dump(rep, open(f"{OUT}/relevel_report.json", "w"), indent=1); print(json.dumps(rep, indent=1))
