"""build_f3w.py (refine1104 ep3 round 2, F3w): direct parquet edit on a COPY of the adopted scene sc_ep3.
usage: build_f3w.py SRC_SCENE OUT_SCENE R1DIR OUTDIR
  SRC_SCENE  local copy of sc_ep3 (read only)      OUT_SCENE  new scene dir (must not exist)
  R1DIR      round-1 dir (fromBabel/f3_select.npz)  OUTDIR     where f3w_track.npz / build_report.json go
Definitions and gates: OUTDIR/gates.json (written before this script ran)."""
import glob, json, os, re, shutil, sys, hashlib
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.ndimage import gaussian_filter1d

SRC, OUT, R1, OD = [os.path.abspath(a) for a in sys.argv[1:5]]
if os.path.exists(OUT): sys.exit(f"{OUT} exists")
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
OFF = np.array([0.074563, -0.320921, 1.514694])
rep = {}


def load_obj(p):  # identical to round-1 d1.py / d2.py
    V, F = [], []
    for line in open(p):
        if line.startswith('v '): V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) for t in line.split()[1:]]; idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            for k in range(1, len(idx) - 1): F.append([idx[0], idx[k], idx[k + 1]])
    return np.array(V, float), np.array(F, int)


def rmean(Rs):  # chordal L2 mean (skB3.py)
    U, _, Vt = np.linalg.svd(Rs.sum(0)); M = U @ Vt
    if np.linalg.det(M) < 0: U[:, -1] *= -1; M = U @ Vt
    return M


def angd(Ra, Rb): return np.degrees(Rot.from_matrix(np.einsum('nji,njk->nik', Ra, Rb)).magnitude())


def twist_split(T, AX):  # round-1 d2.py split of per-frame rotation steps
    Rr = np.einsum('nji,njk->nik', T[:-1, :3, :3], T[1:, :3, :3]); q = Rot.from_matrix(Rr).as_quat()
    tw = np.degrees(2 * np.arctan2(q[:, :3] @ AX, q[:, 3])); tw = (tw + 180) % 360 - 180
    return np.r_[0, np.degrees(Rot.from_matrix(Rr).magnitude())], np.r_[0, tw]


def blend(TA, TB, wt):  # round-1 blend
    out = TA.copy()
    for f in np.nonzero(wt > 0)[0]:
        out[f][:3, :3] = Slerp([0, 1], Rot.from_matrix([TA[f][:3, :3], TB[f][:3, :3]]))([wt[f]]).as_matrix()[0]
        out[f][:3, 3] = (1 - wt[f]) * TA[f][:3, 3] + wt[f] * TB[f][:3, 3]
    return out


# ---------------- inputs
S = np.load(R1 + '/fromBabel/f3_select.npz')
w = S['w']; OC = S['OC_cam'].astype(float); W = S['W_cam'].astype(float); N = len(w)
VS, FS = load_obj(SRC + '/bundle_cleaned/mesh_1.obj')
c = VS.mean(0); _, U = np.linalg.eigh(np.cov((VS - c).T)); AX = U[:, -1]
rep['mesh_md5'] = md5(SRC + '/bundle_cleaned/mesh_1.obj'); rep['AX_body'] = AX.round(5).tolist(); rep['centroid_body_cm'] = (100 * c).round(3).tolist()
Lb = S['L'].astype(float)
Lrv = Rot.from_matrix(Lb[:3, :3]).as_rotvec()
rep['levelling_L'] = dict(total_deg=float(np.degrees(np.linalg.norm(Lrv))), twist_about_AX_deg=float(np.degrees(Lrv @ AX)),
                          axis_tilt_deg=float(np.degrees(np.arccos(np.clip((Lb[:3, :3] @ AX) @ AX, -1, 1)))))
rpt = json.load(open(SRC + '/report.json'))
hold = None
def walk(o):
    global hold
    if isinstance(o, dict):
        for k, v in o.items():
            if str(k).endswith('attached_1') and isinstance(v, list):
                for h in v:
                    if h[0] == 'right': hold = (int(h[1]), int(h[2]))
            walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(rpt); a, b = hold; rep['hold_right_spoon'] = [a, b]
# observed right wrist: build's smooth_hand(sigma=2) on bundle_merged, replicated as in skB3.py -> camera frame
mg = np.load(SRC + '/bundle_merged/result.npz', allow_pickle=True)
rep['bundle_merged_md5'] = md5(SRC + '/bundle_merged/result.npz')
c2wm = mg['camera_to_world_transform'].astype(float); Rc = c2wm[:, :3, :3]
orient = mg['hand_right_wrist_orient_in_camera'].astype(float)
valid = mg['hand_right_is_valid'].astype(bool) if 'hand_right_is_valid' in mg.files else np.ones(N, bool)
rep['hand_right_valid_frames_in_hold'] = [int(valid[a:b].sum()), b - a]
quat = Rot.from_matrix(Rc @ Rot.from_rotvec(orient).as_matrix()).as_quat()
for i in range(1, len(quat)):
    if np.dot(quat[i], quat[i - 1]) < 0: quat[i] = -quat[i]
Rsm = Rot.from_quat(gaussian_filter1d(quat, 2.0, axis=0, mode='nearest')).as_matrix()
Robs = np.einsum('nji,njk->nik', Rc, Rsm)                      # camera frame
# ---------------- F3w track
G = np.einsum('nji,njk->nik', Robs[a:b], W[a:b, :3, :3])      # spoon-in-wrist, Wentao track
Gm = rmean(G)
Rpred = Robs @ Gm
rep['grip_spread_vs_mean_deg'] = np.percentile(angd(G, np.repeat(Gm[None], len(G), 0)), [50, 90, 100]).round(1).tolist()
aW = W[:, :3, :3] @ AX; pW = np.einsum('nij,j->ni', W[:, :3, :3], c) + W[:, :3, 3]
ap = Rpred @ AX
dots = np.clip(np.einsum('ni,ni->n', ap, aW), -1, 1)
F3W = np.repeat(np.eye(4)[None], N, 0)
for f in range(N):
    u, v = ap[f], aW[f]; cr = np.cross(u, v); s = np.linalg.norm(cr)
    Ra = Rot.from_rotvec(cr / s * np.arctan2(s, u @ v)).as_matrix() if s > 1e-12 else np.eye(3)
    F3W[f, :3, :3] = Ra @ Rpred[f]; F3W[f, :3, 3] = pW[f] - F3W[f, :3, :3] @ c
ed = np.nonzero(w > 0)[0]; sw = np.nonzero(w >= 1)[0]
rep['wrist_axis_vs_wentao_axis_deg_edited'] = np.percentile(np.degrees(np.arccos(dots[ed])), [50, 90, 100]).round(1).tolist()
rep['f3w_vs_W_rot_deg_edited'] = np.percentile(angd(F3W[ed, :3, :3], W[ed, :3, :3]), [50, 90, 100]).round(1).tolist()
rep['f3w_vs_W_centroid_cm_max'] = float(100 * np.abs(np.einsum('nij,j->ni', F3W[:, :3, :3], c) + F3W[:, :3, 3] - pW).max())
NEW = blend(OC, F3W, w)
# smoothness / twist continuity (camera frame = round-1 metric; world frame reported too)
sm = {}
for k, T in (('OC', OC), ('W', W), ('NEW', NEW)):
    tot, tw = twist_split(T, AX)
    sm[k] = dict(frames_gt10_total_f50_255=int((tot[50:256] > 10).sum()), frames_gt10_twist_f50_255=int((np.abs(tw[50:256]) > 10).sum()),
                 max_step_deg=round(float(tot[50:256].max()), 1), sd_twist_step_f56_258=round(float(np.std(tw[56:259])), 2))
rep['smooth_cam'] = sm
# ---------------- world poses
cl = np.load(SRC + '/bundle_cleaned/result.npz', allow_pickle=True)
c2w = cl['camera_to_world_transform'].astype(float).copy(); c2w[:, :3, 3] += OFF
Sold = c2w @ OC; Snew = c2w @ NEW
for k, T in (('NEW_world', Snew), ('OC_world', Sold)):
    tot, tw = twist_split(T, AX)
    rep['smooth_' + k] = dict(frames_gt10_total_f50_255=int((tot[50:256] > 10).sum()), frames_gt10_twist_f50_255=int((np.abs(tw[50:256]) > 10).sum()))
shutil.copytree(SRC, OUT, symlinks=True)
pqf = sorted(glob.glob(f"{OUT}/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet"))[-1]
src_pqf = pqf.replace(OUT, SRC)
t = pq.read_table(src_pqf)
get = lambda k: np.array(t.column(k)[0].as_py(), float)
names = list(t.column('object_body_names')[0].as_py()); ks = next(i for i, n in enumerate(names) if n.endswith('wooden_spoon'))
P = get('object_body_position'); Q = get('object_body_wxyz')
Tpq = np.repeat(np.eye(4)[None], N, 0); Tpq[:, :3, :3] = Rot.from_quat(Q[:, ks][:, [1, 2, 3, 0]]).as_matrix(); Tpq[:, :3, 3] = P[:, ks]
rep['scene_spoon_vs_c2w@OC'] = dict(pos_m=float(np.abs(Tpq[:, :3, 3] - Sold[:, :3, 3]).max()), rot_deg=float(angd(Tpq[:, :3, :3], Sold[:, :3, :3]).max()))
assert rep['scene_spoon_vs_c2w@OC']['pos_m'] < 1e-5 and rep['scene_spoon_vs_c2w@OC']['rot_deg'] < 1e-3, rep['scene_spoon_vs_c2w@OC']
Sold = Tpq.copy()                                               # exact scene poses as the old pose
Snew[w <= 0] = Sold[w <= 0]
Pn, Qn = P.copy(), Q.copy()
for f in ed:
    Pn[f, ks] = Snew[f, :3, 3]
    q = Rot.from_matrix(Snew[f, :3, :3]).as_quat()[[3, 0, 1, 2]]
    Qn[f, ks] = q if q @ Q[f, ks] >= 0 else -q
newcols = {'object_body_position': Pn, 'object_body_wxyz': Qn}
# ---------------- right hand co-motion on edited frames inside the hold
hf = [int(f) for f in ed if a <= f < b]
rep['hand_comove_frames'] = [min(hf), max(hf), len(hf)] if hf else []
M = {f: Snew[f] @ np.linalg.inv(Sold[f]) for f in hf}
mv = lambda X, f: X @ M[f][:3, :3].T + M[f][:3, 3]
Rm = lambda f: Rot.from_matrix(M[f][:3, :3])
Jr = get('mano_right_joints'); Wr = get('robot_right_wrist_position')
scc = get('mano_right_scale_center')
move_scc = bool(np.median(np.linalg.norm(scc - Wr, axis=1)) < 0.3)
rep['scale_center_rule'] = dict(median_dist_to_wrist_m=float(np.median(np.linalg.norm(scc - Wr, axis=1))), moved=move_scc)
FR = get('robot_right_frames'); fnames = t.column('right_robot_frame_names')[0].as_py()
univ = [i for i in range(FR.shape[1]) if np.allclose(FR[:, i, :3], 0) and np.allclose(np.abs(FR[:, i, 3]), 1)]
rep['robot_frames_identity_entries_kept'] = [[i, fnames[i]] for i in univ]
POS = ['mano_right_joints', 'robot_right_wrist_position'] + (['mano_right_scale_center'] if move_scc else [])
CPOS = ['mano_right_link_contact_positions', 'mano_right_object_contact_positions']
NRM = ['mano_right_link_contact_normals', 'mano_right_object_contact_normals']
QUAT = ['mano_right_joints_wxyz', 'robot_right_wrist_wxyz']
for k in POS + CPOS + NRM + QUAT + ['mano_right_global_orient', 'mano_right_trans', 'robot_right_frames']:
    X = get(k); Y = X.copy()
    for f in hf:
        if k in POS: Y[f] = mv(X[f].reshape(-1, 3), f).reshape(X[f].shape)
        elif k in CPOS:
            x = X[f].reshape(-1, 3); v = np.abs(x).sum(-1) > 0; y = x.copy(); y[v] = mv(x[v], f); Y[f] = y.reshape(X[f].shape)
        elif k in NRM: Y[f] = (X[f].reshape(-1, 3) @ M[f][:3, :3].T).reshape(X[f].shape)
        elif k in QUAT:
            q = X[f].reshape(-1, 4); qn = (Rm(f) * Rot.from_quat(q[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
            qn *= np.where((qn * q).sum(1, keepdims=True) < 0, -1, 1); Y[f] = qn.reshape(X[f].shape)
        elif k == 'mano_right_global_orient': Y[f] = (Rm(f) * Rot.from_rotvec(X[f])).as_rotvec()
        elif k == 'mano_right_trans': J0 = Jr[f, 0]; Y[f] = mv(J0[None], f)[0] - (J0 - X[f])
        elif k == 'robot_right_frames':
            Fm = X[f].copy(); idx = [i for i in range(Fm.shape[0]) if i not in univ]
            Fm[idx, :3] = mv(Fm[idx, :3], f)
            q = Fm[idx, 3:7]; qn = (Rm(f) * Rot.from_quat(q[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
            qn *= np.where((qn * q).sum(1, keepdims=True) < 0, -1, 1); Fm[idx, 3:7] = qn; Y[f] = Fm
    newcols[k] = Y
cols = {k: (pa.array([newcols[k].tolist()], type=t.schema.field(k).type) if k in newcols else t.column(k)) for k in t.column_names}
pq.write_table(pa.table(cols, schema=t.schema), pqf)
rep['parquet_src_md5'] = md5(src_pqf); rep['parquet_out_md5'] = md5(pqf); rep['parquet_out'] = pqf
# ---------------- local gates G4, G5, G6
t2 = pq.read_table(pqf); get2 = lambda k: np.array(t2.column(k)[0].as_py(), float)
gl = {}
nb = sm['NEW']['frames_gt10_total_f50_255']; nt = sm['NEW']['frames_gt10_twist_f50_255']
gl['G4'] = dict(pass_=bool(nb <= 30 and rep['smooth_NEW_world']['frames_gt10_total_f50_255'] <= 30), cam_total=nb, cam_twist=nt,
                world_total=rep['smooth_NEW_world']['frames_gt10_total_f50_255'], world_twist=rep['smooth_NEW_world']['frames_gt10_twist_f50_255'],
                scene_cam_total=sm['OC']['frames_gt10_total_f50_255'], wentao_cam_total=sm['W']['frames_gt10_total_f50_255'], max_step_deg=sm['NEW']['max_step_deg'])
files_same = all(md5(p) == md5(p.replace(OUT, SRC)) for p in glob.glob(f"{OUT}/**/*", recursive=True) if os.path.isfile(p) and os.path.abspath(p) != os.path.abspath(pqf))
changed = [k for k in t.column_names if not t.column(k).equals(t2.column(k))]
allowed = {'object_body_position', 'object_body_wxyz'} | {k for k in newcols if 'right' in k}
P2, Q2 = get2('object_body_position'), get2('object_body_wxyz')
pot_same = all(np.array_equal(P2[:, i], P[:, i]) and np.array_equal(Q2[:, i], Q[:, i]) for i in range(len(names)) if i != ks)
fr_changed = sorted({int(f) for k in changed for f in range(N) if not np.array_equal(get2(k)[f], get(k)[f])}) if changed else []
def T_of(p, q):
    T = np.repeat(np.eye(4)[None], len(p), 0); T[:, :3, :3] = Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix(); T[:, :3, 3] = p; return T
S1, S2 = T_of(P[:, ks], Q[:, ks]), T_of(P2[:, ks], Q2[:, ks])
W1, W2 = T_of(get('robot_right_wrist_position'), get('robot_right_wrist_wxyz')), T_of(get2('robot_right_wrist_position'), get2('robot_right_wrist_wxyz'))
rel1 = np.linalg.inv(S1) @ W1; rel2 = np.linalg.inv(S2) @ W2
J1l = np.einsum('nji,nkj->nki', S1[:, :3, :3], Jr - S1[:, None, :3, 3]); J2 = get2('mano_right_joints')
J2l = np.einsum('nji,nkj->nki', S2[:, :3, :3], J2 - S2[:, None, :3, 3])
hfa = np.array(hf)
dpos = float(np.abs(rel1[hfa, :3, 3] - rel2[hfa, :3, 3]).max()); drot = float(np.radians(angd(rel1[hfa, :3, :3], rel2[hfa, :3, :3]).max()))
djnt = float(np.abs(J1l[hfa] - J2l[hfa]).max())
FR2 = get2('robot_right_frames')
frl1 = np.einsum('nji,nkj->nki', S1[:, :3, :3], FR[..., :3] - S1[:, None, :3, 3]); frl2 = np.einsum('nji,nkj->nki', S2[:, :3, :3], FR2[..., :3] - S2[:, None, :3, 3])
nonu = [i for i in range(FR.shape[1]) if i not in univ]
dfr = float(np.abs(frl1[hfa][:, nonu] - frl2[hfa][:, nonu]).max())
unit = max(float(np.abs(1 - np.linalg.norm(get2(k).reshape(-1, 4), axis=1)).max()) for k in ('object_body_wxyz', 'mano_right_joints_wxyz', 'robot_right_wrist_wxyz'))
unit = max(unit, float(np.abs(1 - np.linalg.norm(FR2[..., 3:7].reshape(-1, 4), axis=1)).max()))
nan = any(not np.isfinite(get2(k)).all() for k in newcols)
gl['G5'] = dict(pass_=bool(files_same and set(changed) <= allowed and pot_same and (not fr_changed or (min(fr_changed) >= int(ed.min()) and max(fr_changed) <= int(ed.max())))
                          and set(fr_changed) <= set(ed.tolist()) and dpos < 1e-6 and djnt < 1e-6 and drot < 1e-6 and dfr < 1e-6 and unit < 1e-6 and not nan
                          and t2.schema.equals(t.schema) and t2.num_rows == t.num_rows),
                non_parquet_files_identical=files_same, changed_columns=changed, changed_frames=[min(fr_changed), max(fr_changed), len(fr_changed)] if fr_changed else [],
                other_bodies_identical=pot_same, wrist_in_spoon_max_pos_m=dpos, wrist_in_spoon_max_rot_rad=drot, mano_joints_in_spoon_max_m=djnt,
                robot_frames_in_spoon_max_m=dfr, quat_unit_max_err=unit, nan=nan, schema_same=bool(t2.schema.equals(t.schema)))
gl['G6'] = dict(pass_=bool(np.array_equal(get2('mano_right_object_contact_part_ids'), get('mano_right_object_contact_part_ids'))))
rep['local_gates'] = gl
np.savez(OD + '/f3w_track.npz', w=w, NEW_cam=NEW, F3W_cam=F3W, OC_cam=OC, W_cam=W, POT_cam=S['POT_cam'], AX=AX, c=c, Robs_cam=Robs, Gm=Gm)
json.dump(rep, open(OD + '/build_report.json', 'w'), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))
