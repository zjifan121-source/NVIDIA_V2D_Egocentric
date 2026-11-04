"""CLOSED 2026-11-04: do not use for submissions. Project rule, work/handover/08_operations_and_access.md: 不用新mesh配旧坐标pose.
A registered re-reconstruction under another rollout's poses is a new mesh on old-coordinate poses. The legal path is re-track
and retrain on the better mesh. Kept as a record of the analysis only.

Replace one object's mesh in a packed Track 3 file with our own reconstruction, placed in the incumbent's body frame.

The scorer uses the submitted mesh twice. CD-O compares its shape to the scan after Sim(3), so frame and size do not count.
Every pose board first registers the mesh to the scan and converts the rollout through that result, so a new mesh changes
AUC, SP-SR, MP-SR, RPE and MPPE even with identical rollout rows. To keep that conversion close to the incumbent's, the new
mesh takes the incumbent's surface centroid, rotation and RMS radius: uniform scale plus rigid motion, no shape change.

Checks: all 180-degree flips about the principal axes are scored, since a flip costs nothing on CD-O but wrecks the pose
boards; other meshes of the class stand in for the unknown scan to test that the scorer's registration treats old and new
alike; only one cell of the file may change. No scan and no GT is read. Nothing is submitted.
Usage: mesh_swap_probe.py --base B.parquet --episode E --object NAME --mesh OURS.obj --out OUT.parquet [--stand-ins E,E,...]"""
import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
KIT = os.environ.get('LEAD_KIT', '/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit'); sys.path.insert(0, KIT)
from v2dlb.track3_pose import _decode_object_mesh as dec, _encode_archive as enc, _register_object_body as reg, _rotation_matrix
from v2dlb.object_mesh import _object_chamfer_cm as cd, _surface_moments as moments, _align_rigid as align, _symmetric_mean_distance as symd, _kabsch as kabsch
from v2dlb.mesh_common import _sample_mesh_surface as sample
from v2dlb.mesh_budget import budget_mesh
ROSTER = {int(k): v for k, v in json.load(open(KIT + '/data/track_3_evaluation_objects.json'))['episodes'].items()}

def read_obj(path):
    vs, fs = [], []
    for line in open(path):
        if line.startswith('v '): vs.append(line.split()[1:4])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            fs += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    return np.array(vs, float), np.array(fs, int)

def refine(ref, cand, R, t, n=40):
    from scipy.spatial import cKDTree
    tree = cKDTree(ref)
    for _ in range(n):
        moved = cand @ R.T + t; d, i = tree.query(moved, k=1); keep = np.argsort(d)[:int(len(d) * 0.9)]
        dR, dt = kabsch(moved[keep], ref[i[keep]]); R = dR @ R; t = dR @ t + dt
    return R, t, symd(ref, cand @ R.T + t)

def angle(R): return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--episode', type=int, required=True); ap.add_argument('--object', required=True); ap.add_argument('--mesh', required=True); ap.add_argument('--out', required=True, type=Path); ap.add_argument('--stand-ins', default=''); a = ap.parse_args()
if a.out.exists(): raise SystemExit(f'{a.out} exists; never overwriting')
base = pd.read_parquet(a.base); slot = ROSTER[a.episode].index(a.object); rid = f'track_3/e{a.episode:06d}/w000/f000000/o{slot}'
idx = base.index[base.row_id == rid][0]; vb, fb = dec(base.object_mesh_b64[idx]); vo, fo = read_obj(a.mesh); seed = 1_000_003 * a.episode + 101 * slot
cb, rb = moments(vb, fb); co, ro = moments(vo, fo); s = rb / ro
pb = sample(vb, fb, 3000, seed) - cb; po = (sample(vo, fo, 3000, seed) - co) * s
R, t = align(pb, po); best = symd(pb, po @ R.T + t)
c = (po - po.mean(0)); axes = np.linalg.eigh(c.T @ c)[1][:, ::-1]  # long axis first, in our mesh's frame
hyp = {'found': round(best * 100, 3)}
for name, u in (('flip_about_long_axis', axes[:, 0]), ('flip_about_middle_axis', axes[:, 1]), ('flip_about_short_axis', axes[:, 2])):
    F = 2 * np.outer(u, u) - np.eye(3); hyp[name] = round(refine(pb, po, R @ F, t)[2] * 100, 3)
margin = min(v for k, v in hyp.items() if k != 'found') / hyp['found']
new_v = ((vo - co) * s) @ R.T + t + cb
# which end is wider along the long axis: guards head against handle independently of the Chamfer score
def profile(pts, axis, bins=12):
    x = pts @ axis; edges = np.quantile(x, np.linspace(0, 1, bins + 1)); perp = pts - np.outer(x, axis)
    return np.array([np.linalg.norm(perp[(x >= edges[i]) & (x <= edges[i + 1])], axis=1).mean() for i in range(bins)])
cbb = pb - pb.mean(0); U = np.linalg.eigh(cbb.T @ cbb)[1][:, -1]
head_corr = float(np.corrcoef(profile(pb, U), profile(po @ R.T + t, U))[0, 1])
out_dir = a.out.parent; out_dir.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / 'registered.obj'
    with open(p, 'w') as f:
        for v in new_v: f.write('v %.8f %.8f %.8f\n' % tuple(v))
        for tri in fo + 1: f.write('f %d %d %d\n' % tuple(tri))
    V, Fa = budget_mesh(str(p), 4096, 4096)
payload = enc(vertices=V, faces=Fa); nv, nf = dec(payload)
cn, rn = moments(nv, nf); d_new = cd((vb, fb), (nv, nf), seed); d_raw = cd((vb, fb), (vo, fo), seed)
stand = {}
for e in [int(x) for x in a.stand_ins.split(',') if x]:
    sm = dec(base.object_mesh_b64[base.row_id == f'track_3/e{e:06d}/w000/f000000/o{ROSTER[e].index(a.object)}'].iloc[0]); anchor = moments(*sm)[0]
    ao, Ro = reg((vb, fb), sm, anchor, seed); an, Rn = reg((nv, nf), sm, anchor, seed)
    stand[e] = dict(rotation_change_deg=round(angle(Rn @ Ro.T), 1), anchor_shift_cm=round(float(np.linalg.norm(an - ao)) * 100, 2))
new = base.copy(); new.at[idx, 'object_mesh_b64'] = payload
tmp = a.out.with_suffix('.tmp.parquet'); new.to_parquet(tmp, index=False); back = pd.read_parquet(tmp)
cols = [c for c in base.columns]; diff = {c: int((~((base[c].values == back[c].values) | (base[c].isna().values & back[c].isna().values))).sum()) for c in cols}
if sum(diff.values()) != 1 or diff['object_mesh_b64'] != 1 or back.object_mesh_b64[idx] != payload: tmp.unlink(); raise SystemExit(f'expected exactly one changed cell, got {diff}')
tmp.rename(a.out)
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
rep = dict(base=str(a.base), base_sha256=sha(a.base), episode=a.episode, object=a.object, slot=slot, source_mesh=a.mesh, source_mesh_sha256=sha(a.mesh), out=str(a.out), out_sha256=sha(a.out),
           scale_applied=round(float(s), 4), rotation_from_our_frame_deg=round(angle(R), 1), residual_cm_found=hyp['found'], flip_hypotheses_cm=hyp, next_best_over_found=round(float(margin), 2), head_profile_corr=round(head_corr, 3),
           centroid_shift_cm=round(float(np.linalg.norm(cn - cb)) * 100, 3), rms_radius_ratio=round(float(rn / rb), 4), shape_distance_base_new_cm=round(float(d_new), 3), shape_distance_base_raw_ours_cm=round(float(d_raw), 3),
           board_bound_if_all_gain=round(float(d_new) / 36, 4), budget=dict(verts=int((np.abs(nv - nv[0]).sum(1) > 0).sum() + 1), faces=int((np.ptp(nf, axis=1) > 0).sum())), stand_ins=stand, changed_cells=diff['object_mesh_b64'])
Path(str(a.out) + '.swap.json').write_text(json.dumps(rep, indent=1) + '\n'); print(json.dumps(rep, indent=1))
