"""d3.py (refine1104 ep3 round 2, F3w gates on Babel; GT-free, read-only on all inputs, writes only to OUT).
G1 IoU of the NEW scene track vs SAM 3 spoon masks (round-1 d1.py renderer, SC=2), all frames (OC/W from round-1 d1.json)
G2 depth residual vs FoundationStereo inside the SAM 3 mask, every 2nd edited frame (round-1 d1.py code), NEW/OC/W
G3 spoon-vs-pot interpenetration f36-230 (round-1 d1.py parity test), NEW
G7 kit-registration roll consistency: register the spoon mesh to the masked stereo cloud with AUC.py's _align_rigid
usage: python d3.py OUT  (OUT holds f3w_track.npz, AUC.py)"""
import json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from multiprocessing import Pool
T0 = time.time(); OUT = sys.argv[1]
sys.path.insert(0, OUT)
import AUC
def log(*a): print('[%6.1fs]' % (time.time() - T0), *a, flush=True)
O = '/data/user_data/haisenk/v2d/v2d_recon/baseline/reconstruction/data/outputs/ken_c_scored/episode_000003'
CLEAN = '/data/user_data/haisenk/v2d/hand5/scenes/h5_ep3_A1/bundle_cleaned'
D1 = '/data/user_data/haisenk/v2d/refine1104/ep3/diag/d1.json'
def load_obj(p):
    V, F = [], []
    for line in open(p):
        if line.startswith('v '): V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) for t in line.split()[1:]]; idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            for k in range(1, len(idx) - 1): F.append([idx[0], idx[k], idx[k + 1]])
    return np.array(V, float), np.array(F, int)
VS, FS_ = load_obj(CLEAN + '/mesh_1.obj'); VP, FP_ = load_obj(CLEAN + '/mesh.obj')
cl = np.load(CLEAN + '/result.npz', allow_pickle=True)
fx, fy, cx, cy = cl['camera_intrinsics'].astype(float).ravel()[:4]
Z = np.load(OUT + '/f3w_track.npz')
w = Z['w']; NEW, OC, W, POT = Z['NEW_cam'], Z['OC_cam'], Z['W_cam'], Z['POT_cam']; AX = Z['AX']
def TC(d, key): return np.linalg.inv(d['camera_to_world_transform'].astype(float)) @ d[key].astype(float)
log('OC in npz vs cleaned bundle spoon (cam) max pos diff m', float(np.abs(TC(cl, 'object_1_to_world_transform')[:, :3, 3] - OC[:, :3, 3]).max()))
N = len(w); SWF = np.nonzero(w >= 1)[0]; ANYF = np.nonzero(w > 0)[0]
SC = 2
def silhouette(Tc, V, F, sc=SC):
    X = V @ Tc[:3, :3].T + Tc[:3, 3]; z = X[:, 2]
    if (z <= 0.02).any(): return None
    u = (fx * X[:, 0] / z + cx) / sc; v = (fy * X[:, 1] / z + cy) / sc
    im = Image.new('L', (2028 // sc, 1520 // sc), 0); dr = ImageDraw.Draw(im); P = np.stack([u, v], 1)
    for f in F: dr.polygon([tuple(P[f[0]]), tuple(P[f[1]]), tuple(P[f[2]])], fill=255)
    return np.array(im) > 0
def mask(f, sc=SC):
    m = np.array(Image.open(f'{O}/wooden_spoon/masks/1/{f:06d}.png').convert('L')); return (m[::sc, ::sc] if sc > 1 else m) > 127
def iou_job(f):
    m = mask(f); s = silhouette(NEW[f], VS, FS_)
    return float('nan') if s is None or m.sum() == 0 else float((s & m).sum() / max((s | m).sum(), 1))
with Pool(8) as pool: INEW = np.array(pool.map(iou_job, range(N)))
d1 = json.load(open(D1)); IOU = {k: np.array(d1['iou'][k], float) for k in ('OC', 'W', 'WL')}; IOU['NEW'] = INEW
WIN = {'P0 f0-35': (0, 35), 'P1 f36-55': (36, 55), 'f56-84': (56, 84), 'f85-129': (85, 129), 'f130-185': (130, 185), 'f186-215': (186, 215),
       'f216-258': (216, 258), 'P3 f259-288': (259, 288), 'P4 f289-312': (289, 312), 'all': (0, N - 1)}
for k in ('OC', 'W', 'NEW'):
    log('IoU %-3s' % k, ' | '.join('%s %.3f' % (n, np.nanmean(IOU[k][a:b + 1])) for n, (a, b) in WIN.items()), '| switched %.3f' % np.nanmean(IOU[k][SWF]),
        '| edited %.3f' % np.nanmean(IOU[k][ANYF]))
# ---------------- depth (round-1 code)
H, Wd = 1520, 2028
def zbuffer(Vc, F):
    zb = np.full(H * Wd, np.inf); ok = Vc[:, 2] > 0.05; Fk = F[ok[F].all(1)]
    uv = np.stack([fx * Vc[:, 0] / np.where(ok, Vc[:, 2], 1) + cx, fy * Vc[:, 1] / np.where(ok, Vc[:, 2], 1) + cy], 1)
    tri = uv[Fk]; Lg = np.max(np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=2), axis=1); s = np.clip(np.ceil(Lg / 0.5).astype(int), 1, 400)
    for sv in np.unique(s):
        sel = np.where(s == sv)[0]; ii, jj = np.meshgrid(np.arange(sv + 1), np.arange(sv + 1), indexing='ij'); m = ii + jj <= sv
        b1, b2 = ii[m] / sv, jj[m] / sv; b0 = 1 - b1 - b2; per = len(b0); step = max(1, 3_000_000 // per)
        for c0 in range(0, len(sel), step):
            ff = Fk[sel[c0:c0 + step]]
            Vs = (b0[None, :, None] * Vc[ff[:, 0]][:, None] + b1[None, :, None] * Vc[ff[:, 1]][:, None] + b2[None, :, None] * Vc[ff[:, 2]][:, None]).reshape(-1, 3)
            u = np.round(fx * Vs[:, 0] / Vs[:, 2] + cx).astype(np.int64); v = np.round(fy * Vs[:, 1] / Vs[:, 2] + cy).astype(np.int64)
            inb = (u >= 0) & (u < Wd) & (v >= 0) & (v < H); np.minimum.at(zb, v[inb] * Wd + u[inb], Vs[inb, 2])
    return zb.reshape(H, Wd)
def read_depth(f):
    r = np.asarray(Image.open(f'{O}/white_pot/depth/{f:06d}.png')).astype(np.float64); ok = (r > 0) & (r < 65535)
    return np.where(ok, 65535.0 / np.maximum(r, 1.0) - 1.0, np.nan)
def depth_job(f):
    d = read_depth(f); m = mask(f, 1); out = {'f': int(f)}
    for k, T in (('OC', OC), ('W', W), ('NEW', NEW)):
        Vc = VS @ T[f][:3, :3].T + T[f][:3, 3]; zb = zbuffer(Vc, FS_)
        px = m & np.isfinite(zb) & np.isfinite(d); r = (zb - d)[px]; rr = r[np.abs(r) < 0.10]
        out[k] = float(np.median(rr) * 1000) if len(rr) > 20 else None
    return out
DF = [int(f) for f in ANYF[::2]]
with Pool(8) as pool: DEP = pool.map(depth_job, DF)
SWset = set(int(f) for f in SWF); DS = {}
for lab, sel in (('switched', SWset), ('all_edited', set(DF))):
    DS[lab] = {}
    for k in ('OC', 'W', 'NEW'):
        v = np.array([x[k] for x in DEP if x['f'] in sel and x[k] is not None], float)
        DS[lab][k] = dict(n=len(v), median_abs_med_mm=round(float(np.median(np.abs(v))), 1), med_of_med_mm=round(float(np.median(v)), 1))
    log('DEPTH', lab, json.dumps(DS[lab]))
# ---------------- penetration (round-1 parity test)
G = 0.002; lo = VP.min(0) - 0.004; hi = VP.max(0) + 0.004; nx, ny = int(np.ceil((hi[0] - lo[0]) / G)), int(np.ceil((hi[1] - lo[1]) / G))
cells, zs = [], []
for f in FP_:
    t = VP[f]; xmn, ymn = t[:, :2].min(0); xmx, ymx = t[:, :2].max(0)
    i0, i1 = int(np.floor((xmn - lo[0]) / G - 0.5)), int(np.ceil((xmx - lo[0]) / G - 0.5)); j0, j1 = int(np.floor((ymn - lo[1]) / G - 0.5)), int(np.ceil((ymx - lo[1]) / G - 0.5))
    I, J = np.meshgrid(np.arange(max(i0, 0), min(i1, nx - 1) + 1), np.arange(max(j0, 0), min(j1, ny - 1) + 1), indexing='ij')
    px = lo[0] + (I.ravel() + 0.5) * G; py = lo[1] + (J.ravel() + 0.5) * G
    v0, v1, v2 = t[0, :2], t[1, :2], t[2, :2]; den = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(den) < 1e-14: continue
    a = ((v1[1] - v2[1]) * (px - v2[0]) + (v2[0] - v1[0]) * (py - v2[1])) / den; b = ((v2[1] - v0[1]) * (px - v2[0]) + (v0[0] - v2[0]) * (py - v2[1])) / den; c = 1 - a - b
    ins = (a >= 0) & (b >= 0) & (c >= 0)
    if ins.any(): cells.append((I.ravel() * ny + J.ravel())[ins]); zs.append((a * t[0, 2] + b * t[1, 2] + c * t[2, 2])[ins])
cells = np.concatenate(cells); zs = np.concatenate(zs); order = np.lexsort((zs, cells)); cells, zs = cells[order], zs[order]
cnt = np.bincount(cells, minlength=nx * ny); start = np.r_[0, np.cumsum(cnt)[:-1]]; mc = int(cnt.max()); Zg = np.full((nx * ny, mc), np.nan)
for k in range(mc):
    sel = cnt > k; Zg[sel, k] = zs[start[sel] + k]
def inside_pot(Pb):
    i = np.floor((Pb[:, 0] - lo[0]) / G).astype(int); j = np.floor((Pb[:, 1] - lo[1]) / G).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (Pb[:, 2] > lo[2]) & (Pb[:, 2] < hi[2])
    res = np.zeros(len(Pb), bool); cid = i[ok] * ny + j[ok]; above = (Zg[cid] > Pb[ok, 2][:, None]).sum(1); res[ok] = (above % 2 == 1) & (cnt[cid] % 2 == 0)
    return res
PEN = {}
for k, T in (('OC', OC), ('NEW', NEW)):
    nin = np.zeros(N, int)
    for f in range(36, 231):
        Pb = np.linalg.inv(POT[f]) @ T[f]; nin[f] = inside_pot(VS @ Pb[:3, :3].T + Pb[:3, 3]).sum()
    PEN[k] = int((nin[36:231] > 0).sum())
log('PEN frames with spoon verts inside the pot f36-230:', PEN)
# ---------------- G7 kit-registration roll consistency
CAND = AUC._sample_mesh_surface(VS, FS_, AUC._ALIGNMENT_SAMPLES, 7)
def reg_job(f):
    d = read_depth(f); m = mask(f, 1); px = m & np.isfinite(d)
    if px.sum() < 400: return dict(f=int(f), n=int(px.sum()))
    v, u = np.nonzero(px); z = d[px]; zm = np.median(z); k = np.abs(z - zm) < 0.05; u, v, z = u[k], v[k], z[k]
    X = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1)
    rng = np.random.default_rng(int(f))
    if len(X) > AUC._ALIGNMENT_SAMPLES: X = X[rng.choice(len(X), AUC._ALIGNMENT_SAMPLES, replace=False)]
    R, t = AUC._align_rigid(X, CAND)
    out = dict(f=int(f), n=int(px.sum()), resid_mm=1000 * AUC._symmetric_mean_distance(X, CAND @ R.T + t))
    for kk, T in (('NEW', NEW), ('OC', OC), ('W', W)):
        rel = T[f][:3, :3].T @ R; q = Rot.from_matrix(rel).as_quat()
        tw = float(np.degrees(2 * np.arctan2(q[:3] @ AX, q[3]))); tw = (tw + 180) % 360 - 180
        out[kk] = dict(ang=float(np.degrees(Rot.from_matrix(rel).magnitude())), twist=tw,
                       cen_cm=float(100 * np.linalg.norm((R @ CAND.mean(0) + t) - (T[f][:3, :3] @ CAND.mean(0) + T[f][:3, 3]))))
    return out
RF = [int(f) for f in SWF[::2]]
with Pool(8) as pool: REG = pool.map(reg_job, RF)
REGok = [r for r in REG if 'NEW' in r]
KR = {}
for kk in ('NEW', 'OC', 'W'):
    a = np.array([r[kk]['ang'] for r in REGok]); t_ = np.abs(np.array([r[kk]['twist'] for r in REGok])); cc = np.array([r[kk]['cen_cm'] for r in REGok])
    KR[kk] = dict(n=len(a), median_ang=round(float(np.median(a)), 1), p90_ang=round(float(np.percentile(a, 90)), 1), median_abs_twist=round(float(np.median(t_)), 1),
                  frac_ang_ge30=round(float(np.mean(a >= 30)), 3), median_centroid_cm=round(float(np.median(cc)), 2))
log('KITREG frames', len(RF), 'usable', len(REGok), 'resid mm median', round(float(np.median([r['resid_mm'] for r in REGok])), 2), json.dumps(KR))
m = lambda a, x, y: float(np.nanmean(a[x:y + 1]))
gates = {
    'G1a': dict(value=round(m(INEW, 130, 185), 3), thr='>=0.65', pass_=bool(m(INEW, 130, 185) >= 0.65)),
    'G1b': dict(value=round(m(INEW, 85, 129), 3), thr='>=0.65', pass_=bool(m(INEW, 85, 129) >= 0.65)),
    'G1c': dict(value=round(float(np.nanmean(INEW)), 3), thr='>=0.70', pass_=bool(np.nanmean(INEW) >= 0.70)),
    'G1d': dict(value=[round(m(INEW, 0, 35), 3), round(m(INEW, 36, 55), 3)], scene=[round(m(IOU['OC'], 0, 35), 3), round(m(IOU['OC'], 36, 55), 3)],
                pass_=bool(m(INEW, 0, 35) >= m(IOU['OC'], 0, 35) - 0.02 and m(INEW, 36, 55) >= m(IOU['OC'], 36, 55) - 0.02)),
    'G2': dict(value=DS['switched']['NEW']['median_abs_med_mm'], scene=DS['switched']['OC']['median_abs_med_mm'],
               pass_=bool(DS['switched']['NEW']['median_abs_med_mm'] <= DS['switched']['OC']['median_abs_med_mm'] + 5)),
    'G3': dict(value=PEN['NEW'], scene=PEN['OC'], pass_=bool(PEN['NEW'] <= PEN['OC'])),
    'G7': dict(value=dict(median_ang=KR['NEW']['median_ang'], median_abs_twist=KR['NEW']['median_abs_twist']), context=KR,
               pass_=bool(len(REGok) >= 10 and KR['NEW']['median_ang'] < 30 and KR['NEW']['median_abs_twist'] < 30)),
}
for k, v in gates.items(): log('GATE', k, 'PASS' if v['pass_'] else 'FAIL', json.dumps({a: b for a, b in v.items() if a != 'pass_'}))
json.dump(dict(gates=gates, iou_new=np.round(INEW, 4).tolist(), depth=DEP, depth_summary=DS, pen=PEN, kitreg=REG, kitreg_summary=KR), open(OUT + '/d3.json', 'w'))
log('ALL_PASS' if all(v['pass_'] for v in gates.values()) else 'SOME_GATE_FAILED')
