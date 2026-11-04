"""dtchk.py -- independent minimal check of the depth table vs support_0 vs resting object bottoms (ep3 bob1/A1).
Own code; uses render_review only for loaders/projection. Annulus around each object's SAM 3 mask -> FS depth
(inverse-depth u16: d = 65535/v - 1) -> sim world (A @ bundle c2w) -> RANSAC plane -> plane z at the object's
bottom xy. Also: mesh-vs-depth check inside the mask (median depth difference)."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.expanduser('~/v2d_ops/kenc/review/tool'))
import render_review as rr
from PIL import Image
from scipy.ndimage import binary_dilation

S = '/data/user_data/haisenk/v2d/hand5/scenes/h5_ep3_A1'
KR = '/data/user_data/haisenk/v2d/v2d_recon/baseline/reconstruction/data/outputs/ken_c_scored/episode_000003'
SUP = 0.9878071156409292
pqp = rr.find_parquet(S)
P = rr.load_scene_parquet(pqp)
B = rr.load_bundle(S + '/bundle_cleaned/result.npz')
N = P['N']; names = B['names'] or P.get('names') or ['white_pot', 'wooden_spoon']
proc = os.path.dirname(os.path.dirname(os.path.dirname(pqp)))
objs = []
for k in range(len(names)):
    mp = P['mesh_paths'][k]
    c = next(c for c in [mp, os.path.join(proc, os.path.basename(mp))] if os.path.isfile(c))
    V, F = rr.load_obj(c)
    R = rr.wxyz_to_R(P['wxyz'][:, k]).reshape(N, 3, 3)
    T = rr.to_T(R, P['pos'][:, k])
    lo, hi = V.min(0), V.max(0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    objs.append(dict(name=names[k], V=V, F=F, T=T, corners=corners))
A, res = rr.estimate_A([o['T'] for o in objs], [B['objT'][k] for k in range(len(objs))], [o['corners'] for o in objs])
c2w = A[None] @ B['c2w']; w2c = np.linalg.inv(c2w); K = B['K']
print('A t', np.round(A[:3, 3], 5), 'K', np.round(K, 2) if np.ndim(K) else K, flush=True)
H, W = 1520, 2028
fx, fy, cx, cy = 1149.804931640625, 1149.730712890625, 994.8043212890625, 767.544677734375
uu, vv = np.meshgrid(np.arange(W), np.arange(H))
OBJDIR = {'white_pot': 'white_pot', 'wooden_spoon': 'wooden_spoon'}
out = {}
rng = np.random.default_rng(0)
for f in (0, 10, 20, 30, 295, 300, 305, 312):
    v = np.asarray(Image.open(f'{KR}/white_pot/depth/{f:06d}.png')).astype(np.float64)
    d = 65535.0 / np.maximum(v, 1.0) - 1.0
    d[(v >= 65535) | (d < 0.15) | (d > 2.4)] = np.nan
    masks = {}
    for o in objs:
        key = [k for k in OBJDIR if o['name'].endswith(k)][0]
        masks[key] = np.asarray(Image.open(f'{KR}/{key}/masks/1/{f:06d}.png').convert('L')) > 127
    allm = binary_dilation(masks['white_pot'] | masks['wooden_spoon'], iterations=12)
    rec = {}
    for o in objs:
        key = [k for k in OBJDIR if o['name'].endswith(k)][0]
        m = masks[key]
        ring = binary_dilation(m, iterations=140) & ~allm & np.isfinite(d)
        ys, xs = np.where(ring)
        if len(xs) > 60000:
            sel = rng.choice(len(xs), 60000, replace=False); ys, xs = ys[sel], xs[sel]
        z = d[ys, xs]
        Pc = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z, np.ones_like(z)], 1)
        Pw = (c2w[f] @ Pc.T).T[:, :3]
        # object bottom
        Wv = rr.xf(o['T'][f], o['V'])
        bz = Wv[:, 2].min(); low = Wv[:, 2] < bz + 0.003; bxy = Wv[low, :2].mean(0)
        Pw = Pw[np.abs(Pw[:, 2] - bz) < 0.10]
        best = None
        for it in range(300):
            i3 = rng.choice(len(Pw), 3, replace=False)
            nrm = np.cross(Pw[i3[1]] - Pw[i3[0]], Pw[i3[2]] - Pw[i3[0]])
            if np.linalg.norm(nrm) < 1e-9: continue
            nrm /= np.linalg.norm(nrm)
            if abs(nrm[2]) < 0.9: continue
            dist = np.abs((Pw - Pw[i3[0]]) @ nrm)
            inl = dist < 0.004
            if best is None or inl.sum() > best[0].sum(): best = (inl, nrm, Pw[i3[0]])
        inl = best[0]; Q = Pw[inl]
        Am = np.c_[Q[:, 0], Q[:, 1], np.ones(len(Q))]
        coef, *_ = np.linalg.lstsq(Am, Q[:, 2], rcond=None)
        rms = float(np.std(Q[:, 2] - Am @ coef))
        zt = float(coef[0] * bxy[0] + coef[1] * bxy[1] + coef[2])
        tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
        # mesh-vs-depth along the ray inside the eroded mask: project vertices, compare vertex depth to depth map
        uvw = rr.project(Wv, w2c[f], K[f])
        xy, ok, dep = uvw if isinstance(uvw, tuple) and len(uvw) == 3 else (None, None, None)
        dd = None
        if xy is not None:
            xi = np.round(xy[:, 0]).astype(int); yi = np.round(xy[:, 1]).astype(int)
            inb = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            inb[inb] &= m[yi[inb], xi[inb]]
            diff = d[yi[inb], xi[inb]] - dep[inb]
            diff = diff[np.isfinite(diff)]
            # visible-surface vertices are those whose depth ~ min: keep diffs > -2 cm (behind-surface verts have big negative)
            vis = diff[diff > -0.02]
            dd = round(float(np.median(vis)) * 100, 2) if len(vis) else None
        rec[key] = dict(n_ring=int(len(Pw)), inlier_frac=round(float(inl.mean()), 3), plane_rms_mm=round(rms * 1000, 2),
                        plane_tilt_deg=round(tilt, 2), table_z_at_bottom=round(zt, 4), bottom_z=round(float(bz), 4),
                        bottom_minus_table_cm=round((bz - zt) * 100, 2), table_minus_support_cm=round((zt - SUP) * 100, 2),
                        mesh_vs_depth_median_cm=dd)
    out[f] = rec
    print(f, json.dumps(rec), flush=True)
json.dump(out, open(os.path.expanduser('~/v2d_ops/kenc/diag/ep3_skB2/dtchk.json'), 'w'), indent=1)
print('DONE')
