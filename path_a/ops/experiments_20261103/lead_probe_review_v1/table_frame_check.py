"""Is the depth in the same frame as our object poses? Projects our meshes into the frame and compares with observed depth.

dz = observed depth minus mesh depth, on object pixels, in cm: near zero means depth and object poses agree, so a table
height read from depth can be compared with our meshes and supports. IoU compares the projected silhouette with the mask.
No GT, no scans. Usage: table_frame_check.py FRAMES_ROOT EP=SCENE_DIR ..."""
import json, sys, os
from pathlib import Path
import numpy as np, cv2

def read_depth(p):
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED); d = np.full(raw.shape, np.nan); v = (raw > 0) & (raw < 65535)
    d[v] = 65535.0 / raw[v].astype(np.float64) - 1.0; return d

def read_mesh(p):
    vs, fs = [], []
    for l in open(p):
        if l.startswith('v '): vs.append(l.split()[1:4])
        elif l.startswith('f '):
            i = [int(t.split('/')[0]) - 1 for t in l.split()[1:]]; fs += [[i[0], i[k], i[k + 1]] for k in range(1, len(i) - 1)]
    return np.array(vs, float), np.array(fs, int)

def sample(v, f, n, rng):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]; w = np.linalg.norm(np.cross(b - a, c - a), axis=1); i = rng.choice(len(f), n, p=w / w.sum())
    r = rng.random((n, 2)); s = np.sqrt(r[:, 0]); return (1 - s)[:, None] * a[i] + (s * (1 - r[:, 1]))[:, None] * b[i] + (s * r[:, 1])[:, None] * c[i]

root = sys.argv[1]
for arg in sys.argv[2:]:
    ep, scene = arg.split('=', 1); ep = int(ep); b = Path(scene) / 'bundle'; d = np.load(b / 'result.npz', allow_pickle=False); K = json.load(open(f'{root}/ep{ep:02d}/K.json'))
    print(f"ep{ep} K.json fx {K['fx']:.3f} fy {K['fy']:.3f} cx {K['cx']:.3f} cy {K['cy']:.3f} | bundle camera_intrinsics {np.round(np.asarray(d['camera_intrinsics']).reshape(-1)[:9], 3).tolist()}", flush=True)
    names = [str(n) for n in d['object_names']]; c2w = d['camera_to_world_transform'].astype(float)
    for t in (0, 10):
        depth = read_depth(f'{root}/ep{ep:02d}/depth/{t:06d}.png'); h, w = depth.shape; Rc, tc = c2w[t, :3, :3], c2w[t, :3, 3]
        for i, n in enumerate(names):
            P = d['object_to_world_transform' if i == 0 else f'object_{i}_to_world_transform'].astype(float)[t]; s = float(d['object_scale' if i == 0 else f'object_{i}_scale']); v, f = read_mesh(b / ('mesh.obj' if i == 0 else f'mesh_{i}.obj')); rng = np.random.default_rng(ep * 100 + i + t)
            pts = sample(v * s, f, 400000, rng) @ P[:3, :3].T + P[:3, 3]; cam = (pts - tc) @ Rc; z = cam[:, 2]; keep = z > 0.05
            u = np.round(cam[keep, 0] / z[keep] * K['fx'] + K['cx']).astype(int); vv = np.round(cam[keep, 1] / z[keep] * K['fy'] + K['cy']).astype(int); zz = z[keep]
            ok = (u >= 0) & (u < w) & (vv >= 0) & (vv < h); u, vv, zz = u[ok], vv[ok], zz[ok]
            zbuf = np.full((h, w), np.inf); np.minimum.at(zbuf, (vv, u), zz); sil = np.isfinite(zbuf)
            mf = f'{root}/ep{ep:02d}/masks/{n}/{t:06d}.png'; m = cv2.imread(mf, cv2.IMREAD_UNCHANGED); m = (m if m.ndim == 2 else m.max(-1)) > 0
            sil2 = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0; iou = (sil2 & m).sum() / max(1, (sil2 | m).sum())
            core = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0; g = core & sil & np.isfinite(depth); dz = (depth - zbuf)[g] * 100
            print(f"  f{t:<3} {n:20} mask px {int(m.sum()):6} iou {iou:.3f} compared px {int(g.sum()):6} dz median {np.median(dz) if len(dz) else float('nan'):+.2f} cm, p10 {np.percentile(dz, 10) if len(dz) else 0:+.2f}, p90 {np.percentile(dz, 90) if len(dz) else 0:+.2f}", flush=True)
