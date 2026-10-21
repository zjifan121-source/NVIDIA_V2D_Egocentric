"""Fit each object mesh's proportions to many frames of mask and depth (no GPU, no rerun).

SAM3D builds the mesh from one frame, so it has to guess how deep the object is and often guesses
too thick. Here the mesh gets a separate scale along each of its three principal axes, plus one
rigid correction in the object frame, chosen to fit the SAM2 masks and FoundationStereo depth over
~40 frames spread across the clip (other views show the sides the reference frame could not).
Silhouette pixels hidden behind something nearer (a hand) are not counted against the mesh, and
the worst 30% of frames are dropped so a broken stretch of track cannot drive the fit.
The same rule runs on every object. Writes outputs/<dst>/<episode>/<object>/mesh_fit.obj (metres,
same object frame as the run's mesh) + fit_shape.json.
Usage: python tools/track3/fit_shape.py <src_ns> <dst_ns> <ep>:<object> [...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

OUT = Path(__file__).resolve().parents[2] / 'baseline/reconstruction/data/outputs'
S = 4              # work at 1/4 resolution
NF = 40            # frames used in the fit
TAU = 0.03         # depth residual cap (m)
OCC = 0.02         # observed depth this much nearer than the mesh = occluded
KEEP = 0.7         # fraction of best frames kept (trimmed mean)
LOGCLIP = 0.8      # per-axis scale limited to [0.45, 2.2]
KERNEL = np.ones((3, 3), np.uint8)


def decode(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float64)
    return np.where(a >= 65535, 0.0, 65535.0 / np.maximum(a, 1.0) - 1.0)


def render(pc, K, shape):
    fx, fy, cx, cy = K
    pc = pc[pc[:, 2] > 0.05]
    h, w = shape
    u = ((pc[:, 0] / pc[:, 2] * fx + cx) / S).astype(int)
    v = ((pc[:, 1] / pc[:, 2] * fy + cy) / S).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    zb = np.full(shape, np.inf)
    np.minimum.at(zb, (v[ok], u[ok]), pc[ok, 2])
    sil = cv2.morphologyEx(np.isfinite(zb).astype(np.uint8), cv2.MORPH_CLOSE, KERNEL) > 0
    return sil, zb


def frame_cost(m, d, sil, zb):
    occ = sil & ~m & (d > 0) & np.isfinite(zb) & (d < zb - OCC)  # hidden behind something nearer
    union = (m | sil).sum() - occ.sum()
    iou = (m & sil).sum() / union if union > 0 else 0.0
    sel = m & np.isfinite(zb) & (d > 0)
    dep = float(np.mean(np.minimum(np.abs(d[sel] - zb[sel]), TAU)) / TAU) if sel.sum() >= 40 else 1.0
    return (1.0 - iou) + dep


class Fit:
    def __init__(self, run):
        b = run / 'result_slam_gravity_aligned'
        d = np.load(b / 'result.npz')
        self.oc = d['object_to_camera_transform'].astype(np.float64)
        self.K = d['camera_intrinsics'].astype(np.float64)
        self.scale = float(d['object_scale']) if 'object_scale' in d.files else 1.0
        self.mesh = trimesh.load(b / 'mesh.obj', force='mesh', process=False)
        self.mesh.apply_scale(self.scale)
        self.pts = self.mesh.sample(20000, seed=0)
        self.c = self.pts.mean(0)
        _, _, vt = np.linalg.svd(self.pts - self.c, full_matrices=False)
        self.P = vt.T
        oid = json.loads((run / 'object_track.json').read_text())['object_id']
        n = len(self.oc)
        lost = np.linalg.norm(self.oc[:, :3, 3], axis=1) < 0.10
        area = np.zeros(n)
        masks = {}
        for i in range(n):
            mp = run / 'masks' / str(oid) / f'{i:06d}.png'
            if lost[i] or not mp.is_file():
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0
            area[i] = m.sum()
            masks[i] = m
        ok = np.flatnonzero(area >= 0.3 * area.max()) if area.max() > 0 else np.array([], int)
        self.frames = ok[np.linspace(0, len(ok) - 1, min(NF, len(ok))).astype(int)] if len(ok) else ok
        self.obs = []
        for i in self.frames:
            m = masks[i]
            h, w = m.shape[0] // S, m.shape[1] // S
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            dm = cv2.resize(decode(run / 'depth' / f'{i:06d}.png'), (w, h), interpolation=cv2.INTER_NEAREST)
            self.obs.append((m, dm))

    def transform(self, x):
        """Mesh-frame points after per-axis scaling about the centre and the rigid correction."""
        s = np.exp(np.clip(x[:3], -LOGCLIP, LOGCLIP))
        A = self.P @ np.diag(s) @ self.P.T
        Rc = R.from_rotvec(x[3:6]).as_matrix()
        return A, Rc, x[6:9]

    def costs(self, x):
        A, Rc, t = self.transform(x)
        p = ((self.pts - self.c) @ A.T + self.c) @ Rc.T + t
        out = []
        for i, (m, dm) in zip(self.frames, self.obs):
            T = self.oc[i]
            out.append(frame_cost(m, dm, *render(p @ T[:3, :3].T + T[:3, 3], self.K, m.shape)))
        return np.array(out)

    def objective(self, x):
        c = np.sort(self.costs(x))
        return float(c[:max(1, int(KEEP * len(c)))].mean())


def fit(src, dst, ep, obj):
    run = OUT / src / f'episode_{ep:06d}' / obj
    F = Fit(run)
    out = OUT / dst / f'episode_{ep:06d}' / obj
    out.mkdir(parents=True, exist_ok=True)
    if len(F.frames) < 5:
        rep = {'ep': ep, 'obj': obj, 'skipped': f'only {len(F.frames)} usable frames'}
        (out / 'fit_shape.json').write_text(json.dumps(rep, indent=1))
        return rep
    x0 = np.zeros(9)
    c0 = F.objective(x0)
    best = None
    for start in ([0, 0, 0], [-0.3, 0, 0], [0, -0.3, 0], [0, 0, -0.3]):  # a few starts: shrink each axis
        xs = np.r_[start, np.zeros(6)]
        simplex = [xs] + [xs + np.eye(9)[k] * (0.2 if k < 3 else 0.05 if k < 6 else 0.02) for k in range(9)]
        r = minimize(F.objective, xs, method='Nelder-Mead',
                     options={'initial_simplex': np.array(simplex), 'maxfev': 700, 'xatol': 1e-3, 'fatol': 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    A, Rc, t = F.transform(best.x)
    m = F.mesh.copy()
    v = ((m.vertices - F.c) @ A.T + F.c) @ Rc.T + t
    m.vertices = v
    # undo the rigid correction for the exported shape: keep only the proportions, in the original frame
    m.vertices = (v - t) @ Rc
    m.export(out / 'mesh_fit.obj')
    s = np.exp(np.clip(best.x[:3], -LOGCLIP, LOGCLIP))
    rep = {'ep': ep, 'obj': obj, 'frames': [int(i) for i in F.frames], 'cost_before': c0, 'cost_after': float(best.fun),
           'axis_scale': s.round(3).tolist(), 'correction_deg': float(np.rad2deg(np.linalg.norm(best.x[3:6]))),
           'correction_cm': float(np.linalg.norm(best.x[6:9]) * 100),
           'extent_before_cm': sorted((F.mesh.extents * 100).round(1).tolist()),
           'extent_after_cm': sorted((m.extents * 100).round(1).tolist())}
    (out / 'fit_shape.json').write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    for a in sys.argv[3:]:
        ep, obj = a.split(':')
        print(json.dumps(fit(src, dst, int(ep), obj)), flush=True)
