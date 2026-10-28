"""R9 referee, part 2 (CPU): does a FoundationPose sweeper track flip to a 180-deg twin, judged by CoTracker3's 2D
point tracks (cotrack_ep31.py)? GT-free.
Each query point (pixel at its query frame q, inside the SAM 3 mask) is lifted with the stereo depth and kept if it
lies within 1.5 cm of the posed mesh; P = inv(T_q) X_q in the mesh frame. At another frame t it should project to its
2D track at K T_t F_h P, where F_h (h = 0..3: I and 180-deg turns about the mesh's PCA axes, r9_unwrap.klein) is the
twin of frame t relative to frame q. Per (q, t): h = argmin of the median reprojection error, accepted when it beats
the runner-up by --margin. The four twins form a Klein group (labels XOR), so per-frame labels L(t) relative to frame
0 follow by vote propagation from L(0) = 0. L(t) != 0 = the track at t is a twin of frame 0's branch.
--anchor majority re-bases the labels on the most common branch instead of frame 0's (frame 0 may be the outlier);
--anchor ref re-bases them on the run's registration frame (run.json reference_frame), where FoundationPose registered.
Writes <out_dir>/<frame>.json = T_t @ F_L(t) (unlabeled frames keep the nearest labeled frame's L) and a report.
Usage (stereo venv): python cotrack_eval.py <run_dir> <tracks.npz> <out_dir> [--poses poses] [--margin 0.6]
                     [--anchor 0|majority|ref]"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ep31_stage_images import depth_m
from r9_unwrap import klein, load_T, obj_mesh, save_T
from r7_select import surface_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('npz', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--poses', default='poses')
    ap.add_argument('--margin', type=float, default=0.6)
    ap.add_argument('--min_pts', type=int, default=6)
    ap.add_argument('--anchor', default='0', choices=['0', 'majority', 'ref'])
    a = ap.parse_args()
    run = a.run
    mesh = run / 'mesh_scaled.obj' if (run / 'mesh_scaled.obj').is_file() else run / 'scan_mesh' / 'mini_sweeper.obj'
    v, f = obj_mesh(mesh)
    F, info = klein(v, f)
    tree = cKDTree(surface_points(v, f, 100000))
    pdir = run / a.poses if not Path(a.poses).is_absolute() else Path(a.poses)
    files = sorted(pdir.glob('*.json'))
    n = len(files)
    TD = [load_T(p) for p in files]
    T = np.stack([x[0] for x in TD])
    K = json.loads((run / 'intrinsics_stable.json').read_text())
    fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']
    Z = np.load(a.npz)
    tracks, vis, Q = Z['tracks'], Z['vis'] > 0.5, Z['queries']
    qf = Q[:, 0].astype(int)

    # lift query points into the mesh frame
    P = np.full((len(Q), 3), np.nan)
    for q in np.unique(qf):
        dm = depth_m(run / 'depth' / f'{q:06d}.png')
        if dm is None or q >= n:
            continue
        idx = np.flatnonzero(qf == q)
        x, y = Q[idx, 1], Q[idx, 2]
        z = dm[np.clip(y.round().astype(int), 0, dm.shape[0] - 1), np.clip(x.round().astype(int), 0, dm.shape[1] - 1)]
        ok = np.isfinite(z) & (z > 0.1) & (z < 2.5)
        Xc = np.stack([(x - cx) * z / fx, (y - cy) * z / fy, z], 1)
        Ti = np.linalg.inv(T[q])
        Pm = Xc @ Ti[:3, :3].T + Ti[:3, 3]
        d = np.full(len(Pm), np.inf)
        d[ok] = tree.query(Pm[ok])[0]
        ok &= d < 0.015
        P[idx[ok]] = Pm[ok]
    good = np.isfinite(P[:, 0])
    qs = [q for q in np.unique(qf) if (good & (qf == q)).sum() >= a.min_pts]

    # relative twin per (q, t)
    B = {}
    for q in qs:
        idx = np.flatnonzero(good & (qf == q))
        for t in range(n):
            if t == q:
                continue
            sel = idx[vis[t, idx]]
            if len(sel) < a.min_pts:
                continue
            err = []
            for h in range(4):
                M = T[t] @ F[h]
                Xt = P[sel] @ M[:3, :3].T + M[:3, 3]
                uv = np.stack([fx * Xt[:, 0] / Xt[:, 2] + cx, fy * Xt[:, 1] / Xt[:, 2] + cy], 1)
                err.append(np.median(np.linalg.norm(uv - tracks[t, sel], axis=1)))
            err = np.array(err)
            o = np.argsort(err)
            if err[o[0]] < a.margin * err[o[1]]:
                B[(q, t)] = (int(o[0]), float(err[o[0]]), float(err[o[1]]))

    # labels relative to frame 0 by vote propagation (Klein group: compose = XOR)
    L = np.full(n, -1)
    L[0] = 0
    for _ in range(4):
        for t in range(n):
            votes = np.zeros(4)
            for q in [0] + [x for x in qs if x != 0]:     # frame 0 votes through B[(t, 0)] even if not a query frame
                if (q, t) in B and L[q] >= 0:
                    votes[B[(q, t)][0] ^ L[q]] += 1
                if (t, q) in B and L[q] >= 0 and t in qs:      # t is itself a query frame tracked to q
                    votes[B[(t, q)][0] ^ L[q]] += 1
            if t != 0 and votes.sum() >= 1 and votes.max() > 0.6 * votes.sum():
                L[t] = int(votes.argmax())
    lab = np.flatnonzero(L >= 0)
    Lf = L.copy()
    for t in np.flatnonzero(L < 0):
        Lf[t] = L[lab[np.argmin(np.abs(lab - t))]]
    if len(lab) == 1 and n > 1:
        raise SystemExit(f'no label propagated from frame 0 (query frames {qs}): not writing a "corrected" track')
    if a.anchor == 'majority':
        Lf = Lf ^ int(np.bincount(Lf, minlength=4).argmax())
    elif a.anchor == 'ref':
        Lf = Lf ^ int(Lf[int(json.loads((run / 'run.json').read_text()).get('reference_frame', 0))])
    a.out.mkdir(parents=True, exist_ok=False)
    for p, (Tt, d), h in zip(files, TD, Lf):
        save_T(a.out / p.name, Tt @ F[h], d)
    segs, s = [], 0
    for t in range(1, n + 1):
        if t == n or Lf[t] != Lf[s]:
            segs.append([int(s), int(t - 1), int(Lf[s])])
            s = t
    rep = dict(run=str(run), poses=str(pdir), anchor=a.anchor, mesh=mesh.name, pca=info, query_frames=[int(q) for q in qs],
               points=int(good.sum()), pairs=len(B), labeled=int(len(lab)), twin_frames=int((Lf != 0).sum()),
               segments=segs)
    (a.out.parent / f'{a.out.name}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ('poses', 'anchor', 'query_frames', 'points', 'pairs', 'labeled', 'twin_frames', 'segments')}))


if __name__ == '__main__':
    main()
