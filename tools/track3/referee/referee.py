"""The referee for one object run (CPU, GT-free): which frames of a FoundationPose track sit on a 180-degree twin of the
frame-0 branch, judged by CoTracker3 2D point tracks, and the track with those frames turned back.

Each query point (a pixel inside the object mask at its query frame q) is lifted with the stereo depth and kept if it
lies within 1.5 cm of the posed mesh: P = inv(T_q) X_q in the mesh frame. At another frame t it should project onto its
2D track at K T_t F_h P, where F_h (h = 0..3: the identity and the 180-degree turns about the mesh's PCA axes) is the
twin of frame t relative to frame q. Per (q, t): h = argmin of the median reprojection error, accepted when it beats
the runner-up by the margin. The four twins form a Klein group (labels compose by XOR), so per-frame labels L(t)
relative to frame 0 follow by vote propagation from L(0) = 0. The output track is T_t @ F_L(t); frames without a label
take the nearest labelled frame's L.

anchor '0' keeps frame 0's branch (the frame the metric aligns the scene on). 'ref' re-bases on the run's registration
frame (run.json reference_frame), for runs registered later than frame 0. 'majority' re-bases on the most common branch;
it was harmful in our tests (see README) and is kept only for diagnostics."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geom import depth_m, klein, load_T, obj_mesh, save_T, surface_points


def find_mesh(run):
    """The mesh the run's poses refer to: mesh_scaled.obj; else the provided mesh named in run.json (--object_mesh, an
    error if that file is gone); else the last .obj path in mesh_source.txt; else scan_mesh/<object>.obj."""
    run = Path(run)
    if (run / 'mesh_scaled.obj').is_file():
        return run / 'mesh_scaled.obj'
    rj = run / 'run.json'
    if rj.is_file():
        cmd = json.loads(rj.read_text()).get('command', [])
        cmd = cmd.split() if isinstance(cmd, str) else [str(c) for c in cmd]
        for i, c in enumerate(cmd):
            for flag in ('--object_mesh', '--object-mesh'):
                path = c[len(flag) + 1:] if c.startswith(flag + '=') else (cmd[i + 1] if c == flag and i + 1 < len(cmd) else None)
                if path:
                    if not Path(path).is_file():
                        raise SystemExit(f'{rj} tracked the mesh {path}, which does not exist: pass --mesh')
                    return Path(path)
    ms = run / 'mesh_source.txt'
    if ms.is_file():
        objs = [t for t in ms.read_text().replace(',', ' ').split() if t.endswith('.obj') and Path(t).is_file()]
        if objs:
            return Path(objs[-1])
    scan = run / 'scan_mesh' / f'{run.name}.obj'
    if scan.is_file():
        return scan
    raise SystemExit(f'no mesh found for {run}: pass --mesh')


def run(run_dir, tracks, output, poses='poses', mesh=None, margin=0.6, min_pts=6, anchor='0'):
    if anchor not in ('0', 'ref', 'majority'):
        raise SystemExit(f"anchor must be '0', 'ref' or 'majority', not {anchor!r}")
    run_dir = Path(run_dir)
    out = Path(output)
    pdir = Path(poses) if Path(poses).is_absolute() else run_dir / poses
    files = sorted(pdir.glob('*.json'))
    n = len(files)
    if n == 0:
        raise SystemExit(f'no poses in {pdir}')
    if [p.stem for p in files] != [f'{i:06d}' for i in range(n)]:
        raise SystemExit(f'{pdir}: pose files are not 000000.json .. {n - 1:06d}.json')
    Z = np.load(tracks)
    for key in ('tracks', 'vis', 'queries'):
        if key not in Z:
            raise SystemExit(f'{tracks} has no "{key}" array (not a `tracks` output)')
    trk, vis, Q = Z['tracks'], Z['vis'] > 0.5, Z['queries']
    if trk.shape[0] != n:
        raise SystemExit(f'{tracks} has {trk.shape[0]} frames but {pdir} has {n} poses (different video or frames?)')
    kp = run_dir / 'intrinsics_stable.json'
    if not kp.is_file():
        raise SystemExit(f'no {kp}')
    K = json.loads(kp.read_text())
    fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']
    mesh = Path(mesh) if mesh else find_mesh(run_dir)
    v, f = obj_mesh(mesh)
    F, info = klein(v, f)
    tree = cKDTree(surface_points(v, f, 100000))
    TD = [load_T(p) for p in files]
    T = np.stack([x[0] for x in TD])
    qf = Q[:, 0].astype(int)

    # lift query points into the mesh frame
    P = np.full((len(Q), 3), np.nan)
    for q in np.unique(qf):
        dm = depth_m(run_dir / 'depth' / f'{q:06d}.png')
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
    qs = [q for q in np.unique(qf) if (good & (qf == q)).sum() >= min_pts]

    # relative twin per (q, t)
    B = {}
    for q in qs:
        idx = np.flatnonzero(good & (qf == q))
        for t in range(n):
            if t == q:
                continue
            sel = idx[vis[t, idx]]
            if len(sel) < min_pts:
                continue
            err = []
            for h in range(4):
                M = T[t] @ F[h]
                Xt = P[sel] @ M[:3, :3].T + M[:3, 3]
                uv = np.stack([fx * Xt[:, 0] / Xt[:, 2] + cx, fy * Xt[:, 1] / Xt[:, 2] + cy], 1)
                err.append(np.median(np.linalg.norm(uv - trk[t, sel], axis=1)))
            err = np.array(err)
            o = np.argsort(err)
            if err[o[0]] < margin * err[o[1]]:
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
                if (t, q) in B and L[q] >= 0 and t in qs:  # t is itself a query frame tracked to q
                    votes[B[(t, q)][0] ^ L[q]] += 1
            if t != 0 and votes.sum() >= 1 and votes.max() > 0.6 * votes.sum():
                L[t] = int(votes.argmax())
    lab = np.flatnonzero(L >= 0)
    if len(lab) == 1 and n > 1:
        raise SystemExit(f'no label propagated from frame 0 (query frames {qs}): not writing a corrected track')
    Lf = L.copy()
    for t in np.flatnonzero(L < 0):
        Lf[t] = L[lab[np.argmin(np.abs(lab - t))]]
    if anchor == 'majority':
        Lf = Lf ^ int(np.bincount(Lf, minlength=4).argmax())
    elif anchor == 'ref':
        ref = int(json.loads((run_dir / 'run.json').read_text()).get('reference_frame', 0) or 0)
        if not 0 <= ref < n:
            raise SystemExit(f'reference_frame {ref} is outside the {n} poses')
        Lf = Lf ^ int(Lf[ref])
    out.mkdir(parents=True, exist_ok=False)
    for p, (Tt, d), h in zip(files, TD, Lf):
        save_T(out / p.name, Tt @ F[h], d)
    segs, s = [], 0
    for t in range(1, n + 1):
        if t == n or Lf[t] != Lf[s]:
            segs.append([int(s), int(t - 1), int(Lf[s])])
            s = t
    rep = dict(run=str(run_dir), poses=str(pdir), tracks=str(tracks), anchor=anchor, mesh=str(mesh), pca=info,
               query_frames=[int(q) for q in qs], points=int(good.sum()), pairs=len(B), labeled=int(len(lab)),
               twin_frames=int((Lf != 0).sum()), segments=segs)
    (out.parent / f'{out.name}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ('anchor', 'query_frames', 'points', 'pairs', 'labeled', 'twin_frames', 'segments')}))
    return rep
