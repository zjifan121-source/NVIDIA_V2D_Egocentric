"""Shared smooth metric shape and auxiliary poses from observed contact/RGB-D."""
from pathlib import Path
import time, json, shutil
import numpy as np, cv2, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import minimum_filter, maximum_filter, distance_transform_edt
from common import sha, put, fields, mesh_name, bundle_files
from renderer import raster_surface
from geometry import skew, projection

def sample(m, n):
    y, x = np.where(m)
    a = np.linspace(0, len(x) - 1, min(n, len(x)), dtype=int)
    return (y[a], x[a])

def boundary(m):
    return m & ~(cv2.erode(m.astype('uint8'), np.ones((3, 3), np.uint8)) > 0)

def run(spec, H):
    start = time.monotonic()
    cfg = spec['parameters']
    inputs = spec['inputs']
    out = H / 'output'
    out.mkdir(exist_ok=False)
    src = Path(inputs['bundle'])
    mesh = trimesh.load(src / mesh_name(spec['object_slot']), process=False)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    cen = (V.min(0) + V.max(0)) / 2
    bundle = np.load(src / 'result.npz')
    T0 = bundle[fields(spec['object_slot'])[0]].astype(float)
    P = T0.copy()
    N = cfg['graph_nodes']
    dmin = np.full(len(V), np.inf)
    ix = int(np.linalg.norm(V - cen, axis=1).argmin())
    nodes = []
    for _ in range(N):
        nodes.append(ix)
        dmin = np.minimum(dmin, np.linalg.norm(V - V[ix], axis=1))
        ix = int(dmin.argmax())
    controls = V[nodes]
    dist = np.linalg.norm(controls[:, None] - controls[None, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    sigma = float(np.median(dist.min(1)))

    def weights(q):
        w = np.exp(-np.sum((q[:, None] - controls[None, :]) ** 2, axis=-1) / (2 * sigma * sigma))
        return w / w.sum(1, keepdims=True)
    B = weights(V) - weights(cen[None])
    edgeset = set()
    for i in range(N):
        for j in np.argsort(dist[i])[:3]:
            edgeset.add(tuple(sorted([i, int(j)])))
    edges = np.array(sorted(edgeset))
    me = mesh.edges_unique
    el = np.linalg.norm(V[me[:, 0]] - V[me[:, 1]], axis=1)
    n0 = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ng = np.linalg.norm(n0, axis=1) > 1e-15
    I = Path(inputs['observations'])
    k = json.loads((I / 'K_fp.json').read_text())
    K = np.array([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1.0]])
    obs = {}
    support = cfg['support']
    held = cfg['heldout']
    for t in support + held:
        mask = cv2.imread(str(I / f'masks/{t:06d}.png'), 0) > 0
        raw = cv2.imread(str(I / f'depth/{t:06d}.png'), -1)
        valid = (raw > 0) & (raw < 65535)
        dep = np.zeros(raw.shape)
        dep[valid] = 65535.0 / raw[valid] - 1.0
        v = valid & mask & (cv2.imread(str(Path(inputs['validity']) / f'{t:06d}.png'), 0) > 0)
        yy, xx = np.indices(mask.shape)
        check = (xx // 8 * 17 + yy // 8 * 31) % 5 == 0
        y, x = sample(v & ~check, 1200)
        xyz = np.c_[x, y, np.ones(len(x))] @ np.linalg.inv(K).T * dep[y, x, None]
        bd = boundary(mask)
        obs[t] = dict(mask=mask, depth=dep, valid=valid, support=v, check=v & check, xyz=xyz, boundary=sample(bd, 256), nearest=distance_transform_edt(~bd, return_distances=False, return_indices=True))
    ann = json.loads(Path(inputs['contact_annotation']).read_text())
    pix = np.array(ann['mask_boundary_pixel'])
    plane = np.load(Path(inputs['table_plane']))
    C = bundle['camera_to_world_transform'][ann['frame']]
    nc = C[:3, :3].T @ plane[:3]
    dc = plane[3] + plane[:3] @ C[:3, 3]
    ray = np.r_[pix, 1.0] @ np.linalg.inv(K).T
    target = ray * (-dc / (nc @ ray))
    theta = np.zeros((N, 3))
    history = []

    def geom(v):
        n = np.cross(v[F[:, 1]] - v[F[:, 0]], v[F[:, 2]] - v[F[:, 0]])
        ratio = np.linalg.norm(v[me[:, 0]] - v[me[:, 1]], axis=1) / np.maximum(el, 1e-20)
        good = el > 1e-10
        return bool(((n * n0).sum(1)[ng] > 0).all() and (ratio[good] >= 0.5).all() and (ratio[good] <= 2).all())

    def render(t, v, T):
        return raster_surface(v @ T[:3, :3].T + T[:3, 3], F, K, obs[t]['mask'].shape)
    contact_local = None

    def optimize(ids, shape):
        nonlocal theta, P, contact_local
        sd = N * 3 if shape else 0
        dim = sd + 6 * len(ids)
        status = 'ITERATION_LIMIT'
        for it in range(15):
            if time.monotonic() - start > cfg['maximum_seconds']:
                return 'TIME_LIMIT'
            v = V + B @ theta
            HH = np.zeros((dim, dim))
            gg = np.zeros(dim)
            stats = {}

            def add(J, r, norm, huber, weight=1.0):
                nonlocal HH, gg
                if not len(r):
                    return
                mag = np.linalg.norm(r, axis=1)
                w = weight * np.minimum(1.0, huber / np.maximum(mag, 1e-12)) / len(r)
                J = J / norm
                r = r / norm
                HH += np.einsum('nki,n,nkj->ij', J, w, J)
                gg += np.einsum('nki,n,nk->i', J, w, r)
            for j, t in enumerate(ids):
                o = obs[t]
                T = P[t]
                R = T[:3, :3]
                ctr = R @ cen + T[:3, 3]
                z, fi, bc = render(t, v, T)
                mod = fi >= 0
                bd = boundary(mod)
                if bd.sum() < 50:
                    return 'INSUFFICIENT_SURFACE'

                def surface(y, x):
                    faces = F[fi[y, x]]
                    ww = bc[y, x]
                    loc = (v[faces] * ww[:, :, None]).sum(1)
                    q = loc @ R.T + T[:3, 3]
                    J = np.zeros((len(q), 3, dim))
                    if shape:
                        bb = (B[faces] * ww[:, :, None]).sum(1)
                        J[:, :, :sd] = np.einsum('nb,ij->nibj', bb, R).reshape(len(q), 3, sd)
                    J[:, :, sd + 6 * j:sd + 6 * j + 3] = -skew(q - ctr)
                    J[:, :, sd + 6 * j + 3:sd + 6 * j + 6] = np.eye(3)
                    return (q, J, loc)
                sy, sx = sample(mod, 5000)
                q, J, _ = surface(sy, sx)
                dd, nn = cKDTree(q).query(o['xyz'])
                add(J[nn], q[nn] - o['xyz'], 0.01, 0.01)
                near = distance_transform_edt(~bd, return_distances=False, return_indices=True)
                ty, tx = o['boundary']
                my, mx = near[:, ty, tx]
                q, J, _ = surface(my, mx)
                uv, pj = projection(q, K)
                add(pj @ J, uv - np.c_[tx, ty], 5.0, 5.0)
                my, mx = sample(bd, 256)
                ty, tx = o['nearest'][:, my, mx]
                q, J, _ = surface(my, mx)
                vis = (my > 0) & (mx > 0) & (my < o['mask'].shape[0] - 1) & (mx < o['mask'].shape[1] - 1) & (~o['valid'][my, mx] | (o['depth'][my, mx] >= q[:, 2] - 0.02) | o['mask'][my, mx])
                uv, pj = projection(q[vis], K)
                add(pj @ J[vis], uv - np.c_[tx, ty][vis], 5.0, 5.0)
                if t == ann['frame'] and shape:
                    cy, cx = near[:, pix[1], pix[0]]
                    cq, cj, cl = surface(np.array([cy]), np.array([cx]))
                    add(cj, cq - target[None], 0.003, np.inf)
                    contact_local = cl[0]
                    stats['contact_mm'] = float(np.linalg.norm(cq[0] - target) * 1000)
                prior = np.r_[Rotation.from_matrix(R @ T0[t, :3, :3].T).as_rotvec(), ctr - (T0[t, :3, :3] @ cen + T0[t, :3, 3])]
                norm = np.r_[np.repeat(np.radians(30), 3), np.repeat(0.15, 3)]
                lam = 0.1 / norm ** 2
                sl = slice(sd + 6 * j, sd + 6 * j + 6)
                HH[sl, sl] += np.diag(lam)
                gg[sl] += lam * prior
                stats[str(t)] = dict(IoU=float((mod & o['mask']).sum() / max(1, (mod | o['mask']).sum())), surface_median_mm=float(np.median(dd) * 1000))
            if shape:
                lam = 0.05 / 0.03 ** 2
                HH[:sd, :sd] += np.eye(sd) * lam
                gg[:sd] += theta.ravel() * lam
                for a, b in edges:
                    for k in range(3):
                        ia = a * 3 + k
                        ib = b * 3 + k
                        lam = 0.1 / 0.03 ** 2
                        HH[ia, ia] += lam
                        HH[ib, ib] += lam
                        HH[ia, ib] -= lam
                        HH[ib, ia] -= lam
                        delta = theta[a, k] - theta[b, k]
                        gg[ia] += lam * delta
                        gg[ib] -= lam * delta
            step = np.linalg.solve(HH + np.eye(dim) * 1e-05, -gg)
            ds = step[:sd].reshape(N, 3) if shape else np.zeros_like(theta)
            dp = step[sd:].reshape(len(ids), 6)
            factor = min(1, 0.02 / max(1e-12, np.linalg.norm(dp[:, 3:], axis=1).max()), np.radians(5) / max(1e-12, np.linalg.norm(dp[:, :3], axis=1).max()), 0.005 / max(1e-12, np.linalg.norm(B @ ds, axis=1).max()))
            ds *= factor
            dp *= factor
            back = 0
            while shape and back < 10 and (not geom(V + B @ (theta + ds))):
                ds *= 0.5
                dp *= 0.5
                back += 1
            proposed = {}
            shifts = []
            angles = []
            for j, t in enumerate(ids):
                T = P[t].copy()
                ctr = T[:3, :3] @ cen + T[:3, 3]
                T[:3, :3] = Rotation.from_rotvec(dp[j, :3]).as_matrix() @ T[:3, :3]
                T[:3, 3] = ctr + dp[j, 3:] - T[:3, :3] @ cen
                proposed[t] = T
                shifts.append(np.linalg.norm(T[:3, :3] @ cen + T[:3, 3] - (T0[t, :3, :3] @ cen + T0[t, :3, 3])))
                angles.append(np.degrees(Rotation.from_matrix(T[:3, :3] @ T0[t, :3, :3].T).magnitude()))
            row = dict(frames=ids, iteration=it, shape=shape, max_displacement_m=float(np.linalg.norm(B @ (theta + ds), axis=1).max()), max_pose_shift_m=float(max(shifts)), max_rotation_deg=float(max(angles)), metrics=stats, seconds=time.monotonic() - start)
            if row['max_displacement_m'] > 0.08 or max(shifts) > 0.15 or max(angles) > 30 or (not geom(V + B @ (theta + ds))):
                history.append(row)
                return 'BOUND_OR_GEOMETRY_FAILED'
            theta += ds
            for t, T in proposed.items():
                P[t] = T
            history.append(row)
            print(json.dumps(row), flush=True)
        return status
    states = {'support': optimize(support, True)}
    if states['support'] == 'ITERATION_LIMIT':
        for t in held:
            states[str(t)] = optimize([t], False)
    v = V + B @ theta
    rows = []
    pics = []
    for t in support + held:
        o = obs[t]
        zz = []
        rr = []
        for label, verts, T in [('before', V, T0[t]), ('after', v, P[t])]:
            z, fi, bc = render(t, verts, T)
            mod = fi >= 0
            zz.append(z)
            r = dict(frame=t, arm=label, IoU=float((mod & o['mask']).sum() / max(1, (mod | o['mask']).sum())), coverage=float((mod & o['support']).sum() / o['support'].sum()))
            rr.append(r)
            im = cv2.imread(str(I / f'rgb/{t:06d}.png'))
            im[mod] = (im[mod] * 0.6 + np.array([240, 120, 0]) * 0.4).astype('uint8')
            cv2.drawContours(im, cv2.findContours(o['mask'].astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 240, 0), 2)
            cv2.putText(im, f'{t} {label}', (20, 35), 0, 1, (0, 0, 255), 2)
            pics.append(cv2.resize(im, (507, 380)))
        common = o['check'] & (zz[0] > 0) & (zz[1] > 0)
        for r, z in zip(rr, zz):
            err = abs(z[common] - o['depth'][common]) * 1000
            r.update(common_count=int(common.sum()), common_depth_median_mm=float(np.median(err)), common_depth_p90_mm=float(np.quantile(err, 0.9)))
        rows.extend(rr)
    z, fi, bc = render(ann['frame'], v, P[ann['frame']])
    bd = boundary(fi >= 0)
    near = distance_transform_edt(~bd, return_distances=False, return_indices=True)
    y, x = near[:, pix[1], pix[0]]
    contact_local = (v[F[fi[y, x]]] * bc[y, x, :, None]).sum(0)
    ce = float(np.linalg.norm(P[ann['frame'], :3, :3] @ contact_local + P[ann['frame'], :3, 3] - target) * 1000)
    checks = {'bounded': len(states) == 1 + len(held) and all((s == 'ITERATION_LIMIT' for s in states.values())), 'geometry': geom(v), 'contact_mm_le10': ce <= 10}
    groups = {}
    for label in ['before', 'after']:
        groups[label] = {key: float(np.mean([r[key] for r in rows if r['arm'] == label and r['frame'] in held])) for key in ['IoU', 'coverage', 'common_depth_median_mm', 'common_depth_p90_mm']}
    for key, direction in [('IoU', 1), ('coverage', 1), ('common_depth_median_mm', -1), ('common_depth_p90_mm', -1)]:
        checks['heldout_' + key] = bool(direction * groups['after'][key] >= direction * groups['before'][key])
    checks['no_IoU_large_regression'] = all((next((r['IoU'] for r in rows if r['frame'] == t and r['arm'] == 'after')) >= next((r['IoU'] for r in rows if r['frame'] == t and r['arm'] == 'before')) - 0.02 for t in held))
    new = mesh.copy()
    new.vertices = v
    new.export(out / 'mesh.obj')
    np.save(out / 'contact_point.npy', contact_local)
    np.savez_compressed(out / 'parameters.npz', control_nodes=controls, basis=B, theta=theta, poses=P[support + held], frame_ids=support + held, support=support, heldout=held, contact_camera_target=target)
    for t in support + held:
        np.save(out / f'pose_{t:06d}.npy', P[t])
    cv2.imwrite(str(out / 'comparison.jpg'), np.vstack([np.hstack(pics[i:i + 2]) for i in range(0, len(pics), 2)]))
    put(out / 'RESULTS.json', dict(state='INPUT_PASS_PENDING_VISUAL' if all(checks.values()) else 'INPUT_GATE_FAILED', states=states, checks=checks, groups=groups, contact_mm=ce, graph_sigma_m=sigma, max_shape_displacement_mm=float(np.linalg.norm(B @ theta, axis=1).max() * 1000), per_frame=rows, seconds=time.monotonic() - start, GT_reads=0))
    put(out / 'HISTORY.json', history)
    put(out / 'FREEZE.json', dict(files={p.name: sha(p) for p in out.iterdir() if p.is_file()}, code_SHA=sha(__file__), coordinate_convention=spec['coordinate_convention'], GT_used=False))
    print((out / 'RESULTS.json').read_text())
    put(out / 'MESH_POSE_PAIR.json', dict(mesh_sha256=sha(out / 'mesh.obj'), poses_sha256=sha(out / 'parameters.npz'), frame_ids=support + held, convention=spec['coordinate_convention'], poses_are_auxiliary_not_full_tracking=True, scale_baked=True))
    return {'state': 'INPUT_PASS_PENDING_VISUAL' if all(checks.values()) else 'INPUT_GATE_FAILED', 'input_gate_pass': bool(all(checks.values()))}
