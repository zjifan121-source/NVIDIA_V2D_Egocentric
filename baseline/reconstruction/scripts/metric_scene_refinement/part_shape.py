"""Input-annotated shared body/part deformation with independent support poses."""
from pathlib import Path
import time, json, shutil
import numpy as np, cv2, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import minimum_filter, maximum_filter, distance_transform_edt
from common import sha, put, fields, mesh_name, bundle_files
from renderer import raster_surface
from geometry import skew, projection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

def sample(y, x, n):
    ids = np.linspace(0, len(x) - 1, min(n, len(x)), dtype=int)
    return (y[ids], x[ids])

def build_basis(m, marks):
    V = np.asarray(m.vertices)
    F = np.asarray(m.faces)
    aa = marks['parts']
    center = V[marks['center']]
    U, inv = np.unique(np.round(V, 8), axis=0, return_inverse=True)
    e = np.unique(np.sort(inv[m.edges], axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    length = np.linalg.norm(U[e[:, 0]] - U[e[:, 1]], axis=1)
    adj = coo_matrix((np.r_[length, length], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(U), len(U))).tocsr()
    dist = dijkstra(adj, indices=inv[aa], directed=False)
    radii = dist[:, inv[marks['center']]] * 0.5
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('Part anchors must connect to a distinct center on the mesh')
    u = np.clip(dist[:, inv] / radii[:, None], 0, 1)
    W = ((1 - u) ** 2 * (1 + 2 * u)).T
    body = 1 - W.sum(1)
    pp = V[body > 0.95] - center
    if len(pp) < 3 or np.linalg.matrix_rank(pp) < 2:
        raise ValueError('Insufficient body geometry for the fixed PCA basis')
    _, axes = np.linalg.eigh(pp.T @ pp)
    axes = axes[:, ::-1]
    for j in range(3):
        ix = np.argmax(abs(axes[:, j]))
        axes[:, j] *= np.sign(axes[ix, j])
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    basis = np.zeros((len(V), 3, 9))
    rel = V - center
    for j in range(3):
        basis[:, :, j] = body[:, None] * (rel @ axes[:, j])[:, None] * axes[:, j]
    for j in range(2):
        basis[:, :, 3 + 3 * j:6 + 3 * j] = W[:, j, None, None] * np.eye(3)
    assert np.max(abs(basis[marks['center']])) < 1e-12
    return dict(vertices=V, faces=F, basis=basis, center=center, axes=axes, weights=W, radii=radii)

def run(spec, H):
    start = time.monotonic()
    cfg = dict(spec['parameters'])
    inputs = spec['inputs']
    out = H / 'output'
    out.mkdir(exist_ok=False)
    src = Path(inputs['observations'])
    root = Path(inputs['bundle'])
    mesh = trimesh.load(root / mesh_name(spec['object_slot']), force='mesh', process=False)
    marks = json.loads(Path(inputs['semantic_vertices']).read_text())
    z = build_basis(mesh, marks)
    V = z['vertices']
    F = z['faces']
    basis = z['basis']
    center = z['center']
    T0 = np.load(root / 'result.npz')[fields(spec['object_slot'])[0]].astype(float)
    T = T0.copy()
    support = cfg['support']
    allframes = support + cfg['heldout']
    theta = np.zeros(9)
    radius = np.linalg.norm(V - center, axis=1).max()
    cfg['part_displacement_bound_m'] = 0.25 * z['radii']
    bounds = np.r_[np.repeat(0.25, 3), np.repeat(cfg['part_displacement_bound_m'], 3)]
    dim = 9 + 6 * len(support)
    np.savez_compressed(out / 'basis.npz', **z)
    edges = mesh.edges_unique
    e0 = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    n0 = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    good_normal = np.linalg.norm(n0, axis=1) > 1e-15
    k = json.loads((src / 'K_fp.json').read_text())
    K = np.array([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1.0]])
    obs = {}
    for t in allframes:
        mask = cv2.imread(str(src / f'masks/{t:06d}.png'), 0) > 0
        raw = cv2.imread(str(src / f'depth/{t:06d}.png'), -1)
        valid = (raw > 0) & (raw < 65535)
        dep = np.zeros(mask.shape)
        dep[valid] = 65535.0 / raw[valid] - 1
        lo = minimum_filter(np.where(valid, dep, np.inf), 5)
        hi = maximum_filter(np.where(valid, dep, -np.inf), 5)
        trusted = valid & (hi - lo <= 0.02) & (cv2.erode(mask.astype('uint8'), np.ones((3, 3), np.uint8)) > 0)
        y, x = sample(*np.where(trusted), 1024)
        fitxyz = np.c_[x, y, np.ones(len(x))] @ np.linalg.inv(K).T * dep[y, x, None]
        py, px = sample(*np.where(mask), 256)
        _, near = distance_transform_edt(~mask, return_indices=True)
        obs[t] = dict(mask=mask, depth=dep, valid=valid, trusted=trusted, xyz=fitxyz, positive=np.c_[px, py], near=near, rgb=cv2.imread(str(src / f'rgb/{t:06d}.png')))

    def geometry(v):
        n = np.cross(v[F[:, 1]] - v[F[:, 0]], v[F[:, 2]] - v[F[:, 0]])
        rat = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1) / np.maximum(e0, 1e-20)
        ee = e0 > 1e-10
        return bool(((n * n0).sum(1)[good_normal] >= 0).all() and (rat[ee] >= 0.5).all() and (rat[ee] <= 2).all())

    def newverts(x):
        return V + np.einsum('vij,j->vi', basis, x)
    history = []
    status = 'ITERATION_LIMIT'
    for it in range(cfg['maximum_iterations']):
        if time.monotonic() - start > cfg['maximum_seconds']:
            status = 'TIME_LIMIT'
            break
        v = newverts(theta)
        HH = np.zeros((dim, dim))
        gg = np.zeros(dim)
        statistics = {}

        def add(J, r, norm, depth=False):
            nonlocal HH, gg
            if not len(r):
                return
            mag = np.linalg.norm(r, axis=1)
            if depth:
                w = np.where(mag < 0.04, (1 - (mag / 0.04) ** 2) ** 2, 0.0)
            else:
                w = np.minimum(1, 5 / np.maximum(mag, 1e-12))
            w = w / len(r)
            J = J / norm
            r = r / norm
            HH += np.einsum('nki,n,nkj->ij', J, w, J)
            gg += np.einsum('nki,n,nk->i', J, w, r)
        for si, t in enumerate(support):
            o = obs[t]
            R = T[t, :3, :3]
            tr = T[t, :3, 3]
            cc = R @ center + tr
            rz, fi, bc = raster_surface(v @ R.T + tr, F, K, o['mask'].shape)
            model = fi >= 0
            sy, sx = sample(*np.where(model), 6000)
            if len(sx) < 50:
                raise ValueError(f'UNRENDERABLE_INITIAL_ASSET_FRAME_{t}')

            def surface(y, x):
                f = F[fi[y, x]]
                w = bc[y, x]
                loc = (v[f] * w[:, :, None]).sum(1)
                q = loc @ R.T + tr
                bb = (basis[f] * w[:, :, None, None]).sum(1)
                J = np.zeros((len(q), 3, dim))
                J[:, :, :9] = np.einsum('ij,njk->nik', R, bb)
                J[:, :, 9 + 6 * si:12 + 6 * si] = -skew(q - cc)
                J[:, :, 12 + 6 * si:15 + 6 * si] = np.eye(3)
                return (q, J)
            surfaceq, surfaceJ = surface(sy, sx)
            dd, nn = cKDTree(surfaceq).query(o['xyz'])
            add(surfaceJ[nn], surfaceq[nn] - o['xyz'], 0.01, depth=True)
            pu = o['positive']
            pu = pu[~model[pu[:, 1], pu[:, 0]]]
            if len(pu):
                _, near = distance_transform_edt(~model, return_indices=True)
                my = near[0, pu[:, 1], pu[:, 0]]
                mx = near[1, pu[:, 1], pu[:, 0]]
                q, J = surface(my, mx)
                uv, PJ = projection(q, K)
                add(PJ @ J, uv - pu, 5.0)
            neg = model & ~o['mask'] & o['valid'] & (o['depth'] > rz + 0.02)
            ny, nx = sample(*np.where(neg), 256)
            if len(nx):
                q, J = surface(ny, nx)
                uv, PJ = projection(q, K)
                target = np.c_[o['near'][1, ny, nx], o['near'][0, ny, nx]]
                add(PJ @ J, uv - target, 5.0)
            statistics[str(t)] = {'IoU': float((model & o['mask']).sum() / max(1, (model | o['mask']).sum())), 'depth_target_samples': len(dd), 'depth_residual_median_mm': float(np.median(dd) * 1000), 'depth_positive_influence_samples': int((dd < 0.04).sum())}
        prior = 0.1 / bounds ** 2
        HH[:9, :9] += np.diag(prior)
        gg[:9] += prior * theta
        for si, t in enumerate(support):
            r = Rotation.from_matrix(T[t, :3, :3] @ T0[t, :3, :3].T).as_rotvec()
            d = T[t, :3, :3] @ center + T[t, :3, 3] - (T0[t, :3, :3] @ center + T0[t, :3, 3])
            pn = np.r_[np.repeat(radius, 3), np.ones(3)]
            lam = 0.1 / 0.01 ** 2
            ix = slice(9 + 6 * si, 15 + 6 * si)
            HH[ix, ix] += np.diag(lam * pn ** 2)
            gg[ix] += lam * pn ** 2 * np.r_[r, d]
        step = np.linalg.solve(HH + np.eye(27) * 1e-05, -gg)
        sv = np.einsum('vij,j->vi', basis, step[:9])
        ps = step[9:].reshape(len(support), 6)
        factor = min(1.0, 0.005 / max(np.linalg.norm(sv, axis=1).max(), 1e-12), 0.02 / max(np.linalg.norm(ps[:, 3:], axis=1).max(), 1e-12), np.radians(5) / max(np.linalg.norm(ps[:, :3], axis=1).max(), 1e-12))
        step *= factor
        bt = 0
        while bt < 10 and (not geometry(newverts(theta + step[:9]))):
            step *= 0.5
            bt += 1
        trial = theta + step[:9]
        candidate = T.copy()
        ps = step[9:].reshape(len(support), 6)
        for si, t in enumerate(support):
            cc = T[t, :3, :3] @ center + T[t, :3, 3]
            candidate[t, :3, :3] = Rotation.from_rotvec(ps[si, :3]).as_matrix() @ T[t, :3, :3]
            candidate[t, :3, 3] = cc + ps[si, 3:] - candidate[t, :3, :3] @ center
        movements = np.array([np.linalg.norm(candidate[t, :3, :3] @ center + candidate[t, :3, 3] - T0[t, :3, :3] @ center - T0[t, :3, 3]) for t in support])
        rot = np.array([Rotation.from_matrix(candidate[t, :3, :3] @ T0[t, :3, :3].T).magnitude() * 180 / np.pi for t in support])
        row = dict(iteration=it, theta_proposal=trial.tolist(), pose_max_translation_m=float(movements.max()), pose_max_rotation_deg=float(rot.max()), geometry_backtracks=bt, seconds=time.monotonic() - start, support=statistics)
        if (abs(trial[:3]) > 0.25).any() or any((np.linalg.norm(trial[3 + j * 3:6 + j * 3]) > cfg['part_displacement_bound_m'][j] for j in range(2))):
            status = 'SHAPE_BOUND'
            history.append(row)
            break
        if movements.max() > 0.15 or rot.max() > 30:
            status = 'POSE_BOUND'
            history.append(row)
            break
        if not geometry(newverts(trial)):
            status = 'INVALID_GEOMETRY'
            history.append(row)
            break
        theta = trial
        T = candidate
        history.append(row)
        print(json.dumps(row), flush=True)
    v = newverts(theta)
    new = mesh.copy()
    new.vertices = v
    new.export(out / 'mesh.obj')
    reload = trimesh.load(out / 'mesh.obj', force='mesh', process=False)
    assert np.array_equal(reload.faces, F) and np.max(abs(reload.vertices - v)) < 1e-07
    np.savez_compressed(out / 'candidate_parameters.npz', theta=theta, auxiliary_poses=T[support], support_frames=support, basis=basis)
    put(out / 'FIT_RECEIPT.json', dict(state=status, seconds=time.monotonic() - start, theta=theta.tolist(), history=history, geometry_valid=geometry(v), poses_are_auxiliary_not_delivery=True, GT_reads=0, GPU=0))
    rows = []
    pictures = []
    for t in allframes:
        o = obs[t]
        zs = []
        rr = []
        variants = [('old', V, T0[t]), ('shape_only', v, T0[t])] + ([('joint_fit_only', v, T[t])] if t in support else [])
        for label, verts, pose in variants:
            rz, fi, bc = raster_surface(verts @ pose[:3, :3].T + pose[:3, 3], F, K, o['mask'].shape)
            model = fi >= 0
            sup = o['trusted']
            over = model & sup
            err = abs(rz[over] - o['depth'][over])
            zs.append(rz)
            rr.append({'frame': t, 'variant': label, 'IoU': float((model & o['mask']).sum() / max(1, (model | o['mask']).sum())), 'coverage': float(over.sum() / max(1, sup.sum())), 'trusted_depth_pixels': int(sup.sum()), 'mask_finite_depth_pixels': int((o['mask'] & o['valid']).sum()), 'depth_median_mm': float(np.median(err) * 1000) if len(err) else None, 'depth_p90_mm': float(np.quantile(err, 0.9) * 1000) if len(err) else None})
            if label != 'joint_fit_only':
                im = o['rgb'].copy()
                im[model] = (im[model] * 0.65 + np.array([255, 175, 30]) * 0.35).astype('uint8')
                cv2.drawContours(im, cv2.findContours(o['mask'].astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (20, 220, 20), 1)
                cv2.putText(im, f'f{t} {label}', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 220), 2)
                pictures.append(cv2.resize(im, (507, 380)))
        common = (zs[0] > 0) & (zs[1] > 0) & o['trusted']
        for row, rz in zip(rr, zs):
            e = abs(rz[common] - o['depth'][common])
            row['common_median_mm'] = float(np.median(e) * 1000) if len(e) else None
            row['common_p90_mm'] = float(np.quantile(e, 0.9) * 1000) if len(e) else None
        rows.extend(rr)
    groups = {}
    checks = {}
    for label, fs in [('support', support), ('heldout', cfg['heldout'])]:
        aa = [r for r in rows if r['frame'] in fs and r['variant'] == 'old']
        bb = [r for r in rows if r['frame'] in fs and r['variant'] == 'shape_only']
        metrics = ['IoU', 'coverage', 'common_median_mm', 'common_p90_mm']
        means = {v: {k: float(np.mean([r[k] for r in seq])) for k in metrics} for v, seq in [('old', aa), ('shape_only', bb)]}
        groups[label] = means
        a, b = (means['old'], means['shape_only'])
        checks[label + '_mean_shape_improved'] = bool(b['IoU'] > a['IoU'] and b['coverage'] >= a['coverage'] and (b['common_median_mm'] <= a['common_median_mm']) and (b['common_p90_mm'] <= a['common_p90_mm']))
        checks[label + '_no_large_individual_regression'] = all((b['IoU'] >= a['IoU'] - 0.02 and b['common_median_mm'] <= a['common_median_mm'] + 5 for a, b in zip(aa, bb)))
    checks['valid_geometry'] = geometry(v)
    checks['bounded_execution'] = status == 'ITERATION_LIMIT'
    passed = all(checks.values())
    result = dict(state='INPUT_SUPPORTED_SHAPE_REQUIRES_VISUAL_REVIEW' if passed else 'MULTIVIEW_SHAPE_NOT_ACCEPTED', solver_state=status, seconds=time.monotonic() - start, shape_parameters=theta.tolist(), groups=groups, checks=checks, per_frame=rows, release_to_GPU=False, release_pending_visual=passed, new_AUC=None, GT_reads=0, mesh_sha256=sha(out / 'mesh.obj'), original_mesh_SHA_unchanged=sha(root / mesh_name(spec['object_slot'])) == spec['mesh_pose_pair']['mesh_sha256'])
    put(out / 'RESULTS.json', result)
    cv2.imwrite(str(out / 'comparison.jpg'), np.vstack([np.hstack(pictures[i:i + 2]) for i in range(0, len(pictures), 2)]))
    put(out / 'PREDICTION_FREEZE.json', {'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()}, 'GT_read': False, 'auxiliary_poses_not_for_delivery': True})
    print(json.dumps({k: v for k, v in result.items() if k != 'per_frame'}, indent=2))
    put(out / 'MESH_POSE_PAIR.json', dict(mesh_sha256=sha(out / 'mesh.obj'), poses_sha256=sha(out / 'candidate_parameters.npz'), frame_ids=support, convention=spec['coordinate_convention'], poses_are_auxiliary_not_full_tracking=True, scale_baked=True))
    return {'state': result['state'], 'input_gate_pass': bool(passed)}
