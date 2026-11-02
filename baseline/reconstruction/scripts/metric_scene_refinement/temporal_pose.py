"""Fixed-mesh full-sequence stereo, silhouette and contact pose correction."""
from pathlib import Path
import time, json, shutil
import numpy as np, cv2, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import minimum_filter, maximum_filter, distance_transform_edt
from common import sha, put, fields, mesh_name, bundle_files
from renderer import raster_surface
from geometry import skew, projection
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import datetime

def run(spec, H):
    start = time.monotonic()
    p = spec['parameters']
    inputs = spec['inputs']
    B = Path(inputs['bundle'])
    I = Path(inputs['observations'])
    ck, wk = fields(spec['object_slot'])
    old = dict(np.load(B / 'result.npz'))
    N = len(old[ck])
    W0 = old[wk].astype(float)
    C = old['camera_to_world_transform'].astype(float)
    CW = np.linalg.inv(C)
    mesh = trimesh.load(B / mesh_name(spec['object_slot']), process=False)
    V = np.asarray(mesh.vertices)
    cen = (V.min(0) + V.max(0)) / 2
    plane = np.load(Path(inputs['table_plane']))
    normal = plane[:3]
    offset = plane[3]
    assert np.isclose(np.linalg.norm(normal), 1, atol=1e-06)
    center0 = np.einsum('nij,j->ni', W0[:, :3, :3], cen) + W0[:, :3, 3]
    obs = np.load(Path(inputs['landmarks']))
    ids = obs['frame_ids']
    good = obs['qualified']
    X = obs['points_camera']
    local = np.full((X.shape[1], 3), np.nan)
    calibration = np.isin(ids, p['calibration_frames'])
    baseT = old[ck].astype(float)
    for j in range(X.shape[1]):
        ix = np.where(calibration & good[:, j])[0]
        assert len(ix) >= 2
        w = ids[ix]
        loc = np.einsum('nij,nj->ni', baseT[w, :3, :3].transpose(0, 2, 1), X[ix, j] - baseT[w, :3, 3])
        local[j] = np.median(loc, axis=0)
    put(H / 'LOCAL_LANDMARKS.json', dict(points=local.tolist(), frames=p['calibration_frames'], source='Frozen stereo observations calibrated into the matching mesh frame using declared input frames.'))
    sig = obs['sigma_z']
    use = good & np.isin(ids, p['fit_frames'])[:, None]
    ii, jj = np.where(use)
    times = ids[ii]
    target = X[ii, jj]
    sigmas = np.maximum(0.005, sig[ii, jj])
    weight = np.ones(len(ii))
    for n, (i, j) in enumerate(zip(ii, jj)):
        group = np.asarray(p['landmark_groups'])
        count = int(use[i, group == group[j]].sum())
        weight[n] = 1 / np.sqrt(count)

    def state(x):
        d = x.reshape(N, 6)
        rot = Rotation.from_rotvec(d[:, 3:]).as_matrix() @ W0[:, :3, :3]
        center = center0 + d[:, :3]
        trans = center - np.einsum('nij,j->ni', rot, cen)
        return (rot, trans, center)

    def points(x):
        r, t, _ = state(x)
        world = np.einsum('nij,kj->nki', r, local) + t[:, None, :]
        return np.einsum('nij,nkj->nki', CW[:, :3, :3], world) + CW[:, None, :3, 3]

    def gaps(r, t):
        contact = np.asarray(json.loads(Path(inputs['calibration']).read_text())['contact_point_local_m'])
        return np.einsum('nij,j->ni', r[p['contact_frames']], contact) @ normal + t[p['contact_frames']] @ normal + offset
    k = json.loads((I / 'K_fp.json').read_text())
    K = np.array([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1.0]])
    masks = []
    target_boundaries = []
    near_targets = []

    def boundary(mask):
        return mask & ~(cv2.erode(mask.astype('uint8'), np.ones((3, 3), np.uint8)) > 0)

    def sample(mask):
        y, x = np.where(mask)
        select = np.linspace(0, len(x) - 1, min(128, len(x)), dtype=int)
        return (y[select], x[select])
    for frame in range(N):
        mask = cv2.imread(str(I / f'masks/{frame:06d}.png'), 0) > 0
        masks.append(mask)
        bd = boundary(mask)
        target_boundaries.append(sample(bd))
        near_targets.append(distance_transform_edt(~bd, return_distances=False, return_indices=True))
    contour = None

    def correspondences(x):
        r, t, c = state(x)
        all_local = []
        all_t = []
        all_uv = []
        all_w = []
        ious = []
        for frame in range(N):
            rc = CW[frame, :3, :3] @ r[frame]
            tc = CW[frame, :3, :3] @ t[frame] + CW[frame, :3, 3]
            z, fi, bc = raster_surface(V @ rc.T + tc, np.asarray(mesh.faces), K, masks[frame].shape)
            mod = fi >= 0
            bd = boundary(mod)
            assert bd.any()
            ious.append(float((mod & masks[frame]).sum() / max(1, (mod | masks[frame]).sum())))
            near_model = distance_transform_edt(~bd, return_distances=False, return_indices=True)
            ty, tx = target_boundaries[frame]
            sy, sx = (near_model[0, ty, tx], near_model[1, ty, tx])
            my, mx = sample(bd)
            n = near_targets[frame]
            uy, ux = (n[0, my, mx], n[1, my, mx])
            for y, x1, uy1, ux1 in [(sy, sx, ty, tx), (my, mx, uy, ux)]:
                face = np.asarray(mesh.faces)[fi[y, x1]]
                pts = (V[face] * bc[y, x1, :, None]).sum(1)
                all_local.append(pts)
                all_t.extend([frame] * len(x1))
                all_uv.append(np.c_[ux1, uy1])
                all_w.extend([1 / np.sqrt(len(x1))] * len(x1))
        return (np.concatenate(all_local), np.array(all_t, dtype=int), np.concatenate(all_uv), np.array(all_w), float(np.mean(ious)))
    count = 0

    def residual(x):
        nonlocal count
        if time.monotonic() - start > p['maximum_seconds']:
            raise TimeoutError('Frozen CPU budget reached')
        r, t, c = state(x)
        world = np.einsum('nij,nj->ni', r[times], local[jj]) + t[times]
        q = np.einsum('nij,nj->ni', CW[times, :3, :3], world) + CW[times, :3, 3]
        data = ((q - target) / sigmas[:, None] * weight[:, None]).ravel()
        d = x.reshape(N, 6)
        prior = (d / np.array([0.04] * 3 + [np.radians(20)] * 3)).ravel()
        smooth = (np.diff(d, 2, axis=0) / np.array([0.003] * 3 + [np.radians(2)] * 3)).ravel()
        contact = gaps(r, t) / 0.002
        count += 1
        rr = np.r_[data, prior, smooth, contact]
        if contour is not None:
            cp, cf, uv, wt, _ = contour
            world = np.einsum('nij,nj->ni', r[cf], cp) + t[cf]
            qc = np.einsum('nij,nj->ni', CW[cf, :3, :3], world) + CW[cf, :3, 3]
            uvc = (qc @ K.T)[:, :2] / qc[:, 2:]
            rr = np.r_[rr, ((uvc - uv) / 5 * wt[:, None]).ravel()]
        return rr
    x0 = np.zeros(N * 6)
    nr = len(times) * 3 + N * 6 + (N - 2) * 6 + len(p['contact_frames'])
    pattern = lil_matrix((nr, N * 6), dtype=np.uint8)
    row = 0
    for t in times:
        pattern[row:row + 3, t * 6:(t + 1) * 6] = 1
        row += 3
    for t in range(N):
        pattern[row:row + 6, t * 6:(t + 1) * 6] = 1
        row += 6
    for t in range(N - 2):
        pattern[row:row + 6, t * 6:(t + 3) * 6] = 1
        row += 6
    for t in p['contact_frames']:
        pattern[row, t * 6:(t + 1) * 6] = 1
        row += 1
    assert row == nr and len(residual(x0)) == nr
    rr, tt, cc = state(x0)
    np.testing.assert_allclose(rr, W0[:, :3, :3], atol=1e-12)
    np.testing.assert_allclose(tt, W0[:, :3, 3], atol=1e-12)
    manual = np.einsum('nij,kj->nki', old[ck][:, :3, :3], local) + old[ck][:, None, :3, 3]
    np.testing.assert_allclose(points(x0), manual, atol=1e-06)
    put(H / 'SOLVER_INPUTS.json', dict(residual_count=nr, parameter_count=len(x0), contract_checks=3, GT_used=False))
    bounds = np.tile([0.06] * 3 + [np.radians(30)] * 3, N)
    print('SOLVE_START', len(times), 'measurements', nr, 'residuals', flush=True)
    current = x0.copy()
    history = []
    for outer in range(10):
        contour = correspondences(current)
        cp, cf, _, _, iou = contour
        sp = lil_matrix((nr + 2 * len(cf), N * 6), dtype=np.uint8)
        sp[:nr] = pattern
        for q, frame in enumerate(cf):
            sp[nr + 2 * q:nr + 2 * q + 2, frame * 6:(frame + 1) * 6] = 1
        fit = least_squares(residual, current, jac_sparsity=sp.tocsr(), bounds=(-bounds, bounds), loss='soft_l1', f_scale=1, max_nfev=8, verbose=0, x_scale=np.tile([0.04] * 3 + [np.radians(20)] * 3, N))
        current = fit.x
        item = dict(outer=outer, IoU_before=iou, cost=float(fit.cost), nfev=int(fit.nfev), correspondences=len(cf), seconds=time.monotonic() - start)
        history.append(item)
        print(json.dumps(item), flush=True)
    put(H / 'OPTIMIZATION_HISTORY.json', history)
    rot, trans, center = state(fit.x)
    W = np.tile(np.eye(4), (N, 1, 1))
    W[:, :3, :3] = rot
    W[:, :3, 3] = trans
    T = CW @ W
    delta = fit.x.reshape(N, 6)
    angle = np.degrees(np.linalg.norm(delta[:, 3:], axis=1))
    shift = np.linalg.norm(delta[:, :3], axis=1)
    bounds_ok = angle.max() <= 30 and shift.max() <= 0.06
    np.testing.assert_allclose(T[:, :3, :3].transpose(0, 2, 1) @ T[:, :3, :3], np.tile(np.eye(3), (N, 1, 1)), atol=1e-05)
    assert np.isfinite(W).all()
    np.savez_compressed(H / 'POSES.npz', world=W, camera=T, frame_ids=np.arange(N), correction=delta, local_landmarks=local)
    groups = {}
    for label, x in [('baseline', x0), ('candidate', fit.x)]:
        pred = points(x)[ids]
        errors = np.linalg.norm(pred - X, axis=-1) * 1000
        groups[label] = {}
        for group, mask in [('support', use), ('heldout', good & np.isin(ids, p['validation_frames'])[:, None]), ('all', good)]:
            e = errors[mask]
            groups[label][group] = dict(count=len(e), median_mm=float(np.median(e)), p90_mm=float(np.quantile(e, 0.9)), mean_mm=float(e.mean()))
    np.savez_compressed(H / 'LANDMARK_ERRORS.npz', baseline=np.linalg.norm(points(x0)[ids] - X, axis=-1) * 1000, candidate=np.linalg.norm(points(fit.x)[ids] - X, axis=-1) * 1000, qualified=good, frame_ids=ids)
    rec = dict(outer_passes=10, outer_state='FIXED_CORRESPONDENCE_BUDGET_COMPLETE', state='COMPLETE_BOUNDED_CANDIDATE' if bounds_ok else 'COMPLETE_CANDIDATE_NORM_BOUND_FAILED', scipy_status=int(fit.status), scipy_message=fit.message, nfev=fit.nfev, function_calls=count, optimality=float(fit.optimality), cost=float(fit.cost), max_center_shift_mm=float(shift.max() * 1000), max_rotation_correction_deg=float(angle.max()), max_abs_rest_gap_mm=float(abs(gaps(rot, trans)).max() * 1000), groups=groups, seconds=time.monotonic() - start, GT_used=False, solver_inputs_SHA=sha(H / 'SOLVER_INPUTS.json'))
    put(H / 'FIT_RECEIPT.json', rec)
    if not bounds_ok:
        return {'state': rec['state'], 'bounds_ok': False, 'input_gate_pass': False}
    out = H / 'bundle'
    out.mkdir(exist_ok=False)
    for f in bundle_files(B):
        if f.name != 'result.npz':
            target_file = out / f.relative_to(B)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target_file)
    b = {k: v.copy() for k, v in old.items()}
    b[wk] = W
    b[ck] = T
    unchanged = []
    for k, v in old.items():
        if k not in [wk, ck]:
            assert np.array_equal(v, b[k], equal_nan=True) if np.issubdtype(v.dtype, np.number) else np.array_equal(v, b[k])
            unchanged.append(k)
    np.savez_compressed(out / 'result.npz', **b)
    run = dict(state='FROZEN_BEFORE_NEW_GT_SCORE', method='Fixed metric mesh; calibrated rigid stereo landmarks, full-sequence silhouette and temporal SE3; input-declared contact intervals. Experimental candidate, not a production replacement.', frames=N, source_bundle_SHA=sha(B / 'result.npz'), mesh_SHA=sha(out / mesh_name(spec['object_slot'])), poses_SHA=sha(H / 'POSES.npz'), unchanged_fields=unchanged, coordinate_convention='Public object-local mesh and matching object-to-camera/world poses, metric meters. No GT alignment in bundle.', input_gate_pending=True, GT_used=False, historical_Public_GT_exposed=spec.get('historical_Public_GT_exposed', False), production_adopted=False)
    put(out / 'run.json', run)
    put(H / 'PREDICTION_FREEZE.json', dict(state='FROZEN_BEFORE_NEW_GT_SCORE', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), files={p.name: sha(p) for p in out.iterdir() if p.is_file()}, fixed_halves=[[0, N // 2], [N // 2, N]], coordinate_convention=spec['coordinate_convention']))
    print(json.dumps(rec, indent=2), flush=True)
    return {'state': rec['state'], 'input_gate_pass': False, 'bounds_ok': bool(bounds_ok), 'needs_external_input_review': True}
