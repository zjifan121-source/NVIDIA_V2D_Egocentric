"""Input-only shared uniform scale and per-frame SE(3), CPU solver.

Mesh points are centered for numerical conditioning only. Observations remain
in each frame's camera coordinates; no SLAM transform or GT is used in fitting.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .geometry_cpu import raster


@dataclass(frozen=True)
class Config:
    iterations: int = 15
    minimum_correspondences: int = 200
    max_correspondence_m: float = 0.15
    retain_fraction: float = 0.9
    huber_m: float = 0.01
    scale_min: float = 0.5
    scale_max: float = 2.0
    translation_bound_m: float = 0.15
    rotation_bound_deg: float = 30.0


def validate_pose(T):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Expected finite 4x4 object-to-camera pose')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('Invalid homogeneous pose')
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError('Pose must be rigid: bake input mesh scale before fitting')
    if not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5):
        raise ValueError('Pose rotation is not SO(3)')
    return T.copy()


def centered_pose(T, center):
    out = validate_pose(T)
    out[:3, 3] += out[:3, :3] @ center
    return out


def public_pose(T_center, center):
    out = validate_pose(T_center)
    out[:3, 3] -= out[:3, :3] @ center
    return out


def sample_surface(vertices, faces, count=18000, seed=0):
    """Deterministic area-uniform samples, including a triangle-normal record."""
    tri = np.asarray(vertices)[np.asarray(faces)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(n, axis=1)
    if area2.sum() <= 0:
        raise ValueError('Mesh has no non-degenerate faces')
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(tri), count, p=area2 / area2.sum())
    a, b = rng.random((2, count))
    a = np.sqrt(a)
    weights = np.stack([1-a, a*(1-b), a*b], axis=1)
    return np.einsum('ni,nij->nj', weights, tri[ids])


def visible_samples(points, pose, scale, K, shape):
    """Conservative sampled z-buffer in the SAME frame, never across frames.

    Retains near-front samples within a fixed 3 mm tolerance per 2px cell.
    This is for ICP correspondences, not a full triangle renderer or mask IoU.
    """
    q = (points * scale) @ pose[:3, :3].T + pose[:3, 3]
    uvz = q @ K.T
    good = q[:, 2] > 1e-5
    uv = np.full((len(q), 2), -1.0)
    uv[good] = uvz[good, :2] / uvz[good, 2, None]
    h, w = shape
    good &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ids = np.flatnonzero(good)
    if not len(ids):
        return ids, q[ids]
    cells = np.floor(uv[ids] / 2).astype(int)
    bins_w = (w + 1) // 2
    bins_h = (h + 1) // 2
    key = cells[:, 1] * bins_w + cells[:, 0]
    front = np.full(bins_h * bins_w, np.inf)
    np.minimum.at(front, key, q[ids, 2])
    ids = ids[q[ids, 2] <= front[key] + 0.003]
    return ids, q[ids]


def unproject(depth, K, valid, maximum=1200):
    y, x = np.where(valid & np.isfinite(depth) & (depth > 0))
    if len(x) > maximum:
        keep = np.linspace(0, len(x)-1, maximum, dtype=int)
        y, x = y[keep], x[keep]
    rays = np.stack([x, y, np.ones(len(x))], axis=1) @ np.linalg.inv(K).T
    return rays * depth[y, x, None]


def model_surface(points, pose, scale, K, shape, mesh=None):
    if mesh is None:
        ids,surface=visible_samples(points,pose,scale,K,shape)
        return points[ids],surface
    vertices,faces=mesh
    h,w=shape;factor=min(1.,512/w);k=K.copy();k[:2]*=factor
    hw=(round(h*factor),round(w*factor))
    depth=raster((vertices*scale)@pose[:3,:3].T+pose[:3,3],faces,k,hw)
    surface=unproject(depth,k,depth>0,maximum=5000)
    object_points=(surface-pose[:3,3])@pose[:3,:3]/scale
    return object_points,surface


def correspondences(points, observations, pose, scale, K, shape, cfg, mesh=None):
    model_points, surface = model_surface(points,pose,scale,K,shape,mesh)
    if len(model_points) < cfg.minimum_correspondences:
        raise ValueError('INSUFFICIENT_VISIBLE_MODEL_SUPPORT')
    distances, neighbors = cKDTree(surface).query(observations, workers=1)
    cutoff = min(cfg.max_correspondence_m, float(np.quantile(distances, cfg.retain_fraction)))
    keep = distances <= cutoff
    if keep.sum() < cfg.minimum_correspondences:
        raise ValueError('INSUFFICIENT_DEPTH_CORRESPONDENCES')
    d = distances[keep]
    weights = np.minimum(1.0, cfg.huber_m / np.maximum(d, 1e-12))
    weights /= weights.sum()
    # The reverse term prevents an oversized visible model from explaining
    # only a small interior subset of itself. Use clear, untruncated support
    # frames; this does not pretend unseen/occluded surfaces are measurements.
    sample_ids = np.linspace(0, len(surface)-1, min(1200, len(surface)), dtype=int)
    reverse_d, reverse_nn = cKDTree(observations).query(surface[sample_ids], workers=1)
    reverse_cutoff = min(cfg.max_correspondence_m, float(np.quantile(reverse_d, cfg.retain_fraction)))
    reverse_keep = reverse_d <= reverse_cutoff
    if reverse_keep.sum() < cfg.minimum_correspondences:
        raise ValueError('INSUFFICIENT_REVERSE_SUPPORT')
    rd = reverse_d[reverse_keep]
    rw = np.minimum(1.0, cfg.huber_m / np.maximum(rd, 1e-12))
    rw /= rw.sum()
    X = np.concatenate([model_points[neighbors[keep]], model_points[sample_ids[reverse_keep]]])
    Y = np.concatenate([observations[keep], observations[reverse_nn[reverse_keep]]])
    weights = .5 * np.concatenate([weights, rw])  # Equal direction and frame mass.
    return X, Y, weights, {
        'correspondences': int(keep.sum()), 'median_distance_m': float(np.median(d)),
        'p90_distance_m': float(np.quantile(d, 0.9)),
        'visible_mesh_samples': len(model_points), 'cutoff_m': cutoff,
        'reverse_correspondences': int(reverse_keep.sum()),
        'reverse_median_distance_m': float(np.median(rd))}


def solve_corresponded_shared_scale(rows, fixed_scale=None):
    """Exact weighted similarity solve with shared s and independent rigid poses.

    rows: (object-frame points, camera-frame measurements, nonnegative weights).
    No common camera transform or pooled dynamic-object point cloud is formed.
    """
    centered = []
    numerator = denominator = 0.0
    for X, Y, weight in rows:
        weight = np.asarray(weight, float)
        if len(X) < 3 or np.any(weight < 0) or weight.sum() <= 0:
            raise ValueError('Degenerate correspondences')
        weight = weight / weight.sum()
        xb = np.sum(X * weight[:, None], axis=0)
        yb = np.sum(Y * weight[:, None], axis=0)
        xc, yc = X-xb, Y-yb
        U, _, Vt = np.linalg.svd(xc.T @ (weight[:, None] * yc))
        correction = np.diag([1., 1., np.linalg.det(Vt.T @ U.T)])
        R = Vt.T @ correction @ U.T
        numerator += np.sum(weight * np.sum(yc * (xc @ R.T), axis=1))
        denominator += np.sum(weight * np.sum(xc * xc, axis=1))
        centered.append((R, xb, yb))
    if denominator < 1e-12:
        raise ValueError('UNOBSERVABLE_SCALE: no object extent')
    scale = float(numerator / denominator) if fixed_scale is None else float(fixed_scale)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('Invalid fitted scale')
    poses = []
    for R, xb, yb in centered:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = yb - scale * R @ xb
        poses.append(T)
    return scale, poses


def fit(points, frames, cfg=Config(), fixed_scale=None):
    """Local robust ICP; returns a bounded candidate or explicit non-success.

    Frames contain camera points, K, image shape and centered initial pose.
    For heldout frames fixed_scale is required; only SE(3) can change.
    """
    if not len(frames):
        raise ValueError('No frames')
    initials = [validate_pose(f['pose']) for f in frames]
    poses = [x.copy() for x in initials]
    # Input-derived coarse size, no global pose/scale search. Translation drops
    # out of quantile extents in the supplied object's orientation. Thin axes
    # below 20% of the largest model extent are excluded as poorly observed.
    ratios = []
    if fixed_scale is None:
        for f, pose in zip(frames, poses):
            _, model_camera = model_surface(points,pose,1.0,f['K'],f['shape'],f.get('mesh'))
            if len(model_camera) < cfg.minimum_correspondences:
                raise ValueError('INSUFFICIENT_COARSE_MODEL_SUPPORT')
            R = pose[:3,:3]
            observed_axes = f['points'] @ R
            model_axes = model_camera @ R
            extent_o = np.diff(np.quantile(observed_axes, [.05,.95], axis=0), axis=0)[0]
            extent_m = np.diff(np.quantile(model_axes, [.05,.95], axis=0), axis=0)[0]
            axes = extent_m > .2 * extent_m.max()
            ratios.extend((extent_o[axes] / extent_m[axes]).tolist())
        scale = float(np.median(ratios))
        if not cfg.scale_min <= scale <= cfg.scale_max:
            raise ValueError('COARSE_SIZE_OUTSIDE_PREDECLARED_BOUNDS')
    else:
        scale = float(fixed_scale)
    initial_scale = scale
    trace = []
    state = 'ITERATION_LIMIT'
    for iteration in range(cfg.iterations):
        rows, info = [], []
        for f, pose in zip(frames, poses):
            X, Y, weights, metrics = correspondences(points, f['points'], pose, scale,
                                                    f['K'], f['shape'], cfg,f.get('mesh'))
            rows.append((X, Y, weights))
            info.append(metrics)
        proposed_s, proposed = solve_corresponded_shared_scale(rows, fixed_scale)
        pose_delta = []
        for T, initial in zip(proposed, initials):
            delta_r = float(np.rad2deg(Rotation.from_matrix(T[:3, :3] @ initial[:3, :3].T).magnitude()))
            delta_t = float(np.linalg.norm(T[:3, 3] - initial[:3, 3]))
            pose_delta.append({'rotation_deg': delta_r, 'translation_m': delta_t})
        if (not cfg.scale_min <= proposed_s <= cfg.scale_max or
            any(x['rotation_deg'] > cfg.rotation_bound_deg or
                x['translation_m'] > cfg.translation_bound_m for x in pose_delta)):
            state = 'BOUND_REACHED'
            trace.append({'iteration': iteration, 'rejected_scale': proposed_s,
                          'pose_delta_from_initial': pose_delta, 'metrics_before_step': info})
            break
        change = max(abs(proposed_s-scale), max(float(np.linalg.norm(a-b)) for a,b in zip(proposed,poses)))
        scale, poses = proposed_s, proposed
        trace.append({'iteration': iteration, 'scale': scale, 'step_norm': change,
                      'pose_delta_from_initial': pose_delta, 'metrics_before_step': info})
        if change < 1e-6:
            state = 'CONVERGED'
            break
    return {'state': state, 'scale': scale, 'poses': poses, 'trace': trace,
            'initial_scale': initial_scale, 'coarse_extent_ratios': ratios,
            'fixed_scale': fixed_scale, 'GT_used': False}
