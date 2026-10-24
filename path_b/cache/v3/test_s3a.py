"""s3a_mesh's CPU parts: view selection and the metric size fit (SAM 3D itself runs on the GPU)."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from pathb.stages import s3a_mesh as S


def test_views_take_both_rest_windows_then_spread_directions():
    cands = {k: 0.03 + 0.001 * k for k in range(0, 100, 5)}
    dirs = {k: np.array([np.cos(k / 30), np.sin(k / 30), 0.0]) for k in cands}
    views = S.select(cands, dirs, before=[0, 12], after=[90, 99])
    assert len(views) == S.VIEWS
    assert 10 in views and 95 in views  # the largest mask in each rest window
    assert len({round(np.degrees(np.arctan2(dirs[k][1], dirs[k][0]))) for k in views}) == S.VIEWS


def test_size_fit_recovers_a_scale_error():
    F = S.fit_shape()
    box = trimesh.creation.box(extents=[0.12, 0.08, 0.06])
    pts = np.asarray(box.sample(4000, seed=0))
    true_scale = 1.3
    K = (600.0, 600.0, 320.0, 240.0)
    shape = (480 // F.S, 640 // F.S)
    views = []
    for yaw in (0.0, 0.5, 1.0):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xy", [0.4, yaw]).as_matrix()
        T[:3, 3] = [0.0, 0.0, 0.6]
        sil, zb = F.render((pts * true_scale) @ T[:3, :3].T + T[:3, 3], K, shape)
        views.append((T, sil, np.where(np.isfinite(zb), zb, 0.0)))
    sized = S.size_fit(F, pts, views, K)
    assert abs(sized["uniform"] * np.mean(sized["axes"]) - true_scale) < 0.06
    assert sized["cost_after"] < sized["cost_before"]
