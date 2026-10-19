import numpy as np
import pytest

from pathb.geometry import fit_plane, fit_planes, plane_in, read_depth, write_depth
from pathb.stages.p3_scale_bridge import planes_in_cam_a, ray_surface, stereo_along, support_plane, table_distance


def test_depth_png_round_trip(tmp_path):
    depth = np.linspace(0.2, 3.0, 400, dtype=np.float32).reshape(20, 20)
    depth[0, 0] = 0.0  # no depth
    write_depth(tmp_path / "d.png", depth)
    back = read_depth(tmp_path / "d.png")
    assert back[0, 0] == 0.0
    np.testing.assert_allclose(back[1:, 1:], depth[1:, 1:], rtol=2e-4)


def plane_samples(normal, d, n=3000, noise=0.002, outliers=0.2, seed=0):
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    u = np.cross(normal, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    a, b = rng.uniform(-0.5, 0.5, (2, n))
    points = -d * normal + a[:, None] * u + b[:, None] * v + rng.normal(0, noise, (n, 1)) * normal
    k = int(n * outliers)
    points[:k] = rng.uniform(-0.6, 0.6, (k, 3)) + [0, 0, 0.8]
    return normal, points


def test_fit_plane_finds_the_table_and_its_distance():
    normal, points = plane_samples([0.0, -0.8, -0.6], 0.5)
    plane = fit_plane(points)
    assert abs(np.dot(plane["normal"], normal)) > 0.999
    assert plane["d"] == pytest.approx(0.5, abs=0.003)
    assert plane["inlier_fraction"] == pytest.approx(0.8, abs=0.05)


def test_plane_moves_with_the_frame():
    normal, points = plane_samples([0.1, -0.7, -0.7], 0.45, outliers=0)
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    T[:3, 3] = [0.03, -0.01, 0.002]
    n2, d2 = plane_in(T, normal, 0.45)
    moved = points @ T[:3, :3].T + T[:3, 3]
    assert np.abs(moved @ n2 + d2).max() < 0.01  # within the sample noise


def test_table_distance_is_the_cameras_height_above_the_objects_bottom():
    cube = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05)])
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.35, 1.0]  # 1 m ahead, centre 35 cm below the optical axis (camera y points down)
    assert table_distance(cube, T, up=np.array([0.0, -1.0, 0.0])) == pytest.approx(0.40)


def test_plane_distance_is_measured_from_ego_cam_a():
    planes = [{"normal": [0.0, -1.0, 0.0], "d": 0.40}]
    a_from_left = np.eye(4)
    a_from_left[:3, 3] = [-0.0375, -0.02, 0.0]  # ego_cam_a's centre is 2 cm lower (+y) in the left frame
    alignment = {"R1": np.eye(3).tolist(), "a_from_left": a_from_left.tolist()}
    [(normal, distance)] = planes_in_cam_a(planes, alignment)
    assert distance == pytest.approx(0.38)
    np.testing.assert_allclose(normal, [0, -1, 0])


def test_support_is_the_largest_plane_under_the_object_not_the_nearest():
    """ep13 frame 0: floor 1.136 m (43% of points), table 0.531 m (21%), dish rack base 0.505 m (7%)."""
    up = np.array([0.0, -1.0, 0.0])  # camera y points down
    point = np.array([0.0, 0.37, 0.6])  # the cup's stereo surface point, 37 cm below the camera
    floor, table, rack = ((up, d, s) for d, s in ((1.136, 0.43), (0.531, 0.21), (0.505, 0.07)))
    distance, tilt = support_plane([floor, table, rack], up, point)
    assert distance == pytest.approx(0.531) and tilt == pytest.approx(0.0)
    assert support_plane([floor], up, point) is None  # the floor is 77 cm below the cup


def test_fit_planes_finds_a_small_table_beside_a_large_floor():
    _, floor = plane_samples([0.0, -0.8, -0.6], 1.1, n=6000, outliers=0, seed=1)
    _, table = plane_samples([0.0, -0.8, -0.6], 0.5, n=900, outliers=0, seed=2)
    planes = fit_planes(np.vstack([floor, table]))
    assert [round(p["d"], 2) for p in planes[:2]] == [1.1, 0.5]


def test_rays_through_the_object_measure_its_surface_in_both_sources():
    sphere = np.random.default_rng(0).normal(size=(20000, 3))
    sphere = 0.1 * sphere / np.linalg.norm(sphere, axis=1, keepdims=True) + [0.0, 0.1, 0.6]
    direction = np.array([0.0, 0.1, 0.6]) / np.linalg.norm([0.0, 0.1, 0.6])
    assert ray_surface(sphere, direction) == pytest.approx(np.linalg.norm([0, 0.1, 0.6]) - 0.1, abs=0.003)
    K = np.array([[500.0, 0, 50], [0, 500, 50], [0, 0, 1]])
    depth = np.full((100, 200), 0.5)
    point = stereo_along(depth, K, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(point, [0, 0, 0.5])
