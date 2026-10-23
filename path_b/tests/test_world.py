"""The v3 world (pathb/world.py) and P3's support plane (v3)."""
import numpy as np
from scipy.spatial.transform import Rotation

from pathb import world
from pathb.stages.p3_scale_bridge import table_in_camera


def test_world_puts_the_table_at_z0_and_looks_along_y():
    # P1c's world: an arbitrary rotation and offset of a scene whose table is z = 0.3, up = +z
    A = np.eye(4)
    A[:3, :3] = Rotation.from_rotvec([0.4, -0.7, 1.2]).as_matrix()
    A[:3, 3] = [2.0, -1.0, 0.5]
    look = Rotation.from_euler("x", -120, degrees=True).as_matrix()  # camera looking down and forward (+y)
    c2w_true = np.tile(np.eye(4), (10, 1, 1))
    c2w_true[:, :3, :3] = look
    c2w_true[:, :3, 3] = [[0.01 * k, 0.0, 0.3 + 0.5] for k in range(10)]  # 50 cm above the table
    c2w = A[None] @ c2w_true
    normal = A[:3, :3] @ [0, 0, 1.0]
    d = -normal @ (A[:3, :3] @ [0, 0, 0.3] + A[:3, 3])
    W = world.from_table(normal, d, c2w, list(range(10)))
    v3 = W[None] @ c2w
    assert np.allclose(v3[:, 2, 3], 0.5)  # camera height above the table
    table_point = W @ np.append(A[:3, :3] @ [0.4, -0.2, 0.3] + A[:3, 3], 1.0)
    assert np.isclose(table_point[2], 0.0)
    forward = v3[0, :3, 2]
    assert forward[1] > 0 and abs(forward[0]) < 1e-9 and forward[2] < 0  # +y forward, looking down
    assert np.isclose(v3[0, 0, 3], 0.0) and np.isclose(v3[0, 1, 3], 0.0)  # origin under the first camera


def test_table_is_the_largest_plane_under_the_objects():
    up = np.array([0.0, -0.8, -0.6])  # camera frame: the table normal points back at the camera
    table = (up, 0.6, 0.4)  # 60 cm from the camera
    rack_base = (up, 0.45, 0.6)  # a larger plane, but above the object
    wall = (np.array([1.0, 0, 0]), 0.5, 0.7)
    point = -up * 0.5  # an object surface point 10 cm above the table
    n, d, share = table_in_camera([table, rack_base, wall], [point])
    assert np.isclose(d, 0.6) and share == 0.4
    flipped = (-up, -0.6, 0.4)  # the same plane with its normal turned away: turned back
    n, d, _ = table_in_camera([flipped], [point])
    assert np.allclose(n, up) and np.isclose(d, 0.6)
