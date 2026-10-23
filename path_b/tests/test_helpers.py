"""Small shared helpers: Transform3d files, the stereo frame offset, the 2-D rest window."""
import numpy as np
from scipy.spatial.transform import Rotation

from pathb.calib import stereo_frame
from pathb.geometry import read_transform, read_transforms, write_transform
from pathb.phases import rest_window_2d


def test_transform_files_round_trip(tmp_path):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 1.1]).as_matrix()
    T[:3, 3] = [0.1, -0.4, 1.2]
    write_transform(tmp_path / "000001.json", T, [0.5, 0.5, 0.5])
    back, scale = read_transform(tmp_path / "000001.json")
    assert np.allclose(back, T) and np.allclose(scale, 0.5)
    all_T, have = read_transforms(tmp_path, 3)
    assert have.tolist() == [False, True, False] and np.allclose(all_T[1], T)


def test_stereo_frame_applies_the_offset_once():
    assert stereo_frame(10, {"offset": -1}) == 9
    assert stereo_frame(0, {"offset": -1}) == 0 and stereo_frame(99, {"offset": 2}, frames=100) == 99


def test_rest_window_2d_ends_before_a_hand_touches():
    h, w, n = 100, 100, 40
    obj = [np.zeros((h, w), bool) for _ in range(n)]
    hand = [np.zeros((h, w), bool) for _ in range(n)]
    for t in range(n):
        obj[t][40:60, 40:60] = True
        x = max(0, 95 - 3 * t)  # the hand comes in from the right
        hand[t][45:55, x:x + 5] = True
    rest = rest_window_2d([obj], [hand], cap=30, dilate_px=5)
    first_touch = next(t for t in range(n) if max(0, 95 - 3 * t) - 5 <= 59)  # dilated hand reaches column 59
    assert rest == [0, first_touch - 1]
    assert rest_window_2d([obj], [[np.zeros((h, w), bool)] * n], cap=30) == [0, 29]
