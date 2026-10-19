import cv2
import numpy as np

from pathb import calib as C


def test_refine_rows_removes_a_fitted_row_offset():
    """A right image whose rows sit 2 px low plus a slight vertical stretch is put back in line."""
    w, h = 320, 200
    I = C.Intrinsics(np.array([[200.0, 0, 160], [0, 200, 100], [0, 0, 1]]), np.zeros(14), (w, h))
    rect = C.rectify_rig("synthetic", "L", "R", I, I, np.eye(3), np.array([-0.075, 0.0, 0.0]))
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 1.5)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = (2.0 + 0.01 * (ys - h / 2)).astype(np.float32)  # y_left - y_right
    right = cv2.remap(scene, xs, ys + dy, cv2.INTER_LINEAR)  # right row r shows scene row r + dy
    fitted = C.refine_rows(rect, [2.0, 0.0, 0.01])  # what epipolar_check measures: y_left - y_right
    _, fixed = fitted.rectify(scene, right)
    inner = (slice(20, h - 20), slice(20, w - 20))
    error_fixed = np.abs(fixed[inner].astype(float) - scene[inner]).mean()
    error_shifted = np.abs(right[inner].astype(float) - scene[inner]).mean()
    assert error_fixed < 2.0 and error_fixed < 0.3 * error_shifted
