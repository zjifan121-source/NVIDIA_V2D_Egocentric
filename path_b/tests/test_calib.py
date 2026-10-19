import cv2
import numpy as np
import pytest

from pathb import calib as C
from pathb.paths import CALIBRATION


def camera_dict(K, D, size):
    return {"cameras": {"cam": {"intrinsic_matrix": K.tolist(), "distortion_coefficients": D.tolist(),
                                "resolution": list(size)}}}


def test_turning_intrinsics_twice_restores_them():
    K = np.array([[1150.0, 0, 994.8], [0, 1149.7, 767.5], [0, 0, 1]])
    D = np.arange(1, 15, dtype=float) / 100
    once = C.intrinsics(camera_dict(K, D, (2028, 1520)), "cam", rotate=True)
    assert once.K[0, 2] == pytest.approx(2027 - 994.8) and once.D[2] == pytest.approx(-0.03)
    twice = C.intrinsics(camera_dict(once.K, once.D, (2028, 1520)), "cam", rotate=True)
    np.testing.assert_allclose(twice.K, K)
    np.testing.assert_allclose(twice.D, D)


def synthetic_rig():
    K = np.array([[570.0, 0, 640], [0, 570, 400], [0, 0, 1]])
    D = np.zeros(14)
    D[[0, 1, 2, 3]] = [-0.05, 0.01, 0.001, -0.0005]
    R = cv2.Rodrigues(np.array([0.01, -0.02, 0.005]))[0]
    T = np.array([-0.075, 0.001, 0.0005])  # right camera 75 mm to the left camera's right
    I = C.Intrinsics(K, D, (1280, 800))
    return C.rectify_rig("synthetic", "L", "R", I, I, R, T), K, D, R, T


def test_rectification_puts_matches_on_one_row_with_positive_disparity():
    rect, K, D, R, T = synthetic_rig()
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-0.4, 0.4, 200), rng.uniform(-0.3, 0.3, 200), rng.uniform(0.4, 2.0, 200)])
    left = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K, D)[0]
    right = cv2.projectPoints(X, cv2.Rodrigues(R)[0], T, K, D)[0]
    rl = cv2.undistortPoints(left, K, D, R=rect.R1, P=rect.P1).reshape(-1, 2)
    rr = cv2.undistortPoints(right, K, D, R=rect.R2, P=rect.P2).reshape(-1, 2)
    assert np.abs(rl[:, 1] - rr[:, 1]).max() < 0.05
    assert (rl[:, 0] - rr[:, 0] > 0).all()
    assert rect.baseline == pytest.approx(np.linalg.norm(T), rel=1e-6)
    # depth from disparity is the point's depth in the rectified left camera
    z = rect.K[0, 0] * rect.baseline / (rl[:, 0] - rr[:, 0])
    np.testing.assert_allclose(z, (X @ rect.R1.T)[:, 2], rtol=1e-3)


@pytest.mark.skipif(not CALIBRATION.is_file(), reason="Track 3 dataset not downloaded")
def test_every_reading_of_the_real_calibration_builds():
    calibration = C.load()
    for reading in C.READINGS:
        rect = C.rectification(calibration, reading)
        assert abs(rect.baseline) == pytest.approx(0.075, abs=0.001)
        assert rect.K[0, 0] > 0
        candidates = C.cam_a_candidates(calibration, rect)
        assert len(candidates) == 8
        for T in candidates.values():
            np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6)
            assert 0.03 < np.linalg.norm(T[:3, 3]) < 0.12  # ego_cam_a sits on the stereo baseline
