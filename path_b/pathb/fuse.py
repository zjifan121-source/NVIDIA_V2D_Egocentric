"""ESKF + RTS smoothing of an object's world poses with rest anchors (V3_BUILD_SPEC.md S3c).

The baseline's smoother (v2d_foundation_pose/lib/run_ekf_smoothing.py: _eskf_rts) re-implemented on
object_to_world poses: random-walk process model, translation by a linear Kalman filter, rotation
in error state (so(3) perturbation), measurement noise scaled by 1/IoU with an IoU floor, then the
Rauch-Tung-Striebel backward pass. Parameters are the ones the pipeline passes it
(run_ego_wilor.py). On world poses:

- the measurement's depth noise (MEAS_Z) lies along the camera's optical axis, so its covariance is
  rotated into the world with camera_to_world; with no camera rotation it is isotropic (MEAS_XY);
- anchor frames (the rest windows) measure the anchor pose with ANCHOR_T / ANCHOR_R instead of the
  track's pose;
- frames the tracker marked recovered get RECOVERED x the noise;
- an optional second stream (Point2Pose) is a further update with SECOND_T / SECOND_R;
- with a symmetry group (pathb/symmetry.py), each measurement is first replaced by its symmetric
  equivalent closest to the prediction, so a tracker that jumps between equivalent poses does not
  pull the track.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .symmetry import closest

PROCESS_T = 0.01  # default: m per frame (the pipeline's process_noise_xy = _z)
PROCESS_R = 0.02  # default: rad per frame
MEAS_XY = 0.01  # default: m, across the optical axis
MEAS_Z = 0.04  # default: m, along it
MEAS_R = 0.02  # default: rad
MIN_IOU = 0.1  # default
ANCHOR_T, ANCHOR_R = 0.002, 0.005  # default: m, rad
RECOVERED = 3.0  # default: noise factor on recovered frames
SECOND_T, SECOND_R = 0.02, 0.05  # default: Point2Pose stream


def gain(P_pred: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.linalg.solve(S, P_pred.T).T


def rest_anchor(T: np.ndarray, G: np.ndarray | None = None) -> np.ndarray:
    """One pose for a resting object: median translation and chordal-mean rotation of T (n, 4, 4),
    each pose first moved to its symmetric equivalent closest to the first."""
    T = np.asarray(T, dtype=np.float64)
    if G is not None and len(G) > 1:
        T = np.stack([T[0]] + [closest(T[0], t, G)[0] for t in T[1:]])
    out = np.eye(4)
    out[:3, 3] = np.median(T[:, :3, 3], axis=0)
    out[:3, :3] = Rotation.from_matrix(T[:, :3, :3]).mean().as_matrix()
    return out


def smooth(T: np.ndarray, valid: np.ndarray | None = None, iou: np.ndarray | None = None,
           recovered: np.ndarray | None = None, cam_rot: np.ndarray | None = None,
           anchors: list[tuple[list[int], np.ndarray]] = (), second: np.ndarray | None = None,
           second_valid: np.ndarray | None = None, G: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Smoothed (N, 4, 4) object_to_world and counts. T: measured poses (N, 4, 4); valid, iou,
    recovered: per frame; cam_rot: (N, 3, 3) camera_to_world rotations; anchors: ([first, last]
    frames, 4x4 pose) per rest window; second, second_valid: the optional second stream."""
    T = np.asarray(T, dtype=np.float64)
    n = len(T)
    valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    iou = np.ones(n) if iou is None else np.asarray(iou, dtype=np.float64)
    recovered = np.zeros(n, bool) if recovered is None else np.asarray(recovered, bool)
    anchor = [None] * n
    for (a, b), pose in anchors:
        for t in range(max(a, 0), min(b, n - 1) + 1):
            anchor[t] = np.asarray(pose, dtype=np.float64)
    use_second = second is not None and second_valid is not None
    G = np.eye(4)[None] if G is None else G

    def measurements(k: int, r_ref: Rotation) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(pose, translation covariance, rotation covariance) per measurement of frame k."""
        ref = np.eye(4)
        ref[:3, :3] = r_ref.as_matrix()
        if anchor[k] is not None:
            return [(closest(ref, anchor[k], G)[0], ANCHOR_T ** 2 * np.eye(3), ANCHOR_R ** 2 * np.eye(3))]
        out = []
        if valid[k]:
            s = (RECOVERED if recovered[k] else 1.0) / max(float(iou[k]), MIN_IOU)
            D = np.diag([MEAS_XY ** 2, MEAS_XY ** 2, MEAS_Z ** 2]) * s ** 2
            Rc = np.eye(3) if cam_rot is None else np.asarray(cam_rot[k], dtype=np.float64)
            out.append((closest(ref, T[k], G)[0], Rc @ D @ Rc.T, (MEAS_R * s) ** 2 * np.eye(3)))
        if use_second and second_valid[k]:
            out.append((closest(ref, np.asarray(second[k], dtype=np.float64), G)[0],
                        SECOND_T ** 2 * np.eye(3), SECOND_R ** 2 * np.eye(3)))
        return out

    observed = [k for k in range(n) if anchor[k] is not None or valid[k] or (use_second and second_valid[k])]
    if not observed:
        raise ValueError("no frame has a measurement")
    first = observed[0]
    Q_t, Q_r, I3 = PROCESS_T ** 2 * np.eye(3), PROCESS_R ** 2 * np.eye(3), np.eye(3)
    t_f, P_t_f, P_t_p = np.zeros((n, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3, 3))
    r_f, P_r_f, P_r_p = [None] * n, np.zeros((n, 3, 3)), np.zeros((n, 3, 3))
    flips = 0
    for k in range(first, n):
        if k == first:  # start from the first measurement with its own uncertainty
            pose, Rt, Rr = measurements(k, Rotation.from_matrix(
                (anchor[k] if anchor[k] is not None else T[k] if valid[k] else second[k])[:3, :3]))[0]
            t, P_t, r, P_r = pose[:3, 3].copy(), Rt, Rotation.from_matrix(pose[:3, :3]), Rr
            P_t_p[k], P_r_p[k] = P_t, P_r
            rest = measurements(k, r)[1:]
        else:
            t, P_t = t_f[k - 1], P_t_f[k - 1] + Q_t
            r, P_r = r_f[k - 1], P_r_f[k - 1] + Q_r
            P_t_p[k], P_r_p[k] = P_t, P_r
            rest = measurements(k, r)
        for pose, Rt, Rr in rest:
            K = gain(P_t, P_t + Rt)
            t = t + K @ (pose[:3, 3] - t)
            P_t = (I3 - K) @ P_t
            K = gain(P_r, P_r + Rr)
            r = Rotation.from_rotvec(K @ (Rotation.from_matrix(pose[:3, :3]) * r.inv()).as_rotvec()) * r
            P_r = (I3 - K) @ P_r
        if valid[k] and anchor[k] is None:
            ref = np.eye(4)
            ref[:3, :3] = (r_f[k - 1] if k > first else r).as_matrix()
            flips += closest(ref, T[k], G)[1] != 0
        t_f[k], P_t_f[k], r_f[k], P_r_f[k] = t, P_t, r, P_r

    t_s, r_s, P_t_s, P_r_s = t_f.copy(), list(r_f), P_t_f.copy(), P_r_f.copy()
    for k in range(n - 2, first - 1, -1):
        Gt = gain(P_t_f[k], P_t_p[k + 1])
        t_s[k] = t_f[k] + Gt @ (t_s[k + 1] - t_f[k])  # random walk: the prediction of k+1 is t_f[k]
        P_t_s[k] = P_t_f[k] + Gt @ (P_t_s[k + 1] - P_t_p[k + 1]) @ Gt.T
        Gr = gain(P_r_f[k], P_r_p[k + 1])
        r_s[k] = Rotation.from_rotvec(Gr @ (r_s[k + 1] * r_f[k].inv()).as_rotvec()) * r_f[k]
        P_r_s[k] = P_r_f[k] + Gr @ (P_r_s[k + 1] - P_r_p[k + 1]) @ Gr.T
    out = np.tile(np.eye(4), (n, 1, 1))
    for k in range(first, n):
        out[k, :3, :3], out[k, :3, 3] = r_s[k].as_matrix(), t_s[k]
    out[:first] = out[first]  # before the first measurement: held
    return out, {"first_measured_frame": first, "anchor_frames": sum(a is not None for a in anchor),
                 "symmetric_switches": int(flips),
                 "position_std_m_median": float(np.median(np.sqrt(np.trace(P_t_s[first:], axis1=1, axis2=2) / 3)))}
