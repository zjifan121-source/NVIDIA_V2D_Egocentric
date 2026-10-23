"""pathb/symmetry.py and pathb/fuse.py (V3_BUILD_SPEC.md task 3): a flipped synthetic track is
recovered; anchors pin a noisy static segment to < 1 mm."""
import numpy as np
from scipy.spatial.transform import Rotation

from pathb import fuse, symmetry


def cylinder(radius=0.08, height=0.06, axis=2, n=400):
    rng = np.random.default_rng(1)
    a, h = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, height, n)
    v = np.stack([radius * np.cos(a), radius * np.sin(a), h], axis=1)
    return np.roll(v, axis - 2, axis=1)


def pose(rotvec, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


def test_groups_follow_the_roster():
    v = cylinder(axis=1)
    assert symmetry.group("white_pot").shape == (1, 4, 4)
    cup = symmetry.group("blue_cup", v)
    assert cup.shape == (36, 4, 4) and np.allclose(cup[0], np.eye(4))
    lid = symmetry.group("white_pot_lid", v)
    assert len(lid) == 2 and np.isclose(np.degrees(symmetry.angle(np.eye(3), lid[1][:3, :3])), 180)
    a, c = symmetry.axis(v)
    assert np.allclose(np.abs(a), [0, 1, 0], atol=0.05)
    # a symmetry maps the mesh onto itself (about its own axis through its centroid)
    moved = v @ lid[1][:3, :3].T + lid[1][:3, 3]
    assert np.allclose(np.sort(np.linalg.norm(moved - c, axis=1)), np.sort(np.linalg.norm(v - c, axis=1)), atol=1e-9)


def test_flipped_track_is_recovered():
    """A lid turning slowly; on 12 frames the tracker reports the half-turned equivalent."""
    v = cylinder()
    G = symmetry.group("white_pot_lid", v)
    n = 80
    truth = np.stack([pose([0.0, 0.0, 0.01 * k], [0.002 * k, 0.0, 0.3]) for k in range(n)])
    measured = truth.copy()
    flipped = list(range(20, 26)) + list(range(50, 56))
    measured[flipped] = measured[flipped] @ G[1]
    out, info = fuse.smooth(measured, G=G)
    err = [np.degrees(symmetry.angle(a[:3, :3], b[:3, :3])) for a, b in zip(out, truth)]
    assert max(err) < 1.0 and info["symmetric_switches"] == len(flipped)
    assert np.abs(out[:, :3, 3] - truth[:, :3, 3]).max() < 0.005
    # without the group the flips drag the track far off
    bad, _ = fuse.smooth(measured)
    assert max(np.degrees(symmetry.angle(a[:3, :3], b[:3, :3])) for a, b in zip(bad, truth)) > 30


def test_anchors_pin_a_noisy_static_segment():
    rng = np.random.default_rng(0)
    n, rest = 120, (0, 39)
    truth = pose([0.1, -0.2, 0.3], [0.1, 0.2, 0.0])
    measured = np.stack([pose(Rotation.from_matrix(truth[:3, :3]).as_rotvec() + rng.normal(0, 0.02, 3),
                              truth[:3, 3] + rng.normal(0, 0.01, 3)) for _ in range(n)])
    measured[40:, 0, 3] += np.linspace(0, 0.3, n - 40)  # carried away after the rest
    anchor = fuse.rest_anchor(measured[rest[0]:rest[1] + 1])
    assert np.linalg.norm(anchor[:3, 3] - truth[:3, 3]) < 0.01  # the median of 40 frames (1 cm noise per axis)
    out, info = fuse.smooth(measured, anchors=[(list(rest), anchor)])
    assert info["anchor_frames"] == 40
    assert np.linalg.norm(out[:40, :3, 3] - anchor[:3, 3], axis=1).max() < 0.001
    assert max(np.degrees(symmetry.angle(o[:3, :3], anchor[:3, :3])) for o in out[:40]) < 0.5


def test_noise_follows_the_camera_and_recovery_flags():
    """Depth noise lies along the optical axis: a camera looking down world -x trusts world y, z."""
    n = 60
    truth = np.stack([pose([0, 0, 0], [0, 0, 0])] * n)
    measured = truth.copy()
    measured[30, :3, 3] = [0.05, 0.05, 0.0]  # one frame off along world x and y
    look_x = Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix()  # camera z -> world x
    out, _ = fuse.smooth(measured, cam_rot=np.repeat(look_x[None], n, axis=0))
    assert abs(out[30, 0, 3]) < abs(out[30, 1, 3])  # x is depth here: pulled less
    flagged, _ = fuse.smooth(measured, cam_rot=np.repeat(look_x[None], n, axis=0),
                             recovered=np.arange(n) == 30)
    assert abs(flagged[30, 1, 3]) < abs(out[30, 1, 3])  # a recovered frame is trusted less
