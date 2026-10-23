"""pathb/reference.py (V3_BUILD_SPEC.md S4b, task 4).

Synthetic tests always run. The task's data test (ground truth in -> playback AUC 1.0 for tracking;
keyframe from ground-truth anchors + interpolation >= still) needs the public dataset and a few
CPU minutes of official scoring: it runs with PATHB_DATA_TESTS=1 (e.g. on ep40 and ep26).
"""
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pathb import fuse, gt
from pathb import phases as P
from pathb import reference as R
from pathb.paths import TRACK3, episode_info


def poses(xyz, rotvec=None):
    T = np.tile(np.eye(4), (len(xyz), 1, 1))
    T[:, :3, 3] = xyz
    if rotvec is not None:
        T[:, :3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    return T


def carried(n=100, a=30, b=70):
    """Rest at x=0 until a, carried to x=0.3 by b, rest after."""
    x = np.zeros(n)
    x[a:b + 1] = np.linspace(0, 0.3, b - a + 1)
    x[b + 1:] = 0.3
    T = poses(np.stack([x, np.zeros(n), np.zeros(n)], axis=1), np.stack([np.zeros(n), np.zeros(n), x], axis=1))
    near = np.zeros(n, bool)
    near[a:b + 1] = True
    return T, P.phases(near, T)


def test_still_holds_the_rest_before_anchor():
    T, ph = carried()
    out, how = R.still(len(T), T[0])
    assert np.allclose(out, T[0]) and set(how) == {"held"}


def test_keyframe_holds_anchors_keeps_trusted_and_interpolates():
    T, ph = carried()
    trusted = np.zeros(len(T), bool)
    trusted[45:55] = True
    out, how = R.keyframe(T, ph, T[0], T[-1], trusted)
    assert np.allclose(out[:30], T[0]) and np.allclose(out[71:], T[-1])
    assert np.allclose(out[45:55], T[45:55]) and how[45:55] == ["trusted"] * 10
    assert how[30:45] == ["interpolated"] * 15 and how[55:71] == ["interpolated"] * 16
    # zero end velocity: the first interpolated step is smaller than a mid one
    x = out[:, 0, 3]
    assert abs(x[30] - x[29]) < abs(x[37] - x[36])
    # no trusted frame at all: one interpolation from anchor to anchor
    out, how = R.keyframe(T, ph, T[0], T[-1])
    assert how.count("interpolated") == 41 and np.isclose(out[50, 0, 3], 0.15, atol=0.02)


def test_tracking_replaces_rest_windows_with_anchors_and_blends():
    T, ph = carried()
    noisy = T.copy()
    noisy[:, 1, 3] += np.random.default_rng(0).normal(0, 0.01, len(T))
    anchor_b, anchor_a = fuse.rest_anchor(noisy[:25]), fuse.rest_anchor(noisy[76:])
    out, how = R.tracking(noisy, ph, anchor_b, anchor_a)
    rb, ra = ph["rest_before"], ph["rest_after"]
    assert np.allclose(out[:rb[1] + 1], anchor_b) and np.allclose(out[ra[0]:], anchor_a)
    assert how[rb[1] + 1:rb[1] + 1 + R.BLEND] == ["blended"] * R.BLEND
    assert np.allclose(out[40:60], noisy[40:60])  # the carry is the track


def test_never_touched_object_is_still_for_every_candidate():
    T = poses(np.zeros((40, 3)))
    ph = P.phases(np.zeros(40, bool), T)
    built = R.build(T, ph, T[0], None)
    assert all(np.allclose(t, T[0]) for t, _ in built.values())


DATA = os.environ.get("PATHB_DATA_TESTS") == "1" and (TRACK3 / "public/data").is_dir()


def gt_track(episode: dict, slot: int, read: list) -> np.ndarray:
    g = gt.ground_truth(episode, read)[:, slot]
    T = np.tile(np.eye(4), (len(g), 1, 1))
    T[:, :3, 3] = g[:, :3]
    T[:, :3, :3] = Rotation.from_quat(g[:, 3:]).as_matrix()
    return T


@pytest.mark.skipif(not DATA, reason="set PATHB_DATA_TESTS=1 (needs the public dataset; minutes of CPU)")
@pytest.mark.parametrize("number", [40, 26])
def test_ground_truth_reference_scores(number, tmp_path):
    """GT in -> tracking playback AUC 1.0; keyframe from GT anchors + interpolation >= still."""
    e = episode_info(number)
    read = []
    tracks = {o["name"]: gt_track(e, i, read) for i, o in enumerate(e["objects"])}
    meshes = {name: gt.scan(name) for name in tracks}
    built = {}
    for name, T in tracks.items():
        # the GT object's own motion stands in for the wrist: near = frames it moves (> 1 cm from rest)
        moved = (np.linalg.norm(T[:, :3, 3] - T[0, :3, 3], axis=1) > 0.01) & \
                (np.linalg.norm(T[:, :3, 3] - T[-1, :3, 3], axis=1) > 0.01)
        ph = P.phases(moved, T)
        rb, ra = ph["rest_before"], ph["rest_after"]
        before = fuse.rest_anchor(T[rb[0]:rb[1] + 1]) if rb else T[0]
        after = fuse.rest_anchor(T[ra[0]:ra[1] + 1]) if ra else None
        built[name] = R.build(T, ph, before, after)
    scores = {}
    for cand in R.CANDIDATES:
        stack = np.stack([np.concatenate([built[n][cand][0][:, :3, 3],
                                          Rotation.from_matrix(built[n][cand][0][:, :3, :3]).as_quat()], axis=-1)
                          for n in tracks], axis=1)
        scores[cand] = gt.score(e, stack, meshes, read)["AUC"]
    print(number, scores)
    assert scores["tracking"] > 0.99
    assert scores["keyframe"] >= scores["still"]


def test_s01_settings_and_world_fit():
    from pathb import oracle
    steps, overrides = oracle.still_settings()
    assert steps == 60_006_400 and "sim.njmax=512" in overrides and "task.motion_speed=1.0" in overrides
    assert not any(o.startswith("sim.njmax=1024") for o in overrides)
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    yaw = np.radians(30)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    dst = 0.6 * src @ Rz.T + [0.1, -0.2, 0.9]
    s, A = oracle.yaw_similarity(src, dst)
    assert np.isclose(s, 0.6) and np.allclose(A[:3, :3], Rz) and np.allclose(s * src @ A[:3, :3].T + A[:3, 3], dst)
