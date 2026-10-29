"""pathb/reference.py (V4_BUILD_SPEC.md S4b).

Synthetic tests always run. The data test (ground truth in as `inverted` -> playback AUC 1.0;
keyframe from ground-truth anchors + interpolation >= still) needs the public dataset and a few
CPU minutes of official scoring: it runs with PATHB_DATA_TESTS=1 (on the dev episodes ep41 and ep13).
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


def test_candidates_in_build_order_and_inverted_passed_through():
    T, ph = carried()
    assert list(R.build(T, ph, T[0], T[-1])) == ["keyframe", "still"]  # no S3c solution yet
    solved = (T.copy(), ["inverted"] * len(T))
    built = R.build(T, ph, T[0], T[-1], inverted=solved)
    assert list(built) == list(R.CANDIDATES) == ["inverted", "keyframe", "still"]
    assert built["inverted"][0] is solved[0] and R.provenance(built["inverted"][1])["inverted"] == len(T)


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
@pytest.mark.parametrize("number", [41, 13])
def test_ground_truth_reference_scores(number, tmp_path):
    """GT in as the inverted solution -> playback AUC 1.0; keyframe from GT anchors + interpolation >= still."""
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
        built[name] = R.build(T, ph, before, after, inverted=(T, ["inverted"] * len(T)))
    scores = {}
    for cand in R.CANDIDATES:
        stack = np.stack([np.concatenate([built[n][cand][0][:, :3, 3],
                                          Rotation.from_matrix(built[n][cand][0][:, :3, :3]).as_quat()], axis=-1)
                          for n in tracks], axis=1)
        scores[cand] = gt.score(e, stack, meshes, read)["AUC"]
    print(number, scores)
    assert scores["inverted"] > 0.99
    assert scores["keyframe"] >= scores["still"]


def test_s01_settings_and_world_fit():
    from pathb import oracle
    steps, overrides = oracle.settings()
    assert steps == 60_002_304 and "sim.njmax=1024" in overrides and "task.motion_speed=1.0" in overrides
    assert "sim.nconmax=512" in overrides and not any(o.startswith("sim.njmax=512") for o in overrides)
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    yaw = np.radians(30)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    dst = 0.6 * src @ Rz.T + [0.1, -0.2, 0.9]
    s, A = oracle.yaw_similarity(src, dst)
    assert np.isclose(s, 0.6) and np.allclose(A[:3, :3], Rz) and np.allclose(s * src @ A[:3, :3].T + A[:3, 3], dst)


def test_s01_world_fit_from_observed_points():
    """V4 S0.1: our world -> ground truth from what a head camera sees of a box that looks the same
    turned 180 degrees, carried 30 cm and turned 60 degrees: the motion fixes the heading."""
    import trimesh
    from pathb import oracle
    M = np.eye(4)  # ground-truth world -> ours (what the fit must undo)
    M[:3, :3], M[:3, 3] = oracle.Rz(np.radians(137.0)), [0.4, -0.2, 0.03]
    box = trimesh.creation.box((0.12, 0.08, 0.05))
    box.apply_translation([0, 0, 0.025])  # bottom on z = 0
    scan, faces = trimesh.sample.sample_surface(box, 20000, seed=0)
    normals = box.face_normals[faces]
    rng = np.random.default_rng(1)
    F, eye = 30, np.array([0.15, -0.45, 0.5])  # the head, ground-truth world
    G = np.tile(np.eye(4), (F, 1, 1))
    s = np.linspace(0, 1, F)
    G[:, :3, :3] = Rotation.from_rotvec(np.outer(np.radians(60) * s, [0, 0, 1])).as_matrix()
    G[:, :3, 3] = np.stack([0.3 * s, 0.1 * s ** 2, np.zeros(F)], axis=1)
    pts = []
    for t in range(F):  # the faces turned towards the camera, 2 mm noise
        world = scan @ G[t, :3, :3].T + G[t, :3, 3]
        seen = world[((normals @ G[t, :3, :3].T) * (eye - world)).sum(axis=1) > 0]
        p = seen[rng.choice(len(seen), 300, replace=False)] + rng.normal(0, 0.002, (300, 3))
        pts.append(p @ M[:3, :3].T + M[:3, 3])
    A, fit = oracle.world_fit([(np.stack(pts), G, np.asarray(scan))], iterations=100, starts=12)
    err = A @ M  # identity when A undoes M
    assert abs(np.degrees(oracle.yaw_of(err[:3, :3]))) < 1 and np.linalg.norm(err[:3, 3]) < 0.01, fit
    assert fit["trimmed_rms_cm"] < 0.5


def test_s01_still_bar_from_s02():
    from pathb import oracle
    assert round(oracle.still_score(41), 3) == 0.356 and round(oracle.still_score(13), 3) == 0.416


def test_s4b_inverted_holds_statics_and_interpolates_failed_grasps(monkeypatch):
    from pathb.stages import s4b_reference as B
    T, _ = carried()
    noisy = T.copy()
    noisy[:, 1, 3] += np.random.default_rng(0).normal(0, 0.002, len(T))
    segs = [{"kind": "static", "start": 0, "end": 29}, {"kind": "grasp", "start": 30, "end": 70},
            {"kind": "static", "start": 71, "end": 99}]
    v4 = {"track": noisy, "segments": segs, "failed": []}
    out, how = B.inverted(v4)
    assert np.allclose(out[:30], noisy[0]) and np.allclose(out[71:], noisy[71]) and how[40] == "inverted"
    assert np.allclose(out[30:71], noisy[30:71])
    out, how = B.inverted(v4 | {"failed": [(30, 70)]})
    assert how[30:71] == ["interpolated"] * 41 and np.allclose(out[29], noisy[0]) and abs(out[50, 0, 3] - 0.15) < 0.03
    before, after = B.rest_anchors(v4)
    assert np.allclose(before, noisy[0]) and np.allclose(after, noisy[71])
    monkeypatch.delenv("PATHB_MOVING", raising=False)
    assert B.moving_candidate(None, {"a": v4})["candidate"] == "inverted"
    assert B.moving_candidate(None, {"a": v4 | {"failed": [(30, 70)]}})["candidate"] == "keyframe"
    assert B.moving_candidate(None, {"a": None})["candidate"] == "keyframe"
    monkeypatch.setenv("PATHB_MOVING", "keyframe")
    assert B.moving_candidate(None, {"a": v4})["candidate"] == "keyframe"
