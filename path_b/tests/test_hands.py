"""pathb/hands.py: P1d's v4 world exports and the hand-gap rule (V4_BUILD_SPEC.md S1 P1d, task 2).

Synthetic tests always run. The fixture test (a real bundle's hands -> fingertips, through the
baseline loader in the stages 2-3 venv) runs with PATHB_DATA_TESTS=1 on the Mac's ep41 P4 bundle of
run 4c2eac31.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pathb import hands as H
from pathb.paths import OUTPUTS


def moving_wrist(n=60):
    """A wrist on a smooth curve, turning about z: (n, 4, 4)."""
    t = np.arange(n, dtype=np.float64)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, 3] = np.stack([0.01 * t, 0.05 * np.sin(t / 8), 0.3 + 0.002 * t], axis=1)
    T[:, :3, :3] = Rotation.from_rotvec(np.outer(t * 0.02, [0, 0, 1])).as_matrix()
    return T


def test_gaps_are_filled_or_long():
    n = 80
    measured = np.ones(n, bool)
    measured[:3] = False  # leading: nothing before it
    measured[10:15] = False  # 5 frames, filled by tracks_interpolate
    measured[30:50] = False  # 20 frames: longer than MAX_GAP
    measured[60:66] = False  # 6 frames, but two had no mask: not filled
    valid = measured.copy()
    valid[10:15] = True
    valid[60:62] = valid[64:66] = True
    got = [(g["start"], g["end"], g["kind"]) for g in H.gaps(measured, valid)]
    assert got == [(0, 2, "long"), (10, 14, "filled"), (30, 49, "long"), (60, 65, "long")]


def test_hermite_follows_a_moving_wrist_through_a_gap():
    T = moving_wrist()
    measured = np.ones(len(T), bool)
    measured[20:32] = False  # 12 frames
    linear = T.copy()  # what a linear fill puts there
    s = (np.arange(20, 32) - 19) / 13
    linear[20:32, :3, 3] = T[19, :3, 3] + s[:, None] * (T[32, :3, 3] - T[19, :3, 3])
    g = H.gaps(measured, np.ones(len(T), bool))
    out = H.hermite_fill(linear, measured, g)
    err_h = np.linalg.norm(out[20:32, :3, 3] - T[20:32, :3, 3], axis=1).max()
    err_l = np.linalg.norm(linear[20:32, :3, 3] - T[20:32, :3, 3], axis=1).max()
    assert err_h < 0.003 and err_h < err_l / 3
    assert np.allclose(out[measured], T[measured])  # measured frames untouched
    angle = Rotation.from_matrix(out[20:32, :3, :3]).as_rotvec()[:, 2]
    assert np.allclose(angle, np.arange(20, 32) * 0.02, atol=1e-9)  # slerp of a constant turn is exact


def mano_dict(n, offsets, W):
    """mano_world.py's layout for a right hand whose joints sit at fixed wrist-frame offsets; left absent."""
    J = np.einsum("nij,kj->nki", W[:, :3, :3], offsets) + W[:, None, :3, 3]
    out = {"right_joints": J, "right_vertices": np.repeat(J[:, :1], 778, axis=1),
           "right_global_orient": Rotation.from_matrix(W[:, :3, :3]).as_rotvec(),
           "right_valid": np.ones(n, bool), "right_faces": np.zeros((1552, 3), int)}
    out.update({f"left_{k}": np.zeros_like(v) for k, v in
                (("joints", J), ("vertices", out["right_vertices"]), ("global_orient", out["right_global_orient"]))})
    out.update(left_valid=np.zeros(n, bool), left_faces=np.zeros((1552, 3), int))
    return out


def test_build_moves_tips_rigidly_and_flags_gaps():
    n = 60
    truth = moving_wrist(n)
    offsets = np.random.default_rng(0).normal(0, 0.04, (21, 3))
    offsets[0] = 0  # joint 0 is the wrist
    seen = truth.copy()
    seen[20:30, :3, 3] += [0.0, 0.02, 0.0]  # tracks_interpolate's frames off the curve by 2 cm
    measured, filled = np.ones(n, bool), np.zeros(n, bool)
    measured[20:30], filled[20:30] = False, True
    measured[45:], filled[45:] = False, False  # the hand leaves: a trailing gap
    r = H.build(mano_dict(n, offsets, seen), {"right": (measured, filled), "left": (np.zeros(n, bool),) * 2})
    R = H.HANDS.index("R")
    assert r["valid"][:45, R].all() and not r["valid"][45:, R].any() and not r["valid"][:, 0].any()
    assert r["filled"][20:30, R].all() and r["filled"][:, R].sum() == 10
    assert np.isnan(r["tips"][45:, R]).all() and np.isnan(r["wrist"][:, 0]).all()
    assert np.linalg.norm(r["wrist"][20:30, R, :3, 3] - truth[20:30, :3, 3], axis=1).max() < 0.002
    tips_rel = np.einsum("nji,nkj->nki", r["wrist"][:45, R, :3, :3], r["tips"][:45, R] - r["wrist"][:45, R, None, :3, 3])
    assert np.allclose(tips_rel, offsets[list(H.TIPS)][None], atol=1e-9)  # the tips moved with the wrist
    assert 1.5 < r["hermite_shift_cm"]["R"] < 2.5
    assert [g["kind"] for g in r["gaps"]["R"]] == ["filled", "long"]


def test_coverage_inside_grasps():
    valid = np.ones((100, 2), bool)
    valid[40:50, 1] = False  # the right wrist is missing for 10 frames
    segs = [{"kind": "static", "start": 0, "end": 19, "hand": None},
            {"kind": "grasp", "start": 20, "end": 59, "hand": "R"},
            {"kind": "grasp", "start": 70, "end": 79, "hand": "L"}]
    c = H.coverage(valid, segs)
    assert c["overall"] == {"L": 1.0, "R": 0.9}
    assert [g["share"] for g in c["grasps"]] == [0.75, 1.0]
    assert c["coverage_carry"] == pytest.approx(40 / 50)
    assert H.coverage(valid, segs[:1])["coverage_carry"] is None


RETARGET = Path(os.environ.get("RETARGET_VENV", Path.home() / "V2D/venvs/retarget"))
MANO = Path(os.environ.get("MANO_DIR", Path.home() / "V2D/mano_v1_2"))
EP41 = OUTPUTS / "4c2eac31/episode_000041/p4_export/episode_000041/white_pot/result_slam_gravity_aligned"
DATA = (os.environ.get("PATHB_DATA_TESTS") == "1" and (RETARGET / "bin/python").exists()
        and (MANO / "models/MANO_RIGHT.pkl").exists() and (EP41 / "result.npz").is_file())


@pytest.mark.skipif(not DATA, reason="set PATHB_DATA_TESTS=1 (needs the stages 2-3 venv, MANO and run 4c2eac31's ep41)")
def test_fixture_hands_give_fingertips(tmp_path, monkeypatch):
    monkeypatch.setenv("RETARGET_VENV", str(RETARGET))
    monkeypatch.setenv("MANO_DIR", str(MANO))
    n = len(np.load(EP41 / "result.npz")["camera_to_world_transform"])
    dirs = []
    for side in H.SIDES:  # P1d-style records: measured everywhere, a filled gap and a missing stretch
        d = tmp_path / side
        d.mkdir()
        for k in range(n):
            if not 200 <= k < 230:
                (d / f"{k:06d}.json").write_text(json.dumps({"interpolated": 100 <= k < 108}))
        dirs.append(d)
    summary = H.export(EP41, tuple(dirs), tmp_path / "hands", tmp_path / "log.txt")
    tips = np.load(tmp_path / "hands/fingertips_world.npz")
    wrist = np.load(tmp_path / "hands/wrist_world.npz")["wrist"]
    assert tips["tips"].shape == (n, 2, 5, 3) and wrist.shape == (n, 2, 4, 4)
    v = tips["valid"]
    assert not v[200:230].any() and v[:200].all() and v[230:].all()
    assert tips["filled"][100:108].all() and tips["filled"].sum() == 16
    d = np.linalg.norm(tips["tips"][v] - wrist[v][:, None, :3, 3], axis=-1)
    assert 0.05 < np.median(d) < 0.12  # tips 5-12 cm from the wrist (hand scale ~0.8-1.1)
    verts = np.load(tmp_path / "hands/hand_verts_world.npz")["verts"]
    assert verts.shape == (n, 2, 778, 3) and np.isfinite(verts[v]).all()
    gaps = json.loads((tmp_path / "hands/hand_gaps.json").read_text())["gaps"]
    assert [g["kind"] for g in gaps["R"]] == ["filled", "long"]
    assert summary["hermite_shift_cm"]["R"] < 5
