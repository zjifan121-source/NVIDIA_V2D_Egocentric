"""A2.4 (mesh-free point tracking) and A2.5 (variant decision) tests, cut from tests/test_a2.py in v4 task 1.
History only: the modules they import (s3b2_track_points, a2) are in this folder, not on the import path."""
import numpy as np

# ---------------------------------------------------------------- A2.4 rigid motion from point tracks
from pathb import symmetry  # noqa: E402
from pathb.stages import s3b2_track_points as A24  # noqa: E402


def test_point_tracks_give_the_rigid_motion():
    rng = np.random.default_rng(3)
    n, q = 30, 5
    P = rng.normal(0, 0.05, (200, 3)) + [0.3, 0.1, 0.05]
    truth = np.stack([pose(t + 10) for t in range(n)])  # moving between rest windows
    move = truth @ np.linalg.inv(truth[q])  # world motion since frame q
    coords = np.einsum("tij,nj->tni", move[:, :3, :3], P) + move[:, None, :3, 3]
    coords += rng.normal(0, 0.001, coords.shape)
    bad = rng.random(200) < 0.2  # a fifth of the tracks drift off the object
    coords[:, bad] += rng.normal(0, 0.05, (n, bad.sum(), 3))
    visible = rng.random((n, 200)) > 0.3
    visible[q] = True
    M, rows = A24.motions(coords, visible, q, symmetry.group("mini_dust_pan"))
    def on_object(T):  # the object's own points (the motion's origin is 0.3 m away: a lever arm)
        return np.einsum("tij,nj->tni", T[:, :3, :3], P) + T[:, None, :3, 3]
    err = np.linalg.norm(on_object(M) - on_object(move), axis=-1).mean(axis=1)
    assert np.isfinite(M[:, 0, 0]).all() and err.max() < 0.002
    assert max(r["rmse_m"] for r in rows) < 0.003 and min(r["iou"] for r in rows if r["frame"] != q) > 0.6


# ---------------------------------------------------------------- A2.5 decision rule
from pathb import a2 as A25  # noqa: E402


def test_a2_decision_rule():
    row = lambda auc, cdo: {"AUC": auc, "mean CD-O cm": cdo}  # noqa: E731
    # better CD-O on both episodes, AUC split: A2 has the higher mean rank
    t = {40: {"phase_a": row(0.54, 0.6), "a2_mesh_fp": row(0.50, 0.3)},
         26: {"phase_a": row(0.30, 1.0), "a2_mesh_fp": row(0.35, 0.4)}}
    assert A25.decide(t)["choice"] == "a2_mesh_fp"
    # equal ranks: the deterministic (A2) variant
    t = {40: {"phase_a": row(0.54, 0.3), "a2_mesh_fp": row(0.50, 0.4)},
         26: {"phase_a": row(0.30, 1.0), "a2_mesh_fp": row(0.35, 0.4)}}
    d = A25.decide(t)
    assert d["tied"] == ["a2_mesh_fp", "phase_a"] and d["choice"] == "a2_mesh_fp"
    # A2 worse on both AUC and CD-O everywhere: phase A stays
    t = {40: {"phase_a": row(0.54, 0.3), "a2_mesh_fp": row(0.50, 0.4)},
         26: {"phase_a": row(0.36, 0.4), "a2_mesh_fp": row(0.35, 0.5)}}
    d = A25.decide(t)
    assert d["a2_lost_on_both"] and d["choice"] == "phase_a"
