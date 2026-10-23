"""Phase 2 (pathb/phase2.py): support relation, relative anchors, common rest frames."""
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pathb import phase2

RETARGET = Path(os.environ.get("RETARGET_VENV", Path.home() / "V2D/venvs/retarget"))


def box(lo, hi, n=400, seed=0):
    return np.random.default_rng(seed).uniform(lo, hi, (n, 3))


def test_relative_anchor_cancels_the_camera():
    rng = np.random.default_rng(1)
    rel = np.eye(4)
    rel[:3, :3] = Rotation.from_rotvec([0, 0, 0.4]).as_matrix()
    rel[:3, 3] = [0.15, -0.05, 0.02]
    A = np.tile(np.eye(4), (20, 1, 1))
    A[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.3, (20, 3))).as_matrix()  # the camera wanders
    A[:, :3, 3] = rng.normal(0, 0.3, (20, 3))
    B = A @ rel
    B[:, :3, 3] += rng.normal(0, 0.002, (20, 3))  # tracker noise
    got = phase2.relative_anchor(A, B, list(range(20)))
    assert np.abs(got[:3, 3] - rel[:3, 3]).max() < 0.003 and np.allclose(got[:3, :3], rel[:3, :3], atol=1e-6)
    assert phase2.relative_anchor(A, B, []) is None


def test_common_rest_frames():
    phases = {"a": {"rest_before": [0, 9], "rest_after": [30, 39]}, "b": {"rest_before": [0, 14], "rest_after": [25, 39]}}
    have = np.ones(40, bool)
    have[3] = False
    assert phase2.common_rest(phases, "a", "b", "rest_before", have) == [0, 1, 2, 4, 5, 6, 7, 8, 9]
    assert phase2.common_rest(phases, "a", "b", "rest_after", have) == list(range(30, 40))


@pytest.mark.skipif(not (RETARGET / "bin/python").exists(), reason="needs the stages 2-3 venv (support_stacked imports robotic_grounding)")
def test_pot_in_the_rack_rests_on_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RETARGET_VENV", str(RETARGET))
    rack = phase2.disk(box([0.0, 0.0, 0.0], [0.4, 0.3, 0.12]))
    pot = phase2.disk(box([0.12, 0.08, 0.03], [0.28, 0.22, 0.15], seed=2))  # inside the rack, 3 cm up
    cup = phase2.disk(box([0.6, 0.0, 0.0], [0.68, 0.08, 0.1], seed=3))  # beside it, on the table
    assert phase2.supports({"plastic_dish_rack": rack, "white_pot": pot, "blue_cup": cup}, tmp_path) == {
        "plastic_dish_rack": "table", "white_pot": "plastic_dish_rack", "blue_cup": "table"}
