"""Support relation (pathb/support.py; kept from v3's phase 2 for V4 S3b/S3d)."""
import os
from pathlib import Path

import numpy as np
import pytest

from pathb import support

RETARGET = Path(os.environ.get("RETARGET_VENV", Path.home() / "V2D/venvs/retarget"))


def box(lo, hi, n=400, seed=0):
    return np.random.default_rng(seed).uniform(lo, hi, (n, 3))


@pytest.mark.skipif(not (RETARGET / "bin/python").exists(), reason="needs the stages 2-3 venv (support_stacked imports robotic_grounding)")
def test_pot_in_the_rack_rests_on_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RETARGET_VENV", str(RETARGET))
    rack = support.disk(box([0.0, 0.0, 0.0], [0.4, 0.3, 0.12]))
    pot = support.disk(box([0.12, 0.08, 0.03], [0.28, 0.22, 0.15], seed=2))  # inside the rack, 3 cm up
    cup = support.disk(box([0.6, 0.0, 0.0], [0.68, 0.08, 0.1], seed=3))  # beside it, on the table
    assert support.supports({"plastic_dish_rack": rack, "white_pot": pot, "blue_cup": cup}, tmp_path) == {
        "plastic_dish_rack": "table", "white_pot": "plastic_dish_rack", "blue_cup": "table"}


def test_a_dropped_disk_at_the_table_comes_back_unless_kept_disks_cover_it():
    from pathb.drivers import support_surfaces as S
    rack, sweeper = (-0.060, -0.003, 1.001, 0.237), (-0.040, -0.005, 0.997, 0.101)  # ep31: both dropped
    lid = (0.0, 0.0, 1.150, 0.08)  # above the table: a real phantom, stays dropped
    out = S.restore({}, {"rack": [rack], "sweeper": [sweeper], "lid": [lid]}, 1.0, 0.05)
    assert out == {"rack": [rack], "sweeper": [sweeper]}
    cup = (0.10, 0.13, 1.000, 0.05)  # ep13's still scene: NVIDIA keeps the cup's disk, drops the rack's
    assert S.restore({"cup": [cup]}, {"cup": [cup], "rack": [rack]}, 1.0, 0.05)["rack"] == [rack]
    pot = (-0.05, 0.0, 0.985, 0.12)  # ep26: the pot's dropped disk lies on the rack's kept one
    assert S.restore({"rack": [rack]}, {"rack": [rack], "pot": [pot]}, 1.0, 0.05) == {"rack": [rack]}
    assert S.covered(pot, [rack], 0.05) == 1.0 and S.covered(rack, [cup], 0.05) < 0.1
