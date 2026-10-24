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
