"""vm/score_checkpoint.py submitted(): the submitted AUC estimate, playback less F2's policy tax by episode type."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("score_checkpoint", Path(__file__).parents[1] / "vm/score_checkpoint.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def ep(objects: list[str], auc: float | None) -> dict:
    return {"objects": objects, "scores": {} if auc is None else {"ref_inverted": {"AUC": auc}}}


def test_submitted_by_type():
    eps = {1: ep(["pot"], 0.9), 2: ep(["pot"], 0.7), 3: ep(["cup", "rack"], 0.6), 4: ep(["cup", "rack"], 0.5),
           5: ep(["cup", "rack"], 0.4)}
    s = C.submitted(eps)
    single, two = 0.8 - C.POLICY_TAX["single-object"], 0.5 - C.POLICY_TAX["two-object"]
    assert s["single-object"] == pytest.approx(single, abs=1e-4)
    assert s["two-object"] == pytest.approx(two, abs=1e-4)
    assert s["all"] == pytest.approx((2 * single + 3 * two) / 5, abs=1e-4)  # combined by episode count


def test_submitted_missing_floor_and_one_type():
    s = C.submitted({1: ep(["pot"], None), 2: ep(["cup", "rack"], 0.1)})  # no candidates: 0 playback, as in means()
    assert (s["single-object"], s["two-object"], s["all"]) == (0.0, 0.0, 0.0)  # floored, never negative
    s = C.submitted({1: ep(["pot"], 0.9)})
    assert s["two-object"] is None and s["all"] == pytest.approx(0.9 - C.POLICY_TAX["single-object"], abs=1e-4)
