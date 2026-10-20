import math

from pathb.report import Metric, worst


def test_metric_verdicts():
    assert Metric("a", 0.5, ok=(0, 1)).status == "PASS"
    assert Metric("a", 1.5, ok=(0, 1), warn=(0, 2)).status == "WARN"
    assert Metric("a", 3.0, ok=(0, 1), warn=(0, 2)).status == "FAIL"
    assert Metric("a", math.nan, ok=(0, 1)).status == "FAIL"
    assert Metric("a", "text").status == "INFO"
    assert Metric("a", 3.0).status == "INFO"
    assert Metric("a", 12, ok=(10, math.inf)).bar == "pass 10..∞"


def test_stage_status_is_the_worst_barred_verdict():
    assert worst(["PASS", "INFO", "WARN"]) == "WARN"
    assert worst(["INFO"]) == "PASS"
    assert worst(["FAIL", "ERROR", "PASS"]) == "ERROR"
