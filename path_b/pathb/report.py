"""What every stage records: each measurement with its unit, its bar and a verdict.

A stage's status is the worst verdict among its barred measurements (INFO ones carry no bar).
PASS, WARN and FAIL let the episode go on, so the loss report can show every stage of a bad run;
ERROR (an exception) stops it, because later stages lack their inputs. SKIP: the stage does not
apply to this episode.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ORDER = ("PASS", "WARN", "FAIL", "ERROR")


def _num(x: float) -> str:
    return "∞" if x == math.inf else "-∞" if x == -math.inf else f"{x:g}"


@dataclass
class Metric:
    name: str
    value: float | int | str | None
    unit: str = ""
    kind: str = "health"  # health: needs no ground truth (all 40 episodes); gt: public ground truth or scans
    ok: tuple[float, float] | None = None  # PASS range, inclusive
    warn: tuple[float, float] | None = None  # WARN range around it; outside both is FAIL
    note: str = ""

    @property
    def status(self) -> str:
        if self.ok is None or not isinstance(self.value, (int, float)):
            return "INFO"
        if not math.isfinite(self.value):
            return "FAIL"
        if self.ok[0] <= self.value <= self.ok[1]:
            return "PASS"
        if self.warn and self.warn[0] <= self.value <= self.warn[1]:
            return "WARN"
        return "FAIL"

    @property
    def bar(self) -> str:
        if self.ok is None:
            return ""
        text = f"pass {_num(self.ok[0])}..{_num(self.ok[1])}"
        return text + (f", warn {_num(self.warn[0])}..{_num(self.warn[1])}" if self.warn else "")

    def to_dict(self) -> dict:
        value = round(self.value, 4) if isinstance(self.value, float) and math.isfinite(self.value) else self.value
        return {"name": self.name, "value": value, "unit": self.unit, "kind": self.kind,
                "bar": self.bar, "status": self.status, "note": self.note}


@dataclass
class StageResult:
    metrics: list[Metric]
    outputs: dict = field(default_factory=dict)
    skip: str | None = None  # why the stage does not apply to this episode
    files_read: list = field(default_factory=list)  # every input file the stage read (spec §7)


def worst(statuses) -> str:
    ranked = [s for s in statuses if s in ORDER]
    return max(ranked, key=ORDER.index) if ranked else "PASS"
