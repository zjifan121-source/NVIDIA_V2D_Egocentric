"""The GT firewall (V3_BUILD_SPEC.md §7): ground truth and scans only through pathb/gt.py, only for
public episodes, every read recorded in report.json: files_read."""
import importlib.machinery
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from pathb import gt, runner

PATHB = Path(__file__).parents[1] / "pathb"
EVALUATION = {"episode": 1, "split": "evaluation", "num_frames": 10, "objects": [{"name": "white_pot"}]}
PUBLIC = {"episode": 41, "split": "public", "num_frames": 299, "objects": [{"name": "white_pot"}]}


def test_evaluation_episodes_are_refused_before_any_read():
    read = []
    for call in (lambda: gt.ground_truth(EVALUATION, read),
                 lambda: gt.score(EVALUATION, np.zeros((10, 1, 7)), {}, read),
                 lambda: gt.scan_size_ratio(EVALUATION, np.zeros((4, 3)), "white_pot", read),
                 lambda: gt.scan_vertices(EVALUATION, "white_pot", read),
                 lambda: gt.rescale_by_scan(EVALUATION, "white_pot", read)):
        with pytest.raises(gt.GroundTruthFirewall):
            call()
    assert read == []


# What reads ground truth or scans: score_local / run_load_retarget calls and the data folders.
FORBIDDEN = re.compile(r"\.ground_truth\(|scan_size_ratio\(|\.score\(\[|rescale_by_scan\s*=|\bSCANS\b|"
                       r"public/data|public/mesh|_visual\.glb")
# The one allowed mention: drivers/run_lr.py points the runner's scan dir at nothing (evaluation episodes).
BLANKS_SCANS = ("drivers/run_lr.py", re.compile(r'^L\.SCANS = Path\("/nonexistent/[^"]*"\)$'))


def test_only_gt_py_reads_ground_truth_or_scans():
    offenders = []
    for path in sorted(PATHB.rglob("*.py")):
        if path.name == "gt.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if str(path.relative_to(PATHB)) == BLANKS_SCANS[0] and BLANKS_SCANS[1].match(line.strip()):
                continue
            code = re.sub(r"\bgt\.(ground_truth|score|scan_size_ratio|scan_vertices|rescale_by_scan)\(", "", line)
            if FORBIDDEN.search(code.split("#")[0]):
                offenders.append(f"{path.relative_to(PATHB)}:{i}: {line.strip()}")
    assert offenders == []


def fake_stage(monkeypatch, name, run):
    module = types.ModuleType(f"pathb.stages.{name}")
    module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, None)
    module.run = run
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_runner_records_reads_and_stops_evaluation_reads(tmp_path, monkeypatch):
    def reads(ctx):
        ctx.files_read.append(gt.parquet(ctx.number)) if ctx.public else gt.ground_truth(ctx.episode, ctx.files_read)
        return runner.StageResult([])
    fake_stage(monkeypatch, "_reads_gt", reads)
    public = runner.Context(PUBLIC, tmp_path / "public", {})
    report, _ = runner.run_stage("_reads_gt", public, force=True)
    assert report["status"] == "PASS" and report["files_read"] == [str(gt.parquet(41))]
    evaluation = runner.Context(EVALUATION, tmp_path / "evaluation", {})
    report, _ = runner.run_stage("_reads_gt", evaluation, force=True)
    assert report["status"] == "ERROR" and "GroundTruthFirewall" in report["error"]
    assert report["files_read"] == []
