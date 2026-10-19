"""Runs Path B's stages for one episode. Every stage writes <out>/episode_XXXXXX/<stage>/report.json.

A stage is skipped when its report exists and is not ERROR, unless --force is given or an earlier
stage ran again in this call (its inputs may have changed). Stages pass results only through files.
"""
from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import loss
from .paths import STEREO_WEIGHTS, episode_info, git_commit
from .report import StageResult, worst

STAGES = ("p0_ingest", "p1_calibrate", "p2_depth", "p3_scale_bridge", "p4_export", "s23_load_retarget")


@dataclass
class Context:
    episode: dict  # this episode's tools/track3/episodes.json entry
    root: Path  # <out>/episode_XXXXXX
    reports: dict  # stage name -> report of the stages already run or loaded

    @property
    def number(self) -> int:
        return int(self.episode["episode"])

    @property
    def split(self) -> str:
        return self.episode["split"]

    @property
    def public(self) -> bool:
        return self.split == "public"

    @property
    def objects(self) -> list[str]:
        return [o["name"] for o in self.episode["objects"]]

    def dir(self, stage: str) -> Path:
        path = self.root / stage
        path.mkdir(parents=True, exist_ok=True)
        return path


def to_json(x):
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON serializable: {type(x)}")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, default=to_json) + "\n")


class GpuPeak:
    """Peak GPU memory in use (MiB, whole GPU) while a stage runs: nvidia-smi every 0.5 s."""

    def __init__(self):
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                values = [int(v) for v in out.split()]
                if values:
                    self.peak = max([self.peak or 0, *values])
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(0.5)

    def __enter__(self) -> GpuPeak:
        if shutil.which("nvidia-smi"):
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(2)


def run_stage(name: str, ctx: Context, force: bool) -> tuple[dict, bool]:
    """(report, whether it ran)."""
    path = ctx.root / name / "report.json"
    if not force and path.is_file():
        report = json.loads(path.read_text())
        if report["status"] != "ERROR":
            return report, False
    started, t0 = datetime.now(timezone.utc), time.monotonic()
    error = None
    with GpuPeak() as gpu:
        try:
            result = importlib.import_module(f"pathb.stages.{name}").run(ctx)
        except Exception:  # recorded in the report; the episode stops here
            result, error = StageResult([]), traceback.format_exc()
    metrics = [m.to_dict() for m in result.metrics]
    report = {
        "stage": name,
        "status": "ERROR" if error else "SKIP" if result.skip else worst(m["status"] for m in metrics),
        "episode": ctx.number, "split": ctx.split, "commit": git_commit(),
        "started_utc": started.isoformat(timespec="seconds"), "seconds": round(time.monotonic() - t0, 1),
        "gpu_peak_mib": gpu.peak, "metrics": metrics, "outputs": result.outputs,
        "skip_reason": result.skip, "error": error,
    }
    write_json(ctx.dir(name) / "report.json", report)
    return report, True


def gpu_name() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip() or None


def run_episode(number: int, out: Path, stages: tuple[str, ...] = STAGES, force: bool = False) -> dict:
    ctx = Context(episode_info(number), out / f"episode_{number:06d}", {})
    ctx.root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"== episode {number} ({ctx.split}, {' + '.join(ctx.objects)})", flush=True)
    rerun = force
    for name in STAGES:
        if name not in stages:
            existing = ctx.root / name / "report.json"
            if existing.is_file():
                ctx.reports[name] = json.loads(existing.read_text())
            continue
        report, ran = run_stage(name, ctx, rerun)
        rerun = rerun or ran
        ctx.reports[name] = report
        flagged = [m["name"] for m in report["metrics"] if m["status"] in ("WARN", "FAIL")]
        print(f"  {name:18s} {report['status']:5s} {report['seconds']:7.1f} s"
              + ("" if ran else "  (kept)") + (f"  flagged: {', '.join(flagged)}" if flagged else ""), flush=True)
        if report["status"] == "ERROR":
            print("\n".join(report["error"].splitlines()[-6:]), flush=True)
            break
    write_json(ctx.root / "run.json", {
        "command": sys.argv, "commit": git_commit(), "host": platform.node(), "gpu": gpu_name(),
        "python": sys.version.split()[0], "stereo_engines": sorted(p.name for p in STEREO_WEIGHTS.glob("*.engine")),
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages": {n: r["status"] for n, r in ctx.reports.items()},
    })
    return loss.write(ctx.root, ctx.reports)
