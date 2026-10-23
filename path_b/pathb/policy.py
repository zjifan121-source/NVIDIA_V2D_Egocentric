"""FlashCHORD policies (V3_BUILD_SPEC.md S0.1 and S6), shared by run.py oracle and the s6_* stages.

Training: scripts/train_flash_sac.py experiment=sharpa_flash_sac with the still submission's settings
(tools/track3/jobs/still_submit.sh: STEPS and EXTRA, Edmond's overrides), contact buffers as spec S6
sets them for the L40S; a rerun resumes from the newest state_*.safetensors. Evaluation: the kit's
examples/run_policy_evaluation.py (one world, explicit reset), which writes episode_XXXXXX.parquet
(the rollout: achieved and reference object poses per step) and episode_XXXXXX.json (FlashCHORD's
metrics). Runs on the GPU machine, in the flash_chord venv ($FLASH_CHORD_VENV).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .paths import REPO

FLASH_CHORD = REPO / "baseline/robotic_grounding/flash_chord"
KIT = REPO / "assets/data/v2d_submission_kit"
STILL_JOB = REPO / "tools/track3/jobs/still_submit.sh"
L40S_BUFFERS = ["sim.njmax=512", "sim.nconmax=512"]  # spec S6 (James 2026-10-20: njmax 512 on the L40S)


def flash_venv() -> str:
    return os.environ.get("FLASH_CHORD_VENV") or "/root/v2d/venvs/flash_chord"


def still_settings() -> tuple[int, list[str]]:
    """(steps, Hydra overrides) of the still submission, read from jobs/still_submit.sh, with the
    contact buffers replaced by the L40S ones."""
    text = STILL_JOB.read_text()
    steps = int(re.search(r"^STEPS=(\d+)", text, re.M).group(1))
    extra = re.search(r'^export EXTRA="([^"]+)"', text, re.M).group(1).split()
    return steps, [o for o in extra if not o.startswith(("sim.njmax=", "sim.nconmax="))] + L40S_BUFFERS


def run_logged(cmd: list, log: Path, cwd: Path, env: dict | None = None) -> None:
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        rc = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"exit {rc}: {' '.join(map(str, cmd[:3]))} ...; see {log}")


def numbered(folder: Path, pattern: str) -> list[Path]:
    """Files matching pattern, by the number in their name (state_123.safetensors)."""
    return sorted(folder.glob(pattern), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))


def train(partition, run: Path, name: str) -> dict:
    """Train on one task partition into run/train; resumable; run/train.json when done."""
    done = run / "train.json"
    if done.is_file():
        return json.loads(done.read_text())
    run.mkdir(parents=True, exist_ok=True)
    steps, overrides = still_settings()
    resume = numbered(run / "train", "state_*.safetensors")
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", WANDB_MODE="offline", WANDB_DIR=str(run),
               WANDB_SILENT="true")
    cmd = [f"{flash_venv()}/bin/python", "scripts/train_flash_sac.py", "experiment=sharpa_flash_sac",
           f"task.parquet='{partition}'", "reset.seed=42", "logging.mode=offline", "logging.upload_checkpoints=false",
           f"logging.run_name={name}", f"training.total_environment_steps={steps}", f"output_dir={run / 'train'}",
           *overrides]
    if resume:
        cmd += ["training.resume=true", f"training.checkpoint_path='{resume[-1]}'"]
    t0 = datetime.now(timezone.utc)
    run_logged(cmd, run / "train.log", FLASH_CHORD, env)
    policies = numbered(run / "train", "policy_*.safetensors")
    if not policies:
        raise RuntimeError(f"training wrote no policy_*.safetensors under {run / 'train'}")
    info = {"checkpoint": str(policies[-1]), "partition": str(partition), "steps": steps, "overrides": overrides,
            "seconds": (datetime.now(timezone.utc) - t0).total_seconds(), "resumed_from": str(resume[-1]) if resume else None,
            "nefc_overflow_lines": (run / "train.log").read_text().count("nefc overflow")}
    done.write_text(json.dumps(info, indent=1) + "\n")
    return info


def evaluate(checkpoint, number: int, run: Path) -> Path:
    """The kit's evaluator on one checkpoint: run/results/episode_XXXXXX.{parquet,json}."""
    results = run / "results"
    if (run / "eval.done").is_file():
        return results
    shutil.rmtree(results, ignore_errors=True)
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints.json").write_text(json.dumps({str(number): str(checkpoint)}) + "\n")
    run_logged([f"{flash_venv()}/bin/python", KIT / "examples/run_policy_evaluation.py", "--checkpoints",
                run / "checkpoints.json", "--results", results, "--runtime-root", FLASH_CHORD],
               run / "eval.log", FLASH_CHORD, dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"))
    (run / "eval.done").touch()
    return results
