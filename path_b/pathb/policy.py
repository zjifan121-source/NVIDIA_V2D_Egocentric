"""FlashCHORD policies (V4_BUILD_SPEC.md S0.1 and S6), shared by run.py oracle and the s6_* stages.

Training: scripts/train_flash_sac.py experiment=sharpa_flash_sac with our own recipe (`settings`; no setting
is read from the team's job scripts, James 2026-10-29): FlashCHORD's own curriculum
(tracking_250m_mixed_reset) with its thresholds scaled by SHORT_FRACTION and rounded to whole steps of
the scene's worlds, the reference played at its own frame rate (task.motion_speed 1.0, sim.fps =
TRACK3_FPS: the kit's packer numbers each evaluator step as one video frame, reference_frame = step +
start_frame), FlashCHORD's defaults otherwise (wrist termination 0.15 m), and the contact buffers spec §14 G1
sets for the L40S (1024 / 1024); a rerun resumes from the newest state_*.safetensors. long=True: FlashCHORD's own 250M
schedule unscaled (250,003,456 steps). $PATHB_OBJECT_STOP (metres, e.g. 0.05) sets
termination.object_position.threshold (V4 S6's optional stop, adopted only after S0.1). Every run
keeps the experiment's action and termination groups: the overrides may not name them, and the run's
own Hydra record (hydra.run.dir = <run>/hydra) must show action=residual_hand_pose and
termination=tracking (V4 S6: the actor outputs a correction to the reference wrist pose). Evaluation: the kit's
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

from .paths import REPO, TRACK3_FPS

FLASH_CHORD = REPO / "baseline/robotic_grounding/flash_chord"
KIT = REPO / "assets/data/v2d_submission_kit"
L40S_BUFFERS = ["sim.njmax=1024", "sim.nconmax=1024"]  # spec §14 G1 (James 2026-10-29; was 512: ep13 needed 708)
CURRICULUM = FLASH_CHORD / "src/flash_chord/configs/curriculum/tracking_250m_mixed_reset.yaml"
SCENE = FLASH_CHORD / "src/flash_chord/configs/scene/train.yaml"
LONG_STEPS = 250_003_456  # the last threshold of tracking_250m_mixed_reset.yaml
SHORT_FRACTION = 0.24  # default: the short schedule is this share of the 250M one (~60M steps, ~20 min on the L40S)
GROUPS = {"action": "residual_hand_pose", "termination": "tracking"}  # what V4 S6 relies on


def flash_venv() -> str:
    return os.environ.get("FLASH_CHORD_VENV") or "/root/v2d/venvs/flash_chord"


def short_thresholds() -> list[int]:
    """FlashCHORD's 250M curriculum thresholds times SHORT_FRACTION, each rounded to a whole number of
    environment steps of the scene's world count (the last is the run's step count)."""
    import yaml
    thresholds = yaml.safe_load(CURRICULUM.read_text())["thresholds"]
    worlds = int(yaml.safe_load(SCENE.read_text())["world_count"])
    if thresholds[-1] != LONG_STEPS:
        raise ValueError(f"{CURRICULUM.name} ends at {thresholds[-1]}, not {LONG_STEPS}")
    return [max(1, round(t * SHORT_FRACTION / worlds)) * worlds for t in thresholds]


def settings(long: bool = False) -> tuple[int, list[str]]:
    """(steps, Hydra overrides): the short schedule (FlashCHORD's curriculum scaled, see `short_thresholds`)
    or FlashCHORD's own 250M one; both play the reference at its frame rate, one step per video frame."""
    overrides = ["task.motion_speed=1.0", f"sim.fps={TRACK3_FPS}", *L40S_BUFFERS]
    if long:
        steps = LONG_STEPS
    else:
        thresholds = short_thresholds()
        steps = thresholds[-1]
        overrides.append("curriculum.thresholds=[" + ",".join(map(str, thresholds)) + "]")
    stop = os.environ.get("PATHB_OBJECT_STOP")
    if stop:
        overrides = overrides + [f"termination.object_position.threshold={float(stop)}"]
    check_overrides(overrides)
    return steps, overrides


def check_overrides(overrides: list[str]) -> None:
    """None of them may replace the experiment's action or termination group."""
    for o in overrides:
        key = o.lstrip("+~").split("=")[0].removeprefix("override ").strip("/")
        if key in GROUPS:
            raise ValueError(f"override {o!r} replaces the {key} group (V4 S6 needs {GROUPS[key]})")


def chosen_groups(hydra_dir: Path) -> dict[str, str]:
    """The config groups a run composed (Hydra's .hydra/hydra.yaml, runtime.choices)."""
    import yaml
    f = hydra_dir / ".hydra/hydra.yaml"
    if not f.is_file():
        return {}
    choices = yaml.safe_load(f.read_text()).get("hydra", {}).get("runtime", {}).get("choices", {})
    return {k: choices.get(k) for k in GROUPS}


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


def train(partition, run: Path, name: str, long: bool = False, seed: int = 42) -> dict:
    """Train on one task partition into run/train; resumable; run/train.json when done."""
    run = Path(run).resolve()  # FlashCHORD runs in its own folder: a relative output_dir would land there
    done = run / "train.json"
    if done.is_file():
        return json.loads(done.read_text())
    run.mkdir(parents=True, exist_ok=True)
    steps, overrides = settings(long)
    resume = numbered(run / "train", "state_*.safetensors")
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", WANDB_MODE="offline", WANDB_DIR=str(run),
               WANDB_SILENT="true")
    cmd = [f"{flash_venv()}/bin/python", "scripts/train_flash_sac.py", "experiment=sharpa_flash_sac",
           f"task.parquet='{partition}'", f"reset.seed={seed}", "logging.mode=offline", "logging.upload_checkpoints=false",
           f"logging.run_name={name}", f"training.total_environment_steps={steps}", f"output_dir={run / 'train'}",
           f"hydra.run.dir={run / 'hydra'}", *overrides]
    if resume:
        cmd += ["training.resume=true", f"training.checkpoint_path='{resume[-1]}'"]
    t0 = datetime.now(timezone.utc)
    run_logged(cmd, run / "train.log", FLASH_CHORD, env)
    policies = numbered(run / "train", "policy_*.safetensors")
    if not policies:
        raise RuntimeError(f"training wrote no policy_*.safetensors under {run / 'train'}")
    groups = chosen_groups(run / "hydra")
    if groups != GROUPS:
        raise RuntimeError(f"the run composed {groups}, not {GROUPS} ({run / 'hydra/.hydra/hydra.yaml'})")
    info = {"checkpoint": str(policies[-1]), "partition": str(partition), "steps": steps, "overrides": overrides,
            "seed": seed, "long": long, "groups": groups,
            "seconds": (datetime.now(timezone.utc) - t0).total_seconds(), "resumed_from": str(resume[-1]) if resume else None,
            "nefc_overflow_lines": (run / "train.log").read_text().count("nefc overflow")}
    done.write_text(json.dumps(info, indent=1) + "\n")
    return info


def evaluate(checkpoint, number: int, run: Path) -> Path:
    """The kit's evaluator on one checkpoint: run/results/episode_XXXXXX.{parquet,json}."""
    run = Path(run).resolve()
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
