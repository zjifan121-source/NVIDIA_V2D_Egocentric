"""S4 scene: SPIDER's bimanual scene for this episode (one or two objects, table at z = 0), and a
settle test that it holds the frame-0 objects still before any optimization runs.

  1. pathb/spider_scene.py (RETARGET_VENV): stage 2-3 output -> SPIDER's processed layout
     (trajectory_keypoints.npz, task_info.json, one visual.obj per object), slots, support plan.
  2. SPIDER (SPIDER_DIR, SPIDER_PY): decompose.py (CoACD) and generate_xml.py -> scene.xml.
  3. pathb/spider_settle.py (SPIDER_PY): hands lifted, objects at frame 0, 1 s of gravity.

Environment: RETARGET_VENV; SPIDER_DIR (a facebookresearch/spider checkout at a pinned commit, never
copied into this repo: CC BY-NC 4.0); SPIDER_PY (its Python); SPIDER_ROBOT (default xhand, which
SPIDER ships; sharpa once its MJCF assets exist, from SPIDER_ROBOT_ASSETS).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from ..paths import PATH_B, REPO
from ..report import Metric, StageResult
from ..spider_scene import DATASET, EMBODIMENT, processed_dir, task_name


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        raise RuntimeError(f"set {name} (see pathb/stages/s4_scene.py)")
    return value


def _run(cmd: list[str], cwd: Path, log: Path) -> None:
    with open(log, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        result = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join(log.read_text().splitlines()[-15:])
        raise RuntimeError(f"{Path(str(cmd[1])).name} failed (exit {result.returncode}):\n{tail}")


def robot_assets(spider_dir: Path, robot: str, dataset_dir: Path) -> Path:
    """SPIDER reads robots from <dataset>/processed/<name>/assets/robots/<robot>/bimanual.xml."""
    src = Path(os.environ.get("SPIDER_ROBOT_ASSETS") or spider_dir / "spider/assets/robots" / robot)
    if not (src / "bimanual.xml").is_file():
        raise RuntimeError(f"no bimanual.xml for robot '{robot}' in {src}; Sharpa needs SPIDER_ROBOT_ASSETS")
    dest = dataset_dir / "processed" / DATASET / "assets/robots" / robot
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest


def run(ctx) -> StageResult:
    s23 = ctx.reports.get("s23_load_retarget")
    if not s23 or s23["status"] in ("SKIP", "ERROR"):
        return StageResult([], skip="needs a stage 2-3 run (s23_load_retarget)")
    venv, spider_dir = _env("RETARGET_VENV"), Path(_env("SPIDER_DIR"))
    spider_py, robot = _env("SPIDER_PY"), _env("SPIDER_ROBOT", "xhand")
    d = ctx.dir("s4_scene")
    dataset = d / "spider_dataset"
    if dataset.exists():
        shutil.rmtree(dataset)
    log = d / "stdout.log"
    log.write_text("")
    task = task_name(ctx.number)

    _run([f"{venv}/bin/python", PATH_B / "pathb/spider_scene.py", "--run", ctx.root / "s23_load_retarget/run",
          "--episode", ctx.number, "--out", dataset], REPO, log)
    robot_assets(spider_dir, robot, dataset)
    common = ["--dataset-dir", dataset, "--dataset-name", DATASET, "--embodiment-type", EMBODIMENT,
              "--task", task, "--data-id", 0]
    _run([spider_py, "spider/preprocess/decompose.py", *common, "--robot-type", robot], spider_dir, log)
    _run([spider_py, "spider/preprocess/generate_xml.py", *common, "--robot-type", robot,
          "--hand-floor-collision", "--no-show-viewer"], spider_dir, log)
    scene = processed_dir(dataset, robot, task).parent / "scene.xml"
    keypoints = processed_dir(dataset, "mano", task) / "trajectory_keypoints.npz"
    _run([spider_py, PATH_B / "pathb/spider_settle.py", "--scene", scene, "--keypoints", keypoints,
          "--out", d / "settle.json"], REPO, log)

    info = json.loads((processed_dir(dataset, "mano", task).parent / "task_info.json").read_text())["v2d"]
    settle = json.loads((d / "settle.json").read_text())
    metrics = [
        Metric("objects in the scene minus manifest", len(info["slots"]) - len(ctx.objects), "", ok=(0, 0)),
        Metric("slots", ", ".join(f"{n} -> {s}" for n, s in info["slots"].items()), "",
               note=f"fingertip contact frames: {info['contact_frames']}"),
        Metric("contact guidance safe", int(info["guidance_safe"]), "", ok=(1, 1), warn=(0, 0),
               note="0: one hand handles both objects; run S4 without contact guidance"),
        Metric("every object supported", int(info["supported"]), "", ok=(1, 1),
               note="; ".join(f"{n}: {s['rests_on']} ({s['gap_cm']} cm)" for n, s in info["support"].items())),
        Metric("table height (world)", info["table_z_world_m"], "m", note=info["table_source"]),
        Metric("frame-0 deepest contact", settle["frame0_deepest_contact_cm"], "cm", ok=(0, 0.5), warn=(0, 1)),
        *[Metric(f"{side} object drift after {settle['seconds']:g} s", o["drift_cm"], "cm", ok=(0, 1), warn=(0, 2))
          for side, o in settle["objects"].items()],
        *[Metric(f"{side} object turn after {settle['seconds']:g} s", o["turn_deg"], "deg", ok=(0, 2), warn=(0, 5))
          for side, o in settle["objects"].items()],
    ]
    return StageResult(metrics, {"scene": scene, "keypoints": keypoints, "dataset_dir": dataset,
                                 "robot": robot, "settle": d / "settle.json"})
