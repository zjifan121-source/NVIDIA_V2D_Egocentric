"""Call one baseline host wrapper with keyword arguments from a JSON file. Each wrapper starts its
module's container (baseline/reconstruction/modules/*/docker/run_*.py); baseline/ is used as it is.

    baseline/reconstruction/.venv/bin/python path_b/pathb/drivers/baseline_call.py <call.json>
    call.json: {"function": "v2d.droid_slam.docker.run_video_to_slam:run_video_to_slam", "kwargs": {...}}

Run with cwd = baseline/reconstruction (the wrappers' relative weight paths start there).
"""
import importlib
import json
import sys
import time
from pathlib import Path


def main(path: str) -> int:
    call = json.loads(Path(path).read_text())
    module, name = call["function"].split(":")
    t0 = time.time()
    getattr(importlib.import_module(module), name)(**call["kwargs"])
    print(f"[baseline_call] {call['function']} finished in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
