"""The team runner's output checks (tools/track3/run_load_retarget.py check_output) on one partition,
run in the stages 2-3 venv ($RETARGET_VENV: trimesh, pxr).

    $RETARGET_VENV/bin/python path_b/pathb/drivers/lr_check.py <request.json>

request.json: {"partition": dir, "meshes": [obj, ...], "support_usd": usda, "frames": n, "fps": Hz,
"out": json}. Writes check_output's report to "out".
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/track3"))
import run_load_retarget as L  # noqa: E402


def main(request: str) -> int:
    r = json.loads(Path(request).read_text())
    report = L.check_output(Path(r["partition"]), [Path(p) for p in r["meshes"]], Path(r["support_usd"]),
                            int(r["frames"]), float(r["fps"]))
    Path(r["out"]).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
