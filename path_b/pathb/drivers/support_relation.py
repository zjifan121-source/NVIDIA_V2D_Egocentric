"""The team's support rule (tools/track3/support_stacked.py drop_stacked_supports) on Path B's disks, run
in the stages 2-3 venv (support_stacked imports robotic_grounding).

    $RETARGET_VENV/bin/python path_b/pathb/drivers/support_relation.py <request.json>

request.json: {"disks": {object: [[cx, cy, z, r], ...]}, "out": json}; writes {"kept": [objects whose
disk stays: they rest on the table]}.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/track3"))
import support_stacked  # noqa: E402


def main(request: str) -> int:
    r = json.loads(Path(request).read_text())
    disks = {name: [tuple(d) for d in ds] for name, ds in r["disks"].items()}
    kept = support_stacked.drop_stacked_supports(disks, list(disks), None)
    Path(r["out"]).write_text(json.dumps({"kept": sorted(kept)}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
