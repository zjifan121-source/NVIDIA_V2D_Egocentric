"""tools/track3/run_load_retarget.py for an evaluation episode: the runner unchanged, with its scan
directory pointed at nothing, so its bundle check reads no public scan (GT firewall, V3_BUILD_SPEC.md §7).

    $RETARGET_VENV/bin/python path_b/pathb/drivers/run_lr.py <run_load_retarget.py arguments>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/track3"))
import run_load_retarget as L  # noqa: E402

L.SCANS = Path("/nonexistent/evaluation-reads-no-scans")
sys.argv = ["run_load_retarget.py", *sys.argv[1:]]
sys.exit(L.main())
