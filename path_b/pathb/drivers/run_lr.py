"""tools/track3/run_load_retarget.py as Path B's stages run it: the runner unchanged, two of its constants
set from the environment.

    [PATHB_NO_SCANS=1] [MANO_DIR=<dir with models/>] $RETARGET_VENV/bin/python path_b/pathb/drivers/run_lr.py <args>

PATHB_NO_SCANS=1 (evaluation episodes): its scan directory points at nothing, so its bundle check reads
no public scan (GT firewall, V3_BUILD_SPEC.md §7). MANO_DIR: where models/MANO_{LEFT,RIGHT}.pkl are
when not under baseline/reconstruction/data/weights/hand (the Mac keeps them in ~/V2D/mano_v1_2).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/track3"))
import run_load_retarget as L  # noqa: E402

if os.environ.get("PATHB_NO_SCANS") == "1":
    L.SCANS = Path("/nonexistent/evaluation-reads-no-scans")
if os.environ.get("MANO_DIR"):
    L.MANO_DIR = Path(os.environ["MANO_DIR"])
sys.argv = ["run_load_retarget.py", *sys.argv[1:]]
sys.exit(L.main())
