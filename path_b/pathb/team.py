"""Team tooling Path B reuses in-process (tools/track3 is shared by both paths): bundle I/O and the
stage 2 repairs (run_load_retarget), the local scorer (score_local) and the S0.2 oracle study."""
from __future__ import annotations

import sys

from .paths import TOOLS_TRACK3

if str(TOOLS_TRACK3) not in sys.path:
    sys.path.insert(0, str(TOOLS_TRACK3))


def retarget_tools():
    import run_load_retarget
    return run_load_retarget


def scorer():
    import score_local
    return score_local


def playback_oracle():
    """Edmond's S0.2 study (tools/track3/playback_oracle.py): its moving_mask is the ground truth's
    moving / resting split every S0.2 variant uses."""
    import playback_oracle
    return playback_oracle


def lr_command(ctx_public: bool, args: list) -> tuple[list[str], dict]:
    """(command, environment) running run_load_retarget.py through pathb/drivers/run_lr.py in
    $RETARGET_VENV: scans blanked on evaluation episodes (PATHB_NO_SCANS), the device from
    $RETARGET_DEVICE when set (cpu on the Mac), MANO_DIR passed through."""
    import os
    from .paths import PATH_B
    venv = os.environ.get("RETARGET_VENV")
    if not venv:
        raise RuntimeError("set RETARGET_VENV to the stages 2-3 environment (README §5.12)")
    env = dict(os.environ)
    env.pop("PATHB_NO_SCANS", None)
    if not ctx_public:
        env["PATHB_NO_SCANS"] = "1"
    device = os.environ.get("RETARGET_DEVICE")
    cmd = [f"{venv}/bin/python", str(PATH_B / "pathb/drivers/run_lr.py"), *map(str, args)]
    return cmd + (["--device", device] if device else []), env
