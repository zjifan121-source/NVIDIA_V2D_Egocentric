"""Team tooling Path B reuses in-process (tools/track3 is shared by both paths): bundle I/O and the
stage 2 repairs (run_load_retarget), and the local scorer (score_local)."""
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
