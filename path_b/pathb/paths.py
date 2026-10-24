"""Where Path B reads from. It writes only under path_b/data/ (gitignored)."""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

PATH_B = Path(__file__).resolve().parents[1]
REPO = PATH_B.parent
TRACK3 = REPO / "assets/data/track_3"
# evaluation/ ships no meta/; both splits were recorded with the same head rig.
CALIBRATION = TRACK3 / "public/meta/camera_calibration.json"
MANIFEST = REPO / "tools/track3/episodes.json"
TOOLS_TRACK3 = REPO / "tools/track3"
RECON = REPO / "baseline/reconstruction"  # used read-only
HOST_PY = RECON / ".venv/bin/python"  # baseline host wrappers (README §5.2) that launch its containers
STEREO_WEIGHTS = RECON / "data/weights/foundation_stereo"
OUTPUTS = PATH_B / "data/outputs"
TRACK3_FPS = 20.25411910142744  # assets/data/track_3/public/meta/info.json
DEV_EPISODES = (41, 13)  # V4_BUILD_SPEC.md: ep41 (white_pot) and ep13 (blue_cup in the dish rack)


def video(split: str, episode: int, camera: str) -> Path:
    """camera: a (head colour), b or c (head mono stereo)."""
    return TRACK3 / split / f"videos/chunk-000/observation.images.ego_cam_{camera}/episode_{episode:06d}.mp4"


@lru_cache(maxsize=None)
def manifest() -> dict[int, dict]:
    return {e["episode"]: e for e in json.loads(MANIFEST.read_text())["episodes"]}


def episode_info(episode: int) -> dict:
    try:
        return manifest()[episode]
    except KeyError:
        raise SystemExit(f"episode {episode} is not in {MANIFEST}") from None


def git_commit() -> dict:
    """HEAD of this checkout and whether tracked files differ from it."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()
    return {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
