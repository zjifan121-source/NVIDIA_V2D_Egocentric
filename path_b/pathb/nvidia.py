"""NVIDIA's stage 2-3 code exactly as imported (James, 2026-10-29: Path B runs no team code; baseline/ holds
team changes since the import, e.g. the ego loader's fps and two-object merge, the MANO scale columns).

`pristine(rel)` mirrors baseline/<rel> under path_b/data/nvidia_<commit>/<rel> as a tree of symlinks to
the checkout's files (so LFS assets are the real ones), except every file the team changed since
NVIDIA_IMPORT: modified and deleted files are written from that commit, files added since are left out.
baseline/ itself is never written. Python finds the mirror before the venv's editable installs: those
append their finders to sys.meta_path, after the PathFinder that reads PYTHONPATH (scripts), and
drivers/nvidia_stage23.py maps the loader's package names onto the mirror itself.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .paths import PATH_B, REPO

NVIDIA_IMPORT = "d9264a4e"  # baseline/ = NVIDIA video_to_data main@8c778f9, imported unmodified (PR #10)
LOADER = "reconstruction/modules/v2d_task_library_loader"
ROBOTIC = "robotic_grounding"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def changed(rel: str) -> dict[str, str]:
    """{path under baseline/<rel>: git status letter} for files that differ from NVIDIA_IMPORT (M, A, D)."""
    out = {}
    for line in git("diff", "--name-status", "--no-renames", NVIDIA_IMPORT, "--", f"baseline/{rel}").splitlines():
        status, path = line.split("\t", 1)
        out[str(Path(path).relative_to(f"baseline/{rel}"))] = status[0]
    return out


def pristine(rel: str) -> Path:
    """The mirror of baseline/<rel> at NVIDIA_IMPORT (built once per set of team changes)."""
    src = REPO / "baseline" / rel
    dst = PATH_B / f"data/nvidia_{NVIDIA_IMPORT}" / rel
    diff = changed(rel)
    marker = dst / ".pristine.json"
    if marker.is_file() and json.loads(marker.read_text()) == diff:
        return dst
    restore = {p for p, s in diff.items() if s in ("M", "D")}
    skip = {p for p, s in diff.items() if s == "A"}
    special = restore | skip
    keep = {str(Path(p).parent) for p in special}  # directories that must be real, not links
    keep |= {str(parent) for p in special for parent in Path(p).parents}
    if dst.exists():
        subprocess.run(["rm", "-rf", str(dst)], check=True)
    dst.mkdir(parents=True)
    for root, dirs, files in os.walk(src):
        here = Path(root).relative_to(src)
        for name in list(dirs):
            r = str(here / name)
            if r not in keep:
                os.symlink(Path(root) / name, dst / r)
                dirs.remove(name)  # linked whole
            else:
                (dst / r).mkdir(exist_ok=True)
        for name in files:
            r = str(here / name)
            if r not in special:
                os.symlink(Path(root) / name, dst / r)
    for r in sorted(restore):
        (dst / r).parent.mkdir(parents=True, exist_ok=True)
        content = subprocess.run(["git", "-C", str(REPO), "show", f"{NVIDIA_IMPORT}:baseline/{rel}/{r}"],
                                 check=True, capture_output=True).stdout
        (dst / r).write_bytes(content)
    marker.write_text(json.dumps(diff, indent=1, sort_keys=True) + "\n")
    return dst
