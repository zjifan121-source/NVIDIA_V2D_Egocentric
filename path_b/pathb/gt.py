"""The only way into ground truth and the public scans (V3_BUILD_SPEC.md §7, the GT firewall).

Public ground truth (assets/data/track_3/public/data/*.parquet) and the scans (public/mesh/) are
read only here, only for public episodes, and every read is appended to the caller's `read` list,
which the runner writes to report.json: files_read. An evaluation episode raises
GroundTruthFirewall before anything is opened. The team's score_local and run_load_retarget do
the reading; this module decides whether they may. tests/test_gt_firewall.py checks that no other
pathb module calls them for ground truth or scans.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import TRACK3
from .team import retarget_tools, scorer

GT_DATA = TRACK3 / "public/data"
SCANS = TRACK3 / "public/mesh"


class GroundTruthFirewall(PermissionError):
    pass


def allowed(episode: dict) -> None:
    """Raise unless `episode` (its episodes.json entry) is public."""
    if episode.get("split") != "public":
        raise GroundTruthFirewall(f"episode {episode.get('episode')} is {episode.get('split')!r}: "
                                  "ground truth and scans are read for public episodes only")


def parquet(number: int) -> Path:
    return GT_DATA / f"data/chunk-000/episode_{number:06d}.parquet"


def scan(name: str) -> Path | None:
    return next((p for p in (SCANS / f"{name}/{name}_visual.glb", SCANS / f"{name}/{name}.glb") if p.is_file()),
                None)


def ground_truth(episode: dict, read: list) -> np.ndarray:
    """(frames, objects, 7) world_T_object, xyzw, in episodes.json object order."""
    allowed(episode)
    read.append(parquet(episode["episode"]))
    return scorer().ground_truth(episode["episode"])


def score(episode: dict, poses: np.ndarray, meshes: dict[str, Path], read: list) -> dict:
    """The six official metrics of one episode's (frames, objects, 7) poses with one mesh per object."""
    allowed(episode)
    read.append(parquet(episode["episode"]))
    read.extend(p for p in (scan(o["name"]) for o in episode["objects"]) if p)
    n = episode["episode"]
    return scorer().score([episode], {n: poses}, {(n, name): path for name, path in meshes.items()})


def scan_size_ratio(episode: dict, vertices: np.ndarray, name: str, read: list) -> float | None:
    """Reconstructed mesh size / public scan size (run_load_retarget.scan_size_ratio)."""
    allowed(episode)
    path = scan(name)
    if path is None:
        return None
    read.append(path)
    return retarget_tools().scan_size_ratio(vertices, name)


def scan_vertices(episode: dict, name: str, read: list) -> np.ndarray | None:
    import trimesh
    allowed(episode)
    path = scan(name)
    if path is None:
        return None
    read.append(path)
    return np.asarray(trimesh.load(path, force="mesh", process=False).vertices)


def rescale_by_scan(episode: dict, name: str, read: list) -> str:
    """Permission for run_load_retarget's --rescale-by-scan (cleaned_bundle(rescale_by_scan=name))."""
    allowed(episode)
    path = scan(name)
    if path:
        read.append(path)
    return name
