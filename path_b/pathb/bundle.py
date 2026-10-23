"""Copies of reconstruction bundles with every length rescaled (stage 2's rescale_world)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .team import retarget_tools


def bundle_at(root: Path, number: int, name: str) -> Path:
    """An episode's bundle for one object, as P4 exports it (BUNDLES.md layout); S3c completes it.
    root: the episode's run folder (<out>/episode_XXXXXX)."""
    return root / f"p4_export/episode_{number:06d}/{name}/result_slam_gravity_aligned"


def p4_bundle(ctx, name: str) -> Path:
    return bundle_at(ctx.root, ctx.number, name)


def rescaled_copy(src: Path, dest: Path, factor: float, link: bool = True,
                  skip: tuple[str, ...] = ()) -> Path:
    """dest/result.npz = src's with every length times `factor`; the other files are linked (or
    copied, link=False) unchanged: mesh vertices stay put and object_scale carries the factor."""
    data = dict(np.load(src / "result.npz"))
    retarget_tools().rescale_world(data, factor)
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name == "result.npz" or f.name in skip or (dest / f.name).exists():
            continue
        if link:
            (dest / f.name).symlink_to(f.resolve())
        elif f.is_dir():
            shutil.copytree(f, dest / f.name)
        else:
            shutil.copy2(f, dest / f.name)
    np.savez(dest / "result.npz", **data)
    return dest
