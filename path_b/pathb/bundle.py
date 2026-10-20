"""Copies of reconstruction bundles with every length rescaled (stage 2's rescale_world)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .team import retarget_tools


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
