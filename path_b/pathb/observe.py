"""What the head camera sees of an object: P1b's mask x P2's depth (depth_a_half, ego_cam_a) x P4's
camera, as points in the world, per frame. The mask is eroded at half resolution as s3a2_fuse_scan
erodes it, so the mixed depth at its edge stays out. Shared by S3b (segments), S0.1 and S0.3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import read_depth, read_transforms


class View:
    """One episode's camera model, cameras and mask tracks; points(k, t) for object k at frame t."""

    def __init__(self, ctx):
        from .runner import Context  # noqa: F401 (ctx is a runner.Context)
        root = Path(ctx.root)
        intr = json.loads((root / "p2_depth/depth_a_half/intrinsics.json").read_text())
        self.ctx, self.root = ctx, root
        self.K = np.asarray(intr["K"], dtype=np.float64)
        self.shape = (intr["size"][1], intr["size"][0])
        self.c2w, self.have = read_transforms(root / "p4_export/camera_to_world", int(ctx.episode["num_frames"]))

    def points(self, k: int, t: int, limit: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray | None:
        """(M, 3) world points of object k at frame t (None: no mask, no depth or no camera); at most
        `limit` of them, drawn with `rng` (default: every point)."""
        from .stages.s3a2_fuse_scan import frame_masks
        path = self.root / f"p2_depth/depth_a_half/{t:06d}.png"
        if not (path.is_file() and self.have[t]):
            return None
        mask = frame_masks(self.ctx, k, t, self.shape)[0]
        if mask is None:
            return None
        depth = read_depth(path)
        ys, xs = np.nonzero(mask & (depth > 0) & np.isfinite(depth))
        if len(ys) == 0:
            return None
        if limit is not None and len(ys) > limit:
            i = (rng or np.random.default_rng(t)).choice(len(ys), limit, replace=False)
            ys, xs = ys[i], xs[i]
        z = depth[ys, xs]
        K = self.K
        cam = np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], axis=1)
        return cam @ self.c2w[t, :3, :3].T + self.c2w[t, :3, 3]
