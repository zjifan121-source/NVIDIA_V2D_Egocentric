"""What the head camera sees of an object: P1b's mask x P2's depth (depth_a_half, ego_cam_a) x P4's
camera, as points in the world, per frame. The mask is eroded at half resolution as s3a2_fuse_scan
erodes it, so the mixed depth at its edge stays out. Shared by S3b (segments), S0.1 and S0.3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import read_depth, read_transforms

MIN_AREA = 0.01  # spec: the object's mask share of the image for a frame to count (S3a fuse, S3c unstable)
SMALL_AREA = 0.5  # default: a thin or small object's bar, as a share of its own 90th-percentile mask share


def area_bar(shares) -> float:
    """The spec's 1 % mask bar, or half the object's own 90th-percentile mask share when that is lower:
    ep18's mini sweeper lies in full view at 0.87-0.94 % of the image, so the spec bar kept no frame.
    shares: the object's mask share per frame (0 without a mask)."""
    s = np.asarray([x for x in shares if x and x > 0], dtype=np.float64)
    return float(min(MIN_AREA, SMALL_AREA * np.percentile(s, 90))) if len(s) else MIN_AREA


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

    def mask(self, k: int, t: int) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """(eroded mask, depth) of object k at frame t at depth resolution, and the mask's image share
        (None, None, 0 without a mask, depth or camera)."""
        from .stages.s3a2_fuse_scan import frame_masks
        path = self.root / f"p2_depth/depth_a_half/{t:06d}.png"
        if not (path.is_file() and self.have[t]):
            return None, None, 0.0
        mask, _, share = frame_masks(self.ctx, k, t, self.shape)
        if mask is None:
            return None, None, 0.0
        return mask, read_depth(path), share

    def allowed(self, k: int, t: int) -> np.ndarray | None:
        """Where object k may show at frame t, at depth resolution: its P1b mask (not eroded) or a hand's
        (a hand in front of it hides it). None without its mask."""
        import cv2
        from . import baseline as B
        tracks = B.mask_tracks(self.root)
        m = B.read_mask(tracks[f"obj_{k}"], t)
        if m is None:
            return None
        out = m.copy()
        for side in ("hand_L", "hand_R"):
            h = B.read_mask(tracks[side], t) if side in tracks else None
            if h is not None:
                out |= h
        return cv2.resize(out.astype(np.uint8), self.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0

    def pixels(self, k: int, t: int, limit: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float] | None:
        """(uv (M, 2) at depth resolution, observed depth (M,), mask share) of object k's valid-depth
        mask pixels at frame t, at most `limit` of them; None without any."""
        mask, depth, share = self.mask(k, t)
        if mask is None:
            return None
        ys, xs = np.nonzero(mask & (depth > 0) & np.isfinite(depth))
        if len(ys) == 0:
            return None
        if len(ys) > limit:
            i = rng.choice(len(ys), limit, replace=False)
            ys, xs = ys[i], xs[i]
        return np.stack([xs, ys], axis=1).astype(np.float64), depth[ys, xs].astype(np.float64), share

    def masked_depth(self, k: int, t: int) -> np.ndarray | None:
        """Frame t's depth with everything outside object k's eroded mask set to 0 (no depth)."""
        mask, depth, _ = self.mask(k, t)
        if mask is None:
            return None
        depth = depth.copy()
        depth[~mask] = 0.0
        return depth

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
