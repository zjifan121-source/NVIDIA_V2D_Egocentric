"""Phases of one object's clip: rest before the hands reach it, the carry, rest after (V3_BUILD_SPEC.md
S4a). A pure function of wrist and object tracks, used by s4a_phases and, as a helper, by S3a (view
selection) and S3c (rest anchors).

near(t): a valid wrist within REACH_CM of the object surface, measured in the object frame with a
k-d tree over the mesh vertices, as run_load_retarget.reach_frame does. first_reach and last_release
are the first and last near frames themselves (ep41: 19, as P3 reports); the windows come from near
dilated by DILATE frames on both sides:
    rest_before = [0, first_reach - DILATE - 1]
    carry       = [first_reach - DILATE, last_release + DILATE]
    rest_after  = [last_release + DILATE + 1, N - 1]
Windows are inclusive [first, last] frame pairs, None when empty. An object no wrist ever reaches is
at rest for the whole clip. static: the object's position never leaves its frame-0 position by
STATIC_CM or more (the carry-length bar does not apply to it).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

REACH_CM = 10.0  # default: run_load_retarget.reach_frame's and P3's reach distance
DILATE = 5  # default
STATIC_CM = 2.0  # default
MIN_CARRY = 10  # frames; acceptance bar for an object that moves


def near_frames(object_to_world: np.ndarray, vertices: np.ndarray, wrists: list[tuple[np.ndarray, np.ndarray]],
                cm: float = REACH_CM) -> np.ndarray:
    """(N,) bool: some valid wrist is within `cm` of the surface. object_to_world (N, 4, 4); vertices
    (V, 3) in the object frame, metres; wrists: (positions (N, 3) in world, valid (N,)) per hand."""
    T = np.asarray(object_to_world, dtype=np.float64)
    tree = cKDTree(np.asarray(vertices, dtype=np.float64))
    near = np.zeros(len(T), dtype=bool)
    for position, valid in wrists:
        local = np.einsum("tji,tj->ti", T[:, :3, :3], np.asarray(position, dtype=np.float64) - T[:, :3, 3])
        near |= (tree.query(local)[0] < cm / 100) & np.asarray(valid, dtype=bool)
    return near


def displacement_cm(object_to_world: np.ndarray) -> float:
    """Largest distance of the object's position from its frame-0 position, in cm."""
    p = np.asarray(object_to_world, dtype=np.float64)[:, :3, 3]
    return float(np.linalg.norm(p - p[0], axis=1).max() * 100)


def window(a: int, b: int) -> list[int] | None:
    return [int(a), int(b)] if a <= b else None


def phases(near: np.ndarray, object_to_world: np.ndarray, dilate: int = DILATE, static_cm: float = STATIC_CM) -> dict:
    """phases.json content from the near mask and the object track."""
    n = len(near)
    moved = displacement_cm(object_to_world)
    out = {"frames": n, "dilate": dilate, "static_cm": static_cm, "displacement_cm": round(moved, 2),
           "static": bool(moved < static_cm), "near_frames": int(np.count_nonzero(near))}
    hits = np.flatnonzero(near)
    if len(hits) == 0:
        return out | {"first_reach": None, "last_release": None, "rest_before": window(0, n - 1), "carry": None,
                      "rest_after": None}
    first, last = int(hits[0]), int(hits[-1])
    start, end = max(0, first - dilate), min(n - 1, last + dilate)
    return out | {"first_reach": first, "last_release": last, "rest_before": window(0, start - 1),
                  "carry": window(start, end), "rest_after": window(end + 1, n - 1)}


def length(w: list[int] | None) -> int:
    return 0 if w is None else w[1] - w[0] + 1


def from_bundle(data: dict, vertices: np.ndarray, pose_key: str = "object_to_world_transform",
                cm: float = REACH_CM, dilate: int = DILATE) -> dict:
    """phases() for one object of a bundle (BUNDLES.md layout: hand_{side}_wrist_to_world_transform,
    hand_{side}_is_valid); vertices metric, in the object frame (mesh vertices x object scale)."""
    wrists = [(data[f"hand_{side}_wrist_to_world_transform"][:, :3, 3], data[f"hand_{side}_is_valid"])
              for side in ("left", "right")]
    T = data[pose_key]
    return phases(near_frames(T, vertices, wrists, cm), T, dilate)


def rest_window_2d(object_masks: list, hand_masks: list, cap: int = 30, dilate_px: int = 25) -> list[int]:
    """[0, last] opening frames before any hand mask, dilated by dilate_px, touches an object mask,
    capped at `cap` frames (P3's opening-frames rule). Stands in for S4a's rest_before in the stages
    that run before the object is tracked (P1c, P3). masks: per track, a list of (H, W) bool per frame."""
    import cv2
    n = min(len(m) for m in object_masks + hand_masks)
    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    for t in range(min(n, cap)):
        hands = np.zeros_like(object_masks[0][t], dtype=np.uint8)
        for h in hand_masks:
            hands |= h[t].astype(np.uint8)
        near = cv2.dilate(hands, kernel) > 0
        if any((near & o[t]).any() for o in object_masks):
            return [0, t - 1] if t > 0 else [0, 0]
    return [0, min(n, cap) - 1]
