"""Kinematic references of one object (V4_BUILD_SPEC.md S4b), as (N, 4, 4) object_to_world tracks.

  inverted  S3c's segment solution after S3d's settle (static segments held exactly), passed in as
            it is; absent until S3c has run
  keyframe  the rest-before anchor held until first_reach; over the carry, the track on trusted
            frames (IoU >= 0.5 and not recovered, from S3b's track_meta) and an interpolation
            between the nearest trusted frames elsewhere (position: cubic Hermite with zero end
            velocity; rotation: slerp); the rest-after anchor held from last_release. With no
            trusted frame the carry is one interpolation from anchor to anchor. The fallback when
            a grasp segment of S3c failed
  still     the rest-before anchor on every frame (the object of --still-object)
Built in that order; the per-frame provenance (held, trusted, interpolated, tracked, inverted) goes to
reference_report.json. v3's `tracking` candidate is gone (identical to keyframe on every v3 run).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

TRUST_IOU = 0.5  # default
CANDIDATES = ("inverted", "keyframe", "still")  # build order
RETARGETED = ("inverted",)  # the candidates S4c retargets (V4 S4c: one)
KINDS = ("held", "trusted", "interpolated", "tracked", "inverted")  # per-frame provenance


def still(n: int, anchor_before: np.ndarray) -> tuple[np.ndarray, list[str]]:
    return np.repeat(np.asarray(anchor_before, dtype=np.float64)[None], n, axis=0), ["held"] * n


def hermite(p0: np.ndarray, p1: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Cubic Hermite from p0 to p1 with zero velocity at both ends, at s in [0, 1]."""
    h = 3 * s ** 2 - 2 * s ** 3
    return p0[None] + h[:, None] * (p1 - p0)[None]


def keyframe(T: np.ndarray, phases: dict, anchor_before: np.ndarray, anchor_after: np.ndarray | None,
             trusted: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    T = np.asarray(T, dtype=np.float64)
    n = len(T)
    if phases["first_reach"] is None:  # never touched: at rest all clip
        return still(n, anchor_before)
    a, b = phases["first_reach"], phases["last_release"]
    after = anchor_after if anchor_after is not None and phases["rest_after"] is not None else None
    out, how = T.copy(), ["tracked"] * n
    out[:a], how[:a] = anchor_before, ["held"] * a
    if after is not None:
        out[b + 1:], how[b + 1:] = after, ["held"] * (n - b - 1)
    trusted = np.zeros(n, bool) if trusted is None else np.asarray(trusted, bool)
    # keys: the anchors just outside the carry, and the trusted carry frames
    keys = {a - 1: anchor_before} if a > 0 else {}
    keys.update({t: T[t] for t in range(a, b + 1) if trusted[t]})
    if after is not None and b + 1 < n:
        keys[b + 1] = after
    if not keys:  # nothing to interpolate from: hold the rest-before anchor
        keys = {a - 1: anchor_before}
    frames = sorted(keys)
    # without a rest-after anchor the frames after the release are held at the last key
    for t in range(a, (b if after is not None else n - 1) + 1):
        if trusted[t]:
            how[t] = "trusted"
            continue
        lo = max((f for f in frames if f < t), default=None)
        hi = min((f for f in frames if f > t), default=None)
        if lo is None or hi is None:  # past the last key (no rest-after): hold the nearest key
            out[t] = keys[lo if hi is None else hi]
            how[t] = "held"
            continue
        s = np.array([(t - lo) / (hi - lo)])
        out[t, :3, 3] = hermite(keys[lo][:3, 3], keys[hi][:3, 3], s)[0]
        out[t, :3, :3] = Slerp([0, 1], Rotation.from_matrix([keys[lo][:3, :3], keys[hi][:3, :3]]))(s).as_matrix()[0]
        how[t] = "interpolated"
    return out, how


def build(T: np.ndarray, phases: dict, anchor_before: np.ndarray, anchor_after: np.ndarray | None,
          trusted: np.ndarray | None = None,
          inverted: tuple[np.ndarray, list[str]] | None = None) -> dict[str, tuple[np.ndarray, list[str]]]:
    """The candidates in CANDIDATES order; `inverted` only when S3c's solution is given."""
    out = {"inverted": inverted} if inverted is not None else {}
    return out | {"keyframe": keyframe(T, phases, anchor_before, anchor_after, trusted),
                  "still": still(len(T), anchor_before)}


def provenance(how: list[str]) -> dict:
    """Frames per kind: held (anchor), trusted, interpolated, tracked, inverted."""
    return {k: how.count(k) for k in KINDS}
