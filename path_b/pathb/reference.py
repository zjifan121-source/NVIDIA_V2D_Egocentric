"""Kinematic references of one object (V3_BUILD_SPEC.md S4b), as (N, 4, 4) object_to_world tracks.

  still     the rest-before anchor on every frame (the fallback; the object of --still-object)
  keyframe  the rest-before anchor held until first_reach; over the carry, the track on trusted
            frames (IoU >= 0.5 and not recovered, from S3b's track_meta) and an interpolation
            between the nearest trusted frames elsewhere (position: cubic Hermite with zero end
            velocity; rotation: slerp); the rest-after anchor held from last_release. With no
            trusted frame the carry is one interpolation from anchor to anchor
  tracking  the track, with each rest window replaced by its anchor and a 5-frame ease into and
            out of the track (run_load_retarget.hold_until, applied at both ends)
Built in that order (keyframe first, S0.2); the per-frame provenance (held, trusted, interpolated,
tracked, blended) goes to reference_report.json.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .team import retarget_tools

TRUST_IOU = 0.5  # default
BLEND = 5  # default: frames of ease-in/out (hold_until's blend)
CANDIDATES = ("keyframe", "tracking", "still")  # build order


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


def tracking(T: np.ndarray, phases: dict, anchor_before: np.ndarray,
             anchor_after: np.ndarray | None) -> tuple[np.ndarray, list[str]]:
    L = retarget_tools()
    T = np.asarray(T, dtype=np.float64)
    n = len(T)
    if phases["first_reach"] is None:
        return still(n, anchor_before)
    out, how = T.copy(), ["tracked"] * n
    if phases["rest_before"] is not None:
        end = phases["rest_before"][1] + 1
        out[:end] = anchor_before  # hold_until's rest pose then is exactly the anchor
        out = L.hold_until(out, end, blend=BLEND)
        how[:end] = ["held"] * end
        how[end:end + BLEND] = ["blended"] * len(how[end:end + BLEND])
    if phases["rest_after"] is not None and anchor_after is not None:
        start = phases["rest_after"][0]
        rev = out[::-1].copy()
        rev[:n - start] = anchor_after
        out = L.hold_until(rev, n - start, blend=BLEND)[::-1].copy()
        how[start:] = ["held"] * (n - start)
        lo = max(start - BLEND, 0)
        how[lo:start] = ["blended"] * (start - lo)
    return out, how


def build(T: np.ndarray, phases: dict, anchor_before: np.ndarray, anchor_after: np.ndarray | None,
          trusted: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, list[str]]]:
    """All three candidates, keyframe first."""
    return {"keyframe": keyframe(T, phases, anchor_before, anchor_after, trusted),
            "tracking": tracking(T, phases, anchor_before, anchor_after),
            "still": still(len(T), anchor_before)}


def provenance(how: list[str]) -> dict:
    """Frames per kind: held (anchor), trusted, interpolated, tracked, blended."""
    return {k: how.count(k) for k in ("held", "trusted", "interpolated", "tracked", "blended")}
