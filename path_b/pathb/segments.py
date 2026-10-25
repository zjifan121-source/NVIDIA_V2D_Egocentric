"""Contact segmentation of one object's clip (V4_BUILD_SPEC.md S3b): static, grasp and unstable
segments, from what the camera sees of the object and the hands' fingertips; pure functions.

Inputs per frame t: the object's observed points in the world (pathb/observe.py; None when there is
no mask or depth), the fingertips (N, 2, 5, 3) and wrists (N, 2, 4, 4) of P1d's world exports
(pathb/hands.py: hands L, R) with their valid flags, and each hand's long gaps (hand_gaps.json).
  motion(t)  median nearest-point distance between the clouds at t - WINDOW/2 and t + WINDOW/2 (both
             ways): how far the object's surface moved over WINDOW frames
  d_tip      per fingertip, the distance to the nearest object point
Thresholds from the data (spec): `near` = 2 x the median, over certain-contact frames, of the
grasping hand's second-closest fingertip (grasp needs two); certain contact = the object's centroid
moves >= CARRY_M per frame in the direction its wrist moves (cosine >= CARRY_COS), the hand carrying
it. Fewer than MIN_CERTAIN such frames: near = NEAR_DEFAULT. far = 2 x near.
Votes per frame: static = motion < STATIC_M and no fingertip of either hand within far; grasp (one
hand) = >= 2 of its fingertips within near (both hands: the one with more, then the closer); per grasp
run (carrying): static if the object stays still, else the hand the object moves most rigidly with;
unstable otherwise; a frame without object points keeps the previous vote. Hysteresis: the label
changes only where HYSTERESIS consecutive votes agree on the new one. Then grasp runs shorter than
MIN_GRASP become unstable, static runs shorter than MIN_STATIC join their neighbours, and a grasp is
split where its hand has a long gap (> 15 frames: those frames become unstable). A grasp's fingertip
set is the fingers of its hand within near on >= half its frames (at least the two most often).
An object never grasped is one static segment (static: true).
Support of a static segment (the observed points pooled over it): `table` when their lowest 5 %
are within SUPPORT_M of z = 0; else the other object whose footprint holds this one's centre and
whose top reaches within SUPPORT_M of this one's bottom; else `table`, flagged `floating`.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

HANDS = ("L", "R")
WINDOW = 10  # frames (spec)
STATIC_M = 0.005  # spec: 5 mm over the window
NEAR_DEFAULT = 0.025  # m (spec default)
CARRY_M = 0.01  # m per frame (spec)
CARRY_COS = 0.8  # default: "in the same direction as the wrist"
MIN_CERTAIN = 10  # spec
MIN_GRASP = 5  # frames (spec)
STILL_SHARE = 0.8  # default: a grasp run whose object is still on this share of its frames is static
ELIGIBLE_SHARE = 0.3  # default: a hand counts for a grasp run with two fingertips near on this share
MIN_STATIC = 5  # frames (spec)
HYSTERESIS = 3  # frames (spec)
SUPPORT_M = 0.015  # spec: 1.5 cm
LONG_GAP = 15  # frames: P1d's tracks_interpolate limit (hands.MAX_GAP)
QUERY = 2000  # points per frame queried for motion (spec: <= 2 000)


def centroids(clouds: list) -> np.ndarray:
    return np.array([np.median(c, axis=0) if c is not None and len(c) else [np.nan] * 3 for c in clouds])


def motion(clouds: list, window: int = WINDOW, seed: int = 0) -> np.ndarray:
    """(N,) metres; NaN where either cloud is missing."""
    n, h, rng = len(clouds), window // 2, np.random.default_rng(seed)
    out = np.full(n, np.nan)
    trees = [cKDTree(c) if c is not None and len(c) else None for c in clouds]
    for t in range(n):
        a, b = max(0, t - h), min(n - 1, t + h)
        if a == b or trees[a] is None or trees[b] is None:
            continue
        def one_way(src, tree):
            q = src if len(src) <= QUERY else src[rng.choice(len(src), QUERY, replace=False)]
            return np.median(tree.query(q)[0])
        out[t] = 0.5 * (one_way(clouds[a], trees[b]) + one_way(clouds[b], trees[a]))
    return out


def tip_distances(clouds: list, tips: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(N, 2, 5) metres from each fingertip to the nearest object point; NaN without points or hand."""
    n = len(clouds)
    out = np.full((n, 2, 5), np.nan)
    for t, c in enumerate(clouds):
        if c is None or not len(c):
            continue
        tree = cKDTree(c)
        for h in range(2):
            if valid[t, h] and np.isfinite(tips[t, h]).all():
                out[t, h] = tree.query(tips[t, h])[0]
    return out


def certain_contact(cent: np.ndarray, wrist: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(N,) the hand carrying the object at frame t (0 = L, 1 = R), or -1: the centroid moves >= CARRY_M
    from t to t+1 and that hand's wrist moves the same way (cosine >= CARRY_COS)."""
    n = len(cent)
    out = np.full(n, -1)
    for t in range(n - 1):
        dc = cent[t + 1] - cent[t]
        if not np.isfinite(dc).all() or np.linalg.norm(dc) < CARRY_M:
            continue
        best = CARRY_COS
        for h in range(2):
            if not (valid[t, h] and valid[t + 1, h]):
                continue
            dw = wrist[t + 1, h, :3, 3] - wrist[t, h, :3, 3]
            c = float(dc @ dw / (np.linalg.norm(dc) * np.linalg.norm(dw) + 1e-12))
            if c >= best:
                best, out[t] = c, h
    return out


def thresholds(d_tip: np.ndarray, carrier: np.ndarray) -> dict:
    """near and far (m), from the certain-contact frames' second-closest fingertip of the carrier."""
    second = [np.sort(d_tip[t, h])[1] for t, h in enumerate(carrier) if h >= 0 and np.isfinite(d_tip[t, h]).all()]
    if len(second) >= MIN_CERTAIN:
        near, source = 2 * float(np.median(second)), f"2 x median over {len(second)} certain-contact frames"
    else:
        near, source = NEAR_DEFAULT, f"default ({len(second)} certain-contact frames < {MIN_CERTAIN})"
    return {"near_m": near, "far_m": 2 * near, "certain_frames": len(second), "source": source}


def votes(mot: np.ndarray, d_tip: np.ndarray, near: float, far: float) -> list[str | None]:
    """Per frame: 'static', 'grasp:L', 'grasp:R', 'unstable', or None (no object points)."""
    out = []
    for t in range(len(mot)):
        d = d_tip[t]
        if not np.isfinite(d).any() and not np.isfinite(mot[t]):
            out.append(None)
            continue
        within = [int(np.sum(d[h] < near)) if np.isfinite(d[h]).all() else 0 for h in range(2)]
        if max(within) >= 2:
            if within[0] == within[1]:
                h = int(np.nanmean(d[1]) < np.nanmean(d[0]))
            else:
                h = int(within[1] > within[0])
            out.append(f"grasp:{HANDS[h]}")
        elif np.isfinite(mot[t]) and mot[t] < STATIC_M and not (np.nan_to_num(d, nan=np.inf) < far).any():
            out.append("static")
        else:
            out.append("unstable")
    return out


def hysteresis(v: list[str | None], k: int = HYSTERESIS) -> list[str]:
    """Labels that change only where k consecutive votes agree on the new one (None keeps the vote)."""
    filled, last = [], None
    for x in v:
        last = x if x is not None else last
        filled.append(last)
    first = next((x for x in filled if x is not None), "unstable")
    filled = [x if x is not None else first for x in filled]
    out, state = [], filled[0]
    for t, x in enumerate(filled):
        if x != state and all(y == x for y in filled[t:t + k]) and len(filled[t:t + k]) == k:
            state = x
        out.append(state)
    return out


def runs(labels: list[str]) -> list[list]:
    """[[label, start, end], ...] inclusive."""
    out = []
    for t, x in enumerate(labels):
        if out and out[-1][0] == x:
            out[-1][2] = t
        else:
            out.append([x, t, t])
    return out


def merge(rs: list[list]) -> list[list]:
    out = []
    for r in rs:
        if out and out[-1][0] == r[0]:
            out[-1][2] = r[2]
        else:
            out.append(list(r))
    return out


def tidy(labels: list[str]) -> list[list]:
    """Short grasps -> unstable; short static runs join their longer neighbour."""
    rs = [[("unstable" if x.startswith("grasp") and b - a + 1 < MIN_GRASP else x), a, b] for x, a, b in runs(labels)]
    rs = merge(rs)
    changed = True
    while changed and len(rs) > 1:
        changed = False
        for i, (x, a, b) in enumerate(rs):
            if x == "static" and b - a + 1 < MIN_STATIC:
                left, right = rs[i - 1] if i > 0 else None, rs[i + 1] if i + 1 < len(rs) else None
                pick = max((r for r in (left, right) if r), key=lambda r: r[2] - r[1])
                rs[i][0] = pick[0]
                rs, changed = merge(rs), True
                break
    return rs


def split_at_gaps(rs: list[list], long_gaps: dict[str, list[tuple[int, int]]]) -> list[list]:
    """A grasp's frames inside a long gap of its hand become unstable; leftovers shorter than MIN_GRASP too."""
    out = []
    for x, a, b in rs:
        if not x.startswith("grasp"):
            out.append([x, a, b])
            continue
        hand = x.split(":")[1]
        labels = [x] * (b - a + 1)
        for g0, g1 in long_gaps.get(hand, []):
            for t in range(max(a, g0), min(b, g1) + 1):
                labels[t - a] = "unstable"
        for y, s, e in runs(labels):
            out.append([("unstable" if y.startswith("grasp") and e - s + 1 < MIN_GRASP else y), a + s, a + e])
    return merge(out)


def spread(cent: np.ndarray, wrist: np.ndarray, valid: np.ndarray, h: int, a: int, b: int) -> float | None:
    """RMS scatter (m) of the object's centroid in hand h's wrist frame over [a, b]: ~0 when the object
    moves rigidly with that wrist."""
    fr = [t for t in range(a, b + 1) if valid[t, h] and np.isfinite(cent[t]).all()]
    if len(fr) < 3:
        return None
    local = np.array([wrist[t, h, :3, :3].T @ (cent[t] - wrist[t, h, :3, 3]) for t in fr])
    return float(np.sqrt(((local - local.mean(axis=0)) ** 2).sum(axis=1).mean()))


def carrying(rs: list[list], d_tip: np.ndarray, near: float, cent: np.ndarray, wrist: np.ndarray,
             valid: np.ndarray, mot: np.ndarray) -> list[list]:
    """Per grasp run: a run whose object stays still (motion < STATIC_M on >= STILL_SHARE of its frames)
    is static (a hand resting on it does not carry it); else its hand is the one, among the hands with
    two fingertips within near on >= ELIGIBLE_SHARE of its frames, the object moves most rigidly with
    (spread). Neighbouring runs that end up the same merge. ep41's pot, held by both hands in turn,
    came out as 11 short grasps whose hand often moved while the pot did not (a wrist moving 8.7 cm,
    the pot 2.1 cm)."""
    out = []
    for x, a, b in rs:
        if not x.startswith("grasp"):
            out.append([x, a, b])
            continue
        m = mot[a:b + 1]
        if np.mean(np.nan_to_num(m, nan=np.inf) < STATIC_M) >= STILL_SHARE:
            out.append(["static", a, b])
            continue
        near_tips = np.nan_to_num(d_tip[a:b + 1], nan=np.inf) < near  # (frames, 2, 5)
        eligible = [h for h in range(2) if np.mean(near_tips[:, h].sum(axis=1) >= 2) >= ELIGIBLE_SHARE]
        scored = {h: spread(cent, wrist, valid, h, a, b) for h in (eligible or [HANDS.index(x.split(":")[1])])}
        scored = {h: v for h, v in scored.items() if v is not None}
        hand = min(scored, key=scored.get) if scored else HANDS.index(x.split(":")[1])
        out.append([f"grasp:{HANDS[hand]}", a, b])
    return merge(out)


def fingertip_set(d_tip: np.ndarray, hand: int, a: int, b: int, near: float) -> list[list[str]]:
    share = np.nanmean(np.nan_to_num(d_tip[a:b + 1, hand], nan=np.inf) < near, axis=0)
    fingers = [f for f in range(5) if share[f] >= 0.5]
    if len(fingers) < 2:
        fingers = sorted(np.argsort(-share)[:2].tolist())
    return [[HANDS[hand], f] for f in fingers]


def segment(clouds: list, tips: np.ndarray, wrist: np.ndarray, valid: np.ndarray,
            long_gaps: dict[str, list[tuple[int, int]]]) -> tuple[list[dict], dict, dict]:
    """segments.json content (support filled by `support` later), the report, and the per-frame
    arrays (motion, d_tip, centroid: S3b saves them for S0.3)."""
    n = len(clouds)
    cent = centroids(clouds)
    mot = motion(clouds)
    d_tip = tip_distances(clouds, tips, valid)
    th = thresholds(d_tip, certain_contact(cent, wrist, valid))
    v = votes(mot, d_tip, th["near_m"], th["far_m"])
    labels = hysteresis(v)
    rs = split_at_gaps(tidy(labels), long_gaps)
    rs = carrying(rs, d_tip, th["near_m"], cent, wrist, valid, mot)
    never = not any(x.startswith("grasp") for x, _, _ in rs)
    if never:
        rs = [["static", 0, n - 1]]
    segs = []
    for x, a, b in rs:
        kind = x.split(":")[0]
        hand = x.split(":")[1] if kind == "grasp" else None
        segs.append({"kind": kind, "start": int(a), "end": int(b), "hand": hand,
                     "fingertips": fingertip_set(d_tip, HANDS.index(hand), a, b, th["near_m"]) if hand else [],
                     "support": None})
    report = {"frames": n, "thresholds": th, "static": never,
              "votes": {k: sum(1 for x in v if x == k) for k in ("static", "grasp:L", "grasp:R", "unstable", None)},
              "frames_without_points": int(sum(1 for c in clouds if c is None or not len(c))),
              "motion_mm_median": float(np.nanmedian(mot) * 1000) if np.isfinite(mot).any() else None,
              "segments": [[s["kind"], s["start"], s["end"], s["hand"]] for s in segs]}
    return segs, report, {"motion": mot, "d_tip": d_tip, "centroid": cent}


def pooled(clouds: list, a: int, b: int, limit: int = 20000, seed: int = 0) -> np.ndarray | None:
    pts = [c for c in clouds[a:b + 1] if c is not None and len(c)]
    if not pts:
        return None
    p = np.concatenate(pts)
    return p if len(p) <= limit else p[np.random.default_rng(seed).choice(len(p), limit, replace=False)]


def support(points: np.ndarray, others: dict[str, np.ndarray]) -> tuple[str, dict]:
    """What a static segment rests on, from its pooled points and the other objects' pooled points
    over the same frames. Returns (support, evidence)."""
    bottom = float(np.percentile(points[:, 2], 5))
    if bottom <= SUPPORT_M:
        return "table", {"bottom_cm": bottom * 100}
    centre = np.median(points[:, :2], axis=0)
    best = None
    for name, q in others.items():
        if q is None or not len(q):
            continue
        qc = np.median(q[:, :2], axis=0)
        radius = float(np.percentile(np.linalg.norm(q[:, :2] - qc, axis=1), 95))
        top = float(np.percentile(q[:, 2], 95))
        if np.linalg.norm(centre - qc) <= radius and top >= bottom - SUPPORT_M:
            gap = bottom - top
            if best is None or abs(gap) < abs(best[1]):
                best = (name, gap)
    if best:
        return best[0], {"bottom_cm": bottom * 100, "gap_to_support_top_cm": best[1] * 100}
    return "table", {"bottom_cm": bottom * 100, "floating": True}


def phases(segs: list[dict], n: int, cent: np.ndarray) -> dict:
    """phases.json in S4a's layout (pathb/phases.py), from the first and last grasp."""
    grasps = [s for s in segs if s["kind"] == "grasp"]
    statics = [s for s in segs if s["kind"] == "static"]
    moved = None
    if len(statics) >= 2:
        c0 = np.nanmedian(cent[statics[0]["start"]:statics[0]["end"] + 1], axis=0)
        c1 = np.nanmedian(cent[statics[-1]["start"]:statics[-1]["end"] + 1], axis=0)
        moved = float(np.linalg.norm(c1 - c0) * 100) if np.isfinite(c0).all() and np.isfinite(c1).all() else None
    out = {"frames": n, "source": "s3b_segments", "dilate": 0, "displacement_cm": moved,
           "displacement": "net: median centroid of the last static segment minus the first's",
           "static": not grasps, "near_frames": sum(s["end"] - s["start"] + 1 for s in grasps)}
    if not grasps:
        return out | {"first_reach": None, "last_release": None, "rest_before": [0, n - 1], "carry": None,
                      "rest_after": None}
    a, b = grasps[0]["start"], grasps[-1]["end"]
    return out | {"first_reach": a, "last_release": b, "rest_before": [0, a - 1] if a > 0 else None,
                  "carry": [a, b], "rest_after": [b + 1, n - 1] if b < n - 1 else None}
