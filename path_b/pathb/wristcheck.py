"""S0.3 wrist check (V4_BUILD_SPEC.md S0.3, `run.py wristcheck`): is our wrist track steady enough for
S3c's hand-carries-object model? CPU, on S3b's outputs (no ground truth).

Windows: runs of >= MIN_FRAMES frames where the object is still (S3b's motion < segments.STATIC_M)
and a fingertip of one hand is within TOUCH_M of the object's points (that hand's wrist, per window).
Both are still, so the wrist-to-object-centroid distance should be constant: its standard deviation
per window is the wrist's error, and `sigma_cm` pools them (frame-weighted within-window variance).
More readings of the same windows, for the record: `sigma_vec_cm`, the wrist's 3-D scatter about the
object (the distance sees only its radial part; S3c's offset sees all of it); `sigma_contact_cm` on the frames a fingertip
is within S3b's `near` (touching, so the hand is surely still), and `jitter_cm`, the deviation from a
Gaussian-smoothed distance (sigma 3 frames): frame-to-frame noise without slow motion. coverage_carry:
the share of grasp frames with the grasping hand's wrist (S3b's coverage.json, frame-weighted).
Gate (spec): sigma <= 2 cm on both episodes -> build S3c as specified; 2-3 cm -> build it, expect the
carry bar to fail often; > 3 cm -> S3c is not built. coverage_carry < 0.9 -> the hand-gap rule is
mandatory. Out: <run>/wristcheck.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import segments as G
from .hands import HANDS

TOUCH_M = 0.05  # spec: a fingertip within 5 cm
MIN_FRAMES = 5  # default: shortest window
SMOOTH = 3.0  # frames, jitter's smoothing


def windows(motion: np.ndarray, d_tip: np.ndarray, near: float) -> list[dict]:
    """Runs of still-object frames with one hand's fingertip within TOUCH_M, per hand."""
    out = []
    still = np.nan_to_num(motion, nan=np.inf) < G.STATIC_M
    for h in range(2):
        closest = np.nanmin(np.where(np.isfinite(d_tip[:, h]), d_tip[:, h], np.inf), axis=1)
        for inside, a, b in G.runs([bool(x) for x in still & (closest < TOUCH_M)]):
            if inside and b - a + 1 >= MIN_FRAMES:
                out.append({"hand": HANDS[h], "start": int(a), "end": int(b),
                            "touching": [bool(c < near) for c in closest[a:b + 1]]})
    return out


def episode(root: Path, name: str) -> dict:
    """One object of one episode: its windows and their deviations."""
    seg = root / f"s3b_segments/{name}"
    arrays = np.load(seg / "per_frame.npz")
    near = json.loads((seg / "segments_report.json").read_text())["thresholds"]["near_m"]
    wrist = np.load(root / "p4_export/hands/wrist_world.npz")
    W, valid = wrist["wrist"], wrist["valid"]
    cent = arrays["centroid"]
    rows = []
    for w in windows(arrays["motion"], arrays["d_tip"], near):
        h, a, b = HANDS.index(w["hand"]), w["start"], w["end"]
        ok = valid[a:b + 1, h] & np.isfinite(cent[a:b + 1]).all(axis=1)
        if ok.sum() < MIN_FRAMES:
            continue
        rel = (W[a:b + 1, h, :3, 3] - cent[a:b + 1])[ok]
        dist = np.linalg.norm(rel, axis=1)
        touch = np.asarray(w["touching"])[ok]
        smooth = gaussian_filter1d(dist, SMOOTH, mode="nearest")
        rows.append({**{k: w[k] for k in ("hand", "start", "end")}, "frames": int(ok.sum()),
                     "sigma_cm": float(np.std(dist) * 100), "mean_cm": float(np.mean(dist) * 100),
                     "jitter_cm": float(np.std(dist - smooth) * 100),
                     "sigma_vec_cm": float(np.sqrt(((rel - rel.mean(axis=0)) ** 2).sum(axis=1).mean()) * 100),
                     "touching_frames": int(touch.sum()),
                     "sigma_contact_cm": float(np.std(dist[touch]) * 100) if touch.sum() >= MIN_FRAMES else None})
    cov = json.loads((seg / "coverage.json").read_text())
    grasp_frames = sum(g["end"] - g["start"] + 1 for g in cov["grasps"])
    return {"object": name, "near_cm": near * 100, "windows": rows, "coverage_carry": cov["coverage_carry"],
            "grasp_frames": grasp_frames}


def pooled(rows: list[dict], key: str) -> float | None:
    """Frame-weighted within-window standard deviation (cm)."""
    rows = [r for r in rows if r.get(key) is not None]
    n = sum(r["frames"] for r in rows)
    return float(np.sqrt(sum(r["frames"] * r[key] ** 2 for r in rows) / n)) if n else None


def gate(sigma: float | None) -> str:
    if sigma is None:
        return "no window: the check cannot decide"
    if sigma <= 2.0:
        return "build S3c as specified"
    if sigma <= 3.0:
        return "build S3c; expect the carry bar to fail often (keyframe fallback carries the weight)"
    return "do not build S3c: v4 keeps the static/settle gains, the time goes to S6"


def main(run: Path, numbers: list[int]) -> Path:
    from .paths import episode_info
    eps = []
    for number in numbers:
        root = run / f"episode_{number:06d}"
        objs = [episode(root, o["name"]) for o in episode_info(number)["objects"]]
        rows = [r for o in objs for r in o["windows"]]
        weighted = [(o["coverage_carry"], o["grasp_frames"]) for o in objs if o["coverage_carry"] is not None]
        carry = sum(c * n for c, n in weighted) / sum(n for _, n in weighted) if weighted else None
        eps.append({"episode": number, "objects": objs, "sigma_cm": pooled(rows, "sigma_cm"),
                    "sigma_contact_cm": pooled(rows, "sigma_contact_cm"), "jitter_cm": pooled(rows, "jitter_cm"),
                    "sigma_vec_cm": pooled(rows, "sigma_vec_cm"),
                    "window_frames": sum(r["frames"] for r in rows), "coverage_carry": carry})
    sigmas = [e["sigma_cm"] for e in eps]
    worst = None if any(s is None for s in sigmas) else max(sigmas)
    carries = [e["coverage_carry"] for e in eps if e["coverage_carry"] is not None]
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": run.name,
           "episodes": eps, "sigma_cm_worst": worst, "decision": gate(worst),
           "hand_gap_rule": ("mandatory (coverage_carry < 0.9)" if carries and min(carries) < 0.9 else "optional"),
           "settings": {"touch_m": TOUCH_M, "min_frames": MIN_FRAMES, "static_m": G.STATIC_M, "smooth_frames": SMOOTH}}
    path = run / "wristcheck.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path
