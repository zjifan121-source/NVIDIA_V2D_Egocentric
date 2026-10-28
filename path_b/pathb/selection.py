"""S6 selection rules (V3_BUILD_SPEC.md S6): which candidate's rollout an episode submits.

Public: the official metrics against ground truth (through pathb/gt.py) per rollout; best AUC, ties
(within TIE_AUC) to the lower MPPE. Evaluation (no ground truth): each rollout against the observation,
the S4b keyframe reference on its held and trusted frames (rest anchors, and carry frames S3c tracked
with IoU >= 0.5 and no recovery), the scene aligned by the anchor object's frame-0 pose as the official
metric aligns it (the evaluator resets the objects at the task's frame 0, so this removes only the task
world's offset). Per frame: the observed objects' mean position error, symmetry-aware
(pathb/symmetry.py); a termination-equivalent frame (> 10 cm or > 0.5 rad) costs PENALTY_M. Lowest mean
cost wins; ties go to the earlier candidate. Every candidate's own reference would not do: the still
reference never moves, so its rollout always tracks it best.
Phase 2 (two objects, V3_BUILD_SPEC.md §8 "S6 selection adds RPE"): public ties go to the lower RPE before
MPPE; the observation cost also counts, on frames where both objects are observed, the error of the
second object's position relative to the first (capped like the rest).
Reference trust (V4 §13.6 F3, GT-free): the observation cost measures rollouts against our own inverted
reference, so it means nothing where that reference lost the object. `reference_trust` reads S3c's reports:
a rest whose posed mesh misses most of its mask's rays, or one S3c held because the fit lay beyond the hands'
reach, makes the reference untrusted; the episode then takes the still candidate (`pick_reference`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import symmetry

TIE_AUC = 0.001
PENALTY_M = 0.10  # termination-equivalent position error
MAX_ROT = 0.5  # rad
OBSERVED = ("held", "trusted", "inverted")  # S4b provenance with an observation behind it
RAYS_UNTRUSTED = 0.5  # default: a rest whose posed mesh misses this share of its mask's rays is not the object
# (public: worst rests at 0.84-0.95 on ep0, 11, 21, 22, where the inverted reference collapsed; <= 0.37 elsewhere)


def reference_trust(root: Path, names: list[str]) -> dict:
    """Whether the episode's inverted reference holds the objects, from S3c's own reports (no ground truth):
    {"trusted", "why", "rays_missed" (worst rest per object), "beyond_reach" (rests held, per object)}."""
    rays, held, why = {}, {}, []
    for name in names:
        f = Path(root) / f"s3c_invert/{name}/invert_report.json"
        rows = [r for r in json.loads(f.read_text())["segments"] if r["kind"] == "static"] if f.is_file() else []
        worst = max((r["rays_missed"] for r in rows if r.get("rays_missed") is not None), default=None)
        rays[name], held[name] = worst, [[r["start"], r["end"]] for r in rows if r.get("beyond_reach")]
        if worst is not None and worst > RAYS_UNTRUSTED:
            why.append(f"{name}: a rest misses {worst:.0%} of its mask's rays")
        if held[name]:
            why.append(f"{name}: {len(held[name])} rest(s) beyond the hands' reach")
    return {"trusted": not why, "why": "; ".join(why) or "every rest explains its mask and lies within reach",
            "rays_missed": rays, "beyond_reach": held}


def pick_reference(trust: dict) -> tuple[str, str]:
    """The reference an episode takes without ground truth: inverted, or still when it is untrusted."""
    return ("inverted", "the inverted reference is trusted") if trust["trusted"] else ("still", trust["why"])


def matrices(poses: np.ndarray) -> np.ndarray:
    """(n, k, 7) position + xyzw -> (n, k, 4, 4)."""
    T = np.tile(np.eye(4), poses.shape[:2] + (1, 1))
    T[..., :3, :3] = Rotation.from_quat(poses[..., 3:].reshape(-1, 4)).as_matrix().reshape(poses.shape[:2] + (3, 3))
    T[..., :3, 3] = poses[..., :3]
    return T


def observation_cost(rollout: np.ndarray, reference: np.ndarray, observed: np.ndarray, groups: list,
                     relative: bool = False, max_position: float = PENALTY_M, max_rotation: float = MAX_ROT) -> dict:
    """rollout, reference: (n, k, 4, 4) object poses; observed: (n, k) frames to score; groups: per
    object its (K, 4, 4) symmetry transforms; relative: add the objects' relative-position error."""
    A = reference[0, 0] @ np.linalg.inv(rollout[0, 0])  # anchor object's frame 0
    aligned = A @ rollout
    n, k = observed.shape
    cost = np.full((n, k), np.nan)
    terminated = np.zeros((n, k), bool)
    for t, j in zip(*np.nonzero(observed)):
        T, _ = symmetry.closest(reference[t, j], aligned[t, j], groups[j])
        pos = float(np.linalg.norm(T[:3, 3] - reference[t, j, :3, 3]))
        rot = symmetry.angle(reference[t, j, :3, :3], T[:3, :3])
        terminated[t, j] = pos > max_position or rot > max_rotation
        cost[t, j] = max_position if terminated[t, j] else pos
    terms = cost
    if relative and k > 1:
        rel = np.full((n, 1), np.nan)
        for t in np.flatnonzero(observed.all(axis=1)):
            d = np.linalg.norm((aligned[t, 1:, :3, 3] - aligned[t, :1, :3, 3]) - (reference[t, 1:, :3, 3] - reference[t, :1, :3, 3]),
                               axis=1).mean()
            rel[t, 0] = min(float(d), max_position)
        terms = np.concatenate([cost, rel], axis=1)
    frames = observed.any(axis=1)
    per_frame = np.nanmean(terms[frames], axis=1) if frames.any() else np.array([])
    return {"cost_m": float(per_frame.mean()) if len(per_frame) else None, "frames": int(frames.sum()),
            "termination_frames": int(terminated.any(axis=1).sum()),
            "first_termination_frame": int(np.flatnonzero(terminated.any(axis=1))[0]) if terminated.any() else None,
            "median_position_error_m": float(np.nanmedian(np.where(terminated, np.nan, cost))) if frames.any() else None}


def pick_public(scores: dict[str, dict], rpe: bool = False) -> tuple[str, str]:
    best = max(s["AUC"] for s in scores.values())
    tied = [c for c, s in scores.items() if s["AUC"] >= best - TIE_AUC]
    key = (lambda c: (scores[c]["RPE cm"], scores[c]["MPPE cm"])) if rpe else (lambda c: scores[c]["MPPE cm"])
    choice = min(tied, key=key)
    why = f"best AUC {scores[choice]['AUC']:.3f}" + (
        f" (tie with {[c for c in tied if c != choice]}: lower {'RPE, then MPPE' if rpe else 'MPPE'})" if len(tied) > 1 else "")
    return choice, why


def pick_evaluation(costs: dict[str, dict]) -> tuple[str, str]:
    usable = {c: v for c, v in costs.items() if v["cost_m"] is not None}
    if not usable:
        raise RuntimeError("no rollout has an observed frame to compare with")
    choice = min(usable, key=lambda c: usable[c]["cost_m"])  # dicts keep candidate order: ties -> earlier
    return choice, (f"lowest observation cost {usable[choice]['cost_m'] * 100:.2f} cm over {usable[choice]['frames']} "
                    f"frames ({usable[choice]['termination_frames']} termination-equivalent)")
