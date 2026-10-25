"""P1d's v4 exports: the hands per frame in the world, and the hand-gap rule (V4_BUILD_SPEC.md S1 P1d).

The arrays are in the table world (z = 0 on the table), which P4 defines after P1d has run, so P4
writes them from its bundle into p4_export/hands/ (decision 2026-10-24):
  fingertips_world.npz  tips (N, 2, 5, 3): hands L, R; thumb, index, middle, ring, pinky (MANO joints
                        4, 8, 12, 16, 20)
  hand_verts_world.npz  verts (N, 2, 778, 3) float32 and faces (2, 1552, 3)
  wrist_world.npz       wrist (N, 2, 4, 4): MANO's root rotation at joint 0 (the wrist), world
each with valid (N, 2) and filled (N, 2); frames without a hand are NaN. hand_gaps.json lists, per
hand, every run of frames without a measured pose: `filled` (P1d's tracks_interpolate filled it, so
at most MAX_GAP frames between two measured frames) or `long`.
The world hands come from the baseline loader's re-posing (drivers/mano_world.py in $RETARGET_VENV),
as S4c's do. Hand-gap rule: a filled gap's wrist is re-interpolated in the world, position by cubic
Hermite with end tangents from the measured neighbours (default: a carried hand keeps moving through
the gap; zero end velocity would stop it mid-carry), rotation by slerp; the frame's fingertips and
vertices move rigidly with its wrist. tracks_interpolate interpolates cam_t linearly in the moving
camera's frame. A long gap stays empty and listed: S3b splits a grasp segment there.
coverage(valid, segments): share of frames with a wrist, overall and inside each grasp segment
(S0.3's coverage_carry; S3b writes coverage.json).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .paths import PATH_B, REPO

MAX_GAP = 15  # frames; P1d's tracks_interpolate limit (the pipeline's interp_max_gap_frames)
HANDS = ("L", "R")
SIDES = ("left", "right")  # the bundle's names, in HANDS order
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIPS = (4, 8, 12, 16, 20)  # MANO joints, OpenPose order


def retarget_python() -> str:
    return f"{os.environ.get('RETARGET_VENV', '/root/v2d/venvs/retarget')}/bin/python"


def mano_dir() -> str:
    return os.environ.get("MANO_DIR", str(REPO / "baseline/reconstruction/data/weights/hand"))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of every run of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(np.diff(m.astype(int)))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def gaps(measured: np.ndarray, valid: np.ndarray) -> list[dict]:
    """Runs of frames without a measured pose. `filled`: every frame has a pose (tracks_interpolate's)
    and the run lies between two measured frames, at most MAX_GAP long; else `long` (leading and
    trailing runs too: nothing brackets them)."""
    measured, valid = np.asarray(measured, bool), np.asarray(valid, bool)
    n = len(measured)
    out = []
    for a, b in runs(~measured):
        inside = 0 < a and b < n - 1
        filled = inside and b - a + 1 <= MAX_GAP and bool(valid[a:b + 1].all())
        out.append({"start": a, "end": b, "frames": b - a + 1, "kind": "filled" if filled else "long"})
    return out


def hermite_fill(T: np.ndarray, measured: np.ndarray, gap_list: list[dict]) -> np.ndarray:
    """Wrists (N, 4, 4) with each filled gap re-interpolated between its measured neighbours: cubic
    Hermite position, tangents from the measured frame before / after each end (the chord where that
    frame is missing), slerp rotation."""
    T = np.asarray(T, np.float64).copy()
    measured = np.asarray(measured, bool)
    n = len(T)
    for g in gap_list:
        if g["kind"] != "filled":
            continue
        a, b = g["start"] - 1, g["end"] + 1  # the measured brackets
        p0, p1 = T[a, :3, 3], T[b, :3, 3]
        chord = (p1 - p0) / (b - a)  # per frame
        m0 = T[a, :3, 3] - T[a - 1, :3, 3] if a > 0 and measured[a - 1] else chord
        m1 = T[b + 1, :3, 3] - T[b, :3, 3] if b + 1 < n and measured[b + 1] else chord
        frames = np.arange(a + 1, b)
        s = (frames - a) / (b - a)
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        L = b - a  # tangents per frame -> per unit s
        T[frames, :3, 3] = (h00[:, None] * p0 + h10[:, None] * L * m0 + h01[:, None] * p1 + h11[:, None] * L * m1)
        T[frames, :3, :3] = Slerp([0, 1], Rotation.from_matrix([T[a, :3, :3], T[b, :3, :3]]))(s).as_matrix()
    return T


def moved(points: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """points (K, 3) of one frame carried from wrist `old` to wrist `new` (both 4x4)."""
    M = new @ np.linalg.inv(old)
    return points @ M[:3, :3].T + M[:3, 3]


def wrists(joints: np.ndarray, global_orient: np.ndarray) -> np.ndarray:
    """(N, 4, 4): rotation = MANO's root rotation (world rotvec), translation = joint 0."""
    T = np.tile(np.eye(4), (len(joints), 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(global_orient).as_matrix()
    T[:, :3, 3] = joints[:, 0]
    return T


def measured_frames(hand_dir: Path | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(measured, filled) per frame from P1d's hamer_aligned_filled/<track>/ records."""
    measured, filled = np.zeros(n, bool), np.zeros(n, bool)
    if hand_dir is None:
        return measured, filled
    for f in Path(hand_dir).glob("*.json"):
        if f.stem.isdigit() and int(f.stem) < n:
            (filled if json.loads(f.read_text()).get("interpolated") else measured)[int(f.stem)] = True
    return measured, filled


def build(mano: dict, flags: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """The export arrays from drivers/mano_world.py's output and P1d's per-side (measured, filled)
    flags; the hand-gap rule applied. Returns tips, verts, faces, wrist, valid, filled, gaps,
    hermite_shift_cm (per hand: the largest wrist move the re-interpolation made)."""
    n = len(mano["right_joints"])
    tips = np.full((n, 2, 5, 3), np.nan)
    verts = np.full((n, 2, 778, 3), np.nan, np.float32)
    wrist = np.full((n, 2, 4, 4), np.nan)
    valid, filled_out = np.zeros((n, 2), bool), np.zeros((n, 2), bool)
    faces, gap_lists, shift = [], {}, {}
    for h, (hand, side) in enumerate(zip(HANDS, SIDES)):
        measured, filled = flags[side]
        J = np.asarray(mano[f"{side}_joints"], np.float64)
        V = np.asarray(mano[f"{side}_vertices"], np.float64)
        ok = np.asarray(mano[f"{side}_valid"], bool) & np.isfinite(J).all(axis=(1, 2)) & (measured | filled)
        measured, filled = measured & ok, filled & ok
        W = wrists(J, np.asarray(mano[f"{side}_global_orient"], np.float64))
        g = gaps(measured, ok)
        Wn = hermite_fill(W, measured, g)
        tips_h, verts_h = J[:, TIPS].copy(), V.copy()
        moves = [0.0]
        for gap in (x for x in g if x["kind"] == "filled"):
            for t in range(gap["start"], gap["end"] + 1):
                tips_h[t] = moved(tips_h[t], W[t], Wn[t])
                verts_h[t] = moved(verts_h[t], W[t], Wn[t])
                moves.append(float(np.linalg.norm(Wn[t, :3, 3] - W[t, :3, 3])))
        keep = measured | np.isin(np.arange(n), [t for x in g if x["kind"] == "filled"
                                                  for t in range(x["start"], x["end"] + 1)])
        tips[keep, h], verts[keep, h], wrist[keep, h] = tips_h[keep], verts_h[keep], Wn[keep]
        valid[:, h], filled_out[:, h] = keep, keep & ~measured
        faces.append(np.asarray(mano[f"{side}_faces"]))
        gap_lists[hand], shift[hand] = g, max(moves) * 100
    return {"tips": tips, "verts": verts, "faces": np.stack(faces), "wrist": wrist, "valid": valid,
            "filled": filled_out, "gaps": gap_lists, "hermite_shift_cm": shift}


def export(bundle: Path, hand_dirs: tuple[Path | None, Path | None], out: Path, log: Path) -> dict:
    """Writes out/{fingertips_world,hand_verts_world,wrist_world}.npz and hand_gaps.json from one of
    P4's bundles (the hands are the same in every object's bundle). hand_dirs = (left, right) as
    bundle.hand_dirs gives them. Returns build()'s summary (no arrays)."""
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "mano_world.npz"
    source, scratch = bundle, None
    if not (Path(bundle) / "mesh.obj").is_file():  # P4's bundle has no mesh yet; the loader's scene
        # alignment opens one (mano_world.py undoes its shift), so a linked copy gets the writer's placeholder
        from .bundle import PLACEHOLDER_OBJ
        scratch = out / "_bundle"
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir()
        for f in Path(bundle).iterdir():
            (scratch / f.name).symlink_to(f.resolve())
        (scratch / "mesh.obj").write_text(PLACEHOLDER_OBJ)
        source = scratch
    with open(log, "a") as f:
        rc = subprocess.run([retarget_python(), str(PATH_B / "pathb/drivers/mano_world.py"), str(source), str(raw),
                             mano_dir()], stdout=f, stderr=subprocess.STDOUT).returncode
    if scratch is not None:
        shutil.rmtree(scratch, ignore_errors=True)
    if rc:
        raise RuntimeError(f"mano_world.py failed (exit {rc}): " + "\n".join(log.read_text().splitlines()[-15:]))
    mano = dict(np.load(raw))
    n = len(mano["right_joints"])
    r = build(mano, {side: measured_frames(d, n) for side, d in zip(SIDES, hand_dirs)})
    raw.unlink()
    common = {"valid": r["valid"], "filled": r["filled"], "hands": np.array(HANDS)}
    np.savez(out / "fingertips_world.npz", tips=r["tips"], fingers=np.array(FINGERS), **common)
    np.savez(out / "hand_verts_world.npz", verts=r["verts"], faces=r["faces"], **common)
    np.savez(out / "wrist_world.npz", wrist=r["wrist"], **common)
    (out / "hand_gaps.json").write_text(json.dumps({
        "max_gap_frames": MAX_GAP, "gaps": r["gaps"], "hermite_shift_cm": r["hermite_shift_cm"],
        "rule": "filled: re-interpolated in the world (Hermite position, slerp rotation); long: no hand, "
                "S3b splits a grasp there"}, indent=1) + "\n")
    return {k: r[k] for k in ("gaps", "hermite_shift_cm")} | {"valid_share": r["valid"].mean(axis=0).tolist()}


def coverage(valid: np.ndarray, segments: list[dict]) -> dict:
    """Share of frames with a wrist: per hand overall, and inside each grasp segment for its hand
    (`hand` L or R; a segment without a hand counts either hand). `coverage_carry` = over all grasp
    frames, the share with the grasping hand's wrist (S0.3)."""
    valid = np.asarray(valid, bool)
    out = {"overall": {h: float(valid[:, i].mean()) for i, h in enumerate(HANDS)}, "grasps": []}
    hit = total = 0
    for s in segments:
        if s.get("kind") != "grasp":
            continue
        cols = [HANDS.index(s["hand"])] if s.get("hand") in HANDS else [0, 1]
        v = valid[s["start"]:s["end"] + 1][:, cols].any(axis=1)
        out["grasps"].append({"start": s["start"], "end": s["end"], "hand": s.get("hand"), "share": float(v.mean())})
        hit, total = hit + int(v.sum()), total + len(v)
    out["coverage_carry"] = hit / total if total else None
    return out
