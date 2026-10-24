"""Dev-episode analysis page (James, 2026-10-22, step 2): where the score goes on the dev episodes
(V4: ep41 and ep13, paths.DEV_EPISODES, the default when no episode is named).

    uv run --project path_b python path_b/run.py analysis --out <run> [--episodes 41 13] [--tag v2]

Writes path_b/reports/dev_analysis_<date>[_<tag>].md and the self-contained .html made from it
(inline SVG charts, light and dark). Sections, in James's order: (a) headline AUC per episode against
the S0.2 oracle; (b) per-frame error timelines per object with S4a's phases and S3b's recovered
frames; (c) loss attribution by stage from the waterfall; (d) mesh panel; (e) camera panel;
(f) biggest flaws. Every number against ground truth goes through pathb/gt.py (official metric code);
public episodes only. Stand-in stages (run.py standin) are labelled wherever their numbers appear.
"""
from __future__ import annotations

import base64
import html
import json
import re
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import trimesh

from . import gt
from .paths import DEV_EPISODES, PATH_B, episode_info
from .reference import CANDIDATES, RETARGETED
from .runner import Context
from .team import scorer
from .waterfall import SCAN, merged, raw_scene, scene

ORACLE = PATH_B / "oracles/oracle_2026-10-17.json"
REPORTS = PATH_B / "reports"
ORACLE_LINES = ("still_gt0", "place_pos_gt", "place_gt_when_resting", "place_gt_when_moving")
CANDIDATE_COLUMNS = (*[f"rollout_{c}" for c in CANDIDATES], *[f"realised_{c}" for c in RETARGETED],
                     *[f"ref_{c}" for c in CANDIDATES if c != "still"])
STAGES_IN = ("p1b_masks", "p1c_camera", "p1d_hands", "s3a_mesh", "s3b_track")


# ---------------------------------------------------------------- data
def oracle(number: int) -> tuple[dict, dict]:
    rows = [r for r in json.loads(ORACLE.read_text())["variants"] if r["ep"] == number]
    extra = next((r["extra"] for r in rows if r.get("extra") and "rest_common_rms_cm" in r["extra"]), {})
    return {r["v"]: r for r in rows}, extra


def stand_ins(root: Path) -> list[str]:
    return [s for s in STAGES_IN if (root / s / "report.json").is_file()
            and json.loads((root / s / "report.json").read_text()).get("stand_in")]


def best_candidate(scores: dict) -> str | None:
    """The candidate an episode would submit today: the best rollout, else realised, else reference."""
    for prefix in ("rollout_", "realised_", "ref_"):
        cols = [c for c in CANDIDATE_COLUMNS if c.startswith(prefix) and c in scores]
        if cols:
            return max(cols, key=lambda c: scores[c]["AUC"])
    return None


def variant_poses(root: Path, variant: str, names: list[str], frames: int, work: Path):
    """(poses, meshes) of a waterfall column, rebuilt as the waterfall scores it."""
    if variant == "raw":
        return raw_scene(root, names, frames)
    if variant.startswith("rollout_"):  # S6's rollout with S5's runner's meshes, as the waterfall scores it
        rollout = root / f"s6_eval/{variant[8:]}/results/{root.name}.parquet"
        mesh_dir = root / f"s5_task/{variant[8:]}/run/score_input/meshes/{root.name}"
        if not (rollout.is_file() and mesh_dir.is_dir()):
            return None
        poses = scorer().evaluator_poses(rollout, episode_info(int(root.name[8:])))
        return poses, {n: next(mesh_dir.glob(f"{n}.*")) for n in names}
    if variant.startswith("ref_"):
        bundles = {n: root / f"s4b_reference/{n}/{variant[4:]}" for n in names}
    elif variant.startswith("realised_"):
        bundles = {n: root / f"s4c_retarget/{n}/{variant[9:]}/bundle" for n in names}
    elif variant == "fused":
        from .bundle import bundle_at
        bundles = {n: bundle_at(root, int(root.name[8:]), n) for n in names}
    else:
        return None
    if not all((b / "result.npz").is_file() for b in bundles.values()):
        return None
    bundle = bundles[names[0]] if len(names) == 1 else merged(int(root.name[8:]), bundles, work / variant)
    return scene(bundle, names, frames, work, f"analysis_{variant}")


def frame_errors(root: Path, e: dict, variants: list[str], read: list) -> dict:
    names, frames = [o["name"] for o in e["objects"]], int(e["num_frames"])
    work = root / "waterfall"
    out = {}
    for v in variants:
        pm = variant_poses(root, v, names, frames, work)
        if pm is not None:
            out[v] = gt.official_frame_errors(e, pm[0], pm[1], read)
    return out


def recovered_frames(root: Path, name: str) -> list[int] | None:
    f = root / f"s3b_track/{name}/track_meta.json"
    if not f.is_file():
        return None
    return [r["frame"] for r in json.loads(f.read_text())["frames"] if r.get("recovered")]


def mesh_panel(root: Path, e: dict, name: str, read: list) -> dict:
    m = trimesh.load(root / f"s3a_mesh/{name}/mesh.obj", force="mesh", process=False)
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    placed = gt.scan_in_frame(e, name, v, f, read)
    out = {"cdo_cm": gt.cdo_cm(e, name, v, f, read), "size_ratio": gt.scan_size_ratio(e, v, name, read)}
    if placed is not None:
        scan_mesh = trimesh.Trimesh(*placed, process=False)
        pts = scan_mesh.sample(20000, seed=0)
        d = trimesh.Trimesh(v, f, process=False).nearest.on_surface(pts)[1] if len(f) else np.full(len(pts), np.inf)
        out["coverage_1cm"] = float(np.mean(d <= 0.01))
        out["silhouettes"] = silhouettes(v, f, *placed)
    return out


def camera_panel(root: Path, e: dict, extra: dict) -> dict:
    from .geometry import read_transforms
    from .stages.p1c_camera import plane_heights
    n = int(e["num_frames"])
    c2w, ok = read_transforms(root / "p4_export/camera_to_world", n)
    p = c2w[ok, :3, 3]
    scale = json.loads((root / "p3_scale_bridge/scale.json").read_text())
    rest = list(range(scale["rest_window"][0], scale["rest_window"][1] + 1))
    heights, normal_spread = plane_heights(Context(e, root, {}), c2w, rest)
    return {"path_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), "rest_window": scale["rest_window"],
            "plane_height_spread_cm": float(np.ptp(heights) * 100) if len(heights) else None,
            "plane_normal_spread_deg": normal_spread, "scale_factor": scale["factor"],
            "rest_common_rms_cm": extra.get("rest_common_rms_cm"), "rest_diff_rms_cm": extra.get("rest_diff_rms_cm")}


# ---------------------------------------------------------------- drawing (inline SVG / PNG)
def esc(x) -> str:
    return html.escape(str(x))


def bar_chart(bars: list[tuple[str, float, str]], refs: list[tuple[str, float]], width: int = 680) -> str:
    """Horizontal AUC bars (0-1) with the oracle ceilings as labelled vertical lines."""
    left, right, bh, gap, top = 190, 30, 22, 8, 52
    h = top + len(bars) * (bh + gap) + 34
    x = lambda v: left + (width - left - right) * max(0.0, min(1.0, v))  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="AUC bars">']
    for t in np.linspace(0, 1, 6):
        out.append(f'<line x1="{x(t):.1f}" y1="{top - 6}" x2="{x(t):.1f}" y2="{h - 28}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{h - 12}" class="tick" text-anchor="middle">{t:.1f}</text>')
    for i, (label, value, kind) in enumerate(bars):
        y = top + i * (bh + gap)
        out.append(f'<text x="{left - 8}" y="{y + bh * 0.7:.1f}" class="label" text-anchor="end">{esc(label)}</text>')
        if value is not None:
            out.append(f'<rect x="{left}" y="{y}" width="{x(value) - left:.1f}" height="{bh}" class="bar {kind}"/>'
                       f'<text x="{x(value) + 5:.1f}" y="{y + bh * 0.7:.1f}" class="value">{value:.3f}</text>')
    short = {"still_gt0": "true still", "place_pos_gt": "true positions", "place_gt_when_resting": "true at rest",
             "place_gt_when_moving": "true when moving"}
    placed = sorted((v, l) for l, v in refs if v is not None)
    for k, (value, label) in enumerate(placed):  # labels on two staggered rows above the bars
        out.append(f'<line x1="{x(value):.1f}" y1="{top - 8 - 16 * (k % 2)}" x2="{x(value):.1f}" y2="{h - 28}" '
                   f'class="ref"><title>{esc(label)} {value:.3f}</title></line>'
                   f'<text x="{x(value):.1f}" y="{top - 12 - 16 * (k % 2)}" class="reflabel" text-anchor="middle">'
                   f'{esc(short.get(label, label))}</text>')
    out.append("</svg>")
    legend = " · ".join(f'<span class="refkey">┆ {esc(l)} {v:.3f}</span>' for l, v in refs if v is not None)
    return "\n".join(out) + f'\n<p class="legend">Dashed lines, S0.2 oracle ceilings: {legend}</p>'


def timeline(series: list[tuple[str, np.ndarray, str]], phases: dict | None, ticks: list[int] | None, ylabel: str,
             width: int = 680, height: int = 184) -> str:
    left, right, top, bottom = 48, 12, 24, 26
    n = max(len(s[1]) for s in series)
    ymax = max(float(np.nanmax(s[1])) for s in series if np.isfinite(s[1]).any())
    ymax = max(ymax * 1.08, 1e-6)
    x = lambda t: left + (width - left - right) * t / max(n - 1, 1)  # noqa: E731
    y = lambda v: top + (height - top - bottom) * (1 - min(v, ymax) / ymax)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{esc(ylabel)} per frame">']
    if phases:
        for key, cls in (("rest_before", "rest"), ("carry", "carry"), ("rest_after", "rest")):
            w = phases.get(key)
            if w:
                out.append(f'<rect x="{x(w[0]):.1f}" y="{top}" width="{max(x(w[1] + 1) - x(w[0]), 1):.1f}" '
                           f'height="{height - top - bottom}" class="phase {cls}"/>')
    for v in np.linspace(0, ymax, 4):
        out.append(f'<line x1="{left}" y1="{y(v):.1f}" x2="{width - right}" y2="{y(v):.1f}" class="grid"/>'
                   f'<text x="{left - 6}" y="{y(v) + 3:.1f}" class="tick" text-anchor="end">{v:.0f}</text>')
    for t in range(0, n, max(1, n // 6)):
        out.append(f'<text x="{x(t):.1f}" y="{height - 8}" class="tick" text-anchor="middle">{t}</text>')
    for t in ticks or []:
        out.append(f'<line x1="{x(t):.1f}" y1="{height - bottom}" x2="{x(t):.1f}" y2="{height - bottom + 6}" class="tickmark"/>')
    for label, values, cls in series:
        pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in enumerate(values) if np.isfinite(v))
        out.append(f'<polyline points="{pts}" class="line {cls}"><title>{esc(label)}</title></polyline>')
    out.append(f'<text x="{left}" y="12" class="axis">{esc(ylabel)}</text></svg>')
    keys = " ".join(f'<span class="key {cls}">━ {esc(label)}</span>' for label, _, cls in series)
    marks = f" · ticks: {len(ticks)} recovered frames" if ticks else " · no recovered frames"
    return "\n".join(out) + f'\n<p class="legend">{keys} · shaded: S4a rest-before / carry / rest-after{marks}</p>'


def silhouettes(v: np.ndarray, f: np.ndarray, sv: np.ndarray, sf: np.ndarray, px: int = 170) -> str:
    """Three orthographic silhouettes (along the mesh frame's x, y, z): ours and the scan placed in our
    frame, overlaid; a base64 PNG strip."""
    allv = np.vstack([v, sv])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    span = float((hi - lo).max()) * 1.08 or 1.0
    centre = (lo + hi) / 2
    tiles = []
    for drop in (0, 1, 2):
        keep = [a for a in range(3) if a != drop]
        img = np.full((px, px, 4), 0, np.uint8)
        for verts, faces, colour in ((sv, sf, (56, 141, 238, 140)), (v, f, (232, 136, 28, 150))):
            uv = ((verts[:, keep] - centre[keep]) / span + 0.5) * px
            uv[:, 1] = px - uv[:, 1]
            layer = np.zeros((px, px), np.uint8)
            for tri_uv in np.round(uv[faces]).astype(np.int32):  # one by one: fillPoly's even-odd rule
                cv2.fillConvexPoly(layer, tri_uv, 255)            # cancels overlapping triangles
            m = layer > 0
            a = colour[3] / 255
            img[m, :3] = (img[m, :3] * (1 - a) + np.array(colour[:3]) * a).astype(np.uint8)
            img[m, 3] = np.maximum(img[m, 3], int(255 * 0.85))
        tiles.append(img)
    strip = np.concatenate([np.pad(t, ((0, 0), (0, 8), (0, 0))) for t in tiles], axis=1)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(strip, cv2.COLOR_RGBA2BGRA))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------- the report
def fmt(v, d=3):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def attribution(scores: dict) -> list[dict]:
    """Stage by stage: the column before and after it, from the waterfall."""
    s = lambda k: scores.get(k)  # noqa: E731
    best_ref = max((f"ref_{c}" for c in CANDIDATES if c != "still" and f"ref_{c}" in scores),
                   key=lambda c: scores[c]["AUC"], default=None)
    best_real = max((f"realised_{c}" for c in RETARGETED if f"realised_{c}" in scores),
                    key=lambda c: scores[c]["AUC"], default=None)
    best_roll = max((f"rollout_{c}" for c in CANDIDATES if f"rollout_{c}" in scores),
                    key=lambda c: scores[c]["AUC"], default=None)
    truth = {"AUC": 1.0, "MPPE cm": 0.0, "CD-O cm": 0.0}
    rows = [
        ("P1c camera + S3b track", "ground truth", truth, "raw", s("raw"),
         "the camera and the camera-frame track together; no ground-truth camera separates them (camera panel)"),
        ("S3a mesh", "raw@scan", s("raw" + SCAN), "raw", s("raw"), "same poses, the scan as mesh (ceiling, not submittable)"),
        ("S3c invert + S3d settle", "ref_keyframe", s("ref_keyframe"), "ref_inverted", s("ref_inverted"),
         "segment solving against v3's keyframe rule (the v4 go/no-go)"),
        ("S4b reference", "raw", s("raw"), best_ref or "ref_*", s(best_ref) if best_ref else None, "best of inverted / keyframe"),
        ("S4c physics", best_ref or "ref", s(best_ref) if best_ref else None, best_real or "realised_*",
         s(best_real) if best_real else None, "Do as I Do (not run without the VM)" if not best_real else ""),
        ("S6 rollout", best_real or best_ref or "ref", s(best_real or best_ref) if (best_real or best_ref) else None,
         best_roll or "rollout_*", s(best_roll) if best_roll else None, "FlashCHORD (not run without the VM)" if not best_roll else ""),
    ]
    out = []
    for stage, bname, before, aname, after, note in rows:
        loss = (before["AUC"] - after["AUC"]) if before and after else None
        out.append({"stage": stage, "before": bname, "after": aname, "b": before, "a": after, "loss": loss, "note": note})
    ranked = [r for r in out if r["loss"] is not None and not r["stage"].startswith("stage-2")]
    if ranked:
        max(ranked, key=lambda r: r["loss"])["largest"] = True
    return out


def flaws(eps: list[dict]) -> list[tuple[str, str]]:
    """The three biggest flaws: (one sentence with one number, the evidence behind it), in order: the
    position loss, a moving candidate that does not beat still, a two-object scene's relative
    placement, the camera's drift at rest (the first three that apply)."""
    out = []
    first = {e["number"]: next(r for r in e["attribution"] if r["stage"].startswith("P1c")) for e in eps}
    loss = np.mean([r["loss"] for r in first.values() if r["loss"] is not None])
    mesh = {e["number"]: next(r for r in e["attribution"] if r["stage"] == "S3a mesh") for e in eps}
    out.append((f"The object's position in the camera frame (S3b's track seen through P1c's camera) costs "
                f"{loss:.2f} AUC on its own (mean of the dev episodes).",
                "AUC 1.000 → " + " / ".join(f"{fmt(first[e['number']]['a']['AUC'])} (ep{e['number']})" for e in eps)
                + "; the track's depth sits " + " / ".join(f"{fmt(e['s3b_residual_cm'], 1)} cm" for e in eps)
                + " (median) from the stereo depth; S0.2's true positions alone would score "
                + " / ".join(fmt(e["oracle"].get("place_pos_gt", {}).get("AUC")) for e in eps)
                + "; the scan at the same poses gains " + " / ".join(
                    fmt(mesh[e["number"]]["b"]["AUC"] - mesh[e["number"]]["a"]["AUC"]) for e in eps
                    if mesh[e["number"]]["a"] and mesh[e["number"]]["b"]) + " AUC."))
    stuck = [e for e in eps if e["best"] and e["still"] is not None and e["scores"][e["best"]]["AUC"] <= e["still"] + 1e-9]
    if stuck:
        e = stuck[0]
        gap = e["still"] - e["scores"][e["best"]]["AUC"]
        line = (f"On ep{e['number']} the best moving candidate only ties standing still ({e['still']:.3f} AUC)."
                if gap < 1e-9 else
                f"On ep{e['number']} the best moving candidate scores {gap:.2f} AUC below standing still.")
        out.append((line, f"best {e['best']} {fmt(e['scores'][e['best']]['AUC'])} vs still {fmt(e['still'])}"
                    + ("; " + e["s4a_note"] if e["s4a_note"] != "no flag" else "")
                    + (f"; S5's frame-0 seating gap {fmt(e['seating_cm'], 1)} cm" if e["seating_cm"] else "") + "."))
    pairs = [(e, e["scores"][e["best"]]) for e in eps if len(e["names"]) > 1 and e["best"]]
    pairs = [(e, s) for e, s in pairs if s.get("RPE cm") is not None and s["RPE cm"] > 5]
    if pairs:
        e, sc = pairs[0]
        out.append((f"On ep{e['number']} the second object sits {sc['RPE cm']:.0f} cm (RPE) from where it belongs "
                    f"relative to the first.",
                    f"best {e['best']}: RPE {fmt(sc['RPE cm'], 1)} cm, MPPE {fmt(sc['MPPE cm'], 1)} cm; "
                    + ("phase 2's anchoring: p2_best RPE " + fmt(e["scores"]["p2_best"].get("RPE cm"), 1) + " cm"
                       if "p2_best" in e["scores"] else "phase 2 not run") + "."))
    cams = [(e["number"], e["camera"]) for e in eps if e["camera"]["plane_height_spread_cm"] is not None]
    if cams:
        n, worst = max(cams, key=lambda c: c[1]["plane_height_spread_cm"])
        stand = any("p1c_camera" in e["stand_ins"] for e in eps)
        out.append((f"The camera drifts against the table while nothing moves, by {worst['plane_height_spread_cm']:.1f} cm "
                    f"on ep{n} (P1c's bar is 1 cm).",
                    "plane-height spread over the rest window " + " / ".join(f"{fmt(c['plane_height_spread_cm'], 1)} cm (ep{k})"
                                                                           for k, c in cams)
                    + "".join(f"; ep{k}: S0.2's camera common-mode RMS at rest {fmt(c['rest_common_rms_cm'], 1)} cm vs the "
                              f"objects' own {fmt(c['rest_diff_rms_cm'], 1)} cm" for k, c in cams if c["rest_common_rms_cm"])
                    + "; part of the first flaw" + ("; the stand-in camera is the team's DROID path, which v3's "
                                                     "stereo-anchored P1c replaces" if stand else "") + "."))
    return out[:3]


def episode_section(run: Path, number: int, read: list, best: str | None = None) -> dict:
    root = run / f"episode_{number:06d}"
    e = episode_info(number)
    names = [o["name"] for o in e["objects"]]
    wf = json.loads((root / "waterfall.json").read_text())
    scores = wf["scores"]
    orc, extra = oracle(number)
    best = best or best_candidate(scores)
    still = scores.get("ref_still", {}).get("AUC")
    errs = frame_errors(root, e, [c for c in ("raw", "ref_keyframe", best) if c], read)
    residual = []  # median depth residual per object, from S3b's own per-frame records (cm)
    for n in names:
        meta = root / f"s3b_track/{n}/track_meta.json"
        if meta.is_file():
            r = [f["depth_residual_m"] for f in json.loads(meta.read_text())["frames"] if f.get("depth_residual_m") is not None]
            if r:
                residual.append(float(np.median(r)) * 100)
    s4a = json.loads((root / "s4a_phases/report.json").read_text())
    flagged = [m for m in s4a["metrics"] if m["status"] in ("WARN", "FAIL")]
    phases = {n: json.loads((root / f"s4a_phases/{n}/phases.json").read_text()) for n in names}
    anchors = {n: json.loads((root / f"s3c_fuse/{n}/anchors.json").read_text()) for n in names
               if (root / f"s3c_fuse/{n}/anchors.json").is_file()}
    s5 = json.loads((root / "s5_task/report.json").read_text()) if (root / "s5_task/report.json").is_file() else None
    seating = max((abs(m["value"]) for m in (s5 or {}).get("metrics", []) if "seating" in m["name"]
                   and isinstance(m["value"], (int, float))), default=None)
    moving = {n: phases[n].get("displacement_cm") for n in names}
    def gt_moving(n):
        return next((d["gt_moving_frac"] for r in orc.values() if r.get("extra") for d in r["extra"].get("diag", [])
                     if d["name"] == n), None)
    note = "; ".join(f"{n} moves {moving[n]:.1f} cm in our track and is "
                     + (f"reached at frame {phases[n]['first_reach']}" if phases[n].get("first_reach") is not None
                        else "never reached")
                     + (f", while ground truth has it moving on {gt_moving(n):.0%} of frames" if gt_moving(n) is not None else "")
                     for n in names)
    return {"number": number, "root": root, "e": e, "names": names, "scores": scores, "baselines": wf["baselines"],
            "oracle": orc, "extra": extra, "best": best, "still": still, "errors": errs, "phases": phases,
            "attribution": attribution(scores), "stand_ins": stand_ins(root),
            "s3b_residual_cm": float(np.median(residual)) if residual else None,
            "anchorless": ", ".join(f"ep{number} {n}" for n in names if anchors.get(n) and not anchors[n].get("rest_before")) or "none",
            "s4a_note": note if flagged else "no flag", "seating_cm": seating,
            "mesh": {n: mesh_panel(root, e, n, read) for n in names},
            "camera": camera_panel(root, e, extra)}


def summary(eps: list[dict]) -> dict:
    """The page's key numbers, saved next to it, so a later page can say what changed."""
    out = {}
    for e in eps:
        sc = e["scores"]
        out[str(e["number"])] = {
            "best": e["best"], "best_auc": sc[e["best"]]["AUC"] if e["best"] else None, "still": e["still"],
            "columns": {c: sc[c]["AUC"] for c in sc},
            "losses": {r["stage"]: r["loss"] for r in e["attribution"]},
            "cdo": {n: m["cdo_cm"] for n, m in e["mesh"].items()},
            "camera_spread_cm": e["camera"]["plane_height_spread_cm"]}
    return out


def extra_sections(eps: list[dict], run: Path, base: dict | None) -> list[str]:
    """Pages after the first (--tag): phase A2's comparison and decision, phase 2's columns, what changed."""
    md = []
    decision = run / "a2_decision.json"
    if not decision.is_file() and "_a2_" in run.name:  # a variant run: the decision sits in its source run
        decision = run.parent / run.name.split("_a2_")[0] / "a2_decision.json"
    if decision.is_file():
        d = json.loads(decision.read_text())
        md += ["## g. Phase A2: S3 variants (A2.5)", "",
               "| Episode | Variant | Best candidate | AUC | MPPE cm | CD-O cm per object |", "|---|---|---|---|---|---|"]
        for ep, row in d["episodes"].items():
            for v, r in row.items():
                md.append(f"| ep{ep} | `{v}` | `{r['best']}` | {fmt(r['AUC'])} | {fmt(r['MPPE cm'], 2)} | "
                          + ", ".join(f"{n} {fmt(c, 2)}" for n, c in r["CD-O cm"].items()) + " |")
        dec = d["decision"]
        md += ["", f"**Decision (spec A2.5 rank rule):** `{dec['choice']}` — mean rank "
               + ", ".join(f"{v} {m:.2f}" for v, m in dec["mean_rank"].items())
               + ("; A2 lost on both AUC and CD-O" if dec["a2_lost_on_both"] else "")
               + (f". {d['stand_ins']}: the decision is provisional until the dev cache." if d.get("stand_ins") else "."), "",
               "Why the AUC rises while CD-O worsens: the kit registers the submitted mesh to the scan scale-free and "
               "follows the scan's anchor through our poses. The canonical mesh is fused where the object was seen and "
               "registered to the track's rest pose, so that anchor sits on the real object; the team mesh sits off the "
               "track, so its anchor swings with every rotation (same poses: raw 0.460 → 0.687 on ep40).", ""]
    p2 = [(e["number"], {c: e["scores"][c] for c in e["scores"] if c.startswith("p2_")}) for e in eps]
    p2 = [(n, cols) for n, cols in p2 if cols]
    if p2:
        md += ["## h. Phase 2: two-object scope", "", "| Episode | Column | AUC | RPE cm | MPPE cm | Phase 1 (same column) |",
               "|---|---|---|---|---|---|"]
        for n, cols in p2:
            e = next(x for x in eps if x["number"] == n)
            for c, v in cols.items():
                p1 = e["scores"].get(c[3:]) if c != "p2_best" else (e["scores"].get(e["best"]) if e["best"] else None)
                md.append(f"| ep{n} | `{c}`{' (' + v['candidate'] + ')' if 'candidate' in v else ''} | {fmt(v['AUC'])} | "
                          f"{fmt(v.get('RPE cm'), 2)} | {fmt(v.get('MPPE cm'), 2)} | {fmt((p1 or {}).get('AUC'))} |")
        md.append("")
    if base:
        md += ["## What changed since the first page", ""]
        now = summary(eps)
        for ep, b in base.items():
            a = now.get(ep)
            if not a:
                continue
            parts = [f"best candidate {b['best']} {fmt(b['best_auc'])} → {a['best']} {fmt(a['best_auc'])}"]
            new_cols = [f"{c} {fmt(a['columns'][c])}" for c in a["columns"] if c not in b["columns"] and SCAN not in c]
            if new_cols:
                parts.append("new columns: " + ", ".join(new_cols))
            parts += [f"mesh {n} CD-O {fmt(b['cdo'].get(n), 2)} → {fmt(c, 2)} cm" for n, c in a["cdo"].items()
                      if b["cdo"].get(n) is not None and abs(c - b["cdo"][n]) > 1e-6]
            md.append(f"- **ep{ep}:** " + "; ".join(parts))
        md.append("")
    return md


def markdown(eps: list[dict], run: Path, tag: str, base: dict | None = None) -> str:
    today = date.today().isoformat()
    which = " and ".join(f"ep{e['number']}" for e in eps)
    md = [f"# Dev-episode analysis: {which} ({today}{', ' + tag if tag else ''})", "",
          f"Run `{run.name}`; scored with the official Track 3 metric code (`pathb/gt.py`), public ground truth.",
          ""]
    si = sorted({s for e in eps for s in e["stand_ins"]})
    if si:
        md += [f"> **Stand-in inputs:** {', '.join(si)} come from run `4c2eac31`'s v2 bundles (the team's HF "
               "reconstructions at stereo scale), not from v3's own GPU stages; every number downstream of them is "
               "a stand-in number. The dev cache (`vm/dev_cache.sh`) replaces them.", ""]
    fl = flaws(eps)
    md += [f"**At a glance — biggest flaw:** {fl[0][0] if fl else ''}", ""]
    # (a)
    md += ["## a. Headline: today's best candidate vs still vs the S0.2 oracle", ""]
    for e in eps:
        sc, orc = e["scores"], e["oracle"]
        recon = "recon" if len(e["names"]) == 1 else "place"
        bars = [(f"best: {e['best']}" if e["best"] else "best candidate", sc[e["best"]]["AUC"] if e["best"] else None, "ours"),
                ("still (ours, ref_still)", e["still"], "still"),
                (f"S0.2 {recon} (team bundle)", orc.get(recon, {}).get("AUC"), "base"),
                ("S0.2 still", orc.get("still", {}).get("AUC"), "base"),
                ("raw (S3b + camera)", sc.get("raw", {}).get("AUC"), "stage"),
                ("ref_keyframe (v3 rule)", sc.get("ref_keyframe", {}).get("AUC"), "stage")]
        refs = [(k, orc.get(k, {}).get("AUC")) for k in ORACLE_LINES]
        md += [f"### ep{e['number']} ({' + '.join(e['names'])})", "", bar_chart(bars, refs), ""]
    # (b)
    md += ["## b. Per-frame error per object", "",
           "Errors as the official metric sees them: each mesh registered to its scan, the scene aligned by its "
           "frame-0 pose (`gt.official_frame_errors`). Position = the scan anchor's error; rotation is symmetry-aware.",
           ""]
    for e in eps:
        for j, n in enumerate(e["names"]):
            ph, rec = e["phases"][n], recovered_frames(e["root"], n)
            series_p, series_r = [], []
            for v, cls in (("raw", "raw"), ("ref_keyframe", "fused"), (e["best"], "ours")):
                if v in e["errors"]:
                    err = e["errors"][v]
                    series_p.append((v, err["position_m"][:, j] * 100, cls))
                    series_r.append((v, np.degrees(err["rotation_rad"][:, j]), cls))
            any_err = next(iter(e["errors"].values()))
            disp = [("ground truth", any_err["truth_displacement_m"][:, j] * 100, "truth"),
                    (e["best"], e["errors"][e["best"]]["ours_displacement_m"][:, j] * 100, "ours")]
            md += [f"### ep{e['number']} · {n}" + (" (stand-in track: nothing is marked recovered)" if "s3b_track" in e["stand_ins"] else ""), "",
                   timeline(series_p, ph, rec, "position error (cm)"), "",
                   timeline(series_r, ph, rec, "rotation error (deg)"), "",
                   timeline(disp, ph, rec, "displacement from frame 0 (cm)"), ""]
    # (c)
    md += ["## c. Loss attribution by stage", "",
           "AUC / MPPE / CD-O of the waterfall column before and after each stage. The largest single loss is in bold.", ""]
    for e in eps:
        md += [f"### ep{e['number']}", "", "| Stage | Before | After | AUC before → after | AUC lost (− = gained) | MPPE cm | CD-O cm | Note |",
               "|---|---|---|---|---|---|---|---|"]
        for r in e["attribution"]:
            b, a = r["b"] or {}, r["a"] or {}
            cells = [r["stage"], f"`{r['before']}`", f"`{r['after']}`", f"{fmt(b.get('AUC'))} → {fmt(a.get('AUC'))}",
                     fmt(r["loss"]), f"{fmt(b.get('MPPE cm'), 2)} → {fmt(a.get('MPPE cm'), 2)}",
                     f"{fmt(b.get('CD-O cm'), 2)} → {fmt(a.get('CD-O cm'), 2)}", r["note"]]
            if r.get("largest"):
                cells = [f"**{c}**" if c else c for c in cells]
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
    # (d)
    md += ["## d. Meshes against the public scan", "",
           "Silhouettes along the mesh frame's x, y and z axes: ours (orange) over the scan placed in our frame the "
           "way the official metric registers our mesh (blue). Coverage = share of the scan's surface within 1 cm of ours.", ""]
    for e in eps:
        for n, m in e["mesh"].items():
            md += [f"### ep{e['number']} · {n}" + (" (stand-in mesh: the team's SAM3D)" if "s3a_mesh" in e["stand_ins"] else ""), "",
                   f"CD-O **{fmt(m['cdo_cm'], 2)} cm** · size ratio (ours / scan) **{fmt(m['size_ratio'], 2)}** · "
                   f"coverage **{fmt(m.get('coverage_1cm'), 2)}**", ""]
            if m.get("silhouettes"):
                md += [f'<img class="sil" alt="silhouettes of {esc(n)}" src="{m["silhouettes"]}"/>', ""]
    # (e)
    md += ["## e. Camera", "", "| Episode | P1c path | Plane-height spread, rest window | Normal spread | Scale (stereo / P1c) | "
           "S0.2 rest common RMS (camera) | S0.2 rest difference RMS (object) |", "|---|---|---|---|---|---|---|"]
    for e in eps:
        c = e["camera"]
        md.append(f"| ep{e['number']}{' (stand-in camera)' if 'p1c_camera' in e['stand_ins'] else ''} | {fmt(c['path_m'], 2)} m | "
                  f"{fmt(c['plane_height_spread_cm'], 2)} cm (frames {c['rest_window'][0]}–{c['rest_window'][1]}) | "
                  f"{fmt(c['plane_normal_spread_deg'], 2)}° | {fmt(c['scale_factor'], 3)} | "
                  f"{fmt(c['rest_common_rms_cm'], 1) + ' cm' if c['rest_common_rms_cm'] is not None else 'one object: n/a'} | "
                  f"{fmt(c['rest_diff_rms_cm'], 1) + ' cm' if c['rest_diff_rms_cm'] is not None else 'n/a'} |")
    md.append("")
    # (f)
    md += ["## f. Biggest flaws", ""]
    for label, (line, evidence) in zip(("Biggest flaw", "Second", "Third"), fl):
        md += [f"**{label}:** {line}", "", f"Evidence: {evidence}", ""]
    if tag:
        md += extra_sections(eps, run, base)
    return "\n".join(md)


CSS = """
:root { --bg:#fbfbfa; --fg:#1d1d1f; --muted:#6b6b70; --grid:#e3e3e6; --card:#ffffff; --border:#e2e2e5;
  --ours:#d9730d; --still:#7a7a80; --base:#9aa3b5; --stage:#c9a46a; --raw:#a7a7ad; --fused:#3f7fcc; --truth:#2f9e6f;
  --ref:#c0392b; --rest:rgba(47,158,111,.10); --carry:rgba(217,115,13,.08); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141416; --fg:#ececef; --muted:#9c9ca3;
  --grid:#2c2c31; --card:#1c1c1f; --border:#2e2e33; --ours:#f0913a; --still:#9c9ca3; --base:#6f7a90; --stage:#b38f55;
  --raw:#77777e; --fused:#6aa7ef; --truth:#4cc28f; --ref:#ef6b5c; --rest:rgba(76,194,143,.13); --carry:rgba(240,145,58,.11); } }
:root[data-theme="dark"] { --bg:#141416; --fg:#ececef; --muted:#9c9ca3; --grid:#2c2c31; --card:#1c1c1f; --border:#2e2e33;
  --ours:#f0913a; --still:#9c9ca3; --base:#6f7a90; --stage:#b38f55; --raw:#77777e; --fused:#6aa7ef; --truth:#4cc28f;
  --ref:#ef6b5c; --rest:rgba(76,194,143,.13); --carry:rgba(240,145,58,.11); }
body { background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  margin:0; padding:24px 16px 64px; } main { max-width:760px; margin:0 auto; }
h1 { font-size:24px; margin:0 0 6px } h2 { font-size:19px; margin:34px 0 8px; border-top:1px solid var(--border);
  padding-top:22px } h3 { font-size:15px; margin:20px 0 6px; color:var(--muted) }
blockquote { margin:12px 0; padding:10px 14px; background:var(--card); border:1px solid var(--border); border-radius:8px }
table { border-collapse:collapse; width:100%; font-size:13px; margin:8px 0; display:block; overflow-x:auto }
th,td { border-bottom:1px solid var(--border); padding:6px 8px; text-align:left; vertical-align:top } th { color:var(--muted); font-weight:600 }
code { font:12px ui-monospace,Menlo,monospace } .chart { width:100%; height:auto; display:block }
.grid { stroke:var(--grid); stroke-width:1 } .tick,.axis,.value,.label { fill:var(--muted); font-size:11px } .reflabel { fill:var(--ref); font-size:10.5px }
.label { fill:var(--fg); font-size:12px } .value { fill:var(--fg) }
.bar.ours { fill:var(--ours) } .bar.still { fill:var(--still) } .bar.base { fill:var(--base) } .bar.stage { fill:var(--stage) }
.ref { stroke:var(--ref); stroke-width:1.5; stroke-dasharray:4 3 } .refkey { color:var(--ref) }
.line { fill:none; stroke-width:1.4 } .line.ours { stroke:var(--ours); stroke-width:2.2 } .line.fused { stroke:var(--fused) }
.line.raw { stroke:var(--raw); stroke-dasharray:3 2 } .line.truth { stroke:var(--truth); stroke-width:2 }
.key.ours { color:var(--ours) } .key.fused { color:var(--fused) } .key.raw { color:var(--raw) } .key.truth { color:var(--truth) }
.phase.rest { fill:var(--rest) } .phase.carry { fill:var(--carry) } .tickmark { stroke:var(--ref); stroke-width:1.5 }
.legend { color:var(--muted); font-size:12px; margin:2px 0 12px } img.sil { max-width:100%; height:auto; background:var(--card);
  border:1px solid var(--border); border-radius:6px }
ol li { margin-bottom:8px }
"""


def to_html(md: str, title: str, css: str = "") -> str:
    """The subset of markdown this page uses (headings, paragraphs, tables, lists, quotes, bold, code;
    raw HTML blocks pass through)."""
    def inline(t: str) -> str:
        t = html.escape(t, quote=False)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("<"):
            block = []
            while i < len(lines) and lines[i].strip():
                block.append(lines[i])
                i += 1
            out.append("\n".join(block))
            continue
        if ln.startswith("#"):
            level = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{level}>{inline(ln[level:].strip())}</h{level}>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[2:]]
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table>")
            continue
        elif ln.startswith("> "):
            out.append(f"<blockquote>{inline(ln[2:])}</blockquote>")
        elif re.match(r"^\d+\. ", ln):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                items.append(inline(re.sub(r"^\d+\. ", "", lines[i])))
                i += 1
            out.append("<ol>" + "".join(f"<li>{x}</li>" for x in items) + "</ol>")
            continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{CSS}{css}</style></head>'
            f'<body><main>{"".join(out)}</main></body></html>')


def main(run: Path, numbers: list[int] | None = None, tag: str = "") -> tuple[Path, Path]:
    numbers = list(numbers or DEV_EPISODES)
    read: list = []
    eps = [episode_section(run, n, read) for n in numbers if episode_info(n)["split"] == "public"]
    REPORTS.mkdir(exist_ok=True)
    first = sorted(REPORTS.glob("dev_analysis_*.json"))
    base = json.loads(first[0].read_text()) if tag and first else None
    md = markdown(eps, run, tag, base)
    stem = f"dev_analysis_{date.today().isoformat()}" + (f"_{tag}" if tag else "")
    (REPORTS / f"{stem}.json").write_text(json.dumps(summary(eps), indent=1) + "\n")
    (REPORTS / f"{stem}.md").write_text(md)
    (REPORTS / f"{stem}.html").write_text(to_html(md, "Dev episode analysis"))
    return REPORTS / f"{stem}.md", REPORTS / f"{stem}.html"
