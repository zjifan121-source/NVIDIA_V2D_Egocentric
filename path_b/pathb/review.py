"""Phase-1 performance review (James, 2026-10-23, after phase B): how the selected candidates score.

    uv run --project path_b python path_b/run.py review --split all --out path_b/data/outputs/phaseb

Writes path_b/reports/phase1_review_<date>.{md,html,json}. Public episodes are scored with the official
Track 3 metric code through pathb/gt.py; evaluation episodes have no ground truth, so their section
uses only self-consistency signals (the rollout against the S4b observation, termination-equivalent
frames, FlashCHORD's error against the candidate's own reference, the stages' own bars). Sections, in
James's order: (a) headline; (b) best 5 public episodes; (c) worst 5 public episodes with a diagnosis
each; (d) evaluation split; (e) candidate selection; (f) what to fix next, ranked; appendix of file
paths. Charts are inline SVG in light and dark (pathb/analysis.py's helpers and styles).

An episode without S6's selection (a run that stopped before S6) is shown with its best waterfall
column, labelled as not a rollout.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from . import analysis
from .analysis import esc, fmt
from .paths import REPO, episode_info
from .reference import CANDIDATES

REPORTS = analysis.REPORTS
METRICS = ("AUC", "SP-SR", "MP-SR", "MPPE cm", "RPE cm", "CD-O cm")
KAGGLE_STILL_AUC = 0.295  # the team's still submission, public leaderboard (James, 2026-10-23)
LEADERBOARD_TOP = {"AUC": 0.848, "MPPE cm": 2.1, "CD-O cm": 0.065}  # (James, 2026-10-23)
# S0.2 oracle variants (team bundles, Oct 17) that make one thing true, and the spec step that would get there
FIXES = {
    "place_scale": ("true object scale", "P3 scale bridge / S3a mesh size"),
    "place_gt_when_resting": ("true poses while the object rests", "S3c static-segment ICP + S3d settle"),
    "place_gt_when_moving": ("true poses while the object moves", "S3c grasp inversion (hand track carries the object)"),
}
# what to fix next: attribution stage -> (kind, action)
ACTIONS = {
    "P1c camera + S3b track": ("v4", "S3b contact segments + S3c segment solving replace the per-frame track"),
    "S3a mesh": ("existing step", "S3a canonical mesh, pooled over every clip of the object"),
    "S3c fuse": ("v4", "S3c segment solving (static ICP, grasp inversion) replaces S3c fuse"),
    "S4b reference": ("existing step", "S4b: inverted / keyframe reference"),
    "S4c physics": ("existing step", "S4c: Do as I Do settings and its gate"),
    "S6 rollout": ("existing step", "S6: FlashCHORD training (steps, `njmax`)"),
}
EXTRA_CSS = """
.dot { fill:var(--fg) } .dot.still { fill:var(--still) } .dot.gt { fill:none; stroke:var(--truth); stroke-width:1.6 }
.bar.inverted { fill:var(--ours) } .bar.keyframe { fill:var(--fused) } .bar.still { fill:var(--still) }
.bar.gain { fill:var(--truth) } .bar.loss { fill:var(--ref) } .bar.other { fill:var(--base) }
.key.inverted { color:var(--ours) } .key.keyframe { color:var(--fused) } .key.still { color:var(--still) }
.diag { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:4px 16px 10px; margin:16px 0 }
"""


# ---------------------------------------------------------------- data
def oracle_means() -> dict:
    """S0.2's means over the 20 public episodes: still_gt0 and recon (one object) / place (two)."""
    rows = json.loads(analysis.ORACLE.read_text())["variants"]
    by: dict = {}
    for r in rows:
        by.setdefault(r["v"], {})[r["ep"]] = r
    eps = sorted(by["still_gt0"])
    pick = lambda v, e: by[v][e]  # noqa: E731
    base = [pick("recon" if pick("still", e)["n"] == 1 else "place", e) for e in eps]
    keys = {"AUC": "AUC", "SP-SR": "SP-SR", "MP-SR": "MP-SR", "MPPE cm": "MPPE", "RPE cm": "RPE"}
    mean = lambda rs: {m: float(np.mean([r[k] for r in rs if r.get(k) is not None])) for m, k in keys.items()}  # noqa: E731
    return {"still_gt0": mean([pick("still_gt0", e) for e in eps]), "recon/place": mean(base), "episodes": len(eps)}


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def chosen(root: Path) -> dict:
    """The episode's submitted candidate: S6's selection, else the best waterfall column."""
    sel = load(root / "s6_select/selection.json")
    wf = load(root / "waterfall.json") or {}
    scores = wf.get("scores", {})
    if sel:
        cand = sel["candidate"]
        return {"candidate": cand, "column": f"rollout_{cand}", "rollout": True, "selection": sel,
                "scores": (sel.get("scores") or {}).get(cand), "still": (sel.get("scores") or {}).get("still"),
                "all": sel.get("scores") or {}, "waterfall": scores}
    col = analysis.best_candidate(scores)
    return {"candidate": col, "column": col, "rollout": False, "selection": None, "scores": scores.get(col),
            "still": scores.get("ref_still"), "all": {}, "waterfall": scores}


def flagged(root: Path) -> dict:
    """First failing / warning stage and its first flagged bar, from the episode's loss_report.json."""
    lr = load(root / "loss_report.json") or {}
    out = {"first_failing_stage": lr.get("first_failing_stage"), "first_warning_stage": lr.get("first_warning_stage"),
           "errors": [s["stage"] for s in lr.get("stages", []) if s.get("status") == "ERROR"], "bars": []}
    for s in lr.get("stages", []):
        for m in s.get("flagged") or []:
            out["bars"].append({"stage": s["stage"], **m})
    out["first_bar"] = next((b for b in out["bars"] if b["status"] in ("FAIL", "WARN")), None)
    return out


def bar_text(b: dict | None) -> str:
    if not b:
        return "no stage flagged a bar"
    v = b["value"]
    value = fmt(v, 2) if isinstance(v, (int, float)) else str(v)
    return f"{b['status']} at `{b['stage']}`: {b['name']} = {value} {b['unit']} (bar {b['bar']})"


def improvement(ep: dict) -> tuple[str, float | None]:
    """One improvement for an episode's biggest flaw, with its spec step and expected AUC gain."""
    flaw = ep["flaw"]
    if flaw and flaw["stage"] == "S3a mesh" and flaw["loss"] is not None:
        return (f"S3a mesh → A2.1–A2.2 canonical mesh; expected gain up to {flaw['loss']:+.2f} AUC (the scan at the "
                "same poses, this run's `@scan` column)", flaw["loss"])
    if flaw and flaw["stage"] in ("S4b reference", "S4c physics", "S6 rollout") and flaw["loss"] is not None:
        kind, action = ACTIONS[flaw["stage"]]
        return f"{action}; expected gain up to {flaw['loss']:+.2f} AUC (the loss at that stage)", flaw["loss"]
    orc = ep["oracle"]
    base = orc.get("recon" if len(ep["names"]) == 1 else "place", {}).get("AUC")
    gains = {v: orc[v]["AUC"] - base for v in FIXES if v in orc and base is not None}
    if not gains:
        return "no S0.2 oracle variant for this episode", None
    v = max(gains, key=gains.get)
    upper = orc.get("place_pos_gt", {}).get("AUC")
    return (f"{FIXES[v][1]}: S0.2's {FIXES[v][0]} add {gains[v]:+.2f} AUC on the team bundle (`{v}`, Oct 17)"
            + (f"; true positions on every frame, our rotations: {upper - base:+.2f} (`place_pos_gt`)" if upper is not None else ""),
            gains[v])


def public_episode(run: Path, n: int, read: list, deep: bool) -> dict:
    root = run / f"episode_{n:06d}"
    e = episode_info(n)
    c = chosen(root)
    orc, extra = analysis.oracle(n)
    ep = {"number": n, "root": root, "names": [o["name"] for o in e["objects"]], "choice": c, "oracle": orc,
          "extra": extra, "flags": flagged(root), "still_gt0": orc.get("still_gt0", {}).get("AUC")}
    attribution = analysis.attribution(c["waterfall"]) if c["waterfall"] else []
    ranked = [r for r in attribution if r.get("largest")]
    ep["attribution"] = attribution
    ep["flaw"] = ranked[0] if ranked else None
    ep["fix"] = improvement(ep)
    if deep:  # the worst episodes: timelines, mesh and camera panels
        try:
            ep["section"] = analysis.episode_section(run, n, read, best=c["column"])
        except Exception as exc:  # an episode that stopped early: say why, keep the page
            ep["section_error"] = f"{type(exc).__name__}: {exc}"
    return ep


def evaluation_episode(run: Path, n: int) -> dict:
    root = run / f"episode_{n:06d}"
    e = episode_info(n)
    sel = load(root / "s6_select/selection.json")
    ep = {"number": n, "root": root, "names": [o["name"] for o in e["objects"]], "frames": int(e["num_frames"]),
          "flags": flagged(root), "selection": sel}
    if sel:
        cand = sel["candidate"]
        ep["candidate"], ep["cost"] = cand, sel["costs"][cand]
        js = load(root / f"s6_eval/{cand}/results/{root.name}.json") or {}
        ep["own_reference"] = js.get("metrics", {})
    try:
        ep["camera"] = analysis.camera_panel(root, e, {})  # no ground truth: the plane-height drift only
    except Exception:
        ep["camera"] = None
    return ep


# ---------------------------------------------------------------- drawing
def episode_bars(rows: list[tuple[str, float | None, float | None, float | None, str]], width: int = 680) -> str:
    """One row per episode: the selected AUC as a bar, still (grey dot) and S0.2's still at the true
    frame-0 placement (green ring)."""
    left, right, bh, gap, top = 120, 40, 14, 6, 10
    h = top + len(rows) * (bh + gap) + 30
    x = lambda v: left + (width - left - right) * max(0.0, min(1.0, v))  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="AUC per episode">']
    for t in np.linspace(0, 1, 6):
        out.append(f'<line x1="{x(t):.1f}" y1="{top - 4}" x2="{x(t):.1f}" y2="{h - 24}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{t:.1f}</text>')
    for i, (label, auc, still, gt0, kind) in enumerate(rows):
        y = top + i * (bh + gap)
        out.append(f'<text x="{left - 8}" y="{y + bh * 0.8:.1f}" class="label" text-anchor="end">{esc(label)}</text>')
        if auc is not None:
            out.append(f'<rect x="{left}" y="{y}" width="{x(auc) - left:.1f}" height="{bh}" class="bar {kind}">'
                       f'<title>{esc(label)}: {auc:.3f}</title></rect>'
                       f'<text x="{x(auc) + 5:.1f}" y="{y + bh * 0.8:.1f}" class="value">{auc:.3f}</text>')
        if still is not None:
            out.append(f'<circle cx="{x(still):.1f}" cy="{y + bh / 2:.1f}" r="3.6" class="dot still"><title>still {still:.3f}</title></circle>')
        if gt0 is not None:
            out.append(f'<circle cx="{x(gt0):.1f}" cy="{y + bh / 2:.1f}" r="4.6" class="dot gt"><title>S0.2 still at the true '
                       f'frame-0 placement {gt0:.3f}</title></circle>')
    out.append("</svg>")
    return ("\n".join(out) + '\n<p class="legend"><span class="key inverted">━ inverted</span> · <span class="key keyframe">'
            '━ keyframe</span> · <span class="key still">━ still</span> (bar: selected candidate) · ● our still rollout · '
            '○ S0.2 still at the true frame-0 placement</p>')


def value_bars(rows: list[tuple[str, float | None, str, str]], unit: str, width: int = 680, signed: bool = False) -> str:
    """Horizontal bars of one value per row (label, value, css class, note); signed: centred at 0."""
    left, right, bh, gap, top = 120, 120, 14, 6, 10
    vals = [abs(v) for _, v, _, _ in rows if v is not None] or [1.0]
    vmax = max(vals) * 1.05 or 1.0
    h = top + len(rows) * (bh + gap) + 30
    zero = left + (width - left - right) / 2 if signed else left
    span = (width - left - right) / (2 if signed else 1)
    x = lambda v: zero + span * v / vmax  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="{esc(unit)} per episode">']
    for t in (np.linspace(-vmax, vmax, 5) if signed else np.linspace(0, vmax, 5)):
        out.append(f'<line x1="{x(t):.1f}" y1="{top - 4}" x2="{x(t):.1f}" y2="{h - 24}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{t:.2g}</text>')
    for i, (label, v, cls, note) in enumerate(rows):
        y = top + i * (bh + gap)
        out.append(f'<text x="{left - 8}" y="{y + bh * 0.8:.1f}" class="label" text-anchor="end">{esc(label)}</text>')
        if v is not None:
            a, b = sorted((zero, x(v)))
            out.append(f'<rect x="{a:.1f}" y="{y}" width="{max(b - a, 1):.1f}" height="{bh}" class="bar {cls}">'
                       f'<title>{esc(label)}: {v:.3f} {esc(unit)}</title></rect>')
        out.append(f'<text x="{width - right + 6}" y="{y + bh * 0.8:.1f}" class="value">'
                   f'{fmt(v, 3) if v is not None else "–"} {esc(note)}</text>')
    out.append(f'<text x="{left}" y="{h - 8}" class="axis" text-anchor="end">{esc(unit)}</text></svg>')
    return "\n".join(out)


# ---------------------------------------------------------------- the page
def label(ep: dict) -> str:
    return f"ep{ep['number']} · {' + '.join(ep['names'])}"


def metric_cells(s: dict | None) -> list[str]:
    s = s or {}
    return [fmt(s.get("AUC")), fmt(s.get("SP-SR"), 2), fmt(s.get("MP-SR"), 2), fmt(s.get("MPPE cm"), 2),
            fmt(s.get("RPE cm"), 2), fmt(s.get("CD-O cm"), 3)]


def one_sentence(ep: dict) -> str:
    c, s = ep["choice"], ep["choice"]["scores"] or {}
    still = (c["still"] or {}).get("AUC")
    gains = [r for r in ep["attribution"] if r["loss"] is not None and r["loss"] < -0.005 and not r["stage"].startswith("stage-2")]
    gain = min(gains, key=lambda r: r["loss"]) if gains else None
    text = (f"{c['candidate']} scores {fmt(s.get('AUC'))} AUC"
            + (f", {s['AUC'] - still:+.3f} against still" if still is not None and s.get("AUC") is not None else "")
            + f" (MPPE {fmt(s.get('MPPE cm'), 1)} cm, CD-O {fmt(s.get('CD-O cm'), 2)} cm)")
    if gain:
        text += f"; the largest gain is {gain['stage']} ({-gain['loss']:+.3f})"
    if ep["flaw"]:
        text += f"; the largest loss is {ep['flaw']['stage']} ({ep['flaw']['loss']:.3f})"
    return text + "."


def diagnosis(ep: dict) -> list[str]:
    md = [f'<div class="diag">', "", f"### {label(ep)}", ""]
    c = ep["choice"]
    flaw, (fix, gain) = ep["flaw"], ep["fix"]
    md += [f"**Biggest flaw:** " + (f"{flaw['stage']} costs **{flaw['loss']:.3f} AUC** "
                                    f"(`{flaw['before']}` {fmt((flaw['b'] or {}).get('AUC'))} → `{flaw['after']}` "
                                    f"{fmt((flaw['a'] or {}).get('AUC'))})." if flaw else "no waterfall attribution."), "",
           f"**One improvement:** {fix}.", "",
           f"**First failing / warning bar:** {bar_text(ep['flags']['first_bar'])}"
           + (f" · stages in ERROR: {', '.join(ep['flags']['errors'])}" if ep["flags"]["errors"] else ""), ""]
    sec = ep.get("section")
    if sec is None:
        md += [f"Diagnosis panels unavailable: {ep.get('section_error', 'not computed')}.", "", "</div>", ""]
        return md
    # timeline per object: position error with S4a's phases and S3b's recovered frames
    for j, n in enumerate(sec["names"]):
        series = [(v, sec["errors"][v]["position_m"][:, j] * 100, cls)
                  for v, cls in (("raw", "raw"), (c["column"], "ours")) if v in sec["errors"]]
        if series:
            md += [f"**{n}: position error per frame** (official registration and alignment)", "",
                   analysis.timeline(series, sec["phases"][n], analysis.recovered_frames(sec["root"], n), "position error (cm)"), ""]
    # attribution by stage, the largest loss bold
    md += ["| Stage | AUC before → after | AUC lost | MPPE cm | Note |", "|---|---|---|---|---|"]
    for r in sec["attribution"]:
        b, a = r["b"] or {}, r["a"] or {}
        cells = [r["stage"], f"{fmt(b.get('AUC'))} → {fmt(a.get('AUC'))}", fmt(r["loss"]),
                 f"{fmt(b.get('MPPE cm'), 2)} → {fmt(a.get('MPPE cm'), 2)}", r["note"]]
        if r.get("largest"):
            cells = [f"**{x}**" if x else x for x in cells]
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    # mesh vs scan and camera
    for n, m in sec["mesh"].items():
        md += [f"**Mesh {n}:** CD-O {fmt(m['cdo_cm'], 2)} cm · size ours / scan {fmt(m['size_ratio'], 2)} · scan surface "
               f"within 1 cm {fmt(m.get('coverage_1cm'), 2)}", ""]
        if m.get("silhouettes"):
            md += [f'<img class="sil" alt="silhouettes of {esc(n)}" src="{m["silhouettes"]}"/>', ""]
    cam = sec["camera"]
    md += [f"**Camera:** path {fmt(cam['path_m'], 2)} m · plane-height spread at rest {fmt(cam['plane_height_spread_cm'], 2)} cm "
           f"(frames {cam['rest_window'][0]}–{cam['rest_window'][1]}) · S0.2 rest common RMS (camera) "
           f"{fmt(cam['rest_common_rms_cm'], 1)} cm vs rest difference RMS (objects) {fmt(cam['rest_diff_rms_cm'], 1)} cm"
           + (" (one object: n/a)" if cam["rest_common_rms_cm"] is None else ""), "", "</div>", ""]
    return md


def fix_ranking(pub: list[dict]) -> list[dict]:
    """Stages ranked by the public AUC they cost (mean over the public episodes scored), plus the
    placement and selection rows that the chain does not show."""
    n = max(len(pub), 1)
    rows = {}
    for ep in pub:
        for r in ep["attribution"]:
            if r["loss"] is None or r["stage"].startswith("stage-2"):
                continue
            row = rows.setdefault(r["stage"], {"stage": r["stage"], "episodes": [], "largest": 0, "sum": 0.0})
            if r["loss"] > 0.01:
                row["episodes"].append(ep["number"])
            row["sum"] += max(r["loss"], 0.0)
            row["largest"] += bool(r.get("largest"))
    out = []
    for stage, row in rows.items():
        kind, action = ACTIONS.get(stage, ("existing step", stage))
        out.append({"stage": stage, "episodes": row["episodes"], "largest_in": row["largest"], "auc_at_stake": row["sum"] / n,
                    "kind": kind, "action": action,
                    "note": "a ceiling (scan at the same poses); parallel to the chain" if stage == "S3a mesh" else ""})
    # frame-0 placement: our still rollout vs S0.2's still at the true placement (Kaggle aligns by the anchor only)
    two = [ep for ep in pub if len(ep["names"]) > 1 and ep["still_gt0"] is not None and (ep["choice"]["still"] or {}).get("AUC") is not None]
    if two:
        lost = [max(ep["still_gt0"] - ep["choice"]["still"]["AUC"], 0.0) for ep in two]
        out.append({"stage": "frame-0 placement, two objects", "episodes": [ep["number"] for ep, l in zip(two, lost) if l > 0.01],
                    "largest_in": None, "auc_at_stake": float(np.sum(lost)) / n, "kind": "Phase 2",
                    "action": "§8 phase 2: the second object anchored to the first (S4b), RPE in S6's selection",
                    "note": "our still rollout vs S0.2 still at the true frame-0 placement"})
    # the evaluation rule picks on Kaggle; on public episodes it can be checked against the score
    miss = []
    for ep in pub:
        sel = ep["choice"]["selection"]
        if sel and sel.get("evaluation_rule_pick") and sel["evaluation_rule_pick"] != sel["candidate"]:
            miss.append((ep["number"], sel["scores"][sel["candidate"]]["AUC"] - sel["scores"][sel["evaluation_rule_pick"]]["AUC"]))
    if miss:
        out.append({"stage": "S6 evaluation selection rule", "episodes": [m[0] for m in miss if m[1] > 0.01], "largest_in": None,
                    "auc_at_stake": float(sum(m[1] for m in miss)) / n, "kind": "new (proposal for James)",
                    "action": "a better observation cost for evaluation episodes (e.g. weight carry frames, add rotation)",
                    "note": f"the rule picks another candidate than the score on {len(miss)} public episodes"})
    return sorted(out, key=lambda r: -r["auc_at_stake"])


def markdown(run: Path, pub: list[dict], evl: list[dict], ranking: list[dict], refs: dict) -> str:
    today = date.today().isoformat()
    scored = [ep for ep in pub if ep["choice"]["scores"]]
    not_rollouts = [ep["number"] for ep in scored if not ep["choice"]["rollout"]]
    md = [f"# Phase 1 review: run `{run.name}` ({today})", "",
          f"Public episodes: {len(scored)} of {len(pub)} scored with the official Track 3 metric code (`pathb/gt.py`). "
          f"Evaluation episodes: {sum(1 for ep in evl if ep['selection'])} of {len(evl)} selected; **no ground truth "
          "exists for them, so section d uses only self-consistency signals.**", ""]
    if not_rollouts:
        md += [f"> **Not rollouts:** ep{', ep'.join(map(str, not_rollouts))} have no S6 selection; their numbers are the best "
               "waterfall column (a reference or realised trajectory, not a policy).", ""]
    if ranking:
        md += [f"**At a glance:** mean AUC {fmt(np.mean([ep['choice']['scores']['AUC'] for ep in scored])) if scored else '–'} "
               f"over the public episodes; the largest fixable loss is **{ranking[0]['stage']}** "
               f"({ranking[0]['auc_at_stake']:.3f} mean AUC).", ""]
    # (a)
    mean = {m: (float(np.mean([ep["choice"]["scores"][m] for ep in scored if ep["choice"]["scores"].get(m) is not None]))
                if scored else None) for m in METRICS}
    still_mean = [ep["choice"]["still"]["AUC"] for ep in scored if ep["choice"]["still"]]
    md += ["## a. Headline", "",
           f"Mean of the six metrics over {len(scored)} public episodes for the selected candidate.", "",
           "| Metric | Ours (selected) | S0.2 still at true frame 0 | S0.2 recon / place (Oct 17) | Kaggle still submission | "
           "Leaderboard top |", "|---|---|---|---|---|---|"]
    for m in METRICS:
        fm = (lambda v: fmt(v, 3)) if m in ("AUC", "CD-O cm") else (lambda v: fmt(v, 2))
        md.append(f"| {m} | **{fm(mean[m])}** | {fm(refs['still_gt0'].get(m))} | {fm(refs['recon/place'].get(m))} | "
                  f"{fm(KAGGLE_STILL_AUC) if m == 'AUC' else '–'} | {fm(LEADERBOARD_TOP.get(m))} |")
    md += ["", analysis.bar_chart(
        [("ours, selected", mean["AUC"], "ours"), ("ours, still rollout", float(np.mean(still_mean)) if still_mean else None, "still")],
        [("S0.2 still at true frame 0", refs["still_gt0"]["AUC"]), ("S0.2 recon / place", refs["recon/place"]["AUC"]),
         ("Kaggle still submission", KAGGLE_STILL_AUC), ("leaderboard top", LEADERBOARD_TOP["AUC"])]), "",
           "Kaggle still submission and leaderboard top: James, 2026-10-23 (the leaderboard scores the evaluation split; "
           "the other columns are public episodes).", "",
           episode_bars([(f"ep{ep['number']}", ep["choice"]["scores"]["AUC"], (ep["choice"]["still"] or {}).get("AUC"),
                          ep["still_gt0"], ep["choice"]["candidate"] if ep["choice"]["rollout"] else "other")
                         for ep in sorted(scored, key=lambda x: -x["choice"]["scores"]["AUC"])]), ""]
    ranked = sorted(scored, key=lambda x: -x["choice"]["scores"]["AUC"])
    head = "| Episode | Candidate | AUC | SP-SR | MP-SR | MPPE cm | RPE cm | CD-O cm | Still AUC |\n|---|---|---|---|---|---|---|---|---|"
    # (b)
    md += ["## b. Best 5 public episodes", "", head]
    for ep in ranked[:5]:
        md.append(f"| {label(ep)} | `{ep['choice']['candidate']}` | " + " | ".join(metric_cells(ep["choice"]["scores"]))
                  + f" | {fmt((ep['choice']['still'] or {}).get('AUC'))} |")
    md.append("")
    md += [f"- **ep{ep['number']}:** {one_sentence(ep)}" for ep in ranked[:5]] + [""]
    # (c)
    worst = ranked[-5:][::-1]
    md += ["## c. Worst 5 public episodes", "", head]
    for ep in worst:
        md.append(f"| {label(ep)} | `{ep['choice']['candidate']}` | " + " | ".join(metric_cells(ep["choice"]["scores"]))
                  + f" | {fmt((ep['choice']['still'] or {}).get('AUC'))} |")
    md += ["", "Each block: the biggest flaw (the largest AUC loss along the waterfall chain), one improvement with its "
           "spec step and expected gain, the first bar any stage flagged, per-frame error with S4a's phases (shaded) and "
           "S3b's recovered frames (ticks), the stage attribution, the mesh against the scan and the camera.", ""]
    for ep in worst:
        md += diagnosis(ep)
    # (d)
    sel_ev = [ep for ep in evl if ep["selection"]]
    md += ["## d. Evaluation split (no ground truth)", "",
           "**Self-consistency only.** Observation cost: the selected rollout against the S4b keyframe reference on its "
           "held and trusted frames (frames > 10 cm / 0.5 rad off count 10 cm, \"termination-equivalent\"). Own reference: "
           "FlashCHORD's error against the candidate's own task. Neither is the Kaggle score.", ""]
    if pub:
        pairs = [(ep["choice"]["selection"]["costs"][ep["choice"]["candidate"]]["cost_m"], ep["choice"]["scores"]["AUC"])
                 for ep in scored if ep["choice"]["selection"]
                 and ep["choice"]["selection"]["costs"][ep["choice"]["candidate"]]["cost_m"] is not None]
        if len(pairs) >= 4:
            from scipy.stats import spearmanr
            rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])[0]
            md += [f"How far to trust it: on {len(pairs)} public episodes the observation cost and the official AUC have "
                   f"Spearman ρ = **{rho:.2f}** (−1 would be a perfect ranking).", ""]
    if sel_ev:
        rows = sorted(sel_ev, key=lambda x: -(x["cost"]["cost_m"] or 0))
        md += [value_bars([(f"ep{ep['number']}", (ep["cost"]["cost_m"] or 0) * 100, ep["candidate"],
                            f"{ep['cost']['termination_frames']}/{ep['cost']['frames']} term.") for ep in rows],
                          "observation cost (cm)"), "",
               '<p class="legend">Bar colour: selected candidate (<span class="key inverted">inverted</span>, '
               '<span class="key keyframe">keyframe</span>, <span class="key still">still</span>); right: '
               'termination-equivalent frames / observed frames.</p>', "",
               "| Episode | Candidate | Observation cost cm | Termination-equivalent frames | First at frame | Own-reference "
               "position error cm | Own-reference ADD-AUC | First failing / warning stage |", "|---|---|---|---|---|---|---|---|"]
        for ep in sorted(sel_ev, key=lambda x: x["number"]):
            c, o = ep["cost"], ep["own_reference"]
            md.append(f"| {label(ep)} | `{ep['candidate']}` | {fmt((c['cost_m'] or 0) * 100, 2)} | {c['termination_frames']} of "
                      f"{c['frames']} | {c.get('first_termination_frame', '–') if c.get('first_termination_frame') is not None else '–'} | "
                      f"{fmt(o.get('object_position_error_m') * 100 if o.get('object_position_error_m') is not None else None, 2)} | "
                      f"{fmt(o.get('add_auc'))} | {ep['flags']['first_failing_stage'] or '–'} / {ep['flags']['first_warning_stage'] or '–'} |")
        md += ["", "### Five most likely lowest (highest observation cost)", ""]
        for ep in rows[:5]:
            c = ep["cost"]
            bars = [b for b in ep["flags"]["bars"] if b["status"] in ("FAIL", "WARN")][:4]
            cam = ep["camera"] or {}
            md += [f"**{label(ep)}** — `{ep['candidate']}`, cost {fmt((c['cost_m'] or 0) * 100, 2)} cm, "
                   f"{c['termination_frames']} of {c['frames']} observed frames termination-equivalent"
                   + (f" (first at frame {c['first_termination_frame']})" if c.get("first_termination_frame") is not None else "")
                   + f"; camera plane-height spread at rest {fmt(cam.get('plane_height_spread_cm'), 2)} cm.", ""]
            md += [f"- {bar_text(b)}" for b in bars] or ["- no stage flagged a bar"]
            md.append("")
    else:
        md += ["No evaluation episode has an S6 selection yet.", ""]
    # (e)
    md += ["## e. Candidate selection", "", "| Split | " + " | ".join(CANDIDATES) + " | not selected |",
           "|---|" + "---|" * (len(CANDIDATES) + 1)]
    for split, eps, get in (("public", pub, lambda ep: ep["choice"]["candidate"] if ep["choice"]["rollout"] else None),
                            ("evaluation", evl, lambda ep: ep.get("candidate"))):
        picks = [get(ep) for ep in eps]
        md.append(f"| {split} | " + " | ".join(str(picks.count(c)) for c in CANDIDATES)
                  + f" | {picks.count(None)} |")
    md.append("")
    gaps = [(ep, ep["choice"]["scores"]["AUC"] - ep["choice"]["still"]["AUC"]) for ep in scored if ep["choice"]["still"]]
    if gaps:
        md += ["Chosen minus still, per public episode (AUC):", "",
               value_bars([(f"ep{ep['number']}", g, "gain" if g > 1e-9 else "other", ep["choice"]["candidate"])
                           for ep, g in sorted(gaps, key=lambda x: -x[1])], "AUC, chosen − still", signed=True), ""]
        md += ["| Episode | Chosen | Chosen AUC | Still AUC | Gap | Evaluation rule's pick |", "|---|---|---|---|---|---|"]
        for ep, g in sorted(gaps, key=lambda x: x[0]["number"]):
            sel = ep["choice"]["selection"] or {}
            md.append(f"| ep{ep['number']} | `{ep['choice']['candidate']}` | {fmt(ep['choice']['scores']['AUC'])} | "
                      f"{fmt(ep['choice']['still']['AUC'])} | {g:+.3f} | `{sel.get('evaluation_rule_pick', '–')}` |")
        md.append("")
    # (f)
    md += ["## f. What to fix next, ranked", "",
           "AUC at stake = the stage's AUC loss summed over the public episodes, divided by their number (what the public "
           "mean would gain if the stage lost nothing). Losses along the chain add up; the mesh row is a ceiling beside it.", "",
           "| # | Stage | Episodes touched (loss > 0.01) | Largest loss in | AUC at stake | Action | Kind |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranking, 1):
        md.append(f"| {i} | {r['stage']} | {len(r['episodes'])}: " + (", ".join(f"ep{x}" for x in r["episodes"]) or "–")
                  + f" | {r['largest_in'] if r['largest_in'] is not None else '–'} | **{r['auc_at_stake']:.3f}** | {r['action']}"
                  + (f" ({r['note']})" if r["note"] else "") + f" | {r['kind']} |")
    md += ["", "Rows of kind \"new\" are proposals for James, not applied.", ""]
    # appendix
    md += ["## Appendix: files", "",
           f"- Run: `{run}`",
           "- Per episode: `episode_XXXXXX/waterfall.json` (columns), `s6_select/selection.json` (choice, scores, costs), "
           "`s6_eval/<candidate>/results/episode_XXXXXX.{parquet,json}` (rollout), `loss_report.{md,json}` (bars), "
           "`s4a_phases/<object>/phases.json`, `s3b_track/<object>/track_meta.json`",
           f"- S0.2 oracle: `{analysis.ORACLE.relative_to(REPO)}`",
           "- Metric code: `pathb/gt.py` → `tools/track3/score_local.py` (the kit's official Track 3 metrics)",
           f"- This page: `path_b/reports/phase1_review_{today}.{{md,html,json}}` (`pathb/review.py`)", ""]
    return "\n".join(md)


def main(run: Path, numbers: list[int]) -> tuple[Path, Path, list[dict]]:
    read: list = []
    public = [n for n in numbers if episode_info(n)["split"] == "public"]
    evaluation = [n for n in numbers if episode_info(n)["split"] == "evaluation"]
    pub = [public_episode(run, n, read, deep=False) for n in public]
    scored = sorted((ep for ep in pub if ep["choice"]["scores"]), key=lambda x: x["choice"]["scores"]["AUC"])
    for ep in scored[:5]:  # the worst five get the diagnosis panels
        ep.update(public_episode(run, ep["number"], read, deep=True))
    evl = [evaluation_episode(run, n) for n in evaluation]
    ranking = fix_ranking(pub)
    md = markdown(run, pub, evl, ranking, oracle_means())
    REPORTS.mkdir(exist_ok=True)
    stem = REPORTS / f"phase1_review_{date.today().isoformat()}"
    summary = {"run": str(run), "public": {str(ep["number"]): {"candidate": ep["choice"]["candidate"],
                                                               "rollout": ep["choice"]["rollout"],
                                                               "scores": ep["choice"]["scores"]} for ep in pub},
               "evaluation": {str(ep["number"]): {"candidate": ep.get("candidate"), "cost": ep.get("cost")} for ep in evl},
               "ranking": ranking}
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    stem.with_suffix(".md").write_text(md)
    stem.with_suffix(".html").write_text(analysis.to_html(md, "Phase 1 review", EXTRA_CSS))
    return stem.with_suffix(".md"), stem.with_suffix(".html"), ranking

