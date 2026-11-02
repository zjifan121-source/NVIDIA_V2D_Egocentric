"""S4b2 trackability: pathb/trackability.py's GT-free features of every S5 candidate (task "trackability
score", 2026-11-02), after s5_task and before s6_train: the unsupported-motion share of the reference hand
(UMS), the reference's contact coverage (RCC), the rests' settle in S5's scene (RSS, drivers/settle.py in
$SETTLE_PY; CoACD in $COACD_PY, default the same venv) and frame-0 support. Moving frames and rests come
from S3c's invert report of each object. Out: s4b2_trackability/<candidate>/trackability.json (the task asked
for s4b_reference/<candidate>/, but S4b's folder is shared by symlink between runs and the features depend on
this run's S5). Health only: no bar (V4 S6's selection decides what to do with them).
"""
from __future__ import annotations

import json

from .. import trackability
from ..reference import CANDIDATES
from ..report import Metric, StageResult
from ..runner import write_json


def invert_reports(ctx) -> dict:
    out = {}
    for name in ctx.objects:
        f = ctx.root / f"s3c_invert/{name}/invert_report.json"
        out[name] = json.loads(f.read_text()) if f.is_file() else None
    return out


def run(ctx) -> StageResult:
    d = ctx.dir("s4b2_trackability")
    cands = [c for c in CANDIDATES if (ctx.root / f"s5_task/{c}/task.json").is_file()]
    if not cands:
        return StageResult([], skip="no S5 task (s5_task/<candidate>/task.json)")
    reports = invert_reports(ctx)
    metrics, outputs = [], {}
    for cand in cands:  # health only: an error here must not stop s6_train
        try:
            f = trackability.features(ctx.root / f"s5_task/{cand}", cand, reports)
        except Exception as e:  # noqa: BLE001
            metrics.append(Metric(f"{cand}: trackability", "error", note=f"{type(e).__name__}: {e}"[:300]))
            continue
        out = d / f"{cand}/trackability.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, f)
        outputs[cand] = out
        rss = f["rss"]
        metrics += [Metric(f"{cand}: UMS", f["ums"], "", note=f"{f['n_moving']} moving frames, longest unsupported "
                           f"run {f['ums_run']} frames, opposed-tips variant {f['ums_opp']:.2f}"),
                    Metric(f"{cand}: RCC >= 1 contact", f["rcc1"], "",
                           note=f">= 2: {f['rcc2']:.2f}" if f["rcc2"] is not None else "no moving frames"),
                    Metric(f"{cand}: RSS frame 0", rss.get("rss0_cm"), "cm",
                           note=rss.get("error") or f"{rss['rss0_deg']:.1f} deg; worst rest {rss['rss_max_cm']:.2f} cm / "
                           f"{rss['rss_max_deg']:.1f} deg; {rss['unsettled_rests']} of {len(rss['rests'])} rests unsettled"),
                    Metric(f"{cand}: frame-0 support", "yes" if f["support_ok"] else "no", "",
                           note="; ".join(f"{k}: {'disk' if v['disk'] else 'object' if v['object'] else 'none'}"
                                          for k, v in f["support"]["objects"].items()))]
    return StageResult(metrics, outputs)
