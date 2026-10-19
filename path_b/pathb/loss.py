"""loss_report.json and loss_report.md per episode: every stage's verdict, the measurements outside
their bars, and (public episodes, after `run.py waterfall`) how many points each step costs."""
from __future__ import annotations

import json
from pathlib import Path

FLAGGED = ("WARN", "FAIL")


def write(root: Path, reports: dict[str, dict]) -> dict:
    stages = []
    for name, r in reports.items():
        stages.append({"stage": name, "status": r["status"], "seconds": r["seconds"], "gpu_peak_mib": r["gpu_peak_mib"],
                       "skip_reason": r.get("skip_reason"),
                       "error": r["error"].strip().splitlines()[-1] if r.get("error") else None,
                       "flagged": [m for m in r["metrics"] if m["status"] in FLAGGED]})
    waterfall_path = root / "waterfall.json"
    report = {
        "episode": root.name,
        "first_failing_stage": next((s["stage"] for s in stages if s["status"] in ("FAIL", "ERROR")), None),
        "first_warning_stage": next((s["stage"] for s in stages if s["status"] == "WARN"), None),
        "stages": stages,
        "waterfall": json.loads(waterfall_path.read_text()) if waterfall_path.is_file() else None,
        "measurements": {name: r["metrics"] for name, r in reports.items()},
    }
    (root / "loss_report.json").write_text(json.dumps(report, indent=1) + "\n")
    (root / "loss_report.md").write_text(markdown(report))
    return report


def fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else "" if v is None else str(v)


def markdown(r: dict) -> str:
    lines = [f"# Path B loss report: {r['episode']}", "",
             f"First failing stage: **{r['first_failing_stage'] or 'none'}** · "
             f"first warning: {r['first_warning_stage'] or 'none'}", "",
             "| Stage | Status | Time (s) | GPU peak (MiB) | Outside its bar |", "|---|---|---|---|---|"]
    for s in r["stages"]:
        flagged = "; ".join(f"{m['name']} = {fmt(m['value'])} {m['unit']} ({m['bar']})" for m in s["flagged"])
        extra = s["error"] or s["skip_reason"] or ""
        lines.append(f"| {s['stage']} | {s['status']} | {s['seconds']} | {fmt(s['gpu_peak_mib'])} | "
                     f"{'; '.join(x for x in (flagged, extra) if x)} |")
    w = r["waterfall"]
    if w:
        lines += ["", "## Score waterfall (official metric code, public ground truth)", "",
                  "| Step | AUC | MPPE (cm) | MP-SR | CD-O (cm) |", "|---|---|---|---|---|"]
        for step in w["variants"]:
            s = w["scores"].get(step)
            if s:
                lines.append(f"| {step} | {s['AUC']:.3f} | {s['MPPE cm']:.2f} | {s['MP-SR']:.2f} | {s['CD-O cm']:.2f} |")
        lines += ["", "| Where AUC goes | Points |", "|---|---|"]
        lines += [f"| {k} | {fmt(v)} |" for k, v in w["auc_attribution"].items()]
    lines += ["", "## Every measurement", "", "| Stage | Measurement | Value | Unit | Bar | Kind | Status | Note |",
              "|---|---|---|---|---|---|---|---|"]
    for stage, metrics in r["measurements"].items():
        lines += [f"| {stage} | {m['name']} | {fmt(m['value'])} | {m['unit']} | {m['bar']} | {m['kind']} | "
                  f"{m['status']} | {m['note']} |" for m in metrics]
    return "\n".join(lines) + "\n"
