"""Pick each scored episode's policy for a submission plan from several training runs, without ground truth.

For every episode in the base plan, the candidates are the base plan's own policy (if it has one) and every
run named ep<EP>_<prefix>* under flash_runs/diag that finished its evaluation. Each is ranked by its
own-reference score (the evaluator's add_auc against the trajectory the policy was trained on), which ranked
runs of the same episode in the same order as the true score on the public clips. The best candidate is kept if
it reaches --cutoff; otherwise the episode keeps the base plan's entry (a hold-still policy).
Usage: python tools/track3/select_policies.py <base plan.json> <out plan.json> <run prefix> [...] --scenes <runs.json> [--cutoff 0.25]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DIAG = Path('/data/user_data/edmondwa/v2d/flash_runs/diag')


def own_score(run: Path, ep: int) -> float | None:
    f = run / 'results' / f'episode_{ep:06d}.json'
    if not (run / 'eval.done').is_file() or not f.is_file():
        return None
    return float(json.loads(f.read_text())['metrics']['add_auc'])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('base', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('prefixes', nargs='+')
    p.add_argument('--cutoff', type=float, default=0.25)
    p.add_argument('--scenes', type=Path, required=True, help='JSON {run name: load_retarget scene dir} (job logs do not record it)')
    args = p.parse_args()
    base = json.loads(args.base.read_text())
    known = json.loads(args.scenes.read_text())
    out, rows = {}, []
    for ep_s, entry in sorted(base.items(), key=lambda kv: int(kv[0])):
        ep = int(ep_s)
        cands = []
        if entry.get('kind') == 'policy':
            s = own_score(Path(entry['run']), ep)
            if s is not None:
                cands.append((s, entry['run'], entry['scene'], 'base plan'))
        for prefix in args.prefixes:
            for run in sorted(DIAG.glob(f'ep{ep}_{prefix}*')):
                s = own_score(run, ep)
                scene = known.get(run.name)
                if s is not None and scene:
                    cands.append((s, str(run), scene, run.name))
        best = max(cands, default=None)
        if best and best[0] >= args.cutoff:
            out[ep_s] = {'run': best[1], 'scene': best[2], 'kind': 'policy', 'tracking': round(best[0], 3), 'from': best[3]}
        else:
            out[ep_s] = dict(entry, note='kept: no candidate reached the cutoff') if entry.get('kind') != 'policy' else entry
        rows.append((ep, [(round(c[0], 3), c[3]) for c in sorted(cands, reverse=True)], out[ep_s]['kind'], out[ep_s].get('from', '')))
    args.out.write_text(json.dumps(out, indent=1))
    for ep, cs, kind, src in rows:
        print(f'ep{ep:<3} {kind:<13} {src:<22} candidates {cs}')


if __name__ == '__main__':
    main()
