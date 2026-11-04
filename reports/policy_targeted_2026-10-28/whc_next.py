"""Pick, per episode, the best finished WHC/WHCc run whose best checkpoint has a kept state, and write a low-LR continuation trial.

python whc_next.py QUEUED_NAMES_FILE   (prints trial files to submit; a run is continued once)
"""
import json, re, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent
D = Path('/data/user_data/edmondwa/v2d/flash_runs/diag')
queued = set(Path(sys.argv[1]).read_text().split())
MARGIN = 0.08  # continuation gained ~0.08 on Ken's scenes; skip runs further than that below the episode's best


def best(run):
    f = run / 'checkpoint_evaluations.log'
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.startswith('{')] if f.exists() else []
    return max(rows, key=lambda r: r['auc']) if rows else None


top, cands = {}, {}
for run in D.iterdir():
    m = re.match(r'ep(\d+)_(WMIX|WRS|KXW|KWARM|WHC|WHCc|WG)_', run.name)
    if not m or run.name in queued or not (run / 'train.done').exists():
        continue
    b = best(run)
    if not b:
        continue
    ep = int(m[1]); top[ep] = max(top.get(ep, 0), b['auc'])
    st = next((p for p in (run / 'keep' / f'state_{b["step"]}.safetensors', run / 'train' / f'state_{b["step"]}.safetensors') if p.exists()), run / 'keep' / 'none')
    if m[2] in ("WHC", "WHCc", "WG") and st.exists() and not (run / '.continued').exists():
        if b['auc'] > cands.get(ep, (0,))[0]:
            cands[ep] = (b['auc'], run, b['step'])
for ep, (auc, run, step) in sorted(cands.items()):
    if auc < top[ep] - MARGIN:
        continue
    out = subprocess.check_output(['python3', str(H / 'whc_cont.py'), run.name, str(step)], text=True).strip()
    (run / '.continued').write_text(out + '\n')
    print(out, f'# ep{ep} {auc:.3f} (episode best {top[ep]:.3f})', file=sys.stderr)
    print(out)
