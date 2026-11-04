"""Read-only: print one JSON list of training runs on this host (progress, rate, ETA, best evaluated checkpoint)."""
import glob, json, os, re, subprocess, sys
alive = subprocess.run('pgrep -af train_flash_sac', shell=True, capture_output=True, text=True).stdout
out = []
for pat in sys.argv[1:]:
    for d in sorted(glob.glob(pat)):
        n = os.path.basename(d)
        if not re.match(r'ep\d+', n) or not os.path.exists(d + '/train.log'):
            continue
        tail = open(d + '/train.log', errors='ignore').read()[-30000:]
        m = re.findall(r'training progress: ([\d,]+)/([\d,]+) environment steps \(([\d.]+)%\); ([\d,]+) env steps/s; elapsed ([\d:]+); ETA ([\d:]+)', tail)
        last = m[-1] if m else None
        best = None
        try:
            ev = json.load(open(d + '/checkpoint_selection.json'))
            late = [x for x in ev if x['step'] >= 70e6]
            pool = late or ev
            b = max(pool, key=lambda x: x['auc'])
            best = dict(auc=round(b['auc'], 4), step=b['step'], per_body=[round(v, 3) for v in b['per_body']], n=len(ev), late=bool(late), run=b.get('run'))
        except Exception:
            pass
        out.append(dict(name=n, path=d, running=('output_dir=' + d) in alive, done=os.path.exists(d + '/train.done'), pct=float(last[2]) if last else 0.0, rate=int(last[3].replace(',', '')) if last else None, eta=last[5] if last else None, best=best))
print(json.dumps(out))
