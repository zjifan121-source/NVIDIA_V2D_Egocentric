#!/bin/bash
# kqstat.sh [RUN...]: sweep results of the given runs (default: all with sweep.json) + tail of kq.out
cd /tmp/haisenk_v2d/runs
[ $# -eq 0 ] && set -- $(ls -d */ | tr -d /)
for r in "$@"; do
python3 - "$r" <<'PY'
import json, sys
r = sys.argv[1]
try:
    rows = json.load(open(f"{r}/sweep.json"))
    print(f"{r:24s} best {open(f'{r}/best.txt').read().strip()} | " + " ".join(f"{x['step']//1000000}M:{x['auc']:.3f}" for x in sorted(rows, key=lambda x: x['step'])))
except Exception:
    try:
        own = json.load(open(next(iter(__import__('glob').glob(f"{r}/results/episode_*.json")))))["metrics"]
        print(f"{r:24s} (final only) {own['add_auc']:.3f} {[round(x,3) for x in own.get('add_auc_per_body') or []]}")
    except Exception:
        print(f"{r:24s} -")
PY
done
echo "-- kq.out"; tail -${N:-8} /tmp/haisenk_v2d/kq.out
