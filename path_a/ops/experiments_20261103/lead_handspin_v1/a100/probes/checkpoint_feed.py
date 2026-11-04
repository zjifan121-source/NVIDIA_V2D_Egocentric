"""Appends every newly evaluated checkpoint under /root/runs to a feed the review session can watch.
One JSON line per checkpoint in /root/probes/NEW_CHECKPOINTS.jsonl, oldest first, plus NEW_CHECKPOINTS.md.
Read only; no Kaggle, no uploads. Runs every 60 s as a daemon (checkpoint_feed.sh)."""
import glob, json, os, time
from datetime import datetime, timezone
FEED='/root/probes/NEW_CHECKPOINTS.jsonl'; MD='/root/probes/NEW_CHECKPOINTS.md'
STOPPED={'ep30_rawexcl_def_s200','ep25_rawexcl_def_s200','ep28_v8pot_def_s200','ep28_r2_def_s200','ep33_def_s200','ep33_rawexcl_def_s200r','ep33_r1_def_s200','ABANDONED_ep28_oldpot_def_s200'}
seen=set()
if os.path.exists(FEED):
    for l in open(FEED):
        try: d=json.loads(l); seen.add(d['run']+'/'+str(d['step']))
        except Exception: pass
rows=[]
for done in sorted(glob.glob('/root/runs/*/early_*/eval.done'),key=os.path.getmtime):
    early=os.path.dirname(done); run=os.path.basename(os.path.dirname(early)); step=int(early.rsplit('_',1)[1])
    if run in STOPPED or run+'/'+str(step) in seen: continue
    res=glob.glob(early+'/results/episode_*.json')
    if not res: continue
    r=json.load(open(res[0])); ep=int(os.path.basename(res[0])[8:14])
    scene=open(f'/root/runs/{run}/SCENE').read().strip() if os.path.exists(f'/root/runs/{run}/SCENE') else None
    row=dict(seen_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),run=run,episode=ep,step=step,own_ref=round(r['metrics']['add_auc'],4),
             per_body=[round(x,3) for x in r['metrics']['add_auc_per_body']],results_dir=early+'/results',parquet=res[0][:-5]+'.parquet',json=res[0],scene=scene,
             recipe=('kenf16c3m' if 'kenf16c3m' in run else 'kenf16' if 'kenf16' in run else 'wide' if 'wide' in run else 'def'),
             run_finished=os.path.exists(f'/root/runs/{run}/TRAIN_EXITED'))
    rows.append(row); seen.add(run+'/'+str(step))
if rows:
    with open(FEED,'a') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
allrows=[json.loads(l) for l in open(FEED)] if os.path.exists(FEED) else []
L=['# Evaluated checkpoints on the A100 (lead line)','',f'Updated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} by checkpoint_feed.py. Newest last. Machine-readable: NEW_CHECKPOINTS.jsonl. Stopped runs are excluded. No upload has been made from any of these.','',
   '| Seen (UTC) | Run | Ep | Step | Own-ref | Per body | Recipe | Results dir | Scene |','|---|---|---:|---:|---:|---|---|---|---|']
for d in allrows:
    L.append(f"| {d['seen_utc'][11:16]} | `{d['run']}` | {d['episode']} | {d['step']//1000000}M | {d['own_ref']} | {' / '.join(str(x) for x in d['per_body'])} | {d['recipe']} | `{d['results_dir']}` | `{d['scene']}` |")
open(MD,'w').write('\n'.join(L)+'\n')
print(f'{len(rows)} new, {len(allrows)} total')
