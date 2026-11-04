"""Cross-batch summary of every evaluated GPU case (v1..v6) -> ALL_CASES.csv and ALL_CASES.md"""
import json, csv
from pathlib import Path
L=Path('/Volumes/E/Code/V2D/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103'); S=L/'lead_sprint_bundle_v1'
rows=[]
for v in ['v%d'%i for i in range(1,8)]:
    p=L/f'lead_retrack_recovery_{v}'/'run_v1/SELECTION.json'
    if not p.exists(): continue
    sel=json.load(open(p)); batch={c['case']:c for c in json.load(open(L/f'lead_retrack_recovery_{v}'/'FROZEN_BATCH.json'))['cases']}
    for name,c in sel['cases'].items():
        b=batch.get(name,{})
        if c.get('state')!='EVALUATED': rows.append(dict(batch=v,case=name,episode=b.get('episode'),object=b.get('object'),variant=b.get('variant'),state=c.get('state'))); continue
        rows.append(dict(batch=v,case=name,episode=c['episode'],object=c['object'],variant=c['variant'],state='EVALUATED',mesh=b.get('label','own'),anchor=b.get('anchor_frame',0),
            proxy_current=round(c['current']['full_proxy'],4),proxy_candidate=round(c['candidate']['full_proxy'],4),delta=round(c['delta'],4),
            rot_max=round(c['candidate']['motion']['rot_max_deg'],1),n_over30=c['candidate']['motion'].get('n_rot_steps_over_30'),trans_p95=round(c['candidate']['motion']['trans_p95_mm'],1),trans_max=round(c['candidate']['motion'].get('trans_max_mm',0),1),
            recoveries=len(c.get('recovery_frames',[])),passed=c['passed'],passed_strict=c.get('passed_strict')))
rows.sort(key=lambda r:(r.get('episode') or 0,r.get('object') or '',r['batch']))
keys=['batch','case','episode','object','variant','mesh','anchor','state','proxy_current','proxy_candidate','delta','rot_max','n_over30','trans_p95','trans_max','recoveries','passed','passed_strict']
with open(S/'ALL_CASES.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow({k:r.get(k,'') for k in keys}) for r in rows]
md=['| 段 | 物体 | 批次/变体 | mesh | 当前→候选 | Δ | 最大旋转步 | >30° | 平移p95/max | 重注册 | 通过 |','|---|---|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in rows:
    if r['state']!='EVALUATED': md.append(f"| ep{r['episode']:02d} | {r['object']} | {r['batch']}/{r['variant']} | | 未完成 | | | | | | |"); continue
    md.append(f"| ep{r['episode']:02d} | {r['object']} | {r['batch']}/{r['variant']} | {r['mesh']} | {r['proxy_current']:.4f}→{r['proxy_candidate']:.4f} | {r['delta']:+.4f} | {r['rot_max']:.0f}° | {r['n_over30']} | {r['trans_p95']:.0f}/{r['trans_max']:.0f} | {r['recoveries']} | {'是' if r['passed'] else '否'} |")
(S/'ALL_CASES.md').write_text('\n'.join(md)+'\n'); print(len(rows),'cases summarised;',sum(1 for r in rows if r.get('passed')),'pass')
