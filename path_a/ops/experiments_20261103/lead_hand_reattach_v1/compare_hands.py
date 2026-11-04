"""Tabulate the robot-link contact metric (HAND_CONTACT.json) and loader output checks (OUTPUT_CHECK.json) across the
built variants of each episode: control (v1 bundle as delivered), rawhands (loader attach, every hand to every lifted
object), rawclean (cleaned + smoothed raw hands, no attach), rawexcl (exclusive attach). Writes
HAND_REATTACH_COMPARISON.json and .md. Usage: compare_hands.py"""
from pathlib import Path
import json
import os
O=Path(os.environ.get('LEAD_HAND_OUT',str(Path(__file__).resolve().parent))); variants=['control','rawclean','rawhands','rawexcl']
rows=[]; md=['| 段 | 物体/手 | '+' | '.join(f'{v}: 中位距离 cm / ≤1 cm 帧占比' for v in variants)+' |','|---|---|'+'|'.join('---' for _ in variants)+'|']
out={}
for ep in sorted({int(d.name[2:4]) for d in (O/'scenes').iterdir() if d.is_dir()}):
    data={}
    for v in variants:
        sc=O/'scenes'/f'ep{ep:02d}_{v}'
        if (sc/'HAND_CONTACT.json').exists(): data[v]=dict(contact=json.loads((sc/'HAND_CONTACT.json').read_text())['objects'],check={k:vv for k,vv in json.loads((sc/'OUTPUT_CHECK.json').read_text()).items() if k in ['task_error_cm','penetration_frames','frame0_seating_cm','cleaning']} if (sc/'OUTPUT_CHECK.json').exists() else None)
    out[ep]=data
    objs=list(next(iter(data.values()))['contact'].keys()) if data else []
    for obj in objs:
        name=obj.split('_',1)[-1] if obj.startswith(('control_','rawhands_','rawclean_','rawexcl_')) else obj
        for side in ['left','right']:
            cells=[]
            for v in variants:
                d=data.get(v,{}).get('contact',{})
                key=next((k for k in d if k.endswith(name)),None); c=d.get(key,{}).get(side) if key else None
                cells.append(f"{c['closest_link_to_surface_cm_median']} / {c['within_1cm']}" if c else '—')
            md.append(f"| {ep} | {name}/{side} | "+' | '.join(cells)+' |')
    for v in variants:
        ck=data.get(v,{}).get('check')
        if ck: md.append(f"| {ep} | _{v} 任务误差 右/左 (cm 均值,最大)_ | "+' | '.join([f"{ck['task_error_cm']['right']['mean']}/{ck['task_error_cm']['right']['max']} , {ck['task_error_cm']['left']['mean']}/{ck['task_error_cm']['left']['max']}; 穿透>1cm {ck['penetration_frames']['over_1cm']}" if vv==v else '' for vv in variants])+' |')
(O/'HAND_REATTACH_COMPARISON.json').write_text(json.dumps(out,indent=1)+'\n'); (O/'HAND_REATTACH_COMPARISON.md').write_text('\n'.join(md)+'\n'); print('\n'.join(md))
