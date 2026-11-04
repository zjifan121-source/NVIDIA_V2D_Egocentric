"""Summarize diagnostic traces against the unchanged processed reference."""
import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent


def main(ep):
    meta=json.loads((HERE/f'ep{ep}_trained_trace.json').read_text())
    z=np.load(HERE/f'ep{ep}_trained_trace.npz')
    p=Path(meta['reference_parquet'])
    p=p if p.is_file() else next(p.glob('*.parquet'))
    table=pq.read_table(p)
    cell=lambda k:np.asarray(table[k][0].as_py())
    names=table['object_body_names'][0].as_py()
    n=len(z['body_q'])
    ref=cell('object_body_position')[:n]
    obj=z['body_q'][:,z['object_body_ids'],:3]
    objerr=np.linalg.norm(obj-ref,axis=-1)
    werr={}
    for side in ('right','left'):
        bid=next(i for i,name in enumerate(meta['body_labels']) if name.endswith(f'/{side}_hand_C_MC'))
        werr[side]=np.linalg.norm(z['body_q'][:,bid,:3]-cell(f'robot_{side}_wrist_position')[:n],axis=1)
    match=re.search(r'sides=(\([^)]*\))',meta.get('contact_layout',''))
    sides=ast.literal_eval(match[1]) if match else ('right','left')
    actions=z['action'].reshape(n,-1)
    phases=[]
    for a,b in [(0,20),(20,40),(40,60),(60,90),(90,120),(120,160),(160,200),(200,n)]:
        b=min(b,n)
        if a>=b:continue
        row=dict(frames=[a,b],object_mean_error_cm=(100*objerr[a:b].mean(0)).tolist(),
                 wrist_mean_error_cm={s:float(100*e[a:b].mean()) for s,e in werr.items()},
                 reference_z=ref[a:b,:,2].mean(0).tolist(),actual_z=obj[a:b,:,2].mean(0).tolist())
        for hand,side in enumerate(sides):
            for slot,name in enumerate(names):
                mask=(z['slot_hand_ids']==hand)&(z['slot_object_ids']==slot)
                active=z['contact_active'][a:b,mask]
                force=np.linalg.norm(z['contact_force_w'][a:b,mask],axis=-1)
                row.setdefault('contacts',{})[f'{side}:{name}']=dict(
                    frames_with_contact_fraction=float(active.any(1).mean()),
                    mean_active_links=float(active.sum(1).mean()),mean_total_force_N=float(force.sum(1).mean()))
        row['action_abs_mean']=float(np.abs(actions[a:b]).mean())
        row['action_abs_max']=float(np.abs(actions[a:b]).max())
        phases.append(row)
    out=dict(episode=ep,diagnostic_only=True,checkpoint=meta['checkpoint'],objects=names,phases=phases)
    (HERE/f'ep{ep}_trace_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);main(p.parse_args().episode)
