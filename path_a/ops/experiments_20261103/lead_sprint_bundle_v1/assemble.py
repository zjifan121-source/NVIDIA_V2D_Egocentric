"""Assemble the sprint candidate set: start from the fb11 release (current HF) and, per object, apply the best
rule-passing replacement among GPU re-track cases (batches v1/v2) and the pool re-scan. Everything else byte-identical.
Writes scenes/episode_XXXXXX/bundle plus SELECTION_SUMMARY.json. No GT. Not uploaded by this script."""
from pathlib import Path
import json, sys, shutil, hashlib, datetime
import numpy as np
ROOT=Path('/Volumes/E/Code/V2D'); OPS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops'; L=OPS/'experiments_20261103'
OUT=L/'lead_sprint_bundle_v1'; REL=ROOT/'work/scored11-fulltrack-20261103/release'
MIN_GAIN=0.002  # minimum full-frame proxy gain for a replacement to enter the delivered set (noise floor of the proxy)
SCRIPTS=ROOT/'work/scored20-pipeline-release-20261102/baseline/reconstruction/scripts'
sys.path.insert(0,str(SCRIPTS/'scored20_pipeline')); import inputs
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())

REP=L/'lead_fb11_repaired'
def current_bundle(ep):
    r=REP/f'scenes/episode_{ep:06d}/bundle'
    return r if r.exists() else REL/f'scenes/episode_{ep:06d}/bundle'
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
inv={int(r['episode']):r['objects'] for r in read(REL/'inventory.json')}
cands={}  # (ep,obj) -> list of dict(kind, proxy, ...)
for v in ['lead_retrack_recovery_v%d'%i for i in range(1,8)]:
    s=L/v/'run_v1/SELECTION.json'
    if not s.exists(): continue
    for name,c in read(s)['cases'].items():
        if c.get('passed'):
            case=L/v/'cases'/name
            cands.setdefault((c['episode'],c['object']),[]).append(dict(kind='track',proxy=c['candidate']['full_proxy'],current=c['current']['full_proxy'],source=f'{v}/{name}',
                mesh=case/'payload/asset/mesh.obj',poses=L/v/'run_v1/cases'/name/'output/track/matched_mesh_poses.npz',pair=L/v/'run_v1/cases'/name/'output/track/MESH_POSE_PAIR.json',slot=c['slot']))
pool=read(L/'lead_scored_pool_rescan_v1/RESULT.json')
for ep,e in pool['episodes'].items():
    if 'skipped' in e: continue
    for obj,o in e.items():
        if 'error' in o or not o.get('best'): continue
        b=o['results'][o['best']]; cur=[v for v in o['results'].values() if v['is_current']][0]
        cands.setdefault((int(ep),obj),[]).append(dict(kind='pool',proxy=b['full_proxy'],current=cur['full_proxy'],source='lead_scored_pool_rescan_v1:'+o['best'],donor=Path(b['path']),slot=o['slot']))
summary={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':'fb11 release '+str(REL),'min_gain':MIN_GAIN,'episodes':{}}
for ep,objects in sorted(inv.items()):
    src=current_bundle(ep); dest=OUT/f'scenes/episode_{ep:06d}/bundle'
    if dest.exists(): shutil.rmtree(dest.parent)
    chosen=[]
    working=src
    for slot,obj in enumerate(objects):
        cs=cands.get((ep,obj),[])
        if not cs: continue
        best=max(cs,key=lambda c:c['proxy'])
        if best['proxy']<best['current']+MIN_GAIN: continue  # delivery filter: do not churn a trajectory for a noise-level gain
        tmp=OUT/f'_work/ep{ep:02d}_slot{slot}'
        if tmp.exists(): shutil.rmtree(tmp)
        if best['kind']=='track':
            inputs.export_track(working,best['mesh'],best['poses'],best['pair'],tmp,slot)
        else:
            inputs.copy_scene(working,tmp); inputs.install_mesh(best['donor']/inputs.mesh_name(slot),tmp,slot)
            data=inputs.arrays(working); d=np.load(best['donor']/'result.npz',allow_pickle=False)
            ck,wk=inputs.fields(slot); pre=ck.removesuffix('_to_camera_transform')
            P=np.array([np.linalg.solve(c.astype(float),w.astype(float)) for c,w in zip(d['camera_to_world_transform'],d[pre+'_to_world_transform'])])
            res={k:v.copy() for k,v in data.items()}; res[ck]=P.astype(data[ck].dtype) if ck in data else P; res[wk]=data['camera_to_world_transform']@P
            if pre+'_is_valid' not in res: res[pre+'_is_valid']=np.ones(len(P),bool)
            np.savez_compressed(tmp/'result.npz',**res); inputs.validate_scene(tmp)
        working=tmp; chosen.append(dict(object=obj,slot=slot,kind=best['kind'],source=best['source'],proxy_before=best['current'],proxy_after=best['proxy']))
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(working,dest)
    for extra in ['support_v1']:
        if (src.parent/extra).exists(): shutil.copytree(src.parent/extra,dest.parent/extra)
    put(dest.parent/'run.json',dict(episode=ep,objects=objects,base='fb11',changed=chosen,unchanged=(not chosen),created_utc=summary['created_utc']))
    summary['episodes'][ep]=dict(objects=objects,changed=chosen)
    print(f'ep{ep:02d}', 'CHANGED '+', '.join(f"{c['object']}:{c['proxy_before']:.4f}->{c['proxy_after']:.4f} ({c['kind']})" for c in chosen) if chosen else 'unchanged',flush=True)
shutil.rmtree(OUT/'_work',ignore_errors=True)
files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
put(OUT/'SHA256SUMS.json',files); put(OUT/'SELECTION_SUMMARY.json',summary); print('DONE',len(files),'files')
