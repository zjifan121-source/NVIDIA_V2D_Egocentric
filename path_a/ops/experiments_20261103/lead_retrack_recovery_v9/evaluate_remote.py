"""Host-side evaluation (A6000, CPU) for a lead_retrack_recovery package: for every completed case export a candidate
bundle (only this slot's mesh+poses change), score current vs candidate on all frames with the frozen
fixed_observation_proxy_v1, apply the pre-registered (amended) rule, write run_v1/SELECTION.json.
Layout on the host: <pkg>/cases/<case>/{manifest.json,protocol.json,payload/asset,output/track}, <pkg>/lib/{score_fixed_observations.py,
surface_renderer.py,inputs.py,sym_motion.py}, <pkg>/current/epNN/bundle (the current delivered bundle), frames in
/var/tmp/v2d_scored20_dual_anchor_v1/epNN/{rgb,depth,masks,K.json}. Usage: evaluate_remote.py <pkg> [--rerule]"""
from pathlib import Path
import json, sys, time, hashlib, shutil
import numpy as np
X=Path(sys.argv[1]).resolve(); RUN=X/'run_v1'; CACHE=Path('/var/tmp/v2d_scored20_dual_anchor_v1')
sys.path.insert(0,str(X/'lib')); import inputs, score_fixed_observations as score, trimesh
from sym_motion import motion as sym_motion
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def pin(p): return dict(path=str(p),sha256=sha(p))
def current_bundle(ep): return X/'current'/f'ep{ep:02d}'/'bundle'
def motion(W,obj,mesh_path):
    V=np.asarray(trimesh.load(mesh_path,force='mesh',process=False).vertices,float); return sym_motion(W,obj,V)
def rule(cu,ca,frames):
    m=ca['motion']
    return bool(ca['full_proxy']>cu['full_proxy']+1e-12 and m['rot_max_deg']<=90 and m['n_rot_steps_over_30']<=0.10*max(1,frames-1) and m['trans_p95_mm']<=50 and m['trans_max_mm']<=120 and ca['invalid']==0)
def rule_strict(cu,ca):
    m=ca['motion']; return bool(ca['full_proxy']>cu['full_proxy']+1e-12 and m['rot_max_deg']<=max(cu['motion']['rot_max_deg'],30) and m['trans_p95_mm']<=1.5*cu['motion']['trans_p95_mm'] and ca['invalid']==0)
batch=read(X/'FROZEN_BATCH.json'); sel={'rule':batch['selection'],'cases':{}}
for c in batch['cases']:
    case=X/'cases'/c['case']; track=case/'output/track'; rec=track/'RECEIPT.json'
    if not rec.exists() or read(rec).get('state')!='COMPLETE_FULL_TRACK': sel['cases'][c['case']]={'state':'NOT_COMPLETED'}; print(c['case'],'NOT_COMPLETED',flush=True); continue
    r=read(rec); assert r['exit_code']==0 and r['track_calls']==c['frames']-1 and r['protocol_sha256']==sha(case/'protocol.json')
    for n,s in r['outputs_sha256'].items(): assert sha(track/n)==s,(c['case'],n)
    ep,obj,slot=c['episode'],c['object'],c['slot']; current=current_bundle(ep); manifest=read(case/'manifest.json'); frames=manifest['frames']; cache=CACHE/f'ep{ep:02d}'
    dest=RUN/'cases'/c['case']/'candidate_bundle'
    if dest.exists() and not (dest/'MESH_POSE_PAIR.json').exists(): shutil.rmtree(dest)
    if not dest.exists(): inputs.export_track(current,case/'payload/asset/mesh.obj',track/'matched_mesh_poses.npz',track/'MESH_POSE_PAIR.json',dest,slot)
    observations=[dict(frame_id=x['frame_id'],rgb=pin(cache/x['rgb']),depth=pin(cache/x['depth']),mask=pin(cache/x['mask'])) for x in frames]
    for x,o in zip(frames,observations):
        for k in ['rgb','depth','mask']: assert o[k]['sha256']==manifest['files'][x[k]],('frame cache differs from manifest',c['case'],x[k])
    mesh_name=inputs.mesh_name(slot)
    sources=[dict(name=name,schema='bundle_world',object_slot=slot,mesh=pin(bundle/mesh_name),poses=pin(bundle/'result.npz'),mapping_status='VERIFIED_SAME_MEASUREMENT_FRAME') for name,bundle in [('current',current),('candidate',dest)]]
    res=score.evaluate_case(dict(episode=ep,object_name=obj,slot=slot,frame_ids=[x['frame_id'] for x in frames],K=pin(cache/'K.json'),depth_encoding='inverse_uint16_m',observations=observations,sources=sources),X,time.monotonic()+3600)
    m={}
    for name,bundle in [('current',current),('candidate',dest)]:
        z=np.load(bundle/'result.npz',allow_pickle=False); W=z['object_to_world_transform' if slot==0 else f'object_{slot}_to_world_transform']
        s=res['source_summary'][name]['full']; rows=res['rows'][name]
        m[name]=dict(full_proxy=s['proxy_auc_101'],iou_mean=s['iou']['equal_frame_mean'],depth_median_mean_mm=s['depth_median_mm']['equal_frame_mean'],invalid=len(s['invalid_prediction_frames']),motion=motion(W,obj,bundle/mesh_name),
                     iou_blocks=[float(np.mean([x['iou'] for x in rows[b:b+25]])) for b in range(0,len(rows),25)])
    cu,ca=m['current'],m['candidate']; passed=rule(cu,ca,c['frames']); strict=rule_strict(cu,ca)
    sel['cases'][c['case']]=dict(state='EVALUATED',episode=ep,object=obj,slot=slot,variant=c['variant'],label=c.get('label'),current=cu,candidate=ca,delta=ca['full_proxy']-cu['full_proxy'],passed=bool(passed),passed_strict=bool(strict),
        recovery_frames=r.get('recovery_frames',[]),gpu_seconds=r.get('seconds'),candidate_bundle=str(dest),mesh_sha256=c.get('mesh_sha256'))
    print(c['case'],'current %.4f -> candidate %.4f (%+.4f) rot_max %.1f trans_p95 %.1f PASS=%s recoveries=%d'%(cu['full_proxy'],ca['full_proxy'],ca['full_proxy']-cu['full_proxy'],ca['motion']['rot_max_deg'],ca['motion']['trans_p95_mm'],passed,len(r.get('recovery_frames',[]))),flush=True)
best={}
for k,v in sel['cases'].items():
    if v.get('passed'):
        key=f"ep{v['episode']:02d}_{v['object']}"
        if key not in best or v['candidate']['full_proxy']>sel['cases'][best[key]]['candidate']['full_proxy']: best[key]=k
sel['best_per_object']=best; put(RUN/'SELECTION.json',sel); print('SELECTION',best,flush=True)
