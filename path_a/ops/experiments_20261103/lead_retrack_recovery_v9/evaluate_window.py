"""Mac CPU post-processing for collected GPU outputs (run_v1/cases/<case>/output/track).
For every completed case: export a candidate bundle (only this slot's mesh+poses change), score current vs candidate
on all frames with the frozen fixed_observation_proxy_v1, apply the pre-registered rule, write SELECTION.json."""
from pathlib import Path
import json, sys, time, hashlib
import numpy as np
ROOT=Path('/Volumes/E/Code/V2D'); OPS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops'
X=OPS/'experiments_20261103/lead_retrack_recovery_v7'; RUN=X/'run_v1'
B=OPS/'experiments_20261030/scored20_dual_anchor_v1'; REL=ROOT/'work/scored11-fulltrack-20261103/release'
SCRIPTS=ROOT/'work/scored20-pipeline-release-20261102/baseline/reconstruction/scripts'
sys.path.insert(0,str(SCRIPTS/'scored20_pipeline')); import inputs
D=OPS/'experiments_20261029/scored_scene_consistency_v1'
sys.path[:0]=[str(D/'ep5_observed_geometry_v1/runtime'),str(D/'team_cohort_benchmark_20261030')]
import score_fixed_observations as score
sha=score.sha; read=lambda p:json.loads(Path(p).read_text())

REP=OPS/'experiments_20261103/lead_fb11_repaired'
def current_bundle(ep):
    r=REP/f'scenes/episode_{ep:06d}/bundle'
    return r if r.exists() else REL/f'scenes/episode_{ep:06d}/bundle'
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def pin(p): return dict(path=str(p),sha256=sha(p))
import trimesh
from sym_motion import motion as sym_motion
def motion(W,obj,mesh_path):
    V=np.asarray(trimesh.load(mesh_path,force='mesh',process=False).vertices,float); return sym_motion(W,obj,V)
def rule(cu,ca,frames):
    """Amended selection rule (2026-11-03 12:05 UTC), applied uniformly: proxy strictly higher; no flip-sized step
    (>90 deg); fast steps (>30 deg) on at most 10% of frames; absolute translation caps p95 <= 50 mm/frame and
    max <= 120 mm/frame (relative-to-current caps are meaningless when the current track is stuck); no invalid frames."""
    m=ca['motion']
    return bool(ca['full_proxy']>cu['full_proxy']+1e-12 and m['rot_max_deg']<=90 and m['n_rot_steps_over_30']<=0.10*max(1,frames-1) and m['trans_p95_mm']<=50 and m['trans_max_mm']<=120 and ca['invalid']==0)
def rule_strict(cu,ca):
    m=ca['motion']; return bool(ca['full_proxy']>cu['full_proxy']+1e-12 and m['rot_max_deg']<=max(cu['motion']['rot_max_deg'],30) and m['trans_p95_mm']<=1.5*cu['motion']['trans_p95_mm'] and ca['invalid']==0)
batch=read(X/'FROZEN_BATCH.json'); inv={int(r['episode']):r['objects'] for r in read(REL/'inventory.json')}
sel={'rule':batch['selection'],'cases':{}}
if len(sys.argv)>1 and sys.argv[1]=='--rerule':
    old=read(RUN/'SELECTION.json'); best={}
    for k,v in old['cases'].items():
        if v.get('state')=='EVALUATED':
            fr=next(c['frames'] for c in batch['cases'] if c['case']==k); v['passed']=rule(v['current'],v['candidate'],fr); v['passed_strict']=rule_strict(v['current'],v['candidate'])
            print(k,'proxy %.4f -> %.4f rot_max %.1f n>30 %d trans_p95 %.1f trans_max %.1f PASS=%s strict=%s'%(v['current']['full_proxy'],v['candidate']['full_proxy'],v['candidate']['motion']['rot_max_deg'],v['candidate']['motion']['n_rot_steps_over_30'],v['candidate']['motion']['trans_p95_mm'],v['candidate']['motion']['trans_max_mm'],v['passed'],v['passed_strict']))
            if v['passed']:
                key=f"ep{v['episode']:02d}_{v['object']}"
                if key not in best or v['candidate']['full_proxy']>old['cases'][best[key]]['candidate']['full_proxy']: best[key]=k
    old['best_per_object']=best; old['rule_version']='amended 2026-11-03 12:05 UTC'; put(RUN/'SELECTION.json',old); print('SELECTION',best); sys.exit(0)
for c in batch['cases']:
    case=X/'cases'/c['case']; out=RUN/'cases'/c['case']/'output'; track=out/'track'; rec=track/'RECEIPT.json'
    if not rec.exists() or read(rec).get('state')!='COMPLETE_FULL_TRACK': sel['cases'][c['case']]={'state':'NOT_COMPLETED'}; continue
    r=read(rec); assert r['exit_code']==0 and r['track_calls']==c['frames']-1 and r['protocol_sha256']==sha(case/'protocol.json')
    for n,s in r['outputs_sha256'].items(): assert sha(track/n)==s,(c['case'],n)
    ep,obj,slot=c['episode'],c['object'],c['slot']; current=current_bundle(ep)
    dest=RUN/'cases'/c['case']/'candidate_bundle'
    if dest.exists() and not (dest/'MESH_POSE_PAIR.json').exists(): import shutil; shutil.rmtree(dest)  # incomplete earlier export
    if not dest.exists(): inputs.export_track(current,case/'payload/asset/mesh.obj',track/'matched_mesh_poses.npz',track/'MESH_POSE_PAIR.json',dest,slot)
    manifest=read(B/'cases'/f'ep{ep:02d}_{obj}'/'INPUT_MANIFEST.json'); cache=Path(manifest['payload_local']); frames=manifest['frames']
    observations=[dict(frame_id=x['frame_id'],rgb=pin(cache/x['rgb']),depth=pin(cache/x['depth']),mask=pin(cache/x['mask'])) for x in frames]
    mesh_name=inputs.mesh_name(slot)
    sources=[dict(name=name,schema='bundle_world',object_slot=slot,mesh=pin(bundle/mesh_name),poses=pin(bundle/'result.npz'),mapping_status='VERIFIED_SAME_MEASUREMENT_FRAME') for name,bundle in [('current',current),('candidate',dest)]]
    res=score.evaluate_case(dict(episode=ep,object_name=obj,slot=slot,frame_ids=[x['frame_id'] for x in frames],K=pin(cache/'K.json'),depth_encoding='inverse_uint16_m',observations=observations,sources=sources),X,time.monotonic()+1800)
    m={}
    for name,bundle in [('current',current),('candidate',dest)]:
        z=np.load(bundle/'result.npz',allow_pickle=False); W=z['object_to_world_transform' if slot==0 else 'object_1_to_world_transform']
        s=res['source_summary'][name]['full']; rows=res['rows'][name]
        m[name]=dict(full_proxy=s['proxy_auc_101'],iou_mean=s['iou']['equal_frame_mean'],depth_median_mean_mm=s['depth_median_mm']['equal_frame_mean'],invalid=len(s['invalid_prediction_frames']),motion=motion(W,obj,bundle/mesh_name),
                     iou_blocks=[float(np.mean([x['iou'] for x in rows[b:b+25]])) for b in range(0,len(rows),25)])
    cu,ca=m['current'],m['candidate']
    strict=rule_strict(cu,ca); passed=rule(cu,ca,c['frames'])
    sel['cases'][c['case']]=dict(state='EVALUATED',episode=ep,object=obj,slot=slot,variant=c['variant'],current=cu,candidate=ca,delta=ca['full_proxy']-cu['full_proxy'],passed=bool(passed),passed_strict=bool(strict),
        recovery_frames=r.get('recovery_frames',[]),gpu_seconds=r.get('seconds'),candidate_bundle=str(dest))
    print(c['case'],'current %.4f -> candidate %.4f (%+.4f) rot_max %.1f trans_p95 %.1f PASS=%s recoveries=%d'%(cu['full_proxy'],ca['full_proxy'],ca['full_proxy']-cu['full_proxy'],ca['motion']['rot_max_deg'],ca['motion']['trans_p95_mm'],passed,len(r.get('recovery_frames',[]))),flush=True)
# best passing variant per object
best={}
for k,v in sel['cases'].items():
    if v.get('passed'):
        key=f"ep{v['episode']:02d}_{v['object']}"
        if key not in best or v['candidate']['full_proxy']>sel['cases'][best[key]]['candidate']['full_proxy']: best[key]=k
sel['best_per_object']=best
put(RUN/'SELECTION.json',sel); print('SELECTION',best)
