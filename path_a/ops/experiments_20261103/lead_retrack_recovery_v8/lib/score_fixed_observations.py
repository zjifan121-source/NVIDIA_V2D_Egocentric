"""Fixed-observation CPU benchmark. No fitting, anchor selection, or official score.

python score_fixed_observations.py --manifest cases.json --output new_output_dir
Pins are {"path": ..., "sha256": ...}; relative paths resolve beside the manifest.
Schemas: bundle_world, bundle_camera, fp_public. Missing predictions remain q=0.
"""
from pathlib import Path
import argparse, hashlib, json, math, sys, time
import numpy as np

P=Path(__file__).resolve().parent
D=P.parent
THRESHOLDS=np.linspace(0.,1.,101)
SCHEMA_VERSION='fixed_observation_proxy_v1'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def pin_path(pin,root):
    if not isinstance(pin,dict) or not pin.get('sha256'):raise ValueError('Every consumed input requires path and SHA256')
    p=Path(pin['path']);p=p if p.is_absolute() else root/p
    if sha(p)!=pin['sha256']:raise ValueError(f'Input SHA mismatch: {p}')
    return p

def frame_metrics(pred,depth,mask,prediction_valid=True):
    """Own intersection for q; fixed observed valid/mask denominator for coverage."""
    valid=(depth>0)&np.isfinite(depth)&mask;pm=(pred>0)&np.isfinite(pred)
    hit=valid&pm;union=int((pm|mask).sum());err=np.abs(pred[hit]-depth[hit])
    iou=float((pm&mask).sum()/union) if union else 0.
    med=float(np.median(err)) if len(err) else None;p90=float(np.quantile(err,.9)) if len(err) else None
    q=.65*iou+.35*math.exp(-med/.03) if med is not None else .65*iou
    if not prediction_valid:q=0.
    return dict(mask_pixels=int(mask.sum()),valid_depth_points=int(valid.sum()),covered_points=int(hit.sum()),
                coverage=float(hit.sum()/valid.sum()) if valid.any() else 0.,iou=iou,
                depth_median_mm=None if med is None else med*1000,depth_p90_mm=None if p90 is None else p90*1000,
                depth_empty=not bool(hit.any()),prediction_valid=bool(prediction_valid),quality=float(q))

def summary(rows,frame_ids):
    by={r['frame_id']:r for r in rows}
    if set(by)!=set(frame_ids) or len(rows)!=len(frame_ids):raise ValueError('Summary cannot drop/duplicate a requested frame')
    q=np.array([by[f]['quality'] for f in frame_ids]);fraction=(q[:,None]>=THRESHOLDS).mean(0)
    # All-frame denominator is explicit; conditional depth averages are labelled.
    out={'frame_count_requested':len(frame_ids),'frame_count_scored':len(rows),'invalid_prediction_frames':[f for f in frame_ids if not by[f]['prediction_valid']],
         'empty_depth_intersection_frames':[f for f in frame_ids if by[f]['depth_empty']],
         'mean_quality':float(q.mean()),'proxy_auc_101':float(np.sum((fraction[:-1]+fraction[1:])*.5*np.diff(THRESHOLDS))),
         'thresholds':THRESHOLDS.tolist(),'fraction_frames_passing':fraction.tolist()}
    for k in ['iou','coverage','depth_median_mm','depth_p90_mm']:
        v=[by[f][k] for f in frame_ids if by[f][k] is not None]
        out[k]={'equal_frame_mean':float(np.mean(v)) if v else None,'finite_frames':len(v),'requested_frames':len(frame_ids),'all_frame_mean':float(np.mean(v)) if len(v)==len(frame_ids) else None}
    return out

def intrinsics(pin,root):
    z=json.loads(pin_path(pin,root).read_text())
    if 'K' in z:K=np.asarray(z['K'],float);w,h=z['size']
    else:K=np.array([[z['fx'],0,z['cx']],[0,z['fy'],z['cy']],[0,0,1.]]);w,h=z['width'],z['height']
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(w,h)<=0:raise ValueError('Invalid intrinsics/grid')
    return K,(int(h),int(w))

def load_source(source,root):
    import trimesh
    pp=pin_path(source['poses'],root);mp=pin_path(source['mesh'],root)
    with np.load(pp,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    schema=source['schema'];slot=int(source.get('object_slot',0));prefix='object' if slot==0 else f'object_{slot}'
    if schema=='bundle_world':
        raw=a[prefix+'_to_world_transform'];cam=a['camera_to_world_transform']
        if len(raw)!=len(cam):raise ValueError('World/camera length mismatch')
        T=np.full(raw.shape,np.nan,dtype=float)
        for i,(o,c) in enumerate(zip(raw,cam)):
            if np.isfinite(o).all() and np.isfinite(c).all():
                try:T[i]=np.linalg.solve(c.astype(float),o.astype(float))
                except np.linalg.LinAlgError:pass
        scale=float(a.get(prefix+'_scale',1.));valid=np.asarray(a.get(prefix+'_is_valid',np.ones(len(T),bool)),bool).reshape(-1)
    elif schema=='bundle_camera':
        T=np.asarray(a[source.get('pose_key','object_to_camera_transform')],float);scale=float(a.get('object_scale',1.));valid=np.asarray(a.get('object_is_valid',np.ones(len(T),bool)),bool).reshape(-1)
    elif schema=='fp_public':
        T=np.asarray(a[source.get('pose_key','public_poses')],float);scale=float(source.get('scale',1.));valid=np.asarray(a.get('object_is_valid',np.ones(len(T),bool)),bool).reshape(-1)
    else:raise ValueError(f'Unsupported pose schema: {schema}')
    if T.ndim!=3 or T.shape[1:]!=(4,4) or len(valid)!=len(T):raise ValueError('Invalid pose/validity schema')
    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid mesh scale')
    ids=np.asarray(a['frame_ids'] if 'frame_ids' in a else source.get('frame_ids',np.arange(len(T))),int)
    if len(ids)!=len(T) or len(set(ids.tolist()))!=len(ids):raise ValueError('Missing/duplicate source frame mapping')
    if source.get('pair_manifest'):
        pair=json.loads(pin_path(source['pair_manifest'],root).read_text())
        if pair['mesh_sha256']!=source['mesh']['sha256'] or pair['poses_sha256']!=source['poses']['sha256']:raise ValueError('Mesh/pose pair manifest mismatch')
    mesh=trimesh.load(mp,force='mesh',process=False)
    return {'T':T,'valid':valid,'index':{int(f):i for i,f in enumerate(ids)},'vertices':np.asarray(mesh.vertices,float)*scale,'faces':np.asarray(mesh.faces),'mesh_scale_applied_once':scale}

def matrix_ok(T):
    return bool(np.isfinite(T).all() and np.allclose(T[3],[0,0,0,1],atol=1e-6) and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-3) and abs(np.linalg.det(T[:3,:3])-1)<1e-3)

def pair_summary(rows):
    out={'requested_frames':len(rows),'empty_frames':[x['frame_id'] for x in rows if not x['common_count']]}
    for key in ['baseline_median_mm','candidate_median_mm','baseline_p90_mm','candidate_p90_mm']:
        vals=[x[key] for x in rows if x[key] is not None]
        out[key]={'finite_frame_mean':float(np.mean(vals)) if vals else None,'finite_frames':len(vals),'all_frame_mean':float(np.mean(vals)) if len(vals)==len(rows) else None}
    return out

def evaluate_case(case,root,deadline):
    import cv2
    sys.path.insert(0,str(D/'ep5_observed_geometry_v1'))
    from surface_renderer import raster_surface
    ids=case['frame_ids'];obs=case['observations'];sources=case['sources']
    if not ids or ids!=sorted(set(ids)):raise ValueError('Explicit nonempty ordered unique frame_ids required')
    if len(obs)!=len(ids) or {r['frame_id'] for r in obs}!=set(ids):raise ValueError('Every observation frame must be present once')
    if len({s['name'] for s in sources})!=len(sources) or not sources:raise ValueError('Unique nonempty source names required')
    if case['depth_encoding']!='inverse_uint16_m':raise ValueError('Only frozen PathB inverse uint16 metre encoding supported')
    K,shape=intrinsics(case['K'],root);loaded={s['name']:load_source(s,root) for s in sources};byobs={r['frame_id']:r for r in obs}
    rows={s['name']:[] for s in sources};base=sources[0]['name'];pairs={s['name']:[] for s in sources[1:]}
    for f in ids:
        if time.monotonic()>deadline:raise TimeoutError('CPU scoring budget reached')
        r=byobs[f];code=cv2.imread(str(pin_path(r['depth'],root)),cv2.IMREAD_UNCHANGED);mask=cv2.imread(str(pin_path(r['mask'],root)),0)
        if code is None or mask is None or code.shape!=shape or code.dtype!=np.uint16:raise ValueError(f'Invalid depth/mask input frame{f}')
        if mask.shape!=shape:
            if case.get('mask_resize')!='nearest_to_depth_grid':raise ValueError('Mask grid mismatch without explicit nearest resize contract')
            mask=cv2.resize(mask,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
        mask=mask>0;dep=np.zeros(shape,float);good=(code>0)&(code<65535);dep[good]=65535./code[good].astype(float)-1
        preds={}
        for s in sources:
            name=s['name'];z=loaded[name];i=z['index'].get(f);ok=i is not None and bool(z['valid'][i]) and matrix_ok(z['T'][i]);why=None
            if not ok:pred=np.zeros(shape);why='MISSING_OR_INVALID_PREDICTION'
            else:
                T=z['T'][i];vc=z['vertices']@T[:3,:3].T+T[:3,3];pred=raster_surface(vc,z['faces'],K,shape)[0]
            preds[name]=pred;rows[name].append({'frame_id':f,'source':name,'invalid_reason':why,**frame_metrics(pred,dep,mask,ok)})
        valid=good&mask
        for name in pairs:
            common=valid&(preds[base]>0)&(preds[name]>0);n=int(common.sum());q={'frame_id':f,'common_count':n}
            for key in [base,name]:
                err=np.abs(preds[key][common]-dep[common])*1000;stem='baseline' if key==base else 'candidate'
                q[stem+'_median_mm']=float(np.median(err)) if n else None;q[stem+'_p90_mm']=float(np.quantile(err,.9)) if n else None
            pairs[name].append(q)
    half=ids[len(ids)//2:];mapping={s['name']:s.get('mapping_status','UNVERIFIED_MAPPING') for s in sources}
    verified={'VERIFIED_FROZEN_INPUTS','VERIFIED_SAME_MEASUREMENT_FRAME'}
    return {'episode':case['episode'],'object_name':case['object_name'],'state':'COMPLETE_FIXED_INPUT_EVALUATION','frame_ids':ids,'second_half_ids':half,'mapping_status':mapping,'rankable_sources':[s for s,v in mapping.items() if v in verified],'unverified_mapping_sources':[s for s,v in mapping.items() if v not in verified],
            'source_summary':{name:{'full':summary(rr,ids),'second_half':summary([r for r in rr if r['frame_id'] in half],half)} for name,rr in rows.items()},'rows':rows,
            'pairwise_common_depth':{name:{'baseline':base,'candidate':name,'full':pair_summary(rr),'second_half':pair_summary([r for r in rr if r['frame_id'] in half]),'rows':rr} for name,rr in pairs.items()},'mesh_scale_applied_once':{n:z['mesh_scale_applied_once'] for n,z in loaded.items()},'quality_scope':'Own rendered intersection depth; pairwise common-depth is separate and never inserted into q.'}

def macro(results,source_names,total_cases):
    out={}
    for n in source_names:
        rr=[r for r in results if r.get('state')=='COMPLETE_FIXED_INPUT_EVALUATION' and n in r['source_summary']]
        ready=len(rr)==total_cases;out[n]={'episodes_requested':total_cases,'episodes_completed':len(rr),'complete':ready,'rankable':ready and all(n in r['rankable_sources'] for r in rr),
              'macro_full_proxy_auc_101':float(np.mean([r['source_summary'][n]['full']['proxy_auc_101'] for r in rr])) if ready else None,
              'macro_second_half_proxy_auc_101':float(np.mean([r['source_summary'][n]['second_half']['proxy_auc_101'] for r in rr])) if ready else None,
              'aggregation':'Each complete episode equal weight, never pool unequal episode lengths; incomplete macro is null.'}
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--budget-seconds',type=float,default=600);a=ap.parse_args()
    cfg=json.loads(a.manifest.read_text());cases=cfg['cases'];assert cases and cfg.get('state','').startswith('FROZEN')
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+a.budget_seconds;results=[];names=sorted({s['name'] for c in cases for s in c['sources']});msha=sha(a.manifest)
    for case in cases:
        try:result=evaluate_case(case,a.manifest.parent,deadline)
        except Exception as e:result={'episode':case.get('episode'),'state':'FAILED_PRESERVED','error':repr(e)}
        results.append(result);write(a.output/f'ep{case["episode"]:03}.json',result);print(case['episode'],result['state'],flush=True)
    assert sha(a.manifest)==msha
    ok=all(r['state']=='COMPLETE_FIXED_INPUT_EVALUATION' for r in results)
    write(a.output/'RESULTS.json',{'schema':SCHEMA_VERSION,'state':'COMPLETE' if ok else 'INCOMPLETE_NO_COMPLETE_COHORT_CLAIM','manifest_sha256':msha,'script_sha256':sha(__file__),'cases':results,'macro':macro(results,names,len(cases)),'seconds':time.monotonic()-start,'GPU':False,'GT':False,'fit_calls':0,'official_AUC':False,'team_site_exact_scorer_reproduction':False,'zero_quality_threshold_note':'101-point trapezoidal grid gives0.005 for a q=0 frame because threshold0 passes; preserved explicitly, not a hidden success.'})
    if not ok:raise SystemExit(1)
if __name__=='__main__':main()
