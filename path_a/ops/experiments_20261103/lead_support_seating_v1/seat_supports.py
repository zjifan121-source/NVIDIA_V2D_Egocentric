"""Seat the support_v1 tables of all 20 Scored episodes on the lead_sprint_v1 objects (bundle world, Z-up, frame 0).
CPU only, no GT, nothing in the input stage is modified.

Rule (same as haisenk's wseat.py, 2026-11-03): T = min(object bottoms at frame 0) - 2 mm; every support prim whose top
is above T is LOWERED to T (never raised); every object whose bottom is more than 5 mm above T gets its own thin disk
(height 1 cm, top = bottom - 2 mm, radius = its lowest-1-cm footprint radius clipped to 3-15 cm).
Added here: (1) no disk for an object that rests on another object (the other object's vertices lie under its footprint
within 3 cm below its bottom, e.g. the ep01 lid on the pot); (2) a disk that would cut into a neighbour's frame-0
vertices is shrunk to clear it or dropped (haisenk's known bug); (3) a frame-0 / first-10-frames penetration check is
written to VALIDATION.json. ep10 and ep38 have no support_v1 in fb11: their historical same-episode supports
(night_2026-10-24 ken_c_scored_attach) are carried into bundle world with the common consumer transform (the method of
fb11's build), after asserting that our bundle shares the historical bundle's camera arrays, then seated.
Usage: seat_supports.py <stage_dir> <out_dir>"""
from pathlib import Path
import json, sys, re, hashlib, datetime, importlib.util
import numpy as np
ROOT=Path('/Volumes/E/Code/V2D'); OPS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops'
S=Path(sys.argv[1]); OUT=Path(sys.argv[2]); O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
spec=importlib.util.spec_from_file_location('install_support',S/'tools/install_support.py'); inst=importlib.util.module_from_spec(spec); spec.loader.exec_module(inst)
def obj_vertices(path):
    V=[]
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                p=line.split(); V.append((float(p[1]),float(p[2]),float(p[3])))
    return np.array(V,float)
def parse_usda_cylinders(text):
    out=[]
    for m in re.finditer(r'def Cylinder "([^"]+)"(.*?)\n    \}',text,re.S):
        body=m.group(2); r=float(re.search(r'double radius = ([-\d.eE+]+)',body).group(1)); h=float(re.search(r'double height = ([-\d.eE+]+)',body).group(1))
        t=re.search(r'xformOp:translate = \(([^)]+)\)',body).group(1); c=[float(x) for x in t.split(',')]
        assert re.search(r'uniform token axis = "Z"',body); out.append(dict(name=m.group(1),center=c,radius=r,height=h))
    return out
HIST={10:dict(parquet=ROOT/'work/scored11-fulltrack-20261103/historical_support/night_2026-10-24/scenes/episode_000010/ken_c_scored_attach/assets/human_motion_data/ego_recon/processed/sequence_id=ep10_ken_c_scored_attach/robot_name=sharpa_wave/27bd1e7045814c2eba40c890ebfc7e78-0.parquet',
             usda=ROOT/'work/scored11-fulltrack-20261103/historical_support/night_2026-10-24/scenes/episode_000010/ken_c_scored_attach/assets/human_motion_data/ego_recon/reconstructed_stage/ep10_ken_c_scored_attach_sharpa_wave_support.usda',
             remote='night_2026-10-24/scenes/episode_000010/ken_c_scored_attach/assets/human_motion_data/ego_recon/reconstructed_stage/ep10_ken_c_scored_attach_sharpa_wave_support.usda'),
      38:dict(parquet=Path('/private/tmp/claude-501/-Volumes-E-Code-V2D/9e418383-d795-43bf-92a7-b5721bcd24f8/scratchpad/hist_support/night_2026-10-24/scenes/episode_000038/ken_c_scored_attach/assets/human_motion_data/ego_recon/processed/sequence_id=ep38_ken_c_scored_attach/robot_name=sharpa_wave/77e1f0fb90ca4436a7d8b58d19e2cfda-0.parquet'),
             usda=Path('/private/tmp/claude-501/-Volumes-E-Code-V2D/9e418383-d795-43bf-92a7-b5721bcd24f8/scratchpad/hist_support/night_2026-10-24/scenes/episode_000038/ken_c_scored_attach/assets/human_motion_data/ego_recon/reconstructed_stage/ep38_ken_c_scored_attach_sharpa_wave_support.usda'),
             remote='night_2026-10-24/scenes/episode_000038/ken_c_scored_attach/assets/human_motion_data/ego_recon/reconstructed_stage/ep38_ken_c_scored_attach_sharpa_wave_support.usda')}
OLD=OPS/'experiments_20261030/scored20_dual_anchor_v1/hf_inputs/night_2026-10-24/scenes'
def historical_support(ep,bundle):
    import pyarrow.parquet as pq
    h=HIST[ep]; old=np.load(OLD/f'episode_{ep:06d}/ken_c_scored_attach/bundle_cleaned/result.npz',allow_pickle=False); ours=np.load(bundle/'result.npz',allow_pickle=False)
    assert np.array_equal(old['camera_to_world_transform'],ours['camera_to_world_transform']),(ep,'historical bundle world differs from ours')
    table=pq.read_table(h['parquet']); assert table.num_rows==1; record={k:v[0] for k,v in table.to_pydict().items()}
    G,pe,re_=inst.common_transform(old,record); cyl=parse_usda_cylinders(Path(h['usda']).read_text()); out=[]
    for c in cyl:
        center=np.linalg.inv(G)@np.r_[c['center'],1.0]; out.append(dict(center_bundle_world_m=center[:3].tolist(),radius_m=c['radius'],height_m=c['height'],historical_prim=c['name']))
    return out,dict(source='Historical same-episode collision proxies carried into bundle world with the common consumer transform (fb11 method); camera arrays identical to the historical cleaned bundle',source_repo='rose-tea-cafe/v2d-t3-recon',source_support_path=h['remote'],source_support_sha256=sha(h['usda']),source_processed_sha256=sha(h['parquet']),common_transform_position_error_m=pe,common_transform_rotation_error=re_)
def penetration(objs,cyls,frame):
    rows=[]
    for o in objs:
        w=o['V']@o['T'][frame][:3,:3].T+o['T'][frame][:3,3]; worst=dict(fraction=0.0,depth_cm=0.0,cylinder=None)
        for i,c in enumerate(cyls):
            cc=np.array(c['center_bundle_world_m']); ins=(np.linalg.norm(w[:,:2]-cc[:2],axis=1)<=c['radius_m'])&(w[:,2]<=cc[2]+c['height_m']/2)&(w[:,2]>=cc[2]-c['height_m']/2)
            if ins.any():
                d=float((cc[2]+c['height_m']/2-w[ins,2]).max()*100)
                if d>worst['depth_cm']: worst=dict(fraction=float(ins.mean()),depth_cm=d,cylinder=i)
        rows.append(dict(object=o['name'],**worst))
    return rows
summary=[]
for ep_dir in sorted((S/'scenes').iterdir()):
    ep=int(ep_dir.name.split('_')[-1]); b=ep_dir/'bundle'; run=read(ep_dir/'run.json'); objects=run['objects']; z=np.load(b/'result.npz',allow_pickle=False)
    objs=[]
    for slot,name in enumerate(objects):
        k='object_to_world_transform' if slot==0 else f'object_{slot}_to_world_transform'; mesh=b/('mesh.obj' if slot==0 else f'mesh_{slot}.obj')
        objs.append(dict(name=name,slot=slot,T=z[k].astype(float),V=obj_vertices(mesh),mesh=mesh.name))
    srcjson=ep_dir/'support_v1/support_in_bundle_world.json'; log=dict(episode=ep,objects=objects)
    if srcjson.exists():
        base=read(srcjson); cyls=[dict(center_bundle_world_m=c['center_bundle_world_m'],radius_m=c['radius_m'],height_m=c['height_m']) for c in base['cylinders']]; prov={k:base[k] for k in base if k not in ('cylinders','bundle_sha256','warning','schema_version','episode','frames','coordinates')}; log['origin']='fb11 support_v1'
    elif ep in HIST:
        cyls,prov=historical_support(ep,b); log['origin']='historical support carried into bundle world'
    else:
        log['origin']='none'; summary.append(dict(episode=ep,state='NO_SUPPORT_SOURCE')); print(f'ep{ep:02d} NO SUPPORT SOURCE'); continue
    for c in cyls: c['kind']='table'
    # frame-0 bottoms and footprints
    for o in objs:
        w=o['V']@o['T'][0][:3,:3].T+o['T'][0][:3,3]; z0=float(w[:,2].min()); low=w[w[:,2]<z0+0.01]; c=low[:,:2].mean(0)
        o.update(bottom=z0,foot_center=c.tolist(),foot_radius=float(np.clip(np.linalg.norm(low[:,:2]-c,axis=1).max(),0.03,0.15)),W0=w,
                 first10_bottom_range=[float(min((o['V']@o['T'][f][:3,:3].T+o['T'][f][:3,3])[:,2].min() for f in range(min(10,len(o['T']))))),float(max((o['V']@o['T'][f][:3,:3].T+o['T'][f][:3,3])[:,2].min() for f in range(min(10,len(o['T'])))))])
    T=min(o['bottom'] for o in objs)-0.002; before_top=max(c['center_bundle_world_m'][2]+c['height_m']/2 for c in cyls)
    log.update(table_top_before=before_top,table_top_after=T,lowered=[],disks=[],skipped_disks=[],warnings=[],per_object=[])
    for i,c in enumerate(cyls):
        top=c['center_bundle_world_m'][2]+c['height_m']/2
        if top>T: c['center_bundle_world_m'][2]-=(top-T); log['lowered'].append(dict(cylinder=i,by_cm=round((top-T)*100,2)))
    for o in objs:
        gap=o['bottom']-T; log['per_object'].append(dict(object=o['name'],bottom_f0=o['bottom'],gap_to_table_cm=round(gap*100,2),first10_bottom_range=o['first10_bottom_range'],foot_radius=o['foot_radius']))
        if gap<=0.005: continue
        c=np.array(o['foot_center']); r=o['foot_radius']; others=[p for p in objs if p is not o]
        rests=[p['name'] for p in others if ((np.linalg.norm(p['W0'][:,:2]-c,axis=1)<=r)&(p['W0'][:,2]>=o['bottom']-0.03)&(p['W0'][:,2]<=o['bottom']+0.01)).any()]
        if rests: log['skipped_disks'].append(dict(object=o['name'],gap_cm=round(gap*100,2),reason=f"rests on {rests}")); continue
        h=0.01; top=o['bottom']-0.002; cut=[]
        for p in others:
            m=(np.linalg.norm(p['W0'][:,:2]-c,axis=1)<=r)&(p['W0'][:,2]>=top-h)&(p['W0'][:,2]<=top)
            if m.any(): cut.append((p['name'],float(np.linalg.norm(p['W0'][m][:,:2]-c,axis=1).min())))
        if cut:
            r2=min(d for _,d in cut)-0.005
            if r2<0.03: log['skipped_disks'].append(dict(object=o['name'],gap_cm=round(gap*100,2),reason=f'disk would cut into {[n for n,_ in cut]}; radius {r2*100:.1f} cm < 3 cm')); log['warnings'].append(f"{o['name']} floats {gap*100:.1f} cm with no disk"); continue
            log['warnings'].append(f"{o['name']} disk shrunk {r*100:.1f} -> {r2*100:.1f} cm to clear {[n for n,_ in cut]}"); r=r2
        cyls.append(dict(center_bundle_world_m=[float(c[0]),float(c[1]),top-h/2],radius_m=float(r),height_m=h,kind='seat_disk',object=o['name'])); log['disks'].append(dict(object=o['name'],gap_cm=round(gap*100,2),radius_cm=round(r*100,1)))
    pen0=penetration(objs,cyls,0); pen10=[penetration(objs,cyls,f) for f in range(min(10,len(objs[0]['T'])))]
    worst10={o['name']:max(r['depth_cm'] for row in pen10 for r in row if r['object']==o['name']) for o in objs}
    seated=[{k:v for k,v in c.items()} for c in cyls]
    out=dict(schema_version=2,episode=ep,frames=int(len(objs[0]['T'])),coordinates='result.npz shared world, metres, Z-up',**prov,
             seating=dict(rule='T = min(object bottoms at frame 0) - 2 mm; supports lowered to T (never raised); disk (h 1 cm, top = bottom - 2 mm) under objects floating > 5 mm unless resting on another object; disks shrunk/dropped if they cut a neighbour',
                          reference='haisenk hand_fixes_wentao_v1_20261103 tools/wseat.py (2026-11-03), re-implemented in bundle world',**{k:log[k] for k in ['origin','table_top_before','table_top_after','lowered','disks','skipped_disks','warnings','per_object']}),
             bundle_sha256={f.name:sha(f) for f in b.iterdir() if f.suffix in ['.npz','.obj']},cylinders=seated,
             warning='Seated at frame 0 only (objects rest 2 mm above their support; no penetration at frame 0); physics, hand contact and policy success not validated')
    put(OUT/f'episode_{ep:06d}/support_v1/support_in_bundle_world.json',out)
    put(OUT/f'episode_{ep:06d}/support_v1/VALIDATION.json',dict(state='SEATED_FRAME0_NO_PENETRATION' if all(r['depth_cm']==0 for r in pen0) else 'SEATED_WITH_FRAME0_PENETRATION',frame0=pen0,max_depth_first10_cm=worst10,collision_shapes=len(cyls),policy_tested=False))
    summary.append(dict(episode=ep,origin=log['origin'],table_top_before=round(before_top,4),table_top_after=round(T,4),lowered_cm=[l['by_cm'] for l in log['lowered']],disks=log['disks'],skipped=log['skipped_disks'],warnings=log['warnings'],frame0_penetration=[r for r in pen0 if r['depth_cm']>0],max_depth_first10_cm=worst10))
    print(f"ep{ep:02d} {log['origin'][:22]:22s} top {before_top:+.4f} -> {T:+.4f} lowered {[l['by_cm'] for l in log['lowered']]} disks {[(d['object'],d['gap_cm'],d['radius_cm']) for d in log['disks']]} skipped {[(s['object'],s['reason'][:40]) for s in log['skipped_disks']]} pen0 {[(r['object'],round(r['depth_cm'],2)) for r in pen0 if r['depth_cm']>0]} first10max {dict((k,round(v,2)) for k,v in worst10.items())}",flush=True)
put(O/'SEATING_SUMMARY.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=str(S),out=str(OUT),episodes=summary)); print('WROTE',O/'SEATING_SUMMARY.json')
