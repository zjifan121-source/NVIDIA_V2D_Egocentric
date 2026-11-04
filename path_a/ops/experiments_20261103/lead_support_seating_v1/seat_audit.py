"""Frame-0 support audit for the lead_sprint_v1 HF stage: per episode, table top z (support_v1 cylinders, bundle world,
Z-up) versus each object's lowest mesh vertex at frame 0 (and the lowest over the first 10 frames). Positive sink = object
below the table top. CPU only, no GT. Usage: seat_audit.py [stage_dir]"""
from pathlib import Path
import json, sys
import numpy as np, trimesh
S=Path(sys.argv[1] if len(sys.argv)>1 else '/Volumes/E/Code/V2D/work/lead-sprint-release-20261103/stage')
O=Path(__file__).resolve().parent
rows=[]
for ep_dir in sorted((S/'scenes').iterdir()):
    ep=int(ep_dir.name.split('_')[-1]); b=ep_dir/'bundle'; z=np.load(b/'result.npz',allow_pickle=False); objects=json.loads((b/'manifest.json').read_text())['objects'] if (b/'manifest.json').exists() else None
    if objects is None: objects=json.loads((ep_dir/'run.json').read_text())['objects']
    sup=ep_dir/'support_v1/support_in_bundle_world.json'
    rec=dict(episode=ep,objects=objects,has_support=sup.exists())
    if sup.exists():
        s=json.loads(sup.read_text()); cyl=s['cylinders']
        rec['cylinders']=[dict(center=c['center_bundle_world_m'],radius=c['radius_m'],height=c['height_m'],top=c['center_bundle_world_m'][2]+c['height_m']/2) for c in cyl]
        top=max(c['top'] for c in rec['cylinders'])
    else: top=None
    rec['table_top_z']=top; rec['per_object']=[]
    for slot,obj in enumerate(objects):
        k='object_to_world_transform' if slot==0 else f'object_{slot}_to_world_transform'
        T=z[k]; V=np.asarray(trimesh.load(b/('mesh.obj' if slot==0 else f'mesh_{slot}.obj'),force='mesh',process=False).vertices,float)
        w0=V@T[0][:3,:3].T+T[0][:3,3]; minz0=float(w0[:,2].min())
        mins=[float((V@T[f][:3,:3].T+T[f][:3,3])[:,2].min()) for f in range(min(10,len(T)))]
        # footprint at frame 0 (xy extent) for a possible disk
        rec['per_object'].append(dict(object=obj,slot=slot,min_z_f0=minz0,min_z_first10=float(min(mins)),max_z_first10=float(max(mins)),
            xy_center_f0=[float(w0[:,0].mean()),float(w0[:,1].mean())],xy_radius_f0=float(np.sqrt(((w0[:,:2]-w0[:,:2].mean(0))**2).sum(1)).max()),
            sink_cm_f0=(None if top is None else (top-minz0)*100)))
    rows.append(rec)
    line=f"ep{ep:02d} top={'none' if top is None else f'{top:+.4f}'} " + ' | '.join(f"{o['object']}: minz0 {o['min_z_f0']:+.4f} sink {o['sink_cm_f0']:+.1f}cm (first10 {o['min_z_first10']:+.4f}..{o['max_z_first10']:+.4f})" if top is not None else f"{o['object']}: minz0 {o['min_z_f0']:+.4f}" for o in rec['per_object'])
    print(line,flush=True)
(O/'SEAT_AUDIT.json').write_text(json.dumps(dict(stage=str(S),rows=rows),indent=2)+'\n'); print('WROTE',O/'SEAT_AUDIT.json')
