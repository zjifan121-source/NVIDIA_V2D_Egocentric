"""Hand-object contact as the policy sees it: in the retargeted (sim-frame) parquet, for every frame on which an object is
lifted (> 2 cm above its first-10-frame minimum), the distance from the closest robot link of each hand to that object's
surface (processed mesh at the frame's pose). Reports the median distance and the fraction of lifted frames with a link
within 1 cm / 2 cm, per scene, object and hand. This is the criterion the team loader's attach_hands is meant to fix.
Usage: hand_contact.py <scene dir> [...]  -> writes HAND_CONTACT.json in each scene dir."""
from pathlib import Path
import sys, json, glob
import numpy as np, trimesh, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
for sc in [Path(a) for a in sys.argv[1:]]:
    f=glob.glob(str(sc/'assets/human_motion_data/ego_recon/processed/*/robot_name=sharpa_wave/*.parquet'))[0]
    t=pq.read_table(f); d={k:v[0] for k,v in t.to_pydict().items()}
    P=np.asarray(d['object_body_position'],float); Q=np.asarray(d['object_body_wxyz'],float); names=list(d['object_body_names']); meshes=list(d['object_mesh_paths'])
    out=dict(scene=sc.name,frames=int(len(P)),objects={})
    for b,(n,mp) in enumerate(zip(names,meshes)):
        mp=Path(mp); mp=mp if mp.exists() else (sc/'assets/human_motion_data/ego_recon/processed'/mp.name)
        mesh=trimesh.load(mp,force='mesh',process=False); pqry=trimesh.proximity.ProximityQuery(mesh)
        rot=Rotation.from_quat(Q[:,b][:,[1,2,3,0]]).as_matrix(); z=P[:,b,2]; lifted=z>z[:10].min()+0.02
        res=dict(lifted_frames=int(lifted.sum()))
        for side in ('left','right'):
            links=np.asarray(d[f'robot_{side}_frames'],float)[:,2:,:3]  # skip universe/root
            idx=np.flatnonzero(lifted)
            if len(idx)==0: res[side]=None; continue
            local=np.einsum('tji,tkj->tki',rot[idx],links[idx]-P[idx,b][:,None,:]); pts=local.reshape(-1,3)
            sd=np.concatenate([pqry.signed_distance(pts[i:i+4000]) for i in range(0,len(pts),4000)]).reshape(local.shape[:2])
            dist=np.where(sd>0,0.0,-sd).min(axis=1)  # 0 inside the mesh, else distance to the surface
            res[side]=dict(closest_link_to_surface_cm_median=round(float(np.median(dist))*100,2),within_1cm=round(float(np.mean(dist<0.01)),3),within_2cm=round(float(np.mean(dist<0.02)),3),within_5cm=round(float(np.mean(dist<0.05)),3))
        out['objects'][n.split('_',2)[-1] if n.count('_')>=2 else n]=res
    (sc/'HAND_CONTACT.json').write_text(json.dumps(out,indent=1)+'\n')
    print(sc.name,json.dumps(out['objects']),flush=True)
