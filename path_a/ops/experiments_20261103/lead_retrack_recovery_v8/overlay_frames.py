"""Montage of current (yellow) vs candidate (cyan) silhouettes over RGB with the observed mask (green) at 8 frames.
Usage: overlay_frames.py <experiment dir> <case name>"""
from pathlib import Path
import json, sys, numpy as np, cv2
ROOT=Path('/Volumes/E/Code/V2D'); OPS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops'
D=OPS/'experiments_20261029/scored_scene_consistency_v1'; sys.path.insert(0,str(D/'ep5_observed_geometry_v1'))
from surface_renderer import raster_surface
import trimesh
X=Path(sys.argv[1]); name=sys.argv[2]; B=OPS/'experiments_20261030/scored20_dual_anchor_v1'; REL=ROOT/'work/scored11-fulltrack-20261103/release'
batch=json.loads((X/'FROZEN_BATCH.json').read_text()); c=next(x for x in batch['cases'] if x['case']==name)
ep,obj,slot=c['episode'],c['object'],c['slot']
man=json.loads((B/'cases'/f'ep{ep:02d}_{obj}'/'INPUT_MANIFEST.json').read_text()); cache=Path(man['payload_local']); K=json.loads((cache/'K.json').read_text())
Km=np.array([[K['fx'],0,K['cx']],[0,K['fy'],K['cy']],[0,0,1.]]); shape=(K['height'],K['width'])
def load(bundle):
    z=np.load(bundle/'result.npz',allow_pickle=False); k='object_to_world_transform' if slot==0 else 'object_1_to_world_transform'
    T=np.array([np.linalg.solve(cc.astype(float),w.astype(float)) for cc,w in zip(z['camera_to_world_transform'],z[k])])
    m=trimesh.load(bundle/('mesh.obj' if slot==0 else 'mesh_1.obj'),force='mesh',process=False); return T,np.asarray(m.vertices,float),np.asarray(m.faces)
cur=load(REL/f'scenes/episode_{ep:06d}/bundle'); cand=load(X/'run_v1/cases'/name/'candidate_bundle')
n=c['frames']; ids=[int(round(i*(n-1)/7)) for i in range(8)]; tiles=[]
for f in ids:
    rgb=cv2.imread(str(cache/man['frames'][f]['rgb'])); mask=cv2.imread(str(cache/man['frames'][f]['mask']),0)>0
    for (T,V,F),col in [(cur,(0,255,255)),(cand,(255,255,0))]:
        vc=V@T[f][:3,:3].T+T[f][:3,3]; dep=raster_surface(vc,F,Km,shape)[0]; sil=(dep>0).astype(np.uint8)
        cs,_=cv2.findContours(sil,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(rgb,cs,-1,col,2)
    cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(rgb,cs,-1,(0,255,0),1)
    cv2.putText(rgb,f'f{f}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2); tiles.append(cv2.resize(rgb,(507,380)))
rows=[np.hstack(tiles[:4]),np.hstack(tiles[4:])]; out=X/'run_v1/cases'/name/'overlay_montage.png'; cv2.imwrite(str(out),np.vstack(rows)); print(out)
