"""Compare actual successful donor grasp against geometry-only target registrations."""
from pathlib import Path
import sys,json,xml.etree.ElementTree as ET
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[1]/'path_b'));from pathb import fk
j=json.loads((P/'spoon_grasp_registration.json').read_text());plan=json.loads((P/'plan.json').read_text());up=Path(str(fk.FLASH_URDF).format(side='right'));xml=ET.parse(up).getroot();rows=[]
variants=dict(j['registrations']);r=variants['rigid'];R=np.asarray(r['rotation']);pos=R.T@(np.asarray(r['wrist_position_target_object_m'])-r['translation_m']);rot=R.T@Rotation.from_quat(np.asarray(r['wrist_wxyz_target_object'])[[1,2,3,0]]).as_matrix();variants['original_successful']=dict(wrist_position_target_object_m=pos,wrist_wxyz_target_object=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]])
for mode,r in variants.items():
 ep=37 if mode=='original_successful' else 33
 obj=trimesh.load(Path(plan[str(ep)]['scene'])/f'score_input/meshes/episode_{ep:06d}/wooden_spoon.obj',force='mesh');obj.merge_vertices();obj.process(validate=True)
 poses=fk.sharpa('right','urdf').poses(np.array(r['wrist_position_target_object_m']),np.array(r['wrist_wxyz_target_object']),dict(zip(j['finger_joint_names'],j['finger_joint_positions'])));links=[]
 for link in xml.findall('link'):
  name=link.get('name')
  if name not in poses:continue
  for coll in link.findall('collision'):
   me=coll.find('geometry/mesh')
   if me is None:continue
   mesh=trimesh.load((up.parent/me.get('filename')).resolve(),force='mesh');o=coll.find('origin');xyz=np.fromstring(o.get('xyz','0 0 0'),sep=' ') if o is not None else np.zeros(3);rpy=np.fromstring(o.get('rpy','0 0 0'),sep=' ') if o is not None else np.zeros(3);v=mesh.vertices@Rotation.from_euler('xyz',rpy).as_matrix().T+xyz;T=poses[name][0];v=v@T[:3,:3].T+T[:3,3];near,dist,face=trimesh.proximity.closest_point(obj,v);inside=obj.contains(v);idx=int(dist.argmin());links.append(dict(link=name,min_gap_mm=float(dist.min()*1000),max_penetration_mm=float(dist[inside].max()*1000) if inside.any() else 0,closest_point=near[idx].tolist(),surface_normal=obj.face_normals[face[idx]].tolist()))
 row=dict(registration=mode,links=sorted(links,key=lambda x:x['min_gap_mm']),diagnostic_only=True);rows.append(row);print(json.dumps(row),flush=True)
(P/'donor_spoon_grasp_static_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
