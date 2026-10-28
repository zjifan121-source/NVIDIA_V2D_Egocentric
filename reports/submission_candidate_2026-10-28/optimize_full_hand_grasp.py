"""Offline collision-aware anchor fit; produces no policy score or modified trajectory."""
from pathlib import Path
import ast,json,sys,time,xml.etree.ElementTree as ET
import numpy as np,trimesh,pyarrow.parquet as pq
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
P=Path(__file__).resolve().parent;REPO=P.parents[1];sys.path.insert(0,str(REPO/'path_b'));from pathb import fk
S=Path('/data/user_data/edmondwa/v2d/ken_night/variants/ep33_B');side='left';frame=96
T=pq.read_table(next(S.glob('assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet')));c=lambda k:np.asarray(T[k][0].as_py());jn=T['left_robot_finger_joint_names'][0].as_py();q0=c('robot_left_finger_joints')[frame];op=c('object_body_position')[frame,0];oR=Rotation.from_quat(c('object_body_wxyz')[frame,0,[1,2,3,0]]).as_matrix();p0=(c('robot_left_wrist_position')[frame]-op)@oR;R0=oR.T@Rotation.from_quat(c('robot_left_wrist_wxyz')[frame,[1,2,3,0]]).as_matrix();up=Path(str(fk.FLASH_URDF).format(side=side));xml=ET.parse(up).getroot();hand=fk.sharpa(side,'urdf');parents={j.find('child').get('link'):(j.find('parent').get('link'),j.get('type')) for j in xml.findall('joint')};mod=ast.parse((REPO/'baseline/robotic_grounding/flash_chord/src/flash_chord/embodiments/sharpa_collision.py').read_text());roles=next(ast.literal_eval(n.value) for n in mod.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='_OBJECT_CONTACT_LINKS' for a in n.targets));geoms=[];np.random.seed(0)
for link in xml.findall('link'):
 name=link.get('name');owner=name
 while owner in parents and parents[owner][1]=='fixed':owner=parents[owner][0]
 if owner.removeprefix(side+'_') not in roles:continue
 for coll in link.findall('collision'):
  me=coll.find('geometry/mesh')
  if me is None:continue
  mesh=trimesh.load((up.parent/me.get('filename')).resolve(),force='mesh').convex_hull;o=coll.find('origin');origin=np.fromstring(o.get('xyz','0 0 0'),sep=' ') if o is not None else np.zeros(3);rpy=np.fromstring(o.get('rpy','0 0 0'),sep=' ') if o is not None else np.zeros(3);r=Rotation.from_euler('xyz',rpy).as_matrix();points=np.r_[mesh.vertices,mesh.sample(180)]@r.T+origin;geoms.append((name,points,mesh.vertices@r.T+origin))
z=np.load(P/'ep33_cup_collision_geometry.npz');hulls=[]
for i in range(int(z['n'])):
 v=z[f'v{i}'];e=ConvexHull(v).equations;hulls.append((v.min(0),v.max(0),e[:,:3],e[:,3]))
obj=trimesh.Trimesh(z['cup_v'],z['cup_f'],process=True);poses0=hand.poses(p0,Rotation.from_matrix(R0).as_quat()[[3,0,1,2]],dict(zip(jn,q0)));contacts=[]
for digit in ['thumb','index']:
 name=f'left_{digit}_elastomer';v=next(v for n,p,v in geoms if n==name);xf=poses0[name][0];points=v@xf[:3,:3].T+xf[:3,3];nearest,distance,face=trimesh.proximity.closest_point(obj,points);idx=distance.argmin();target=nearest[idx]-obj.face_normals[face[idx]]*.0002;contacts.append((name,v[idx],target))

def decode(x):return p0+.01*x[:3],Rotation.from_rotvec(.1*x[3:6]).as_matrix()@R0,q0+.1*x[6:]

def measure(x):
 p,r,q=decode(x);poses=hand.poses(p,Rotation.from_matrix(r).as_quat()[[3,0,1,2]],dict(zip(jn,q)));parts=[];groups=[];allows=[]
 for name,points,verts in geoms:
  xf=poses[name][0];parts.append(points@xf[:3,:3].T+xf[:3,3]);groups.append(len(points));allows.extend([.0005 if 'elastomer' in name else .0001]*len(points))
 points=np.concatenate(parts);sd=np.full(len(points),.1)
 for lo,hi,norm,off in hulls:
  mask=np.all((points>=lo-.001)&(points<=hi+.001),axis=1)
  if mask.any():sd[mask]=np.minimum(sd[mask],np.max(points[mask]@norm.T+off,axis=1))
 pen=np.maximum(-sd-np.array(allows),0)*1000;start=0;worst=[]
 for count in groups:worst.append(float(pen[start:start+count].max()));start+=count
 contact_error=[]
 for name,v,target in contacts:
  xf=poses[name][0];contact_error.append(xf[:3,:3]@v+xf[:3,3]-target)
 contact_error=np.asarray(contact_error)*1000
 loss=10*np.sum(np.square(worst))+np.mean(pen**2)+10*np.sum(contact_error**2)+.01*np.sum(x[:3]**2)+.005*np.sum(x[3:6]**2)+.0005*np.sum(x[6:]**2)
 return float(loss),dict(max_penetration_beyond_allowance_mm=max(worst),contact_error_mm=np.linalg.norm(contact_error,axis=1).tolist(),per_link_penetration_mm=dict(zip([n for n,p,v in geoms],worst)))

best=[float('inf'),None,None];calls=[0];start=time.time()
def objective(x):
 loss,report=measure(x);calls[0]+=1
 if loss<best[0]:best[:]=[loss,x.copy(),report]
 if calls[0]%100==0:
  report=dict(loss=best[0],calls=calls[0],seconds=time.time()-start,x=best[1].tolist(),**best[2]);tmp=P/'full_hand_fit_progress.tmp.json';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(P/'full_hand_fit_progress.json')
 return loss
limits=[xml.find(f"joint[@name='{n}']/limit") for n in jn];bounds=[(-3.,3.)]*3+[(-3.5,3.5)]*3+[(max(float(j.get('lower'))-q,-.7)/.1,min(float(j.get('upper'))-q,.7)/.1) for q,j in zip(q0,limits)];x0=np.zeros(28);x0=np.array([np.clip(v,a+1e-8,b-1e-8) for v,(a,b) in zip(x0,bounds)]);print('INITIAL',json.dumps(measure(x0)),flush=True);fit=minimize(objective,x0,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=200,maxfun=10000,eps=1e-3,ftol=1e-11,gtol=1e-5));p,r,q=decode(best[1]);out=dict(source=str(S),side=side,object='blue_cup',anchor_frame=frame,success=bool(fit.success),message=str(fit.message),loss=best[0],calls=calls[0],seconds=time.time()-start,wrist_position_object_m=p.tolist(),wrist_wxyz_object=Rotation.from_matrix(r).as_quat()[[3,0,1,2]].tolist(),finger_joint_names=jn,finger_joint_positions=q.tolist(),**best[2]);(P/'full_hand_grasp_fit.json').write_text(json.dumps(out,indent=2)+'\n');print('RESULT',json.dumps(out),flush=True)
