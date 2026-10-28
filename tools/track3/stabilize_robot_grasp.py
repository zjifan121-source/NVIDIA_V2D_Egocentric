"""Stabilize an inspected robot grasp without changing object trajectories.

Uses actual Sharpa collision-pad vertices and bounded finger IK at an inspected
anchor, then retains that robot/object transform during the specified hold.
Contact targets alone do not certify a grasp: audit every runtime contact link
(palm and phalanges included), then test the dynamic approach before training.
Small pad compression models a compliant pinch target; it does not pin objects
or change physics, meshes, evaluator code, or object references.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,xml.etree.ElementTree as ET
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'path_b'));from pathb import fk

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('scene',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--grip',action='append',required=True,help='SIDE:OBJECT:START:END:ANCHOR; inspect the contact feature first');ap.add_argument('--compression-mm',type=float,default=.5);ap.add_argument('--blend',type=int,default=12);ap.add_argument('--max-wrist-shift-m',type=float,default=.15);ap.add_argument('--digits',nargs='+',choices=['thumb','index','middle'],default=['thumb','index']);args=ap.parse_args()
 S=args.scene.resolve();D=args.output.resolve()
 if len(args.digits)<2 or args.digits[0]!='thumb' or len(set(args.digits))!=len(args.digits):raise ValueError('Use thumb first and at least one distinct opposing digit')
 if D.exists():raise FileExistsError(D)
 if not 0<args.max_wrist_shift_m<=.4:raise ValueError('Maximum wrist correction must be positive and <=40cm')
 if not 0<=args.compression_mm<=1:raise ValueError('Compression must be between0 and1mm')
 p=next(S.glob('assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet'));t=pq.read_table(p);original=t;c=lambda k:np.asarray(t[k][0].as_py());pos=c('object_body_position');quat=c('object_body_wxyz');names=t['object_body_names'][0].as_py();changes={};audit=[];holds={}
 for spec in args.grip:
  side,body,a,b,anchor=spec.split(':');a,b,anchor=map(int,[a,b,anchor]);assert side in ['left','right'] and 0<=a<=anchor<b<=len(pos)
  for olda,oldb in holds.setdefault(side,[]):
   if max(a,olda)<min(b,oldb):raise ValueError('Overlapping holds for the same hand')
  holds[side].append((a,b));slot=next(i for i,n in enumerate(names) if n.endswith('_'+body));jn=t[f'{side}_robot_finger_joint_names'][0].as_py();q=changes.get(f'robot_{side}_finger_joints',c(f'robot_{side}_finger_joints')).copy();wp=changes.get(f'robot_{side}_wrist_position',c(f'robot_{side}_wrist_position')).copy();wq=changes.get(f'robot_{side}_wrist_wxyz',c(f'robot_{side}_wrist_wxyz')).copy();hand=fk.sharpa(side,'urdf');framehand=fk.sharpa(side);assert framehand.names==t[f'{side}_robot_frame_names'][0].as_py();assert np.allclose(framehand.frames(c(f'robot_{side}_wrist_position'),c(f'robot_{side}_wrist_wxyz'),dict(zip(jn,c(f'robot_{side}_finger_joints').T))),c(f'robot_{side}_frames'),atol=2e-6)
  up=Path(str(fk.FLASH_URDF).format(side=side));xml=ET.parse(up).getroot();mesh=trimesh.load(next(S.glob(f'score_input/meshes/episode_*/{body}.obj')),force='mesh',process=False);mesh.merge_vertices(merge_tex=True,merge_norm=True);mesh.process(validate=True)
  if not mesh.is_watertight:raise ValueError('Pad compression requires a watertight object mesh')
  R=Rotation.from_quat(quat[:,slot][:,[1,2,3,0]]).as_matrix();poses=hand.poses(wp[anchor],wq[anchor],dict(zip(jn,q[anchor])));pads=[];targets=[];normals=[]
  for digit in args.digits:
   name=f'{side}_{digit}_elastomer';link=xml.find(f"link[@name='{name}']");pad=trimesh.load((up.parent/link.find('collision/geometry/mesh').get('filename')).resolve(),force='mesh');T=poses[name][0];v=(pad.vertices@T[:3,:3].T+T[:3,3]-pos[anchor,slot])@R[anchor];near,dist,face=trimesh.proximity.closest_point(mesh,v);inside=mesh.contains(v);idx=np.where(inside,dist,-1).argmax() if inside.any() else dist.argmin();normal=mesh.face_normals[face[idx]];targets.append(near[idx]-normal*args.compression_mm*.001);normals.append(normal);pads.append((name,pad.vertices[idx]))
  normal_dot=float(min(normals[0]@n for n in normals[1:]))
  if normal_dot>=0:raise ValueError(f'{body} chosen anchor has no opposing thumb/index pad normals: {normal_dot}')
  indices=[i for i,n in enumerate(jn) if any('_'+digit+'_' in n for digit in args.digits)];q0=q[anchor].copy();limits=[xml.find(f"joint[@name='{jn[i]}']/limit") for i in indices];lo=np.maximum([float(x.get('lower')) for x in limits],q0[indices]-.6);hi=np.minimum([float(x.get('upper')) for x in limits],q0[indices]+.6);world_targets=np.asarray(targets)@R[anchor].T+pos[anchor,slot]
  def residual(x):
   angles=q0.copy();angles[indices]=x;poses=hand.poses(wp[anchor],wq[anchor],dict(zip(jn,angles)));points=np.array([poses[name][0,:3,:3]@v+poses[name][0,:3,3] for name,v in pads]);return np.r_[(points-world_targets).ravel(),.0005*(x-q0[indices])]
  fit=least_squares(residual,np.clip(q0[indices],lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=200,gtol=1e-10);err=float(np.linalg.norm(residual(fit.x)[:3*len(pads)].reshape(len(pads),3),axis=1).max())
  if err>.002:raise ValueError(f'{body} anchor contact is unreachable: {err}m')
  target_q=q0.copy();target_q[indices]=fit.x;local_wp=(wp[anchor]-pos[anchor,slot])@R[anchor];local_wr=R[anchor].T@Rotation.from_quat(wq[anchor,[1,2,3,0]]).as_matrix();max_shift=0
  for f in range(a,b):
   weight=min(1,(f-a+1)/(args.blend+1),(b-f)/(args.blend+1));weight=weight*weight*(3-2*weight);target_wp=R[f]@local_wp+pos[f,slot];target_wr=R[f]@local_wr;old_wr=Rotation.from_quat(wq[f,[1,2,3,0]]).as_matrix();max_shift=max(max_shift,float(np.linalg.norm(weight*(target_wp-wp[f]))));wp[f]=(1-weight)*wp[f]+weight*target_wp;delta=Rotation.from_matrix(target_wr@old_wr.T).as_rotvec();wq[f]=Rotation.from_matrix(Rotation.from_rotvec(weight*delta).as_matrix()@old_wr).as_quat()[[3,0,1,2]];q[f]=(1-weight)*q[f]+weight*target_q
  if max_shift>args.max_wrist_shift_m:raise ValueError(f'{body} robot correction {max_shift:.4f}m exceeds {args.max_wrist_shift_m}m: inspect the grasp alignment')
  changes.update({f'robot_{side}_wrist_position':wp,f'robot_{side}_wrist_wxyz':wq,f'robot_{side}_finger_joints':q,f'robot_{side}_frames':framehand.frames(wp,wq,dict(zip(jn,q.T)))})
  audit.append(dict(side=side,body=body,hold=[a,b],anchor=anchor,compression_mm=args.compression_mm,digits=args.digits,opposing_normal_dot=normal_dot,anchor_target_error_m=err,max_wrist_shift_m=max_shift));print(json.dumps(audit[-1]),flush=True)
 for key,value in changes.items():
  idx=t.column_names.index(key);field=t.schema.field(idx);t=t.set_column(idx,field,pa.array([value.tolist()],type=field.type))
 for key in ['object_mesh_paths','object_urdf_paths']:
  values=[str(D/'assets'/v.split('/assets/',1)[1]) for v in t[key][0].as_py()];idx=t.column_names.index(key);field=t.schema.field(idx);t=t.set_column(idx,field,pa.array([values],type=field.type))
 for key in original.column_names:
  if key not in changes and key not in ['object_mesh_paths','object_urdf_paths']:assert original[key].equals(t[key]),key
 shutil.copytree(S,D);pq.write_table(t,D/p.relative_to(S));(D/'robot_grasp_repair.json').write_text(json.dumps(dict(source=str(S),source_parquet_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),changed_columns=list(changes),object_references_unchanged=True,validation_scope="Pad target IK only; full-hand collision and dynamic approach must be checked separately",grasps=audit),indent=2)+'\n')
if __name__=='__main__':main()
