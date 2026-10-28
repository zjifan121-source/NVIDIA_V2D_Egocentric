"""Apply a geometry-only assembly-fit correction to robot/object references.

No ground truth or policy rollout is used by the fit. Object 2 retains observed
pickup/carry, then receives a smooth bounded rigid correction from frame190..220.
Its grasping robot hand follows the same correction. Visual/scoring meshes stay
identical. Collision-piece installation is performed separately after validation.
"""
from pathlib import Path
import json,shutil,sys,hashlib
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
sys.path.insert(0,'path_b');from pathb import fk
P=Path('reports/submission_candidate_2026-10-28').resolve();S=P/'ep25_pinch_scene';D=P/'ep25_assembly_scene'
if D.exists():raise FileExistsError(D)
p=next(S.glob('assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet'));t=pq.read_table(p);original=t;c=lambda k:np.asarray(t[k][0].as_py());pos=c('object_body_position');q=c('object_body_wxyz');wp=c('robot_right_wrist_position');wq=c('robot_right_wrist_wxyz');j=c('robot_right_finger_joints');jn=t['right_robot_finger_joint_names'][0].as_py();fit=json.loads((P/'ep25_slot_rigid_fit.json').read_text());x=np.array(fit['correction_in_board1_local']);assert np.linalg.norm(x[:3])<.02 and np.linalg.norm(x[3:])<np.deg2rad(6)
oldpos=pos.copy();oldq=q.copy()
for f in range(len(pos)):
 weight=np.clip((f-190)/30,0,1);weight=weight*weight*(3-2*weight)
 if weight==0:continue
 R1=Rotation.from_quat(q[f,0,[1,2,3,0]]).as_matrix();R2=Rotation.from_quat(q[f,1,[1,2,3,0]]).as_matrix();dr=R1@Rotation.from_rotvec(weight*x[3:]).as_matrix()@R1.T;delta=weight*R1@x[:3]
 wp[f]=dr@(wp[f]-pos[f,1])+pos[f,1]+delta;wq[f]=Rotation.from_matrix(dr@Rotation.from_quat(wq[f,[1,2,3,0]]).as_matrix()).as_quat()[[3,0,1,2]];pos[f,1]+=delta;q[f,1]=Rotation.from_matrix(dr@R2).as_quat()[[3,0,1,2]]
assert np.array_equal(pos[:191],oldpos[:191]) and np.array_equal(q[:191],oldq[:191]);assert np.array_equal(pos[:,0],oldpos[:,0]) and np.array_equal(q[:,0],oldq[:,0])
frames=fk.sharpa('right').frames(wp,wq,dict(zip(jn,j.T)));changes=dict(object_body_position=pos,object_body_wxyz=q,robot_right_wrist_position=wp,robot_right_wrist_wxyz=wq,robot_right_frames=frames)
for key,value in changes.items():
 idx=t.column_names.index(key);field=t.schema.field(idx);t=t.set_column(idx,field,pa.array([value.tolist()],type=field.type))
for key in ['object_mesh_paths','object_urdf_paths']:
 values=[str(D/'assets'/v.split('/assets/',1)[1]) for v in t[key][0].as_py()];idx=t.column_names.index(key);field=t.schema.field(idx);t=t.set_column(idx,field,pa.array([values],type=field.type))
for key in original.column_names:
 if key not in changes and key not in ['object_mesh_paths','object_urdf_paths']:assert original[key].equals(t[key]),key
shutil.copytree(S,D);pq.write_table(t,D/p.relative_to(S));provenance=dict(source=str(S),source_parquet_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fit='ep25_slot_rigid_fit.json',correction=x.tolist(),blend_frames=[190,220],changed_columns=list(changes),asset_paths_localized=True,meshes_unchanged=True,object1_unchanged=True,pickup_through_frame190_unchanged=True)
(D/'assembly_fit.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance,indent=2))
