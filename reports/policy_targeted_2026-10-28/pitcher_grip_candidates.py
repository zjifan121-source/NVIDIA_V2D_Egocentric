"""Two observed-hold retargeting candidates for the off-camera pitcher grasp.

Video frames20..120 show a continuous right-hand handle grasp while the pitcher
pours. Hand tracking drifts >20cm when it leaves the image. Keep one inspected
robot wrist/finger grasp through that hold, preserving every object pose and mesh.
"""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'path_b'))
from pathb import fk
info=json.loads((HERE/'ep20_reference.json').read_text());S=Path(info['entry']['scene']);p=Path(info['parquet'])
assert hashlib.sha256(p.read_bytes()).hexdigest()==info['parquet_sha256']
original=pq.read_table(p);cell=lambda k:np.asarray(original[k][0].as_py())
pos=cell('object_body_position');names=original['object_body_names'][0].as_py();slot=next(i for i,n in enumerate(names) if n.endswith('_water_pitcher'))
Robj=Rotation.from_quat(cell('object_body_wxyz')[:,slot,[1,2,3,0]]).as_matrix()
wp=cell('robot_right_wrist_position');wq=cell('robot_right_wrist_wxyz');Rw=Rotation.from_quat(wq[:,[1,2,3,0]]).as_matrix()
q=cell('robot_right_finger_joints');jn=original['right_robot_finger_joint_names'][0].as_py()
cases=json.loads((HERE/'pitcher_probe_cases.json').read_text())
for anchor in (30,40):
    variant=f'grip{anchor}';D=HERE/f'ep20_{variant}_scene';out=HERE/f'ep20_{variant}_eval'
    if D.exists() or out.exists():raise FileExistsError(D)
    gp=Robj[anchor].T@(wp[anchor]-pos[anchor,slot]);gR=Robj[anchor].T@Rw[anchor]
    newwp=wp.copy();newwq=wq.copy();newq=q.copy()
    for f in range(22,150):
        u=min(1.,(f-21)/11.,(150-f)/11.);weight=u*u*(3-2*u)
        newwp[f]=(1-weight)*wp[f]+weight*(Robj[f]@gp+pos[f,slot])
        delta=Rotation.from_matrix(Robj[f]@gR@Rw[f].T).as_rotvec()
        newwq[f]=Rotation.from_matrix(Rotation.from_rotvec(weight*delta).as_matrix()@Rw[f]).as_quat()[[3,0,1,2]]
        newq[f]=(1-weight)*q[f]+weight*q[anchor]
    shift=float(np.linalg.norm(newwp-wp,axis=1).max());assert shift<.4
    changes=dict(robot_right_wrist_position=newwp,robot_right_wrist_wxyz=newwq,robot_right_finger_joints=newq,
                 robot_right_frames=fk.sharpa('right').frames(newwp,newwq,dict(zip(jn,newq.T))))
    table=original
    for key,value in changes.items():
        i=table.column_names.index(key);field=table.schema.field(i)
        table=table.set_column(i,field,pa.array([value.tolist()],type=field.type))
    for key in ('object_mesh_paths','object_urdf_paths'):
        i=table.column_names.index(key);field=table.schema.field(i)
        paths=[str(D/'assets'/v.split('/assets/',1)[1]) for v in original[key][0].as_py()]
        table=table.set_column(i,field,pa.array([paths],type=field.type))
    for key in original.column_names:
        if key not in changes and key not in ('object_mesh_paths','object_urdf_paths'):assert original[key].equals(table[key]),key
    D.mkdir()
    for directory in ('assets','score_input'):shutil.copytree(S/directory,D/directory)
    target=D/p.relative_to(S);pq.write_table(table,target)
    raw=Path(info['checkpoint']).read_bytes();size=struct.unpack('<Q',raw[:8])[0]
    header=json.loads(raw[8:8+size]);payload=raw[8+size:];md=header['__metadata__']
    cfg=json.loads(md['user.resolved_config_json']);cfg['task']['parquet']=str(target.parent)
    md['user.resolved_config_json']=json.dumps(cfg,separators=(',',':'));md['user.pitcher_grip_reference_anchor']=str(anchor)
    md['user.original_policy_sha256']=hashlib.sha256(raw).hexdigest()
    h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((-len(h))%8)
    out.mkdir();checkpoint=out/'policy.safetensors';checkpoint.write_bytes(struct.pack('<Q',len(h))+h+payload)
    (out/'checkpoints.json').write_text(json.dumps({'20':str(checkpoint)})+'\n')
    row=dict(variant=variant,scene=str(D),evaluation=str(out),hold=[22,150],anchor=anchor,
             max_wrist_shift_m=shift,changed_columns=list(changes),object_and_left_hand_unchanged=True,
             physics_unchanged=True,actor_tensors_identical=True)
    cases.append(row);print(json.dumps(row),flush=True)
(HERE/'pitcher_probe_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
