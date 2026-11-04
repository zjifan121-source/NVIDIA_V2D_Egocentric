"""Cheap full-rollout timing tests of the selected policy; no object motion changes.

Actual ep5 pregrasp wrist lag was ~10–13cm. Test small advances of only the right
robot wrist during acquisition, with unchanged fingers/left hand/object references.
The zero-advance copy is a same-runtime control. No rollout is uploaded automatically.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def prepare():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation
    sys.path.insert(0,str(REPO/'path_b'))
    from pathb import fk
    info=json.loads((HERE/'ep5_reference.json').read_text());S=Path(info['entry']['scene'])
    source=Path(info['parquet'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==info['parquet_sha256']
    original=pq.read_table(source)
    cell=lambda k:np.asarray(original[k][0].as_py())
    pos=cell('robot_right_wrist_position');quat=cell('robot_right_wrist_wxyz');q=cell('robot_right_finger_joints')
    names=original['right_robot_finger_joint_names'][0].as_py()
    n=len(pos);R=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
    rows=[]
    for advance in (0,2,4):
        D=HERE/f'ep5_timing_{advance}_scene';out=HERE/f'ep5_timing_{advance}_eval'
        if D.exists() or out.exists():raise FileExistsError(D)
        table=original;changed=[];newpos=pos.copy();newq=quat.copy()
        if advance:
            for frame in range(18,70):
                u=min(1.,(frame-17)/9.,(70-frame)/9.);w=u*u*(3-2*u)
                future=min(n-1,frame+advance)
                newpos[frame]=(1-w)*pos[frame]+w*pos[future]
                delta=Rotation.from_matrix(R[future]@R[frame].T).as_rotvec()
                newq[frame]=Rotation.from_matrix(Rotation.from_rotvec(w*delta).as_matrix()@R[frame]).as_quat()[[3,0,1,2]]
            changes=dict(robot_right_wrist_position=newpos,robot_right_wrist_wxyz=newq,
                         robot_right_frames=fk.sharpa('right').frames(newpos,newq,dict(zip(names,q.T))))
            for key,value in changes.items():
                idx=table.column_names.index(key);field=table.schema.field(idx)
                table=table.set_column(idx,field,pa.array([value.tolist()],type=field.type))
            changed=list(changes)
        for key in ('object_mesh_paths','object_urdf_paths'):
            idx=table.column_names.index(key);field=table.schema.field(idx)
            values=[str(D/'assets'/v.split('/assets/',1)[1]) for v in original[key][0].as_py()]
            table=table.set_column(idx,field,pa.array([values],type=field.type))
        for key in original.column_names:
            if key not in changed and key not in ('object_mesh_paths','object_urdf_paths'):
                assert original[key].equals(table[key]),key
        D.mkdir()
        for directory in ('assets','score_input'):shutil.copytree(S/directory,D/directory)
        target=D/source.relative_to(S);pq.write_table(table,target)
        raw=Path(info['checkpoint']).read_bytes();size=struct.unpack('<Q',raw[:8])[0]
        header=json.loads(raw[8:8+size]);payload=raw[8+size:];md=header['__metadata__']
        cfg=json.loads(md['user.resolved_config_json']);cfg['task']['parquet']=str(target.parent)
        md['user.resolved_config_json']=json.dumps(cfg,separators=(',',':'))
        md['user.right_wrist_acquisition_advance_frames']=str(advance)
        md['user.original_policy_sha256']=hashlib.sha256(raw).hexdigest()
        h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((-len(h))%8)
        out.mkdir();checkpoint=out/'policy.safetensors'
        checkpoint.write_bytes(struct.pack('<Q',len(h))+h+payload)
        copied=checkpoint.read_bytes();newsize=struct.unpack('<Q',copied[:8])[0]
        assert copied[8+newsize:]==payload
        (out/'checkpoints.json').write_text(json.dumps({'5':str(checkpoint)})+'\n')
        row=dict(advance_frames=advance,scene=str(D),evaluation=str(out),changed_columns=changed,
                 max_wrist_shift_m=float(np.linalg.norm(newpos-pos,axis=1).max()),
                 object_and_left_hand_and_fingers_unchanged=True,physics_unchanged=True,actor_tensors_identical=True)
        rows.append(row);print(json.dumps(row),flush=True)
    (HERE/'timing_probe_cases.json').write_text(json.dumps(rows,indent=2)+'\n')


def evaluate():
    rows=json.loads((HERE/'timing_probe_cases.json').read_text())
    for row in rows:
        out=Path(row['evaluation'])
        if not (out/'eval.done').exists():
            with (out/'eval.log').open('a') as log:
                subprocess.run([sys.executable,str(REPO/'assets/data/v2d_submission_kit/examples/run_policy_evaluation.py'),
                    '--checkpoints',str(out/'checkpoints.json'),'--results',str(out/'results'),
                    '--runtime-root',str(REPO/'baseline/robotic_grounding/flash_chord')],
                    cwd=REPO/'baseline/robotic_grounding/flash_chord',stdout=log,stderr=subprocess.STDOUT,check=True)
            (out/'eval.done').touch()
        result=json.loads((out/'results/episode_000005.json').read_text())
        row['metrics']=result['metrics'];row['step_count']=result['step_count']
        print(row['advance_frames'],result['metrics']['add_auc'],result['metrics']['add_auc_per_body'],flush=True)
        (HERE/'timing_probe_results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','evaluate']);a=p.parse_args()
    prepare() if a.mode=='prepare' else evaluate()
