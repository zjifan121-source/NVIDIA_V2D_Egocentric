"""Test an object-relative pitcher wrist target with an unchanged trained actor."""
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
    info=json.loads((HERE/'ep20_reference.json').read_text());S=Path(info['entry']['scene'])
    source=Path(info['parquet']);assert hashlib.sha256(source.read_bytes()).hexdigest()==info['parquet_sha256']
    original=pq.read_table(source);cell=lambda k:np.asarray(original[k][0].as_py())
    names=original['object_body_names'][0].as_py();slot=next(i for i,n in enumerate(names) if n.endswith('_water_pitcher'))
    pos=cell('object_body_position');Robj=Rotation.from_quat(cell('object_body_wxyz')[:,slot,[1,2,3,0]]).as_matrix()
    wp=cell('robot_right_wrist_position');wq=cell('robot_right_wrist_wxyz')
    Rw=Rotation.from_quat(wq[:,[1,2,3,0]]).as_matrix();q=cell('robot_right_finger_joints')
    qnames=original['right_robot_finger_joint_names'][0].as_py()
    anchor=40;grip_p=Robj[anchor].T@(wp[anchor]-pos[anchor,slot]);grip_R=Robj[anchor].T@Rw[anchor]
    rows=[]
    for variant in ('control','rigid'):
        D=HERE/f'ep20_{variant}_scene';out=HERE/f'ep20_{variant}_eval'
        if D.exists() or out.exists():raise FileExistsError(D)
        table=original;new_wp=wp.copy();new_wq=wq.copy();changed=[]
        if variant=='rigid':
            for f in range(22,120):
                u=min(1.,(f-21)/11.,(120-f)/11.);weight=u*u*(3-2*u)
                target_p=Robj[f]@grip_p+pos[f,slot];target_R=Robj[f]@grip_R
                new_wp[f]=(1-weight)*wp[f]+weight*target_p
                delta=Rotation.from_matrix(target_R@Rw[f].T).as_rotvec()
                new_wq[f]=Rotation.from_matrix(Rotation.from_rotvec(weight*delta).as_matrix()@Rw[f]).as_quat()[[3,0,1,2]]
            changes=dict(robot_right_wrist_position=new_wp,robot_right_wrist_wxyz=new_wq,
                         robot_right_frames=fk.sharpa('right').frames(new_wp,new_wq,dict(zip(qnames,q.T))))
            for key,value in changes.items():
                idx=table.column_names.index(key);field=table.schema.field(idx)
                table=table.set_column(idx,field,pa.array([value.tolist()],type=field.type))
            changed=list(changes)
        for key in ('object_mesh_paths','object_urdf_paths'):
            idx=table.column_names.index(key);field=table.schema.field(idx)
            values=[str(D/'assets'/v.split('/assets/',1)[1]) for v in original[key][0].as_py()]
            table=table.set_column(idx,field,pa.array([values],type=field.type))
        for key in original.column_names:
            if key not in changed and key not in ('object_mesh_paths','object_urdf_paths'):assert original[key].equals(table[key]),key
        shift=float(np.linalg.norm(new_wp-wp,axis=1).max())
        if shift>.4:raise ValueError(f'Correction {shift:.3f}m exceeds inspected range')
        D.mkdir()
        for directory in ('assets','score_input'):shutil.copytree(S/directory,D/directory)
        target=D/source.relative_to(S);pq.write_table(table,target)
        raw=Path(info['checkpoint']).read_bytes();size=struct.unpack('<Q',raw[:8])[0]
        header=json.loads(raw[8:8+size]);payload=raw[8+size:];md=header['__metadata__']
        cfg=json.loads(md['user.resolved_config_json']);cfg['task']['parquet']=str(target.parent)
        md['user.resolved_config_json']=json.dumps(cfg,separators=(',',':'))
        md['user.robot_wrist_hold_probe']=variant;md['user.original_policy_sha256']=hashlib.sha256(raw).hexdigest()
        h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((-len(h))%8)
        out.mkdir();checkpoint=out/'policy.safetensors';checkpoint.write_bytes(struct.pack('<Q',len(h))+h+payload)
        (out/'checkpoints.json').write_text(json.dumps({'20':str(checkpoint)})+'\n')
        row=dict(variant=variant,scene=str(D),evaluation=str(out),hold=[22,120],anchor=anchor,
                 max_wrist_shift_m=shift,changed_columns=changed,object_and_left_hand_and_fingers_unchanged=True,
                 physics_unchanged=True,actor_tensors_identical=True)
        rows.append(row);print(json.dumps(row),flush=True)
    (HERE/'pitcher_probe_cases.json').write_text(json.dumps(rows,indent=2)+'\n')
    # Inspect the actual manipulation phases before interpreting this candidate.
    import io
    from PIL import Image,ImageDraw
    episode=next(e for e in json.loads((REPO/'tools/track3/episodes.json').read_text())['episodes'] if e['episode']==20)
    canvas=Image.new('RGB',(1200,1200),'white');draw=ImageDraw.Draw(canvas)
    for i,frame in enumerate((0,20,40,60,80,100,120,160)):
        raw=subprocess.check_output(['ffmpeg','-loglevel','error','-threads','1','-i',str(REPO/episode['video']),
            '-vf',f'select=eq(n\\,{frame})','-frames:v','1','-f','image2pipe','-vcodec','png','-'])
        im=Image.open(io.BytesIO(raw)).convert('RGB');im.thumbnail((600,280))
        x,y=(i%2)*600,(i//2)*300;canvas.paste(im,(x,y+20));draw.text((x+4,y+3),f'episode20 frame{frame}',fill='black')
    canvas.save(HERE/'ep20_video_sheet.jpg',quality=90)


def evaluate():
    rows=json.loads((HERE/'pitcher_probe_cases.json').read_text())
    for row in rows:
        out=Path(row['evaluation'])
        if not (out/'eval.done').exists():
            with (out/'eval.log').open('a') as log:
                subprocess.run([sys.executable,str(REPO/'assets/data/v2d_submission_kit/examples/run_policy_evaluation.py'),
                    '--checkpoints',str(out/'checkpoints.json'),'--results',str(out/'results'),
                    '--runtime-root',str(REPO/'baseline/robotic_grounding/flash_chord')],
                    cwd=REPO/'baseline/robotic_grounding/flash_chord',stdout=log,stderr=subprocess.STDOUT,check=True)
            (out/'eval.done').touch()
        result=json.loads((out/'results/episode_000020.json').read_text())
        row['metrics']=result['metrics'];row['step_count']=result['step_count']
        print(row['variant'],result['metrics']['add_auc'],result['metrics']['add_auc_per_body'],flush=True)
        (HERE/'pitcher_probe_results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','evaluate']);a=p.parse_args()
    prepare() if a.mode=='prepare' else evaluate()
