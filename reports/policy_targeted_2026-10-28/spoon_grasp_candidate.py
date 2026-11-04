"""Propose a robot-reference grasp from a successful policy, keeping reconstruction fixed.

This is a diagnostic candidate, not a certified grasp. It must pass dynamic full
rollouts before training/selection. Only right robot wrist/finger/frame columns change.
"""
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'path_b'))
from pathb import fk


def register(source,target):
    rng=np.random.default_rng(7)
    # Local RNG support differs across trimesh versions; use deterministic vertex subsamples.
    X=np.asarray(source.vertices);Y=np.asarray(target.vertices)
    X=X[rng.choice(len(X),min(5000,len(X)),replace=False)]
    Y=Y[rng.choice(len(Y),min(8000,len(Y)),replace=False)]
    cx,cy=X.mean(0),Y.mean(0)
    Ax=np.linalg.svd(X-cx,full_matrices=False)[2]
    Ay=np.linalg.svd(Y-cy,full_matrices=False)[2]
    tree=cKDTree(Y);best=None
    for signs in itertools.product([-1,1],repeat=3):
        R=Ay.T@np.diag(signs)@Ax
        if np.linalg.det(R)<0:continue
        t=cy-R@cx
        for _ in range(60):
            moved=X@R.T+t;d,ids=tree.query(moved);keep=d<=np.quantile(d,.85)
            a,b=moved[keep],Y[ids[keep]];ac,bc=a.mean(0),b.mean(0)
            U,_,Vt=np.linalg.svd((a-ac).T@(b-bc));dR=Vt.T@U.T
            if np.linalg.det(dR)<0:Vt[-1]*=-1;dR=Vt.T@U.T
            dt=bc-dR@ac;R=dR@R;t=dR@t+dt
        moved=X@R.T+t;d=tree.query(moved)[0];rev=cKDTree(moved).query(Y)[0]
        error=float(np.sqrt(np.mean(np.r_[np.sort(d)[:int(.85*len(d))],np.sort(rev)[:int(.85*len(rev))]]**2)))
        if best is None or error<best[0]:best=(error,R,t)
    return best


def main():
    old=REPO/'reports/submission_candidate_2026-10-28'
    plan=json.loads((old/'plan.json').read_text())
    entry=json.loads((HERE/'ep5_reference.json').read_text())
    S=Path(entry['entry']['scene']);D=HERE/'ep5_donor_grasp_scene'
    if D.exists():raise FileExistsError(D)
    source_mesh=trimesh.load(Path(plan['37']['scene'])/'score_input/meshes/episode_000037/wooden_spoon.obj',force='mesh',process=False)
    target_mesh=trimesh.load(S/'score_input/meshes/episode_000005/wooden_spoon.obj',force='mesh',process=False)
    error,Rreg,treg=register(source_mesh,target_mesh)
    if error>.008:raise ValueError(f'Spoon rigid registration error {error:.4f}m exceeds 8mm')
    trace=np.load(old/'ep37_successful_grasp_trace.npz')
    meta=json.loads((old/'ep37_successful_grasp_trace.json').read_text())
    f=60;body=trace['body_q'][f];ob=body[int(trace['object_body_ids'][0])]
    wid=next(i for i,n in enumerate(meta['body_labels']) if n.endswith('/right_hand_C_MC'))
    Ro=Rotation.from_quat(ob[3:]).as_matrix();Rw=Rotation.from_quat(body[wid,3:]).as_matrix()
    grip_p=Rreg@(Ro.T@(body[wid,:3]-ob[:3]))+treg
    grip_R=Rreg@Ro.T@Rw
    ji=[i for i,n in enumerate(meta['joint_labels']) if n.startswith('right_sharpa_wave/')]
    joint_names=[meta['joint_labels'][i].split('/')[-1] for i in ji]
    measured_q=trace['joint_q'][f,trace['joint_q_start'][ji]]
    original=pq.read_table(entry['parquet']);table=original
    cell=lambda k:np.asarray(table[k][0].as_py())
    names=table['object_body_names'][0].as_py();slot=next(i for i,n in enumerate(names) if n.endswith('_wooden_spoon'))
    qnames=table['right_robot_finger_joint_names'][0].as_py()
    target_q=np.array([measured_q[joint_names.index(n)] for n in qnames])
    pos=cell('object_body_position');quat=cell('object_body_wxyz')
    Robj=Rotation.from_quat(quat[:,slot,[1,2,3,0]]).as_matrix()
    wp=cell('robot_right_wrist_position').copy();wq=cell('robot_right_wrist_wxyz').copy()
    q=cell('robot_right_finger_joints').copy();n=len(pos)
    old_wp=wp.copy();weights=np.zeros(n)
    start,end,blend=22,252,10
    for k in range(start,end):
        u=min(1.,(k-start+1)/(blend+1),(end-k)/(blend+1));weights[k]=u*u*(3-2*u)
        target_p=Robj[k]@grip_p+pos[k,slot]
        old_R=Rotation.from_quat(wq[k,[1,2,3,0]]).as_matrix()
        target_R=Robj[k]@grip_R
        delta=Rotation.from_matrix(target_R@old_R.T).as_rotvec()
        wp[k]=(1-weights[k])*wp[k]+weights[k]*target_p
        wq[k]=Rotation.from_matrix(Rotation.from_rotvec(weights[k]*delta).as_matrix()@old_R).as_quat()[[3,0,1,2]]
        q[k]=(1-weights[k])*q[k]+weights[k]*target_q
    shift=float(np.linalg.norm(wp-old_wp,axis=1).max())
    if shift>.25:raise ValueError(f'Wrist shift {shift:.3f}m exceeds 25cm; inspect registration')
    changes=dict(robot_right_wrist_position=wp,robot_right_wrist_wxyz=wq,robot_right_finger_joints=q,
                 robot_right_frames=fk.sharpa('right').frames(wp,wq,dict(zip(qnames,q.T))))
    for key,value in changes.items():
        i=table.column_names.index(key);field=table.schema.field(i)
        table=table.set_column(i,field,pa.array([value.tolist()],type=field.type))
    for key in ('object_mesh_paths','object_urdf_paths'):
        i=table.column_names.index(key);field=table.schema.field(i)
        paths=[str(D/'assets'/s.split('/assets/',1)[1]) for s in table[key][0].as_py()]
        table=table.set_column(i,field,pa.array([paths],type=field.type))
    for key in original.column_names:
        if key not in changes and key not in ('object_mesh_paths','object_urdf_paths'):
            assert table[key].equals(original[key]),key
    # Preserve meshes, collisions, supports and all left-hand data exactly.
    D.mkdir()
    for dirname in ('assets','score_input'):shutil.copytree(S/dirname,D/dirname)
    source_p=Path(entry['parquet']);out_p=D/source_p.relative_to(S)
    pq.write_table(table,out_p)
    audit=dict(diagnostic_only=True,source_scene=str(S),source_donor_episode=37,source_donor_frame=f,
               rigid_registration_rmse_m=error,rotation=Rreg.tolist(),translation_m=treg.tolist(),
               hold=[start,end],blend=blend,max_wrist_shift_m=shift,changed_columns=list(changes),
               object_references_unchanged=True,left_hand_unchanged=True,physics_and_meshes_unchanged=True,
               source_parquet_sha256=entry['parquet_sha256'],target_parquet=str(out_p),
               grasp_position_object_m=grip_p.tolist(),grasp_wxyz_object=Rotation.from_matrix(grip_R).as_quat()[[3,0,1,2]].tolist(),
               source_actual_finger_q=target_q.tolist(),source_finger_joint_names=qnames,
               validation_scope='Rigid mesh fit and measured donor pose only; dynamic validation required')
    (D/'grasp_candidate.json').write_text(json.dumps(audit,indent=2)+'\n')
    # A zero-residual actor exercises only the new moving robot reference. Diagnostic only.
    checkpoint=Path(entry['checkpoint']);raw=checkpoint.read_bytes();size=struct.unpack('<Q',raw[:8])[0]
    header=json.loads(raw[8:8+size]);payload=bytearray(raw[8+size:]);md=header['__metadata__']
    cfg=json.loads(md['user.resolved_config_json']);cfg['task']['parquet']=str(out_p.parent)
    md['user.resolved_config_json']=json.dumps(cfg,separators=(',',':'))
    md['user.nominal_zero_residual']='true';md['user.targeted_grasp_diagnostic']='true'
    for leaf in ('state.0018','state.0020'):
        assert header[leaf]['shape']==[56] and header[leaf]['dtype']=='F32'
        a,b=header[leaf]['data_offsets'];payload[a:b]=b'\0'*(b-a)
    h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((-len(h))%8)
    nominal=HERE/'ep5_donor_grasp_nominal';nominal.mkdir()
    policy=nominal/'policy_zero.safetensors';policy.write_bytes(struct.pack('<Q',len(h))+h+payload)
    (nominal/'checkpoints.json').write_text(json.dumps({'5':str(policy)})+'\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
