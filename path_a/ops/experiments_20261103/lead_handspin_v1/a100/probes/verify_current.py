import glob, json, os
import numpy as np, pyarrow.parquet as pq
SC="/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes"
R="/root/lead_current/experiments/wentacc/scored20_lead_v3_hands_r1_20261104/scenes"
K="/root/lead_current/experiments/haisenk/hand_fixes_wentao_v1_20261103/episode_000001/scene"
cols=['object_body_position','object_body_wxyz','object_body_names','object_mesh_paths','object_urdf_paths','robot_left_wrist_position','robot_right_wrist_position']
def load(s):
    f=sorted(glob.glob(s+"/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet"))
    assert len(f)==1,(s,f)
    t=pq.read_table(f[0],columns=cols); return f[0],{c:t.column(c)[0].as_py() for c in cols}
P=[(25,f"{R}/episode_000025/scene",f"{SC}/ep25_rawexcl"),
   (28,f"{R}/episode_000028/scene","/root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl"),
   (33,f"{R}/episode_000033/scene",f"{SC}/ep33_rawexcl"),
   (1,K,f"{SC}/ep01_rawexcl")]
for ep,a,b in P:
    fa,A=load(a); fb,B=load(b)
    pa=np.array(A['object_body_position'],float); pb=np.array(B['object_body_position'],float)
    qa=np.array(A['object_body_wxyz'],float); qb=np.array(B['object_body_wxyz'],float)
    same=pa.shape==pb.shape
    dp=np.abs(pa-pb).max(axis=(0,2)).round(5).tolist() if same else 'shape differs'
    dq=np.abs(np.abs((qa*qb).sum(-1))-1).max(axis=0).round(6).tolist() if same else '-'
    dl=np.linalg.norm(np.array(A['robot_left_wrist_position'],float)-np.array(B['robot_left_wrist_position'],float),axis=1) if same else np.zeros(1)
    dr=np.linalg.norm(np.array(A['robot_right_wrist_position'],float)-np.array(B['robot_right_wrist_position'],float),axis=1) if same else np.zeros(1)
    print(f"ep{ep} seq={fa.split('sequence_id=')[1].split('/')[0]} bodies={A['object_body_names']}")
    print(f"   paths_exist={all(os.path.exists(p) for p in A['object_mesh_paths']+A['object_urdf_paths'])} score_meshes={len(glob.glob(a+'/score_input/meshes/episode_*/*.obj'))} apple={len(glob.glob(a+'/**/._*',recursive=True))}")
    print(f"   vs running/ref scene: object pos max diff per body={dp} rot diff={dq}; wrist moved median cm L={np.median(dl)*100:.1f} R={np.median(dr)*100:.1f}")
    pv=os.path.join(os.path.dirname(a),'PROVENANCE.json')
    if os.path.exists(pv):
        d=json.load(open(pv)); print("   provenance:",json.dumps({k:d.get(k) for k in ('objects_version','hold_windows','hands','policy_tested') if k in d},ensure_ascii=False)[:360])
