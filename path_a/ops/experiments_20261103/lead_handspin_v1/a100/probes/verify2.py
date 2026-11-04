import glob, json, os, hashlib
import numpy as np, pyarrow.parquet as pq
SC="/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes"
R="/root/lead_current/experiments/wentacc/scored20_lead_v3_hands_r1_20261104/scenes"
K="/root/lead_current/experiments/haisenk/hand_fixes_wentao_v1_20261103/episode_000001/scene"
cols=['object_body_position','object_body_wxyz','object_body_names','object_mesh_paths','object_urdf_paths','robot_left_wrist_position','robot_right_wrist_position','mano_left_trans','mano_right_trans']
def pqs(s): return sorted(glob.glob(s+"/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet"))
def load(f):
    t=pq.read_table(f,columns=cols); return {c:t.column(c)[0].as_py() for c in cols}
def fp(d):
    h=hashlib.sha256()
    for c in ('object_body_position','object_body_wxyz','robot_left_wrist_position','robot_right_wrist_position'): h.update(np.array(d[c],float).tobytes())
    return h.hexdigest()[:12]
print("== ep28 parquets")
r2="/root/v2d/lead_v3/handspin28r2/scenes/ep28_hold"; r1loc="/root/v2d/lead_v3/handspin28/scenes/ep28_hold"
for name,s in (("HF r1 folder",f"{R}/episode_000028/scene"),("A100 handspin28 (r1 build)",r1loc),("A100 handspin28r2",r2)):
    for f in pqs(s):
        d=load(f); print(f"  {name}: {os.path.basename(f)[:14]} size={os.path.getsize(f)} mtime={os.path.getmtime(f):.0f} fp={fp(d)} seq={f.split('sequence_id=')[1].split('/')[0]} paths_ok={all(os.path.exists(p) for p in d['object_mesh_paths']+d['object_urdf_paths'])}")
new=load(pqs("/root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl")[0])
print("== candidates vs refs")
for ep,a,b in [(28,r2,"/root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl"),(33,f"{R}/episode_000033/scene",f"{SC}/ep33_rawexcl"),(1,K,f"{SC}/ep01_rawexcl")]:
    fa=pqs(a); assert len(fa)==1,(a,fa); A=load(fa[0]); B=load(pqs(b)[0])
    pa,pb=np.array(A['object_body_position'],float),np.array(B['object_body_position'],float)
    print(f"ep{ep} {os.path.basename(a)} seq={fa[0].split('sequence_id=')[1].split('/')[0]} bodies={A['object_body_names']} parquets={len(fa)} paths_ok={all(os.path.exists(p) for p in A['object_mesh_paths']+A['object_urdf_paths'])} meshes={len(glob.glob(a+'/score_input/meshes/episode_*/*.obj'))} apple={len(glob.glob(a+'/**/._*',recursive=True))}")
    if pa.shape==pb.shape:
        qa,qb=np.array(A['object_body_wxyz'],float),np.array(B['object_body_wxyz'],float)
        print(f"   object pos max diff per body={np.abs(pa-pb).max(axis=(0,2)).round(5).tolist()} rot(1-|dot|) max={np.abs(1-np.abs((qa*qb).sum(-1))).max(axis=0).round(5).tolist()} wrist moved median cm L={np.median(np.linalg.norm(np.array(A['robot_left_wrist_position'],float)-np.array(B['robot_left_wrist_position'],float),axis=1))*100:.1f} R={np.median(np.linalg.norm(np.array(A['robot_right_wrist_position'],float)-np.array(B['robot_right_wrist_position'],float),axis=1))*100:.1f}")
    else: print("   shapes differ",pa.shape,pb.shape)
