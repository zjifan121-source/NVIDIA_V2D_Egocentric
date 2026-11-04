import glob, hashlib, json, os
import numpy as np, pyarrow.parquet as pq, trimesh
P="assets/human_motion_data/ego_recon/processed"
new="/root/v2d/lead_v3/handspin28oldr2com/scenes/ep28_hold"; cmp_new="/root/v2d/lead_v3/handspin28r2com/scenes/ep28_hold"; cmp_old="/root/v2d/lead_v3/handspin28oldr2/scenes/ep28_hold"
def pqs(s): return sorted(glob.glob(f"{s}/{P}/sequence_id=*/robot_name=*/*.parquet"))
def load(f):
    t=pq.read_table(f); return {c:t.column(c)[0].as_py() for c in t.column_names}
fn=pqs(new); print("parquets",len(fn),"seq",fn[0].split('sequence_id=')[1].split('/')[0] if fn else None)
A=load(fn[0]); us=A['object_urdf_paths']+A['object_mesh_paths']
print("paths own:",all(p.startswith(new) for p in us),"exist:",all(os.path.exists(p) for p in us),"score meshes:",sorted(os.path.basename(x) for x in glob.glob(new+"/score_input/meshes/episode_*/*.obj")),"apple:",len(glob.glob(new+"/**/._*",recursive=True)),"bodies",A['object_body_names'],"frames",len(A['object_body_position']))
for u in sorted(glob.glob(f"{new}/{P}/*_rigid.urdf")): print("  ",os.path.basename(u),[l.strip() for l in open(u) if '<origin' in l][:1])
for lab,s in (("OLD r2com",new),("NEW r2com",cmp_new)):
    for m in sorted(glob.glob(f"{s}/{P}/*white_pot.obj")):
        v=np.asarray(trimesh.load(m,force='mesh',process=False).vertices); ext=np.ptp(v,axis=0)*100
        print(f"  {lab} pot mesh {os.path.basename(m)}: extents cm {ext.round(1).tolist()} max {ext.max():.1f} verts {len(v)} sha {hashlib.sha256(open(m,'rb').read()).hexdigest()[:12]}")
for f in ("OUTPUT_CHECK.json","HAND_CONTACT.json","SCENE_COM_FIX.json"):
    p=f"{new}/{f}"
    if os.path.exists(p):
        d=json.load(open(p))
        if f=="OUTPUT_CHECK.json": print(" penetration:",d.get("penetration_frames"))
        elif f=="HAND_CONTACT.json": print(" contact:",{o:{k:(c.get("within_1cm") if isinstance(c,dict) else c) for k,c in v.items()} for o,v in d["objects"].items()})
        else: print(" com:",[(o.get("body"),o.get("offset_cm"),o.get("patched")) for o in d.get("objects",[])])
    else: print(f,"missing")
for lab,base in (("NEW r2com",cmp_new),("OLD r2 (no COM)",cmp_old),("lead_v3 ep28",f"/root/v2d/lead_v3/stage/scenes/episode_000028/scene")):
    fb=pqs(base)
    if not fb: print("vs",lab,": no parquet"); continue
    B=load(fb[0]); pa=np.array(A['object_body_position'],float); pb=np.array(B['object_body_position'],float)
    if pa.shape!=pb.shape: print("vs",lab,"shape differs"); continue
    qa=np.array(A['object_body_wxyz'],float); qb=np.array(B['object_body_wxyz'],float)
    dl=np.linalg.norm(np.array(A['robot_left_wrist_position'],float)-np.array(B['robot_left_wrist_position'],float),axis=1); dr=np.linalg.norm(np.array(A['robot_right_wrist_position'],float)-np.array(B['robot_right_wrist_position'],float),axis=1)
    print(f"vs {lab}: obj pos max cm {(np.abs(pa-pb).max(axis=(0,2))*100).round(2).tolist()} rot(1-|dot|) max {np.abs(1-np.abs((qa*qb).sum(-1))).max(axis=0).round(4).tolist()} wrist med cm L={np.median(dl)*100:.1f} R={np.median(dr)*100:.1f} radius diff {np.round(np.array(A['object_mesh_radius'])-np.array(B['object_mesh_radius']),4).tolist()}")
